/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier/poplar_algebraic_simplifier_convolution.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier/poplar_algebraic_simplifier_dot.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier/poplar_algebraic_simplifier_utils.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/arg_min_max.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/overflow_util.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {

namespace {

namespace m = match;
namespace pp = poplarplugin;

constexpr int64 very_small_gather_size = 4;

}  // namespace

HloInstruction* AlgebraicSimplifierVisitor::PreserveFrontendAttributesIfNeeded(
    HloInstruction* new_inst, const HloInstruction* old_inst) {
  if (new_inst->frontend_attributes().map().empty() &&
      !old_inst->frontend_attributes().map().empty()) {
    new_inst->set_frontend_attributes(old_inst->frontend_attributes());
  }
  return new_inst;
}

StatusOr<HloInstruction*>
AlgebraicSimplifierVisitor::PreserveFrontendAttributesIfNeeded(
    StatusOr<HloInstruction*> new_inst, const HloInstruction* old_inst) {
  if (new_inst.ok()) {
    return PreserveFrontendAttributesIfNeeded(new_inst.ValueOrDie(), old_inst);
  }
  return new_inst;
}

// Converts to primitive type if the input hlo is not that type, otherwise
// returns the original hlo.
HloInstruction* AlgebraicSimplifierVisitor::AsType(
    HloInstruction* hlo, const PrimitiveType element_type) {
  if (hlo->shape().element_type() == element_type) {
    return hlo;
  }
  Shape changed_shape =
      ShapeUtil::ChangeElementType(hlo->shape(), element_type);
  simplifier_->UpdateLayout(&changed_shape);

  return PreserveFrontendAttributesIfNeeded(
      computation_->AddInstruction(
          HloInstruction::CreateConvert(changed_shape, hlo)),
      hlo);
}

// Helper method to perform and add reduction on a list of dimensions.
HloInstruction* AlgebraicSimplifierVisitor::AddReduce(
    HloInstruction* hlo, absl::Span<const int64> dims, PrimitiveType type) {
  HloInstruction* zero =
      computation_->AddInstruction(simplifier_->CreateConstantWithLayoutUpdated(
          LiteralUtil::Zero(hlo->shape().element_type()).Clone()));
  HloComputation* AddReduce_computation = GetOrCreateScalarAddComputation(type);
  Shape shape = ShapeUtil::FilterDimensions(
      [&](int64 dim) { return !absl::c_linear_search(dims, dim); },
      hlo->shape());
  simplifier_->UpdateLayout(&shape);
  return computation_->AddInstruction(HloInstruction::CreateReduce(
      shape, hlo, zero, dims, AddReduce_computation));
}

HloComputation* AlgebraicSimplifierVisitor::GetOrCreateScalarAddComputation(
    PrimitiveType type) {
  if (scalar_add_computations_.find(type) != scalar_add_computations_.end()) {
    return scalar_add_computations_.at(type);
  }

  HloComputation::Builder b("scalar_add_computation");
  Shape shape = ShapeUtil::MakeShape(type, {});
  simplifier_->UpdateLayout(&shape);
  auto scalar_lhs =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "scalar_lhs"));
  auto scalar_rhs =
      b.AddInstruction(HloInstruction::CreateParameter(1, shape, "scalar_rhs"));
  auto scalar_op = b.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, scalar_lhs, scalar_rhs));
  auto scalar_add_computation =
      computation_->parent()->AddEmbeddedComputation(b.Build(scalar_op));
  scalar_add_computations_.insert({type, scalar_add_computation});

  return scalar_add_computation;
}

void AlgebraicSimplifierVisitor::ResetState(HloComputation* computation) {
  changed_ = false;
  ResetVisitStates();
  computation_ = computation;
}

bool AlgebraicSimplifierVisitor::Run(HloComputation* computation,
                                     PoplarAlgebraicSimplifier* simplifier) {
  ResetState(computation);
  TF_CHECK_OK(computation->Accept(this));
  return changed_ || changed();
}

bool AlgebraicSimplifierVisitor::ReplaceInstructionIfSameShape(
    HloInstruction* old_instruction, HloInstruction* new_instruction) {
  if (!ShapeUtil::Compatible(old_instruction->shape(),
                             new_instruction->shape())) {
    return false;
  }
  TF_CHECK_OK(ReplaceInstruction(old_instruction, new_instruction));
  return true;
}

Status AlgebraicSimplifierVisitor::HandleAdd(HloInstruction* add) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(add, m::Add(m::Op(&lhs), m::Op(&rhs))));

  // A + 0 => A
  VLOG(10) << "trying transform [A + 0 => A]: " << add->ToString();
  if (pp::algebraic_simplifier::util::IsAll(rhs, 0) &&
      ReplaceInstructionIfSameShape(add, lhs)) {
    return Status::OK();
  }
  // 0 + A => A
  VLOG(10) << "trying transform [0 + A => A]: " << add->ToString();
  if (pp::algebraic_simplifier::util::IsAll(lhs, 0) &&
      ReplaceInstructionIfSameShape(add, rhs)) {
    return Status::OK();
  }

  // Canonicalization: Put constants on the right.  This makes the reassociation
  // rules below simpler.
  VLOG(10) << "trying transform [Const + A => A + Const]";
  if (Match(add, m::Add(m::Constant(), m::NonConstant()))) {
    return ReplaceWithNewInstruction(
        add,
        HloInstruction::CreateBinary(add->shape(), HloOpcode::kAdd, rhs, lhs));
  }

  // Reassociate to allow constant folding.
  //
  // Note: This is not general.  For example, we won't reassociate
  //
  //   (A + C1) + (B + C2) =>  A + B + (C1 + C2).
  //
  VLOG(10) << "trying transform [(A + C1) + C2 => A + (C1 + C2)]";
  HloInstruction *a, *c1, *c2;
  if (Match(add, m::Add(m::Add(m::NonConstant(&a), m::Constant(&c1)),
                        m::Constant(&c2))) ||
      Match(add, m::Add(m::Add(m::NonConstant(&a),
                               m::Broadcast(m::ConstantScalar(&c1))),
                        m::Broadcast(m::ConstantScalar(&c2))))) {
    TF_ASSIGN_OR_RETURN(auto* sum_of_constants,
                        MakeBinaryHlo(HloOpcode::kAdd, c1, c2));
    if (ShapeUtil::IsScalar(sum_of_constants->shape()) &&
        !ShapeUtil::IsScalar(add->shape())) {
      sum_of_constants = computation_->AddInstruction(
          HloInstruction::CreateBroadcast(add->shape(), sum_of_constants, {}));
    }
    return ReplaceWithNewInstruction(
        add, HloInstruction::CreateBinary(add->shape(), HloOpcode::kAdd, a,
                                          sum_of_constants));
  }

  // Convert add with fullshape into add with partial shape when a
  // portion of add is effective:
  //             zero (fullshape)   rhs (partialshape)
  // .           |                  |
  // . lhs .    dynamic_update_slice (fullshape)
  // . |         |
  // Add (fullshape)
  //
  // to:
  //              lhs
  //              |
  //             dynamic_slice (partialshape)   rhs (partialshape)
  // .           |                      |
  // . lhs .    add (partial_shape)+----+
  // . |         |
  // dynamic_update_slice (fullshape)
  //
  // This is pattern is discovered in control flow V2 gradient update.
  if (Match(add,
            m::Add(m::Op(&lhs),
                   m::Op(&rhs)
                       .WithOpcode(HloOpcode::kDynamicUpdateSlice)
                       .WithOperand(
                           0, m::Broadcast(m::ConstantEffectiveScalar(0)))))) {
    const Shape& partial_shape = rhs->operand(1)->shape();
    auto sliced_lhs =
        computation_->AddInstruction(HloInstruction::CreateDynamicSlice(
            partial_shape, lhs, absl::MakeSpan(rhs->operands()).subspan(2),
            partial_shape.dimensions()));

    auto add_partial = computation_->AddInstruction(
        HloInstruction::CreateBinary(rhs->operand(1)->shape(), HloOpcode::kAdd,
                                     sliced_lhs, rhs->mutable_operand(1)));

    auto dynamic_update_slice_full = HloInstruction::CreateDynamicUpdateSlice(
        lhs->shape(), lhs, add_partial,
        absl::MakeSpan(rhs->operands()).subspan(2));

    return ReplaceWithNewInstruction(add, std::move(dynamic_update_slice_full));
  }

  // A*C + B*C => (A+B)*C
  //
  //  - If A, B, and C are integers, do this unconditionally. Proof of
  //    correctness: https://rise4fun.com/Alive/u9X.
  //
  //  - If A, B, and C are floating point, do this if C is a scalar constant or
  //    broadcast of scalar constant and is equal to +/- 2^k for some (possibly
  //    negative) integer k.
  //
  //    Multiplying by a power of 2 just moves the exponent, so our answer is
  //    exact modulo rounding of intermediate results so long as
  //
  //     - none of the three products has an exponent which underflows (so the
  //       result is 0 or denormal), and
  //     - none of the three products overflows to inf.
  //
  //    Proof: See algebraic_simplifier_proof_distributive_property.py.
  //
  //    We deem these differences in rounding, underflow, and overflow
  //    acceptable in the ML context.
  HloInstruction *b, *c;
  if (((Match(lhs, m::Multiply(m::Op(&a), m::Op(&c))) &&
        Match(rhs, m::MultiplyAnyOrder(m::Op().Is(c), m::Op(&b)))) ||
       (Match(lhs, m::Multiply(m::Op(&c), m::Op(&a))) &&
        Match(rhs, m::MultiplyAnyOrder(m::Op().Is(c), m::Op(&b))))) &&
      (ShapeUtil::ElementIsIntegral(add->shape()) ||
       pp::algebraic_simplifier::util::IsAllFpConstantPowerOf2(c))) {
    return ReplaceWithNewInstruction(
        add, HloInstruction::CreateBinary(
                 add->shape(), HloOpcode::kMultiply,
                 computation_->AddInstruction(HloInstruction::CreateBinary(
                     add->shape(), HloOpcode::kAdd, a, b)),
                 c));
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleAnd(HloInstruction* logical_and) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(logical_and, m::And(m::Op(&lhs), m::Op(&rhs))));
  // Simplify logical and
  if (ShapeUtil::HasPrimitiveType(lhs->shape(), xla::PRED) &&
      ShapeUtil::HasPrimitiveType(rhs->shape(), xla::PRED)) {
    // A && True => A
    VLOG(10) << "trying transform [A && True => A]: "
             << logical_and->ToString();
    if (pp::algebraic_simplifier::util::IsAll(rhs, 1) &&
        ReplaceInstructionIfSameShape(logical_and, lhs)) {
      return Status::OK();
    }
    // True && A => A
    VLOG(10) << "trying transform [True && A => A]: "
             << logical_and->ToString();
    if (pp::algebraic_simplifier::util::IsAll(lhs, 1) &&
        ReplaceInstructionIfSameShape(logical_and, rhs)) {
      return Status::OK();
    }
  }

  // A && False => False or A & 0 => 0
  VLOG(10) << "trying transform [A && False => False]: "
           << logical_and->ToString();
  if (pp::algebraic_simplifier::util::IsAll(rhs, 0) &&
      ReplaceInstructionIfSameShape(logical_and, rhs)) {
    return Status::OK();
  }

  // False && A => False or A & 0 => 0
  VLOG(10) << "trying transform [False && A => False]: "
           << logical_and->ToString();
  if (pp::algebraic_simplifier::util::IsAll(lhs, 0) &&
      ReplaceInstructionIfSameShape(logical_and, lhs)) {
    return Status::OK();
  }

  // A && A => A
  VLOG(10) << "trying transform [A && A => A]: " << logical_and->ToString();
  if (lhs->Identical(*rhs) && ReplaceInstructionIfSameShape(logical_and, lhs)) {
    return Status::OK();
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleBitcast(HloInstruction* bitcast) {
  // If a bitcast feeds a bitcast, make it a single bitcast.
  HloInstruction* op;
  if (Match(bitcast, m::Bitcast(m::Bitcast(m::Op(&op))))) {
    return ReplaceWithNewInstruction(
        bitcast, HloInstruction::CreateBitcast(bitcast->shape(), op));
  }
  // All bitcasts can be eliminated (assuming layout constraints are
  // satisified).
  ReplaceInstructionIfSameShape(bitcast, bitcast->mutable_operand(0));
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleBitcastConvert(
    HloInstruction* bitcast) {
  // If a bitcast convert feeds a bitcast convert, make it a single bitcast
  // convert.
  HloInstruction* op;
  if (Match(bitcast, m::BitcastConvert(m::BitcastConvert(m::Op(&op))))) {
    return ReplaceWithNewInstruction(
        bitcast, HloInstruction::CreateBitcastConvert(bitcast->shape(), op));
  }
  // Eliminate bitcast converts between same shape.
  ReplaceInstructionIfSameShape(bitcast, bitcast->mutable_operand(0));
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleCopy(HloInstruction* copy) {
  // If a copy feeds a copy, make it a single copy.
  HloInstruction* op;
  if (Match(copy, m::Copy(m::Copy(m::Op(&op))))) {
    return ReplaceWithNewInstruction(
        copy, HloInstruction::CreateUnary(copy->shape(), HloOpcode::kCopy, op));
  }
  // All copies can be eliminated (assuming layout constraints are satisified).
  if (ReplaceInstructionIfSameShape(copy, copy->mutable_operand(0))) {
    return Status::OK();
  }

  // Replace Copy(Reshape()) with Reshape() if the Reshape is a logical bitcast.
  if (copy->operand(0)->opcode() == HloOpcode::kReshape &&
      copy->operand(0)->user_count() == 1 &&
      ShapeUtil::ReshapeIsBitcast(copy->operand(0)->shape(), copy->shape())) {
    return ReplaceWithNewInstruction(
        copy,
        copy->operand(0)->CloneWithNewOperands(
            copy->shape(), {copy->mutable_operand(0)->mutable_operand(0)}));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleConcatenate(
    HloInstruction* concatenate) {
  absl::Span<HloInstruction* const> operands(concatenate->operands());
  if (operands.size() == 1) {
    // Unary concatenates are useless.
    ReplaceInstructionIfSameShape(concatenate, operands[0]);
    return Status::OK();
  }
  // Filter out and remove empty operands.
  std::vector<HloInstruction*> nonempty_operands;
  for (HloInstruction* operand : operands) {
    if (!ShapeUtil::IsZeroElementArray(operand->shape())) {
      nonempty_operands.push_back(operand);
    }
  }
  if (nonempty_operands.size() < operands.size()) {
    HloInstruction* replacement;
    if (nonempty_operands.empty()) {
      replacement = operands[0];
    } else if (nonempty_operands.size() == 1) {
      replacement = nonempty_operands[0];
    } else {
      replacement =
          computation_->AddInstruction(concatenate->CloneWithNewOperands(
              concatenate->shape(), nonempty_operands));
    }
    VLOG(10) << "trying to replace " << concatenate->ToString() << " with "
             << replacement->ToString();
    ReplaceInstructionIfSameShape(concatenate, replacement);
    return Status::OK();
  }

  // Check if we can merge "adjacent" concatenate operations
  for (size_t i = 0; i < operands.size(); ++i) {
    // Make sure the other Concatenate is along the same dimension and this
    // concatenate is its only user.
    if (operands[i]->opcode() != HloOpcode::kConcatenate ||
        concatenate->concatenate_dimension() !=
            operands[i]->concatenate_dimension() ||
        operands[i]->user_count() > 1) {
      continue;
    }
    HloInstruction::InstructionVector new_operands = concatenate->operands();
    new_operands.erase(new_operands.begin() + i);
    new_operands.insert(new_operands.begin() + i,
                        operands[i]->operands().begin(),
                        operands[i]->operands().end());

    HloInstruction* new_instruction =
        computation_->AddInstruction(HloInstruction::CreateConcatenate(
            concatenate->shape(), new_operands,
            concatenate->concatenate_dimension()));
    TF_CHECK_OK(ReplaceInstruction(concatenate, new_instruction));
    return Status::OK();
  }

  // Check if we can merge "adjacent" slice operands which take slices from the
  // same other op. For simplicity we only merge unstrided slices.
  int64 concatenate_dimension = concatenate->concatenate_dimension();
  for (int64 i = 0; i < static_cast<int64>(operands.size()); ++i) {
    if (operands[i]->opcode() != HloOpcode::kSlice ||
        !pp::algebraic_simplifier::util::IsUnstridedSlice(operands[i])) {
      continue;
    }
    int64 slice_end = operands[i]->slice_limits(concatenate_dimension);
    HloInstruction* slice_operand = operands[i]->mutable_operand(0);
    int64 j = i + 1;
    while (j < static_cast<int64>(operands.size()) &&
           operands[j]->opcode() == HloOpcode::kSlice &&
           pp::algebraic_simplifier::util::IsUnstridedSlice(operands[j]) &&
           operands[j]->operand(0) == slice_operand &&
           operands[j]->slice_starts(concatenate_dimension) == slice_end) {
      // Check that all the slice_start values are the same in all other
      // dimensions. This implies that the slice_limit values are also the same,
      // because operands of concatenate need to have the same shape, and we
      // already checked that the slices are unstrided.
      bool same_other_starts = true;
      for (int64 k = 0;
           k < static_cast<int64>(operands[j]->slice_starts().size()); ++k) {
        if (k == concatenate_dimension) {
          continue;
        }
        if (operands[i]->slice_starts(k) != operands[j]->slice_starts(k)) {
          same_other_starts = false;
          break;
        }
      }
      if (!same_other_starts) {
        break;
      }
      slice_end = operands[j]->slice_limits(concatenate_dimension);
      ++j;
    }
    if (j - i > 1) {
      Shape new_slice_shape = operands[i]->shape();
      new_slice_shape.set_dimensions(
          concatenate_dimension,
          slice_end - operands[i]->slice_starts(concatenate_dimension));
      simplifier_->UpdateLayout(&new_slice_shape);
      auto new_limit_indices = operands[i]->slice_limits();
      new_limit_indices[concatenate_dimension] = slice_end;
      auto new_slice_op =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation_->AddInstruction(HloInstruction::CreateSlice(
                  new_slice_shape, slice_operand,
                  /*start_indices=*/operands[i]->slice_starts(),
                  /*limit_indices=*/new_limit_indices,
                  /*strides=*/operands[i]->slice_strides())),
              slice_operand);
      std::vector<HloInstruction*> new_operands;
      for (int64 k = 0; k < i; ++k) {
        new_operands.push_back(operands[k]);
      }
      new_operands.push_back(new_slice_op);
      for (int64 k = j; k < static_cast<int64>(operands.size()); ++k) {
        new_operands.push_back(operands[k]);
      }
      auto replacement =
          computation_->AddInstruction(concatenate->CloneWithNewOperands(
              concatenate->shape(), new_operands));
      ReplaceInstructionIfSameShape(concatenate, replacement);
      return Status::OK();
    }
  }

  if (operands.size() == 2) {
    // A binary concat with a broadcasted scalar as an operand can be converted
    // into a pad which is simpler to fold into other operations.
    bool is_effective_low_pad = Match(
        operands[0], m::Broadcast(m::Op().WithShape(m::Shape().IsScalar())));
    bool is_effective_high_pad = Match(
        operands[1], m::Broadcast(m::Op().WithShape(m::Shape().IsScalar())));
    if (!is_effective_low_pad && !is_effective_high_pad) {
      return Status::OK();
    }
    PaddingConfig padding_config;
    for (int64 dim = 0; dim < operands[0]->shape().rank(); ++dim) {
      auto padding_config_dim = padding_config.add_dimensions();
      padding_config_dim->set_edge_padding_high(0);
      padding_config_dim->set_edge_padding_low(0);
      padding_config_dim->set_interior_padding(0);
      if (dim == concatenate_dimension) {
        if (is_effective_low_pad) {
          padding_config_dim->set_edge_padding_low(
              operands[0]->shape().dimensions(dim));
        } else {
          padding_config_dim->set_edge_padding_high(
              operands[1]->shape().dimensions(dim));
        }
      }
    }
    int64 operand_to_pad = is_effective_low_pad ? 1 : 0;
    int64 pad_value_operand = is_effective_low_pad ? 0 : 1;
    HloInstruction* pad =
        computation_->AddInstruction(HloInstruction::CreatePad(
            concatenate->shape(), operands[operand_to_pad],
            operands[pad_value_operand]->mutable_operand(0), padding_config));
    return ReplaceInstruction(concatenate, pad);
  }
  return Status::OK();
}

static HloInstruction* BuildTupleConstant(
    HloComputation* computation, const LiteralSlice& literal,
    PoplarAlgebraicSimplifier* simplifier) {
  if (literal.shape().IsTuple()) {
    std::vector<HloInstruction*> elems;
    elems.reserve(ShapeUtil::TupleElementCount(literal.shape()));
    for (int i = 0; i < ShapeUtil::TupleElementCount(literal.shape()); ++i) {
      elems.push_back(BuildTupleConstant(
          computation, LiteralSlice(literal, {i}), simplifier));
    }
    return computation->AddInstruction(HloInstruction::CreateTuple(elems));
  } else {
    return computation->AddInstruction(
        simplifier->CreateConstantWithLayoutUpdated(literal.Clone()));
  }
}

Status AlgebraicSimplifierVisitor::HandleConstant(HloInstruction* constant) {
  // Tuple constants aren't directly supported by any backend. Expand them into
  // explicit Tuple instructions.
  if (constant->shape().IsTuple()) {
    return ReplaceInstruction(
        constant,
        BuildTupleConstant(computation_, constant->literal(), simplifier_));
  }

  if (constant->shape().element_type() == TOKEN) {
    return Status::OK();
  }

  // If a literal is all the same element replace it with a scalar broadcast.
  if (ShapeUtil::ElementsIn(constant->shape()) > 1 &&
      constant->literal().IsAllFirst()) {
    Literal unique_scalar(
        LiteralUtil::GetFirstScalarLiteral(constant->literal()));
    HloInstruction* scalar = computation_->AddInstruction(
        simplifier_->CreateConstantWithLayoutUpdated(std::move(unique_scalar)));
    return ReplaceWithNewInstruction(
        constant,
        HloInstruction::CreateBroadcast(constant->shape(), scalar, {}));
  }

  // If a literal is an increasing sequence from zero, replace it with an iota.
  if (constant->shape().rank() == 1 &&
      ShapeUtil::ElementsIn(constant->shape()) > 1 &&
      constant->literal().IsR1Iota()) {
    return ReplaceWithNewInstruction(
        constant, HloInstruction::CreateIota(constant->shape(), 0));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleSubtract(HloInstruction* sub) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(sub, m::Subtract(m::Op(&lhs), m::Op(&rhs))));
  // A - 0 => A
  VLOG(10) << "trying transform [A - 0 => A]: " << sub->ToString();
  if (pp::algebraic_simplifier::util::IsAll(rhs, 0) &&
      ReplaceInstructionIfSameShape(sub, lhs)) {
    return Status::OK();
  }

  // Canonicalize subtraction of a constant to addition.
  VLOG(10) << "trying transform [A - Const => A + (-Const)]";
  if (Match(sub, m::Subtract(m::NonConstant(&lhs), m::Constant(&rhs))) ||
      Match(sub, m::Subtract(m::NonConstant(&lhs),
                             m::Broadcast(m::Constant(&rhs))))) {
    HloInstruction* negative_const =
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(HloInstruction::CreateUnary(
                rhs->shape(), HloOpcode::kNegate, rhs)),
            rhs);
    if (const HloInstruction* broadcast =
            DynCast<HloBroadcastInstruction>(sub->operand(1))) {
      negative_const =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation_->AddInstruction(HloInstruction::CreateBroadcast(
                  broadcast->shape(), negative_const, broadcast->dimensions())),
              negative_const);
    }
    return ReplaceWithNewInstruction(
        sub, HloInstruction::CreateBinary(sub->shape(), HloOpcode::kAdd, lhs,
                                          negative_const));
  }

  return Status::OK();
}
namespace {
template <typename T>
Status InvertConstant(const HloInstruction& constant, Literal* result) {
  return result->Populate<T>([&](absl::Span<const int64> indices) {
    return T{1.0} / constant.literal().Get<T>(indices);
  });
}

template <typename T>
std::unique_ptr<HloInstruction> TryDivideToShift(
    HloInstruction* divide, HloComputation* computation,
    PoplarAlgebraicSimplifier* simplifier) {
  HloInstruction *a, *b, *c;
  CHECK(Match(divide, m::Divide(m::Op(&a), m::Op(&b))));

  if (ShapeUtil::ElementIsIntegral(divide->shape()) &&
      !Match(b, m::ConstantEffectiveScalar(&c)) &&
      !Match(b, m::Broadcast(m::ConstantEffectiveScalar(&c)))) {
    return nullptr;
  }

  if (ShapeUtil::ElementIsSigned(divide->shape())) {
    int64 b_value = c->literal().GetFirstElement<T>();
    if (b_value > 0 && IsPowerOfTwo(static_cast<uint64>(b_value))) {
      // Handle negative dividends by negating the result of the division.
      HloInstruction* zero_like_a = BroadcastZeros(
          computation, a->shape().element_type(), a->shape().dimensions());

      Shape changed_shape = ShapeUtil::ChangeElementType(a->shape(), PRED);
      simplifier->UpdateLayout(&changed_shape);
      auto* dividend_is_negative =
          computation->AddInstruction(HloInstruction::CreateCompare(
              changed_shape, a, zero_like_a, ComparisonDirection::kLt));

      auto* negated_dividend =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation->AddInstruction(HloInstruction::CreateUnary(
                  a->shape(), HloOpcode::kNegate, a)),
              a);

      auto* abs_dividend =
          computation->AddInstruction(HloInstruction::CreateTernary(
              a->shape(), HloOpcode::kSelect, dividend_is_negative,
              negated_dividend, a));

      int log2_abs_b_value = tensorflow::Log2Floor64(b_value);

      auto* shift_amount = computation->AddInstruction(
          simplifier->CreateConstantWithLayoutUpdated(
              LiteralUtil::CreateR0<T>(log2_abs_b_value)));
      if (!ShapeUtil::IsScalar(b->shape())) {
        shift_amount =
            pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
                computation->AddInstruction(HloInstruction::CreateBroadcast(
                    b->shape(), shift_amount, {})),
                b);
      }

      auto* quotient =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation->AddInstruction(HloInstruction::CreateBinary(
                  divide->shape(), HloOpcode::kShiftRightLogical, abs_dividend,
                  shift_amount)),
              divide);

      auto* neqated_quotient =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation->AddInstruction(HloInstruction::CreateUnary(
                  quotient->shape(), HloOpcode::kNegate, quotient)),
              quotient);

      return HloInstruction::CreateTernary(divide->shape(), HloOpcode::kSelect,
                                           dividend_is_negative,
                                           neqated_quotient, quotient);
    }
  } else {
    uint64 b_value = c->literal().GetFirstElement<T>();
    if (IsPowerOfTwo(b_value)) {
      int log2_abs_b_value = tensorflow::Log2Floor64(b_value);
      HloInstruction* shift_amount =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation->AddInstruction(
                  simplifier->CreateConstantWithLayoutUpdated(
                      LiteralUtil::CreateR0<T>(log2_abs_b_value))),
              c);
      if (!ShapeUtil::IsScalar(b->shape())) {
        shift_amount =
            pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
                computation->AddInstruction(HloInstruction::CreateBroadcast(
                    b->shape(), shift_amount, {})),
                shift_amount);
      }
      return HloInstruction::CreateBinary(
          divide->shape(), HloOpcode::kShiftRightLogical, a, shift_amount);
    }
  }

  return nullptr;
}
}  // namespace

Status AlgebraicSimplifierVisitor::HandleDivide(HloInstruction* divide) {
  HloInstruction *a, *b, *c, *d;
  CHECK(Match(divide, m::Divide(m::Op(&a), m::Op(&b))));
  // A/1 => A
  VLOG(10) << "trying transform [A/1 => A]: " << divide->ToString();
  if (pp::algebraic_simplifier::util::IsAll(b, 1) &&
      ReplaceInstructionIfSameShape(divide, a)) {
    return Status::OK();
  }

  // A / B => A >> log2(B) if B is a power of 2.
  switch (divide->shape().element_type()) {
    case S8:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<int8>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case S16:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<int16>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case S32:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<int32>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case S64:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<int64>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case U8:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<uint8>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case U16:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<uint16>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case U32:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<uint32>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case U64:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<uint64>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    default:
      break;
  }

  Shape* shape;
  // exp(A)/exp(B) => exp(A-B)
  if (Match(divide, m::Divide(m::Exp(m::Op(&a)), m::Exp(m::Op(&b)))
                        .WithShape(m::Shape(&shape)))) {
    VLOG(10) << "transform [exp(A)/exp(B) => exp(A-B)]: " << divide->ToString();
    HloInstruction* subtract =
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(HloInstruction::CreateBinary(
                *shape, HloOpcode::kSubtract, a, b)),
            divide);
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateUnary(*shape, HloOpcode::kExp, subtract));
  }

  // A/exp(B) => A*exp(-B)
  if (Match(divide, m::Divide(m::Op(&a), m::Exp(m::Op(&b))))) {
    VLOG(10) << "transform [A/exp(B) => A*exp(-B)]: " << divide->ToString();
    HloInstruction* negate =
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(HloInstruction::CreateUnary(
                divide->shape(), HloOpcode::kNegate, b)),
            b);
    HloInstruction* new_exp =
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(HloInstruction::CreateUnary(
                divide->shape(), HloOpcode::kExp, negate)),
            negate);
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateBinary(divide->shape(),
                                             HloOpcode::kMultiply, a, new_exp));
  }

  // A/pow(B,C) => A*pow(B,-C)
  if (Match(divide, m::Divide(m::Op(&a), m::Power(m::Op(&b), m::Op(&c))))) {
    VLOG(10) << "transform [A/pow(B,C) => A*pow(B,-C)]: " << divide->ToString();
    // The output shape of the created negate operator should be the same as the
    // input.
    const Shape& negate_shape = c->shape();
    HloInstruction* negate =
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(HloInstruction::CreateUnary(
                negate_shape, HloOpcode::kNegate, c)),
            c);
    // And the power operator should retain the output shape of the old one.
    const Shape& new_power_shape = b->shape();
    HloInstruction* new_power =
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(HloInstruction::CreateBinary(
                new_power_shape, HloOpcode::kPower, b, negate)),
            divide);
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateBinary(
                    divide->shape(), HloOpcode::kMultiply, a, new_power));
  }

  // A/sqrt(B) => A*rsqrt(X).
  if (Match(divide, m::Divide(m::Op(&a), m::Sqrt(m::Op(&b))))) {
    auto* rsqrt =
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(HloInstruction::CreateUnary(
                divide->shape(), HloOpcode::kRsqrt, b)),
            divide);
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateBinary(rsqrt->shape(),
                                             HloOpcode::kMultiply, a, rsqrt));
  }

  // A/rsqrt(B) => A*sqrt(B).
  if (Match(divide, m::Divide(m::Op(&a), m::Rsqrt(m::Op(&b))))) {
    auto* sqrt =
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(HloInstruction::CreateUnary(
                divide->shape(), HloOpcode::kSqrt, b)),
            divide);
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateBinary(sqrt->shape(),
                                             HloOpcode::kMultiply, a, sqrt));
  }

  // Simplifying integral division would produce unexpected results.
  if (ShapeUtil::ElementIsIntegral(divide->shape())) {
    return Status::OK();
  }

  // A / Const => A * (1 / Const)
  //
  // (Backends can do this transformation, but generally only if the constant is
  // a scalar.)
  if (Match(divide, m::Divide(m::NonConstant(&a), m::Op(&b))) &&
      (Match(b, m::Constant(&c)) || Match(b, m::Broadcast(m::Constant(&c))))) {
    Shape result_shape = c->literal().shape();
    Literal new_literal(result_shape);
    switch (result_shape.element_type()) {
      case F16:
        TF_RETURN_IF_ERROR(InvertConstant<half>(*c, &new_literal));
        break;
      case F32:
        TF_RETURN_IF_ERROR(InvertConstant<float>(*c, &new_literal));
        break;
      case BF16:
        TF_RETURN_IF_ERROR(InvertConstant<bfloat16>(*c, &new_literal));
        break;
      case F64:
        TF_RETURN_IF_ERROR(InvertConstant<double>(*c, &new_literal));
        break;
      case C64:
        TF_RETURN_IF_ERROR(InvertConstant<complex64>(*c, &new_literal));
        break;
      case C128:
        TF_RETURN_IF_ERROR(InvertConstant<complex128>(*c, &new_literal));
        break;
      default:
        return Status::OK();
    }
    auto inverse = computation_->AddInstruction(
        simplifier_->CreateConstantWithLayoutUpdated(new_literal.Clone()));
    if (b != c) {
      inverse = computation_->AddInstruction(HloInstruction::CreateBroadcast(
          b->shape(), inverse, b->dimensions()));
    }
    TF_ASSIGN_OR_RETURN(auto new_divide,
                        MakeBinaryHlo(HloOpcode::kMultiply, a, inverse));
    return ReplaceInstruction(divide, new_divide);
  }

  // A / Broadcast(B) => A * Broadcast((1 / B))
  if (config_.enable_fast_math() &&
      Match(divide, m::Divide(m::NonConstant(&a), m::Op(&b))) &&
      Match(b, m::Broadcast(m::Op(&c)))) {
    const Shape& divisor_shape = c->shape();
    const PrimitiveType element_type = divisor_shape.element_type();
    if (primitive_util::IsFloatingPointType(element_type)) {
      HloInstruction* scalar_one = computation_->AddInstruction(
          simplifier_->CreateConstantWithLayoutUpdated(
              LiteralUtil::One(element_type)));

      TF_ASSIGN_OR_RETURN(
          HloInstruction * inverse,
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              MakeBinaryHlo(HloOpcode::kDivide, scalar_one, c), divide));

      inverse = computation_->AddInstruction(HloInstruction::CreateBroadcast(
          b->shape(), inverse, b->dimensions()));

      TF_ASSIGN_OR_RETURN(
          HloInstruction * new_divide,
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              MakeBinaryHlo(HloOpcode::kMultiply, a, inverse), divide));

      return ReplaceInstruction(divide, new_divide);
    }
  }

  // (A / B) / (C / D)  =>  (A / B)*(D / C) => (A * D) / (B * C)
  if (Match(divide, m::Divide(m::Divide(m::Op(&a), m::Op(&b)),
                              m::Divide(m::Op(&c), m::Op(&d))))) {
    TF_ASSIGN_OR_RETURN(
        auto a_times_d,
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            MakeBinaryHlo(HloOpcode::kMultiply, a, d), divide));
    TF_ASSIGN_OR_RETURN(
        auto b_times_c,
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            MakeBinaryHlo(HloOpcode::kMultiply, b, c), divide));
    TF_ASSIGN_OR_RETURN(auto new_divide, MakeBinaryHlo(HloOpcode::kDivide,
                                                       a_times_d, b_times_c));

    return ReplaceInstruction(divide, new_divide);
  }

  // (A / B) / C => A / (B * C)
  if (Match(divide, m::Divide(m::Divide(m::Op(&a), m::Op(&b)), m::Op(&c)))) {
    TF_ASSIGN_OR_RETURN(
        auto b_times_c,
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            MakeBinaryHlo(HloOpcode::kMultiply, b, c), divide));
    TF_ASSIGN_OR_RETURN(auto new_divide,
                        MakeBinaryHlo(HloOpcode::kDivide, a, b_times_c));
    return ReplaceInstruction(divide, new_divide);
  }

  // A / (B / C) => (A*C) / B
  if (Match(divide, m::Divide(m::Op(&a), m::Divide(m::Op(&b), m::Op(&c))))) {
    TF_ASSIGN_OR_RETURN(
        auto a_times_c,
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            MakeBinaryHlo(HloOpcode::kMultiply, a, c), divide));
    TF_ASSIGN_OR_RETURN(auto new_divide,
                        MakeBinaryHlo(HloOpcode::kDivide, a_times_c, b));
    return ReplaceInstruction(divide, new_divide);
  }

  return Status::OK();
}

StatusOr<bool> AlgebraicSimplifierVisitor::RemoveDegenerateDimensionFromDot(
    HloInstruction* dot) {
  const Shape& lhs_shape = dot->operand(0)->shape();
  int64 num_degenerate_lhs_dims = 0;
  std::vector<int64> lhs_dimension_map(lhs_shape.rank(), -1);
  for (int64 i = 0; i < lhs_shape.rank(); ++i) {
    if (lhs_shape.dimensions(i) == 1) {
      ++num_degenerate_lhs_dims;
    } else {
      lhs_dimension_map[i] = i - num_degenerate_lhs_dims;
    }
  }

  const Shape& rhs_shape = dot->operand(1)->shape();
  int64 num_degenerate_rhs_dims = 0;
  std::vector<int64> rhs_dimension_map(rhs_shape.rank(), -1);
  for (int64 i = 0; i < rhs_shape.rank(); ++i) {
    if (rhs_shape.dimensions(i) == 1) {
      ++num_degenerate_rhs_dims;
    } else {
      rhs_dimension_map[i] = i - num_degenerate_rhs_dims;
    }
  }
  if (num_degenerate_lhs_dims == 0 && num_degenerate_rhs_dims == 0) {
    return false;
  }
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  DotDimensionNumbers new_dnums;
  for (int64 dim : dnums.lhs_batch_dimensions()) {
    int64 new_dim = lhs_dimension_map[dim];
    if (new_dim != -1) {
      new_dnums.add_lhs_batch_dimensions(new_dim);
    }
  }
  for (int64 dim : dnums.lhs_contracting_dimensions()) {
    int64 new_dim = lhs_dimension_map[dim];
    if (new_dim != -1) {
      new_dnums.add_lhs_contracting_dimensions(new_dim);
    }
  }

  for (int64 dim : dnums.rhs_batch_dimensions()) {
    int64 new_dim = rhs_dimension_map[dim];
    if (new_dim != -1) {
      new_dnums.add_rhs_batch_dimensions(new_dim);
    }
  }
  for (int64 dim : dnums.rhs_contracting_dimensions()) {
    int64 new_dim = rhs_dimension_map[dim];
    if (new_dim != -1) {
      new_dnums.add_rhs_contracting_dimensions(new_dim);
    }
  }

  HloInstruction* new_lhs =
      num_degenerate_lhs_dims > 0
          ? pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
                dot->parent()->AddInstruction(HloInstruction::CreateReshape(
                    ShapeUtil::DropDegenerateDimensions(lhs_shape),
                    dot->mutable_operand(0))),
                dot->operand(0))
          : dot->mutable_operand(0);
  HloInstruction* new_rhs =
      num_degenerate_rhs_dims > 0
          ? pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
                dot->parent()->AddInstruction(HloInstruction::CreateReshape(
                    ShapeUtil::DropDegenerateDimensions(rhs_shape),
                    dot->mutable_operand(1))),
                dot->operand(1))
          : dot->mutable_operand(1);
  TF_ASSIGN_OR_RETURN(
      auto new_dot,
      pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
          MakeDotHlo(new_lhs, new_rhs, new_dnums, dot->precision_config(),
                     /*preferred_element_type=*/dot->shape().element_type()),
          dot));
  if (ShapeUtil::Compatible(dot->shape(), new_dot->shape())) {
    TF_RETURN_IF_ERROR(ReplaceInstruction(dot, new_dot));
  } else {
    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
        dot, HloInstruction::CreateReshape(dot->shape(), new_dot)));
  }
  return true;
}

Status AlgebraicSimplifierVisitor::HandleDot(HloInstruction* dot) {
  CHECK(computation_ == dot->parent());
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));
  // Replace a zero element dot with a broadcast of the constant 0.
  if (ShapeUtil::IsZeroElementArray(dot->shape()) ||
      ShapeUtil::IsZeroElementArray(lhs->shape()) ||
      ShapeUtil::IsZeroElementArray(rhs->shape())) {
    auto zero = computation_->AddInstruction(
        simplifier_->CreateConstantWithLayoutUpdated(
            LiteralUtil::Zero(dot->shape().element_type())));
    return ReplaceWithNewInstruction(
        dot, HloInstruction::CreateBroadcast(dot->shape(), zero, {}));
  }

  // If either lhs or rhs is a zero tensor (all elements 0, rather than rank 0)
  // then just substitute the computation with a zero tensor.
  if (config_.enable_fast_math() &&
      (pp::algebraic_simplifier::util::IsAllFloat(lhs, 0) ||
       pp::algebraic_simplifier::util::IsAllFloat(rhs, 0))) {
    auto zero = computation_->AddInstruction(
        simplifier_->CreateConstantWithLayoutUpdated(
            LiteralUtil::Zero(dot->shape().element_type())));
    return ReplaceWithNewInstruction(
        dot, HloInstruction::CreateBroadcast(dot->shape(), zero, {}));
  }

  // If the lhs or rhs have only batch and contracting dimensions, a dot can be
  // rewritten as reduce(mul(broadcast(transpose(x)),broadcast(transpose(y))))
  TF_ASSIGN_OR_RETURN(
      HloInstruction * dot_strength_reduction_optimized,
      pp::algebraic_simplifier::dot::OptimizeDotStrengthReduction(this, dot));
  if (dot_strength_reduction_optimized) {
    VLOG(10) << " Replaced dot " << dot->ToString()
             << " with new dot operation: "
             << dot_strength_reduction_optimized->ToString();
    return ReplaceInstruction(dot, dot_strength_reduction_optimized);
  }

  // Simplify dot(reshape(transpose(A)), Const) to:
  // dot(reshape(A), reshape(transpose(reshape(Const)))), so that the reshape
  // and transpose on the Const side can be constant folded.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * dot_of_reorder_optimized,
      pp::algebraic_simplifier::dot::OptimizeDotOfReorderContractingDims(this,
                                                                         dot));
  if (dot_of_reorder_optimized) {
    VLOG(10) << " Replaced dot " << dot->ToString()
             << " with new dot operation: "
             << dot_of_reorder_optimized->ToString();
    return ReplaceInstruction(dot, dot_of_reorder_optimized);
  }

  TF_ASSIGN_OR_RETURN(
      HloInstruction * dot_of_concat_optimized,
      pp::algebraic_simplifier::dot::OptimizeDotOfConcat(this, dot));
  if (dot_of_concat_optimized) {
    VLOG(10) << "Replaced dot(concat(...), constant) with add(dot(..., "
                "constant)...)";
    return ReplaceInstruction(dot, dot_of_concat_optimized);
  }

  // Simplify dot(ConstA, Gather(Index, ConstB)) to:
  // Gather(Index, dot*(ConstA, ConstB)), where dot* is an appropriately
  // batched version of dot.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * dot_of_gather_optimized,
      pp::algebraic_simplifier::dot::OptimizeDotOfGather(this, dot));
  if (dot_of_gather_optimized) {
    VLOG(10) << "Replaced dot(constA, gather(i, constB)) with "
                "gather(i, dot*(constA, constB))";
    return ReplaceInstruction(dot, dot_of_gather_optimized);
  }

  TF_ASSIGN_OR_RETURN(bool removed_degenerate_dimensions,
                      RemoveDegenerateDimensionFromDot(dot));
  if (removed_degenerate_dimensions) {
    return Status::OK();
  }

  // Simplify dot(transpose(a), transpose(b)) to transpose(dot(b,a)).
  if (dot->dot_dimension_numbers().lhs_batch_dimensions_size() == 0 &&
      dot->dot_dimension_numbers().lhs_contracting_dimensions_size() == 1 &&
      dot->dot_dimension_numbers().lhs_contracting_dimensions(0) == 1 &&
      dot->dot_dimension_numbers().rhs_contracting_dimensions(0) == 0 &&
      lhs->IsRank2Transpose() && rhs->IsRank2Transpose()) {
    DotDimensionNumbers dot_dimension_numbers;
    dot_dimension_numbers.add_lhs_contracting_dimensions(1);
    dot_dimension_numbers.add_rhs_contracting_dimensions(0);
    auto new_dot =
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(HloInstruction::CreateDot(
                ShapeUtil::PermuteDimensions({1, 0}, dot->shape()),
                rhs->mutable_operand(0), lhs->mutable_operand(0),
                dot_dimension_numbers, dot->precision_config())),
            dot);
    return ReplaceWithNewInstruction(
        dot, HloInstruction::CreateTranspose(dot->shape(), new_dot, {1, 0}));
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleGather(HloInstruction* gather) {
  const Shape& operand_shape = gather->operand(0)->shape();
  // If the operand of a gather is very small, it is easier to fuse a
  // sequence of selects.
  if (operand_shape.rank() == 1 &&
      operand_shape.dimensions(0) <= very_small_gather_size &&
      gather->gather_dimension_numbers().index_vector_dim() ==
          gather->operand(1)->shape().rank() &&
      gather->gather_dimension_numbers().collapsed_slice_dims_size() == 1) {
    const Shape& index_shape = gather->operand(1)->shape();
    const int64 operand_elements = operand_shape.dimensions(0);
    auto get_value = [&](int64 i) {
      auto slice =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation_->AddInstruction(HloInstruction::CreateSlice(
                  ShapeUtil::MakeShape(operand_shape.element_type(), {1}),
                  gather->mutable_operand(0), {i}, {i + 1}, {1})),
              gather->operand(0));
      auto scalar =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation_->AddInstruction(HloInstruction::CreateReshape(
                  ShapeUtil::MakeShape(operand_shape.element_type(), {}),
                  slice)),
              slice);
      return pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
          computation_->AddInstruction(
              HloInstruction::CreateBroadcast(gather->shape(), scalar, {})),
          scalar);
    };
    auto result = get_value(0);
    auto one =
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::One(index_shape.element_type()))),
            gather->operand(1));
    auto index = one;
    auto pred_shape = ShapeUtil::ChangeElementType(gather->shape(), PRED);
    auto iter_shape = ShapeUtil::ChangeElementType(gather->shape(),
                                                   index_shape.element_type());
    for (int64 i = 1; i < operand_elements; ++i) {
      auto broadcasted_index =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation_->AddInstruction(
                  HloInstruction::CreateBroadcast(iter_shape, index, {})),
              index);
      auto index_mask =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation_->AddInstruction(HloInstruction::CreateCompare(
                  pred_shape, gather->mutable_operand(1), broadcasted_index,
                  ComparisonDirection::kGe)),
              broadcasted_index);
      result =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation_->AddInstruction(HloInstruction::CreateTernary(
                  gather->shape(), HloOpcode::kSelect, index_mask, get_value(i),
                  result)),
              index_mask);
      index =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation_->AddInstruction(HloInstruction::CreateBinary(
                  index->shape(), HloOpcode::kAdd, index, one)),
              index);
    }
    return ReplaceInstruction(gather, result);
  }
  return Status::OK();
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> MinMaxToClamp(
    HloInstruction* clamp_lower_bound_bcast, HloInstruction* to_clamp,
    HloInstruction* clamp_upper_bound_bcast) {
  HloInstruction* clamp_lower_bound;
  CHECK(Match(clamp_lower_bound_bcast,
              m::Broadcast(m::ConstantEffectiveScalar(&clamp_lower_bound))))
      << clamp_lower_bound_bcast->ToString();

  HloInstruction* clamp_upper_bound;
  CHECK(Match(clamp_upper_bound_bcast,
              m::Broadcast(m::ConstantEffectiveScalar(&clamp_upper_bound))))
      << clamp_upper_bound_bcast->ToString();

  const Literal& lower_bound =
      Cast<HloConstantInstruction>(clamp_lower_bound)->literal();
  const Literal& upper_bound =
      Cast<HloConstantInstruction>(clamp_upper_bound)->literal();

  std::unique_ptr<HloInstruction> lower_bound_instr =
      HloInstruction::CreateConstant(lower_bound.Clone());
  std::unique_ptr<HloInstruction> upper_bound_instr =
      HloInstruction::CreateConstant(upper_bound.Clone());

  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateCompare(
          ShapeUtil::ChangeElementType(lower_bound_instr->shape(), PRED),
          lower_bound_instr.get(), upper_bound_instr.get(),
          ComparisonDirection::kLt);

  HloEvaluator evaluator;
  TF_ASSIGN_OR_RETURN(auto result,
                      evaluator.Evaluate(cloned_instruction.get()));
  if (result.IsAll(true)) {
    return HloInstruction::CreateTernary(to_clamp->shape(), HloOpcode::kClamp,
                                         clamp_lower_bound_bcast, to_clamp,
                                         clamp_upper_bound_bcast);
  }
  return std::unique_ptr<HloInstruction>();
}
}  // namespace

Status AlgebraicSimplifierVisitor::HandleMaximum(HloInstruction* maximum) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(maximum, m::Maximum(m::Op(&lhs), m::Op(&rhs))));

  HloInstruction* clamp_upper_bound_bcast;
  HloInstruction* clamp_lower_bound_bcast;
  HloInstruction* to_clamp;
  if (Match(maximum, m::MaximumAnyOrder(
                         m::Broadcast(&clamp_lower_bound_bcast,
                                      m::ConstantEffectiveScalar()),
                         m::MinimumAnyOrder(
                             m::Op(&to_clamp),
                             m::Broadcast(&clamp_upper_bound_bcast,
                                          m::ConstantEffectiveScalar()))))) {
    TF_ASSIGN_OR_RETURN(auto clamp,
                        MinMaxToClamp(clamp_lower_bound_bcast, to_clamp,
                                      clamp_upper_bound_bcast));
    if (clamp) {
      return ReplaceWithNewInstruction(maximum, std::move(clamp));
    }
  }

  HloInstruction* clamp_lower_bound;
  HloInstruction* clamp_upper_bound;
  HloInstruction* max_operand;
  HloInstruction* clamp;
  if (Match(maximum,
            m::MaximumAnyOrder(
                m::Op(&max_operand),
                m::Clamp(&clamp, m::Op(&clamp_lower_bound), m::Op(&to_clamp),
                         m::Op(&clamp_upper_bound))))) {
    if (max_operand == clamp_lower_bound &&
        ReplaceInstructionIfSameShape(maximum, clamp)) {
      return Status::OK();
    }
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleMinimum(HloInstruction* minimum) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(minimum, m::Minimum(m::Op(&lhs), m::Op(&rhs))));

  HloInstruction* clamp_upper_bound_bcast;
  HloInstruction* clamp_lower_bound_bcast;
  HloInstruction* to_clamp;
  if (Match(minimum, m::MinimumAnyOrder(
                         m::Broadcast(&clamp_upper_bound_bcast,
                                      m::ConstantEffectiveScalar()),
                         m::MaximumAnyOrder(
                             m::Op(&to_clamp),
                             m::Broadcast(&clamp_lower_bound_bcast,
                                          m::ConstantEffectiveScalar()))))) {
    TF_ASSIGN_OR_RETURN(auto clamp,
                        MinMaxToClamp(clamp_lower_bound_bcast, to_clamp,
                                      clamp_upper_bound_bcast));
    if (clamp) {
      return ReplaceWithNewInstruction(minimum, std::move(clamp));
    }
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleClamp(HloInstruction* clamp) {
  HloInstruction* clamp_lower_bound;
  HloInstruction* clamp_upper_bound;
  HloInstruction* to_clamp;
  CHECK(Match(clamp, m::Clamp(m::Op(&clamp_lower_bound), m::Op(&to_clamp),
                              m::Op(&clamp_upper_bound))));

  // clamp(a, clamp(a, x, b), b) -> clamp(a, x, b)
  if (Match(to_clamp, m::Clamp(m::Op().Is(clamp_lower_bound), m::Op(),
                               m::Op().Is(clamp_upper_bound))) &&
      ReplaceInstructionIfSameShape(clamp, to_clamp)) {
    return Status::OK();
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleMultiply(HloInstruction* multiply) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(multiply, m::Multiply(m::Op(&lhs), m::Op(&rhs))));
  // A*1 => A
  VLOG(10) << "trying transform [A*1 => A]: " << multiply->ToString();
  if (pp::algebraic_simplifier::util::IsAll(rhs, 1) &&
      ReplaceInstructionIfSameShape(multiply, lhs)) {
    return Status::OK();
  }
  // 1*A => A
  VLOG(10) << "trying transform [1*A => A]: " << multiply->ToString();
  if (pp::algebraic_simplifier::util::IsAll(lhs, 1) &&
      ReplaceInstructionIfSameShape(multiply, rhs)) {
    return Status::OK();
  }

  // A*-1 => -A
  VLOG(10) << "trying transform [A*-1 => A]: " << multiply->ToString();
  if (pp::algebraic_simplifier::util::IsAll(rhs, -1)) {
    return ReplaceWithNewInstruction(
        multiply, HloInstruction::CreateUnary(multiply->shape(),
                                              HloOpcode::kNegate, lhs));
  }
  // -1*A => -A
  VLOG(10) << "trying transform [-1*A => A]: " << multiply->ToString();
  if (pp::algebraic_simplifier::util::IsAll(lhs, -1)) {
    return ReplaceWithNewInstruction(
        multiply, HloInstruction::CreateUnary(multiply->shape(),
                                              HloOpcode::kNegate, rhs));
  }

  // 0*A => 0. Only applies for integral types for correct NaN-handling, unless
  // fast math is enabled.
  if (pp::algebraic_simplifier::util::IsAll(lhs, 0) &&
      (primitive_util::IsIntegralType(multiply->shape().element_type()) ||
       config_.enable_fast_math()) &&
      ReplaceInstructionIfSameShape(multiply, lhs)) {
    return Status::OK();
  }
  // A*0 => 0
  if (pp::algebraic_simplifier::util::IsAll(rhs, 0) &&
      (primitive_util::IsIntegralType(multiply->shape().element_type()) ||
       config_.enable_fast_math()) &&
      ReplaceInstructionIfSameShape(multiply, rhs)) {
    return Status::OK();
  }

  VLOG(10) << "trying transform [(A * C1) * C2 => A * (C1 * C2)]";
  HloInstruction *a, *c1, *c2;
  if (Match(multiply,
            m::Multiply(m::Multiply(m::NonConstant(&a), m::Constant(&c1)),
                        m::Constant(&c2))) ||
      Match(multiply,
            m::Multiply(
                m::Multiply(m::Op(&a), m::Broadcast(m::ConstantScalar(&c1))),
                m::Broadcast(m::ConstantScalar(&c2))))) {
    TF_ASSIGN_OR_RETURN(
        auto* product_of_constants,
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            MakeBinaryHlo(HloOpcode::kMultiply, c1, c2), multiply));
    if (ShapeUtil::IsScalar(product_of_constants->shape()) &&
        !ShapeUtil::IsScalar(multiply->shape())) {
      product_of_constants =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation_->AddInstruction(HloInstruction::CreateBroadcast(
                  multiply->shape(), product_of_constants, {})),
              product_of_constants);
    }
    return ReplaceWithNewInstruction(
        multiply,
        HloInstruction::CreateBinary(multiply->shape(), HloOpcode::kMultiply, a,
                                     product_of_constants));
  }

  // exp(A) * exp(B) => exp(A+B)
  if (Match(multiply, m::Multiply(m::Exp(m::Op(&lhs)), m::Exp(m::Op(&rhs))))) {
    auto add =
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(HloInstruction::CreateBinary(
                multiply->shape(), HloOpcode::kAdd, lhs, rhs)),
            multiply);
    return ReplaceWithNewInstruction(
        multiply,
        HloInstruction::CreateUnary(multiply->shape(), HloOpcode::kExp, add));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleNegate(HloInstruction* negate) {
  // negate(negate(x)) => x
  HloInstruction* x;
  if (Match(negate, m::Negate(m::Negate(m::Op(&x)))) &&
      ReplaceInstructionIfSameShape(negate, x)) {
    return Status::OK();
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleNot(HloInstruction* logical_not) {
  // not(not(x)) => x
  HloInstruction* x;
  if (Match(logical_not, m::Not(m::Not(m::Op(&x)))) &&
      ReplaceInstructionIfSameShape(logical_not, x)) {
    return Status::OK();
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleOr(HloInstruction* logical_or) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(logical_or, m::Or(m::Op(&lhs), m::Op(&rhs))));

  // Simplify logical or
  if (ShapeUtil::HasPrimitiveType(lhs->shape(), xla::PRED) &&
      ShapeUtil::HasPrimitiveType(rhs->shape(), xla::PRED)) {
    // A || True => True
    VLOG(10) << "trying transform [A || True => True]: "
             << logical_or->ToString();
    if (pp::algebraic_simplifier::util::IsAll(rhs, 1) &&
        ReplaceInstructionIfSameShape(logical_or, rhs)) {
      return Status::OK();
    }
    // True || A => True
    VLOG(10) << "trying transform [True || A => True]: "
             << logical_or->ToString();
    if (pp::algebraic_simplifier::util::IsAll(lhs, 1) &&
        ReplaceInstructionIfSameShape(logical_or, lhs)) {
      return Status::OK();
    }

    // A || A => A
    VLOG(10) << "trying transform [A || A => A]: " << logical_or->ToString();
    if (lhs->Identical(*rhs) &&
        ReplaceInstructionIfSameShape(logical_or, lhs)) {
      return Status::OK();
    }
  }

  // A || False => A and A | 0 => A
  VLOG(10) << "trying transform [A || False => A]: " << logical_or->ToString();
  if (pp::algebraic_simplifier::util::IsAll(rhs, 0) &&
      ReplaceInstructionIfSameShape(logical_or, lhs)) {
    return Status::OK();
  }

  // False || A => A and 0 | A => A
  VLOG(10) << "trying transform [False || A => A]: " << logical_or->ToString();
  if (pp::algebraic_simplifier::util::IsAll(lhs, 0) &&
      ReplaceInstructionIfSameShape(logical_or, rhs)) {
    return Status::OK();
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleLog(HloInstruction* log) {
  // ln(exp(A)) => A
  VLOG(10) << "trying transform [ln(exp(A)) => A]: " << log->ToString();
  HloInstruction *a, *b;
  if (Match(log, m::Log(m::Exp(m::Op(&a)))) &&
      ReplaceInstructionIfSameShape(log, a)) {
    return Status::OK();
  }

  // ln(pow(A,B)) => B*ln(abs(A))
  if (Match(log, m::Log(m::Power(m::Op(&a), m::Op(&b))))) {
    auto abs_a =
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(
                HloInstruction::CreateUnary(log->shape(), HloOpcode::kAbs, a)),
            log);
    auto new_log =
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(HloInstruction::CreateUnary(
                log->shape(), HloOpcode::kLog, abs_a)),
            log);
    return ReplaceWithNewInstruction(
        log, HloInstruction::CreateBinary(log->shape(), HloOpcode::kMultiply,
                                          new_log, b));
  }

  if (Match(log, m::Log(m::Sqrt(m::Op(&a))))) {
    auto new_log = PreserveFrontendAttributesIfNeeded(
        computation_->AddInstruction(
            HloInstruction::CreateUnary(log->shape(), HloOpcode::kLog, a)),
        log);
    return ReplaceWithNewInstruction(
        log, HloInstruction::CreateBinary(log->shape(), HloOpcode::kMultiply,
                                          new_log, MakeScalarLike(log, 0.5)));
  }

  if (Match(log, m::Log(m::Rsqrt(m::Op(&a))))) {
    auto new_log = PreserveFrontendAttributesIfNeeded(
        computation_->AddInstruction(
            HloInstruction::CreateUnary(log->shape(), HloOpcode::kLog, a)),
        log);
    return ReplaceWithNewInstruction(
        log, HloInstruction::CreateBinary(log->shape(), HloOpcode::kMultiply,
                                          new_log, MakeScalarLike(log, -0.5)));
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleGetTupleElement(
    HloInstruction* get_tuple_element) {
  auto operand = get_tuple_element->mutable_operand(0);
  if (operand->opcode() == HloOpcode::kTuple) {
    // get_tuple_element(make_tuple({A_0, A_1, ..., A_n}), i) => A_i
    VLOG(10) << "trying transform "
             << "[get_tuple_element(make_tuple({...,A_i,...}), i)] => A_i: "
             << get_tuple_element->ToString();
    if (ReplaceInstructionIfSameShape(
            get_tuple_element,
            operand->mutable_operand(get_tuple_element->tuple_index()))) {
      return Status::OK();
    }
  }
  return Status::OK();
}

namespace {

absl::optional<std::vector<int64>> ReshapeLeavesDimensionsUnmodified(
    const HloInstruction* hlo, absl::Span<const int64> input_dim_indices) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kReshape);
  return ShapeUtil::ReshapeLeavesDimensionsUnmodified(
      hlo->operand(0)->shape(), hlo->shape(), input_dim_indices);
}

// Returns true if the output of "instruction" is a permutation of the
// elements of "operand". Precondition: "operand" is an operand of
// "instruction".
bool OutputIsPermutationOfOperandElements(HloInstruction* instruction,
                                          HloInstruction* operand) {
  DCHECK(!instruction->OperandIndices(operand).empty());
  switch (instruction->opcode()) {
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kTranspose:
      return true;
    case HloOpcode::kSort:
      return (!instruction->shape().IsTuple());
    default:
      return false;
  }
}

// Returns true if the output of "instruction" is a subset of the elements of
// "operand". Precondition: "operand" is an operand of "instruction".
bool OutputIsSubsetOfOperandElements(HloInstruction* instruction,
                                     HloInstruction* operand) {
  auto operand_indices = instruction->OperandIndices(operand);
  CHECK(!operand_indices.empty());
  if (operand_indices.size() != 1) {
    return false;
  }
  int64 operand_index = operand_indices[0];
  switch (instruction->opcode()) {
    case HloOpcode::kSlice:
      CHECK_EQ(0, operand_index);
      return true;
    case HloOpcode::kDynamicSlice:
      return operand_index == 0;
    default:
      return false;
  }
}

}  // namespace

Status AlgebraicSimplifierVisitor::HandleBroadcast(HloInstruction* broadcast) {
  HloInstruction* operand;
  CHECK(Match(broadcast, m::Broadcast(m::Op(&operand))));
  auto dims = broadcast->dimensions();
  // A degenerate broadcast of a reshape that does not change the number of
  // elements can be replaced by a reshape.
  if (std::is_sorted(dims.begin(), dims.end()) &&
      ShapeUtil::ElementsIn(broadcast->shape()) ==
          ShapeUtil::ElementsIn(operand->shape())) {
    VLOG(10) << "transform broadcast(X) -> reshape(X) where "
                "n(broadcast(X)) == n(X)";
    return ReplaceWithNewInstruction(
        broadcast, HloInstruction::CreateReshape(broadcast->shape(), operand));
  }

  // A degenerate broadcast that has the same input and output rank can be
  // converted into a transpose.
  if (broadcast->shape().rank() == operand->shape().rank() &&
      ShapeUtil::ElementsIn(broadcast->shape()) ==
          ShapeUtil::ElementsIn(operand->shape())) {
    VLOG(10) << "transform broadcast(X) -> transpose(X) where "
                "n(broadcast(X)) == n(X)";
    return ReplaceWithNewInstruction(
        broadcast,
        HloInstruction::CreateTranspose(broadcast->shape(), operand, dims));
  }

  // A broadcast of a reshape which merely inserts 1-sized dimensions can
  // elide its operand.
  {
    bool merely_inserts_or_deletes_1_sized_dimensions;
    std::vector<int64> inserted_indices, deleted_indices;
    std::tie(merely_inserts_or_deletes_1_sized_dimensions, deleted_indices,
             inserted_indices) =
        operand->ReshapeMerelyInsertsOrDeletes1SizedDimensions();
    if (merely_inserts_or_deletes_1_sized_dimensions &&
        deleted_indices.empty()) {
      std::reverse(inserted_indices.begin(), inserted_indices.end());
      for (auto inserted_index : inserted_indices) {
        dims.erase(dims.begin() + inserted_index);
      }
      return ReplaceWithNewInstruction(
          broadcast,
          HloInstruction::CreateBroadcast(broadcast->shape(),
                                          operand->mutable_operand(0), dims));
    }
  }

  // A Broadcast that feeds a unary element-wise operation can sink the
  // broadcast after the unary element-wise operation.
  TF_ASSIGN_OR_RETURN(
      bool sink_succeeded,
      TryToSinkBroadcastAfterOpWithUniqueNonScalarOperand(broadcast));
  changed_ |= sink_succeeded;
  if (sink_succeeded) {
    return Status::OK();
  }

  // A scalar broadcast feeding an instruction which only permutes (reshape,
  // transpose, sort, reverse) or selects a subset of operand elements (slice,
  // dynamic slice) can be replaced with a broadcast directly to the output
  // shape of the instruction.
  if (ShapeUtil::IsScalar(operand->shape())) {
    for (HloInstruction* user : broadcast->users()) {
      // Skip if the broadcast user has no uses itself.
      if (user->user_count() == 0 && user != computation_->root_instruction()) {
        continue;
      }
      if (OutputIsPermutationOfOperandElements(user, broadcast) ||
          OutputIsSubsetOfOperandElements(user, broadcast)) {
        VLOG(10) << "transform permuting/subset  of a scalar broadcast into "
                 << "a single broadcast";
        HloInstruction* new_broadcast =
            pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
                computation_->AddInstruction(HloInstruction::CreateBroadcast(
                    user->shape(), operand, {})),
                operand);
        // Use HloInstruction::ReplaceAllUsesWith instead of
        // HloComputation::ReplaceWithNewInstruction because we are replacing an
        // instruction other than the visited instruction.
        changed_ = true;
        return user->ReplaceAllUsesWith(new_broadcast);
      }
    }
    return Status::OK();
  }

  // broadcast(iota) -> iota.
  if (operand->opcode() == HloOpcode::kIota) {
    return ReplaceWithNewInstruction(
        broadcast,
        HloInstruction::CreateIota(
            broadcast->shape(),
            dims[Cast<HloIotaInstruction>(operand)->iota_dimension()]));
  }

  // Merge two consecutive broadcasts into a single one.
  if (operand->opcode() == HloOpcode::kBroadcast) {
    std::vector<int64> new_dimensions;
    for (auto dim : operand->dimensions()) {
      new_dimensions.push_back(dims[dim]);
    }
    return ReplaceWithNewInstruction(
        broadcast,
        HloInstruction::CreateBroadcast(
            broadcast->shape(), operand->mutable_operand(0), new_dimensions));
  }

  return Status::OK();
}

namespace {
StatusOr<ComparisonDirection> inverse_comparison_direction(
    ComparisonDirection direction) {
  switch (direction) {
    case ComparisonDirection::kEq:
      return ComparisonDirection::kEq;
    case ComparisonDirection::kGt:
      return ComparisonDirection::kLt;
    case ComparisonDirection::kGe:
      return ComparisonDirection::kLe;
    case ComparisonDirection::kLt:
      return ComparisonDirection::kGt;
    case ComparisonDirection::kLe:
      return ComparisonDirection::kGe;
    case ComparisonDirection::kNe:
      return ComparisonDirection::kNe;
    default:
      return FailedPrecondition("Invalid direction %s",
                                ComparisonDirectionToString(direction));
  }
}
}  // namespace

Status AlgebraicSimplifierVisitor::HandleCompare(HloInstruction* compare) {
  HloInstruction* lhs;
  HloInstruction* rhs;
  CHECK(Match(compare, m::Compare(m::Op(&lhs), m::Op(&rhs))));
  {
    // compare(broadcast(a) + x, broadcast(b)) ==>
    //   compare(x, broadcast(b-a)), only enabled for integral types.
    HloInstruction *x, *a, *b;
    if (Match(compare,
              m::Compare(
                  m::AddAnyOrder(m::Op(&x), m::Broadcast(m::Op(&a).WithShape(
                                                m::Shape().IsScalar()))),
                  m::Broadcast(m::Op(&b).WithShape(m::Shape().IsScalar()))))) {
      if (ShapeUtil::ElementIsSigned(x->shape()) &&
          ShapeUtil::ElementIsIntegral(x->shape())) {
        HloInstruction* sub =
            computation_->AddInstruction(HloInstruction::CreateBinary(
                b->shape(), HloOpcode::kSubtract, b, a));
        HloInstruction* broadcast = computation_->AddInstruction(
            HloInstruction::CreateBroadcast(x->shape(), sub, {}));
        HloInstruction* new_compare = PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(
                HloInstruction::CreateCompare(compare->shape(), x, broadcast,
                                              compare->comparison_direction())),
            compare);
        return ReplaceInstruction(compare, new_compare);
      }
    }
  }

  {
    HloInstruction* lhs_delta;
    // Canonicalizing: Replacing compare(X +/- C, Y) => compare(X, Y -/+ C)
    // This allows constant folding on the right hand side later.
    if (Match(compare,
              m::Compare(m::Add(m::Op(&lhs), m::ConstantScalar(&lhs_delta)),
                         m::Op(&rhs)))) {
      const HloInstruction* add = compare->operand(0);
      return ReplaceWithNewInstruction(
          compare,
          compare->CloneWithNewOperands(
              compare->shape(),
              {lhs, computation_->AddInstruction(HloInstruction::CreateBinary(
                        add->shape(), HloOpcode::kSubtract, rhs, lhs_delta))}));
    } else if (Match(compare,
                     m::Compare(m::Add(m::Op(&lhs), m::Negate(m::ConstantScalar(
                                                        &lhs_delta))),
                                m::Op(&rhs)))) {
      const HloInstruction* add = compare->operand(0);
      return ReplaceWithNewInstruction(
          compare,
          compare->CloneWithNewOperands(
              compare->shape(),
              {lhs, computation_->AddInstruction(HloInstruction::CreateBinary(
                        add->shape(), HloOpcode::kAdd, rhs, lhs_delta))}));
    } else if (Match(compare,
                     m::Compare(m::Negate(m::Op(&lhs)), m::Constant(&rhs)))) {
      HloInstruction* new_rhs = computation_->AddInstruction(
          HloInstruction::CreateUnary(rhs->shape(), HloOpcode::kNegate, rhs));
      TF_ASSIGN_OR_RETURN(
          auto new_direction,
          inverse_comparison_direction(compare->comparison_direction()));
      return ReplaceWithNewInstruction(
          compare, compare->CreateCompare(compare->shape(), lhs, new_rhs,
                                          new_direction));
    }
  }

  if (Cast<HloCompareInstruction>(compare)->type() ==
      Comparison::Type::kUnsigned) {
    // X u<  0 -> false
    if (compare->comparison_direction() == ComparisonDirection::kLt &&
        pp::algebraic_simplifier::util::IsAll(rhs, 0)) {
      return ReplaceInstruction(compare, MakeScalarLike(compare, false));
    }
    // X u>= 0 -> true
    if (compare->comparison_direction() == ComparisonDirection::kGe &&
        pp::algebraic_simplifier::util::IsAll(rhs, 0)) {
      return ReplaceInstruction(compare, MakeScalarLike(compare, true));
    }
    // 0 u>  X -> false
    if (compare->comparison_direction() == ComparisonDirection::kGt &&
        pp::algebraic_simplifier::util::IsAll(lhs, 0)) {
      return ReplaceInstruction(compare, MakeScalarLike(compare, false));
    }
    // 0 u<= X -> true
    if (compare->comparison_direction() == ComparisonDirection::kLe &&
        pp::algebraic_simplifier::util::IsAll(lhs, 0)) {
      return ReplaceInstruction(compare, MakeScalarLike(compare, true));
    }
  }

  auto replace_with_pred_broadcast = [&](bool value) {
    return ReplaceWithNewInstruction(
        compare,
        HloInstruction::CreateBroadcast(
            compare->shape(),
            computation_->AddInstruction(
                HloInstruction::CreateConstant(LiteralUtil::CreateR0(value))),
            {}));
  };
  if (compare->comparison_direction() == ComparisonDirection::kLt &&
      lhs->opcode() == HloOpcode::kIota &&
      pp::algebraic_simplifier::util::IsAll(rhs, 0)) {
    return replace_with_pred_broadcast(false);
  } else if (compare->comparison_direction() == ComparisonDirection::kGt &&
             pp::algebraic_simplifier::util::IsAll(lhs, 0) &&
             rhs->opcode() == HloOpcode::kIota) {
    return replace_with_pred_broadcast(false);
  } else if (compare->comparison_direction() == ComparisonDirection::kGe &&
             lhs->opcode() == HloOpcode::kIota &&
             pp::algebraic_simplifier::util::IsAll(rhs, 0)) {
    return replace_with_pred_broadcast(true);
  } else if (compare->comparison_direction() == ComparisonDirection::kLe &&
             pp::algebraic_simplifier::util::IsAll(lhs, 0) &&
             rhs->opcode() == HloOpcode::kIota) {
    return replace_with_pred_broadcast(true);
  }
  if (lhs == rhs &&
      primitive_util::IsIntegralType(lhs->shape().element_type())) {
    switch (compare->comparison_direction()) {
      case ComparisonDirection::kGt:
      case ComparisonDirection::kLt:
      case ComparisonDirection::kNe:
        return replace_with_pred_broadcast(false);
      case ComparisonDirection::kEq:
      case ComparisonDirection::kGe:
      case ComparisonDirection::kLe:
        return replace_with_pred_broadcast(true);
    }
  }
  return Status::OK();
}

// A conversion to the same element type as the operand is a nop and can be
// removed.  A conversion of a constant can be simplified by making a new
// constant.
Status AlgebraicSimplifierVisitor::HandleConvert(HloInstruction* convert) {
  PrimitiveType src_type = convert->operand(0)->shape().element_type();
  PrimitiveType dest_type = convert->shape().element_type();
  if (src_type == dest_type) {
    return ReplaceInstruction(convert, convert->mutable_operand(0));
  }
  return Status::OK();
}

// Complex(Real(c), Imag(c)) -> c
Status AlgebraicSimplifierVisitor::HandleComplex(HloInstruction* complex) {
  HloInstruction *c0, *c1;
  if (Match(complex, m::Complex(m::Real(m::Op(&c0)), m::Imag(m::Op(&c1)))) &&
      c0 == c1) {
    return ReplaceInstruction(complex, c0);
  }
  return Status::OK();
}

// Real(Complex(r, i)) -> r
Status AlgebraicSimplifierVisitor::HandleReal(HloInstruction* real) {
  HloInstruction* op;
  if (Match(real, m::Real(m::Complex(m::Op(&op), m::Op())))) {
    return ReplaceInstruction(real, op);
  }
  return Status::OK();
}

// Imag(Complex(r, i)) -> i
Status AlgebraicSimplifierVisitor::HandleImag(HloInstruction* imag) {
  HloInstruction* op;
  if (Match(imag, m::Imag(m::Complex(m::Op(), m::Op(&op))))) {
    return ReplaceInstruction(imag, op);
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleIota(HloInstruction* instruction) {
  // iota -> zero if the iota dimension never produces an element other than
  // zero.
  auto* iota = Cast<HloIotaInstruction>(instruction);
  if (iota->shape().dimensions(iota->iota_dimension()) <= 1) {
    auto zero = computation_->AddInstruction(
        simplifier_->CreateConstantWithLayoutUpdated(
            LiteralUtil::Zero(iota->shape().element_type()).Clone()));
    return ReplaceWithNewInstruction(
        iota, HloInstruction::CreateBroadcast(iota->shape(), zero, {}));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandlePad(HloInstruction* pad) {
  if (ShapeUtil::IsZeroElementArray(pad->operand(0)->shape())) {
    return ReplaceWithNewInstruction(
        pad, HloInstruction::CreateBroadcast(pad->shape(),
                                             pad->mutable_operand(1), {}));
  }

  // Interior padding on one sized dimensions have no effect. As a result it
  // makes other simplifications possible if there is no interior padding.
  if (HasInteriorPadding(pad->padding_config())) {
    PaddingConfig padding_config = pad->padding_config();
    bool cleared_interior_padding = false;
    for (int64 i = 0; i < pad->shape().rank(); ++i) {
      if (padding_config.dimensions(i).interior_padding() > 0 &&
          pad->operand(0)->shape().dimensions(i) == 1) {
        cleared_interior_padding = true;
        padding_config.mutable_dimensions(i)->set_interior_padding(0);
      }
    }
    if (cleared_interior_padding) {
      return ReplaceWithNewInstruction(
          pad,
          HloInstruction::CreatePad(pad->shape(), pad->mutable_operand(0),
                                    pad->mutable_operand(1), padding_config));
    }
  }

  // Eliminate nop pads (padding all zero), and replace a pad with negative
  // padding with a pad with non-negative padding followed by a slice.
  bool all_zero = true;
  bool has_negative = false;
  for (auto& padding_dimension : pad->padding_config().dimensions()) {
    if (padding_dimension.edge_padding_low() < 0 ||
        padding_dimension.edge_padding_high() < 0) {
      has_negative = true;
    }
    if (padding_dimension.edge_padding_low() != 0 ||
        padding_dimension.edge_padding_high() != 0) {
      all_zero = false;
    }
  }

  if (all_zero) {
    ReplaceInstructionIfSameShape(pad, pad->mutable_operand(0));
    return Status::OK();
  }

  if (has_negative) {
    // Pad has negative padding. Replace with a pad with the non-negative
    // padding followed by a slice which effectively performs the negative
    // padding.
    // TODO(b/34628603): Add support for negative padding in the backends, or
    // change kPad semantics to dpp::algebraic_simplifier::util::IsAllow
    // negative padding and use slice instead.

    // First construct the padding config with non-negative entries and the
    // compute the shape of this new pad instruction.
    PaddingConfig nonzero_padding = pad->padding_config();
    for (int i = 0; i < pad->padding_config().dimensions_size(); ++i) {
      PaddingConfig::PaddingConfigDimension* padding_dimension =
          nonzero_padding.mutable_dimensions(i);
      // Set negative padding to zero.
      if (padding_dimension->edge_padding_low() < 0) {
        padding_dimension->set_edge_padding_low(0);
      }
      if (padding_dimension->edge_padding_high() < 0) {
        padding_dimension->set_edge_padding_high(0);
      }
    }

    TF_ASSIGN_OR_RETURN(HloInstruction * nonzero_pad,
                        MakePadHlo(pad->mutable_operand(0),
                                   pad->mutable_operand(1), nonzero_padding));
    // Copy the layout from the original pad instructions. The new pad and the
    // slice instruction should all have the same layout.
    TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
        pad->shape(), nonzero_pad->mutable_shape()));

    // Second, construct the slice instruction to perform the negative padding.
    std::vector<int64> start_indices;
    std::vector<int64> end_indices;
    std::vector<int64> strides;
    for (int64 i = 0; i < pad->padding_config().dimensions_size(); ++i) {
      const PaddingConfig::PaddingConfigDimension& padding_dimension =
          pad->padding_config().dimensions(i);
      int64 start = 0;
      if (padding_dimension.edge_padding_low() < 0) {
        start = -1 * padding_dimension.edge_padding_low();
      }
      int64 end = nonzero_pad->shape().dimensions(i);
      if (padding_dimension.edge_padding_high() < 0) {
        end += padding_dimension.edge_padding_high();
      }
      start_indices.push_back(start);
      end_indices.push_back(end);
      strides.push_back(1);
    }

    TF_ASSIGN_OR_RETURN(
        HloInstruction * slice,
        MakeSliceHlo(nonzero_pad, start_indices, end_indices, strides));
    TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
        pad->shape(), slice->mutable_shape()));

    // Verify that the slice shape matches the pad shape.
    TF_RET_CHECK(ShapeUtil::Equal(slice->shape(), pad->shape()));

    return ReplaceInstruction(pad, slice);
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandlePower(HloInstruction* power) {
  VLOG(10) << "trying transform [pow(A, 0) => 1]: " << power->ToString();
  HloInstruction *lhs, *rhs;
  CHECK(Match(power, m::Power(m::Op(&lhs), m::Op(&rhs))));
  if (pp::algebraic_simplifier::util::IsAll(rhs, 0)) {
    auto one = simplifier_->CreateConstantWithLayoutUpdated(
        LiteralUtil::One(power->shape().element_type()).Clone());
    std::unique_ptr<HloInstruction> ones;
    if (ShapeUtil::IsScalar(power->shape())) {
      ones = std::move(one);
    } else {
      ones = HloInstruction::CreateBroadcast(
          power->shape(), computation_->AddInstruction(std::move(one)), {});
    }
    return ReplaceWithNewInstruction(power, std::move(ones));
  }

  VLOG(10) << "trying transform [pow(A, 1) => A]: " << power->ToString();
  if (pp::algebraic_simplifier::util::IsAll(rhs, 1) &&
      ReplaceInstructionIfSameShape(power, lhs)) {
    return Status::OK();
  }

  // pow(exp(A),B) => exp(A*B)
  HloInstruction *a, *b;
  if (Match(power, m::Power(m::Exp(m::Op(&a)), m::Op(&b)))) {
    auto a_times_b =
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(HloInstruction::CreateBinary(
                power->shape(), HloOpcode::kMultiply, a, b)),
            power);
    return ReplaceWithNewInstruction(
        power, HloInstruction::CreateUnary(power->shape(), HloOpcode::kExp,
                                           a_times_b));
  }
  VLOG(10) << "trying transform [pow(A, 2) => A*A]: " << power->ToString();
  if (pp::algebraic_simplifier::util::IsAll(rhs, 2)) {
    return ReplaceWithNewInstruction(
        power, HloInstruction::CreateBinary(power->shape(),
                                            HloOpcode::kMultiply, lhs, lhs));
  }

  VLOG(10) << "trying transform [pow(A, -2) => 1 / (A*A)]: "
           << power->ToString();
  if (pp::algebraic_simplifier::util::IsAll(rhs, -2)) {
    auto* one = computation_->AddInstruction(
        simplifier_->CreateConstantWithLayoutUpdated(
            LiteralUtil::One(rhs->shape().element_type()).Clone()));

    auto* broadcast_one = computation_->AddInstruction(
        HloInstruction::CreateBroadcast(power->shape(), one, {}));

    auto* sq = computation_->AddInstruction(HloInstruction::CreateBinary(
        broadcast_one->shape(), HloOpcode::kMultiply, lhs, lhs));

    return ReplaceWithNewInstruction(
        power, HloInstruction::CreateBinary(sq->shape(), HloOpcode::kDivide,
                                            broadcast_one, sq));
  }

  VLOG(10) << "trying transform [pow(A, -1) => 1/A]: " << power->ToString();
  if (pp::algebraic_simplifier::util::IsAll(rhs, -1)) {
    auto* one = computation_->AddInstruction(
        simplifier_->CreateConstantWithLayoutUpdated(
            LiteralUtil::One(rhs->shape().element_type()).Clone()));

    // Explicitly broadcast scalar 1 to the output shape, to avoid implicit
    // broadcast in divide HLO as we are trying to eliminate implicit
    // broadcasting at HLO level.
    auto* broadcast_one = computation_->AddInstruction(
        HloInstruction::CreateBroadcast(power->shape(), one, {}));
    return ReplaceWithNewInstruction(
        power, HloInstruction::CreateBinary(power->shape(), HloOpcode::kDivide,
                                            broadcast_one, lhs));
  }

  if (config_.enable_fast_math()) {
    VLOG(10) << "trying transform [pow(A, 0.5) => sqrt(A)]: "
             << power->ToString();
    if (pp::algebraic_simplifier::util::IsAllFloat(rhs, 0.5)) {
      return ReplaceWithNewInstruction(
          power,
          HloInstruction::CreateUnary(lhs->shape(), HloOpcode::kSqrt, lhs));
    }

    VLOG(10) << "trying transform [pow(A, -0.5) => 1/sqrt(A)]: "
             << power->ToString();
    if (pp::algebraic_simplifier::util::IsAllFloat(rhs, -0.5)) {
      return ReplaceWithNewInstruction(
          power,
          HloInstruction::CreateUnary(lhs->shape(), HloOpcode::kRsqrt, lhs));
    }
  }

  VLOG(10) << "trying transform [pow(pow(A, X), Y) => pow(A, X*Y)]: "
           << power->ToString();

  // Don't perform this optimization if either of the exponents is complex; this
  // identity is true only for real-valued exponents.  In addition, we cowardly
  // refuse to do this transformation if the two expontents have different
  // element types.
  if (lhs->opcode() == HloOpcode::kPower &&
      !ShapeUtil::ElementIsComplex(lhs->operand(1)->shape()) &&
      !ShapeUtil::ElementIsComplex(rhs->shape()) &&
      ShapeUtil::SameElementType(lhs->operand(1)->shape(), rhs->shape())) {
    auto exponent_product =
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(
                HloInstruction::CreateBinary(rhs->shape(), HloOpcode::kMultiply,
                                             lhs->mutable_operand(1), rhs)),
            power);
    return ReplaceWithNewInstruction(
        power, HloInstruction::CreateBinary(power->shape(), HloOpcode::kPower,
                                            lhs->mutable_operand(0),
                                            exponent_product));
  }

  return Status::OK();
}

namespace {
bool IsHloInstructionElementWiseWithCompatibleBroadcastOperands(
    const HloInstruction* instruction) {
  if (!instruction->IsElementwise() || instruction->operand_count() == 0) {
    return false;
  }

  // Fetch the first operand and make sure it's a broadcast instruction. We'll
  // use it to compare to the other operands.
  const auto* broadcast = instruction->operand(0);

  if (broadcast->opcode() != HloOpcode::kBroadcast) {
    return false;
  }

  const auto is_operand_compatible =
      [&broadcast](const HloInstruction* operand) {
        return operand->shape() == broadcast->shape() &&
               operand->dimensions() == broadcast->dimensions();
      };

  // Check for all operands whether their shapes and dimensions are equal.
  const auto& operands = instruction->operands();

  return std::all_of(operands.begin(), operands.end(),
                     std::move(is_operand_compatible));
}

bool ShouldSinkBroadcastAndOperandWithSameUser(const HloInstruction* user,
                                               const HloInstruction* broadcast,
                                               const HloInstruction* operand) {
  return broadcast == operand ||
         IsHloInstructionElementWiseWithCompatibleBroadcastOperands(user);
}

bool IsHloInstructionBroadcastWithScalarOperand(
    const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kBroadcast &&
         ShapeUtil::IsScalar(instruction->operand(0)->shape());
}
}  // namespace

StatusOr<bool>
AlgebraicSimplifierVisitor::TryToSinkBroadcastAfterOpWithUniqueNonScalarOperand(
    HloInstruction* broadcast) {
  TF_RET_CHECK(broadcast->opcode() == HloOpcode::kBroadcast);
  bool changed = false;
  if (ShapeUtil::IsScalar(broadcast->shape())) {
    return false;
  }
  HloInstruction* operand = broadcast->mutable_operand(0);
  for (HloInstruction* user : broadcast->users()) {
    if (user->user_count() == 0 && user != computation_->root_instruction()) {
      continue;
    }
    // Do not move reshapes or broadcasts past copies since the shape the copy
    // will operate on will change.
    if (user->opcode() == HloOpcode::kCopy) {
      continue;
    }
    // Do not change the shape of fusion nodes in case there a multiple shapes
    // inside the fusion node already.
    if (user->opcode() == HloOpcode::kFusion) {
      continue;
    }
    if (!user->IsElementwise()) {
      continue;
    }

    // Find the unique non-scalar operand or continue if there isn't one.
    int64 scalar_broadcast_count = 0;
    int64 broadcast_use_count = 0;
    for (HloInstruction* user_operand : user->operands()) {
      if (user_operand->opcode() == HloOpcode::kBroadcast &&
          ShapeUtil::IsScalar(user_operand->operand(0)->shape())) {
        ++scalar_broadcast_count;
      } else if (broadcast == user_operand) {
        ++broadcast_use_count;
      }
    }
    if (scalar_broadcast_count + broadcast_use_count != user->operand_count()) {
      continue;
    }
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(user->operand_count());

    Shape changed_shape;

    for (auto* user_operand : user->operands()) {
      if (ShouldSinkBroadcastAndOperandWithSameUser(user, broadcast,
                                                    user_operand)) {
        // Sink original broadcast X -b-> Y => -b-> X -> Y.
        new_operands.emplace_back(user_operand->mutable_operand(0));
      } else if (IsHloInstructionBroadcastWithScalarOperand(user_operand)) {
        // Make sure that any scalar operand gets the same shape as the user.
        changed_shape = ShapeUtil::ChangeElementType(
            operand->shape(), user_operand->shape().element_type());
        simplifier_->UpdateLayout(&changed_shape);
        new_operands.push_back(
            computation_->AddInstruction(HloInstruction::CreateBroadcast(
                changed_shape, user_operand->mutable_operand(0), {})));
      } else {
        // No optimization possible, so just insert the old operand.
        new_operands.emplace_back(operand);
      }
    }

    VLOG(4) << "Sinking broadcast after user:";
    VLOG(4) << "  old broadcast: " << broadcast->ToString();
    VLOG(4) << "  old user: " << user->ToString();
    changed_shape = ShapeUtil::ChangeElementType(operand->shape(),
                                                 user->shape().element_type());
    simplifier_->UpdateLayout(&changed_shape);

    HloInstruction* new_user = computation_->AddInstruction(
        user->CloneWithNewOperands(changed_shape, new_operands));
    VLOG(4) << "  new user: " << new_user->ToString();
    HloInstruction* new_broadcast =
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(HloInstruction::CreateBroadcast(
                user->shape(), new_user, broadcast->dimensions())),
            broadcast);
    VLOG(4) << "  new broadcast: " << new_broadcast->ToString();
    TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(new_broadcast));

    changed = true;

    TF_RETURN_IF_ERROR(
        TryToSinkBroadcastAfterOpWithUniqueNonScalarOperand(new_broadcast)
            .status());
  }

  return changed;
}

namespace {
template <typename T>
std::unique_ptr<HloInstruction> TryRemainderToAnd(
    HloInstruction* remainder, HloComputation* computation,
    PoplarAlgebraicSimplifier* simplifier) {
  HloInstruction *a, *b, *c;
  CHECK(Match(remainder, m::Remainder(m::Op(&a), m::Op(&b))));

  if (ShapeUtil::ElementIsIntegral(remainder->shape()) &&
      !Match(b, m::ConstantEffectiveScalar(&c)) &&
      !Match(b, m::Broadcast(m::ConstantEffectiveScalar(&c)))) {
    return nullptr;
  }

  if (ShapeUtil::ElementIsSigned(remainder->shape())) {
    int64 b_value = c->literal().GetFirstElement<T>();
    if (b_value > 0 && IsPowerOfTwo(static_cast<uint64>(b_value))) {
      // Handle negative dividends by negating the result of the division.
      HloInstruction* zero_like_a = BroadcastZeros(
          computation, a->shape().element_type(), a->shape().dimensions());

      auto* dividend_is_negative =
          computation->AddInstruction(HloInstruction::CreateCompare(
              ShapeUtil::ChangeElementType(a->shape(), PRED), a, zero_like_a,
              ComparisonDirection::kLt));

      auto* negated_dividend =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation->AddInstruction(HloInstruction::CreateUnary(
                  a->shape(), HloOpcode::kNegate, a)),
              a);

      auto* abs_dividend =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation->AddInstruction(HloInstruction::CreateTernary(
                  a->shape(), HloOpcode::kSelect, dividend_is_negative,
                  negated_dividend, a)),
              a);

      auto* mask_amount = computation->AddInstruction(
          simplifier->CreateConstantWithLayoutUpdated(
              LiteralUtil::CreateR0<T>(b_value - 1)));
      if (!ShapeUtil::IsScalar(b->shape())) {
        mask_amount =
            pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
                computation->AddInstruction(HloInstruction::CreateBroadcast(
                    b->shape(), mask_amount, {})),
                b);
      }

      auto* quotient =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation->AddInstruction(HloInstruction::CreateBinary(
                  remainder->shape(), HloOpcode::kAnd, abs_dividend,
                  mask_amount)),
              remainder);

      auto* neqated_quotient =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation->AddInstruction(HloInstruction::CreateUnary(
                  quotient->shape(), HloOpcode::kNegate, quotient)),
              quotient);

      return HloInstruction::CreateTernary(
          remainder->shape(), HloOpcode::kSelect, dividend_is_negative,
          neqated_quotient, quotient);
    }
  } else {
    uint64 b_value = c->literal().GetFirstElement<T>();
    if (IsPowerOfTwo(b_value)) {
      HloInstruction* mask_amount =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation->AddInstruction(
                  simplifier->CreateConstantWithLayoutUpdated(
                      LiteralUtil::CreateR0<T>(b_value - 1))),
              c);
      if (!ShapeUtil::IsScalar(b->shape())) {
        mask_amount =
            pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
                computation->AddInstruction(HloInstruction::CreateBroadcast(
                    b->shape(), mask_amount, {})),
                mask_amount);
      }
      return HloInstruction::CreateBinary(remainder->shape(), HloOpcode::kAnd,
                                          a, mask_amount);
    }
  }
  return nullptr;
}
}  // namespace

Status AlgebraicSimplifierVisitor::HandleRemainder(HloInstruction* remainder) {
  HloInstruction *a, *b;
  CHECK(Match(remainder, m::Remainder(m::Op(&a), m::Op(&b))));

  // (A % B) % B == A % B.
  if (Match(a, m::Remainder(m::Op(), m::Op().Is(b)))) {
    return ReplaceInstruction(remainder, a);
  }

  // A % B => A & (B - 1) if B is a power of 2.
  switch (remainder->shape().element_type()) {
    case S8:
      if (std::unique_ptr<HloInstruction> shift =
              TryRemainderToAnd<int8>(remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case S16:
      if (std::unique_ptr<HloInstruction> shift =
              TryRemainderToAnd<int16>(remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case S32:
      if (std::unique_ptr<HloInstruction> shift =
              TryRemainderToAnd<int32>(remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case S64:
      if (std::unique_ptr<HloInstruction> shift =
              TryRemainderToAnd<int64>(remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case U8:
      if (std::unique_ptr<HloInstruction> shift =
              TryRemainderToAnd<uint8>(remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case U16:
      if (std::unique_ptr<HloInstruction> shift =
              TryRemainderToAnd<uint16>(remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case U32:
      if (std::unique_ptr<HloInstruction> shift =
              TryRemainderToAnd<uint32>(remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case U64:
      if (std::unique_ptr<HloInstruction> shift =
              TryRemainderToAnd<uint64>(remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    default:
      break;
  }

  // If M < N, then {0, ..., M} % N ==> {0, ..., M}.
  //
  // Currently this only covers the case when N is a broadcasted constant
  // scalar.  We could also cover the case when N is a non-broadcasted constant
  // with the same value repeated.
  HloInstruction* iota;
  HloInstruction* divisor;
  if (Match(remainder,
            m::Remainder(m::Iota(&iota),
                         m::Broadcast(m::ConstantEffectiveScalar(&divisor))))) {
    // The iota counts {0, ..., iota_upper_bound - 1}.  (Actually this is
    // conservative; the iota may overflow and count up to a smaller value than
    // this.  But that's OK for our purposes here.)
    int64 iota_upper_bound = iota->shape().dimensions(
        Cast<HloIotaInstruction>(iota)->iota_dimension());
    absl::optional<int64> divisor_val = divisor->literal().GetIntegralAsS64(
        std::vector<int64>(0, divisor->shape().dimensions_size()));
    if (divisor_val && *divisor_val >= iota_upper_bound) {
      return ReplaceInstruction(remainder, iota);
    }
  }

  // (X + N) % N = X % N, so long as X + N does not overflow.
  //
  // We don't have range tracking in XLA that would let us know whether X + N
  // overflows, so for now we only do this simplification when X is an iota.  We
  // could add other operations where it's easy to see a range, such as
  // remainder, convert, etc., though at some point we'd probably want a
  // range-tracking analysis.
  HloInstruction* bcast;
  HloInstruction* addend;
  if (Match(
          remainder,
          m::Remainder(
              m::AddAnyOrder(m::Iota(&iota),
                             m::Broadcast(m::ConstantEffectiveScalar(&addend))),
              m::Broadcast(&bcast, m::ConstantEffectiveScalar(&divisor)))) &&
      addend == divisor) {
    // The iota counts {0, ...iota_upper_bound - 1}, with the same caveat above
    // that iota_upper_bound is conservative, and the true upper bound may be
    // smaller.
    int64 iota_upper_bound = iota->shape().dimensions(
        Cast<HloIotaInstruction>(iota)->iota_dimension());
    absl::optional<int64> divisor_val = divisor->literal().GetIntegralAsS64(
        std::vector<int64>(0, divisor->shape().dimensions_size()));
    if (divisor_val) {
      // Check whether divisor_val + iota_upper_bound - 1 overflows.
      absl::optional<int64> max_val =
          OverflowSafeAdd(*divisor_val, iota_upper_bound);
      if (max_val.has_value() &&
          FitsInIntegralType(*max_val, iota->shape().element_type())) {
        return ReplaceWithNewInstruction(
            remainder,
            HloInstruction::CreateBinary(remainder->shape(),
                                         HloOpcode::kRemainder, iota, bcast));
      }
    }
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleReshape(HloInstruction* reshape) {
  auto operand = reshape->mutable_operand(0);

  // Reshape directly to empty constant if the shape contains zero-element
  // dimension.
  if (ShapeUtil::IsZeroElementArray(reshape->shape())) {
    // If the instruction doesn't have a layout, use a default layout for
    // the literal result.
    Shape reshaped_shape = reshape->shape();
    if (!LayoutUtil::HasLayout(reshaped_shape)) {
      LayoutUtil::SetToDefaultLayout(&reshaped_shape);
    }
    auto empty_constant = simplifier_->CreateConstantWithLayoutUpdated(
        Literal::CreateFromShape(reshaped_shape));

    return ReplaceWithNewInstruction(reshape, std::move(empty_constant));
  }

  // Delete no-op reshapes, i.e. where shape = operand shape.
  if (ShapeUtil::Compatible(reshape->shape(), operand->shape())) {
    VLOG(10) << "deleting no-op reshape";
    return ReplaceInstruction(reshape, operand);
  }

  // Merge reshapes.
  if (HloOpcode::kReshape == operand->opcode()) {
    return ReplaceWithNewInstruction(
        reshape, HloInstruction::CreateReshape(reshape->shape(),
                                               operand->mutable_operand(0)));
  }

  if (operand->opcode() == HloOpcode::kRng && operand->user_count() == 1) {
    *operand->mutable_shape() = reshape->shape();
    return ReplaceInstruction(reshape, operand);
  }

  if (HloOpcode::kBroadcast == reshape->operand(0)->opcode()) {
    auto opt_dims = ReshapeLeavesDimensionsUnmodified(
        reshape, reshape->operand(0)->dimensions());
    if (opt_dims.has_value()) {
      return ReplaceWithNewInstruction(
          reshape,
          HloInstruction::CreateBroadcast(
              reshape->shape(), reshape->mutable_operand(0)->mutable_operand(0),
              *opt_dims));
    }
  }

  // reshape(iota) -> iota or a mixed radix calculation like
  // s32[2,3,4] reshape(s32[24] iota()) to
  // add(
  //    add(s32[2,3,4] iota() iota_dimension=2,
  //        4 * s32[2,3,4] iota() iota_dimension=1),
  //    12 * s32[2,3,4] iota() iota_dimension=0).
  if (operand->opcode() == HloOpcode::kIota) {
    auto* iota = Cast<HloIotaInstruction>(operand);
    auto opt_dims =
        ReshapeLeavesDimensionsUnmodified(reshape, {iota->iota_dimension()});
    auto common_factors =
        CommonFactors(reshape->operand(0)->shape().dimensions(),
                      reshape->shape().dimensions());
    auto iota_dim = absl::c_find_if(
        common_factors, [&](const std::pair<int64, int64>& dim_pair) {
          return dim_pair.first == iota->iota_dimension() &&
                 reshape->shape().dimensions(dim_pair.second) > 1;
        });
    auto next_dim = absl::c_find_if(
        common_factors, [&](const std::pair<int64, int64>& dim_pair) {
          return dim_pair.first == iota->iota_dimension() + 1;
        });
    if (iota_dim != common_factors.end() && next_dim != common_factors.end()) {
      int64 multiplier = 1;
      HloInstruction* new_reshape = nullptr;

      for (int64 dim = (iota_dim + 1)->second - 1; dim >= iota_dim->second;
           --dim) {
        HloInstruction* new_iota = computation_->AddInstruction(
            HloInstruction::CreateIota(reshape->shape(), dim));
        iota->SetupDerivedInstruction(new_iota);
        if (new_reshape) {
          new_reshape =
              computation_->AddInstruction(HloInstruction::CreateBinary(
                  reshape->shape(), HloOpcode::kAdd, new_reshape,
                  computation_->AddInstruction(HloInstruction::CreateBinary(
                      reshape->shape(), HloOpcode::kMultiply, new_iota,
                      MakeScalarLike(reshape, multiplier)))));
          reshape->SetupDerivedInstruction(new_reshape);
        } else {
          new_reshape = new_iota;
        }
        multiplier *= reshape->shape().dimensions(dim);
      }
      reshape->SetupDerivedInstruction(new_reshape);
      return ReplaceInstruction(reshape, new_reshape);
    }
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleReverse(HloInstruction* reverse) {
  // When all the dimensions to reverse are trivial (i.e. the bound is 1),
  // there is nothing to be done.
  auto dim_is_one = [&](int64 i) -> bool {
    return reverse->shape().dimensions(i) == 1;
  };
  if (absl::c_all_of(reverse->dimensions(), dim_is_one)) {
    return ReplaceInstruction(reverse, reverse->mutable_operand(0));
  }
  return Status::OK();
}

StatusOr<bool> AlgebraicSimplifierVisitor::TrySimplifyScalarSlice(
    HloInstruction* slice) {
  // Only try to do this for effective scalars. We could do the same for slicing
  // out larger pieces of padding (replacing with a broadcast of the padding
  // value), but this is probably not worth it.
  if (!ShapeUtil::IsEffectiveScalar(slice->shape())) {
    return false;
  }

  if (slice->operand(0)->opcode() == HloOpcode::kPad) {
    VLOG(10) << "Trying to simplify scalar slice of pad";
    // Check there's no internal padding. Again, we could handle that too, since
    // everything is statically known, but it's not worth it.
    auto pad = Cast<HloPadInstruction>(slice->mutable_operand(0));
    auto padding_config = pad->padding_config();
    int64 rank = padding_config.dimensions_size();
    if (HasInteriorPadding(padding_config)) {
      VLOG(10) << "Not folding scalar slice of pad, pad has interior padding";
      return false;
    }

    // Check whether the scalar we're slicing out falls into the padding.
    bool in_padding = [&]() {
      for (int64 i = 0; i < rank; ++i) {
        int64 start = slice->slice_starts(i);
        int64 low = padding_config.dimensions(i).edge_padding_low();
        int64 data = pad->operand(0)->shape().dimensions(i);
        if (start < low || start >= low + data) {
          return true;
        }
      }
      return false;
    }();

    if (in_padding) {
      VLOG(10) << "Folding scalar slice of pad into padding value";
      TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
          slice, HloInstruction::CreateReshape(slice->shape(),
                                               pad->mutable_padding_value())));
      return true;
    } else {
      // We already know the output of the slice is scalar. If the padded
      // value is scalar, and it's not in the padding, then it's exactly the
      // output value.
      bool replaced =
          ReplaceInstructionIfSameShape(slice, pad->mutable_operand(0));
      if (replaced) {
        VLOG(10) << "Folding scalar slice of pad into padded value";
      } else {
        VLOG(10) << "Not folding scalar slice of pad into padded value as they "
                    "have different shapes.";
      }
      return replaced;
    }
  }

  if (slice->operand(0)->opcode() == HloOpcode::kConcatenate) {
    VLOG(10) << "Trying to simplify scalar slice of concat";
    // Only do this for R1, there's no chance of this being useful otherwise.
    if (slice->shape().rank() != 1) {
      VLOG(10) << "Not folding, slice is not rank 1";
      return false;
    }
    HloConcatenateInstruction* concat =
        Cast<HloConcatenateInstruction>(slice->mutable_operand(0));
    int64 operand_start = 0;
    int64 operand_num = 0;
    // Weird loop structure to avoid annoying off-by-one errors.
    while (true) {
      TF_RET_CHECK(operand_num < concat->operand_count());
      const HloInstruction* operand = concat->operand(operand_num);
      int64 next_operand_start = operand_start + operand->shape().dimensions(0);
      if (next_operand_start > slice->slice_starts(0)) {
        break;
      }
      operand_start = next_operand_start;
      operand_num++;
    }

    bool replaced = ReplaceInstructionIfSameShape(
        slice, concat->mutable_operand(operand_num));
    if (replaced) {
      VLOG(10) << "Folding scalar slice of concat into concat operand";
    } else {
      VLOG(10) << "Folding scalar slice of concat into slice of concat operand";
      TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
          slice, HloInstruction::CreateSlice(
                     slice->shape(), concat->mutable_operand(operand_num),
                     {slice->slice_starts(0) - operand_start},
                     {slice->slice_starts(0) - operand_start + 1},
                     slice->slice_strides())));
    }
    return true;
  }

  return false;
}

StatusOr<bool> AlgebraicSimplifierVisitor::TryToReorderSliceAndReshape(
    HloInstruction* slice) {
  CHECK_EQ(slice->opcode(), HloOpcode::kSlice);
  if (!pp::algebraic_simplifier::util::IsUnstridedSlice(slice)) {
    return false;
  }
  HloInstruction* reshape = slice->mutable_operand(0);
  if (reshape->opcode() != HloOpcode::kReshape) {
    return false;
  }
  HloInstruction* new_slice_operand = reshape->mutable_operand(0);
  int64 slice_rank = slice->shape().rank();
  std::vector<int64> sliced_dims;
  for (int64 i = 0; i < slice_rank; ++i) {
    if (slice->slice_starts(i) != 0 ||
        slice->slice_limits(i) != reshape->shape().dimensions(i)) {
      sliced_dims.push_back(i);
    }
  }

  if (sliced_dims.size() == 1 && sliced_dims[0] == 0 &&
      slice->slice_starts(0) == 0) {
    const Shape& new_slice_shape = new_slice_operand->shape();
    const int64 rank = new_slice_shape.rank();
    std::vector<int64> new_slice_starts(rank, 0);
    std::vector<int64> new_slice_stides(rank, 1);
    std::vector<int64> new_slice_limits(new_slice_shape.dimensions().begin(),
                                        new_slice_shape.dimensions().end());
    int64 slice_elements = ShapeUtil::ElementsIn(slice->shape());
    for (int64 i = rank - 1; i >= 0; --i) {
      if (slice_elements >= new_slice_limits[i]) {
        if (slice_elements % new_slice_limits[i] != 0) {
          return false;
        }
        slice_elements /= new_slice_limits[i];
      } else {
        new_slice_limits[i] = slice_elements;
        slice_elements = 1;
      }
    }
    HloInstruction* new_slice =
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(HloInstruction::CreateSlice(
                ShapeUtil::MakeShape(new_slice_shape.element_type(),
                                     new_slice_limits),
                new_slice_operand, new_slice_starts, new_slice_limits,
                new_slice_stides)),
            slice);
    simplifier_->UpdateLayout(new_slice->mutable_shape());
    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
        slice, HloInstruction::CreateReshape(slice->shape(), new_slice)));
    return true;
  }
  return false;
}

Status AlgebraicSimplifierVisitor::HandleSlice(HloInstruction* slice) {
  // Delete no-op slices, i.e. where shape = operand shape.
  if (ReplaceInstructionIfSameShape(slice, slice->mutable_operand(0))) {
    return Status::OK();
  }

  if (slice->operand(0)->opcode() == HloOpcode::kSlice &&
      pp::algebraic_simplifier::util::IsUnstridedSlice(slice) &&
      pp::algebraic_simplifier::util::IsUnstridedSlice(slice->operand(0))) {
    HloInstruction* operand_slice = slice->mutable_operand(0);
    std::vector<int64> new_slice_starts = slice->slice_starts();
    std::vector<int64> new_slice_limits = slice->slice_limits();
    for (size_t i = 0; i < new_slice_starts.size(); ++i) {
      new_slice_starts[i] += operand_slice->slice_starts(i);
      new_slice_limits[i] += operand_slice->slice_starts(i);
    }
    return ReplaceWithNewInstruction(
        slice, HloInstruction::CreateSlice(
                   slice->shape(), operand_slice->mutable_operand(0),
                   new_slice_starts, new_slice_limits, slice->slice_strides()));
  }

  auto only_broadcast_dims_sliced = [&] {
    if (slice->operand(0)->opcode() != HloOpcode::kBroadcast) {
      return false;
    }
    for (int64 dim : slice->operand(0)->dimensions()) {
      if (slice->slice_starts(dim) != 0 || slice->slice_strides(dim) != 1 ||
          slice->slice_limits(dim) !=
              slice->operand(0)->shape().dimensions(dim)) {
        return false;
      }
    }
    return true;
  };
  if (only_broadcast_dims_sliced()) {
    return ReplaceWithNewInstruction(
        slice,
        HloInstruction::CreateBroadcast(
            slice->shape(), slice->mutable_operand(0)->mutable_operand(0),
            slice->mutable_operand(0)->dimensions()));
  }

  TF_ASSIGN_OR_RETURN(bool replaced, TrySimplifyScalarSlice(slice));
  if (replaced) {
    return Status::OK();
  }

  // Try to simplify concat -> slice to an operand of concat.
  if (slice->operand(0)->opcode() == HloOpcode::kConcatenate &&
      pp::algebraic_simplifier::util::IsUnstridedSlice(slice)) {
    auto concat = slice->operand(0);
    int64 concat_dim = concat->concatenate_dimension();
    int64 piece_start = 0;
    for (auto piece : concat->operands()) {
      if (!ShapeUtil::Compatible(piece->shape(), slice->shape())) {
        piece_start += piece->shape().dimensions(concat_dim);
        continue;
      }
      if (slice->slice_starts(concat_dim) == piece_start) {
        return ReplaceInstruction(slice, piece);
      }
      piece_start += piece->shape().dimensions(concat_dim);
    }
  }

  // Do not try to reorder slices and reshapes after layout assignment as it may
  // be invalid.
  TF_ASSIGN_OR_RETURN(replaced, TryToReorderSliceAndReshape(slice));
  if (replaced) {
    return Status::OK();
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleDynamicSlice(
    HloInstruction* dynamic_slice) {
  auto operand = dynamic_slice->mutable_operand(0);
  if (ShapeUtil::IsScalar(dynamic_slice->shape())) {
    return ReplaceInstruction(dynamic_slice, operand);
  }
  // DynamicSlice where operand has the same size as the output is simply equal
  // to operand.
  if (ShapeUtil::Compatible(operand->shape(), dynamic_slice->shape())) {
    return ReplaceInstruction(dynamic_slice, operand);
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update_slice) {
  // Rewriting DynamicUpdateSlice when it matches
  // dynamic_update_slice(broadcast(constant),data,constant_index0,...)
  // to a Pad(x, constant)
  // Only Broadcast considered currently, other ops need to be considered
  // in the future.
  HloInstruction* updated = dynamic_update_slice->mutable_operand(0);
  HloInstruction* dus_update = dynamic_update_slice->mutable_operand(1);
  HloInstruction* pad_value;
  if (Match(updated,
            m::Broadcast(m::Op(&pad_value).WithShape(m::Shape().IsScalar())))) {
    auto updated_shape = updated->shape();
    auto update_shape = dus_update->shape();
    auto update_start_indx = dynamic_update_slice->operand(2);
    int64 offset = 0;
    bool compatible = true;
    // Whether the start indices to dynamic update slice is a list,
    // output of a tuple/concatenate, we setup the update_start_indx
    // appropriately.
    if (ShapeUtil::IsScalar(update_start_indx->shape())) {
      update_start_indx = dynamic_update_slice;
      offset = 2;
    } else {
      if (update_start_indx->opcode() == HloOpcode::kTuple ||
          update_start_indx->opcode() == HloOpcode::kConcatenate) {
        offset = 0;
      } else {
        compatible = false;
      }
    }
    PaddingConfig padding_config;
    if (compatible) {
      for (int64 dim = 0; dim < updated_shape.rank(); ++dim) {
        auto padding_config_dim = padding_config.add_dimensions();
        auto slice_dim_start = update_start_indx->operand(dim + offset);
        if (!Match(slice_dim_start, m::ConstantScalar())) {
          compatible = false;
          break;
        }
        VLOG(2) << "slice: " << slice_dim_start->ToString();
        absl::optional<int64> beg =
            slice_dim_start->literal().GetFirstInteger();
        if (!beg) {
          compatible = false;
          break;
        }
        VLOG(2) << "beg value: " << *beg;
        auto update_width = ShapeUtil::GetDimension(update_shape, dim);
        auto bcast_width = ShapeUtil::GetDimension(updated_shape, dim);
        // Clamp beg so that it is non-negative.
        *beg = std::max<int64>(0, *beg);
        // Clamp beg so that it is in-bounds.
        *beg = std::min<int64>(bcast_width - update_width, *beg);
        VLOG(2) << "adjusted beg value: " << *beg;
        padding_config_dim->set_edge_padding_low(*beg);
        padding_config_dim->set_edge_padding_high(bcast_width -
                                                  (*beg + update_width));
        // dynamic_update_slice does not specify a stride
        padding_config_dim->set_interior_padding(0);
      }
    }

    if (compatible) {
      HloInstruction* pad =
          computation_->AddInstruction(HloInstruction::CreatePad(
              updated_shape, dus_update, pad_value, padding_config));
      VLOG(2) << dynamic_update_slice->ToString();
      VLOG(2) << " with pad:" << pad->ToString();
      VLOG(2) << " Computation before rewrite is: "
              << dynamic_update_slice->parent()->ToString();
      return ReplaceInstruction(dynamic_update_slice, pad);
    }
  }

  // DynamicUpdateSlice where operand and dus_update have the same size is
  // equal to dus_update.
  if (ShapeUtil::Compatible(dynamic_update_slice->shape(),
                            dus_update->shape())) {
    return ReplaceInstruction(dynamic_update_slice, dus_update);
  }

  // If any dimension of dus_update is 0, elide the DynamicUpdateSlice.  This
  // optimization becomes invalid should we later prefer to warn about out of
  // bound indices.
  if (ShapeUtil::IsZeroElementArray(dus_update->shape())) {
    return ReplaceInstruction(dynamic_update_slice, updated);
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleAllReduce(HloInstruction* all_reduce) {
  /// Replace all-reduce(replication-normalise(all-reduce(arg))) with
  /// all-reduce(arg)
  if (all_reduce->operand_count() != 1 ||
      !pp::algebraic_simplifier::util::IsGlobalAllReduceWithSum(all_reduce)) {
    return Status::OK();
  }
  HloInstruction* normalise = all_reduce->mutable_operand(0);
  if (!pp::IsPoplarInstruction(PoplarOp::ReplicationNormalise, normalise) ||
      normalise->operand_count() != 1) {
    return Status::OK();
  }
  HloInstruction* top_all_reduce = normalise->mutable_operand(0);
  if (top_all_reduce->opcode() == HloOpcode::kAllReduce &&
      top_all_reduce->operand_count() == 1 &&
      pp::algebraic_simplifier::util::IsGlobalAllReduceWithSum(
          top_all_reduce)) {
    return ReplaceInstruction(all_reduce, top_all_reduce);
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleReduce(HloInstruction* hlo) {
  HloReduceInstruction* reduce = Cast<HloReduceInstruction>(hlo);
  bool multi_output_reduce = reduce->shape().IsTuple();

  // For tuple reduce, we require all reduce shapes to be the same, up to the
  // element types, so we can just the first operand and the first result as a
  // representative.
  auto arg = reduce->inputs()[0];
  auto init_value = reduce->init_values()[0];
  Shape& reduce_result_shape = const_cast<Shape&>(
      multi_output_reduce ? reduce->shape().tuple_shapes(0) : reduce->shape());

  absl::Span<const int64> dimensions(reduce->dimensions());
  HloComputation* function = reduce->to_apply();
  if (ShapeUtil::IsZeroElementArray(arg->shape()) ||
      ShapeUtil::IsZeroElementArray(reduce_result_shape)) {
    if (multi_output_reduce) {
      std::vector<HloInstruction*> broadcast_inits;
      int64 inputs = reduce->input_count();
      for (int64 i = 0; i < inputs; ++i) {
        broadcast_inits.push_back(computation_->AddInstruction(
            HloInstruction::CreateBroadcast(reduce->shape().tuple_shapes(i),
                                            reduce->init_values()[i], {})));
      }
      return ReplaceWithNewInstruction(
          reduce, HloInstruction::CreateTuple(broadcast_inits));
    } else {
      return ReplaceWithNewInstruction(
          reduce,
          HloInstruction::CreateBroadcast(reduce_result_shape, init_value, {}));
    }
  }

  // If the reduction results in the same number of elements, then the only
  // possible side effect would be a reshape. Since the init_value is an
  // identity of the reduction function, we can therefore replace the reduce
  // with a simple reshape, ignoring the reduction function completely.
  if (ShapeUtil::ElementsIn(reduce_result_shape) ==
      ShapeUtil::ElementsIn(arg->shape())) {
    if (multi_output_reduce) {
      std::vector<HloInstruction*> reshaped_args;
      int64 inputs = reduce->input_count();
      for (int64 i = 0; i < inputs; ++i) {
        reshaped_args.push_back(
            pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
                computation_->AddInstruction(HloInstruction::CreateReshape(
                    reduce->shape().tuple_shapes(i), reduce->inputs()[i])),
                reduce));
      }
      return ReplaceWithNewInstruction(
          reduce, HloInstruction::CreateTuple(reshaped_args));
    } else {
      return ReplaceWithNewInstruction(
          reduce, HloInstruction::CreateReshape(reduce_result_shape, arg));
    }
  }

  if (hlo->operand_count() == 4 && hlo->operand(0)->shape().rank() < 3) {
    // Match for ArgMin/Max generated by BuildArgMinMaxReductionBody in
    // tensorflow/compiler/mlir/xla/transforms/legalize_tf.cc.
    const bool arg_max_or_min = Match(
        function->root_instruction(),
        m::Tuple(
            m::Select(m::Compare(m::Parameter(0), m::Parameter(2)),
                      m::Parameter(0), m::Parameter(2)),
            m::Select(m::Compare(m::Parameter(0), m::Parameter(2))
                          .WithComparisonDirection(ComparisonDirection::kEq),
                      m::Minimum(m::Parameter(1), m::Parameter(3)),
                      m::Select(m::Compare(m::Parameter(0), m::Parameter(2)),
                                m::Parameter(1), m::Parameter(3)))));

    if (arg_max_or_min) {
      const HloInstruction* output_0 = function->root_instruction()->operand(0);
      const HloInstruction* output_1 = function->root_instruction()->operand(1);

      // Check that the compare is the same.
      bool valid = output_0->operand(0) == output_1->operand(2)->operand(0);
      // Get the instruction inputs and verify them.
      HloInstruction* value;
      HloInstruction* iota;
      HloInstruction* init_value;
      HloInstruction* init_iota;
      valid = valid &&
              Match(hlo, m::Reduce(m::Op(&value), m::Op(&iota),
                                   m::Op(&init_value), m::Op(&init_iota))) &&
              Match(iota, m::Iota()) &&
              Match(init_value, m::ConstantScalar()) &&
              Match(init_iota, m::ConstantScalar(0));

      // Check that the reduction is done over one dimension matching iota.
      if (valid) {
        valid = hlo->dimensions().size() == 1 &&
                hlo->dimensions(0) ==
                    Cast<HloIotaInstruction>(iota)->iota_dimension();
      }

      bool is_max = false;
      // Find whether this is min or max and verify initial value.
      if (valid) {
        const HloInstruction* compare = output_0->operand(0);
        if (compare->comparison_direction() == ComparisonDirection::kLe) {
          // Min.
          valid = init_value->literal() ==
                  LiteralUtil::MaxValue(value->shape().element_type());
          is_max = false;
        } else if (compare->comparison_direction() ==
                   ComparisonDirection::kGe) {
          // Max.
          valid = init_value->literal() ==
                  LiteralUtil::MinValue(value->shape().element_type());
          is_max = true;
        }
      }

      if (valid) {
        const Shape& values_shape = ShapeUtil::GetSubshape(hlo->shape(), {0});
        const Shape& indices_shape = ShapeUtil::GetSubshape(hlo->shape(), {1});
        const Shape output_shape =
            ShapeUtil::MakeTupleShape({values_shape, indices_shape});
        HloInstruction* arg_min_max;
        if (is_max) {
          arg_min_max = computation_->AddInstruction(pp::CreateHloMaxAndArgMax(
              value, output_shape, hlo->dimensions(0)));
        } else {
          arg_min_max = computation_->AddInstruction(pp::CreateHloMinAndArgMin(
              value, output_shape, hlo->dimensions(0)));
        }
        arg_min_max =
            pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
                arg_min_max, hlo);
        return ReplaceInstruction(hlo, arg_min_max);
      }
    }
  }

  // TODO(b/131122694): Most of those optimizations below can be done for
  // multi-output reduces.
  if (multi_output_reduce) {
    return Status::OK();
  }

  // A Transpose feeding a reduce can simply permute the reduction dimensions
  // field if the output of the reduce is a vector or scalar. Higher ranked
  // result may require a transpose of the output.
  if (reduce_result_shape.rank() <= 1 &&
      arg->opcode() == HloOpcode::kTranspose) {
    auto transpose_dimensions = arg->dimensions();
    std::vector<int64> new_reduce_dimensions;
    for (auto dim : dimensions) {
      new_reduce_dimensions.push_back(transpose_dimensions[dim]);
    }
    return ReplaceWithNewInstruction(
        reduce, HloInstruction::CreateReduce(
                    reduce_result_shape, arg->mutable_operand(0), init_value,
                    new_reduce_dimensions, function));
  }

  // If a reduce feeds a reduce with the same computation and initial value,
  // they can be combined into a single reduce.
  if (arg->opcode() == HloOpcode::kReduce &&
      init_value->Identical(*arg->operand(1)) &&
      *function == *arg->to_apply()) {
    // Create a new reduce with the combined reduction dimensions of both
    // reduces.
    std::vector<int64> arg_dims = arg->dimensions();
    absl::c_sort(arg_dims);
    std::vector<int64> reduce_dims = reduce->dimensions();
    absl::c_sort(reduce_dims);
    // Transform reduce_dims to the same rank as the operand of the operand.
    for (int64 arg_dim : arg_dims) {
      for (int64& dim : reduce_dims) {
        if (dim >= arg_dim) {
          ++dim;
        }
      }
    }
    std::vector<int64> new_dimensions;
    new_dimensions.reserve(arg->dimensions().size() +
                           reduce->dimensions().size());
    std::merge(arg_dims.begin(), arg_dims.end(), reduce_dims.begin(),
               reduce_dims.end(), std::back_inserter(new_dimensions));
    return ReplaceWithNewInstruction(
        reduce, HloInstruction::CreateReduce(
                    reduce_result_shape, arg->mutable_operand(0), init_value,
                    new_dimensions, function));
  }

  // A reshape that collapses multiple dimensions into a dimension being
  // reduced can just reduce all of those dimensions instead of doing a
  // collapsing reshape before a reduction.
  if (arg->opcode() == HloOpcode::kReshape) {
    std::vector<std::pair<int64, int64>> unmodified_dims =
        ShapeUtil::DimensionsUnmodifiedByReshape(arg->operand(0)->shape(),
                                                 arg->shape());
    std::vector<bool> arg_dim_in_output(arg->shape().rank(), true);
    std::vector<bool> arg_dim_unmodified(arg->shape().rank(), false);
    for (auto dim : dimensions) {
      arg_dim_in_output[dim] = false;
    }
    for (auto dim_pair : unmodified_dims) {
      arg_dim_unmodified[dim_pair.second] = true;
    }
    // The goal is to verify that all dimensions that are not removed in the
    // reduce are unmodified by the reshape. For example:
    // reduce(reshape([A,B*C], a[A,B,C]),[1]) = reduce(a[A, B, C], [1, 2])
    bool can_move_reshape_into_reduce = true;
    for (size_t i = 0; i < arg_dim_in_output.size(); ++i) {
      if (arg_dim_in_output[i] && !arg_dim_unmodified[i]) {
        can_move_reshape_into_reduce = false;
      }
    }
    if (can_move_reshape_into_reduce) {
      changed_ = true;
      absl::flat_hash_set<int64> dimensions_not_to_reduce;
      for (auto dim_pair : unmodified_dims) {
        if (arg_dim_in_output[dim_pair.second]) {
          dimensions_not_to_reduce.insert(dim_pair.first);
        }
      }
      std::vector<int64> new_reduce_dimensions;
      for (int64 i = 0; i < arg->operand(0)->shape().rank(); ++i) {
        if (!dimensions_not_to_reduce.contains(i)) {
          new_reduce_dimensions.push_back(i);
        }
      }
      return ReplaceWithNewInstruction(
          reduce, HloInstruction::CreateReduce(
                      reduce_result_shape, arg->mutable_operand(0), init_value,
                      new_reduce_dimensions, function));
    }
  }
  // Convert Reduce(concat({a,b,...})) to
  //  map(reduce(a),map(reduce(b),...,))
  //
  // This should make fusion easier or use less memory bandwidth in the
  // unfused case.
  if (arg->opcode() == HloOpcode::kConcatenate &&
      absl::c_linear_search(reduce->dimensions(),
                            arg->concatenate_dimension())) {
    HloInstruction* old_reduce = nullptr;
    for (HloInstruction* operand : arg->operands()) {
      HloInstruction* new_reduce =
          pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
              computation_->AddInstruction(HloInstruction::CreateReduce(
                  reduce_result_shape, operand, init_value,
                  reduce->dimensions(), function)),
              reduce);
      if (old_reduce != nullptr) {
        new_reduce =
            pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
                computation_->AddInstruction(HloInstruction::CreateMap(
                    reduce_result_shape, {old_reduce, new_reduce}, function)),
                reduce);
      }
      old_reduce = new_reduce;
    }
    return ReplaceInstruction(reduce, old_reduce);
  } else if (arg->opcode() == HloOpcode::kDot &&
             (Match(function->root_instruction(),
                    m::Add(m::Parameter(0), m::Parameter(1))) ||
              Match(function->root_instruction(),
                    m::Add(m::Parameter(1), m::Parameter(0))))) {
    bool valid = true;
    for (auto value : reduce->init_values()) {
      if (!pp::IsConstantZero(value)) {
        valid = false;
        break;
      }
    }

    auto* dot = Cast<HloDotInstruction>(arg);
    if (dot->user_count() != 1) {
      valid = false;
    }

    if (valid) {
      VLOG(10) << "found candidate for reduce(dot) reduction "
               << reduce->ToString();
      auto dims = arg->dot_dimension_numbers();
      auto& lhs_batch_dimensions = *dims.mutable_lhs_batch_dimensions();
      auto& rhs_batch_dimensions = *dims.mutable_rhs_batch_dimensions();
      auto& lhs_contracting_dimensions =
          *dims.mutable_lhs_contracting_dimensions();
      auto& rhs_contracting_dimensions =
          *dims.mutable_rhs_contracting_dimensions();

      for (auto dim : dimensions) {
        VLOG(10) << "trying to reduce dimension " << dim;
        if (absl::c_count(lhs_batch_dimensions, dim) == 1 &&
            absl::c_count(rhs_batch_dimensions, dim) == 1) {
          // if reduce dimension is in both batch dimensions,
          // put it on contracting dimension list
          lhs_batch_dimensions.erase(
              std::remove(lhs_batch_dimensions.begin(),
                          lhs_batch_dimensions.end(), dim),
              lhs_batch_dimensions.end());
          rhs_batch_dimensions.erase(
              std::remove(rhs_batch_dimensions.begin(),
                          rhs_batch_dimensions.end(), dim),
              rhs_batch_dimensions.end());
          lhs_contracting_dimensions.Add(dim);
          rhs_contracting_dimensions.Add(dim);
        } else {
          VLOG(10) << "dimension " << dim << " is not on batch dimension list";
          valid = false;
        }
      }
      if (valid) {
        VLOG(10) << "replace reduce/dot combination";
        auto new_dot = HloInstruction::CreateDot(
            reduce->shape(), dot->mutable_operand(0), dot->mutable_operand(1),
            dims, dot->precision_config());
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            new_dot.get(), dot);
        return ReplaceWithNewInstruction(reduce, std::move(new_dot));
      }
    }
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleReduceWindow(
    HloInstruction* reduce_window) {
  // TODO(b/73062247) Variadic reduce window is not yet supported in simplifier.
  if (reduce_window->shape().IsTuple()) {
    return Status::OK();
  }

  if (ShapeUtil::IsZeroElementArray(reduce_window->operand(0)->shape())) {
    return ReplaceWithNewInstruction(
        reduce_window,
        HloInstruction::CreateBroadcast(reduce_window->shape(),
                                        reduce_window->mutable_operand(1), {}));
  }
  auto operand = reduce_window->mutable_operand(0);
  const Window& window = reduce_window->window();
  auto function = reduce_window->to_apply();
  if (ShapeUtil::IsScalar(operand->shape())) {
    TF_RET_CHECK(ShapeUtil::IsScalar(reduce_window->shape()));
    return ReplaceWithNewInstruction(
        reduce_window,
        HloInstruction::CreateMap(reduce_window->shape(),
                                  {reduce_window->mutable_operand(1), operand},
                                  function));
  }

  // This optimization folds a pad op into reduce_window.
  HloInstruction* pad;
  const HloInstruction* convert = nullptr;
  if (operand->opcode() == HloOpcode::kPad) {
    pad = operand;
  } else if (operand->opcode() == HloOpcode::kConvert &&
             operand->operand(0)->opcode() == HloOpcode::kPad) {
    convert = operand;
    pad = operand->mutable_operand(0);
  } else {
    VLOG(10) << "Not folding pad into reduce-window as there is no pad.";
    return Status::OK();
  }

  VLOG(10) << "Considering folding Pad: " << pad->ToString()
           << "\ninto reduce-window: " << reduce_window->ToString()
           << (convert != nullptr
                   ? absl::StrCat("\nvia convert: ", convert->ToString())
                   : "");

  // Do not fold interior padding into ReduceWindow since the backends do not
  // support it.
  const PaddingConfig& pad_config = pad->padding_config();
  if (HasInteriorPadding(pad_config) && window_util::HasBaseDilation(window)) {
    VLOG(10) << "Not folding interior pad into base-dilated reduce-window.";
    return Status::OK();
  }

  // If reduce_window already has padding, the pad value of the pad op and the
  // init value of reduce_window must match to allow folding the pad.
  const HloInstruction* pad_value = pad->operand(1);
  const HloInstruction* reduce_init_value = reduce_window->operand(1);
  if (pad_value != reduce_init_value) {
    auto literals_are_equivalent = [&] {
      auto& pad_literal = pad_value->literal();
      auto& reduce_init_literal = reduce_init_value->literal();
      if (pad_literal == reduce_init_literal) {
        return true;
      }
      auto converted_pad_literal =
          pad_literal.ConvertToShape(reduce_init_value->shape());
      if (!converted_pad_literal.ok()) {
        return false;
      }
      return converted_pad_literal.ValueOrDie() == reduce_init_literal;
    };
    // The pad value is usually a constant, so we handle that case and do not
    // try to get more fancy about proving equivalence in cases beyond that.
    if (pad_value->opcode() != HloOpcode::kConstant ||
        reduce_init_value->opcode() != HloOpcode::kConstant ||
        !literals_are_equivalent()) {
      VLOG(10) << "Not folding pad into reduce-window due to different pad "
                  "values.";
      return Status::OK();
    }
  }

  // If the pad puts a single non-identity value in each window that we're
  // reducing, then this is a broadcast.
  HloInstruction* pad_operand = pad->mutable_operand(0);
  auto is_effective_broadcast = [&] {
    if (window_util::HasStride(window)) {
      VLOG(10) << "Window has stride.";
      return false;
    }
    if (!window_util::HasSymmetricPadding(pad_config)) {
      VLOG(10) << "Window has uneven padding.";
      return false;
    }
    if (HasInteriorPadding(pad_config)) {
      VLOG(10) << "Window has interior padding.";
      return false;
    }
    for (int64 i = 0; i < pad_config.dimensions_size(); ++i) {
      const auto& pad_dimension = pad_config.dimensions(i);
      if ((pad_dimension.edge_padding_low() != 0 ||
           pad_dimension.edge_padding_high() != 0) &&
          pad_operand->shape().dimensions(i) != 1) {
        VLOG(10) << "Found non-trivial dimension being padded: " << i;
        return false;
      }
    }
    VLOG(10) << "Found to be padding trivial dimensions only.";

    for (int64 i = 0; i < window.dimensions_size(); ++i) {
      const auto& pad_dimension = pad_config.dimensions(i);
      const WindowDimension& window_dimension = window.dimensions(i);
      bool dimension_has_padding = (pad_dimension.edge_padding_low() != 0 ||
                                    pad_dimension.edge_padding_high() != 0);
      if (dimension_has_padding &&
          window_dimension.size() < pad_dimension.edge_padding_low() + 1) {
        VLOG(10) << "Found window did not cover single unpadded element in "
                    "dimension: "
                 << i;
        return false;
      }
      if (pad_operand->shape().dimensions(i) != 1 &&
          window_dimension.size() != 1) {
        VLOG(10) << "Found window covers more than one element in non-trivial "
                    "dimension: "
                 << i;
        return false;
      }
    }
    VLOG(10) << "Found window covers a single unpadded element.";
    return true;
  };

  HloInstruction* new_reduce_window_operand;
  if (convert != nullptr) {
    Shape changed_shape = ShapeUtil::ChangeElementType(
        pad_operand->shape(), convert->shape().element_type());
    simplifier_->UpdateLayout(&changed_shape);
    new_reduce_window_operand =
        pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
            computation_->AddInstruction(
                HloInstruction::CreateConvert(changed_shape, pad_operand)),
            pad_operand);
  } else {
    new_reduce_window_operand = pad_operand;
  }

  if (is_effective_broadcast()) {
    VLOG(10) << "Replacing pad/reduce-window with broadcast.";
    auto fadd = [this](std::unique_ptr<HloInstruction> x) {
      return computation_->AddInstruction(std::move(x));
    };
    return ReplaceWithNewInstruction(
        reduce_window, HloInstruction::CreateBroadcastSequence(
                           /*output_shape=*/reduce_window->shape(),
                           /*operand=*/new_reduce_window_operand, fadd));
  }

  // Carry out the folding of the pad into reduce_window.
  VLOG(10) << "Folding pad into reduce-window.";
  Window new_window = window;
  const int64 rank = reduce_window->shape().rank();
  TF_RET_CHECK(pad_config.dimensions_size() == rank);
  TF_RET_CHECK(window.dimensions_size() == rank);
  for (int64 i = 0; i < rank; ++i) {
    const auto& pad_dim = pad_config.dimensions(i);
    auto& window_dim = *new_window.mutable_dimensions(i);
    window_dim.set_padding_low(window_dim.padding_low() +
                               pad_dim.edge_padding_low());
    window_dim.set_padding_high(window_dim.padding_high() +
                                pad_dim.edge_padding_high());
    if (pad_dim.interior_padding() != 0) {
      CHECK_EQ(window_dim.base_dilation(), 1);
      window_dim.set_base_dilation(1 + pad_dim.interior_padding());
    }
  }

  return ReplaceWithNewInstruction(
      reduce_window, HloInstruction::CreateReduceWindow(
                         /*shape=*/reduce_window->shape(),
                         /*operand=*/new_reduce_window_operand,
                         /*init_value=*/reduce_window->mutable_operand(1),
                         /*window=*/new_window,
                         /*reduce_computation=*/function));
}

Status AlgebraicSimplifierVisitor::HandleSelect(HloInstruction* select) {
  // select(x, y, y) -> y.
  if (select->operand(1) == select->operand(2)) {
    return ReplaceInstruction(select, select->mutable_operand(1));
  }
  // select(true, x, y) -> x.
  if (pp::algebraic_simplifier::util::IsAll(select->operand(0), true)) {
    return ReplaceInstruction(select, select->mutable_operand(1));
  }
  // select(false, x, y) -> y.
  if (pp::algebraic_simplifier::util::IsAll(select->operand(0), false)) {
    return ReplaceInstruction(select, select->mutable_operand(2));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleScatter(HloInstruction* scatter) {
  if (ShapeUtil::IsZeroElementArray(scatter->operand(2)->shape()) &&
      ReplaceInstructionIfSameShape(scatter, scatter->mutable_operand(0))) {
    return Status::OK();
  }
  if (ShapeUtil::IsZeroElementArray(scatter->operand(1)->shape()) &&
      ShapeUtil::Compatible(scatter->shape(), scatter->operand(0)->shape()) &&
      ShapeUtil::Compatible(scatter->shape(), scatter->operand(2)->shape())) {
    return ReplaceWithNewInstruction(
        scatter, HloInstruction::CreateMap(
                     scatter->shape(),
                     {scatter->mutable_operand(0), scatter->mutable_operand(2)},
                     scatter->to_apply()));
  }
  return Status::OK();
}
Status AlgebraicSimplifierVisitor::HandleSort(HloInstruction* sort) {
  auto operand = sort->mutable_operand(0);
  int64 dimension_to_sort = sort->dimensions(0);
  if (ShapeUtil::IsZeroElementArray(operand->shape()) ||
      operand->shape().dimensions(dimension_to_sort) <= 1) {
    if (sort->operand_count() == 1) {
      return ReplaceInstruction(sort, operand);
    }
    // If it is key/value sort, the output of sort is a tuple.
    return ReplaceWithNewInstruction(
        sort, HloInstruction::CreateTuple(sort->operands()));
  }
  return Status::OK();
}

namespace {
bool OnlyPermutesDegenerateDims(const Shape& shape,
                                absl::Span<const int64> perm) {
  std::vector<int64> new_permutation;
  int64 degenerate_count = 0;
  for (size_t i = 0; i < perm.size(); ++i) {
    if (shape.dimensions(i) != 1) {
      new_permutation.push_back(perm[i]);
    } else {
      ++degenerate_count;
    }
  }
  return degenerate_count > 0 && absl::c_is_sorted(new_permutation);
}
}  // namespace

Status AlgebraicSimplifierVisitor::HandleTranspose(HloInstruction* transpose) {
  auto operand = transpose->mutable_operand(0);
  if (std::is_sorted(transpose->dimensions().begin(),
                     transpose->dimensions().end())) {
    VLOG(10) << "deleting no-op transpose";
    return ReplaceInstruction(transpose, operand);
  }

  if (HloOpcode::kTranspose == operand->opcode()) {
    return ReplaceWithNewInstruction(
        transpose, HloInstruction::CreateTranspose(
                       transpose->shape(), operand->mutable_operand(0),
                       ComposePermutations(operand->dimensions(),
                                           transpose->dimensions())));
  }

  // Convert transpose(dot(a,b)) to dot(b,a).
  if (operand->opcode() == HloOpcode::kDot && operand->user_count() == 1 &&
      operand->shape().rank() == 2) {
    TF_ASSIGN_OR_RETURN(bool did_transform, [&]() -> StatusOr<bool> {
      const auto& dnums = operand->dot_dimension_numbers();
      if (dnums.lhs_batch_dimensions_size() != 0) {
        return false;
      }
      HloInstruction* lhs = operand->mutable_operand(0);
      if (lhs->shape().rank() != 1 + dnums.lhs_contracting_dimensions_size()) {
        return false;
      }
      HloInstruction* rhs = operand->mutable_operand(1);
      if (rhs->shape().rank() != 1 + dnums.rhs_contracting_dimensions_size()) {
        return false;
      }
      DotDimensionNumbers new_dnums;
      *new_dnums.mutable_lhs_contracting_dimensions() =
          dnums.rhs_contracting_dimensions();
      *new_dnums.mutable_rhs_contracting_dimensions() =
          dnums.lhs_contracting_dimensions();
      TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
          transpose, HloInstruction::CreateDot(transpose->shape(), /*lhs=*/rhs,
                                               /*rhs=*/lhs, new_dnums,
                                               operand->precision_config())));
      return true;
    }());
    if (did_transform) {
      return Status::OK();
    }
  }

  // Replace transpose with a reshape if more than one degenerate method is
  // permuted.
  if (OnlyPermutesDegenerateDims(transpose->shape(), transpose->dimensions())) {
    return ReplaceWithNewInstruction(
        transpose, HloInstruction::CreateReshape(
                       transpose->shape(), transpose->mutable_operand(0)));
  }

  if (operand->opcode() == HloOpcode::kRng && operand->user_count() == 1) {
    *operand->mutable_shape() = transpose->shape();
    return ReplaceInstruction(transpose, operand);
  }

  return Status::OK();
}

StatusOr<bool> AlgebraicSimplifierVisitor::FoldConvFilterPad(
    HloInstruction* convolution) {
  auto* lhs = convolution->mutable_operand(0);
  auto* rhs = convolution->mutable_operand(1);
  const ConvolutionDimensionNumbers& dnums =
      convolution->convolution_dimension_numbers();

  if (rhs->opcode() != HloOpcode::kPad) {
    return false;
  }

  // Convolution's padding is always zero, so bail if the kPad is adding
  // something other than zero.
  if (!pp::algebraic_simplifier::util::IsAll(rhs->operand(1), 0)) {
    return false;
  }

  const auto& padding = rhs->padding_config();

  // Can't pad or dilate feature dims.
  for (int64 dim : {dnums.kernel_input_feature_dimension(),
                    dnums.kernel_output_feature_dimension()}) {
    const auto& p = padding.dimensions(dim);
    if (p.edge_padding_low() != 0 || p.edge_padding_high() != 0 ||
        p.interior_padding() != 0) {
      return false;
    }
  }

  // Compute the window which is the result of merging the kPad and the
  // convolution's existing window.
  Window new_window = convolution->window();
  for (int64 dim = 0; dim < dnums.kernel_spatial_dimensions_size(); ++dim) {
    auto& w = *new_window.mutable_dimensions(dim);
    const auto& p = padding.dimensions(dnums.kernel_spatial_dimensions(dim));

    // We can only do this transformation if p adds dilation to the filter --
    // edge padding on the filter is not supported in conv.
    if (p.edge_padding_low() != 0 || p.edge_padding_high() != 0) {
      return false;
    }

    // Nothing to do if the kPad for this dim is entirely a nop.
    if (p.interior_padding() == 0) {
      continue;
    }

    // We cowardly refuse to think about how dilation composes with itself;
    // bail if both the kPad and conv have dilation on this dimension.
    if (w.window_dilation() > 1) {
      return false;
    }
    CHECK_EQ(w.window_dilation(), 1);
    w.set_window_dilation(1 + p.interior_padding());
    w.set_size(rhs->operand(0)->shape().dimensions(
        dnums.kernel_spatial_dimensions(dim)));
  }

  auto new_conv = convolution->CloneWithNewOperands(
      convolution->shape(), {lhs, rhs->mutable_operand(0)});
  new_conv->set_window(new_window);
  TF_RETURN_IF_ERROR(
      ReplaceWithNewInstruction(convolution, std::move(new_conv)));
  return true;
}

Status AlgebraicSimplifierVisitor::HandleConvolution(
    HloInstruction* convolution) {
  // Zero-sized input or filter.
  if (ShapeUtil::IsZeroElementArray(convolution->operand(0)->shape()) ||
      ShapeUtil::IsZeroElementArray(convolution->operand(1)->shape())) {
    return ReplaceWithNewInstruction(
        convolution,
        HloInstruction::CreateBroadcast(
            convolution->shape(),
            pp::algebraic_simplifier::util::PreserveFrontendAttributesIfNeeded(
                computation_->AddInstruction(
                    simplifier_->CreateConstantWithLayoutUpdated(
                        LiteralUtil::Zero(
                            convolution->shape().element_type()))),
                convolution),
            {}));
  }

  // Try to merge padding/dilation of the input with the convolution's window.
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloInstruction> folded_input_pad,
                      pp::algebraic_simplifier::convolution::FoldConvInputPad(
                          simplifier_, convolution));
  if (folded_input_pad) {
    TF_RETURN_IF_ERROR(
        ReplaceWithNewInstruction(convolution, std::move(folded_input_pad)));

    return Status::OK();
  }

  // Try to merge dilation of the filter with the convolution's window.
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloInstruction> folded_filter_pad,
                      pp::algebraic_simplifier::convolution::FoldConvFilterPad(
                          simplifier_, convolution));
  if (folded_filter_pad) {
    TF_RETURN_IF_ERROR(
        ReplaceWithNewInstruction(convolution, std::move(folded_filter_pad)));

    return Status::OK();
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleMap(HloInstruction* map) {
  auto* map_computation = map->to_apply();
  auto* map_root = map_computation->root_instruction();
  if (map_root->opcode() == HloOpcode::kParameter) {
    ReplaceInstructionIfSameShape(
        map, map->mutable_operand(map_root->parameter_number()));
    return Status::OK();
  }
  if (map_root->opcode() == HloOpcode::kConstant) {
    if (!ShapeUtil::IsScalar(map_root->shape())) {
      return Status::OK();
    }
    auto clone = map_root->CloneWithNewOperands(map_root->shape(), {});
    if (ShapeUtil::IsScalar(map->shape())) {
      return ReplaceWithNewInstruction(map, std::move(clone));
    }
    return ReplaceWithNewInstruction(
        map,
        HloInstruction::CreateBroadcast(
            map->shape(), computation_->AddInstruction(std::move(clone)), {}));
  }
  // Inline the map if the map computation only contains an elementwise
  // operation that can accept arbitrary shapes.
  if (map_root->opcode() == HloOpcode::kFusion || !map_root->IsElementwise()) {
    return Status::OK();
  }
  std::vector<HloInstruction*> new_operands;
  for (auto* root_operand : map_root->operands()) {
    if (root_operand->opcode() != HloOpcode::kParameter) {
      return Status::OK();
    }
    new_operands.push_back(
        map->mutable_operand(root_operand->parameter_number()));
  }
  auto clone = map_root->CloneWithNewOperands(map->shape(), new_operands);
  return ReplaceWithNewInstruction(map, std::move(clone));
}

Status AlgebraicSimplifierVisitor::HandleCustomCall(
    HloInstruction* custom_call) {
  // We elide gradient accumulation ops with `num_mini_batches=1`.
  if (pp::IsPoplarInstruction(PoplarOp::StatefulGradientAccumulate)(
          custom_call)) {
    auto* inst = Cast<pp::HloStatefulGradientAccumulate>(custom_call);
    if (inst->MiniBatchesToAccumulate() == 1) {
      changed_ = true;
      return inst->ReplaceAllUsesWith(inst->mutable_operand(0));
    }
  }
  return Status::OK();
}

PoplarAlgebraicSimplifier::PoplarAlgebraicSimplifier(
    poplarplugin::IpuOptions_IpuAlgebraicSimplifierConfig config)
    : config_(std::move(config)) {}

StatusOr<bool> PoplarAlgebraicSimplifier::Run(HloModule* module) {
  XLA_VLOG_LINES(
      2, "PoplarAlgebraicSimplifier::Run(), before:\n" + module->ToString());
  bool changed = false;
  AlgebraicSimplifierVisitor visitor(this, config_);
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (pp::IsPopOpsFusion(comp)) {
      continue;
    }

    if (visitor.Run(comp, this)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(
      2, "PoplarAlgebraicSimplifier::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
