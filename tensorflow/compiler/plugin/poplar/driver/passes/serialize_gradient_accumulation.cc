/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/passes/serialize_gradient_accumulation.h"

#include <memory>
#include <queue>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/passes/slice_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/scaled_inplace.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {
constexpr char kFusionName[] = "_pop_op_serialized_gradient_accumulation";

HloInstruction* ConvertOperand(HloInstruction* inst, int64 operand_index,
                               PrimitiveType type) {
  CHECK_LT(operand_index, inst->operand_count()) << inst->ToString();
  HloInstruction* convert =
      MakeConvertToHlo(inst->mutable_operand(operand_index), type);
  TF_CHECK_OK(inst->ReplaceOperandWith(operand_index, convert));
  return convert;
}

Status ConvertGradientAccumulatorAdd(HloInstruction* inst) {
  HloComputation* comp = inst->parent();

  // Convert the GradientAccumulatorAdd into a normal add.
  HloInstruction* accumulator = inst->mutable_operand(0);
  HloInstruction* to_serialize = inst->mutable_operand(1);

  const PrimitiveType accumulator_type = accumulator->shape().element_type();
  CHECK(primitive_util::IsFloatingPointType(accumulator_type));

  // The accumulator type may be lower precision than the Add output, but
  // MakeBinaryHlo chooses the higher precision type as the output, so create
  // the Add op manually with the accumulator type.
  HloInstruction* add = comp->AddInstruction(HloInstruction::CreateBinary(
      accumulator->shape(), HloOpcode::kAdd, accumulator, to_serialize));

  CHECK_EQ(add->shape().element_type(), accumulator_type) << add->ToString();
  inst->SetupDerivedInstruction(add);
  TF_RETURN_IF_ERROR(comp->ReplaceInstruction(inst, add));
  if (add->user_count() != 1) {
    return FailedPrecondition(
        "Expected the gradient accumulation buffer to have a single user.");
  }
  HloInstruction* accumulator_user = add->users()[0];

  // Collect all the convert instructions that we are adding, as they need
  // to be outlined later.
  std::vector<HloInstruction*> convert_instructions;

  // Try and serialize the gradient accumulation application.
  HloInstruction* output = add;
  while (output != accumulator) {
    // When trying to serialize the gradients, handle the following
    // patterns:
    // ( 1) Add(a, MultiUpdateAdd(b, c, idx)) =>
    //      MultiUpdateAdd(Add(a, b), c, idx)
    // ( 2) Add(a, Concat(b, c, ...)) =>
    //      SliceApply(SliceApply(a, b), c) ...
    // ( 3) Add(a, Multiply(Concat(b, c, ...), Broadcast(d)) =>
    //      SliceApplyabY(SliceApplyabY(a, b, d), c, d) ...
    // ( 4) Add(a, Multiply(b, Broadcast(c))) =>
    //      ScaledInplaceXbY(a, b, c)
    // ( 5) Add(a, Add(b, c)) =>
    //      Add(Add(a, b), c)
    // ( 6) Add(a, 0) =>
    //      a
    // Following patterns also handle the RHS being a transpose.
    // ( 7) Add(a, Transpose(Concat(b, c, ...))) =>
    //      Add(a, Concat(Transpose(b), Transpose(c), ...)
    // ( 8) Add(a, Transpose(Multiply(Concat(b, c, ...), Broadcast(d))) =>
    //      Add(a, Multiply(Concat(Transpose(b), Transpose(c), ...),
    //                      Broadcast(d))
    // ( 9) Add(a, Transpose(Add(b, c))) =>
    //      Add(a, Add(Transpose(b), Transpose(c)))
    // These patterns try and move the transpose so that patterns 1-5 can be
    // applied.
    // Note that all these patterns keep the accumulator ('a') on the LHS.
    // TODO(T20227) support `Add(a, Transpose(MultiUpdateAdd(b, c, idx)))`.

    HloInstruction* lhs = output->mutable_operand(0);
    HloInstruction* rhs = output->mutable_operand(1);

    // Throughout the accumulation here the lhs and output should have the
    // correct accumulator type. The rhs may however not be in the correct
    // type yet if we are accumulating in a different type than the type of
    // the gradients. This will be handled below in the loop for non-add ops,
    // and after the loop for add ops as converting these in the loop would
    // interfere with later pattern matching.
    CHECK_EQ(lhs->shape().element_type(), accumulator_type) << lhs->ToString();
    CHECK_EQ(output->shape().element_type(), accumulator_type)
        << output->ToString();

    // Skip if rhs has more than a single user.
    if (rhs->user_count() > 1) {
      output = lhs;
      continue;
    }

    if (IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(rhs)) {
      // Case 1:
      // Add(lhs, MultiUpdateAdd(b, ...)) =>
      // MultiUpdateAdd(Add(lhs, b), ...)
      HloInstruction* multi_update_add = rhs;

      // Convert the `updates` and `scale` inputs to MultiUpdateAdd.
      if (multi_update_add->shape().element_type() != accumulator_type) {
        for (int64 i : {2, 3}) {
          convert_instructions.push_back(
              ConvertOperand(multi_update_add, i, accumulator_type));
        }
        multi_update_add->mutable_shape()->set_element_type(accumulator_type);
      }

      HloInstruction* b = multi_update_add->mutable_operand(0);
      TF_RETURN_IF_ERROR(output->ReplaceOperandWith(1, b));
      TF_RETURN_IF_ERROR(multi_update_add->ReplaceOperandWith(0, output));
      TF_RETURN_IF_ERROR(output->ReplaceAllUsesWith(multi_update_add));

    } else if (rhs->opcode() == HloOpcode::kConcatenate) {
      // Case 2:
      // Add(lhs, Concat(b, c, ...)) =>
      // SliceApply(SliceApply(lhs, b), c) ...
      HloInstruction* concat = rhs;

      // Convert all the inputs to Concat.
      if (concat->shape().element_type() != accumulator_type) {
        for (int64 i = 0; i < concat->operand_count(); ++i) {
          convert_instructions.push_back(
              ConvertOperand(concat, i, accumulator_type));
        }
        concat->mutable_shape()->set_element_type(accumulator_type);
      }

      TF_ASSIGN_OR_RETURN(
          HloInstruction * new_output,
          SliceOptimizer::ConvertToSliceApply(HloOpcode::kAdd, lhs, concat));

      TF_RETURN_IF_ERROR(output->ReplaceAllUsesWith(new_output));
      TF_RETURN_IF_ERROR(comp->RemoveInstructionAndUnusedOperands(output));

      output = lhs;

    } else if (Match(rhs, m::Multiply(m::Concatenate(),
                                      m::Broadcast(m::Op().WithShape(
                                          m::Shape().IsScalar()))))) {
      // Case 3:
      // Add(a, Multiply(Concat(b, c, ...), Broadcast(d)) =>
      // SliceApplyabY(SliceApplyabY(a, b, d), c, d)
      HloInstruction* mul = rhs;
      HloInstruction* concat = mul->mutable_operand(0);

      // Convert all the inputs to Concat.
      if (concat->shape().element_type() != accumulator_type) {
        for (int64 i = 0; i < concat->operand_count(); ++i) {
          convert_instructions.push_back(
              ConvertOperand(concat, i, accumulator_type));
        }
      }

      HloInstruction* scalar = mul->mutable_operand(1)->mutable_operand(0);
      if (scalar->shape().element_type() != accumulator_type) {
        scalar = MakeConvertToHlo(scalar, accumulator_type);
        convert_instructions.push_back(scalar);
      }

      TF_ASSIGN_OR_RETURN(HloInstruction * new_output,
                          SliceOptimizer::ConvertToSliceApplyabY(
                              HloOpcode::kAdd, lhs, concat, scalar));
      TF_RETURN_IF_ERROR(comp->ReplaceInstruction(output, new_output));
      output = lhs;

    } else if (Match(rhs, m::Multiply(m::Op(), m::Broadcast(m::Op().WithShape(
                                                   m::Shape().IsScalar()))))) {
      // Case 4:
      // Add(a, Multiply(b, Broadcast(c))) =>
      // ScaledInplaceXbY(a, b, c)
      HloInstruction* b = rhs->mutable_operand(0);
      HloInstruction* scale = rhs->mutable_operand(1)->mutable_operand(0);

      if (b->shape().element_type() != accumulator_type) {
        b = MakeConvertToHlo(b, accumulator_type);
        convert_instructions.push_back(b);
        scale = MakeConvertToHlo(scale, accumulator_type);
        convert_instructions.push_back(scale);
      } else if (scale->opcode() == HloOpcode::kConvert && IsF16(scale) &&
                 IsF32(scale->operand(0))) {
        scale = scale->mutable_operand(0);
      }

      TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
          output, CreateScaledInplaceXbY(lhs, b, scale, HloOpcode::kAdd)));
      output = lhs;

    } else if (rhs->opcode() == HloOpcode::kAdd) {
      // Case 5:
      // Add(lhs, Add(a, b)) =>
      // Add(Add(lhs, a), b)
      HloInstruction* a = rhs->mutable_operand(0);
      HloInstruction* b = rhs->mutable_operand(1);
      TF_RETURN_IF_ERROR(rhs->ReplaceOperandWith(0, lhs));
      TF_RETURN_IF_ERROR(rhs->ReplaceOperandWith(1, a));
      TF_RETURN_IF_ERROR(output->ReplaceOperandWith(0, rhs));
      TF_RETURN_IF_ERROR(output->ReplaceOperandWith(1, b));

      // Make sure that the add we now moved to the lhs is accumulating in the
      // correct type, as this will impact type inferece for later iterations.
      // Converting of the rhs operand is handled below after all the
      // transformations are performed (we do not do it here as it would
      // complicate later pattern matching).
      CHECK(primitive_util::IsFloatingPointType(rhs->shape().element_type()));
      rhs->mutable_shape()->set_element_type(accumulator_type);

    } else if (IsWideConstantZero(rhs)) {
      // Case 6:
      // Add(lhs, zeros) =>
      // lhs
      TF_RETURN_IF_ERROR(comp->ReplaceInstruction(output, lhs));
      output = lhs;

    } else if (Match(rhs, m::Transpose(m::Concatenate())) ||
               Match(rhs, m::Transpose(m::Multiply(
                              m::Concatenate(),
                              m::Broadcast(m::ConstantScalar()))))) {
      // Case 7:
      // Add(a, Transpose(Concat(b, c, ...))) =>
      // Add(a, Concat(Transpose(b), Transpose(c), ...)
      // Case 8:
      // Add(a, Transpose(Multiply(Concat(b, c, ...), Broadcast(d))) =>
      // Add(a, Multiply(Concat(Transpose(b), Transpose(c), ...),
      //                 Broadcast(d))
      HloInstruction* transpose = rhs;
      HloInstruction* concat = transpose->mutable_operand(0);

      // Get the scalar for case 7.
      HloInstruction* scalar = nullptr;
      if (concat->opcode() == HloOpcode::kMultiply) {
        scalar = concat->mutable_operand(1)->mutable_operand(0);
        CHECK_EQ(scalar->opcode(), HloOpcode::kConstant);
        concat = concat->mutable_operand(0);
      }
      CHECK_EQ(concat->opcode(), HloOpcode::kConcatenate);

      // We can push a transpose through a concatenate if only a single
      // dimension is being transposed.
      const std::vector<int64> permutation = transpose->dimensions();
      int64 num_differences = 0;
      for (int64 i = 0; i != permutation.size(); ++i) {
        num_differences += permutation[i] == i ? 0 : 1;
      }
      if (num_differences != 2) {
        // We cannot move this transpose.
        output = lhs;
        continue;
      }

      // Transpose the individual operands of the concat operation.
      auto operands = concat->operands();
      std::vector<HloInstruction*> new_operands(concat->operand_count());
      for (int64 i = 0; i != operands.size(); ++i) {
        TF_ASSIGN_OR_RETURN(new_operands[i],
                            MakeTransposeHlo(operands[i], permutation));
      }

      // Create a concat on the transposed operands, on the permuted
      // concat dimension.
      const int64 concat_dimension = concat->concatenate_dimension();
      const int64 new_concat_dimension = permutation[concat_dimension];
      TF_ASSIGN_OR_RETURN(HloInstruction * new_output,
                          MakeConcatHlo(new_operands, new_concat_dimension));

      // Add a multiply if there was one.
      if (scalar) {
        // Create a new broadcast.
        HloInstruction* bcast =
            MakeBroadcastHlo(scalar, {}, new_output->shape().dimensions());

        // Apply the scalar.
        TF_ASSIGN_OR_RETURN(
            new_output, MakeBinaryHlo(HloOpcode::kMultiply, new_output, bcast));
      }

      TF_RETURN_IF_ERROR(comp->ReplaceInstruction(rhs, new_output));

    } else if (Match(rhs, m::Transpose(m::Add()))) {
      // Case 9:
      // Add(a, Transpose(Add(b, c))) =>
      // Add(a, Add(Transpose(b), Transpose(c)))
      HloInstruction* add = rhs->mutable_operand(0);
      HloInstruction* b = add->mutable_operand(0);
      HloInstruction* c = add->mutable_operand(1);

      // Tranpose b and c.
      TF_ASSIGN_OR_RETURN(HloInstruction * b_t,
                          MakeTransposeHlo(b, rhs->dimensions()));
      TF_ASSIGN_OR_RETURN(HloInstruction * c_t,
                          MakeTransposeHlo(c, rhs->dimensions()));

      // Create a new add.
      TF_ASSIGN_OR_RETURN(HloInstruction * new_add,
                          MakeBinaryHlo(HloOpcode::kAdd, b_t, c_t));

      TF_RETURN_IF_ERROR(comp->ReplaceInstruction(rhs, new_add));

    } else {
      output = lhs;
    }
  }

  // To avoid any other passes modifying the serialized chain of instruction,
  // temporarily put it inside of a fusion computation. The
  // PostSerializeGradientAccumulation pass will inline this back after all the
  // optimizations are performed.
  std::vector<HloInstruction*> to_outline;
  CHECK_EQ(accumulator->user_count(), 1);
  HloInstruction* next_inst = accumulator->users()[0];

  while (next_inst != accumulator_user) {
    to_outline.push_back(next_inst);
    if (next_inst->user_count() != 1) {
      return InternalErrorStrCat("Expected instruction ", next_inst->ToString(),
                                 " to have a single user, but has ",
                                 next_inst->user_count(), ".");
    }
    next_inst = next_inst->users()[0];
  }

  // Add any missing converts for the add rhs operands that we left out above.
  for (auto* inst : to_outline) {
    if (inst->opcode() == HloOpcode::kAdd) {
      // The instruction and lhs operand should already have the correct type.
      CHECK_EQ(inst->shape().element_type(), accumulator_type)
          << inst->ToString();
      CHECK_EQ(inst->operand(0)->shape().element_type(), accumulator_type)
          << inst->ToString();

      // Convert the rhs operand if needed.
      if (inst->operand(1)->shape().element_type() != accumulator_type) {
        convert_instructions.push_back(
            ConvertOperand(inst, 1, accumulator_type));
      }
    }
  }

  if (to_outline.size()) {
    // Insert the convert instructions at the beginning, which gives a valid
    // topological order, as required by the fusion outlining.
    to_outline.insert(to_outline.begin(), convert_instructions.cbegin(),
                      convert_instructions.cend());

    // Make sure that the accumulator is the first argument, as required by the
    // post serialize pass.
    auto* fusion = OutlineExpressionFromComputationWithFusion(
        to_outline, kFusionName, comp, {accumulator});
    CHECK_EQ(fusion->operand_index(accumulator), 0);
  }

  return Status::OK();
}

StatusOr<bool> ConvertGradientAccumulatorAdds(HloModule* module) {
  bool changed = false;
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    // Get all the accumulators.
    std::vector<HloInstruction*> accumulator_adds;
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (IsPoplarInstruction(PoplarOp::GradientAccumulatorAdd)(inst)) {
        accumulator_adds.push_back(inst);
      }
    }

    // Serialize them.
    for (HloInstruction* accumulator_add : accumulator_adds) {
      changed = true;
      TF_RETURN_IF_ERROR(ConvertGradientAccumulatorAdd(accumulator_add));
    }
  }
  return changed;
}
}  // namespace

StatusOr<bool> SerializeGradientAccumulation::Run(HloModule* module) {
  VLOG(2) << "Before SerializeGradientAccumulation:";
  XLA_VLOG_LINES(2, module->ToString());
  TF_ASSIGN_OR_RETURN(const bool changed,
                      ConvertGradientAccumulatorAdds(module));
  if (changed) {
    VLOG(2) << "After SerializeGradientAccumulation:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the module.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
