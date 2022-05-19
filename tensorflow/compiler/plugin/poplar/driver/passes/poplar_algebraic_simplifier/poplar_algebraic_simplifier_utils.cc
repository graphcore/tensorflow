/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier/poplar_algebraic_simplifier_utils.h"

#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"

namespace xla::poplarplugin::algebraic_simplifier::util {
namespace m = match;

bool IsAll(const HloInstruction* op, int8 value) {
  switch (op->opcode()) {
    case HloOpcode::kBroadcast:
      return IsAll(op->operand(0), value);
    case HloOpcode::kConstant:
      return op->literal().IsAll(value);
    default:
      return false;
  }
}

bool IsAllFloat(const HloInstruction* op, float value) {
  switch (op->opcode()) {
    case HloOpcode::kBroadcast:
      return IsAllFloat(op->operand(0), value);
    case HloOpcode::kConstant:
      return op->literal().IsAllFloat(value);
    default:
      return false;
  }
}

// Checks whether `op` is a floating-point constant or broadcast of a constant
// of the form +/- 2^k for some integer k positive, negative, or zero.  Such
// values are interesting because multiplying by a power of 2 just moves the
// exponent.
bool IsAllFpConstantPowerOf2(const HloInstruction* op) {
  // Unwrap the broadcast if necessary.
  const HloInstruction* c;
  if (!Match(op, m::ConstantEffectiveScalar(&c)) &&
      !Match(op, m::Broadcast(m::Constant(&c).WithShape(
                     m::Shape().IsEffectiveScalar())))) {
    return false;
  }
  auto val = [&]() -> absl::optional<double> {
    switch (c->shape().element_type()) {
      case BF16:
        return static_cast<double>(c->literal().GetFirstElement<bfloat16>());
      case F16:
        return static_cast<double>(c->literal().GetFirstElement<Eigen::half>());
      case F32:
        return c->literal().GetFirstElement<float>();
      case F64:
        return c->literal().GetFirstElement<double>();
      default:
        // Cowardly refuse to consider complex types.
        return absl::nullopt;
    }
  }();
  if (!val) {
    return false;
  }

  int exp;
  double mantissa = std::frexp(*val, &exp);
  // frexp returns a value in the range (-1, -0.5] U [0.5, 1).  A return value
  // of +/-0.5 therefore indicates that the floating point value is a power of
  // 2.
  return mantissa == 0.5 || mantissa == -0.5;
}

// Recursive helper for method below.
HloInstruction* BitcastingOperandOfReshapeOrCopyChainHelper(
    HloInstruction* instr, HloInstruction* operand) {
  // If the operand is a copy or reshape try to see if the operand's operand
  // would produce a bitcast with initial instruction.
  if (HloOpcode::kReshape == operand->opcode() ||
      HloOpcode::kCopy == operand->opcode()) {
    return BitcastingOperandOfReshapeOrCopyChainHelper(
        instr, operand->mutable_operand(0));
  }
  return nullptr;
}

bool IsUnstridedSlice(const HloInstruction* hlo) {
  return absl::c_all_of(hlo->slice_strides(),
                        [](int64_t stride) { return stride == 1; });
}

HloInstruction* PreserveFrontendAttributesIfNeeded(
    HloInstruction* new_inst, const HloInstruction* old_inst) {
  if (new_inst->frontend_attributes().map().empty() &&
      !old_inst->frontend_attributes().map().empty()) {
    new_inst->set_frontend_attributes(old_inst->frontend_attributes());
  }
  return new_inst;
}
StatusOr<HloInstruction*> PreserveFrontendAttributesIfNeeded(
    StatusOr<HloInstruction*> new_inst, const HloInstruction* old_inst) {
  if (new_inst.ok()) {
    return PreserveFrontendAttributesIfNeeded(new_inst.ValueOrDie(), old_inst);
  }
  return new_inst;
}

bool IsGlobalAllReduceWithSum(const HloInstruction* all_reduce) {
  if (all_reduce->opcode() != HloOpcode::kAllReduce ||
      !all_reduce->replica_groups().empty()) {
    return false;
  }
  auto& called_computations = all_reduce->called_computations();
  if (called_computations.size() != 1) {
    return false;
  }
  const HloComputation* comp = called_computations.front();
  return Match(comp->root_instruction(),
               m::Add(m::Parameter(), m::Parameter()));
}
}  // namespace xla::poplarplugin::algebraic_simplifier::util
