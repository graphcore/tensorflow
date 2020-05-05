/* Copyright 2018 Graphcore Ltd

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/commutative_instruction_reorder_operands.h"

#include <map>

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {
namespace {
bool ShouldAlwaysBeRhs(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kAddDependency) {
    inst = inst->operand(0);
  }
  switch (inst->opcode()) {
    case HloOpcode::kBroadcast:
      return true;
    default:
      return false;
  }
}
bool PreferRhs(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kAddDependency) {
    inst = inst->operand(0);
  }
  switch (inst->opcode()) {
    case HloOpcode::kConcatenate:
    case HloOpcode::kReshape:
    case HloOpcode::kPad:
      return true;
    default:
      return false;
  }
}

bool IsElementwiseBinaryCommutative(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
      return true;
    default:
      return false;
  }
}
}  // namespace

StatusOr<bool> CommutativeInstructionReorderOperands::Run(HloModule* module) {
  bool changed = false;
  for (auto* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (auto* inst : comp->MakeInstructionPostOrder()) {
      if (!IsElementwiseBinaryCommutative(inst)) {
        continue;
      }
      HloInstruction* lhs = inst->mutable_operand(0);
      HloInstruction* rhs = inst->mutable_operand(1);

      bool reorder = false;
      // Check if this is something which is always wanted as rhs.
      reorder |= ShouldAlwaysBeRhs(lhs) && !ShouldAlwaysBeRhs(rhs);
      // Check if this is something that should be on the rhs.
      reorder |= PreferRhs(lhs) && !(PreferRhs(rhs) || ShouldAlwaysBeRhs(rhs));
      if (reorder) {
        TF_RETURN_IF_ERROR(inst->ReplaceOperandWith(0, rhs));
        TF_RETURN_IF_ERROR(inst->ReplaceOperandWith(1, lhs));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
