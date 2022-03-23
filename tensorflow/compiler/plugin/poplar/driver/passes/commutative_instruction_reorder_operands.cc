/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/scaled_inplace.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
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
    default: { break; }
  }
  return false;
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
    default: { break; }
  }
  return false;
}

bool IsElementwiseBinaryCommutative(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply: {
      return true;
    }
    case HloOpcode::kCustomCall: {
      return IsAnyScaledInplace(inst) &&
             Cast<HloScaledInplaceBase>(inst)->GetOperation() ==
                 HloOpcode::kAdd;
    }
    default: { break; }
  }
  return false;
}

Status ReorderOperands(HloInstruction* inst) {
  VLOG(2) << "Reordering operands for " << inst->ToString();
  HloInstruction* lhs = inst->mutable_operand(0);
  HloInstruction* rhs = inst->mutable_operand(1);
  switch (inst->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply: {
      TF_RETURN_IF_ERROR(inst->ReplaceOperandWith(0, rhs));
      TF_RETURN_IF_ERROR(inst->ReplaceOperandWith(1, lhs));
      return Status::OK();
    }
    case HloOpcode::kCustomCall: {
      HloComputation* comp = inst->parent();
      // ScaledInplaceXbY is converted into a ScaledInplaceaXbY first as there
      // is no ScaledInplaceaXY variant.
      if (IsPoplarInstruction(PoplarOp::ScaledInplaceXbY)(inst)) {
        HloInstruction* scale = inst->mutable_operand(2);
        HloInstruction* one =
            comp->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::One(scale->shape().element_type())));
        HloInstruction* new_inst = comp->AddInstruction(CreateScaledInplaceaXbY(
            lhs, rhs, one, scale,
            Cast<HloScaledInplaceXbY>(inst)->GetOperation()));
        TF_RETURN_IF_ERROR(comp->ReplaceInstruction(inst, new_inst));
        inst->SetupDerivedInstruction(new_inst);
        inst = new_inst;
      }
      if (IsPoplarInstruction(PoplarOp::ScaledInplaceaXbY)(inst)) {
        HloInstruction* a_scale = inst->mutable_operand(2);
        HloInstruction* b_scale = inst->mutable_operand(3);
        TF_RETURN_IF_ERROR(inst->ReplaceOperandWith(0, rhs));
        TF_RETURN_IF_ERROR(inst->ReplaceOperandWith(1, lhs));
        TF_RETURN_IF_ERROR(inst->ReplaceOperandWith(2, b_scale));
        TF_RETURN_IF_ERROR(inst->ReplaceOperandWith(3, a_scale));
        return Status::OK();
      }
      break;
    }
    default: { break; }
  }
  return InternalErrorStrCat("Cannot reorder operands for ", inst->ToString());
}
}  // namespace

StatusOr<bool> CommutativeInstructionReorderOperands::Run(HloModule* module) {
  bool changed = false;
  for (auto* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    auto reachability_map = HloReachabilityMap::Build(comp);

    std::vector<HloInstruction*> to_reorder;
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      if (!IsElementwiseBinaryCommutative(inst)) {
        continue;
      }
      const HloInstruction* lhs = inst->operand(0);
      const HloInstruction* rhs = inst->operand(1);

      // Check if this is something which is always wanted as rhs.
      if (ShouldAlwaysBeRhs(lhs) && !ShouldAlwaysBeRhs(rhs)) {
        to_reorder.push_back(inst);
        continue;
      }
      // Check if this is something that should be on the rhs.
      if (PreferRhs(lhs) && !(PreferRhs(rhs) || ShouldAlwaysBeRhs(rhs))) {
        to_reorder.push_back(inst);
        continue;
      }
      // Check whether the lhs can still be live after this instruction
      // executes, but rhs is not live. Swap them to allow this operation to be
      // inplace.
      auto is_live_after =
          [&inst, &reachability_map](const HloInstruction* input) -> bool {
        for (const HloInstruction* user : input->users()) {
          if (inst == user) {
            continue;
          }
          if (reachability_map->IsReachable(inst, user)) {
            return true;
          }
        }
        return false;
      };

      if (!ShouldAlwaysBeRhs(rhs) && is_live_after(lhs) &&
          !is_live_after(rhs)) {
        to_reorder.push_back(inst);
        continue;
      }
    }

    for (HloInstruction* inst : to_reorder) {
      TF_RETURN_IF_ERROR(ReorderOperands(inst));
      changed = true;
    }
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
