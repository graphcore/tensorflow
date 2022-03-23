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

#include "tensorflow/compiler/plugin/poplar/driver/passes/copy_inserter.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {
namespace {
bool ShouldAddCopiesForDuplicateOperands(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kTuple: {
      if (inst->GetModule()->entry_computation()->root_instruction() == inst) {
        // Don't add copies if the tuple is the root of the entry computation
        // and it does not need compilation.
        return absl::c_any_of(
            inst->operands(), [](const HloInstruction* operand) {
              return operand->opcode() != HloOpcode::kParameter &&
                     operand->opcode() != HloOpcode::kConstant;
            });
      } else {
        return true;
      }
    }
    case HloOpcode::kCall: {
      return IsRepeatLoop(inst) || IsPipelineOp(inst);
    }
    default: { return false; }
  }
}

StatusOr<bool> AddCopiesForDuplicateOperands(HloInstruction* inst) {
  absl::flat_hash_set<HloInstruction*> seen_operands;
  bool changed = false;
  for (int64 i = 0; i != inst->operand_count(); ++i) {
    HloInstruction* operand = inst->mutable_operand(i);
    if (seen_operands.contains(operand)) {
      // Add a copy if the operand was seen before.
      HloInstruction* copy =
          inst->parent()->AddInstruction(HloInstruction::CreateUnary(
              operand->shape(), HloOpcode::kCopy, operand));
      operand->SetupDerivedInstruction(copy);
      TF_RETURN_IF_ERROR(inst->ReplaceOperandWith(i, copy));
      changed = true;
    } else {
      // First time the operand was seen, don't need to add a copy.
      seen_operands.insert(operand);
    }
  }

  return changed;
}

StatusOr<bool> AddCopiesForResourceUpdate(HloInstruction* resource_update) {
  HloComputation* comp = resource_update->parent();
  HloInstruction* root = comp->root_instruction();
  if (root->opcode() != HloOpcode::kTuple) {
    return false;
  }

  bool changed = false;
  for (int64 i = 0; i != resource_update->operand_count(); ++i) {
    HloInstruction* operand = resource_update->mutable_operand(i);
    if (root->OperandIndices(operand).empty()) {
      continue;
    }

    // Found an operand used by the root tuple - insert a copy to the resource
    // update.
    HloInstruction* copy = comp->AddInstruction(HloInstruction::CreateUnary(
        operand->shape(), HloOpcode::kCopy, operand));
    operand->SetupDerivedInstruction(copy);
    TF_RETURN_IF_ERROR(resource_update->ReplaceOperandWith(i, copy));
    changed = true;
  }
  return changed;
}
}  // namespace

StatusOr<bool> CopyInserter::Run(HloModule* module) {
  VLOG(2) << "Before CopyInserter:";
  XLA_VLOG_LINES(2, module->ToString(HloPrintOptions::ShortParsable()));

  bool changed = false;
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (inst->operand_count() != inst->unique_operands().size() &&
          ShouldAddCopiesForDuplicateOperands(inst)) {
        VLOG(10) << "Removing duplicate operands in " << inst->ToString();
        TF_ASSIGN_OR_RETURN(bool inst_changed,
                            AddCopiesForDuplicateOperands(inst));
        changed |= inst_changed;
      }
      if (IsResourceUpdate(inst)) {
        TF_ASSIGN_OR_RETURN(bool inst_changed,
                            AddCopiesForResourceUpdate(inst));
        changed |= inst_changed;
      }
    }
  }
  if (changed) {
    VLOG(2) << "After CopyInserter:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
