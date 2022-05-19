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

#include "tensorflow/compiler/plugin/poplar/driver/passes/resource_update_copy_inserter.h"

#include <algorithm>
#include <stack>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/copy_into.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {

StatusOr<bool> ResourceUpdateCopyInserter::InsertCopiesInCall(
    HloInstruction* const call) {
  HloComputation* call_comp = call->to_apply();

  // Find a resource update.
  HloInstruction* resource_update;
  {
    std::vector<HloInstruction*> resource_updates;
    absl::c_copy_if(call_comp->MakeInstructionPostOrder(),
                    std::back_inserter(resource_updates), IsResourceUpdate);
    if (resource_updates.size() != 1) {
      return false;
    }
    resource_update = resource_updates[0];
  }
  HloComputation* resource_update_comp = resource_update->to_apply();

  bool changed = false;

  // Make sure the root instructions are tuples.
  TF_ASSIGN_OR_RETURN(bool changed_call_root, FixRootInstruction(call_comp));
  changed |= changed_call_root;

  TF_ASSIGN_OR_RETURN(bool changed_resource_update_root,
                      FixRootInstruction(resource_update_comp));
  changed |= changed_resource_update_root;

  const HloInstruction* call_root = call_comp->root_instruction();
  HloInstruction* resource_update_root =
      resource_update_comp->root_instruction();

  if (call_comp->root_instruction()->opcode() != HloOpcode::kTuple ||
      resource_update_comp->root_instruction()->opcode() != HloOpcode::kTuple) {
    return changed;
  }

  VLOG(2) << "Looking for copies to add inside of " << resource_update->name();
  for (int64_t operand_idx = 0; operand_idx != resource_update->operand_count();
       ++operand_idx) {
    // Only consider adding copies for parameters of the call as those have to
    // alias the output.
    const HloInstruction* operand = resource_update->operand(operand_idx);
    if (operand->opcode() != HloOpcode::kParameter) {
      continue;
    }
    // Check the parameter is modified by the resource update.
    const HloInstruction* gte = call_root->operand(operand->parameter_number());
    if (gte->opcode() != HloOpcode::kGetTupleElement) {
      continue;
    }
    const int64_t output_idx = gte->tuple_index();

    if (gte->operand(0) != resource_update) {
      continue;
    }

    HloInstruction* parameter =
        resource_update_comp->parameter_instruction(operand_idx);
    HloInstruction* output = resource_update_root->mutable_operand(output_idx);
    // Traverse the graph from the output to all its inputs traversing through
    // nodes of inplace type.
    std::stack<const HloInstruction*> to_visit;
    absl::flat_hash_set<const HloInstruction*> visited;
    to_visit.push(output);

    bool found = false;
    while (!to_visit.empty()) {
      const HloInstruction* inst = to_visit.top();
      to_visit.pop();

      // Found a path to the parameter being used inplace.
      if (inst == parameter) {
        found = true;
        break;
      }

      if (visited.contains(inst)) {
        continue;
      }
      visited.insert(inst);
      auto description = GetInplaceDescription(inst);
      if (description.IsInplaceType()) {
        for (auto idx : description.GetInplaceOperandIndices()) {
          to_visit.push(inst->operand(idx));
        }
      }
    }

    if (!found) {
      VLOG(2) << "Adding a copy from " << output->ToString() << " to "
              << parameter->ToString();
      HloInstruction* copy = resource_update_comp->AddInstruction(
          CreateCopyInto(parameter, output));
      TF_RETURN_IF_ERROR(
          resource_update_root->ReplaceOperandWith(output_idx, copy));
      changed = true;
    }
  }

  return changed;
}

StatusOr<bool> ResourceUpdateCopyInserter::Run(HloModule* module) {
  VLOG(2) << "Before ResourceUpdateCopyInserter:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (IsPipelineOp(inst) || IsRepeatLoop(inst)) {
        TF_ASSIGN_OR_RETURN(bool inserted, InsertCopiesInCall(inst));
        changed |= inserted;
      }
    }
  }

  if (changed) {
    VLOG(2) << "After ResourceUpdateCopyInserter:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the Pipeline.";
  }
  return changed;
}
}  // namespace poplarplugin
}  // namespace xla
