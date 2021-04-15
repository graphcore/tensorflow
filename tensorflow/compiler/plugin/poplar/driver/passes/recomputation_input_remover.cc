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

#include "tensorflow/compiler/plugin/poplar/driver/passes/recomputation_input_remover.h"

#include <queue>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"

namespace xla {
namespace poplarplugin {
namespace {

// Get all the source instructions which need to be executed before `inst`.
HloInstructionSet GetSources(HloInstruction* const inst) {
  HloInstructionSet sources;
  std::queue<HloInstruction*> to_visit;
  to_visit.push(inst);
  while (!to_visit.empty()) {
    HloInstruction* top = to_visit.front();
    to_visit.pop();
    if (ContainsKey(sources, top)) {
      continue;
    }
    sources.insert(top);
    for (HloInstruction* operand : top->operands()) {
      to_visit.push(operand);
    }
  }
  return sources;
}

Status InsertControlDependencies(HloInstruction* const checkpointed_input,
                                 HloInstruction* const old_input,
                                 HloComputation* const comp,
                                 HloReachabilityMap* const reachability_map) {
  // Find all the sources to the old input and add control dependencies to make
  // sure any operations with a data path from the checkpointed input are
  // executed before them.
  HloInstructionSet old_input_sources = GetSources(old_input);

  absl::flat_hash_set<HloInstruction*> visited;
  std::queue<HloInstruction*> to_visit;
  to_visit.push(checkpointed_input);

  while (!to_visit.empty()) {
    HloInstruction* user = to_visit.front();
    to_visit.pop();
    if (visited.contains(user)) {
      continue;
    }
    visited.insert(user);

    for (HloInstruction* source : old_input_sources) {
      if (!reachability_map->IsReachable(source, user)) {
        VLOG(2) << "Adding control dependency from " << user->ToString()
                << " to " << source->ToString();
        TF_RETURN_IF_ERROR(user->AddControlDependencyTo(source));
        reachability_map->UpdateReachabilityThroughInstruction(source);
      }
    }

    for (HloInstruction* user : user->users()) {
      to_visit.push(user);
    }
  }

  return Status::OK();
}

StatusOr<bool> HandleComputation(HloComputation* const comp) {
  bool changed = false;
  std::vector<HloInstruction*> to_remove;
  for (auto inst : comp->MakeInstructionPostOrder()) {
    if (IsPoplarInstruction(PoplarOp::RecomputationInput, inst)) {
      to_remove.push_back(inst);
    }
  }
  if (to_remove.size()) {
    auto reachability_map = HloReachabilityMap::Build(comp);
    // Reverse the instructions into reverse post order to allow looking up
    // through the instructions and their operands.
    absl::c_reverse(to_remove);

    // Remove the instructions and add the control dependencies.
    for (HloInstruction* inst : to_remove) {
      VLOG(2) << "Removing " << inst->ToString();
      HloInstruction* checkpointed_input = inst->mutable_operand(0);
      HloInstruction* old_input = inst->mutable_operand(1);

      // Make sure all the users of old_input use the checkpointed value if
      // possible.
      for (HloInstruction* user : old_input->users()) {
        if (user != inst &&
            !reachability_map->IsReachable(user, checkpointed_input)) {
          TF_RETURN_IF_ERROR(
              old_input->ReplaceUseWith(user, checkpointed_input));
          reachability_map->UpdateReachabilityThroughInstruction(user);
        }
      }

      // Remove the recomputation input instruction and make sure all the users
      // use the checkpointed input.
      TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(checkpointed_input));
      TF_RETURN_IF_ERROR(inst->DropAllControlDeps());
      TF_RETURN_IF_ERROR(comp->RemoveInstruction(inst));

      reachability_map = HloReachabilityMap::Build(comp);

      TF_RETURN_IF_ERROR(InsertControlDependencies(
          checkpointed_input, old_input, comp, reachability_map.get()));

      // Remove the old input from the computation if it has no users.
      if (old_input->user_count() == 0) {
        TF_RETURN_IF_ERROR(old_input->DropAllControlDeps());
        TF_RETURN_IF_ERROR(comp->RemoveInstruction(old_input));
      }
    }
    changed = true;
  }
  return changed;
}
}  // namespace

StatusOr<bool> RecomputationInputRemover::Run(HloModule* module) {
  VLOG(2) << "Before RecomputationInputRemover:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  for (auto comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(bool changed_comp, HandleComputation(comp));
    changed |= changed_comp;
  }

  if (changed) {
    VLOG(2) << "After RecomputationInputRemover:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
