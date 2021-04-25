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

#include "tensorflow/compiler/plugin/poplar/driver/passes/recompute_casts.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"

namespace xla {
namespace poplarplugin {

namespace {

StatusOr<absl::flat_hash_set<const HloComputation*>> FindResourceUpdateCallees(
    HloModule* module) {
  auto call_graph = CallGraph::Build(module);

  absl::flat_hash_set<const HloComputation*> resource_update_computations;
  for (auto comp : module->computations()) {
    for (auto inst : comp->instructions()) {
      if (IsResourceUpdate(inst)) {
        TF_ASSIGN_OR_RETURN(absl::flat_hash_set<HloComputation*> called_comps,
                            GetAllComputationsCalledBy(inst, call_graph.get()));
        resource_update_computations.insert(called_comps.begin(),
                                            called_comps.end());
      }
    }
  }

  return {resource_update_computations};
}
}  // namespace

StatusOr<bool> RecomputeCasts::Run(HloModule* module) {
  VLOG(2) << "Before RecomputeCasts:";
  XLA_VLOG_LINES(2, module->ToString());

  bool made_recomputations = false;

  TF_ASSIGN_OR_RETURN(const auto resource_update_computations,
                      FindResourceUpdateCallees(module));

  for (auto comp : module->MakeComputationPostOrder()) {
    const auto recomputation_blocked =
        IsPopOpsFusion(comp) || resource_update_computations.contains(comp);
    if (recomputation_blocked) {
      continue;
    }

    for (auto inst : comp->MakeInstructionPostOrder()) {
      const auto parameter_cast =
          inst->opcode() == HloOpcode::kConvert &&
          inst->operand(0)->opcode() == HloOpcode::kParameter &&
          inst->user_count() > 1;

      if (parameter_cast) {
        TF_RETURN_IF_ERROR(SetupRecomputation(comp, inst));
        made_recomputations = true;
      }
    }
  }

  if (made_recomputations) {
    VLOG(2) << "After RecomputeCasts:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made.";
  }

  return made_recomputations;
}

Status RecomputeCasts::SetupRecomputation(HloComputation* comp,
                                          HloInstruction* inst) {
  VLOG(2) << "Recomputing " << inst->ToString();
  auto users = inst->users();
  for (auto i = 0; i < users.size(); ++i) {
    auto user = users[i];
    auto replacement = comp->AddInstruction(inst->Clone());
    inst->SetupDerivedInstruction(replacement);
    auto reachability_map = HloReachabilityMap::Build(comp);

    for (auto operand : user->operands()) {
      if (operand != inst &&
          !reachability_map->IsReachable(replacement, operand)) {
        TF_RETURN_IF_ERROR(operand->AddControlDependencyTo(replacement));
        reachability_map->UpdateReachabilityThroughInstruction(replacement);
      }
    }

    for (auto predecessor : inst->control_predecessors()) {
      if (!reachability_map->IsReachable(replacement, predecessor)) {
        TF_RETURN_IF_ERROR(predecessor->AddControlDependencyTo(replacement));
        reachability_map->UpdateReachabilityThroughInstruction(replacement);
      }
    }

    for (auto successor : inst->control_successors()) {
      if (!reachability_map->IsReachable(successor, replacement)) {
        TF_RETURN_IF_ERROR(replacement->AddControlDependencyTo(successor));
        reachability_map->UpdateReachabilityThroughInstruction(successor);
      }
    }

    TF_RETURN_IF_ERROR(inst->ReplaceUseWith(user, replacement));
  }
  // Cant remove the instruction while it has control dependencies
  TF_RETURN_IF_ERROR(inst->DropAllControlDeps());
  TF_RETURN_IF_ERROR(comp->RemoveInstruction(inst));

  return Status::OK();
}
}  // namespace poplarplugin
}  // namespace xla
