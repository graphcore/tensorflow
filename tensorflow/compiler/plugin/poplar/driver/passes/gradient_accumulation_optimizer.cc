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

#include "tensorflow/compiler/plugin/poplar/driver/passes/gradient_accumulation_optimizer.h"

#include <memory>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"

namespace xla {
namespace poplarplugin {
namespace {
StatusOr<bool> ConvertGradientAccumulatorAdds(HloModule* module) {
  bool changed = false;
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }
    auto reachability_map = HloReachabilityMap::Build(comp);
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (!IsPoplarInstruction(PoplarOp::GradientAccumulatorAdd)(inst)) {
        continue;
      }
      // Convert the GradientAccumulatorAdd into a normal add.
      HloInstruction* lhs = inst->mutable_operand(0);
      HloInstruction* rhs = inst->mutable_operand(1);
      TF_ASSIGN_OR_RETURN(HloInstruction * new_add,
                          MakeBinaryHlo(HloOpcode::kAdd, lhs, rhs));
      inst->SetupDerivedInstruction(new_add);
      // Copy control dependencies.
      TF_RETURN_IF_ERROR(new_add->CopyAllControlDepsFrom(inst));
      TF_RETURN_IF_ERROR(inst->DropAllControlDeps());
      // Add a control dependency from the rhs to lhs to make sure the gradient
      // accumulation buffer is allocated as late as possible.
      if (!reachability_map->IsReachable(rhs, lhs)) {
        TF_CHECK_OK(rhs->AddControlDependencyTo(lhs));
      }
      TF_RETURN_IF_ERROR(comp->ReplaceInstruction(inst, new_add));
      changed = true;

      // Rebuild the map.
      reachability_map = HloReachabilityMap::Build(comp);
    }
  }
  return changed;
}

// Find all the gradient accumulation buffer creators and add dependencies so
// that these are executed as late as possible to make sure the variable has
// been allocated before (incase it was a deferred allocation).
StatusOr<bool> AddAllocationControlDependencies(HloModule* module) {
  bool changed = false;
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }
    auto reachability_map = HloReachabilityMap::Build(comp);
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (!IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(inst)) {
        continue;
      }

      HloInstruction* layout_input = inst->mutable_operand(0);
      // Try and add control dependencies to other layout input users.
      for (HloInstruction* peer : layout_input->users()) {
        // Skip current instruction.
        if (peer == inst) {
          continue;
        }
        // Skip if there already is a dependency.
        if (reachability_map->IsReachable(inst, peer)) {
          continue;
        }
        // Skip if the peer uses layout_input inplace - we prioritize inplace
        // instructions.
        const HloInstructionDescription description =
            HloInstructionDescription(peer);

        absl::flat_hash_set<int64> inplace_indicies = {
            description.GetInplaceOperandIndexes().begin(),
            description.GetInplaceOperandIndexes().end()};

        const bool can_be_inplace =
            absl::c_any_of(peer->OperandIndices(layout_input),
                           [&inplace_indicies](int64 operand_idx) {
                             return inplace_indicies.contains(operand_idx);
                           });

        if (can_be_inplace) {
          continue;
        }

        TF_RETURN_IF_ERROR(peer->AddControlDependencyTo(inst));
        reachability_map->UpdateReachabilityThroughInstruction(inst);
        changed = true;
      }
    }
  }

  return changed;
}
}  // namespace

StatusOr<bool> GradientAccumulationOptimizer::Run(HloModule* module) {
  VLOG(2) << "Before GradientAccumulationOptimizer:";
  XLA_VLOG_LINES(2, module->ToString());
  TF_ASSIGN_OR_RETURN(const bool changed_accumulators,
                      ConvertGradientAccumulatorAdds(module));
  TF_ASSIGN_OR_RETURN(const bool changed_creators,
                      AddAllocationControlDependencies(module));
  const bool changed = changed_accumulators || changed_creators;
  if (changed) {
    VLOG(2) << "After GradientAccumulationOptimizer:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the module.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
