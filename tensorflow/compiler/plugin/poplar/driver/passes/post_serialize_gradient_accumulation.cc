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

#include "tensorflow/compiler/plugin/poplar/driver/passes/post_serialize_gradient_accumulation.h"

#include <memory>
#include <stack>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"

namespace xla {
namespace poplarplugin {
namespace {

Status InlineSerializedGradientAccumulation(HloInstruction* fusion) {
  HloComputation* fusion_comp = fusion->fused_instructions_computation();
  HloComputation* comp = fusion->parent();
  HloInstruction* accumulator = fusion->mutable_operand(0);
  // Inline the computation.
  TF_ASSIGN_OR_RETURN(auto mapping, InlineComputation(fusion, fusion_comp,
                                                      /*copy_sharding*/ true));

  auto reachability_map = HloReachabilityMap::Build(comp);

  auto add_control_dependency = [&reachability_map](
                                    HloInstruction* const from,
                                    HloInstruction* const to) -> Status {
    // Add a control dependency as long as it doesn't create a cycle.
    if (from != to && !reachability_map->IsReachable(to, from)) {
      TF_RETURN_IF_ERROR(from->AddControlDependencyTo(to));
      reachability_map->UpdateReachabilityThroughInstruction(to);
    }
    return Status::OK();
  };

  // Add a control dependency to the accumulator from all other operands
  // from its user to make sure it is allocated late as possible.
  CHECK_EQ(accumulator->user_count(), 1);
  HloInstruction* accumulator_user = accumulator->users()[0];
  for (HloInstruction* operand : accumulator_user->operands()) {
    TF_RETURN_IF_ERROR(add_control_dependency(operand, accumulator));
  }

  // Add control dependencies so that each instruction adding to the gradient is
  // executed as soon as possible.
  for (auto pair : mapping) {
    if (pair.first->opcode() == HloOpcode::kParameter) {
      continue;
    }

    HloInstruction* inst = pair.second;
    HloInstructionSet input_set;
    for (int64 op_idx = 1; op_idx != inst->operand_count(); ++op_idx) {
      HloInstruction* operand = inst->mutable_operand(op_idx);
      input_set.insert(operand);
      input_set.insert(operand->LatestNonGteAncestor());
    }

    for (HloInstruction* input : input_set) {
      // Skip scalar dependencies.
      if (ShapeUtil::ElementsIn(input->shape()) == 1) {
        continue;
      }

      for (HloInstruction* user : input->users()) {
        TF_RETURN_IF_ERROR(add_control_dependency(inst, user));
      }
    }
  }

  return Status::OK();
}

StatusOr<bool> InlineSerializedGradientAccumulations(HloModule* module) {
  bool changed = false;
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    // Find the fusions inside of the computations.
    std::vector<HloInstruction*> accumulator_fusions;
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (IsSerializedGradientAccumulation(inst)) {
        accumulator_fusions.push_back(inst);
      }
    }

    // Inline them.
    for (HloInstruction* accumulator_fusion : accumulator_fusions) {
      changed = true;
      TF_RETURN_IF_ERROR(
          InlineSerializedGradientAccumulation(accumulator_fusion));
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

      // GradientAccumulatorCreate also accepts shape
      // as its ctor argument as an alternative to instruction.
      // This means no peer instruction is available here.
      if (inst->operand_count() == 0) {
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
        auto description = GetInplaceDescription(peer);
        const auto& inplace_indices = description.GetInplaceOperandSet();

        const bool can_be_inplace =
            absl::c_any_of(peer->OperandIndices(layout_input),
                           [&inplace_indices](int64 operand_idx) {
                             return inplace_indices.contains(operand_idx);
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

StatusOr<bool> PostSerializeGradientAccumulation::Run(HloModule* module) {
  VLOG(2) << "Before PostSerializeGradientAccumulation:";
  XLA_VLOG_LINES(2, module->ToString());
  TF_ASSIGN_OR_RETURN(const bool inlined_accumulators,
                      InlineSerializedGradientAccumulations(module));
  TF_ASSIGN_OR_RETURN(const bool changed_creators,
                      AddAllocationControlDependencies(module));

  const bool changed = inlined_accumulators || changed_creators;
  if (changed) {
    VLOG(2) << "After PostSerializeGradientAccumulation:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the module.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
