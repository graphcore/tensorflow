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

#include "tensorflow/compiler/plugin/poplar/driver/passes/call_optimizer.h"

#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {

std::vector<HloInstruction*> OrderInnerPipelineFunctions(
    PipelineStages& stages) {
  const bool has_resource_update = stages.resource_update.has_value();

  std::vector<HloInstruction*> ordered_stages(stages.forward.size() +
                                              stages.backward.size() +
                                              (has_resource_update ? 1 : 0));
  absl::c_copy(stages.forward, ordered_stages.begin());
  std::copy(stages.backward.rbegin(), stages.backward.rend(),
            std::next(ordered_stages.begin(), stages.forward.size()));
  if (has_resource_update) {
    ordered_stages.back() = *stages.resource_update;
  }
  absl::c_reverse(ordered_stages);
  return ordered_stages;
}

Status ReplaceOutputUses(HloInstruction* stage,
                         std::map<int64, std::set<int64>> duplicate_outputs) {
  // Get all the GTEs by tuple index.
  absl::flat_hash_map<int64, HloInstructionSet> gte_users;
  for (HloInstruction* user : stage->users()) {
    CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
    gte_users[user->tuple_index()].insert(user);
  }

  // Replace all duplicate uses with a single GTE.
  for (auto pair : duplicate_outputs) {
    int64 output_idx = pair.first;
    std::set<int64> duplicate_indices = pair.second;
    VLOG(3) << "Replacing duplicate output indices "
            << absl::StrJoin(duplicate_indices, ", ") << " with output index "
            << output_idx;
    TF_ASSIGN_OR_RETURN(HloInstruction * output_gte,
                        MakeGetTupleElementHlo(stage, output_idx));

    for (int64 duplicate_idx : duplicate_indices) {
      for (HloInstruction* duplicate_inst : gte_users[duplicate_idx]) {
        TF_RETURN_IF_ERROR(duplicate_inst->ReplaceAllUsesWith(output_gte));
      }
    }
    // Also replace all the GTEs for output_idx (if any) with the new gte.
    for (HloInstruction* old_gte : gte_users[output_idx]) {
      TF_RETURN_IF_ERROR(old_gte->ReplaceAllUsesWith(output_gte));
    }
  }
  return Status::OK();
}

Status ReplaceDuplicateInputs(
    HloInstruction* stage, std::map<int64, std::set<int64>> duplicate_inputs) {
  HloComputation* stage_comp = stage->to_apply();
  // Replace any duplicate inputs which will make parameters unused.
  for (auto pair : duplicate_inputs) {
    int64 param_number = pair.first;
    std::set<int64> duplicate_indices = pair.second;
    VLOG(3) << "Replacing duplicate parameter numbers "
            << absl::StrJoin(duplicate_indices, ", ")
            << " with parameter number " << param_number;
    HloInstruction* parameter = stage_comp->parameter_instruction(param_number);
    for (int64 duplicate_idx : duplicate_indices) {
      HloInstruction* parameter_to_replace =
          stage_comp->parameter_instruction(duplicate_idx);
      TF_RETURN_IF_ERROR(parameter_to_replace->ReplaceAllUsesWith(parameter));
    }
  }
  return Status::OK();
}

absl::flat_hash_map<int64, std::vector<HloInstruction*>> GetAllGtes(
    HloInstruction* const stage) {
  absl::flat_hash_map<int64, std::vector<HloInstruction*>> gte_users;
  for (HloInstruction* user : stage->users()) {
    CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
    gte_users[user->tuple_index()].push_back(user);
  }
  return gte_users;
}

StatusOr<bool> PropagatePadsAndBroadcasts(HloInstruction* const stage) {
  bool changed = false;
  std::map<int64, HloInstruction*> outputs_to_propagate;
  HloComputation* pipeline_comp = stage->parent();
  HloComputation* stage_comp = stage->to_apply();
  HloInstruction* stage_root = stage_comp->root_instruction();
  CHECK_EQ(stage_root->opcode(), HloOpcode::kTuple);

  for (int64 i = 0; i != stage_root->operand_count(); ++i) {
    HloInstruction* operand = stage_root->mutable_operand(i);
    switch (operand->opcode()) {
      case HloOpcode::kBroadcast:
      case HloOpcode::kPad: {
        outputs_to_propagate[i] = operand;
        break;
      }
      default: { break; }
    }
  }

  absl::flat_hash_map<int64, std::vector<HloInstruction*>> gte_users =
      GetAllGtes(stage);

  // Go through all the users of padded/broadcasted outputs and do the
  // padding/broadcast in the user.
  for (auto& output_pair : outputs_to_propagate) {
    auto itr = gte_users.find(output_pair.first);
    if (itr == gte_users.end()) {
      continue;
    }

    // Go through all the users of the output.
    for (HloInstruction* gte : itr->second) {
      for (HloInstruction* user : gte->users()) {
        if (user->opcode() != HloOpcode::kCall) {
          continue;
        }
        const int64 num_outputs = ShapeUtil::TupleElementCount(stage->shape());
        // Add the operands of the output being cloned as an output of the
        // stage.
        HloInstruction* output = output_pair.second;
        {
          // New root has all the same inputs, plus the operands of the
          // instruction being propagated.
          auto new_outputs = stage_root->operands();
          new_outputs.insert(new_outputs.end(), output->operands().begin(),
                             output->operands().end());

          // Create the new root.
          HloInstruction* new_stage_root = stage_comp->AddInstruction(
              HloInstruction::CreateTuple(new_outputs));
          stage_root->SetupDerivedInstruction(new_stage_root);

          // Use the new root and change the shape of the output.
          stage_comp->set_root_instruction(new_stage_root, true);
          stage_root = new_stage_root;
          *stage->mutable_shape() = new_stage_root->shape();
        }

        // Add GTEs for the new outputs.
        std::vector<HloInstruction*> new_gtes(output->operand_count());
        for (int64 i = 0; i != new_gtes.size(); ++i) {
          TF_ASSIGN_OR_RETURN(new_gtes[i],
                              MakeGetTupleElementHlo(stage, num_outputs + i));
        }

        // Clone the instruction inside the pipeline with the new outputs of the
        // stage.
        HloInstruction* output_clone = pipeline_comp->AddInstruction(
            output->CloneWithNewOperands(output->shape(), new_gtes));

        // Lower the output into the stage and replace all the uses with it.
        std::map<int64, HloInstruction*> replacements;
        absl::c_for_each(user->OperandIndices(gte), [&](int64 operand_idx) {
          replacements[operand_idx] = output_clone;
        });
        TF_RETURN_IF_ERROR(
            AddInstructionsToPipelineStage(user, {output_clone}, replacements)
                .status());
        changed = true;
      }
    }
  }
  return changed;
}

StatusOr<bool> PropagateConstantOutputs(HloInstruction* const stage) {
  std::map<int64, HloInstruction*> constant_outputs;
  HloComputation* stage_comp = stage->to_apply();
  HloInstruction* root = stage_comp->root_instruction();

  // Find any constant outputs.
  for (int64 i = 0; i != root->operand_count(); ++i) {
    if (root->mutable_operand(i)->opcode() == HloOpcode::kConstant) {
      constant_outputs[i] = root->mutable_operand(i);
    }
  }

  absl::flat_hash_map<int64, std::vector<HloInstruction*>> gte_users =
      GetAllGtes(stage);

  // Go through all the constant outputs and replace the users of the
  // constants with constants.
  bool changed = false;
  for (auto& constant_output_pair : constant_outputs) {
    auto itr = gte_users.find(constant_output_pair.first);
    if (itr == gte_users.end()) {
      continue;
    }

    // Go through all the users of the output.
    for (HloInstruction* gte : itr->second) {
      for (HloInstruction* gte_user : gte->users()) {
        // Only propagate the constant to other computations.
        if (gte_user->opcode() == HloOpcode::kCall) {
          HloComputation* comp = gte_user->to_apply();
          // Create the constant inside of the computation.
          HloInstruction* propagated_const =
              comp->AddInstruction(constant_output_pair.second->Clone());

          for (int64 input_index : gte_user->OperandIndices(gte)) {
            // Get the parameter instruction for this operand.
            HloInstruction* parameter =
                comp->parameter_instruction(input_index);
            // Replace all the users with the constant.
            TF_RETURN_IF_ERROR(parameter->ReplaceAllUsesWith(propagated_const));
            changed = true;
          }
        }
      }
    }
  }

  return changed;
}
}  // namespace

StatusOr<HloInstruction*> CallOptimizer::OptimizeCallInstruction(
    HloInstruction* inst, bool* changed) {
  VLOG(2) << "Optimizing: " << inst->ToString();
  // Propagate operations which are safe to be moved and save on size/aliasing.
  TF_ASSIGN_OR_RETURN(bool propagated_pads_and_broadcasts,
                      PropagatePadsAndBroadcasts(inst));
  TF_ASSIGN_OR_RETURN(bool propagated_constants,
                      PropagateConstantOutputs(inst));

  // Find any duplicate outputs.
  TF_ASSIGN_OR_RETURN(auto duplicate_outputs, GetDuplicateCallOutputs(inst));
  if (duplicate_outputs.size()) {
    VLOG(3) << "Replacing duplicate outputs.";
    TF_RETURN_IF_ERROR(ReplaceOutputUses(inst, duplicate_outputs));
  }

  // Find any unused outputs.
  TF_ASSIGN_OR_RETURN(std::set<int64> unused_outputs,
                      GetUnusedCallOutputIndices(inst));
  if (unused_outputs.size()) {
    VLOG(3) << "Removing unused outputs.";
    TF_RETURN_IF_ERROR(RemoveOutputsFromCall(inst, unused_outputs));
    TF_RETURN_IF_ERROR(
        HloDCE()
            .RunOnComputation(inst->to_apply(),
                              /*remove_cross_partition_collective_ops=*/false)
            .status());
  }

  // Find any duplicate inputs and change the parameters such that they become
  // unused.
  TF_ASSIGN_OR_RETURN(auto duplicate_inputs, GetDuplicateCallInputs(inst));
  if (duplicate_inputs.size()) {
    VLOG(3) << "Replacing duplicate inputs.";
    TF_RETURN_IF_ERROR(ReplaceDuplicateInputs(inst, duplicate_inputs));
  }

  // Find any unused inputs and remove them.
  TF_ASSIGN_OR_RETURN(std::set<int64> unused_parameters,
                      GetUnusedParametersInCall(inst));
  if (unused_parameters.size()) {
    VLOG(3) << "Removing unused inputs.";
    TF_ASSIGN_OR_RETURN(inst,
                        RemoveParametersFromCall(inst, unused_parameters));
    TF_RETURN_IF_ERROR(
        HloDCE()
            .RunOnComputation(inst->to_apply(),
                              /*remove_cross_partition_collective_ops=*/false)
            .status());
  }

  (*changed) |= (propagated_pads_and_broadcasts || propagated_constants ||
                 duplicate_outputs.size() || unused_outputs.size() ||
                 duplicate_inputs.size() || unused_parameters.size());
  return inst;
}

StatusOr<bool> CallOptimizer::OptimizePipeline(HloInstruction* pipeline_op) {
  bool changed = false;

  HloComputation* pipeline_comp = pipeline_op->to_apply();

  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  // Make sure that the root of each stage is a tuple.
  TF_RETURN_IF_ERROR(FixRootInstructions(stages));
  // For each stage, starting from the last stage to last.
  std::vector<HloInstruction*> ordered = OrderInnerPipelineFunctions(stages);
  for (HloInstruction* call : ordered) {
    bool call_changed = false;
    TF_ASSIGN_OR_RETURN(call, OptimizeCallInstruction(call, &call_changed));
    changed |= call_changed;
  }

  return changed;
}

StatusOr<bool> CallOptimizer::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to optimize.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  VLOG(2) << "Before CallOptimizer:";
  XLA_VLOG_LINES(2, module->ToString());

  TF_ASSIGN_OR_RETURN(bool changed, OptimizePipeline(pipeline_ops[0]));

  VLOG(2) << "After CallOptimizer:";
  XLA_VLOG_LINES(2, module->ToString());
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
