/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_optimizer.h"

#include <algorithm>
#include <set>
#include <utility>

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
  absl::flat_hash_map<int64, absl::flat_hash_set<HloInstruction*>> gte_users;
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

StatusOr<bool> MoveParameterInputsToBackwardStages(
    HloComputation* pipeline_comp) {
  bool changed = false;
  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  // Skip this optimisation if we do not have the backward stages.
  if (stages.backward.empty()) {
    return changed;
  }
  // Go through forward stages, if any stage has a Pipeline parameter as an
  // input, which is passed straight through to the root tuple and then passed
  // as an input to the bwd pass, then make the bwd stage use the parameter
  // instead.
  // For example, replaces:
  // p = parameter(10)
  //        ||
  //        \/
  // c = forward_pipeline_stage(p)
  //  ________________________________       __________________________________
  // | p_in_comp = parameter(0)       |     | p = parameter(0)                 |
  // | log = log(p_in_comp)           |     | ...                              |
  // | ROOT t = tuple(p_in_comp, log) |     |__________________________________|
  // |________________________________|     c_bwd = forward_pipeline_stage(gte)
  //        ||                                            /\
  //        \/                                            ||
  //    gte = gte(c), index=0 =============================
  //
  // with:
  // p = parameter(10)====================
  //        ||                           ||
  //        \/                           ||
  // c = forward_pipeline_stage(p)       ||
  //  ________________________________   ||  __________________________________
  // | p_in_comp = parameter(0)       |  || | p = parameter(0)                 |
  // | log = log(p_in_comp)           |  || | ...                              |
  // | ROOT t = tuple(p_in_comp, log) |  || |__________________________________|
  // |________________________________|   => c_bwd = forward_pipeline_stage(gte)
  // This will avoid having FIFOs for variables.

  for (size_t stage_id = 0; stage_id != stages.forward.size(); ++stage_id) {
    HloInstruction* fwd_stage = stages.forward[stage_id];
    HloInstruction* bwd_stage = stages.backward[stage_id];
    // Go through the inputs to the fwd stage and identify operand indices
    // which are parameters.
    std::set<int64> parameter_input_indices;
    for (int64 op_idx = 0; op_idx != fwd_stage->operand_count(); ++op_idx) {
      if (fwd_stage->operand(op_idx)->opcode() == HloOpcode::kParameter) {
        parameter_input_indices.insert(op_idx);
      }
    }
    if (parameter_input_indices.empty()) {
      continue;
    }
    // Map all the output GTEs from their tuple_index to the instruction.
    absl::flat_hash_map<int64, HloInstruction*> gtes;
    for (HloInstruction* user : fwd_stage->users()) {
      gtes[user->tuple_index()] = user;
    }

    // Given those indices, check which of them are used in the root tuple and
    // then used by the bwd stage.
    HloComputation* fwd_stage_comp = fwd_stage->to_apply();
    HloInstruction* fwd_stage_root = fwd_stage_comp->root_instruction();
    CHECK_EQ(fwd_stage_root->opcode(), HloOpcode::kTuple);
    for (int64 param_idx : parameter_input_indices) {
      HloInstruction* parameter =
          fwd_stage_comp->parameter_instruction(param_idx);
      // Given the parameter, go through all the uses in the root.
      for (int64 output_idx : fwd_stage_root->OperandIndices(parameter)) {
        if (!gtes.contains(output_idx)) {
          continue;
        }
        // Get all the indices the bwd stages uses this GTE.
        auto uses = bwd_stage->OperandIndices(gtes[output_idx]);
        for (int64 use_idx : uses) {
          TF_RETURN_IF_ERROR(bwd_stage->ReplaceOperandWith(
              use_idx, fwd_stage->mutable_operand(param_idx)));
          changed = true;
        }
      }
    }
  }
  return changed;
}
}  // namespace

StatusOr<HloInstruction*> PipelineOptimizer::OptimizeCallInstruction(
    HloInstruction* inst, bool* changed) {
  VLOG(2) << "Optimizing: " << inst->ToString();
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
    TF_RETURN_IF_ERROR(HloDCE::RunOnComputation(inst->to_apply()).status());
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
    TF_RETURN_IF_ERROR(HloDCE::RunOnComputation(inst->to_apply()).status());
  }

  (*changed) |= (duplicate_outputs.size() || unused_outputs.size() ||
                 duplicate_inputs.size() || unused_parameters.size());
  return inst;
}

StatusOr<bool> PipelineOptimizer::OptimizePipeline(
    HloInstruction* pipeline_op) {
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
  TF_ASSIGN_OR_RETURN(bool moved_parameters,
                      MoveParameterInputsToBackwardStages(pipeline_comp));
  changed |= moved_parameters;

  return changed;
}

StatusOr<bool> PipelineOptimizer::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> pipeline_ops,
                      GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to optimize.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  VLOG(2) << "Before PipelineOptimizer:";
  XLA_VLOG_LINES(2, module->ToString());

  TF_ASSIGN_OR_RETURN(bool changed, OptimizePipeline(pipeline_ops[0]));

  if (changed) {
    VLOG(2) << "After PipelineOptimizer:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the Pipeline.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
