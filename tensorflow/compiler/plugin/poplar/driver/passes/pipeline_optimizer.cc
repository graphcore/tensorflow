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
std::vector<HloInstruction*> OrderStagesLastToFirst(PipelineStages& stages) {
  std::vector<HloInstruction*> ordered_stages(stages.forward.size() +
                                              stages.backward.size());
  absl::c_copy(stages.backward, ordered_stages.begin());
  std::copy(stages.forward.rbegin(), stages.forward.rend(),
            std::next(ordered_stages.begin(), stages.backward.size()));
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
}  // namespace

StatusOr<bool> PipelineOptimizer::OptimizePipeline(
    HloInstruction* pipeline_op) {
  bool changed = false;
  HloComputation* pipeline_comp = pipeline_op->to_apply();

  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  // For each stage, starting from the last stage to last.
  std::vector<HloInstruction*> ordered_stages = OrderStagesLastToFirst(stages);
  for (HloInstruction* stage : ordered_stages) {
    VLOG(2) << "Optimizing stage: " << stage->ToString();
    // Find any duplicate outputs.
    TF_ASSIGN_OR_RETURN(auto duplicate_outputs,
                        GetDuplicatePipelineStageOutputs(stage));
    if (duplicate_outputs.size()) {
      VLOG(3) << "Replacing duplicate outputs.";
      TF_RETURN_IF_ERROR(ReplaceOutputUses(stage, duplicate_outputs));
    }

    // Find any unused outputs.
    TF_ASSIGN_OR_RETURN(std::set<int64> unused_outputs,
                        GetUnusedPipelineStageOutputIndices(stage));
    if (unused_outputs.size()) {
      VLOG(3) << "Removing unused outputs.";
      TF_RETURN_IF_ERROR(RemoveOutputsFromStage(stage, unused_outputs));
      TF_RETURN_IF_ERROR(HloDCE::RunOnComputation(stage->to_apply()).status());
    }

    // Find any duplicate inputs and change the parameters such that they become
    // unused.
    TF_ASSIGN_OR_RETURN(auto duplicate_inputs,
                        GetDuplicatePipelineStageInputs(stage));
    if (duplicate_inputs.size()) {
      VLOG(3) << "Replacing duplicate inputs.";
      TF_RETURN_IF_ERROR(ReplaceDuplicateInputs(stage, duplicate_inputs));
    }

    // Find any unused inputs and remove them.
    TF_ASSIGN_OR_RETURN(std::set<int64> unused_parameters,
                        GetUnusedParametersInPipelineStage(stage));
    if (unused_parameters.size()) {
      VLOG(3) << "Removing unused inputs.";
      TF_ASSIGN_OR_RETURN(stage,
                          RemoveParametersFromStage(stage, unused_parameters));
      TF_RETURN_IF_ERROR(HloDCE::RunOnComputation(stage->to_apply()).status());
    }

    changed |= (duplicate_outputs.size() || unused_outputs.size() ||
                duplicate_inputs.size() || unused_parameters.size());
  }
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
