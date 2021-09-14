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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_stage_merger.h"

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_noop.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {

// Moves all the computation from the last forward stage to the corresponding
// backward stage. This means when batch serialization is done, the forward and
// backward stage instruction of each micro batch can be done as a single step.
StatusOr<bool> MergeLastForwardAndBackwardStage(
    HloInstruction* const pipeline_op) {
  HloComputation* pipeline_comp = pipeline_op->to_apply();
  HloModule* module = pipeline_comp->parent();
  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  if (stages.backward.empty()) {
    return false;
  }

  TF_RETURN_IF_ERROR(FixRootInstructions(stages));
  TF_ASSIGN_OR_RETURN(auto analysis,
                      PipelineDataflowAnalysis::GetAnalysis(stages, true));

  HloInstruction* last_fwd_stage = stages.forward.back();
  HloSharding last_fwd_stage_sharding = last_fwd_stage->sharding();
  HloInstruction* last_bwd_stage = stages.backward.back();

  HloComputation* fwd_comp = last_fwd_stage->to_apply();
  HloComputation* bwd_comp = last_bwd_stage->to_apply();

  // Inline the bwd computation into the pipeline.
  TF_ASSIGN_OR_RETURN(auto inline_info, CallInliner::Inline(last_bwd_stage));

  // Lower all the instructions into the forward computation.
  std::vector<HloInstruction*> to_lower_forward;
  for (HloInstruction* inst : bwd_comp->MakeInstructionPostOrder()) {
    auto itr = inline_info.find(inst);
    CHECK(itr != inline_info.end());
    TF_ASSIGN_OR_RETURN(bool lower, analysis->HasToBeLowered(itr->second));
    if (lower) {
      to_lower_forward.push_back(itr->second);
    }
  }
  // Make sure all the users end up using the outputs of this stage.
  HloInstruction* root = inline_info.at(bwd_comp->root_instruction());
  to_lower_forward.insert(to_lower_forward.end(), root->users().begin(),
                          root->users().end());

  TF_ASSIGN_OR_RETURN(last_fwd_stage, AddInstructionsToPipelineStage(
                                          last_fwd_stage, to_lower_forward));

  // Convert the forward stage into a backward stage - all the computation has
  // to happen in the backward stage as the inputs might contain gradient
  // accumulation buffers which are only allowed in the backward stages.
  {
    TF_ASSIGN_OR_RETURN(PoplarBackendConfig cfg,
                        last_fwd_stage->backend_config<PoplarBackendConfig>());
    cfg.mutable_call_config()->set_type(
        PoplarBackendConfig::CallConfig::PipelineStageBackward);
    TF_RETURN_IF_ERROR(last_fwd_stage->set_backend_config(cfg));
    last_bwd_stage = last_fwd_stage;
    last_bwd_stage->clear_sharding();
  }

  // To preserve the dataflow analysis, force outputs of the second to last
  // forward stage used by the backward stage through the new forward stage.
  // Note that the communication optimizer might remove these links.
  {
    std::vector<HloInstruction*> stage_inputs;
    std::vector<int64> indices;
    for (int64 op_idx = 0; op_idx != last_bwd_stage->operand_count();
         ++op_idx) {
      HloInstruction* operand = last_bwd_stage->mutable_operand(op_idx);
      if (operand->opcode() == HloOpcode::kGetTupleElement) {
        stage_inputs.push_back(operand);
        indices.push_back(op_idx);
      }
    }

    // Create the pass through computation.
    HloComputation::Builder builder(
        absl::StrCat("pipeline_stage_", stages.forward.size() - 1));
    std::vector<HloInstruction*> comp_parameters(stage_inputs.size());
    for (int64 param_idx = 0; param_idx != stage_inputs.size(); ++param_idx) {
      comp_parameters[param_idx] =
          builder.AddInstruction(HloInstruction::CreateParameter(
              param_idx, stage_inputs[param_idx]->shape(),
              absl::StrCat("Parameter", param_idx)));
    }
    builder.AddInstruction(CreateStatefulNoop());
    builder.AddInstruction(HloInstruction::CreateTuple(comp_parameters));
    fwd_comp = module->AddEmbeddedComputation(builder.Build());

    // Create the pipeline stage.
    const int64 stage_id = stages.forward.size() - 1;
    TF_ASSIGN_OR_RETURN(
        last_fwd_stage,
        CreatePipelineStage(pipeline_comp, stage_inputs, fwd_comp,
                            PoplarBackendConfig::CallConfig::PipelineStage,
                            stage_id, absl::StrCat("PipelineStage", stage_id)));
    last_fwd_stage->set_sharding(last_fwd_stage_sharding);

    // Rewire the outputs so that the backward stage uses the forward stage.
    for (int64 output_idx = 0; output_idx != indices.size(); ++output_idx) {
      TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                          MakeGetTupleElementHlo(last_fwd_stage, output_idx));
      TF_RETURN_IF_ERROR(
          last_bwd_stage->ReplaceOperandWith(indices[output_idx], gte));
    }
  }
  TF_RETURN_IF_ERROR(
      TupleSimplifier::RunOnComputation(last_bwd_stage->to_apply()).status());

  return true;
}

StatusOr<bool> PipelineStageMerger::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to fix.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  TF_ASSIGN_OR_RETURN(const auto schedule,
                      GetPipelineSchedule(pipeline_ops[0]));
  if (schedule != PoplarBackendConfig::CallConfig::PipelineConfig::Sequential) {
    return false;
  }

  VLOG(2) << "Before PipelineStageMerger:";
  XLA_VLOG_LINES(2, module->ToString(HloPrintOptions::ShortParsable()));

  TF_ASSIGN_OR_RETURN(bool changed,
                      MergeLastForwardAndBackwardStage(pipeline_ops[0]));

  if (changed) {
    VLOG(2) << "After PipelineStageMerger:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the Pipeline.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
