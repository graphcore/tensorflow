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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_fifo_inserter.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/fifo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {

StatusOr<bool> PipelineFIFOInserter::InsertInPipeline(
    HloInstruction* pipeline_op) {
  bool changed = false;
  HloComputation* pipeline_comp = pipeline_op->to_apply();
  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  // Make sure that the root of each stage is a tuple.
  TF_RETURN_IF_ERROR(FixRootInstructions(stages));
  TF_ASSIGN_OR_RETURN(auto analysis,
                      PipelineDataflowAnalysis::GetAnalysis(stages, true));

  const int64 last_stage_id = stages.forward.size() - 1;
  TF_ASSIGN_OR_RETURN(const int fifo_depth_multiplier,
                      GetFifoDepthMultiplier(pipeline_op));

  for (HloInstruction* stage : stages.backward) {
    TF_ASSIGN_OR_RETURN(StageID stage_id, analysis->GetStageID(stage));
    TF_ASSIGN_OR_RETURN(StageID previous_stage_id,
                        analysis->GetPreviousStageID(stage));
    HloInstructionSet fwd_stage_inputs;
    for (HloInstruction* operand : stage->unique_operands()) {
      switch (operand->opcode()) {
        case HloOpcode::kGetTupleElement: {
          const HloInstruction* source = operand->operand(0);
          TF_ASSIGN_OR_RETURN(StageID source_stage_id,
                              analysis->GetStageID(source));
          if (source_stage_id.id == previous_stage_id.id) {
            // We do not need fifos between consecutive stages (including the
            // last forward stage and first backward).
          } else {
            if (stage_id.id != source_stage_id.id) {
              return FailedPrecondition(
                  "Invalid dataflow in pipeline, trying to access outputs of "
                  "%s in %s.",
                  source_stage_id.ToString().c_str(),
                  stage_id.ToString().c_str());
            }
            fwd_stage_inputs.insert(operand);
          }
          break;
        }
        case HloOpcode::kParameter: {
          // We don't need to do anything for parameters.
          break;
        }
        case HloOpcode::kCustomCall: {
          if (IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(
                  operand)) {
            // We don't need to do anything for gradient accumulator creators.
            break;
          }
          // Fall through.
        }
        default: {
          return InternalErrorStrCat("Invalid input ", operand->ToString(),
                                     " to pipeline stage ", stage_id.ToString(),
                                     ".");
        }
      }
    }

    for (HloInstruction* fwd_stage_input : fwd_stage_inputs) {
      // Insert the FIFO between forward and backward stage.
      VLOG(3) << "Inserting FIFO for stage " << stage_id.id;
      HloInstruction* fifo_inst = pipeline_comp->AddInstruction(
          CreateFifo(fwd_stage_input,
                     fifo_depth_multiplier * (last_stage_id - stage_id.id)));
      TF_RETURN_IF_ERROR(fwd_stage_input->ReplaceUseWith(stage, fifo_inst));
      changed = true;
    }
  }
  return changed;
}

StatusOr<bool> PipelineFIFOInserter::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> pipeline_ops,
                      GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to fix.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  VLOG(2) << "Before PipelineFIFOInserter:";
  XLA_VLOG_LINES(2, module->ToString(HloPrintOptions::ShortParsable()));

  TF_ASSIGN_OR_RETURN(bool changed, InsertInPipeline(pipeline_ops[0]));

  if (changed) {
    VLOG(2) << "After PipelineFIFOInserter:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the Pipeline.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
