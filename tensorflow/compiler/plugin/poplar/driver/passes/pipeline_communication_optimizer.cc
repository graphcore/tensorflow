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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_communication_optimizer.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/fifo.h"
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

StatusOr<bool> PipelineCommunicationOptimizer::OptimizePipeline(
    HloInstruction* pipeline_op) {
  HloComputation* pipeline_comp = pipeline_op->to_apply();

  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  TF_ASSIGN_OR_RETURN(std::vector<PipelinePath> paths,
                      FindPassthroughPipelinePaths(stages));

  if (paths.empty()) {
    return false;
  }

  // Convert the paths into FIFOs.
  for (auto& path : paths) {
    const auto& visited_stages = path.GetVisitedStages();

    VLOG(1) << "Adding a FIFO between " << visited_stages[0] << " and "
            << visited_stages.back();

    // Get the output which will be used for the FIFO - this is the input to the
    // second to last stage which was visited.
    HloInstruction* user = path.GetOldConsumerStage();
    HloInstruction* operand =
        user->mutable_operand(path.GetInputsPath().back());

    CHECK_EQ(operand->opcode(), HloOpcode::kGetTupleElement);
    TF_ASSIGN_OR_RETURN(const uint64 fifo_depth,
                        path.GetFifoDepth(pipeline_op));
    // Create the FIFO.
    HloInstruction* fifo_inst =
        pipeline_comp->AddInstruction(CreateFifo(operand, fifo_depth));
    fifo_inst->SetAndSanitizeName(operand->name() + ".fifo");
    fifo_inst->set_sharding(operand->sharding());

    // Connect it to the right input.
    HloInstruction* stage = path.GetNewConsumerStage();
    TF_RETURN_IF_ERROR(
        stage->ReplaceOperandWith(path.GetInputsPath()[0], fifo_inst));
  }

  return true;
}

StatusOr<bool> PipelineCommunicationOptimizer::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> pipeline_ops,
                      GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to optimize.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  HloInstruction* pipeline_op = pipeline_ops[0];
  VLOG(2) << "Before PipelineCommunicationOptimizer:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  TF_ASSIGN_OR_RETURN(const auto schedule, GetPipelineSchedule(pipeline_op));
  switch (schedule) {
    case PoplarBackendConfig::CallConfig::PipelineConfig::Grouped:
    case PoplarBackendConfig::CallConfig::PipelineConfig::Sequential: {
      TF_ASSIGN_OR_RETURN(changed, OptimizePipeline(pipeline_op));
      break;
    }
    case PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved: {
      VLOG(1) << "Interleaved schedule is not supported by the "
                 "PipelineCommunicationOptimizer pass.";
      break;
    }
    default: { return FailedPrecondition("Unknown pipeline schedule."); }
  }

  if (changed) {
    VLOG(2) << "After PipelineCommunicationOptimizer:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the Pipeline.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
