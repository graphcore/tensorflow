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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_fifo_inserter.h"
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
  TF_ASSIGN_OR_RETURN(const auto schedule, GetPipelineSchedule(pipeline_op));
  TF_ASSIGN_OR_RETURN(std::vector<PipelinePath> paths,
                      FindPassthroughPipelinePaths(stages, schedule));

  if (paths.empty()) {
    return false;
  }
  // Get whether any Fifos which will be inserted should be offloaded.
  TF_ASSIGN_OR_RETURN(const bool offload_fifos,
                      PipelineFIFOInserter::OffloadFifos(
                          pipeline_op, remote_memory_supported_));

  // Convert the paths into FIFOs.
  for (auto& path : paths) {
    if (schedule ==
            PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved &&
        path.GetType() != PipelinePath::Type::kForwardToBackward) {
      continue;
    }

    const auto& visited_stages = path.GetVisitedStages();
    HloInstruction* stage = path.GetNewConsumerStage();

    VLOG(3) << "Inserting a connection between " << visited_stages[0] << " and "
            << visited_stages.back();

    // Get the output which will be used for the FIFO - this is the input to the
    // second to last stage which was visited.
    HloInstruction* user = path.GetOldConsumerStage();
    HloInstruction* operand =
        user->mutable_operand(path.GetInputsPath().back());

    CHECK_EQ(operand->opcode(), HloOpcode::kGetTupleElement);
    HloInstruction* stage_input;
    if (schedule ==
        PoplarBackendConfig::CallConfig::PipelineConfig::Sequential) {
      stage_input = operand;
    } else {
      TF_ASSIGN_OR_RETURN(const uint64 fifo_depth, path.GetFifoDepth());
      // Create the FIFO.
      stage_input = pipeline_comp->AddInstruction(
          CreateFifo(operand, fifo_depth, offload_fifos));
      stage_input->SetAndSanitizeName(operand->name() + ".fifo");
    }

    // Connect it to the right input.
    TF_RETURN_IF_ERROR(
        stage->ReplaceOperandWith(path.GetInputsPath()[0], stage_input));
  }

  return true;
}

PipelineCommunicationOptimizer::PipelineCommunicationOptimizer(
    bool remote_memory_supported)
    : remote_memory_supported_(remote_memory_supported) {}

StatusOr<bool> PipelineCommunicationOptimizer::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
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
    case PoplarBackendConfig::CallConfig::PipelineConfig::Sequential:
    case PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved: {
      TF_ASSIGN_OR_RETURN(changed, OptimizePipeline(pipeline_op));
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
