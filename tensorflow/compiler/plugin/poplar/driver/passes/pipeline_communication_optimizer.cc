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
namespace {
// Helper class for storing paths by the optimizer and getting the correct FIFO
// depth.
class Path {
 public:
  Path(uint64 stage_idx, uint64 input_idx, uint64 output_idx)
      : visited_stages_({stage_idx}),
        inputs_path_({input_idx}),
        outputs_path_({output_idx}) {}

  bool ValidatePathAndCalculateFIFODepth(const PipelineStages& stages) {
    validated_ = true;
    // Only add a path if we visited more than two stages otherwise we
    // would be inserting FIFOs of size zero between consecutive stages.
    if (visited_stages_.size() < 3) {
      return false;
    }
    // We can only have a path between stages that are either:
    // (1) all in forward stages,
    // (2) all in backward stages,
    // (3) between a forward and a backward stage with the same id.
    const uint64 start_stage_idx = visited_stages_[0];
    const uint64 end_stage_idx = visited_stages_.back();

    const uint64 num_backward_stages = stages.backward.size();
    // It is worth remembering that the path can only be a chain between
    // consecutive stages.
    const bool start_is_backward_stage = start_stage_idx < num_backward_stages;
    const bool end_is_backward_stage = end_stage_idx < num_backward_stages;

    if (start_is_backward_stage == end_is_backward_stage) {
      // Both start and end are of same type - handle cases (1) and (2).
      fifo_depth_ = end_stage_idx - start_stage_idx - 1;
      fifo_between_fwd_and_bwd_ = false;
      return true;
    } else if (start_is_backward_stage && !end_is_backward_stage) {
      // Handle case (3).
      fifo_depth_ = end_stage_idx - num_backward_stages;
      fifo_between_fwd_and_bwd_ = true;
      return start_stage_idx == (2 * num_backward_stages - end_stage_idx - 1);
    } else {
      return false;
    }
  }

  std::vector<uint64>& GetVisitedStages() { return visited_stages_; }

  std::vector<uint64>& GetInputsPath() { return inputs_path_; }

  std::vector<uint64>& GetOutputsPath() { return outputs_path_; }

  StatusOr<int64> GetFifoDepth(const HloInstruction* pipeline_op) {
    if (validated_) {
      TF_ASSIGN_OR_RETURN(const auto schedule,
                          GetPipelineSchedule(pipeline_op));
      // We need to take schedule into account.
      uint64 multiplier = 1;
      switch (schedule) {
        case PoplarBackendConfig::CallConfig::PipelineConfig::Grouped:
          multiplier = fifo_between_fwd_and_bwd_ ? 2 : 1;
          break;
        case PoplarBackendConfig::CallConfig::PipelineConfig::Sequential:
          multiplier = 0;
          break;
        default:
          return FailedPrecondition("Unsupported pipeline schedule.");
      }

      return fifo_depth_ * multiplier;
    } else {
      return InternalErrorStrCat("Expected the path to have been validated.");
    }
  }

 private:
  // The fields below are populated by the validate function.
  bool validated_ = false;
  int64 fifo_depth_ = -1;
  bool fifo_between_fwd_and_bwd_ = false;

  std::vector<uint64> visited_stages_;
  std::vector<uint64> inputs_path_;
  std::vector<uint64> outputs_path_;
};
}  // namespace

StatusOr<bool> PipelineCommunicationOptimizer::OptimizePipeline(
    HloInstruction* pipeline_op) {
  HloComputation* pipeline_comp = pipeline_op->to_apply();

  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  const uint64 num_stages = stages.forward.size() + stages.backward.size();
  // Set up stages in last to to first order.
  std::vector<HloInstruction*> stages_last_to_first(num_stages);
  absl::c_copy(stages.backward, stages_last_to_first.begin());
  std::copy(stages.forward.rbegin(), stages.forward.rend(),
            std::next(stages_last_to_first.begin(), stages.backward.size()));

  // For each stage we store a map which represents tensors being passed-through
  // the stage. For example given a stage:
  // {
  //   p0 = parameter(0)
  //   p1 = parameter(1)
  //   ...
  //   ROOT t = tuple(a, b, p0, p1)
  // }
  // the inputs p0 and p1 are passed-through the stage, with p0 having an output
  // index 2 (its location in the root tuple) and input index 0 (its parameter
  // number). Similarly p1 has output index 3 and input index 1. Other outputs
  // (a and b) are not inputs to the stage therefore they are not in the map.
  std::vector<absl::flat_hash_map<uint64, uint64>>
      intra_stage_output_to_input_map(num_stages);

  // For each stage we store a map which represents connections between stage
  // inputs and the outputs of the previous stage. For example given stages:
  // s0 = (fp32[], fp32[], fp32[]) pipeline_stage(x, y, z) stage_id = 0
  // s0.0 = gte(s0), index=0
  // s0.1 = gte(s0), index=1
  // s0.2 = gte(s0), index=2
  // s1 = (fp32[]) pipeline_stage(s0.0, s0.1) stage_id = 1
  // s1.0 = gte(s1), index=0
  // s2 = (fp32[], fp32[], fp32[]) pipeline_stage(s0.2, s1.0) stage_id = 2
  //
  // * s1 at input index 0 uses output of s0 at output index 0
  // * s1 at input index 1 uses output of s0 at output index 1
  // * s2 at input index 1 uses output of s1 at output index 0
  // Note that s2 also uses an output from s0, however the previous stage of s2
  // is s1 so it is not included in the map.
  std::vector<absl::flat_hash_map<uint64, uint64>>
      inter_stage_input_to_previous_output_map(num_stages);

  // Build the intra and inter maps.
  for (uint64 stage_idx = 0; stage_idx != num_stages; ++stage_idx) {
    HloInstruction* stage = stages_last_to_first[stage_idx];
    // We expect all stages to have sharding for this algorithm to work.
    if (!stage->sharding_unique_device()) {
      return InternalErrorStrCat("Expected stage ", stage->ToString(),
                                 " to have sharding information.");
    }
    HloComputation* stage_computation = stage->to_apply();
    // Building the intra map.
    // To build up the intra map, we go through the tuple elements of the root
    // instruction of the pipeline stage computation which is expected to be a
    // tuple.
    HloInstruction* root = stage_computation->root_instruction();
    CHECK_EQ(root->opcode(), HloOpcode::kTuple);
    for (uint64 output_idx = 0; output_idx != root->operand_count();
         ++output_idx) {
      const HloInstruction* operand = root->operand(output_idx);
      if (operand->opcode() == HloOpcode::kParameter) {
        const uint64 input_idx = operand->parameter_number();
        if (intra_stage_output_to_input_map[stage_idx].contains(output_idx)) {
          return InternalErrorStrCat("Expected pipeline stage ",
                                     stage->ToString(),
                                     " to not have duplicate outputs.");
        }
        intra_stage_output_to_input_map[stage_idx][output_idx] = input_idx;
      }
    }
    // Building the inter map.
    // Note that the last stage does not a previous stage so we do not build a
    // map for it.
    if (stage_idx == (num_stages - 1)) {
      continue;
    }
    // Go through all the operands to the stage and if they are GTEs on the
    // previous stage, then add them to the map.
    for (uint64 input_idx = 0; input_idx != stage->operand_count();
         ++input_idx) {
      const HloInstruction* operand = stage->operand(input_idx);
      if (operand->opcode() == HloOpcode::kGetTupleElement &&
          operand->operand(0) == stages_last_to_first[stage_idx + 1]) {
        if (inter_stage_input_to_previous_output_map[stage_idx].contains(
                input_idx)) {
          return InternalErrorStrCat("Expected pipeline stage ",
                                     stage->ToString(),
                                     " to not have duplicate inputs.");
        }
        inter_stage_input_to_previous_output_map[stage_idx][input_idx] =
            operand->tuple_index();
      }
    }
  }

  std::vector<Path> paths;
  // Build the paths.
  // We build the path by traversing a path, note that for each starting point
  // there can only be at most one path.
  for (uint64 stage_idx = 0; stage_idx != num_stages; ++stage_idx) {
    HloInstruction* stage = stages_last_to_first[stage_idx];
    const int64 shard = *stage->sharding_unique_device();
    for (uint64 input_idx = 0; input_idx != stage->operand_count();
         ++input_idx) {
      if (!inter_stage_input_to_previous_output_map[stage_idx].contains(
              input_idx)) {
        // No path to build - skip.
        continue;
      }
      // Create the start of a path.
      Path path{
          stage_idx, input_idx,
          inter_stage_input_to_previous_output_map[stage_idx].at(input_idx)};

      for (uint64 next_stage_idx = stage_idx + 1; next_stage_idx != num_stages;
           ++next_stage_idx) {
        HloInstruction* next_stage = stages_last_to_first[next_stage_idx];
        const int64 next_shard = *next_stage->sharding_unique_device();
        const bool shard_matches = next_shard == shard;
        path.GetVisitedStages().push_back(next_stage_idx);

        if (shard_matches) {
          // Stop if we reached a stage on the same shard.
          // We stop as soon as possible to avoid creating parallel pipelines.
          if (path.ValidatePathAndCalculateFIFODepth(stages)) {
            paths.push_back(path);
          }
          break;
        }

        // Try to extend the path.
        // Use the intra map to see if this output has been threaded through.
        const uint64 output_idx = path.GetOutputsPath().back();
        auto intra_itr =
            intra_stage_output_to_input_map[next_stage_idx].find(output_idx);
        if (intra_itr ==
            intra_stage_output_to_input_map[next_stage_idx].end()) {
          // The output at this index has not been threaded through - therefore
          // there is no path.
          break;
        }
        const uint64 input_idx = intra_itr->second;
        // Now see if the input at that index is the `next` stage by looking at
        // the inter map.
        auto inter_itr =
            inter_stage_input_to_previous_output_map[next_stage_idx].find(
                input_idx);
        if (inter_itr ==
            inter_stage_input_to_previous_output_map[next_stage_idx].end()) {
          // The output at this index has not been threaded through - therefore
          // there is no path.
          break;
        }
        const uint64 next_output_idx = inter_itr->second;
        path.GetInputsPath().push_back(input_idx);
        path.GetOutputsPath().push_back(next_output_idx);
      }
    }
  }

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
    HloInstruction* user =
        stages_last_to_first.at(visited_stages[visited_stages.size() - 2]);
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
    HloInstruction* stage = stages_last_to_first.at(visited_stages[0]);
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
