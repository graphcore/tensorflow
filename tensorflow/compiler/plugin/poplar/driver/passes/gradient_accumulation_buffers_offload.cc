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

#include "tensorflow/compiler/plugin/poplar/driver/passes/gradient_accumulation_buffers_offload.h"

#include <algorithm>
#include <set>
#include <string>
#include <utility>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {
bool IsGradientAccumulatorCreate(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(inst);
}

bool IsGradientAccumulatorSink(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(inst);
}
}  // namespace

GradientAccumulationBuffersOffload::GradientAccumulationBuffersOffload(
    bool remote_memory_supported, int64_t minimum_remote_tensor_size)
    : remote_memory_supported_(remote_memory_supported),
      minimum_remote_tensor_size_(minimum_remote_tensor_size) {}

StatusOr<bool> GradientAccumulationBuffersOffload::ShouldOffloadInPipeline(
    HloInstruction* const pipeline_op) {
  switch (GetPipelineOffloadGradientAccumulationBuffers(pipeline_op)) {
    case THREESTATE_OFF: {
      return false;
    }
    case THREESTATE_ON: {
      if (!remote_memory_supported_) {
        return FailedPrecondition(
            "Gradient accumulation buffer offloading has been enabled, however "
            "the current configuration of the IPU devices does not support "
            "remote memory. Set the `offload_gradient_accumulation_buffers` "
            "argument of `pipelining_ops.pipeline` to `False` to stop seeing "
            "this message.");
      }
      return true;
    }
    case THREESTATE_UNDEFINED: {
      // Don't try to offload if there is no remote memory.
      if (!remote_memory_supported_) {
        return false;
      }
      // Only offload if batch serialization and Sequential schedules are used.
      const int64_t batch_serialization_iterations =
          GetPipelineBatchSerializationIterations(pipeline_op);
      TF_ASSIGN_OR_RETURN(const auto schedule,
                          GetPipelineSchedule(pipeline_op));
      if (batch_serialization_iterations > 1 &&
          schedule ==
              PoplarBackendConfig::CallConfig::PipelineConfig::Sequential) {
        return true;
      }
      return false;
    }
    default: { return FailedPrecondition("Unknown state."); }
  }
}

StatusOr<bool> GradientAccumulationBuffersOffload::OffloadInPipeline(
    HloInstruction* const pipeline_op) {
  HloComputation* pipeline_comp = pipeline_op->to_apply();
  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));

  // If it's not a training pipeline then there is nothing to do.
  if (!stages.resource_update) {
    return false;
  }

  TF_RETURN_IF_ERROR(FixRootInstructions(stages));
  auto should_offload_buffer = [this](const HloInstruction* inst) {
    return minimum_remote_tensor_size_ <= ShapeUtil::ByteSizeOf(inst->shape());
  };

  // Insert load and store instructions inside of stages. Also keep track
  // of which sink operations are now in remote memory.
  HloInstructionSet buffers_to_offload;
  absl::flat_hash_set<HloInstruction*> offloaded_sinks;
  for (HloInstruction* stage : stages.backward) {
    HloComputation* stage_comp = stage->to_apply();
    HloInstruction* root = stage_comp->root_instruction();

    for (int64_t i = 0; i != stage->operand_count(); ++i) {
      HloInstruction* operand = stage->mutable_operand(i);
      if (!(IsGradientAccumulatorCreate(operand) &&
            should_offload_buffer(operand))) {
        continue;
      }

      HloInstruction* parameter = stage_comp->parameter_instruction(i);
      CHECK_EQ(parameter->user_count(), 1);
      HloInstruction* user = parameter->users()[0];
      // Don't insert load/stores in a stage if the accumulator is just being
      // passed through.
      if (user == root) {
        continue;
      }
      VLOG(2) << "Offloading the gradient accumulation buffer input " << i
              << " of pipeline stage " << stage->ToString();
      CHECK(IsSerializedGradientAccumulation(user));
      CHECK_EQ(user->user_count(), 1);
      CHECK_EQ(user->users()[0], root);

      // Add the load inside of the stage and use that for the accumulator.
      HloInstruction* loaded_parameter =
          stage_comp->AddInstruction(CreateHloRemoteParameterLoad({parameter}));
      CHECK_EQ(user->operand_index(parameter), 0);
      TF_RETURN_IF_ERROR(user->ReplaceOperandWith(0, loaded_parameter));

      // Store the result of the accumulation.
      HloInstruction* stored_parameter = stage_comp->AddInstruction(
          CreateHloRemoteParameterStore({parameter, user}));

      // Output the updated remote buffer.
      int64_t output_idx = root->operand_index(user);
      TF_RETURN_IF_ERROR(
          root->ReplaceOperandWith(output_idx, stored_parameter));

      // Find the sink for this accumulator.
      TF_ASSIGN_OR_RETURN(HloInstruction * output_gte,
                          GetUniqueGTEUser(stage, output_idx));
      CHECK_EQ(output_gte->user_count(), 1);
      HloInstruction* sink = output_gte->users()[0];
      CHECK(IsGradientAccumulatorSink(sink));

      // Mark this accumulator for offloading.
      buffers_to_offload.insert(operand);
      offloaded_sinks.insert(sink);
    }
  }

  for (HloInstruction* buffer : buffers_to_offload) {
    VLOG(2) << "Replacing buffer " << buffer->ToString()
            << " with a remote memory variant.";
    HloInstruction* new_buffer = pipeline_comp->AddInstruction(
        absl::make_unique<HloGradientAccumulatorCreate>(
            buffer->shape(), buffer->operands(), /*is_remote=*/true));
    buffer->SetupDerivedInstruction(new_buffer);
    TF_RETURN_IF_ERROR(buffer->ReplaceAllUsesWith(new_buffer));
    TF_RETURN_IF_ERROR(pipeline_comp->ForceRemoveInstruction(buffer));
  }

  // Add loads for the buffers inside of the resource update.
  HloInstruction* resource_update = *stages.resource_update;
  HloComputation* resource_update_comp = resource_update->to_apply();

  for (int64_t i = 0; i != resource_update->operand_count(); ++i) {
    HloInstruction* operand = resource_update->mutable_operand(i);
    // The operand is not an offloaded sink, therefore doesn't require a load.
    if (!offloaded_sinks.contains(operand)) {
      continue;
    }
    HloInstruction* parameter = resource_update_comp->parameter_instruction(i);
    HloInstruction* loaded_parameter = resource_update_comp->AddInstruction(
        CreateHloRemoteParameterLoad({parameter}));
    TF_RETURN_IF_ERROR(parameter->ReplaceAllUsesWith(loaded_parameter));
  }

  return buffers_to_offload.size();
}

StatusOr<bool> GradientAccumulationBuffersOffload::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to fix.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);

  TF_ASSIGN_OR_RETURN(bool offload, ShouldOffloadInPipeline(pipeline_ops[0]));
  if (!offload) {
    return false;
  }

  VLOG(2) << "Before GradientAccumulationBuffersOffload.";
  XLA_VLOG_LINES(2, module->ToString());

  TF_ASSIGN_OR_RETURN(bool changed, OffloadInPipeline(pipeline_ops[0]));

  VLOG(2) << "After GradientAccumulationBuffersOffload.";
  XLA_VLOG_LINES(2, module->ToString());
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
