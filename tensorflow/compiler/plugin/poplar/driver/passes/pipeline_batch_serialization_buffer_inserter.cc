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
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_batch_serialization_buffer_inserter.h"

#include <map>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/execution_counter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

namespace {
Shape GetBufferShape(const Shape& shape, int64 size) {
  CHECK(!shape.IsTuple());
  std::vector<int64> dimensions(shape.rank() + 1);
  dimensions[0] = size;
  absl::c_copy(shape.dimensions(), std::next(dimensions.begin()));
  return ShapeUtil::MakeShape(shape.element_type(), dimensions);
}

Shape GetSliceShape(const Shape& shape) {
  CHECK(!shape.IsTuple());
  Shape output_shape = shape;
  output_shape.set_dimensions(0, 1);
  return output_shape;
}

StatusOr<HloInstructionSet> GetOutputUsers(HloInstruction* output) {
  CHECK_EQ(output->opcode(), HloOpcode::kGetTupleElement);
  HloInstructionSet users;

  for (HloInstruction* user : output->users()) {
    auto error_msg =
        absl::StrCat("Invalid user ", user->ToString(),
                     " of pipeline stage output ", output->ToString(), ".");
    switch (user->opcode()) {
      case HloOpcode::kCall: {
        if (IsAnyPipelineStageOp(user)) {
          users.insert(user);
        } else {
          CHECK(IsResourceUpdate(user));
        }
        break;
      }
      case HloOpcode::kCustomCall: {
        if (IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(user)) {
          // We don't need to do anything for gradient accumulator sink.
          break;
        } else {
          return InternalErrorStrCat(error_msg);
        }
      }
      case HloOpcode::kTuple: {
        if (user == output->parent()->root_instruction()) {
          break;
        } else {
          return InternalErrorStrCat(error_msg);
        }
      }
      default: { return InternalErrorStrCat(error_msg); }
    }
  }
  return users;
}

StatusOr<HloInstruction*> CreateBufferStore(
    bool offload_activations, HloComputation* const pipeline_comp,
    HloInstruction* const buffer, HloInstruction* const value,
    std::vector<HloInstruction*>* const instructions_to_lower) {
  const Shape& buffer_shape = buffer->shape();
  const Shape slice_shape = GetSliceShape(buffer_shape);
  TF_ASSIGN_OR_RETURN(HloInstruction * reshaped_value,
                      MakeReshapeHlo(slice_shape, value));

  HloInstruction* counter =
      pipeline_comp->AddInstruction(CreateExecutionCounter());

  HloInstruction* update;
  if (offload_activations) {
    update = pipeline_comp->AddInstruction(
        CreateBufferStoreSlice(buffer, reshaped_value, counter));

    (*instructions_to_lower) = {counter, reshaped_value, update};
  } else {
    // Create a dynamic update which sets the output values at the end of each
    // batch serialization loop iteration. Note that the dynamic update is
    // only done on the first dimension.
    std::vector<HloInstruction*> update_start_indices(buffer_shape.rank());
    HloInstruction* zero = MakeR0ConstantHlo<int32>(pipeline_comp, 0);
    update_start_indices[0] = counter;
    std::fill(std::next(update_start_indices.begin()),
              update_start_indices.end(), zero);
    update =
        pipeline_comp->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            buffer_shape, buffer, reshaped_value, update_start_indices));

    (*instructions_to_lower) = {counter, zero, reshaped_value, update};
  }

  return update;
}

StatusOr<HloInstruction*> CreateBufferLoad(
    bool offload_activations, HloComputation* const pipeline_comp,
    HloInstruction* const buffer, const Shape& output_shape,
    std::vector<HloInstruction*>* const instructions_to_lower) {
  const Shape& buffer_shape = buffer->shape();
  const Shape slice_shape = GetSliceShape(buffer_shape);

  HloInstruction* counter =
      pipeline_comp->AddInstruction(CreateExecutionCounter());

  HloInstruction* slice;
  if (offload_activations) {
    slice = pipeline_comp->AddInstruction(
        CreateBufferLoadSlice(slice_shape, buffer, counter));

    (*instructions_to_lower) = {counter, slice};
  } else {
    // Create a dynamic slice which gets the input slice at the beginning of
    // each batch serialization loop iteration.
    HloInstruction* zero = MakeR0ConstantHlo<int32>(pipeline_comp, 0);

    std::vector<HloInstruction*> start_indices(buffer_shape.rank());
    start_indices[0] = counter;
    std::fill(std::next(start_indices.begin()), start_indices.end(), zero);

    slice = pipeline_comp->AddInstruction(HloInstruction::CreateDynamicSlice(
        slice_shape, buffer, start_indices, slice_shape.dimensions()));

    (*instructions_to_lower) = {counter, zero, slice};
  }

  TF_ASSIGN_OR_RETURN(HloInstruction * reshape_slice,
                      MakeReshapeHlo(output_shape, slice));

  instructions_to_lower->push_back(reshape_slice);

  return reshape_slice;
}

}  // namespace

PipelineBatchSerializationBufferInserter::
    PipelineBatchSerializationBufferInserter(bool remote_memory_supported)
    : remote_memory_supported_(remote_memory_supported) {}

Status PipelineBatchSerializationBufferInserter::InsertIntoPipeline(
    HloInstruction* pipeline_op) {
  const int64 batch_serialization_iterations =
      GetPipelineBatchSerializationIterations(pipeline_op);
  HloComputation* pipeline_comp = pipeline_op->to_apply();
  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  // Make sure that the root of each stage is a tuple.
  TF_RETURN_IF_ERROR(FixRootInstructions(stages));
  OrderedPipelineStages ordered_stages(stages,
                                       /*include_resource_update*/ false);

  bool offload_activations;
  switch (GetPipelineOffloadActivations(pipeline_op)) {
    case THREESTATE_OFF: {
      offload_activations = false;
      break;
    }
    case THREESTATE_ON: {
      if (!remote_memory_supported_) {
        return FailedPrecondition(
            "Activations offloading has been enabled, however the current "
            "configuration of the IPU devices does not support remote memory. "
            "Set the `offload_activations` argument of "
            "`pipelining_ops.pipeline` to `False` to stop seeing this "
            "message.");
      }
      offload_activations = true;
      break;
    }
    case THREESTATE_UNDEFINED: {
      // If the option has not been specified, then offload if there is remote
      // memory support.
      offload_activations = remote_memory_supported_;
      break;
    }
    default: { return FailedPrecondition("Unknown state."); }
  }

  // Go through all the pipeline stages in order, and insert buffers for all the
  // outputs used.
  for (int64 i = 0; i != ordered_stages.GetNumberOfStages(); ++i) {
    HloInstruction* stage = ordered_stages.GetStage(i);

    // Go through all the users which are expected to be GTEs.
    std::vector<HloInstruction*> gtes = stage->users();
    for (HloInstruction* gte : gtes) {
      TF_ASSIGN_OR_RETURN(HloInstructionSet users, GetOutputUsers(gte));

      // Don't need to insert buffers as there are no other stages which use
      // this output.
      if (users.empty()) {
        continue;
      }

      // Insert a buffer for storing outputs of a pipeline stage.
      // 1. Create a buffer which can hold all the outputs N times (where N is
      // the number of times a stage gets executed).
      // 2. Pass that buffer into the stage, and update the outputs into the
      // buffer. Return the updated buffer as an output of the stage.
      // 3. Pass the updated buffer as an input to other stages which use it,
      // and use a slice.

      // Create the buffer.
      const Shape& output_shape = gte->shape();
      CHECK(!output_shape.IsTuple());
      const Shape buffer_shape =
          GetBufferShape(output_shape, batch_serialization_iterations);
      const Shape slice_shape = GetSliceShape(buffer_shape);

      HloInstruction* buffer = pipeline_comp->AddInstruction(
          CreateHloCreateBuffer(buffer_shape, offload_activations));

      // Keep track of which instructions need to be lowered into the current
      // stage.
      std::vector<HloInstruction*> lower_to_current_stage;

      TF_ASSIGN_OR_RETURN(
          HloInstruction * update,
          CreateBufferStore(offload_activations, pipeline_comp, buffer, gte,
                            &lower_to_current_stage));

      // Create a dynamic slice on the updated buffer inside of each pipeline
      // stage which uses the buffer.
      for (HloInstruction* user : users) {
        VLOG(3) << "Adding slice operations for buffer in " << user->ToString();

        const int64 user_idx = ordered_stages.GetIndex(user);
        // Keep track of which instructions need to be lowered into the user
        // stage.
        std::vector<HloInstruction*> lower_to_user_stage;

        TF_ASSIGN_OR_RETURN(
            HloInstruction * slice,
            CreateBufferLoad(offload_activations, pipeline_comp, update,
                             output_shape, &lower_to_user_stage));

        // Replace the uses of the previous output with the new (reshaped)
        // slice.
        std::map<int64, HloInstruction*> replacements;
        absl::c_for_each(user->OperandIndices(gte), [&](int64 operand_idx) {
          replacements[operand_idx] = slice;
        });
        TF_ASSIGN_OR_RETURN(user, AddInstructionsToPipelineStage(
                                      user, lower_to_user_stage, replacements));

        ordered_stages.UpdateStage(user_idx, user);
      }

      VLOG(3) << "Adding update operations for buffer in " << stage->ToString();

      TF_ASSIGN_OR_RETURN(
          stage, AddInstructionsToPipelineStage(stage, lower_to_current_stage));
      ordered_stages.UpdateStage(i, stage);
    }
  }

  return Status::OK();
}

StatusOr<bool> PipelineBatchSerializationBufferInserter::Run(
    HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  if (GetPipelineBatchSerializationIterations(pipeline_ops[0]) < 2) {
    // Do not need to insert buffers.
    return false;
  }

  VLOG(2) << "Before PipelineBatchSerializationBufferInserter.";
  XLA_VLOG_LINES(2, module->ToString());

  TF_RETURN_IF_ERROR(InsertIntoPipeline(pipeline_ops[0]));

  VLOG(2) << "After PipelineBatchSerializationBufferInserter.";
  XLA_VLOG_LINES(2, module->ToString());
  return true;
}

}  // namespace poplarplugin
}  // namespace xla
