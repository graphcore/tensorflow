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
        }
        TF_FALLTHROUGH_INTENDED;
      }
      default: {
        return InternalErrorStrCat("Invalid user ", user->ToString(),
                                   " of pipeline stage output ",
                                   output->ToString(), ".");
      }
    }
  }
  return users;
}
}  // namespace

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
      // 2. Pass that buffer into the stage, and dynamic slice update the
      // outputs into the buffer. Return the updated buffer as an output of the
      // stage.
      // 3. Pass the updated buffer as an input to other stages which use it,
      // and use a dynamic slice.

      // Create the buffer.
      const Shape& output_shape = gte->shape();
      CHECK(!output_shape.IsTuple());
      const Shape buffer_shape =
          GetBufferShape(output_shape, batch_serialization_iterations);
      const Shape slice_shape = GetSliceShape(buffer_shape);

      HloInstruction* buffer = pipeline_comp->AddInstruction(
          CreateHloCreateBuffer(buffer_shape, /*is_remote*/ false));

      // Create a dynamic update which sets the output values at the end of each
      // batch serialization loop iteration. Note that the dynamic update is
      // only done on the first dimension.
      std::vector<HloInstruction*> update_start_indices(buffer_shape.rank());
      HloInstruction* counter =
          pipeline_comp->AddInstruction(CreateExecutionCounter());
      HloInstruction* zero = MakeR0ConstantHlo<int32>(pipeline_comp, 0);
      update_start_indices[0] = counter;
      std::fill(std::next(update_start_indices.begin()),
                update_start_indices.end(), zero);

      TF_ASSIGN_OR_RETURN(HloInstruction * reshaped_gte,
                          MakeReshapeHlo(slice_shape, gte));
      HloInstruction* update = pipeline_comp->AddInstruction(
          HloInstruction::CreateDynamicUpdateSlice(
              buffer_shape, buffer, reshaped_gte, update_start_indices));

      // Create a dynamic slice on the updated buffer inside of each pipeline
      // stage which uses the buffer.
      for (HloInstruction* user : users) {
        VLOG(3) << "Adding slice operations for buffer in " << user->ToString();

        const int64 user_idx = ordered_stages.GetIndex(user);

        // Need to create counter and zero per user as they are lowered directly
        // into the stage.
        HloInstruction* slice_counter =
            pipeline_comp->AddInstruction(CreateExecutionCounter());
        HloInstruction* slice_zero = MakeR0ConstantHlo<int32>(pipeline_comp, 0);

        std::vector<HloInstruction*> slice_start_indices(buffer_shape.rank());
        slice_start_indices[0] = slice_counter;
        std::fill(std::next(slice_start_indices.begin()),
                  slice_start_indices.end(), slice_zero);

        HloInstruction* slice =
            pipeline_comp->AddInstruction(HloInstruction::CreateDynamicSlice(
                slice_shape, update, slice_start_indices,
                slice_shape.dimensions()));
        TF_ASSIGN_OR_RETURN(HloInstruction * reshape_slice,
                            MakeReshapeHlo(output_shape, slice));

        // Lower the instructions into the stage and replace the uses with the
        // new (reshaped) slice.
        std::vector<HloInstruction*> lower_to_user_stage = {
            slice_counter, slice_zero, slice, reshape_slice};
        std::map<int64, HloInstruction*> replacements;
        absl::c_for_each(user->OperandIndices(gte), [&](int64 operand_idx) {
          replacements[operand_idx] = reshape_slice;
        });
        TF_ASSIGN_OR_RETURN(user, AddInstructionsToPipelineStage(
                                      user, lower_to_user_stage, replacements));

        ordered_stages.UpdateStage(user_idx, user);
      }

      VLOG(3) << "Adding update operations for buffer in " << stage->ToString();
      std::vector<HloInstruction*> lower_to_current_stage = {
          counter, zero, reshaped_gte, update};

      TF_ASSIGN_OR_RETURN(
          stage, AddInstructionsToPipelineStage(stage, lower_to_current_stage));
      ordered_stages.UpdateStage(i, stage);
    }
  }

  return Status::OK();
}

StatusOr<bool> PipelineBatchSerializationBufferInserter::Run(
    HloModule* module) {
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> pipeline_ops,
                      GetPipelines(module));
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
