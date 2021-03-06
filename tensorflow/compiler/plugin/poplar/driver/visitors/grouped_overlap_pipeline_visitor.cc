/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/grouped_overlap_pipeline_visitor.h"

#include <stddef.h>
#include <string.h>

#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <poplar/Engine.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/exceptions.hpp>
#include <poputil/Util.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/deferred_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_stage_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_visitor_utils.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/initialize.h"

namespace xla {
namespace poplarplugin {

namespace util = ::xla::poplarplugin::pipelinevisitorutils;

Status GroupedOverlapPipelineVisitor::VerifyPipelineArguments(
    int64 iterations) const {
  const int64 overlap_length =
      util::ScheduleOffsets(schedule_, stage_ipu_mapping_).size();

  if (iterations < (overlap_length + 2)) {
    return FailedPrecondition(
        "The pipeline depth of the pipeline must be at least %d, but it is %d.",
        overlap_length + 2, iterations);
  }

  return Status::OK();
}

PipelineVisitor::RepeatBlock
GroupedOverlapPipelineVisitor::GetPipelineRampUpSequence(
    const poplar::DebugNameAndId& debug_name_and_id) const {
  std::vector<int> offsets =
      util::ScheduleOffsets(schedule_, stage_ipu_mapping_);

  // Build schedules for the compute and copy programs.
  // Each schedule is 2D, where each column represents a time-slice and each row
  // represents the "mini-batch",
  auto infeed_sequences =
      util::ConstructRampUpScheduleOverlapIO(offsets, infeed_sequences_, 0);
  auto program_sequences =
      util::ConstructRampUpScheduleOverlapIO(offsets, program_sequences_, 1);
  auto fifo_sequences =
      util::ConstructRampUpScheduleOverlapIO(offsets, fifo_sequences_, 1);
  auto recomputation_sequences =
      util::ConstructRecomputationRampUpScheduleOverlapIO(
          offsets, recomputation_sequences_, num_backward_stages_, 1);
  auto copy_sequences =
      util::ConstructScheduleOverlapIO(offsets, copy_sequences_);
  auto inter_ipu_copy_sequences =
      util::ConstructScheduleOverlapIO(offsets, inter_ipu_copy_sequences_);
  auto inter_tileset_copy_in_sequences = util::ConstructScheduleOverlapIO(
      offsets, inter_tileset_copy_in_sequences_);
  auto inter_tileset_copy_out_sequences = util::ConstructScheduleOverlapIO(
      offsets, inter_tileset_copy_out_sequences_);
  auto outfeed_sequences =
      util::ConstructRampUpScheduleOverlapIO(offsets, outfeed_sequences_, 2);

  // Concatenate the programs in the correct order.
  // For overlapped IO, we execute in following order:
  // inter-tileset-copy in, other copies, inter-tileset-copy out, outfeeds,
  // infeeds, programs, fifos, recomputation stages, and then inter-ipu-copies.
  std::vector<std::vector<poplar::program::Sequence>> pipeline_sequences;
  pipeline_sequences.insert(pipeline_sequences.end(),
                            inter_tileset_copy_out_sequences.begin(),
                            inter_tileset_copy_out_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(), copy_sequences.begin(),
                            copy_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(),
                            inter_tileset_copy_in_sequences.begin(),
                            inter_tileset_copy_in_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(), outfeed_sequences.begin(),
                            outfeed_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(), infeed_sequences.begin(),
                            infeed_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(), program_sequences.begin(),
                            program_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(), fifo_sequences.begin(),
                            fifo_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(),
                            recomputation_sequences.begin(),
                            recomputation_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(),
                            inter_ipu_copy_sequences.begin(),
                            inter_ipu_copy_sequences.end());

  // Flatten the schedule to a linear sequence.
  auto repeat_block_sequences = util::FlattenSchedule(pipeline_sequences);

  poplar::program::Sequence repeat_block({}, debug_name_and_id);
  for (const auto& seq : repeat_block_sequences) {
    repeat_block.add(seq);
  }

  return {repeat_block, (offsets.size() / 2) + 1};
}

PipelineVisitor::RepeatBlock
GroupedOverlapPipelineVisitor::GetPipelineRampDownSequence(
    const poplar::DebugNameAndId& debug_name_and_id,
    int additional_iterations) const {
  // Find the set of non-overlapping program offsets.
  std::vector<int> offsets =
      util::ScheduleOffsets(schedule_, stage_ipu_mapping_);

  // Build schedules for the compute and copy programs.
  // Each schedule is 2D, where each column represents a time-slice and each row
  // represents the "mini-batch",
  auto infeed_sequences =
      util::ConstructRampDownScheduleOverlapIO(offsets, infeed_sequences_, 0);
  auto program_sequences =
      util::ConstructRampDownScheduleOverlapIO(offsets, program_sequences_, 1);
  auto fifo_sequences =
      util::ConstructScheduleOverlapIO(offsets, fifo_sequences_);
  auto recomputation_sequences =
      util::ConstructRecomputationRampDownScheduleOverlapIO(
          offsets, recomputation_sequences_, num_backward_stages_, 1);
  auto copy_sequences =
      util::ConstructScheduleOverlapIO(offsets, copy_sequences_);
  auto inter_ipu_copy_sequences =
      util::ConstructScheduleOverlapIO(offsets, inter_ipu_copy_sequences_);
  auto inter_tileset_copy_in_sequences = util::ConstructScheduleOverlapIO(
      offsets, inter_tileset_copy_in_sequences_);
  auto inter_tileset_copy_out_sequences = util::ConstructScheduleOverlapIO(
      offsets, inter_tileset_copy_out_sequences_);
  auto outfeed_sequences =
      util::ConstructRampDownScheduleOverlapIO(offsets, outfeed_sequences_, 2);

  // Concatenate the programs in the correct order.
  // For overlapped IO, we execute in following order:
  // inter-tileset-copy in, other copies, inter-tileset-copy out, outfeeds,
  // infeeds, programs, fifos, recomputation stages, and then inter-ipu-copies.
  std::vector<std::vector<poplar::program::Sequence>> pipeline_sequences;
  pipeline_sequences.insert(pipeline_sequences.end(),
                            inter_tileset_copy_out_sequences.begin(),
                            inter_tileset_copy_out_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(), copy_sequences.begin(),
                            copy_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(),
                            inter_tileset_copy_in_sequences.begin(),
                            inter_tileset_copy_in_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(), outfeed_sequences.begin(),
                            outfeed_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(), infeed_sequences.begin(),
                            infeed_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(), program_sequences.begin(),
                            program_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(), fifo_sequences.begin(),
                            fifo_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(),
                            recomputation_sequences.begin(),
                            recomputation_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(),
                            inter_ipu_copy_sequences.begin(),
                            inter_ipu_copy_sequences.end());

  // Flatten the schedule to a linear sequence.
  auto repeat_block_sequences = util::FlattenSchedule(pipeline_sequences);

  poplar::program::Sequence repeat_block({}, debug_name_and_id);
  for (const auto& seq : repeat_block_sequences) {
    repeat_block.add(seq);
  }

  return {repeat_block, offsets.size() / 2 + 1};
}

PipelineVisitor::RepeatBlock
GroupedOverlapPipelineVisitor::GetPipelineRepeatBlockSequence(
    const poplar::DebugNameAndId& debug_name_and_id, int64 iterations) const {
  // Find the set of non-overlapping program offsets.
  std::vector<int> offsets =
      util::ScheduleOffsets(schedule_, stage_ipu_mapping_);

  const int64 num_repeats = ((iterations / offsets.size()) - 1);
  if (num_repeats < 1) {
    return {poplar::program::Sequence({}, debug_name_and_id), 0};
  }

  // Build schedules for the compute and copy programs.
  // Each schedule is 2D, where each column represents a time-slice and each row
  // represents the "mini-batch",
  auto fifo_sequences =
      util::ConstructSchedule(offsets, fifo_sequences_, false);
  auto infeed_sequences =
      util::ConstructSchedule(offsets, infeed_sequences_, false);
  auto program_sequences =
      util::ConstructSchedule(offsets, program_sequences_, false);
  auto recomputation_sequences =
      util::ConstructSchedule(offsets, recomputation_sequences_, false);
  auto copy_sequences =
      util::ConstructSchedule(offsets, copy_sequences_, false);
  auto inter_ipu_copy_sequences =
      util::ConstructSchedule(offsets, inter_ipu_copy_sequences_, false);
  auto inter_tileset_copy_in_sequences =
      util::ConstructSchedule(offsets, inter_tileset_copy_in_sequences_, false);
  auto inter_tileset_copy_out_sequences = util::ConstructSchedule(
      offsets, inter_tileset_copy_out_sequences_, false);
  auto outfeed_sequences =
      util::ConstructSchedule(offsets, outfeed_sequences_, false);

  // Concatenate the programs in the correct order.
  // For overlapped IO, we execute in following order:
  // inter-tileset-copy in, other copies, inter-tileset-copy out, outfeeds,
  // infeeds, programs, fifos, recomputation stages, and then inter-ipu-copies.
  std::vector<std::vector<poplar::program::Sequence>> pipeline_sequences;
  pipeline_sequences.insert(pipeline_sequences.end(),
                            inter_tileset_copy_out_sequences.begin(),
                            inter_tileset_copy_out_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(), copy_sequences.begin(),
                            copy_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(),
                            inter_tileset_copy_in_sequences.begin(),
                            inter_tileset_copy_in_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(), outfeed_sequences.begin(),
                            outfeed_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(), infeed_sequences.begin(),
                            infeed_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(), program_sequences.begin(),
                            program_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(), fifo_sequences.begin(),
                            fifo_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(),
                            recomputation_sequences.begin(),
                            recomputation_sequences.end());
  pipeline_sequences.insert(pipeline_sequences.end(),
                            inter_ipu_copy_sequences.begin(),
                            inter_ipu_copy_sequences.end());

  for (auto& seq : pipeline_sequences) {
    seq.resize(1);
  }

  // Flatten the schedule to a linear sequence.
  auto repeat_block_sequences = util::FlattenSchedule(pipeline_sequences);

  poplar::program::Sequence repeat_block({}, debug_name_and_id);
  for (const auto& seq : repeat_block_sequences) {
    repeat_block.add(seq);
  }

  return {poplar::program::Repeat(num_repeats * offsets.size() - 2,
                                  repeat_block, {debug_name_and_id}),
          num_repeats * offsets.size() - 2};
}

std::unique_ptr<PipelineVisitor> GroupedOverlapPipelineVisitor::Create(
    const HloInstruction* pipeline, CompilerResources& res,
    const DeferredArgRBVectors& inputs,
    const HloInstructionDescription& description,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return absl::make_unique<GroupedOverlapPipelineVisitor>(
      pipeline, res, inputs, description, debug_name_and_id);
}
}  // namespace poplarplugin
}  // namespace xla
