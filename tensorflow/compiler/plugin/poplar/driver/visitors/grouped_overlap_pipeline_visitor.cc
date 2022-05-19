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
#include <popops/ElementWise.hpp>
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

static Status VerifyPipelineArgumentsFixed(int64_t iterations,
                                           int64_t overlap_length) {
  if (iterations < (overlap_length + 2)) {
    return FailedPrecondition(
        "The number of iterations of the pipeline must be at least %d, "
        "but it is %d. This number might be called "
        "`gradient_accumulation_count`, "
        "`gradient_accumulation_steps_per_replica` or `steps_per_execution` "
        "depending on the API used.",
        overlap_length + 2, iterations);
  }

  return Status::OK();
}

static StatusOr<DriverProgramSequence> VerifyPipelineArgumentsRuntime(
    const HloInstruction* accumulation_count, int64_t overlap_length,
    DriverTensor accumulation_count_tensor, DriverGraph& graph,
    const poplar::DebugContext& debug_context) {
  DriverProgramSequence prog(graph, debug_context);
  auto cond =
      popops::map(graph, popops::expr::_1 < (overlap_length + 2),
                  {std::move(accumulation_count_tensor)}, prog, debug_context);
  std::string message = absl::StrCat(
      "Grouped overlap pipeline depth is invalid. Check that pipeline depth"
      " is >= the overlap length (",
      (overlap_length + 2),
      ").\n"
      "This number might be called `gradient_accumulation_count`, "
      "`gradient_accumulation_steps_per_replica` or `steps_per_execution` "
      "depending on the API used.");
  prog.add(poplar::program::AbortOnCondition(cond, message, debug_context));
  return prog;
}

StatusOr<DriverProgramSequence>
GroupedOverlapPipelineVisitor::VerifyPipelineArguments(
    const HloInstruction* accumulation_count,
    DriverTensor accumulation_count_tensor, DriverGraph& graph) const {
  const auto iterations = GetAccumulationConstantsValue(accumulation_count);
  const int64_t overlap_length =
      pipeline_scheduler_util_->ScheduleOffsets(stage_ipu_mapping_).size();
  if (iterations) {
    TF_RETURN_IF_ERROR(
        VerifyPipelineArgumentsFixed(*iterations, overlap_length));
    return DriverProgramSequence(graph, dnai_);
  }
  return VerifyPipelineArgumentsRuntime(
      accumulation_count, overlap_length, std::move(accumulation_count_tensor),
      graph, {dnai_, "VerifyPipelineConditions"});
}

PipelineVisitor::IterationsType
GroupedOverlapPipelineVisitor::RampDownAdditionalIterations(
    PipelineVisitor::IterationsType iterations, const size_t overlap_length,
    DriverProgramSequence& program) const {
  return absl::visit(
      make_visitor<PipelineVisitor::IterationsType>(
          [&](int64_t& i) {
            return PipelineVisitor::IterationsType(i % overlap_length);
          },
          [&](PipelineVisitor::CountAndGraph& i) {
            return PipelineVisitor::IterationsType(
                PipelineVisitor::CountAndGraph(
                    i.graph,
                    DriverTensor(
                        popops::map(i.graph, popops::expr::_1 % overlap_length,
                                    {i.count}, program),
                        i.graph)));
          }),
      iterations);
}

PipelineVisitor::RepeatBlock
GroupedOverlapPipelineVisitor::GetPipelineRampUpSequence(
    DriverGraph& graph, const poplar::DebugNameAndId& debug_name_and_id) const {
  std::vector<int> offsets =
      pipeline_scheduler_util_->ScheduleOffsets(stage_ipu_mapping_);

  // Build schedules for the compute and copy programs.
  // Each schedule is 2D, where each column represents a time-slice and each row
  // represents the "mini-batch",
  auto infeed_sequences = util::ConstructRampUpScheduleOverlapIO(
      offsets, infeed_sequences_, 0, {graph});
  auto program_sequences = util::ConstructRampUpScheduleOverlapIO(
      offsets, program_sequences_, 1, {graph});
  auto fifo_sequences = util::ConstructRampUpScheduleOverlapIO(
      offsets, fifo_sequences_, 1, {graph});
  auto recomputation_sequences =
      util::ConstructRecomputationRampUpScheduleOverlapIO(
          offsets, recomputation_sequences_, num_backward_stages_, 1, {graph});
  auto copy_sequences =
      util::ConstructScheduleOverlapIO(offsets, copy_sequences_, {graph});
  auto inter_ipu_copy_sequences = util::ConstructScheduleOverlapIO(
      offsets, inter_ipu_copy_sequences_, {graph});
  auto inter_tileset_copy_in_sequences = util::ConstructScheduleOverlapIO(
      offsets, inter_tileset_copy_in_sequences_, {graph});
  auto inter_tileset_copy_out_sequences = util::ConstructScheduleOverlapIO(
      offsets, inter_tileset_copy_out_sequences_, {graph});
  auto outfeed_sequences = util::ConstructRampUpScheduleOverlapIO(
      offsets, outfeed_sequences_, 2, {graph});

  // Concatenate the programs in the correct order.
  // For overlapped IO, we execute in following order:
  // inter-tileset-copy in, other copies, inter-tileset-copy out, outfeeds,
  // infeeds, programs, fifos, recomputation stages, and then inter-ipu-copies.
  std::vector<std::vector<DriverProgramSequence>> pipeline_sequences;
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

  auto repeat_block = util::DefaultScheduler().CreateRepeatBlock(
      graph, pipeline_sequences, debug_name_and_id, offsets.size());

  return {std::move(repeat_block), (offsets.size() / 2) + 1};
}

DriverProgramSequence
GroupedOverlapPipelineVisitor::GetPipelineRampDownSequence(
    DriverGraph& graph, const poplar::DebugNameAndId& debug_name_and_id,
    const IterationsType& additional_iterations) const {
  // Find the set of non-overlapping program offsets.
  std::vector<int> offsets =
      pipeline_scheduler_util_->ScheduleOffsets(stage_ipu_mapping_);

  // Build schedules for the compute and copy programs.
  // Each schedule is 2D, where each column represents a time-slice and each row
  // represents the "mini-batch",
  auto infeed_sequences = util::ConstructRampDownScheduleOverlapIO(
      offsets, infeed_sequences_, 0, {graph});
  auto program_sequences = util::ConstructRampDownScheduleOverlapIO(
      offsets, program_sequences_, 1, {graph});
  auto fifo_sequences =
      util::ConstructScheduleOverlapIO(offsets, fifo_sequences_, {graph});
  auto recomputation_sequences =
      util::ConstructRecomputationRampDownScheduleOverlapIO(
          offsets, recomputation_sequences_, num_backward_stages_, 1, {graph});
  auto copy_sequences =
      util::ConstructScheduleOverlapIO(offsets, copy_sequences_, {graph});
  auto inter_ipu_copy_sequences = util::ConstructScheduleOverlapIO(
      offsets, inter_ipu_copy_sequences_, {graph});
  auto inter_tileset_copy_in_sequences = util::ConstructScheduleOverlapIO(
      offsets, inter_tileset_copy_in_sequences_, {graph});
  auto inter_tileset_copy_out_sequences = util::ConstructScheduleOverlapIO(
      offsets, inter_tileset_copy_out_sequences_, {graph});
  auto outfeed_sequences = util::ConstructRampDownScheduleOverlapIO(
      offsets, outfeed_sequences_, 2, {graph});

  // Concatenate the programs in the correct order.
  // For overlapped IO, we execute in following order:
  // inter-tileset-copy in, other copies, inter-tileset-copy out, outfeeds,
  // infeeds, programs, fifos, recomputation stages, and then inter-ipu-copies.
  std::vector<std::vector<DriverProgramSequence>> pipeline_sequences;
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

  auto repeat_block = util::DefaultScheduler().CreateRepeatBlock(
      graph, pipeline_sequences, debug_name_and_id, offsets.size());

  return repeat_block;
}

DriverProgramSequence
GroupedOverlapPipelineVisitor::GetPipelineRepeatBlockSequence(
    DriverGraph& graph, const poplar::DebugNameAndId& debug_name_and_id,
    const IterationsType& iterations) const {
  // Find the set of non-overlapping program offsets.
  std::vector<int> offsets =
      pipeline_scheduler_util_->ScheduleOffsets(stage_ipu_mapping_);

  // Build schedules for the compute and copy programs.
  // Each schedule is 2D, where each column represents a time-slice and each row
  // represents the "mini-batch",
  DCHECK(absl::holds_alternative<util::GroupedScheduler>(
      pipeline_scheduler_util_->type));
  auto fifo_sequences =
      pipeline_scheduler_util_->ConstructSchedule(offsets, fifo_sequences_);
  auto infeed_sequences =
      pipeline_scheduler_util_->ConstructSchedule(offsets, infeed_sequences_);
  auto program_sequences =
      pipeline_scheduler_util_->ConstructSchedule(offsets, program_sequences_);
  auto recomputation_sequences = pipeline_scheduler_util_->ConstructSchedule(
      offsets, recomputation_sequences_);
  auto copy_sequences =
      pipeline_scheduler_util_->ConstructSchedule(offsets, copy_sequences_);
  auto inter_ipu_copy_sequences = pipeline_scheduler_util_->ConstructSchedule(
      offsets, inter_ipu_copy_sequences_);
  auto inter_tileset_copy_in_sequences =
      pipeline_scheduler_util_->ConstructSchedule(
          offsets, inter_tileset_copy_in_sequences_);
  auto inter_tileset_copy_out_sequences =
      pipeline_scheduler_util_->ConstructSchedule(
          offsets, inter_tileset_copy_out_sequences_);
  auto outfeed_sequences =
      pipeline_scheduler_util_->ConstructSchedule(offsets, outfeed_sequences_);

  // Concatenate the programs in the correct order.
  // For overlapped IO, we execute in following order:
  // inter-tileset-copy in, other copies, inter-tileset-copy out, outfeeds,
  // infeeds, programs, fifos, recomputation stages, and then inter-ipu-copies.
  std::vector<std::vector<DriverProgramSequence>> pipeline_sequences;
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
    seq.resize(1, {graph});
  }

  auto repeat_block = util::DefaultScheduler().CreateRepeatBlock(
      graph, pipeline_sequences, debug_name_and_id, offsets.size());

  return absl::visit(
      make_visitor<DriverProgramSequence>(
          [&](const int64_t i) {
            const int64_t num_repeats = ((i / offsets.size()) - 1);
            if (num_repeats < 1) {
              return DriverProgramSequence(graph, debug_name_and_id);
            }

            return DriverProgramSequence(
                {DriverProgramRepeat(num_repeats * offsets.size() - 2,
                                     repeat_block, {debug_name_and_id})},
                graph, {debug_name_and_id});
          },
          [&](const PipelineVisitor::CountAndGraph i) {
            DriverProgramSequence result(graph, {debug_name_and_id});
            auto expr =
                (((popops::expr::_1 / offsets.size()) - 1) * offsets.size()) -
                1;
            auto repeat_counter =
                popops::map(i.graph, std::move(expr), {i.count}, result,
                            {debug_name_and_id});
            result.add(popops::countedForLoop(
                i.graph, 0, i.count, 1, repeat_block, {debug_name_and_id}));
            return result;
          }),
      iterations);
}

std::unique_ptr<PipelineVisitor> GroupedOverlapPipelineVisitor::Create(
    DriverGraph& graph, const HloInstruction* pipeline, CompilerResources& res,
    const DeferredArgRBVectors& inputs,
    const HloPoplarInplaceDescription& description,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return absl::make_unique<GroupedOverlapPipelineVisitor>(
      graph, pipeline, res, inputs, description, debug_name_and_id);
}
}  // namespace poplarplugin
}  // namespace xla
