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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_PIPELINE_VISITOR_UTILS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_PIPELINE_VISITOR_UTILS_H_

#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_visitor.h"

#include <stddef.h>
#include <string.h>

#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/types/variant.h"

#include <popops/ElementWise.hpp>
#include <popops/Loop.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/inter_tileset_copy.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/make_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/deferred_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_stage_visitor.h"
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
namespace pipelinevisitorutils {

/**
 * Construct a unary predicate which checks if a given HloInstruction has the
 * same opcode as the one captured in the closure.
 *
 * @param opcode The opcode to capture and compare against.
 *
 * @returns The unary predicate.
 */
inline std::function<bool(const HloInstruction*)> HasHloOpcode(
    HloOpcode opcode) {
  return [opcode](const HloInstruction* inst) -> bool {
    return inst->opcode() == opcode;
  };
}

/**
 * Construct a unary predicate which checks if a given HloInstruction is an
 * HloFifoInstruction.
 *
 * @returns The unary predicate.
 */
inline std::function<bool(const HloInstruction*)> IsFifoInstruction() {
  return IsPoplarInstruction(PoplarOp::Fifo);
}

/**
 * Construct a unary predicate which checks if a given HloInstruction is an
 * HloFifoInstruction.
 *
 * @returns The unary predicate.
 */
inline std::function<bool(const HloInstruction*)>
IsInterTilesetCopyInstruction() {
  return IsPoplarInstruction(PoplarOp::InterTilesetCopy);
}

/**
 * Test whether a HLO instruction is a intertileset copy from the IO tiles.
 *
 * @param inst The instruction to test
 *
 * @returns true if the instruction is a HLO instruction is a intertileset copy
 * from the IO tiles, otherwise returns false.
 */
inline bool IsInterTilesetCopyInInstruction(const HloInstruction* inst) {
  if (!IsInterTilesetCopyInstruction()(inst)) {
    return false;
  }

  const auto* copy_inst = Cast<HloInterTilesetCopy>(inst);
  return !copy_inst->IsCopyToIoTiles();
}

/**
 * Test whether a HLO instruction is a intertileset copy to the IO tiles.
 *
 * @param inst The instruction to test
 *
 * @returns true if the instruction is a HLO instruction is a intertileset copy
 * to the IO tiles, otherwise returns false.
 */
inline bool IsInterTilesetCopyOutInstruction(const HloInstruction* inst) {
  if (!IsInterTilesetCopyInstruction()(inst)) {
    return false;
  }

  const auto* copy_inst = Cast<HloInterTilesetCopy>(inst);
  return copy_inst->IsCopyToIoTiles();
}

/**
 * Construct a unary predicate which checks if a given HloInstruction is an
 * HloInterIpuCopy.
 *
 * @returns The unary predicate.
 */
inline std::function<bool(const HloInstruction*)> IsInterIpuCopyInstruction() {
  return IsPoplarInstruction(PoplarOp::InterIpuCopy);
}

/**
 * Construct a unary predicate which checks if a given HloInstruction is an
 * HloGradientAccumulatorCreate.
 *
 * @returns The unary predicate.
 */
inline std::function<bool(const HloInstruction*)>
IsGradientAccumulatorCreateInstruction() {
  return IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate);
}

/**
 * Construct a unary predicate which checks if a given HloInstruction is an
 * HloGradientAccumulatorSink.
 *
 * @returns The unary predicate.
 */
inline std::function<bool(const HloInstruction*)>
IsGradientAccumulatorSinkInstruction() {
  return IsPoplarInstruction(PoplarOp::GradientAccumulatorSink);
}

/**
 * Construct a unary predicate which checks if a given HloInstruction is an
 * HloCreateBuffer.
 *
 * @returns The unary predicate.
 */
inline std::function<bool(const HloInstruction*)> IsCreateBuffer() {
  return IsPoplarInstruction(PoplarOp::CreateBuffer);
}

/**
 * Construct a unary predicate which checks if a given HloInstruction is an
 * HloExecutionCounter.
 *
 * @returns The unary predicate.
 */
inline std::function<bool(const HloInstruction*)> IsExecutionCounter() {
  return IsPoplarInstruction(PoplarOp::ExecutionCounter);
}

/**
 * Find the indices of all possible non-overlapping circular unions.
 *
 * @param input The sequence of input elements.
 * @param predicate The user defined predicate function which compares members
 *                  of the input sequence for "equality".
 *
 * @returns a list of indices of valid rotations of the input that do not
 *          overlap.
 *
 *
 * Suppose our we have:
 *   ElementType = int,
 *   BinaryPredicateType = bool(int, int),
 *   input = [0, 1, 2, 0, 0, 2, 1, 0],
 *   predicate = [](int a, int b){ return a == b; }
 *
 * The result will be [0, 2].
 * We can see this is the case by drawing the rotated input
 *   rotate(input, 0) = [0, 1, 2, 0, 0, 2, 1, 0]
 *   rotate(input, 2) = [2, 0, 0, 2, 1, 0, 0, 1]
 *
 * It can also be seen that no other rotations would work
 *   rotate(input, 0) = [0, 1, 2, 0, 0, 2, 1, 0] Trivially a member of the set
 *   rotate(input, 1) = [0, 0, 1, 2, 0, 0, 2, 1] Overlaps at position 0
 *   rotate(input, 2) = [1, 0, 0, 1, 2, 0, 0, 2] Add to set
 *   rotate(input, 3) = [2, 1, 0, 0, 1, 2, 0, 0] Overlaps at position 1
 *   rotate(input, 4) = [0, 2, 1, 0, 0, 1, 2, 0] Overlaps at position 0
 *   rotate(input, 5) = [0, 0, 2, 1, 0, 0, 1, 2] Overlaps at position 0
 *   rotate(input, 6) = [2, 0, 0, 2, 1, 0, 0, 1] Overlaps at position 1
 *   rotate(input, 7) = [1, 2, 0, 0, 2, 1, 0, 0] Overlaps at position 3
 */
template <typename ElementType,
          typename BinaryPredicateType = std::equal_to<ElementType>>
std::vector<int> CircularUnion(const std::vector<ElementType>& input,
                               BinaryPredicateType predicate = {}) {
  // The 0th rotation is always a valid result.
  std::vector<int> result = {0};

  // Create a temporary storage area the same size of the input.
  std::vector<ElementType> temp_0(input.size());
  std::vector<ElementType> temp_1(input.size());

  // Invert the user predicate.
  const auto not_predicate = [&predicate](const ElementType& a,
                                          const ElementType& b) -> bool {
    return !predicate(a, b);
  };

  // For each possible valid rotation, check if it is non-overlapping with the
  // input rotations.
  for (size_t i = 1; i < input.size(); ++i) {
    // Take the ith rotated input.
    std::rotate_copy(input.begin(), std::next(input.begin(), i), input.end(),
                     temp_0.begin());

    bool non_overlapping = true;

    // Compare against all accept rotations of the input
    for (size_t k = 0; k < result.size() && non_overlapping; ++k) {
      std::rotate_copy(input.begin(), std::next(input.begin(), result[k]),
                       input.end(), temp_1.begin());

      // Map-reduce where the map is the negation of the user predicate and the
      // reduction is logical and. This means we will accept rotations where the
      // corresponding elements are not equal.
      non_overlapping = std::inner_product(
          temp_1.begin(), temp_1.end(), temp_0.begin(), non_overlapping,
          std::logical_and<bool>{}, not_predicate);
    }

    // If the rotation is non-overlapping with all existing
    if (non_overlapping) {
      // Add this rotation index to the result.
      result.push_back(i);
    }
  }

  return result;
}

/**
 * Find the indices of all possible circular unions, including overlaps.
 *
 * @param input The sequence of input elements.
 *
 * @returns a list of indices of valid rotations of the input that do not
 *          overlap.
 */
template <typename ElementType>
std::vector<int> AllUnion(const std::vector<ElementType>& input) {
  std::vector<int> result(input.size());

  // This is trivially just every offset.
  absl::c_iota(result, 0);

  return result;
}

/**
 * Create indices for the simple sharded case.
 *
 * @param input The sequence of input elements.
 *
 * @returns a "list" of indices {0}.
 */
template <typename ElementType>
std::vector<int> NoUnion(const std::vector<ElementType>& input) {
  std::vector<int> result(1);

  // This is trivially just a single offset.
  absl::c_iota(result, 0);

  return result;
}

template <typename ElementType>
std::vector<int> ScheduleOffsets(
    PoplarBackendConfig::CallConfig::PipelineConfig::Schedule schedule,
    const std::vector<ElementType>& input) {
  switch (schedule) {
    case PoplarBackendConfig::CallConfig::PipelineConfig::Grouped:
      return AllUnion(input);
    case PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved:
      return CircularUnion(input);
    default:
      return NoUnion(input);
  }
}

template <typename ElementType>
std::vector<std::vector<ElementType>> TransposeSchedule(
    const std::vector<std::vector<ElementType>>& input) {
  std::vector<std::vector<ElementType>> result(input[0].size());

  for (size_t i = 0; i < input.size(); ++i) {
    for (size_t k = 0; k < input[i].size(); ++k) {
      result[k].push_back(input[i][k]);
    }
  }

  return result;
}

/**
 * Construct a pipeline schedule given an offset and some schedulable
 * components.
 *
 * @param offsets The offsets of each parallel sequence of inputs.
 * @param input The input sequence to schedule.
 *
 * @returns A 2D array of pipeline schedule where each row represents the
 *          parallel sequence, and each column represents a single timestep
 *          where a single step of the input is scheduled.
 */
template <typename ElementType>
std::vector<std::vector<ElementType>> ConstructScheduleInternal(
    const std::vector<int>& offsets, const std::vector<ElementType>& input) {
  std::vector<std::vector<ElementType>> result(offsets.size(), input);

  for (size_t i = 0; i < offsets.size(); ++i) {
    std::rotate(result[i].begin(),
                std::next(result[i].begin(), result[i].size() - offsets[i]),
                result[i].end());
  }

  return result;
}

template <typename ElementType>
std::vector<std::vector<ElementType>> RotateSchedule(
    const std::vector<std::vector<ElementType>>& input) {
  std::vector<std::vector<ElementType>> result = input;

  for (int i = 0; i < static_cast<int>(result.size()) - 1; ++i) {
    std::rotate(result[i].begin(), std::next(result[i].begin(), i + 1),
                result[i].end());
  }

  return result;
}

template <typename ElementType>
std::vector<std::vector<ElementType>> LeftPadSchedule(
    const std::vector<std::vector<ElementType>>& input,
    ElementType padding = {}) {
  auto result = input;  // TransposeSchedule(input);

  for (auto& timestep : result) {
    timestep.insert(timestep.begin(), padding);
  }

  return result;
}

template <typename ElementType>
std::vector<std::vector<ElementType>> ReverseSchedule(
    std::vector<std::vector<ElementType>> input) {
  for (auto& timestep : input) {
    std::reverse(timestep.begin(), timestep.end());
  }

  return input;
}

template <typename ElementType>
std::vector<std::vector<ElementType>> RightPadSchedule(
    const std::vector<std::vector<ElementType>>& input,
    ElementType padding = {}) {
  return ReverseSchedule(LeftPadSchedule(ReverseSchedule(input), padding));
}

template <typename ElementType>
std::vector<std::vector<ElementType>> SliceSchedule(
    std::vector<std::vector<ElementType>> input, std::size_t size) {
  for (auto& timestep : input) {
    timestep.resize(size);
  }

  return input;
}

template <typename ElementType>
std::vector<std::vector<ElementType>> ConcatSchedule(
    const std::vector<std::vector<ElementType>>& A,
    const std::vector<std::vector<ElementType>>& B) {
  CHECK_EQ(A.size(), B.size());
  std::vector<std::vector<ElementType>> result(A.size());

  std::transform(
      A.begin(), A.end(), B.begin(), result.begin(),
      [](std::vector<ElementType> a,
         const std::vector<ElementType>& b) -> std::vector<ElementType> {
        a.insert(a.end(), b.begin(), b.end());
        return a;
      });

  return result;
}

template <typename ElementType>
static std::vector<std::vector<ElementType>> ForceInterleavedStageOrders(
    std::vector<std::vector<ElementType>> result) {
  result = TransposeSchedule(result);
  result = RotateSchedule(result);
  return TransposeSchedule(result);
}

/**
 * Construct a pipeline schedule given an offset and some schedulable
 * components.
 *
 * @param offsets The offsets of each parallel sequence of inputs.
 * @param input The input sequence to schedule.
 *
 * @returns A 2D array of pipeline schedule where each row represents the
 *          parallel sequence, and each column represents a single timestep
 *          where a single step of the input is scheduled.
 */
template <typename ElementType>
std::vector<std::vector<ElementType>> ConstructSchedule(
    const std::vector<int>& offsets, const std::vector<ElementType>& input) {
  return ConstructScheduleInternal(offsets, input);
}

/**
 * Construct a pipeline schedule given an offset and some schedulable
 * components.
 *
 * @param offsets The offsets of each parallel sequence of inputs.
 * @param input The input sequence to schedule.
 *
 * @returns A 2D array of pipeline schedule where each row represents the
 *          parallel sequence, and each column represents a single timestep
 *          where a single step of the input is scheduled.
 */
template <typename ElementType>
std::vector<std::vector<ElementType>> ConstructScheduleOverlapIO(
    const std::vector<int>& offsets, const std::vector<ElementType>& input) {
  auto result = ForceInterleavedStageOrders(ConstructSchedule(offsets, input));
  auto right_padding = SliceSchedule(
      ForceInterleavedStageOrders(ConstructSchedule(offsets, input)), 2);

  return ConcatSchedule(result, right_padding);
}

/**
 * Construct a "ramp-up" pipeline schedule given an offset and some schedulable
 * components. Additionally, empty stages are inserted into the schedule where a
 * stage cannot be executed.
 *
 * @param offsets The offsets of each parallel sequence of inputs.
 * @param input The input sequence to schedule.
 * @param empty_element The empty element, or identity element, on the
 *                      ElementType. This is what is inserted into the "empty
 *                      stages" of the schedule.
 *
 * @returns A 2D array of pipeline schedule where each row represents the
 *          parallel sequence, and each column represents a single timestep
 *          where a single step of the input is scheduled.
 */
template <typename ElementType>
std::vector<std::vector<ElementType>> ConstructRampUpSchedule(
    const std::vector<int>& offsets, const std::vector<ElementType>& input,
    ElementType empty_element = {}) {
  auto result = ConstructScheduleInternal(offsets, input);

  for (size_t i = 0; i < offsets.size(); ++i) {
    std::fill(result[i].begin(), std::next(result[i].begin(), offsets[i]),
              empty_element);
  }

  return result;
}

/**
 * Construct a "ramp-up" pipeline schedule for overlapped IO given an offset and
 * some schedulable components. Additionally, empty stages are inserted into the
 * schedule where a stage cannot be executed.
 *
 * @param offsets The offsets of each parallel sequence of inputs.
 * @param input The input sequence to schedule.
 * @param offset An additional offset to apply to all stages.
 * @param empty_element The empty element, or identity element, on the
 *                      ElementType. This is what is inserted into the "empty
 *                      stages" of the schedule.
 *
 * @returns A 2D array of pipeline schedule where each row represents the
 *          parallel sequence, and each column represents a single timestep
 *          where a single step of the input is scheduled.
 */
template <typename ElementType>
std::vector<std::vector<ElementType>> ConstructRampUpScheduleOverlapIO(
    const std::vector<int>& offsets, const std::vector<ElementType>& input,
    std::size_t offset, ElementType empty_element = {}) {
  CHECK_LT(offset, 3);
  auto result = ConstructRampUpSchedule(offsets, input, empty_element);

  for (std::size_t i = 0; i < offset; ++i) {
    result = LeftPadSchedule(result, empty_element);
  }

  auto right_padding = SliceSchedule(
      ForceInterleavedStageOrders(ConstructSchedule(offsets, input)),
      2 - offset);
  return ConcatSchedule(result, right_padding);
}

// Given the current stage at the current timestep, we do not need to
// perform recomputation iff there is another stage in the future time
// steps with the same stage_id executed before the corresponding bwd
// stage.
static bool ReplaceStageWithEmpty(
    const std::vector<std::vector<int64>>& stage_schedule,
    const std::vector<absl::flat_hash_set<int64>>& stage_schedule_lookup,
    const int64 t1, const int64 stage_id, const int64 bwd_stage_id) {
  bool replace_with_empty_element = false;
  for (int64 t2 = t1 + 1; t2 != stage_schedule.size(); ++t2) {
    if (stage_schedule_lookup[t2].contains(bwd_stage_id)) {
      // Found a corresponding bwd stage before another forward stage with
      // the same id, need to recompute.
      break;
    }
    if (stage_schedule_lookup[t2].contains(stage_id)) {
      // Found another forward stage with the same id before a corresponding
      // bwd stage, don't need to recompute.
      replace_with_empty_element = true;
      break;
    }
  }
  return replace_with_empty_element;
}

/**
 * Construct a "ramp-up" pipeline schedule for recomputation given an offset and
 * some schedulable components. Additionally, empty stages are inserted into the
 * schedule where a recomputation stage cannot be executed.
 *
 * @param offsets The offsets of each parallel sequence of inputs.
 * @param input The input sequence to schedule.
 * @param empty_element The empty element, or identity element, on the
 *                      ElementType. This is what is inserted into the "empty
 *                      stages" of the schedule.
 *
 * @returns A 2D array of pipeline recomputation schedule where each row
 *          represents the parallel sequence, and each column represents a
 *          single timestep where a single step of the input is scheduled.
 */
template <typename ElementType>
std::vector<std::vector<ElementType>> ConstructRecomputationRampUpSchedule(
    const std::vector<int>& offsets, const std::vector<ElementType>& input,
    int64 num_backward_stages, ElementType empty_element = {}) {
  // If there are no backward stages, there is no recomputation.
  if (num_backward_stages == 0) {
    return std::vector<std::vector<ElementType>>(
        offsets.size(), std::vector<ElementType>(input.size(), empty_element));
  }
  CHECK_EQ(input.size() / num_backward_stages, 2);
  CHECK_EQ(input.size() % num_backward_stages, 0);

  // The pipelining implementation expects the recomputation for backward stage
  // X at timestep 't2' to be done along with forward stage X at timestep `t1`
  // such that `t1` < `t2`.

  // Worked example:
  // num_backward_stages = 4
  // offsets = {0, 2, 4, 6}
  // First construct the ramp up schedule given those parameters
  // {-1,-1,-1,-1,-1,-1,-1, 0}
  // {-1,-1,-1,-1,-1, 0, 1, 2}
  // {-1,-1,-1, 0, 1, 2, 3, 4}
  // {-1, 0, 1, 2, 3, 4, 5, 6}
  // where each column represents a timestep, and each value represents a
  // pipeline stage and -1 represents nothing being executed.
  // Transpose it so that each row represents a timestep.
  // t = 0 {-1,-1,-1,-1}
  // t = 1 {-1,-1,-1, 0}
  // t = 2 {-1,-1,-1, 1}
  // t = 3 {-1,-1, 0, 2}
  // t = 4 {-1,-1, 1, 3}
  // t = 5 {-1, 0, 2, 4}
  // t = 6 {-1, 1, 3, 5}
  // t = 7 { 0, 2, 4, 6}
  // Now a recomputation needs to be placed at timestep t1 for stage with id x,
  // if there is a corresponding stage with id (2*num_backward_stages - 1 - x)
  // in timestep t2, where t2 > t1 and there is no other t3 for between t1 and
  // t2 in which x is executed.
  // For example at t=2, stage 1 is executed, its backward stage is 6, which is
  // executed in t=7. However between those stages, 1 is also executed at t=4
  // and t=6, so recomputation only needs to be placed at time step 6.
  // If the stage, for example stage 0, executed at t = 1, t = 3, t = 5 and
  // t = 7, has no corresponding backward stage in the ramp up, then only the
  // last execution of that stage in the ramp up needs to perform recomputation.

  // Create an execution trace which shows which fwd/bwd stages are exected
  // during ramp up - a value of -1 indicates no stage being executed.
  std::vector<int64> stages(input.size());
  absl::c_iota(stages, 0);
  std::vector<std::vector<int64>> stage_schedule =
      ConstructRampUpSchedule(offsets, stages, -1LL);
  // Transpose so that each row represents a single timestep.
  stage_schedule = TransposeSchedule(stage_schedule);
  // Create a lookup for what stages are executed in each timestep.
  std::vector<absl::flat_hash_set<int64>> stage_schedule_lookup(
      stage_schedule.size());
  absl::c_transform(
      stage_schedule, stage_schedule_lookup.begin(),
      [](const std::vector<int64>& timestep_schedule)
          -> absl::flat_hash_set<int64> {
        return {timestep_schedule.begin(), timestep_schedule.end()};
      });

  std::vector<std::vector<ElementType>> result =
      ConstructScheduleInternal(offsets, input);
  // Transpose so that each row represents a single timestep.
  result = TransposeSchedule(result);

  // Go through all the elements in the ramp up schedule, and insert empty
  // elements as appropriate.
  for (int64 t1 = 0; t1 != stage_schedule.size(); ++t1) {
    for (int64 j = 0; j != stage_schedule[t1].size(); ++j) {
      const int64 stage_id = stage_schedule[t1][j];
      // Mask if the stage is not meant to be executed or it is a backward
      // stage.
      if (stage_id == -1 || stage_id >= num_backward_stages) {
        result[t1][j] = empty_element;
        continue;
      }

      const int64 bwd_stage_id = num_backward_stages * 2 - 1 - stage_id;
      if (ReplaceStageWithEmpty(stage_schedule, stage_schedule_lookup, t1,
                                stage_id, bwd_stage_id)) {
        result[t1][j] = empty_element;
      }
    }
  }
  // Transpose the schedule back to the expected format.
  result = TransposeSchedule(result);
  return result;
}

/**
 * Construct a "ramp-up" pipeline schedule for recomputation with overlapped IO
 * given an offset and some schedulable components. Additionally, empty stages
 * are inserted into the schedule where a recomputation stage cannot be
 * executed.
 *
 * @param offsets The offsets of each parallel sequence of inputs.
 * @param input The input sequence to schedule.
 * @param offset An additional offset to apply to all stages.
 * @param empty_element The empty element, or identity element, on the
 *                      ElementType. This is what is inserted into the "empty
 *                      stages" of the schedule.
 *
 * @returns A 2D array of pipeline recomputation schedule where each row
 *          represents the parallel sequence, and each column represents a
 *          single timestep where a single step of the input is scheduled.
 */
template <typename ElementType>
std::vector<std::vector<ElementType>>
ConstructRecomputationRampUpScheduleOverlapIO(
    const std::vector<int>& offsets, const std::vector<ElementType>& input,
    int64 num_backward_stages, std::size_t offset,
    ElementType empty_element = {}) {
  CHECK_LT(offset, 3);
  auto result = ConstructRecomputationRampUpSchedule(
      offsets, input, num_backward_stages, empty_element);

  for (std::size_t i = 0; i < offset; ++i) {
    result = LeftPadSchedule(result, empty_element);
  }

  auto right_padding = SliceSchedule(
      ForceInterleavedStageOrders(ConstructSchedule(offsets, input)),
      2 - offset);
  return ConcatSchedule(result, right_padding);
}

template <typename ElementType>
static std::vector<std::vector<ElementType>> MaskOutPastAdditional(
    const std::vector<int>& offsets,
    std::vector<std::vector<ElementType>> input, ElementType empty_element,
    const int additional_iterations) {
  for (size_t i = additional_iterations; i < offsets.size(); ++i) {
    std::fill(std::next(input[i].begin(), offsets[i]), input[i].end(),
              empty_element);
  }
  return input;
}

template <typename ElementType>
static std::vector<std::vector<ElementType>> MaskOutPastAdditional(
    const std::vector<int>& offsets,
    std::vector<std::vector<ElementType>> input, ElementType empty_element,
    const PipelineVisitor::CountAndGraph& additional_iterations,
    const poplar::DebugContext& debug_context) {
  LOG(FATAL)
      << "Dynamic iterations makes no sense for non program type schedules";
}

template <>
std::vector<std::vector<poplar::program::Sequence>>
MaskOutPastAdditional<poplar::program::Sequence>(
    const std::vector<int>& offsets,
    std::vector<std::vector<poplar::program::Sequence>> input,
    poplar::program::Sequence empty_element,
    const PipelineVisitor::CountAndGraph& additional_iterations,
    const poplar::DebugContext& debug_context) {
  for (size_t i = 0; i < offsets.size(); ++i) {
    auto start = std::next(input[i].begin(), offsets[i]);
    std::transform(
        start, input[i].end(), start,
        [&](const poplar::program::Sequence& seq) {
          poplar::program::Sequence result({}, debug_context);
          auto predicate =
              popops::map(additional_iterations.graph,
                          popops::expr::Const(i) < popops::expr::_1,
                          {additional_iterations.count}, result, debug_context);
          result.add(poplar::program::If(std::move(predicate), seq,
                                         empty_element, debug_context));
          return result;
        });
  }
  return input;
}

/**
 * Construct a "ramp-down" pipeline schedule given an offset and some
 * schedulable components. Additionally, empty stages are inserted into the
 * schedule where a stage cannot be executed.
 *
 * @param offsets The offsets of each parallel sequence of inputs.
 * @param input The input sequence to schedule.
 * @param empty_element The empty element, or identity element, on the
 *                      ElementType. This is what is inserted into the "empty
 *                      stages" of the schedule.
 * @param additional_iterations The number of additional iterations that should
 *                              be executed to completely flush the pipeline.
 *
 * @returns A 2D array of pipeline schedule where each row represents the
 *          parallel sequence, and each column represents a single timestep
 *          where a single step of the input is scheduled.
 */
template <typename ElementType>
std::vector<std::vector<ElementType>> ConstructRampDownSchedule(
    const std::vector<int>& offsets, const std::vector<ElementType>& input,
    ElementType empty_element,
    const PipelineVisitor::IterationsType& additional_iterations,
    const poplar::DebugContext& debug_context) {
  auto result = ConstructScheduleInternal(offsets, input);

  return absl::visit(make_visitor<ElementType>(
                         [&](const int64 i) {
                           return MaskOutPastAdditional(
                               offsets, std::move(result),
                               std::move(empty_element), i);
                         },
                         [&](const PipelineVisitor::CountAndGraph& i) {
                           return MaskOutPastAdditional<ElementType>(
                               offsets, std::move(result),
                               std::move(empty_element), i, debug_context);
                         }),
                     additional_iterations);
}

/**
 * Construct a "ramp-down" pipeline schedule for overlapped IO given an offset
 * and some schedulable components. Additionally, empty stages are inserted into
 * the schedule where a stage cannot be executed.
 *
 * @param offsets The offsets of each parallel sequence of inputs.
 * @param input The input sequence to schedule.
 * @param offset An additional offset to apply to all stages.
 * @param empty_element The empty element, or identity element, on the
 *                      ElementType. This is what is inserted into the "empty
 *                      stages" of the schedule.
 *
 * @returns A 2D array of pipeline schedule where each row represents the
 *          parallel sequence, and each column represents a single timestep
 *          where a single step of the input is scheduled.
 */
template <typename ElementType>
std::vector<std::vector<ElementType>> ConstructRampDownScheduleOverlapIO(
    const std::vector<int>& offsets, const std::vector<ElementType>& input,
    std::size_t offset, ElementType empty_element = {},
    const poplar::DebugContext& debug_context = {}) {
  CHECK_LT(offset, 3);
  auto result = ConstructRampDownSchedule(offsets, input, empty_element, 0,
                                          debug_context);

  for (std::size_t i = 0; i < (2 - offset); ++i) {
    result = RightPadSchedule(result, empty_element);
  }

  auto left_padding = SliceSchedule(
      ForceInterleavedStageOrders(ConstructSchedule(offsets, input)), offset);
  return ConcatSchedule(left_padding, result);
}

/**
 * Construct a "ramp-down" pipeline schedule for recomputation given an offset
 * and some schedulable components. Additionally, empty stages are inserted into
 * the schedule where a recomputation stage cannot be executed.
 *
 * @param offsets The offsets of each parallel sequence of inputs.
 * @param input The input sequence to schedule.
 * @param empty_element The empty element, or identity element, on the
 *                      ElementType. This is what is inserted into the "empty
 *                      stages" of the schedule.
 *
 * @returns A 2D array of pipeline recomputation schedule where each row
 *          represents the parallel sequence, and each column represents a
 *          single timestep where a single step of the input is scheduled.
 */
template <typename ElementType>
std::vector<std::vector<ElementType>> ConstructRecomputationRampDownSchedule(
    const std::vector<int>& offsets, const std::vector<ElementType>& input,
    int64 num_backward_stages, ElementType empty_element = {},
    const PipelineVisitor::IterationsType& additional_iterations = 0,
    const poplar::DebugContext debug_context = {}) {
  // If there are no backward stages, there is no recomputation.
  if (num_backward_stages == 0) {
    return std::vector<std::vector<ElementType>>(
        offsets.size(), std::vector<ElementType>(input.size(), empty_element));
  }
  CHECK_EQ(input.size() / num_backward_stages, 2);
  CHECK_EQ(input.size() % num_backward_stages, 0);
  // The pipelining implementation expects the recomputation for backward stage
  // X at timestep 't2' to be done along with forward stage X at timestep `t1`
  // such that `t1` < `t2`.

  // Worked example:
  // num_backward_stages = 4
  // offsets = {0, 2, 4, 6}
  // First construct the ramp down schedule given those parameters
  // { 1, 2, 3, 4, 5, 6, 7,-1}
  // { 3, 4, 5, 6, 7,-1,-1,-1}
  // { 5, 6, 7,-1,-1,-1,-1,-1}
  // { 7,-1,-1,-1,-1,-1,-1,-1}
  // where each column represents a timestep, and each value represents a
  // pipeline stage and -1 represents nothing being executed.
  // Also construct a schedule without ramp down:
  // { 1, 2, 3, 4, 5, 6, 7, 0}
  // { 3, 4, 5, 6, 7, 0, 1, 2}
  // { 5, 6, 7, 0, 1, 2, 3, 4}
  // { 7, 0, 1, 2, 3, 4, 5, 6}
  // This shows where a recomputation stage needs to be placed - as those are
  // only placed in 'slots' with the forward stages.
  // Transpose both of those:
  // Ramp down schedule:     "Normal" schedule:
  // t = 0 { 1, 3, 5, 7}     { 1, 3, 5, 7}
  // t = 1 { 2, 4, 6,-1}     { 2, 4, 6, 0}
  // t = 2 { 3, 5, 7,-1}     { 3, 5, 7, 1}
  // t = 3 { 4, 6,-1,-1}     { 4, 6, 0, 2}
  // t = 4 { 5, 7,-1,-1}     { 5, 7, 1, 3}
  // t = 5 { 6,-1,-1,-1}     { 6, 0, 2, 4}
  // t = 6 { 7,-1,-1,-1}     { 7, 1, 3, 5}
  // t = 7 {-1,-1,-1,-1}     { 0, 2, 4, 6}
  // Now a recomputation needs to be placed at timestep t1 for stage with id x,
  // if there is a corresponding stage with id (2*num_backward_stages - 1 - x)
  // in timestep t2, where t2 > t1.
  // For example at t=1, stage 6 is executed, its recomputation stage is 1,
  // which in the "normal" schedule is executed at t=0, so a recomputation stage
  // needs to be placed there.

  // Create an execution trace which shows which fwd/bwd stages are exected
  // during ramp down.
  std::vector<int64> stages(input.size());
  absl::c_iota(stages, 0);
  std::vector<std::vector<int64>> ramp_down_schedule =
      ConstructRampDownSchedule(offsets, stages, -1LL, additional_iterations,
                                debug_context);

  // Transpose so that each row represents a single timestep.
  ramp_down_schedule = TransposeSchedule(ramp_down_schedule);
  // Create a lookup for what stages are executed in each timestep.
  std::vector<absl::flat_hash_set<int64>> ramp_down_schedule_lookup(
      ramp_down_schedule.size());
  absl::c_transform(
      ramp_down_schedule, ramp_down_schedule_lookup.begin(),
      [](const std::vector<int64>& timestep_schedule)
          -> absl::flat_hash_set<int64> {
        return {timestep_schedule.begin(), timestep_schedule.end()};
      });

  // Create an execution trace without ramp-down to show at what time slots the
  // recomputation has to be performed in (it has to be performed in a time-slot
  // for the corresponding forward stage).
  std::vector<std::vector<int64>> stage_schedule =
      ConstructScheduleInternal(offsets, stages);
  // Transpose so that each row represents a single timestep.
  stage_schedule = TransposeSchedule(stage_schedule);

  std::vector<std::vector<ElementType>> result =
      ConstructScheduleInternal(offsets, input);
  // Transpose so that each row represents a single timestep.
  result = TransposeSchedule(result);
  // Go through all the elements in the ramp down schedule, and insert empty
  // elements as appropriate.
  for (int64 t1 = 0; t1 != stage_schedule.size(); ++t1) {
    for (int64 j = 0; j != stage_schedule[t1].size(); ++j) {
      const int64 stage_id = stage_schedule[t1][j];
      // Mask if the stage is a backward stage.
      if (stage_id >= num_backward_stages) {
        result[t1][j] = empty_element;
        continue;
      }
      // Given the current stage at the current timestep, we only need to
      // perform recomputation if there is a corresponding bwd stage in a
      // future time step being executed.
      const int64 bwd_stage_id = num_backward_stages * 2 - 1 - stage_id;

      bool replace_with_empty_element = true;
      for (int64 t2 = t1 + 1; t2 != ramp_down_schedule.size(); ++t2) {
        const auto& timestep_schedule = ramp_down_schedule[t2];
        if (ramp_down_schedule_lookup[t2].contains(bwd_stage_id)) {
          // Found a corresponding bwd stage, need to recompute.
          replace_with_empty_element = false;
          break;
        }
      }

      if (replace_with_empty_element) {
        result[t1][j] = empty_element;
      }
    }
  }
  // Transpose the schedule back to the expected format.
  result = TransposeSchedule(result);
  return result;
}

/**
 * Construct a "ramp-down" pipeline schedule for recomputation given an offset
 * and some schedulable components. Additionally, empty stages are inserted into
 * the schedule where a recomputation stage cannot be executed.
 *
 * @param offsets The offsets of each parallel sequence of inputs.
 * @param input The input sequence to schedule.
 * @param offset An additional offset to apply to all stages.
 * @param empty_element The empty element, or identity element, on the
 *                      ElementType. This is what is inserted into the "empty
 *                      stages" of the schedule.
 *
 * @returns A 2D array of pipeline recomputation schedule where each row
 *          represents the parallel sequence, and each column represents a
 *          single timestep where a single step of the input is scheduled.
 */
template <typename ElementType>
std::vector<std::vector<ElementType>>
ConstructRecomputationRampDownScheduleOverlapIO(
    const std::vector<int>& offsets, const std::vector<ElementType>& input,
    int64 num_backward_stages, std::size_t offset,
    ElementType empty_element = {}) {
  CHECK_LT(offset, 3);
  auto result = ConstructRecomputationRampDownSchedule(
      offsets, input, num_backward_stages, empty_element);

  for (std::size_t i = 0; i < (2 - offset); ++i) {
    result = RightPadSchedule(result, empty_element);
  }

  auto left_padding = SliceSchedule(
      ForceInterleavedStageOrders(ConstructSchedule(offsets, input)), offset);
  return ConcatSchedule(left_padding, result);
}

/**
 * Given a schedule, like the ones produced by `ConstructSchedule`, flatten the
 * time axis to produce a single sequence.
 *
 * @param inputs The input parallel schedule.
 *
 * @returns The flattened schedule.
 */
template <typename ElementType>
std::vector<ElementType> FlattenSchedule(
    const std::vector<std::vector<ElementType>>& inputs) {
  std::vector<ElementType> result;

  auto inputs_transpose = TransposeSchedule(inputs);

  for (const auto& inputs : inputs_transpose) {
    result.insert(result.end(), inputs.begin(), inputs.end());
  }

  return result;
}

struct DefaultScheduler {
  template <typename ElementType>
  std::vector<int> ScheduleOffsets(
      const std::vector<ElementType>& input) const {
    return NoUnion(input);
  }

  template <typename ElementType>
  std::vector<std::vector<ElementType>> ConstructSchedule(
      const std::vector<int>& offsets,
      const std::vector<ElementType>& input) const {
    return ConstructScheduleInternal(offsets, input);
  }

  template <typename ElementType>
  poplar::program::Sequence CreateRepeatBlock(
      std::vector<std::vector<ElementType>>& infeed_sequences,
      const poplar::DebugNameAndId& debug_name_and_id,
      std::size_t offset_size) const {
    // Flatten the schedule to a linear sequence.
    auto repeat_block_sequences = FlattenSchedule(infeed_sequences);
    poplar::program::Sequence repeat_block({}, debug_name_and_id);
    for (const auto& seq : repeat_block_sequences) {
      repeat_block.add(seq);
    }
    return repeat_block;
  }
};

struct GroupedScheduler : public DefaultScheduler {
  template <typename ElementType>
  std::vector<int> ScheduleOffsets(
      const std::vector<ElementType>& input) const {
    return AllUnion(input);
  }

  template <typename ElementType>
  std::vector<std::vector<ElementType>> ConstructSchedule(
      const std::vector<int>& offsets,
      const std::vector<ElementType>& input) const {
    return ForceInterleavedStageOrders(
        ConstructScheduleInternal(offsets, input));
  }

  template <typename ElementType>
  poplar::program::Sequence CreateRepeatBlock(
      std::vector<std::vector<ElementType>>& infeed_sequences,
      const poplar::DebugNameAndId& debug_name_and_id,
      std::size_t offset_size) const {
    for (auto& seq : infeed_sequences) {
      seq.resize(1);
    }
    auto repeat_block = DefaultScheduler().CreateRepeatBlock(
        infeed_sequences, debug_name_and_id, offset_size);
    return poplar::program::Sequence({poplar::program::Repeat(
        offset_size, repeat_block, {debug_name_and_id})});
  }
};

struct InterleavedScheduler : public DefaultScheduler {
  template <typename ElementType>
  std::vector<int> ScheduleOffsets(
      const std::vector<ElementType>& input) const {
    return CircularUnion(input);
  }
};

struct PipelineSchedulerUtil {
  absl::variant<DefaultScheduler, GroupedScheduler, InterleavedScheduler> type;

  PipelineSchedulerUtil(
      PoplarBackendConfig::CallConfig::PipelineConfig::Schedule schedule) {
    switch (schedule) {
      case PoplarBackendConfig::CallConfig::PipelineConfig::Grouped:
        type = GroupedScheduler();
        break;
      case PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved:
        type = InterleavedScheduler();
        break;
      default:
        type = DefaultScheduler();
        break;
    }
  }

  template <typename ElementType>
  std::vector<int> ScheduleOffsets(
      const std::vector<ElementType>& input) const {
    // If using c++14 don't need this make visitor can use an auto lambda
    // absl::visit([&] (const auto s) {return s.ScheduleOffsets(input)});
    auto vis = make_visitor<std::vector<int>>(
        [&](const DefaultScheduler& scheduler) {
          return scheduler.ScheduleOffsets(input);
        },
        [&](const GroupedScheduler& scheduler) {
          return scheduler.ScheduleOffsets(input);
        },
        [&](const InterleavedScheduler& scheduler) {
          return scheduler.ScheduleOffsets(input);
        });
    return absl::visit(vis, type);
  }

  template <typename ElementType>
  std::vector<std::vector<ElementType>> ConstructSchedule(
      const std::vector<int>& offsets,
      const std::vector<ElementType>& input) const {
    auto vis = make_visitor<std::vector<std::vector<ElementType>>>(
        [&](const DefaultScheduler& scheduler) {
          return scheduler.ConstructSchedule(offsets, input);
        },
        [&](const GroupedScheduler& scheduler) {
          return scheduler.ConstructSchedule(offsets, input);
        },
        [&](const InterleavedScheduler& scheduler) {
          return scheduler.ConstructSchedule(offsets, input);
        });
    return absl::visit(vis, type);
  }

  template <typename ElementType>
  poplar::program::Sequence CreateRepeatBlock(
      std::vector<std::vector<ElementType>>& infeed_sequences,
      const poplar::DebugNameAndId& debug_name_and_id,
      std::size_t offset_size) const {
    auto vis = make_visitor<poplar::program::Sequence>(
        [&](const DefaultScheduler& scheduler) {
          return scheduler.CreateRepeatBlock(infeed_sequences,
                                             debug_name_and_id, offset_size);
        },
        [&](const GroupedScheduler& scheduler) {
          return scheduler.CreateRepeatBlock(infeed_sequences,
                                             debug_name_and_id, offset_size);
        },
        [&](const InterleavedScheduler& scheduler) {
          return scheduler.CreateRepeatBlock(infeed_sequences,
                                             debug_name_and_id, offset_size);
        });
    return absl::visit(vis, type);
  }
};

inline poplar::program::Sequence ForProgram(
    const absl::variant<int64, PipelineVisitor::CountAndGraph>& count,
    const poplar::program::Sequence& body,
    const poplar::DebugContext& debug_context) {
  return absl::visit(
      make_visitor<poplar::program::Sequence>(
          [&](const int64 i) -> poplar::program::Sequence {
            return poplar::program::Sequence(
                {poplar::program::Repeat(i, body, debug_context)});
          },
          [&](const PipelineVisitor::CountAndGraph i)
              -> poplar::program::Sequence {
            return popops::countedForLoop(i.graph, 0, i.count, 1, body,
                                          debug_context);
          }),
      count);
}

}  // namespace pipelinevisitorutils
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_PIPELINE_VISITOR_UTILS_H_
