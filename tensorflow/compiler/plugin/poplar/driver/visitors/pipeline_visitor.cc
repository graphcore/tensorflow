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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_visitor.h"

#include <stddef.h>
#include <string.h>

#include <map>
#include <memory>
#include <poplar/Engine.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/exceptions.hpp>
#include <poputil/Util.hpp>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
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

namespace {
/**
 * Construct a unary predicate which checks if a given HloInstruction has the
 * same opcode as the one captured in the closure.
 *
 * @param opcode The opcode to capture and compare against.
 *
 * @returns The unary predicate.
 */
std::function<bool(const HloInstruction*)> HasHloOpcode(HloOpcode opcode) {
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
std::function<bool(const HloInstruction*)> IsFifoInstruction() {
  return [](const HloInstruction* inst) -> bool {
    return IsPoplarInstruction(PoplarOp::Fifo)(inst);
  };
}

/**
 * Construct a unary predicate which checks if a given HloInstruction is an
 * HloIpuInterCopy.
 *
 * @returns The unary predicate.
 */
std::function<bool(const HloInstruction*)> IsIpuInterCopyInstruction() {
  return [](const HloInstruction* inst) -> bool {
    return IsPoplarInstruction(PoplarOp::IpuInterCopy)(inst);
  };
}

/**
 * Construct a unary predicate which checks if a given HloInstruction is an
 * HloGradientAccumulatorCreate.
 *
 * @returns The unary predicate.
 */
std::function<bool(const HloInstruction*)>
IsGradientAccumulatorCreateInstruction() {
  return [](const HloInstruction* inst) -> bool {
    return IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(inst);
  };
}

/**
 * Construct a unary predicate which checks if a given HloInstruction is an
 * HloGradientAccumulatorSink.
 *
 * @returns The unary predicate.
 */
std::function<bool(const HloInstruction*)>
IsGradientAccumulatorSinkInstruction() {
  return [](const HloInstruction* inst) -> bool {
    return IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(inst);
  };
}

/**
 * Get the number of stages in a pipeline.
 *
 * @param pipeline The outer pipeline instruction.
 *
 * @returns The number of stages inside the pipeline computation.
 *
 * @note Assumes the pipeline is correctly constructed.
 */
int64 GetPipelineStageCount(const HloInstruction* pipeline) {
  HloComputation* pipeline_computation = pipeline->to_apply();

  return absl::c_count_if(pipeline_computation->instructions(),
                          [](const HloInstruction* inst) {
                            return IsPipelineStageOrBackwardOp(inst);
                          });
}

/**
 * Get the pipeline stage to device mapping.
 *
 * @param pipeline The outer pipeline instruction.
 *
 * @returns The mapping of the ith stage to a IPU device.
 *
 * @note Assumes the pipeline is correctly constructed.
 */
std::vector<int> GetPipelineStageDeviceMapping(const HloInstruction* pipeline) {
  HloComputation* pipeline_computation = pipeline->to_apply();
  std::vector<HloInstruction*> instructions(
      pipeline_computation->instructions().begin(),
      pipeline_computation->instructions().end());

  // Cannot reasonably return StatusOr because this is called inside a
  // constructor.
  auto stage = GetPipelineStages(pipeline_computation).ValueOrDie();
  stage.forward.insert(stage.forward.end(), stage.backward.rbegin(),
                       stage.backward.rend());

  std::vector<int> result(stage.forward.size());

  const auto get_stage_shard = [](const HloInstruction* hlo) -> int {
    return *hlo->sharding_unique_device();
  };
  absl::c_transform(stage.forward, result.begin(), get_stage_shard);

  return result;
}

/**
 * Get the pipeline instruction to stage mapping. When an instruction isn't a
 * stage call, it must be associated with a stage.
 *
 * @param pipeline The outer pipeline instruction.
 *
 * @returns The mapping from Hlo instructions to pipeline stage index.
 *
 * @note Assumes the pipeline is correctly constructed.
 */
absl::flat_hash_map<const HloInstruction*, int> GetPipelineInstStageMapping(
    const HloInstruction* pipeline) {
  absl::flat_hash_map<const HloInstruction*, int> result;
  HloComputation* pipeline_computation = pipeline->to_apply();
  auto instructions = pipeline_computation->MakeInstructionPostOrder();

  // Cannot reasonably return StatusOr because this is called inside a
  // constructor.
  auto stage = GetPipelineStages(pipeline_computation).ValueOrDie();
  stage.forward.insert(stage.forward.end(), stage.backward.rbegin(),
                       stage.backward.rend());

  // Loop through all of the pipeline stage calls.
  // These trivially belong to the stage id that corresponds to their position.
  for (auto itr = stage.forward.begin(); itr != stage.forward.end(); ++itr) {
    result.insert(
        std::make_pair(*itr, std::distance(stage.forward.begin(), itr)));
  }

  // Assign the recomputation stages to the same stage as the forward stage.
  for (auto pair : stage.recomputation) {
    result[pair.second] = pair.first;
  }

  // Partition out the stage calls instructions and skip them.
  auto stages_end = std::stable_partition(
      instructions.begin(), instructions.end(), HasHloOpcode(HloOpcode::kCall));

  // Comparison of HloInstructions with assigned stage index.
  const auto inst_comparison = [&](HloInstruction* a,
                                   HloInstruction* b) -> bool {
    return result.at(a) < result.at(b);
  };

  // Assign the root instruction to the last stage. Note that we expect the root
  // instruction to be a tuple which does not modify the sequences.
  HloInstruction* root = pipeline_computation->root_instruction();
  CHECK_EQ(root->opcode(), HloOpcode::kTuple);
  result[root] = stage.forward.size() - 1;

  // Assign the resource update stage (if there is one) to the last stage. Note
  // that it won't actually be executed with the last stage.
  if (stage.resource_update) {
    result[*stage.resource_update] = stage.forward.size() - 1;
  }

  // Get the stage given the users. Requires all the users to already have a
  // stage.
  auto get_stage_from_users = [&](const HloInstruction* inst) {
    auto users = inst->users();
    return result.at(*absl::c_min_element(users, inst_comparison));
  };

  // Get the stage given the operands. Requires all the operands to already have
  // a stage.
  auto get_stage_from_operands = [&](const HloInstruction* inst) {
    auto operands = inst->operands();
    return result.at(*absl::c_max_element(operands, inst_comparison));
  };

  // Partition out infeeds.
  auto infeeds_end = std::stable_partition(stages_end, instructions.end(),
                                           HasHloOpcode(HloOpcode::kInfeed));
  for (auto itr = stages_end; itr != infeeds_end; ++itr) {
    HloInstruction* inst = *itr;
    // For an infeed, assign the stages for the infeed, its gte user, and
    // the input token.
    const HloInstruction* token = inst->operand(0);
    CHECK_EQ(inst->user_count(), 1);
    const HloInstruction* gte = inst->users()[0];
    // Expect at least one user of GTE to be a forward stage.
    auto fwd_stage_itr = absl::c_find_if(
        gte->users(),
        [](const HloInstruction* inst) { return IsPipelineStage(inst); });
    int64 stage = result.at(*fwd_stage_itr);
    result[inst] = stage;
    result[gte] = stage;
    result[token] = stage;
  }

  // Partition out the outfeeds.
  auto outfeeds_end = std::stable_partition(infeeds_end, instructions.end(),
                                            HasHloOpcode(HloOpcode::kOutfeed));
  for (auto itr = infeeds_end; itr != outfeeds_end; ++itr) {
    HloInstruction* inst = *itr;
    // For an outfeed, assign the stages for the outfeed, its gte operand, and
    // the input token.
    const HloInstruction* copy = inst->operand(0);
    const HloInstruction* gte = copy->operand(0);
    const HloInstruction* token = inst->operand(1);
    int64 stage = result.at(gte->operand(0));
    result[inst] = stage;
    result[gte] = stage;
    result[token] = stage;
  }

  // Partition out the Inter IPU copies and also assign stage to their operands.
  auto inter_ipu_copies_end = std::stable_partition(
      outfeeds_end, instructions.end(), IsIpuInterCopyInstruction());
  for (auto itr = outfeeds_end; itr != inter_ipu_copies_end; ++itr) {
    HloInstruction* inst = *itr;
    // Assign stages to the operands of the inter IPU copy.
    for (HloInstruction* operand : inst->operands()) {
      CHECK_EQ(operand->opcode(), HloOpcode::kGetTupleElement);
      result[operand] = get_stage_from_operands(operand);
    }
    // Then assign it to the copy.
    result[inst] = get_stage_from_operands(inst);
  }

  // Partition out GTEs which have not been assigned a stage - these are
  // assigned to the same stage as their input.
  auto gtes_end = std::stable_partition(
      inter_ipu_copies_end, instructions.end(),
      [&result](const HloInstruction* inst) {
        return HasHloOpcode(HloOpcode::kGetTupleElement)(inst) &&
               !result.contains(inst);
      });
  for (auto itr = inter_ipu_copies_end; itr != gtes_end; ++itr) {
    HloInstruction* inst = *itr;
    result[inst] = get_stage_from_operands(inst);
  }

  // Partition out the copies.
  auto copies_end = std::stable_partition(gtes_end, instructions.end(),
                                          HasHloOpcode(HloOpcode::kCopy));
  for (auto itr = gtes_end; itr != copies_end; ++itr) {
    result[*itr] = get_stage_from_operands(*itr);
  }

  // Partition out FIFOs - if the FIFO is an input to a recomputation stage,
  // then it is assigned to that stage, otherwise it it assigned to the same
  // stage as its input.
  auto fifos_end = std::stable_partition(copies_end, instructions.end(),
                                         IsFifoInstruction());
  for (auto itr = copies_end; itr != fifos_end; ++itr) {
    HloInstruction* inst = *itr;
    if (inst->user_count() == 1) {
      if (IsPipelineStageRecomputation(inst->users()[0])) {
        result[inst] = get_stage_from_users(inst);
      } else {
        result[inst] = get_stage_from_operands(inst);
      }
    } else {
      CHECK_EQ(inst->user_count(), 2);
      CHECK(!absl::c_any_of(inst->users(), IsPipelineStageRecomputation));
      result[inst] = get_stage_from_operands(inst);
    }
  }

  // Partition out the gradient accumulation buffers - these are assigned to the
  // first stage in which they are used in.
  auto gradient_accumulators_end = std::stable_partition(
      fifos_end, instructions.end(), IsGradientAccumulatorCreateInstruction());
  for (auto itr = fifos_end; itr != gradient_accumulators_end; ++itr) {
    HloInstruction* inst = *itr;
    result[inst] = get_stage_from_users(inst);
  }

  // Partition out parameters - these are assigned to the first stage in which
  // they are used in.
  auto parameters_end =
      std::stable_partition(gradient_accumulators_end, instructions.end(),
                            HasHloOpcode(HloOpcode::kParameter));
  for (auto itr = gradient_accumulators_end; itr != parameters_end; ++itr) {
    HloInstruction* inst = *itr;
    result[inst] = get_stage_from_users(inst);
  }

  // Go through the remaining instructions and assign them to stages given their
  // operands. Note that we are visiting in post-order.
  for (auto itr = parameters_end; itr != instructions.end(); ++itr) {
    HloInstruction* inst = *itr;
    // Only assign the stage if no other instruction assigned it for us.
    if (!result.contains(inst)) {
      result[inst] = get_stage_from_operands(inst);
    }
  }

  if (result.size() !=
      static_cast<size_t>(pipeline_computation->instruction_count())) {
    LOG(FATAL) << "Could not assign all the instructions to Pipeline Stages.";
  }
  return result;
}

/**
 * Get the pipeline stages which have stateless recomputation.
 *
 * @param pipeline The outer pipeline instruction.
 *
 * @returns The mapping of the ith stage to a IPU device.
 *
 * @note Assumes the pipeline is correctly constructed.
 */
absl::flat_hash_set<int> GetPipelineStagesWithStatelessRecomputation(
    const HloInstruction* pipeline) {
  HloComputation* pipeline_computation = pipeline->to_apply();
  // Cannot reasonably return StatusOr because this is called inside a
  // constructor.
  auto stages = GetPipelineStages(pipeline_computation).ValueOrDie();
  absl::flat_hash_set<int> result, tmp;
  absl::c_transform(stages.recomputation, std::inserter(tmp, tmp.begin()),
                    [&stages](const std::pair<int64, HloInstruction*>& pair) {
                      // Recomputation stages for stages containing stateful ops
                      // have a different number of operands
                      return (pair.second->operand_count() ==
                              stages.forward[pair.first]->operand_count())
                                 ? pair.first
                                 : -1;
                    });
  // Remove the -1s introduced by the previous transform.
  absl::c_remove_copy(tmp, std::inserter(result, result.begin()), -1);
  return result;
}

/**
 * Get the number of backward stages in the pipeline.
 *
 * @param pipeline The outer pipeline instruction.
 *
 * @returns The number of backward stages in the pipeline.
 *
 * @note Assumes the pipeline is correctly constructed.
 */
int64 GetNumberOfBackwardPipelineStages(const HloInstruction* pipeline) {
  HloComputation* pipeline_computation = pipeline->to_apply();
  // Cannot reasonably return StatusOr because this is called inside a
  // constructor.
  auto stages = GetPipelineStages(pipeline_computation).ValueOrDie();
  return stages.backward.size();
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
    const std::vector<int>& offsets, const std::vector<ElementType>& input,
    bool interleave) {
  auto result = ConstructScheduleInternal(offsets, input);

  // Force the stages to be added to poplar in a consistent order.
  if (!interleave) {
    result = TransposeSchedule(result);
    result = RotateSchedule(result);
    result = TransposeSchedule(result);
  }

  return result;
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
      // Given the current stage at the current timestep, we do not need to
      // perform recomputation iff there is another stage in the future time
      // steps with the same stage_id executed before the corresponding bwd
      // stage.
      const int64 bwd_stage_id = num_backward_stages * 2 - 1 - stage_id;

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
    ElementType empty_element = {}, const int additional_iterations = 0) {
  auto result = ConstructScheduleInternal(offsets, input);

  for (size_t i = additional_iterations; i < offsets.size(); ++i) {
    std::fill(std::next(result[i].begin(), offsets[i]), result[i].end(),
              empty_element);
  }

  return result;
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
    const int additional_iterations = 0) {
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
      ConstructRampDownSchedule(offsets, stages, -1LL, additional_iterations);
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

  for (const auto inputs : inputs_transpose) {
    result.insert(result.end(), inputs.begin(), inputs.end());
  }

  return result;
}

// Return the pipeline stage index for the given hlo instruction
StatusOr<int> GetPipelineStage(
    const absl::flat_hash_map<const HloInstruction*, int>& inst_stage_mapping,
    const HloInstruction* hlo) {
  if (inst_stage_mapping.count(hlo) == 0) {
    return FailedPrecondition(
        "Hlo instruction \"%s\" does not have an assigned pipeline stage.",
        hlo->ToString());
  }

  return inst_stage_mapping.at(hlo);
}

/**
 * Get all the inputs for the pipeline stage/resource update, making sure to
 * preserve aliasing. Note that there is a mix of inplace and not inplace inputs
 * - we get all of them.
 *
 * @param seq The sequence to use if any copies are inserted.
 * @param res The compiler resources.
 * @param inst The instruction for which we are getting inputs.
 * @param tensor_map The map which stores the tensors.
 *
 * @returns A 2D array of pipeline stage inputs.
 */
StatusOr<TensorOrRemoteBufferVectors> GetInputs(poplar::program::Sequence& seq,
                                                CompilerResources& res,
                                                const HloInstruction* inst,
                                                TensorMap& tensor_map) {
  TensorOrRemoteBufferVectors inputs(inst->operand_count());
  // First get all the inplace inputs - we do not expand constants and we
  // preserve all the aliasing.
  TF_ASSIGN_OR_RETURN(
      TensorOrRemoteBufferVectors inplace_inputs,
      FindInplaceOutputs(tensor_map, res, inst, seq, false, true));
  auto inplace_inputs_itr = inplace_inputs.begin();
  auto inst_description = HloInstructionDescription(inst);
  // Keep track of inputs which are not inplace (i.e. parameters for forward
  // stages).
  absl::flat_hash_set<int64> non_inplace_operand_indices;
  for (int64 op_idx = 0; op_idx != inst->operand_count(); ++op_idx) {
    non_inplace_operand_indices.insert(op_idx);
  }

  // Populate the inputs with the inplace inputs first.
  for (int64 inplace_idx : inst_description.GetInplaceOperandIndexes()) {
    inputs[inplace_idx] = *inplace_inputs_itr;
    inplace_inputs_itr++;
    non_inplace_operand_indices.erase(inplace_idx);
  }
  // Get all the non inplace inputs.
  if (inst_description.GetInplaceOperandIndexes().size() !=
      static_cast<size_t>(inst->operand_count())) {
    CHECK(IsAnyPipelineStageOp(inst));
    for (int64 op_idx : non_inplace_operand_indices) {
      inputs[op_idx] =
          FindInstructionInputs(tensor_map, res, inst, op_idx, seq, false);
    }
  }
  return inputs;
}
}  // namespace

PipelineVisitor::PipelineVisitor(
    PoplarBackendConfig::CallConfig::PipelineConfig::Schedule schedule,
    int64 stage_count, const std::vector<int>& stage_ipu_mapping,
    const absl::flat_hash_map<const HloInstruction*, int>& inst_stage_mapping,
    const absl::flat_hash_set<int> stages_with_recomputation,
    int64 num_backward_stages, CompilerResources& res,
    const DeferredArgRBVectors& inputs, const std::string& name)
    : InplaceDeferredVisitor(res, inputs, name, {}),
      schedule_(schedule),
      copy_sequences_(stage_count),
      inter_ipu_copy_sequences_(stage_count),
      fifo_sequences_(stage_count),
      infeed_sequences_(stage_count),
      outfeed_sequences_(stage_count),
      program_sequences_(stage_count),
      recomputation_sequences_(stage_count),
      stage_ipu_mapping_(stage_ipu_mapping),
      inst_stage_mapping_(inst_stage_mapping),
      stages_with_recomputation_(stages_with_recomputation),
      num_backward_stages_(num_backward_stages) {
  // Push a new vector for the zeroing sequences onto the stack.
  res.gradient_accumulation_zeroing_sequences.push({});
  // Push a new vector for the write undef sequences onto the stack.
  res.pipelining_write_undef_sequences.push({});
}

PipelineVisitor::PipelineVisitor(const HloInstruction* pipeline,
                                 CompilerResources& res,
                                 const DeferredArgRBVectors& inputs,
                                 const std::string& name)
    : PipelineVisitor(GetPipelineSchedule(pipeline).ValueOrDie(),
                      GetPipelineStageCount(pipeline),
                      GetPipelineStageDeviceMapping(pipeline),
                      GetPipelineInstStageMapping(pipeline),
                      GetPipelineStagesWithStatelessRecomputation(pipeline),
                      GetNumberOfBackwardPipelineStages(pipeline), res, inputs,
                      name) {}

StatusOr<poplar::program::Sequence> PipelineVisitor::GetPipelineSequence(
    int64 iterations) const {
  const int64 overlap_length =
      ScheduleOffsets(schedule_, stage_ipu_mapping_).size();

  if (iterations % overlap_length) {
    // TODO(T11404)
    return FailedPrecondition(
        "The pipeline depth of the pipeline must be a multiple of %d, but it "
        "is %d.",
        overlap_length, iterations);
  }
  // To account for ramp up and ramp down we need at least overlap_length
  // iterations.
  if (iterations < overlap_length) {
    return FailedPrecondition(
        "The pipeline depth of the pipeline must be at least %d, but it is %d.",
        overlap_length, iterations);
  }

  poplar::program::Program ramp_up = GetPipelineRampUpSequence();
  poplar::program::Program repeat_block = GetPipelineRepeatBlockSequence();
  poplar::program::Program ramp_down =
      GetPipelineRampDownSequence(iterations % overlap_length);

  poplar::program::Sequence program;
  program.add(pipeline_execution_counters_initialize_sequence_);
  program.add(pipeline_tensors_zeroing_sequence_);
  program.add(pipeline_write_undef_sequence_);
  program.add(ramp_up);
  if ((iterations / overlap_length) - 1 > 0) {
    program.add(poplar::program::Repeat((iterations / overlap_length) - 1,
                                        repeat_block));
  }
  program.add(ramp_down);

  // Add the resource update sequence.
  program.add(resource_update_);

  return program;
}

// Collect the pipeline stage programs and call CreateRampSequences
poplar::program::Program PipelineVisitor::GetPipelineRampUpSequence() const {
  std::vector<int> offsets = ScheduleOffsets(schedule_, stage_ipu_mapping_);
  const bool is_grouped =
      schedule_ == PoplarBackendConfig::CallConfig::PipelineConfig::Grouped;

  // Build schedules for the compute and copy programs.
  // Each schedule is 2D, where each column represents a time-slice and each row
  // represents the "mini-batch",
  auto infeed_sequences = ConstructRampUpSchedule(offsets, infeed_sequences_);
  auto program_sequences = ConstructRampUpSchedule(offsets, program_sequences_);
  auto fifo_sequences = ConstructRampUpSchedule(offsets, fifo_sequences_);
  auto recomputation_sequences = ConstructRecomputationRampUpSchedule(
      offsets, recomputation_sequences_, num_backward_stages_);
  auto copy_sequences =
      ConstructSchedule(offsets, copy_sequences_, !is_grouped);
  auto inter_ipu_copy_sequences =
      ConstructSchedule(offsets, inter_ipu_copy_sequences_, !is_grouped);
  auto outfeed_sequences = ConstructRampUpSchedule(offsets, outfeed_sequences_);

  // Concatenate the programs in the correct order.
  // We always execute in following order - infeeds, fwd/bwd stages, fifos,
  // recomputation stages, outfeeds and then inter-ipu-copies.
  infeed_sequences.insert(infeed_sequences.end(), program_sequences.begin(),
                          program_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), fifo_sequences.begin(),
                          fifo_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(),
                          recomputation_sequences.begin(),
                          recomputation_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), copy_sequences.begin(),
                          copy_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(),
                          inter_ipu_copy_sequences.begin(),
                          inter_ipu_copy_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), outfeed_sequences.begin(),
                          outfeed_sequences.end());

  // Flatten the schedule to a linear sequence.
  auto repeat_block_sequences = FlattenSchedule(infeed_sequences);

  poplar::program::Sequence repeat_block;
  for (const auto& seq : repeat_block_sequences) {
    repeat_block.add(seq);
  }

  return repeat_block;
}

// Collect the pipeline stage programs and call CreateRampSequences
poplar::program::Program PipelineVisitor::GetPipelineRampDownSequence(
    int additional_iterations) const {
  // Find the set of non-overlapping program offsets.
  std::vector<int> offsets = ScheduleOffsets(schedule_, stage_ipu_mapping_);

  const bool is_grouped =
      schedule_ == PoplarBackendConfig::CallConfig::PipelineConfig::Grouped;

  // Build schedules for the compute and copy programs.
  // Each schedule is 2D, where each column represents a time-slice and each row
  // represents the "mini-batch",
  auto infeed_sequences = ConstructRampDownSchedule(offsets, infeed_sequences_,
                                                    {}, additional_iterations);
  auto program_sequences = ConstructRampDownSchedule(
      offsets, program_sequences_, {}, additional_iterations);
  auto fifo_sequences =
      ConstructSchedule(offsets, fifo_sequences_, !is_grouped);
  auto recomputation_sequences = ConstructRecomputationRampDownSchedule(
      offsets, recomputation_sequences_, num_backward_stages_, {},
      additional_iterations);
  auto copy_sequences =
      ConstructSchedule(offsets, copy_sequences_, !is_grouped);
  auto inter_ipu_copy_sequences =
      ConstructSchedule(offsets, inter_ipu_copy_sequences_, !is_grouped);
  auto outfeed_sequences = ConstructRampDownSchedule(
      offsets, outfeed_sequences_, {}, additional_iterations);

  // Concatenate the programs in the correct order.
  // We always execute in following order - infeeds, fwd/bwd stages, fifos,
  // recomputation stages, outfeeds and then inter-ipu-copies.
  infeed_sequences.insert(infeed_sequences.end(), program_sequences.begin(),
                          program_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), fifo_sequences.begin(),
                          fifo_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(),
                          recomputation_sequences.begin(),
                          recomputation_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), copy_sequences.begin(),
                          copy_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(),
                          inter_ipu_copy_sequences.begin(),
                          inter_ipu_copy_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), outfeed_sequences.begin(),
                          outfeed_sequences.end());

  // Flatten the schedule to a linear sequence.
  auto repeat_block_sequences = FlattenSchedule(infeed_sequences);

  poplar::program::Sequence repeat_block;
  for (const auto& seq : repeat_block_sequences) {
    repeat_block.add(seq);
  }

  return repeat_block;
}

// Collect the pipeline stage programs and build the repeat block
poplar::program::Program PipelineVisitor::GetPipelineRepeatBlockSequence()
    const {
  // Find the set of non-overlapping program offsets.
  std::vector<int> offsets = ScheduleOffsets(schedule_, stage_ipu_mapping_);

  const bool is_grouped =
      schedule_ == PoplarBackendConfig::CallConfig::PipelineConfig::Grouped;

  // Build schedules for the compute and copy programs.
  // Each schedule is 2D, where each column represents a time-slice and each row
  // represents the "mini-batch",
  auto fifo_sequences =
      ConstructSchedule(offsets, fifo_sequences_, !is_grouped);
  auto infeed_sequences =
      ConstructSchedule(offsets, infeed_sequences_, !is_grouped);
  auto program_sequences =
      ConstructSchedule(offsets, program_sequences_, !is_grouped);
  auto recomputation_sequences =
      ConstructSchedule(offsets, recomputation_sequences_, !is_grouped);
  auto copy_sequences =
      ConstructSchedule(offsets, copy_sequences_, !is_grouped);
  auto inter_ipu_copy_sequences =
      ConstructSchedule(offsets, inter_ipu_copy_sequences_, !is_grouped);
  auto outfeed_sequences =
      ConstructSchedule(offsets, outfeed_sequences_, !is_grouped);

  // Concatenate the programs in the correct order.
  // We always execute in following order - infeeds, fwd/bwd stages, fifos,
  // recomputation stages, outfeeds and then inter-ipu-copies.
  infeed_sequences.insert(infeed_sequences.end(), program_sequences.begin(),
                          program_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), fifo_sequences.begin(),
                          fifo_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(),
                          recomputation_sequences.begin(),
                          recomputation_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), copy_sequences.begin(),
                          copy_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(),
                          inter_ipu_copy_sequences.begin(),
                          inter_ipu_copy_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), outfeed_sequences.begin(),
                          outfeed_sequences.end());

  if (is_grouped) {
    for (auto& seq : infeed_sequences) {
      seq.resize(1);
    }
  }

  // Flatten the schedule to a linear sequence.
  auto repeat_block_sequences = FlattenSchedule(infeed_sequences);

  poplar::program::Sequence repeat_block;
  for (const auto& seq : repeat_block_sequences) {
    repeat_block.add(seq);
  }

  if (!is_grouped) {
    return repeat_block;
  } else {
    return poplar::program::Repeat(offsets.size(), repeat_block);
  }
}

StatusOr<poplar::program::Sequence*> PipelineVisitor::GetSequenceForInstruction(
    const HloInstruction* hlo) {
  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, hlo));
  switch (hlo->opcode()) {
    case HloOpcode::kCall: {
      if (IsResourceUpdate(hlo)) {
        return &resource_update_;
      } else {
        return IsPipelineStageRecomputation(hlo)
                   ? &recomputation_sequences_[stage]
                   : &program_sequences_[stage];
      }
    }
    case HloOpcode::kGetTupleElement: {
      const HloInstruction* gte_input = hlo->operand(0);
      if (IsResourceUpdate(gte_input)) {
        return &resource_update_;
      } else {
        return IsPipelineStageRecomputation(gte_input)
                   ? &recomputation_sequences_[stage]
                   : &program_sequences_[stage];
      }
    }
    case HloOpcode::kInfeed: {
      return &infeed_sequences_[stage];
    }
    case HloOpcode::kParameter: {
      return &program_sequences_[stage];
    }
    case HloOpcode::kTuple: {
      CHECK_EQ(hlo->parent()->root_instruction(), hlo);
      return &resource_update_;
    }
    default: {
      return InternalErrorStrCat("Trying to get a sequence for ",
                                 hlo->ToString(), " which is not supported.");
    }
  }
}

Status PipelineVisitor::HandleNotImplemented(HloInstruction* hlo) {
  return xla::Unimplemented(
      "%s (%s) is not a valid pipeline stage Hlo instruction",
      hlo->name().c_str(), HloOpcodeString(hlo->opcode()).c_str());
}

/**
 * Creates the PipelineStageVisitor for a PiplineStage or PipelineStageBackward
 * instruction and populates the sequence ready for the execution.
 *
 * @param inst The PiplineStage or PipelineStageBackward instruction which is
 * being lowered.
 *
 * @returns The Poplar sequence with lowering of the stage.
 */
StatusOr<poplar::program::Sequence> PipelineVisitor::CreatePipelineStageOp(
    const HloInstruction* inst) {
  poplar::program::Sequence seq;
  poplar::Graph& graph = GetGraph(resources_, inst);
  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, inst));
  const std::string debug_name = GetDebugName(inst);

  TF_ASSIGN_OR_RETURN(DeferredArgRBVectors inputs,
                      GetInputsForDeferredInplaceRBInstruction(
                          inst, /*preserve_aliasing*/ true));

  const bool has_recomputation = stages_with_recomputation_.contains(stage);

  std::unique_ptr<PipelineStageVisitor> visitor;
  if (has_recomputation) {
    DeferredArgRBVectors visitor_inputs = inputs;
    // When recomputation is enabled, we need to add clones for inplace inputs
    // of the pipeline stage (i.e. non parameters/weights), so that we can
    // reuse the code for the recomputation stage.
    auto inst_description = HloInstructionDescription(inst);
    for (int64 inplace_idx : inst_description.GetInplaceOperandIndexes()) {
      for (size_t flat_idx = 0; flat_idx != inputs[inplace_idx].size();
           ++flat_idx) {
        auto optional_tensor = visitor_inputs[inplace_idx][flat_idx];
        if (optional_tensor) {
          const std::string name =
              absl::StrCat(debug_name, "/clone/", inplace_idx, "/", flat_idx);
          VLOG(1) << "Adding a clone for inplace input (" << inplace_idx << ", "
                  << flat_idx << ").";
          visitor_inputs[inplace_idx][flat_idx] = graph.clone(
              *optional_tensor, name,
              poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
        }
      }
    }
    visitor = absl::make_unique<ReusablePipelineStageVisitor>(
        resources_, visitor_inputs, debug_name);
  } else {
    visitor =
        absl::make_unique<PipelineStageVisitor>(resources_, inputs, debug_name);
  }

  HloComputation* stage_computation = inst->to_apply();
  auto order = stage_computation->parent()
                   ->schedule()
                   .sequence(stage_computation)
                   .instructions();

  TF_RETURN_IF_ERROR(stage_computation->AcceptOrdered(visitor.get(), order));
  // Make sure any deferred inputs to the instruction are pushed up.
  TF_RETURN_IF_ERROR(visitor->PropagateDeferredAllocations(inst));

  // Get the sequence for the stage.
  if (has_recomputation) {
    ReusablePipelineStageVisitor* reusable_visitor =
        static_cast<ReusablePipelineStageVisitor*>(visitor.get());

    // Since the sequence is reused, separate execution counters are required to
    // make sure they are correct and independent of the sequence being used for
    // the recomputation stage.
    ExecutionCounters& sequence_counters =
        reusable_visitor->GetExecutionCounters();
    ExecutionCounters forward_counters = sequence_counters.Clone();

    // Initialize the counters once from the outer scope.
    TF_RETURN_IF_ERROR(CopyExecutionCountersFromScope(
        resources_, forward_counters,
        pipeline_execution_counters_initialize_sequence_));

    // Before every execution of the sequence, copy the counters in.
    TF_ASSIGN_OR_RETURN(
        poplar::program::Sequence counters_in,
        sequence_counters.SetInitialValuesFrom(&forward_counters));
    seq.add(counters_in);

    // Execute the shared sequence.
    seq.add(
        reusable_visitor->GetForwardStageSequence(inst, inputs, tensor_map));

    // After every execution of the sequence, copy the counters out.
    TF_ASSIGN_OR_RETURN(poplar::program::Sequence counters_out,
                        sequence_counters.UpdateCounters(&forward_counters));
    seq.add(counters_out);
  } else {
    // Initialize the counters once from the outer scope.
    TF_RETURN_IF_ERROR(CopyExecutionCountersFromScope(
        resources_, visitor->GetExecutionCounters(),
        pipeline_execution_counters_initialize_sequence_));
    // Execute the sequence.
    seq.add(visitor->GetCachedSequence());
  }

  // Set the outputs.
  const TensorOrRemoteBufferVector& pipeline_outputs = visitor->outputs();
  const ShapeTree<bool> add_copies = visitor->GetOutputCopies(inst);
  size_t flat_tuple_index = 0;
  for (const auto& leaf : add_copies.leaves()) {
    poplar::Tensor output = pipeline_outputs[flat_tuple_index];
    if (leaf.second) {
      output = poputil::duplicate(
          graph, output, seq,
          absl::StrCat(debug_name, "/output/", flat_tuple_index),
          poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    }
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, flat_tuple_index, output));

    flat_tuple_index++;
  }
  CHECK_EQ(pipeline_outputs.size(), flat_tuple_index);

  fwd_stage_visitors_[stage] = std::move(visitor);

  return seq;
}

/**
 * Lowers a PipelineStageRecomputation into Poplar by reusing the sequence from
 * the corresponding PipelineStage visitor if possible, otherwise create a new
 * sequence.
 *
 * @param inst The PipelineStageRecomputation instruction which is being
 * lowered.
 *
 * @returns The Poplar sequence with lowering of the stage.
 */
StatusOr<poplar::program::Sequence>
PipelineVisitor::CreatePipelineStageRecomputationOp(
    const HloInstruction* inst) {
  poplar::program::Sequence seq;
  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, inst));
  // Get the non-deferred inputs for the pipeline stage.
  TF_ASSIGN_OR_RETURN(auto inputs,
                      GetInputs(seq, resources_, inst, tensor_map));
  // Recomputation stages reuse the forward stage visitor.
  PipelineStageVisitor* forward_stage_visitor =
      fwd_stage_visitors_.at(stage).get();

  // If the number of inputs of the recomputation doesn't match the number of
  // inputs of the forward stage then this is a stateful recomputation and we
  // need to create a new sequence.
  if (forward_stage_visitor->inputs().size() != inputs.size()) {
    PipelineStageVisitor visitor(
        resources_, ConvertInputsToDeferredInputs(inputs), GetDebugName(inst));
    HloComputation* stage_computation = inst->to_apply();
    auto order = stage_computation->parent()
                     ->schedule()
                     .sequence(stage_computation)
                     .instructions();
    TF_RETURN_IF_ERROR(stage_computation->AcceptOrdered(&visitor, order));

    // Initialize the counters once from the outer scope.
    TF_RETURN_IF_ERROR(CopyExecutionCountersFromScope(
        resources_, visitor.GetExecutionCounters(),
        pipeline_execution_counters_initialize_sequence_));

    // Note that it is not required to propagate any deferred allocations here
    // as recomputations do not have any deferred inputs.

    // Get the sequence for the stage.
    seq.add(visitor.GetCachedSequence());

    // Set the outputs.
    const TensorOrRemoteBufferVector& pipeline_outputs = visitor.outputs();
    for (size_t i = 0; i < pipeline_outputs.size(); i++) {
      poplar::Tensor output = pipeline_outputs[i];
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, output));
    }
  } else {
    ReusablePipelineStageVisitor* reusable_visitor =
        static_cast<ReusablePipelineStageVisitor*>(forward_stage_visitor);

    // Since the sequence is reused, separate execution counters are required to
    // make sure they are correct and independent of the sequence being used for
    // the forward stage.
    ExecutionCounters& sequence_counters =
        reusable_visitor->GetExecutionCounters();
    ExecutionCounters recomputation_counters = sequence_counters.Clone();
    // Initialize the counters once from the outer scope.
    TF_RETURN_IF_ERROR(CopyExecutionCountersFromScope(
        resources_, recomputation_counters,
        pipeline_execution_counters_initialize_sequence_));

    // Before every execution of the sequence, copy the counters in.
    TF_ASSIGN_OR_RETURN(
        poplar::program::Sequence counters_in,
        sequence_counters.SetInitialValuesFrom(&recomputation_counters));
    seq.add(counters_in);

    // Execute the shared sequence.
    seq.add(reusable_visitor->GetRecomputationStageSequence(inst, inputs));

    // After every execution of the sequence, copy the counters out.
    TF_ASSIGN_OR_RETURN(
        poplar::program::Sequence counters_out,
        sequence_counters.UpdateCounters(&recomputation_counters));
    seq.add(counters_out);

    // Set the outputs.
    const TensorOrRemoteBufferVector& pipeline_outputs =
        forward_stage_visitor->outputs();
    for (size_t i = 0; i < pipeline_outputs.size(); i++) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, pipeline_outputs[i]));
    }
  }
  return seq;
}

Status PipelineVisitor::HandleDeferredAllocationCall(HloInstruction* hlo) {
  HloComputation* comp = hlo->to_apply();

  if (IsResourceUpdate(hlo)) {
    TF_ASSIGN_OR_RETURN(DeferredArgRBVectors inputs,
                        GetInputsForDeferredInplaceRBInstruction(
                            hlo, /*preserve_aliasing*/ true));

    TF_ASSIGN_OR_RETURN(resource_update_,
                        CreateResourceUpdateOp(resources_, hlo, inputs,
                                               hlo->shape(), tensor_map));
  } else {
    TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, hlo));

    VLOG(1) << "Processing " << hlo->name() << " : " << comp->name()
            << " as a pipeline stage";

    if (IsPipelineStageOrBackwardOp(hlo)) {
      TF_ASSIGN_OR_RETURN(poplar::program::Sequence seq,
                          CreatePipelineStageOp(hlo));
      program_sequences_[stage].add(seq);
    } else if (IsPipelineStageRecomputation(hlo)) {
      TF_ASSIGN_OR_RETURN(poplar::program::Sequence seq,
                          CreatePipelineStageRecomputationOp(hlo));
      recomputation_sequences_[stage].add(seq);
    } else {
      return HandleNotImplemented(hlo);
    }
  }

  return Status::OK();
}

Status PipelineVisitor::HandleCopy(HloInstruction* hlo) {
  VLOG(1) << "Processing " << hlo->name();

  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, hlo));
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateCopy(resources_, hlo, GetOutputShape(hlo), tensor_map));
  copy_sequences_[stage].add(prog);

  return Status::OK();
}

Status PipelineVisitor::HandleNonDeferredCustomCall(HloInstruction* hlo) {
  if (IsFifoInstruction()(hlo)) {
    return HandleFifo(hlo);
  } else if (IsIpuInterCopyInstruction()(hlo)) {
    return HandleInterIpuCopy(hlo);
  } else if (IsGradientAccumulatorSinkInstruction()(hlo)) {
    return HandleGradientAccumulatorSink(hlo);
  } else {
    return HandleNotImplemented(hlo);
  }
}

Status PipelineVisitor::HandleFifo(HloInstruction* hlo) {
  VLOG(1) << "Processing " << hlo->ToString();
  if (!IsPoplibsHloCustomOp(hlo)) {
    return HandleNotImplemented(hlo);
  }

  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, hlo));
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateCustomCallOp(resources_, hlo, hlo->shape(), tensor_map));

  fifo_sequences_[stage].add(prog);

  return Status::OK();
}

Status PipelineVisitor::HandleInterIpuCopy(HloInstruction* hlo) {
  VLOG(1) << "Processing " << hlo->name();
  if (!IsPoplibsHloCustomOp(hlo)) {
    return HandleNotImplemented(hlo);
  }

  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, hlo));
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateCustomCallOp(resources_, hlo, hlo->shape(), tensor_map));

  inter_ipu_copy_sequences_[stage].add(prog);

  return Status::OK();
}

Status PipelineVisitor::HandleGradientAccumulatorSink(HloInstruction* hlo) {
  VLOG(1) << "Processing " << hlo->name();
  if (!IsPoplibsHloCustomOp(hlo)) {
    return HandleNotImplemented(hlo);
  }

  // The sink op just makes sure that all the uses of the gradient
  // accumulation buffer in different pipeline stages on the same device result
  // in the same buffer.
  // No sequence is created.
  TF_RETURN_IF_ERROR(
      CreateCustomCallOp(resources_, hlo, hlo->shape(), tensor_map).status());

  return Status::OK();
}

Status PipelineVisitor::HandleOutfeed(HloInstruction* hlo) {
  VLOG(1) << "Processing " << hlo->ToString();
  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, hlo));
  TF_ASSIGN_OR_RETURN(poplar::program::Program prog,
                      CreateOutfeed(resources_, hlo, tensor_map));

  outfeed_sequences_[stage].add(prog);
  return Status::OK();
}

Status PipelineVisitor::HandleDeferredAllocationTuple(HloInstruction* hlo) {
  if (hlo->parent()->root_instruction() != hlo) {
    return FailedPrecondition(
        "Hlo tuple instructions are only allowed in a pipeline when they are "
        "the root instruction. Hlo instruction \"%s\" is not.",
        hlo->name());
  }
  return InplaceDeferredVisitor::HandleDeferredAllocationTuple(hlo);
}

Status PipelineVisitor::HandleDeferredAllocationWhile(HloInstruction* hlo) {
  return HandleNotImplemented(hlo);
}

Status PipelineVisitor::FinishDeferedAllocationVisit(HloInstruction* inst) {
  // Create a sequence for all the zeroing of pipeline tensors (gradient
  // accumulation).
  auto& zeroing_seqs = resources_.gradient_accumulation_zeroing_sequences.top();
  for (poplar::program::Sequence& zeroing_seq : zeroing_seqs) {
    pipeline_tensors_zeroing_sequence_.add(zeroing_seq);
  }
  resources_.gradient_accumulation_zeroing_sequences.pop();

  // Create a sequence for all the write undefs of pipeline tensors (FIFOs).
  auto& write_undefs = resources_.pipelining_write_undef_sequences.top();
  for (poplar::program::Sequence& write_undef : write_undefs) {
    pipeline_write_undef_sequence_.add(write_undef);
  }
  resources_.pipelining_write_undef_sequences.pop();

  // Wrap each of the poplar sequences in a poplar function to maximise code
  // reuse.
  poplar::Graph& graph = GetMasterGraph(resources_);

  // Transform a given sequence into a poplar function call sequence.
  auto to_function = [&graph](const poplar::program::Sequence& seq) mutable
      -> poplar::program::Sequence {
    auto f = graph.addFunction(seq);
    return poplar::program::Sequence(poplar::program::Call(f));
  };

  // Transform all of the pipeline stage sequences into poplar function calls.
  absl::c_transform(copy_sequences_, copy_sequences_.begin(), to_function);
  absl::c_transform(inter_ipu_copy_sequences_,
                    inter_ipu_copy_sequences_.begin(), to_function);
  absl::c_transform(fifo_sequences_, fifo_sequences_.begin(), to_function);
  absl::c_transform(infeed_sequences_, infeed_sequences_.begin(), to_function);
  absl::c_transform(outfeed_sequences_, outfeed_sequences_.begin(),
                    to_function);
  absl::c_transform(program_sequences_, program_sequences_.begin(),
                    to_function);
  absl::c_transform(recomputation_sequences_, recomputation_sequences_.begin(),
                    to_function);

  return Status::OK();
}

poplar::program::Sequence& PipelineVisitor::GetSequenceForAliasingCopy() {
  return resource_update_;
}

}  // namespace poplarplugin
}  // namespace xla
