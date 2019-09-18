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
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"

#include <stddef.h>
#include <string.h>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/stream_executor/lib/initialize.h"

#include <poplar/Engine.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/exceptions.hpp>

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
std::function<bool(HloInstruction*)> HasHloOpcode(HloOpcode opcode) {
  return [opcode](const HloInstruction* inst) -> bool {
    return inst->opcode() == opcode;
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
                          HasHloOpcode(HloOpcode::kCall));
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

  // Partition out the stage calls instructions and skip them.
  auto p1 = std::stable_partition(instructions.begin(), instructions.end(),
                                  HasHloOpcode(HloOpcode::kCall));

  // Partition out the parameter instructions.
  auto p2 = std::stable_partition(p1, instructions.end(),
                                  HasHloOpcode(HloOpcode::kParameter));

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

  // Loop through all of the parameters and assign them to the earliest stage of
  // the users. The users must be pipeline stage calls/root instruction which
  // have already been visited.
  for (auto itr = p1; itr != p2; ++itr) {
    auto inst = *itr;

    auto users = inst->users();
    auto min_elem =
        std::min_element(users.begin(), users.end(), inst_comparison);

    result.insert(std::make_pair(inst, result.at(*min_elem)));
  }

  // Loop through the remaining instructions, assigning them to the latest stage
  // of their operand(s). Since we are visiting in post-order, the operands
  // must've already been visited.
  for (auto itr = p2; itr != instructions.end(); ++itr) {
    auto inst = *itr;
    auto operands = inst->operands();
    auto max_elem =
        std::max_element(operands.begin(), operands.end(), inst_comparison);

    result.insert(std::make_pair(inst, result.at(*max_elem)));
  }

  return result;
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
  for (int i = 1; i < input.size(); ++i) {
    // Take the ith rotated input.
    std::rotate_copy(input.begin(), std::next(input.begin(), i), input.end(),
                     temp_0.begin());

    bool non_overlapping = true;

    // Compare against all accept rotations of the input
    for (int k = 0; k < result.size() && non_overlapping; ++k) {
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
  std::vector<std::vector<ElementType>> result(offsets.size(), input);

  for (int i = 0; i < offsets.size(); ++i) {
    std::rotate(result[i].begin(),
                std::next(result[i].begin(), result[i].size() - offsets[i]),
                result[i].end());
  }

  return result;
}

/**
 * Construct a "ramp-up" pipeline schedule given an offset and some schedulable
 * components. Additionally, blank spaces are inserted into the schedule where a
 * stage cannot be executed.
 *
 * @param offsets The offsets of each parallel sequence of inputs.
 * @param input The input sequence to schedule.
 * @param empty_element The empty element, or identity element, on the
 *                      ElementType. This is what is inserted into the "blank
 *                      spaces" of the schedule.
 *
 * @returns A 2D array of pipeline schedule where each row represents the
 *          parallel sequence, and each column represents a single timestep
 *          where a single step of the input is scheduled.
 */
template <typename ElementType>
std::vector<std::vector<ElementType>> ConstructRampUpSchedule(
    const std::vector<int>& offsets, const std::vector<ElementType>& input,
    ElementType empty_element = {}) {
  std::vector<std::vector<ElementType>> result =
      ConstructSchedule(offsets, input);

  for (int i = 0; i < offsets.size(); ++i) {
    std::fill(result[i].begin(), std::next(result[i].begin(), offsets[i]),
              empty_element);
  }

  return result;
}

/**
 * Construct a "ramp-down" pipeline schedule given an offset and some
 * schedulable components. Additionally, blank spaces are inserted into the
 * schedule where a stage cannot be executed.
 *
 * @param offsets The offsets of each parallel sequence of inputs.
 * @param input The input sequence to schedule.
 * @param empty_element The empty element, or identity element, on the
 *                      ElementType. This is what is inserted into the "blank
 *                      spaces" of the schedule.
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
  std::vector<std::vector<ElementType>> result =
      ConstructSchedule(offsets, input);

  for (int i = additional_iterations; i < offsets.size(); ++i) {
    std::fill(std::next(result[i].begin(), offsets[i]), result[i].end(),
              empty_element);
  }
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

  for (int i = 0; i < inputs[0].size(); ++i) {
    for (const auto& input : inputs) {
      result.push_back(input[i]);
    }
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
}  // namespace

PipelineVisitor::PipelineVisitor(
    int64 stage_count, const std::vector<int>& stage_ipu_mapping,
    const absl::flat_hash_map<const HloInstruction*, int>& inst_stage_mapping,
    CompilerResources& res, const ArgVectors& inputs,
    const std::vector<const SubComputationVisitor*>& dependent_subcomputations)
    : InplaceSubComputationVisitor(res, inputs, dependent_subcomputations),
      copy_sequences_(stage_count),
      fifo_sequences_(stage_count),
      program_sequences_(stage_count),
      stage_ipu_mapping_(stage_ipu_mapping),
      inst_stage_mapping_(inst_stage_mapping) {}

PipelineVisitor::PipelineVisitor(
    const HloInstruction* pipeline, CompilerResources& res,
    const ArgVectors& inputs,
    const std::vector<const SubComputationVisitor*>& dependent_subcomputations)
    : PipelineVisitor(GetPipelineStageCount(pipeline),
                      GetPipelineStageDeviceMapping(pipeline),
                      GetPipelineInstStageMapping(pipeline), res, inputs,
                      dependent_subcomputations) {}

StatusOr<poplar::program::Sequence> PipelineVisitor::GetPipelineSequence(
    int64 iterations) const {
  const auto overlap_length = CircularUnion(stage_ipu_mapping_).size();
  if (iterations % overlap_length) {
    // TODO(T11404)
    return FailedPrecondition(
        "The number of iterations of the pipeline must be a multiple of %d.",
        overlap_length);
  }
  // To account for ramp up and ramp down we need at least overlap_length * 2
  // iterations.
  if (iterations < overlap_length * 2) {
    return FailedPrecondition(
        "The number of iterations of the pipeline must be at least %d.",
        overlap_length * 2);
  }
  poplar::program::Program ramp_up = GetPipelineRampUpSequence();
  poplar::program::Program repeat_block = GetPipelineRepeatBlockSequence();

  poplar::program::Sequence program;

  poplar::program::Program ramp_down =
      GetPipelineRampDownSequence(iterations % overlap_length);

  program.add(ramp_up);
  program.add(
      poplar::program::Repeat((iterations / overlap_length) - 1, repeat_block));
  program.add(ramp_down);

  return program;
}

// Collect the pipeline stage programs and call CreateRampSequences
poplar::program::Program PipelineVisitor::GetPipelineRampUpSequence() const {
  // Find the set of non-overlapping program offsets.
  auto offsets = CircularUnion(stage_ipu_mapping_);

  // Build schedules for the compute and copy programs.
  // Each schedule is 2D, where each column represents a time-slice and each row
  // represents the "mini-batch",
  auto program_sequences = ConstructRampUpSchedule(offsets, program_sequences_);
  auto copy_sequences = ConstructRampUpSchedule(offsets, copy_sequences_);
  auto fifo_sequences = ConstructRampUpSchedule(offsets, fifo_sequences_);

  // Concatenate the compute, copy and fifo programs.
  program_sequences.insert(program_sequences.end(), copy_sequences.begin(),
                           copy_sequences.end());
  program_sequences.insert(program_sequences.end(), fifo_sequences.begin(),
                           fifo_sequences.end());

  // Flatten the schedule to a linear sequence.
  auto repeat_block_sequences = FlattenSchedule(program_sequences);

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
  auto offsets = CircularUnion(stage_ipu_mapping_);
  // Build schedules for the compute and copy programs.
  // Each schedule is 2D, where each column represents a time-slice and each row
  // represents the "mini-batch",
  auto program_sequences = ConstructRampDownSchedule(
      offsets, program_sequences_, {}, additional_iterations);
  auto copy_sequences = ConstructRampDownSchedule(offsets, copy_sequences_, {},
                                                  additional_iterations);
  auto fifo_sequences = ConstructSchedule(offsets, fifo_sequences_);

  // Concatenate the compute, copy and fifo programs.
  program_sequences.insert(program_sequences.end(), copy_sequences.begin(),
                           copy_sequences.end());
  program_sequences.insert(program_sequences.end(), fifo_sequences.begin(),
                           fifo_sequences.end());

  // Flatten the schedule to a linear sequence.
  auto repeat_block_sequences = FlattenSchedule(program_sequences);

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
  auto offsets = CircularUnion(stage_ipu_mapping_);
  // Build schedules for the compute and copy programs.
  // Each schedule is 2D, where each column represents a time-slice and each row
  // represents the "mini-batch",
  auto program_sequences = ConstructSchedule(offsets, program_sequences_);
  auto copy_sequences = ConstructSchedule(offsets, copy_sequences_);
  auto fifo_sequences = ConstructSchedule(offsets, fifo_sequences_);

  // Concatenate the compute, copy and fifo programs.
  program_sequences.insert(program_sequences.end(), copy_sequences.begin(),
                           copy_sequences.end());
  program_sequences.insert(program_sequences.end(), fifo_sequences.begin(),
                           fifo_sequences.end());

  // Flatten the schedule to a linear sequence.
  auto repeat_block_sequences = FlattenSchedule(program_sequences);

  poplar::program::Sequence repeat_block;
  for (const auto& seq : repeat_block_sequences) {
    repeat_block.add(seq);
  }

  return repeat_block;
}

Status PipelineVisitor::HandleNotImplemented(HloInstruction* hlo) {
  return xla::Unimplemented(
      "%s (%s) is not a valid pipeline stage hlo instruction",
      hlo->name().c_str(), HloOpcodeString(hlo->opcode()).c_str());
}

Status PipelineVisitor::HandleCall(HloInstruction* hlo) {
  HloComputation* comp = hlo->to_apply();
  VLOG(1) << "Processing " << hlo->name() << " : " << comp->name()
          << " as a pipeline stage";

  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, hlo));
  TF_ASSIGN_OR_RETURN(poplar::program::Program prog,
                      CreateCallOp(resources_, hlo, hlo->shape(), tensor_map));

  program_sequences_[stage].add(prog);

  return Status::OK();
}

Status PipelineVisitor::HandleCustomCall(HloInstruction* hlo) {
  if (!IsPoplibsHloCustomOp(hlo)) {
    return HandleNotImplemented(hlo);
  }

  const bool is_inter_ipu_copy_hlo =
      hlo->custom_call_target() ==
      GetPoplibsCustomOpTargetString(PoplibsOp::Poputil,
                                     PoplibsOp::IpuInterCopy);

  if (is_inter_ipu_copy_hlo) {
    return HandleInterIpuCopy(hlo);
  }

  const bool is_fifo_hlo =
      hlo->custom_call_target() ==
      GetPoplibsCustomOpTargetString(PoplibsOp::Poputil, PoplibsOp::Fifo);

  if (is_fifo_hlo) {
    return HandleFifo(hlo);
  }

  return HandleNotImplemented(hlo);
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

  copy_sequences_[stage].add(prog);

  return Status::OK();
}

Status PipelineVisitor::HandleGetTupleElement(HloInstruction* hlo) {
  VLOG(1) << "Processing " << hlo->name();

  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, hlo));
  TF_ASSIGN_OR_RETURN(
      ArgVectors output_tensors,
      FindInplaceOutputTensors(tensor_map, resources_, hlo,
                               program_sequences_[stage], false));
  CHECK_EQ(output_tensors.size(), 1);
  CHECK_EQ(output_tensors[0].size(), CountShapes(hlo->shape()));
  for (int64 i = 0; i < output_tensors[0].size(); i++) {
    TF_CHECK_OK(AddOutputTensor(tensor_map, hlo, i, output_tensors[0][i]));
  }
  return Status::OK();
}

Status PipelineVisitor::FinishVisit(HloInstruction* inst) {
  outputs_ = FindInstructionOutputs(tensor_map, inst);
  resources_.tensor_maps[inst->parent()->name()] = std::move(tensor_map);
  return Status::OK();
}

Status PipelineVisitor::HandleTuple(HloInstruction* hlo) {
  if (hlo->parent()->root_instruction() != hlo) {
    return FailedPrecondition(
        "Hlo tuple instructions are only allowed in a pipeline when they are "
        "the root instruction. Hlo instruction \"%s\" is not.",
        hlo->name());
  }

  VLOG(1) << "Processing " << hlo->name();

  // Tuple just forwards the input tensors.
  uint64 n = 0;
  for (int64 op_idx = 0; op_idx != hlo->operand_count(); ++op_idx) {
    const HloInstruction* operand = hlo->operand(op_idx);
    ArgVector inputs = FindInstructionOutputs(tensor_map, operand);
    CHECK_EQ(inputs.size(), CountShapes(operand->shape()));

    for (uint64 j = 0; j < inputs.size(); j++) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, hlo, n++, inputs[j]));
    }
  }

  return Status::OK();
}

poplar::program::Sequence& PipelineVisitor::GetSequenceForAliasingCopy(
    int64 flat_tensor_index, const HloComputation* computation) {
  const HloInstruction* root = computation->root_instruction();
  CHECK_EQ(root->operand_count(), computation->num_parameters());
  // Get the parameter for this input to the tuple.
  auto param_num_index = GetParameterNumberAndFlatIndex(flat_tensor_index);
  int64 param_number = param_num_index.first;

  // Get the stage of the input to the tuple.
  int64 stage =
      GetPipelineStage(inst_stage_mapping_, root->operand(param_number))
          .ValueOrDie();
  return copy_sequences_[stage];
}

}  // namespace poplarplugin
}  // namespace xla
