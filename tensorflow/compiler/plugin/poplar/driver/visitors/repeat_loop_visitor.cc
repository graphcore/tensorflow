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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/repeat_loop_visitor.h"

#include <string>
#include <vector>

#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

namespace xla {
namespace poplarplugin {

RepeatLoopVisitor::RepeatLoopVisitor(
    CompilerResources& res, const DeferredArgVectors& inputs,
    const ReallocateInputsInfo& reallocate_inputs_info, const std::string& name)
    : InplaceDeferredVisitor(res, inputs, name, {}, reallocate_inputs_info) {
  // Push a new vector for the zeroing sequences onto the stack.
  res.gradient_accumulation_zeroing_sequences.push({});
}

Status RepeatLoopVisitor::HandleDeferredAllocationCall(HloInstruction* inst) {
  if (IsResourceUpdate(inst)) {
    if (has_resource_update_) {
      return FailedPrecondition(
          "Detected multiple resource update instructions inside of a training "
          "loop - only one resource update is allowed.");
    }

    has_resource_update_ = true;
    num_mini_batches_to_accumulate_ =
        GetResourceUpdateBatchesToAccumulate(inst);

    TF_ASSIGN_OR_RETURN(DeferredArgVectors inputs,
                        GetInputsForDeferredInplaceInstruction(
                            inst, /*preserve_aliasing*/ true));

    TF_ASSIGN_OR_RETURN(resource_update_sequence_,
                        CreateResourceUpdateOp(resources_, inst, inputs,
                                               inst->shape(), tensor_map));

  } else {
    return InplaceDeferredVisitor::HandleDeferredAllocationCall(inst);
  }
}

Status RepeatLoopVisitor::FinishDeferedAllocationVisit(HloInstruction* inst) {
  // Create the sequence which is only executed once before the loops is
  // executed.
  // Add any copies if the inputs were reallocated.
  TF_ASSIGN_OR_RETURN(pre_loop_sequence_, GetPreambleCopies());

  // Only add the copies for the execution counters in the body visitor once
  // before the execution of the loop so that they are not reset at the
  // beginning of each iteration.
  TF_RETURN_IF_ERROR(CopyExecutionCountersFromScope(
      resources_, execution_counters_, pre_loop_sequence_));

  // Create a sequence for all the zeroing gradient accumulation buffers.
  auto& zeroing_seqs = resources_.gradient_accumulation_zeroing_sequences.top();
  for (poplar::program::Sequence& zeroing_seq : zeroing_seqs) {
    tensors_zeroing_sequence_.add(zeroing_seq);
  }
  resources_.gradient_accumulation_zeroing_sequences.pop();

  // Add the aliasing copies for the loop so that the outputs of one iteration
  // are aliased to the inputs of the next one.
  poplar::Graph& graph = GetGraph(resources_, inst);
  TF_ASSIGN_OR_RETURN(loop_state_, AddLoopInputOutputAliasingCopies(
                                       graph, inst->parent(), name_));

  return Status::OK();
}

StatusOr<poplar::program::Sequence*>
RepeatLoopVisitor::GetSequenceForInstruction(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kGetTupleElement: {
      return IsResourceUpdate(inst->operand(0)) ? &resource_update_sequence_
                                                : &sequence;
    }
    case HloOpcode::kTuple: {
      return has_resource_update_ && inst->parent()->root_instruction() == inst
                 ? &resource_update_sequence_
                 : &sequence;
    }
    default: { return InplaceDeferredVisitor::GetSequenceForInstruction(inst); }
  }
}

poplar::program::Sequence& RepeatLoopVisitor::GetSequenceForAliasingCopy() {
  return has_resource_update_ ? resource_update_sequence_ : sequence;
}

poplar::program::Sequence RepeatLoopVisitor::GetRepeatLoopSequence(
    const HloInstruction* inst) {
  const int64 repeat_count = GetRepeatLoopCount(inst);

  poplar::program::Sequence seq;
  seq.add(pre_loop_sequence_);

  poplar::program::Sequence repeat_seq;
  {
    repeat_seq.add(GetSequence(/*copy_execution_counters*/ false));
    // Increase the local execution counters at the end of each iteration.
    repeat_seq.add(execution_counters_.IncrementLiveCounters());
  }

  if (has_resource_update_) {
    CHECK_GT(num_mini_batches_to_accumulate_, 0);
    CHECK_EQ(repeat_count % num_mini_batches_to_accumulate_, 0);
    // Create a double loop - the inner loop executes for
    // `num_mini_batches_to_accumulate_` iterations and then performs the
    // resource update.
    poplar::program::Sequence inner_seq;
    // Zero the gradient accumulation buffers.
    inner_seq.add(tensors_zeroing_sequence_);
    inner_seq.add(
        poplar::program::Repeat(num_mini_batches_to_accumulate_, repeat_seq));
    inner_seq.add(resource_update_sequence_);

    // Repeat the inner loop.
    seq.add(poplar::program::Repeat(
        repeat_count / num_mini_batches_to_accumulate_, inner_seq));
  } else {
    seq.add(poplar::program::Repeat(repeat_count, repeat_seq));
  }
  return seq;
}

const TensorVector& RepeatLoopVisitor::GetLoopState() const {
  return loop_state_;
}

}  // namespace poplarplugin
}  // namespace xla
