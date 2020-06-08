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

RepeatLoopVisitor::RepeatLoopVisitor(CompilerResources& res,
                                     const DeferredArgVectors& inputs,
                                     bool reallocate_inputs,
                                     const std::string& name)
    : InplaceDeferredVisitor(res, inputs, name, {}, reallocate_inputs) {}

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

  // Add the aliasing copies for the loop so that the outputs of one iteration
  // are aliased to the inputs of the next one.
  poplar::Graph& graph = GetGraph(resources_, inst);
  TF_ASSIGN_OR_RETURN(loop_state_, AddLoopInputOutputAliasingCopies(
                                       graph, inst->parent(), name_));

  return Status::OK();
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
  seq.add(poplar::program::Repeat(repeat_count, repeat_seq));
  return seq;
}

const TensorVector& RepeatLoopVisitor::GetLoopState() const {
  return loop_state_;
}
}  // namespace poplarplugin
}  // namespace xla
