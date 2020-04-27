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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_stage_visitor.h"

#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

namespace xla {
namespace poplarplugin {

PipelineStageVisitor::PipelineStageVisitor(CompilerResources& res,
                                           const DeferredArgVectors& inputs)
    : InplaceDeferredVisitor(res, inputs) {}

bool PipelineStageVisitor::TupleOutputsNeedToPreserveAliasing(
    const HloInstruction* inst) {
  // We preserve aliasing when this is the output tuple.
  return inst->parent()->root_instruction() == inst;
}

poplar::program::Sequence PipelineStageVisitor::GetSequence() const {
  if (!has_function_) {
    poplar::program::Sequence seq = InplaceDeferredVisitor::GetSequence();
    function_ = GetMasterGraph(resources_).addFunction(seq);
  }
  has_function_ = true;

  return poplar::program::Sequence(poplar::program::Call(function_));
}

ShapeTree<bool> PipelineStageVisitor::GetOutputCopies(
    const HloInstruction* inst) const {
  ShapeTree<bool> output_tree(inst->shape(), true);
  // Don't add copies for any outputs used by the gradient accumulation sinks.
  for (HloInstruction* user : inst->users()) {
    CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
    // If the user is a sink.
    if (user->user_count() == 1 &&
        IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(
            user->users()[0])) {
      ShapeTree<bool> user_tree(user->shape(), false);
      output_tree.CopySubtreeFrom(user_tree, {}, {user->tuple_index()});
    }
  }
  return output_tree;
}

ReusablePipelineStageVisitor::ReusablePipelineStageVisitor(
    CompilerResources& res, const DeferredArgVectors& inputs)
    : PipelineStageVisitor(res, inputs) {}

Status ReusablePipelineStageVisitor::PropagateDeferredAllocations(
    const HloInstruction* callsite_inst) {
  std::vector<bool> add_clones(callsite_inst->operand_count(), false);

  // Mark all the inplace operands as requiring clones so that the input tensors
  // are not clobbered when reusing the sequence.
  auto inst_description = HloInstructionDescription(callsite_inst);
  for (int64 inplace_idx : inst_description.GetInplaceOperandIndexes()) {
    add_clones[inplace_idx] = true;
  }

  return DeferredVisitor::PropagateDeferredAllocations(callsite_inst,
                                                       add_clones);
}

poplar::program::Sequence ReusablePipelineStageVisitor::GetSequence(
    const HloInstruction* callsite, const DeferredArgVectors& deferred_inputs,
    TensorMap& callsite_tensor_map) const {
  poplar::program::Sequence seq;
  // Convert deferred args to actual tensors, filling gaps where required.
  CHECK_EQ(callsite->operand_count(), deferred_inputs.size());
  TensorVectors inputs(deferred_inputs.size());
  for (uint64 operand_idx = 0; operand_idx != deferred_inputs.size();
       ++operand_idx) {
    const uint64 num_tensors = deferred_inputs[operand_idx].size();
    inputs[operand_idx].resize(num_tensors);
    for (uint64 flat_idx = 0; flat_idx != num_tensors; ++flat_idx) {
      if (deferred_inputs[operand_idx][flat_idx]) {
        // We have a tensor already, forward it.
        inputs[operand_idx][flat_idx] = *deferred_inputs[operand_idx][flat_idx];
      } else {
        // The tensor had a deferred allocation, so we get it now after the
        // stage has been built.
        auto input = FindInstructionInputsInRange(
            callsite_tensor_map, resources_, callsite, operand_idx,
            {flat_idx, flat_idx + 1}, seq, false);
        CHECK_EQ(input.size(), 1);
        // We do not need to check whether this input is used inplace, because
        // we add copies for all inplace inputs in a reusable pipeline stage
        // anyway.
        inputs[operand_idx][flat_idx] = input[0];
      }
    }
  }
  seq.add(GetSequence(callsite, inputs));
  return seq;
}

poplar::program::Sequence ReusablePipelineStageVisitor::GetSequence(
    const HloInstruction* callsite, const TensorVectors& inputs) const {
  poplar::Graph& graph = GetGraph(resources_, callsite);
  // When recomputation is enabled, copies need to be inserted for all the non
  // parameter inputs as we are re-using the forward stage Poplar
  // Sequence/visitor for both the forward and recomputation stages. Note that
  // we do not add copies for parameters as these are always the same/are not
  // modified. Note that since we are adding these copies, the FIFO instructions
  // can be executed after the PipelineStage and before the
  // PipelineStageRecomputation since the values won't be modified inplace.
  poplar::program::Sequence seq;

  auto inst_description = HloInstructionDescription(callsite);
  // For each inplace operand, go through all the tensors for that operand and
  // add copies from the instruction input tensors to the visitor input tensors
  // (preserving the aliasing).
  for (int64 inplace_idx : inst_description.GetInplaceOperandIndexes()) {
    // If inplace_idx is out of bounds then it's a state and therefore no copy
    // is needed.
    if (inplace_idx < static_cast<int64>(computation_inputs_.size())) {
      CHECK_EQ(inputs[inplace_idx].size(),
               computation_inputs_[inplace_idx].size());
      for (size_t flat_idx = 0; flat_idx != inputs[inplace_idx].size();
           ++flat_idx) {
        seq.add(
            TensorCopyWithAliasing(graph, inputs[inplace_idx][flat_idx],
                                   computation_inputs_[inplace_idx][flat_idx]));
      }
    }
  }
  // Add the actual sequence for the stage.
  seq.add(PipelineStageVisitor::GetSequence());
  return seq;
}

ShapeTree<bool> ReusablePipelineStageVisitor::GetOutputCopies(
    const HloInstruction* inst) const {
  // Reusable stages need a copy on all their outputs.
  ShapeTree<bool> output_tree(inst->shape(), true);
  return output_tree;
}

}  // namespace poplarplugin
}  // namespace xla
