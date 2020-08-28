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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"

#include <gcl/Collectives.hpp>
#include <poplar/Program.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Zero.hpp>
#include <poputil/Broadcast.hpp>
#include <poputil/Util.hpp>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {
namespace {

class StatefulGradientAccumulateOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;

    const HloStatefulGradientAccumulate* grad_inst =
        Cast<HloStatefulGradientAccumulate>(inst);

    const bool do_all_reduce =
        IsPoplarInstruction(PoplarOp::StatefulGradientAccumulateAndAllReduce)(
            inst) &&
        res.replication_factor > 1;

    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, seq, false));
    CHECK_EQ(inputs.size(), inst->operand_count());
    std::vector<poplar::Tensor> input_tensors(inst->operand_count());
    for (size_t i = 0; i < inputs.size(); ++i) {
      CHECK_EQ(inputs[i].size(), 1);
      input_tensors[i] = inputs[i][0];
    }
    // Create a concatenated and flattened tensor of the input tensors.
    poplar::Tensor input = FlattenAndConcatenateTensors(input_tensors);
    poplar::Tensor counter = graph.addVariable(poplar::UNSIGNED_INT, {},
                                               GetDebugName(inst) + "/Counter");
    // Map counter to the next tile.
    MappingHelper::MapTensorLinearly(res.linear_mapping_state, graph, counter);
    AddZeroTensorToPreamble(res, counter);

    poplar::Tensor accumulator =
        graph.clone(input, GetDebugName(inst) + "/Accumulator");
    AddZeroTensorToPreamble(res, accumulator);
    // Accumulate the input into the buffer.
    popops::addInPlace(graph, accumulator, input, seq,
                       GetDebugName(inst) + "/Accumulate");

    // Output the accumulated gradients if counter == MiniBatchesToAccumulate -
    // 1 otherwise output all zeros.
    poplar::Tensor output_grads = popops::map(
        graph,
        pe::Equal(pe::_1, pe::Const(grad_inst->MiniBatchesToAccumulate() - 1)),
        {counter}, seq, GetDebugName(inst) + "/CheckOutputGradients");

    poplar::Tensor output = input;
    poplar::program::Sequence if_true;
    {
      if (do_all_reduce) {
        // All reduce the accumulator tensor into the output.
        gcl::allReduceToDestination(
            GetMasterGraph(res), accumulator, output, popops::Operation::ADD,
            if_true, GetDebugName(inst), GetReplicateAllReduceOptions(res));
      } else {
        // Copy accumulator into output.
        if_true.add(poplar::program::Copy(accumulator, output));
      }

      // Zero the accumulator.
      popops::zero(graph, accumulator, if_true,
                   GetDebugName(inst) + "/ZeroAccumulator");
      // Zero the counter.
      popops::zero(graph, counter, if_true,
                   GetDebugName(inst) + "/ZeroCounter");
    }
    poplar::program::Sequence if_false;
    {
      // Set output to all zeros.
      popops::zero(graph, output, if_false, GetDebugName(inst) + "/ZeroOutput");
      // Increase counter.
      popops::mapInPlace(graph, pe::Add(pe::_1, pe::Const(1)), {counter},
                         if_false, GetDebugName(inst) + "/IncreaseCounter");
    }
    seq.add(poplar::program::If(output_grads, if_true, if_false));

    // This op is completely inplace, so just set the input tensors to outputs.
    for (size_t i = 0; i < inputs.size(); ++i) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, input_tensors[i]));
    }

    return seq;
  }
};
REGISTER_POPLAR_OP(StatefulGradientAccumulate, StatefulGradientAccumulateOp);
REGISTER_POPLAR_OP(StatefulGradientAccumulateAndAllReduce,
                   StatefulGradientAccumulateOp);

class StatefulGradientAccumulateWithMomentumOp : public PoplarOpDef {
  StatusOr<poplar::Tensor> Allocator(poplar::Graph& graph,
                                     CompilerResources& res,
                                     const std::string& name,
                                     const TensorTarget& tensor_target,
                                     const TensorMap& tensor_map) override {
    const HloInstruction* inst = tensor_target.tgt;
    const int64 accumulator_index = tensor_target.input_index;
    // Expect to allocate for the accumulator.
    const uint64 num_grads = inst->operand_count() / 2;
    if (accumulator_index > num_grads) {
      return xla::FailedPrecondition(
          "Trying to allocate StatefulGradientAccumulateWithMomentumOp tensor "
          "for an index out of range (%d >= %d).",
          accumulator_index, num_grads);
    }
    TF_ASSIGN_OR_RETURN(
        TensorVector outputs,
        FindInstructionOutputTensors(
            tensor_map, res, inst->operand(num_grads + accumulator_index)));

    if (outputs.size() != 1) {
      return xla::FailedPrecondition("Could not find layout input for %s",
                                     GetDebugName(inst));
    }
    return graph.clone(outputs[0], GetDebugName(inst));
  }

  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    const HloStatefulGradientAccumulate* grad_inst =
        Cast<HloStatefulGradientAccumulate>(inst);

    const bool do_all_reduce_and_norm =
        IsPoplarInstruction(
            PoplarOp::
                StatefulGradientAccumulateWithMomentumAndAllReduceWithNorm)(
            inst) &&
        res.replication_factor > 1;

    poplar::program::Sequence seq;

    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, seq, false));
    CHECK_EQ(inputs.size(), inst->operand_count() - 1);

    const uint64 num_grads = inst->operand_count() / 2;

    // Combine all the accumulator tensors and gradient tensors into a gradient
    // and accumulation tensor.
    std::vector<poplar::Tensor> accumulator_tensors(num_grads);
    std::vector<poplar::Tensor> grad_tensors(num_grads);
    for (uint64 i = 0; i != num_grads; ++i) {
      CHECK_EQ(inputs[i].size(), 1);
      accumulator_tensors[i] = inputs[i][0];

      CHECK_EQ(inputs[num_grads + i].size(), 1);
      grad_tensors[i] = inputs[num_grads + i][0];
    }
    poplar::Tensor accumulator =
        FlattenAndConcatenateTensors(accumulator_tensors);
    poplar::Tensor grad = FlattenAndConcatenateTensors(grad_tensors);

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor momentum,
        FindInstructionInput(tensor_map, res, inst, inst->operand_count() - 1,
                             seq, false));

    poplar::Tensor counter = graph.addVariable(poplar::UNSIGNED_INT, {},
                                               GetDebugName(inst) + "/Counter");
    // Map counter to the next tile.
    MappingHelper::MapTensorLinearly(res.linear_mapping_state, graph, counter);
    AddZeroTensorToPreamble(res, counter);

    // Apply momentum to the accumulator when counter == 0.
    {
      poplar::Tensor apply_momentum =
          popops::map(graph, pe::Equal(pe::_1, pe::Const(0)), {counter}, seq,
                      GetDebugName(inst) + "/CheckApplyMomentum");
      poplar::program::Sequence if_true;
      {
        // Apply the momentum.
        popops::mulInPlace(graph, accumulator, momentum, if_true,
                           GetDebugName(inst) + "/ApplyMomentum");
      }
      // Do nothing in false case.
      poplar::program::Sequence if_false;
      seq.add(poplar::program::If(apply_momentum, if_true, if_false));
    }

    // Add the gradient.
    popops::addInPlace(graph, accumulator, grad, seq,
                       GetDebugName(inst) + "/MomentumAddGrad");

    poplar::Tensor output = grad;

    // Output the accumulated gradients if counter == MiniBatchesToAccumulate -
    // 1 otherwise output all zeros.
    {
      poplar::Tensor output_grads = popops::map(
          graph,
          pe::Equal(pe::_1,
                    pe::Const(grad_inst->MiniBatchesToAccumulate() - 1)),
          {counter}, seq, GetDebugName(inst) + "/CheckOutputGradients");

      poplar::program::Sequence if_true;
      {
        if (do_all_reduce_and_norm) {
          // All reduce the accumulator tensor into the output.
          gcl::allReduceToDestination(
              GetMasterGraph(res), accumulator, output, popops::Operation::ADD,
              if_true, GetDebugName(inst), GetReplicateAllReduceOptions(res));

          // Normalize it - we normalize after the all reduce otherwise we risk
          // the gradients becoming zeros.
          popops::mapInPlace(
              graph, pe::Divide(pe::_1, pe::Const(res.replication_factor)),
              {output}, if_true, GetDebugName(inst) + "/NormalizeAccumulator");

          // Copy the normalized output into accumulator.
          if_true.add(poplar::program::Copy(output, accumulator));

        } else {
          // No all reduce (and therefore no norm) - just copy the accumulator
          // into output.
          if_true.add(poplar::program::Copy(accumulator, output));
        }

        // Zero the counter.
        popops::zero(graph, counter, if_true,
                     GetDebugName(inst) + "/ZeroCounter");
      }
      poplar::program::Sequence if_false;
      {
        // Set output to all zeros.
        popops::zero(graph, output, if_false,
                     GetDebugName(inst) + "/ZeroOutput");
        // Increase counter.
        popops::mapInPlace(graph, pe::Add(pe::_1, pe::Const(1)), {counter},
                           if_false, GetDebugName(inst) + "/IncreaseCounter");
      }
      seq.add(poplar::program::If(output_grads, if_true, if_false));
    }

    // This op is completely inplace, so just set the input tensors to outputs.
    for (uint64 i = 0; i != num_grads; ++i) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, accumulator_tensors[i]));
      TF_CHECK_OK(
          AddOutputTensor(tensor_map, inst, num_grads + i, grad_tensors[i]));
    }

    return seq;
  }
};
REGISTER_POPLAR_OP(StatefulGradientAccumulateWithMomentum,
                   StatefulGradientAccumulateWithMomentumOp);
REGISTER_POPLAR_OP(StatefulGradientAccumulateWithMomentumAndAllReduceWithNorm,
                   StatefulGradientAccumulateWithMomentumOp);

// A gradient accumulation creation op creates a gradient accumulation buffer
// which can be used by multiple pipeline stages on the same IPU.
// It is however handeled by the deferred allocation visitor.
class GradientAccumulatorCreateOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    return InternalErrorStrCat(
        "Instruction ", inst->name(),
        " should have been allocated by the DeferredVisitor. This error is "
        "most likely caused by inappropriate use of gradient accumulation. "
        "Please use the Pipelinining API.");
  }
};
REGISTER_POPLAR_OP(GradientAccumulatorCreate, GradientAccumulatorCreateOp);

// A gradient accumulation sink combines accumulators from different pipeline
// stages on the same IPU into a single buffer.
class GradientAccumulatorSinkOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    if (!IsLoweredInplace(inst)) {
      return InternalErrorStrCat("Expected the instruction ", inst->name(),
                                 " to have been lowered inplace.");
    }
    poplar::program::Sequence seq;
    TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                        FindInplaceOutputTensors(tensor_map, res, inst, seq));
    CHECK_EQ(inputs.size(), inst->operand_count());
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor output = inputs[0][0];

    // Make sure that all the merged gradient accumulation buffers are the same
    // location.
    for (uint64 i = 1; i != inputs.size(); ++i) {
      CHECK_EQ(inputs[i].size(), 1);
      if (inputs[i][0] != output) {
        return InternalErrorStrCat(
            "Expected all the gradient accumulation buffers for instruction ",
            inst->name(), " to match.");
      }
    }

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));
    return seq;
  }
};
REGISTER_POPLAR_OP(GradientAccumulatorSink, GradientAccumulatorSinkOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
