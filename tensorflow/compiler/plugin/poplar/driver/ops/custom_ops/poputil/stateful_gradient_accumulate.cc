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

#include <poplar/Program.hpp>
#include <popops/Collectives.hpp>
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
    res.zeroed_tensors.push_back(counter);

    poplar::Tensor accumulator =
        graph.clone(input, GetDebugName(inst) + "/Accumulator");
    res.zeroed_tensors.push_back(accumulator);
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
        popops::replicatedAllReduceWithOutput(
            GetMasterGraph(res), accumulator, output, popops::Operation::ADD,
            if_true, GetDebugName(inst), GetReplicateAllReduceOptions());
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

class PipelineStatefulGradientAccumulateOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    const std::string debug_name = GetDebugName(inst);

    // Create a sequence for zeroing the accumulator before the pipeline is
    // executed.
    poplar::program::Sequence zeroing_seq;

    // Create a sequence to be executed during the pipeline.
    poplar::program::Sequence seq;
    TensorVector inputs = FindInstructionInputs(tensor_map, res, inst, 0, seq);

    // Clone the inputs into accumulators which are the outputs.
    TensorVector accumulators(inputs.size());
    absl::c_transform(inputs, accumulators.begin(),
                      [&graph, &debug_name](const poplar::Tensor& in) {
                        return graph.clone(in, debug_name + "/Accumulator");
                      });

    for (size_t i = 0; i != inputs.size(); ++i) {
      // Add the initial zeroing.
      popops::zero(graph, accumulators[i], zeroing_seq,
                   debug_name + "/ZeroAccumulator");
      // Add input to the accumulator.
      popops::addInPlace(graph, accumulators[i], inputs[i], seq,
                         debug_name + "/Accumulate");
    }

    // Add the zeroing sequence.
    if (res.pipelining_buffer_zeroing_sequences.empty()) {
      return FailedPrecondition(
          "Cannot zero Pipeline's gradient accumulators.");
    }
    res.pipelining_buffer_zeroing_sequences.top().push_back(zeroing_seq);

    for (size_t i = 0; i != accumulators.size(); ++i) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, accumulators[i]));
    }

    return seq;
  }
};
REGISTER_POPLAR_OP(PipelineStatefulGradientAccumulate,
                   PipelineStatefulGradientAccumulateOp);

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
    TensorVector outputs = FindInstructionOutputs(
        tensor_map, res, inst->operand(num_grads + accumulator_index));

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
    res.zeroed_tensors.push_back(counter);

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
          popops::replicatedAllReduceWithOutput(
              GetMasterGraph(res), accumulator, output, popops::Operation::ADD,
              if_true, GetDebugName(inst), GetReplicateAllReduceOptions());

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
};  // namespace
REGISTER_POPLAR_OP(StatefulGradientAccumulateWithMomentum,
                   StatefulGradientAccumulateWithMomentumOp);
REGISTER_POPLAR_OP(StatefulGradientAccumulateWithMomentumAndAllReduceWithNorm,
                   StatefulGradientAccumulateWithMomentumOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
