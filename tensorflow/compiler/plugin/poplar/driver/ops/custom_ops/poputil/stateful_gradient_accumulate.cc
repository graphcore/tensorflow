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
    // Unconcat the result and unflatten.
    auto output_tensors = SliceTensorIntoTensorsLike(output, input_tensors);
    for (size_t i = 0; i != output_tensors.size(); ++i) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, output_tensors[i]));
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
    TensorVector inputs =
        FindInstructionInputs(tensor_map, res, inst, 0, seq, false);

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

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
