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
#include <gcl/Collectives.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Pad.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

int64 PerReplicaLength(int64 length, int64 num_replicas) {
  return tensorflow::MathUtil::CeilOfRatio(length, num_replicas);
}

poplar::Tensor InterleavePerReplica(poplar::Graph& graph,
                                    const std::vector<poplar::Tensor>& inputs,
                                    uint32 num_replicas) {
  std::vector<poplar::Tensor> interleaved;

  // Interleave padded input slices in a replica-major order.
  for (uint32 replica_id = 0; replica_id < num_replicas; ++replica_id) {
    for (const poplar::Tensor& input : inputs) {
      const int64 total_length = input.numElements();
      const int64 replica_length = PerReplicaLength(total_length, num_replicas);
      const int64 offset = std::min(total_length, replica_id * replica_length);
      const int64 remaining = total_length - offset;
      const int64 actual_length = std::min(replica_length, remaining);
      const int64 padding = replica_length - actual_length;

      poplar::Tensor slice = input.slice(offset, offset + actual_length);
      poplar::Tensor padded_slice = popops::pad(graph, slice, {0}, {padding});
      interleaved.push_back(padded_slice);
    }
  }

  poplar::Tensor result = poplar::concat(interleaved);
  CHECK_EQ(result.numElements() % num_replicas, 0);
  return result;
}

class ReduceScatterOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;

    // Collect all the inputs.
    const int64 num_inputs = inst->operand_count();
    std::vector<poplar::Tensor> inputs(num_inputs);
    for (int64 i = 0; i < num_inputs; ++i) {
      TF_ASSIGN_OR_RETURN(inputs[i],
                          FindInstructionInput(tensor_map, res, inst, i, seq));
      // The op requires rank-1 input.
      CHECK_EQ(inputs[i].rank(), 1);
    }

    // Interleave inputs per replica such that each replica gets its own chunk
    // of the scattered result for each input (with necessary padding applied).
    // Example with two replicas:
    // Before: inputs = [[a0, a1, a2], [b0]] (per replica)
    // After: interleaved_input = [a0, a1, b0, a2, 0, 0] (per replica)
    poplar::Tensor interleaved_input =
        InterleavePerReplica(graph, inputs, res.replication_factor);

    // Do the actual reduce scatter on the interleaved input.
    poplar::Tensor interleaved_output =
        gcl::reduceScatter(graph, interleaved_input, popops::Operation::ADD,
                           seq, GetDebugName(inst) + "/ReduceScatter",
                           GetReplicatedCollectiveOptions(res));

    // Deinterleave output.
    // Before: interleaved_output = [sum(a0), sum(a1), sum(b0)] (on replica 1/2)
    // Before: interleaved_output = [sum(a2), sum(0),  sum(0)]  (on replica 2/2)
    //   (where sum(x) denotes a cross-replica sum of x)
    // After: outputs = [[sum(a0), sum(a1)], [sum(b0)]] (on replica 1/2)
    // After: outputs = [[sum(a2), sum(0)],  [sum(0)]]  (on replica 2/2)
    int64 output_offset = 0;
    for (int64 i = 0; i < num_inputs; ++i) {
      const int64 replica_length =
          PerReplicaLength(inputs[i].numElements(), res.replication_factor);

      poplar::Tensor output = interleaved_output.slice(
          output_offset, output_offset + replica_length);

      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, output));
      output_offset += replica_length;
    }

    // Ensure all the output has been consumed.
    CHECK_EQ(output_offset, interleaved_output.numElements());

    return seq;
  }
};

REGISTER_POPLAR_OP(ReduceScatter, ReduceScatterOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
