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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/recv_from_host.h"

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

class RecvFromHostOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;

    const auto* recv = Cast<HloRecvFromHostInstruction>(inst);

    poplar::Tensor dst_tensor;
    if (IsLoweredInplace(recv) && inst->operand_count() > 0) {
      CHECK_EQ(inst->operand_count(), 1);
      // Input provided and lowered in-place: Use the provided input tensor for
      // the output.
      TF_ASSIGN_OR_RETURN(dst_tensor,
                          FindInstructionInput(tensor_map, res, inst, 0, seq));
    } else {
      // Either:
      // 1. No input provided: Must allocate new tensor for output.
      // 2. Input provided but not in-place: Allocate new tensor for output
      //    (rather than copying the input tensor and using that) to get a
      //    layout desired by the consumer.
      TF_ASSIGN_OR_RETURN(dst_tensor, AddTensor(graph, TensorLocation{inst, 0},
                                                output_shape, res, tensor_map));
    }

    // Use the rendezvous key also for the Poplar stream handle.
    const poplar::DataStream src_stream = graph.addHostToDeviceFIFO(
        recv->RendezvousKey(), dst_tensor.elementType(),
        dst_tensor.numElements(), poplar::ReplicatedStreamMode::BROADCAST);

    seq.add(poplar::program::Copy(src_stream, dst_tensor,
                                  res.always_rearrange_copies_on_host));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, dst_tensor));

    res.annotations.recv_infos.emplace_back(
        src_stream.handle(), recv->RendezvousKey(), output_shape);

    return seq;
  }
};

REGISTER_POPLAR_OP(RecvFromHost, RecvFromHostOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
