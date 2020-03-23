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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/send_to_host.h"

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

class SendToHostOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;

    const auto* send = Cast<HloSendToHostInstruction>(inst);

    const int64 num_inputs = send->operand_count();

    // As long as the stream copies are scheduled right after each other,
    // Poplar will attempt to merge them according to `opt.maxCopyMergeSize`.
    for (int64 i = 0; i < num_inputs; ++i) {
      TF_ASSIGN_OR_RETURN(const poplar::Tensor tensor,
                          FindInstructionInput(tensor_map, res, inst, i, seq));

      const std::string& rendezvous_key = send->RendezvousKeys()[i];

      // Use the rendezvous key also for the Poplar stream handle.
      const poplar::DataStream stream = graph.addDeviceToHostFIFO(
          rendezvous_key, tensor.elementType(), tensor.numElements());

      seq.add(poplar::program::Copy(tensor, stream,
                                    res.always_rearrange_copies_on_host));

      const Shape& shape = inst->operand(0)->shape();
      res.annotations.send_infos.emplace_back(stream.handle(), rendezvous_key,
                                              shape, send->ConcatReplicas());
    }

    return seq;
  }
};

REGISTER_POPLAR_OP(SendToHost, SendToHostOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
