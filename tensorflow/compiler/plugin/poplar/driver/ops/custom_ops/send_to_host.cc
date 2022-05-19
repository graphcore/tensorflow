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
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

class SendToHostOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "SendToHostOp");
    DriverProgramSequence seq(graph, debug_info);

    const auto* send = Cast<HloSendToHostInstruction>(inst);

    const int64_t num_inputs = send->operand_count();

    // As long as the stream copies are scheduled right after each other,
    // Poplar will attempt to merge them according to `opt.maxCopyMergeSize`.
    for (int64_t i = 0; i < num_inputs; ++i) {
      TF_ASSIGN_OR_RETURN(
          const poplar::Tensor tensor,
          FindInstructionInput(tensor_map, res, inst, i, seq, debug_info));

      const std::string& rendezvous_key = send->RendezvousKeys()[i];

      // Use the rendezvous key also for the Poplar stream handle.
      const auto stream = graph.addDeviceToHostFIFO(
          rendezvous_key, tensor.elementType(), tensor.numElements());

      seq.add(poplar::program::Copy(
          tensor, stream, res.always_rearrange_copies_on_host, {debug_info}));

      const Shape& shape = inst->operand(i)->shape();
      res.annotations.send_infos.emplace_back(stream.handle(), rendezvous_key,
                                              shape);
    }

    return seq;
  }

  StatusOr<poplar::Tensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "SendToHostOp");
    const int64_t input_index = tensor_target.input_index;
    const Shape& input_shape = tensor_target.tgt->operand(input_index)->shape();
    return AddHostCopyTensor(graph, {debug_info}, input_shape);
  }
};

REGISTER_POPLAR_OP(SendToHost, SendToHostOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
