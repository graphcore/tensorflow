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
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

#include <poplar/DebugContext.hpp>

namespace xla {
namespace poplarplugin {
namespace {

std::vector<xla::Shape> GetOutputShapes(const xla::Shape& output_shape,
                                        int64 num_outputs) {
  if (num_outputs > 1) {
    CHECK(output_shape.IsTuple());
    return output_shape.tuple_shapes();
  }

  CHECK(output_shape.IsArray());
  return {output_shape};
}

StatusOr<TensorVector> GetOutputTensors(
    DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
    const std::vector<xla::Shape>& output_shapes, int64 num_outputs,
    TensorMap& tensor_map, poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id) {
  TensorVector result(num_outputs);

  // If inputs provided and lowered in-place, use the input tensors.
  if (inst->operand_count() > 0 && IsLoweredInplace(inst)) {
    TF_ASSIGN_OR_RETURN(TensorVectors nested,
                        FindInplaceOutputTensors(tensor_map, res, inst, seq,
                                                 debug_name_and_id));
    CHECK_EQ(nested.size(), num_outputs);
    for (int64 i = 0; i < num_outputs; ++i) {
      CHECK_EQ(nested[i].size(), 1);
      result[i] = nested[i][0];
    }
  } else {
    // Otherwise, allocate new tensors with layout optimised for host copies.
    for (int64 i = 0; i < num_outputs; ++i) {
      TF_ASSIGN_OR_RETURN(
          result[i],
          AddHostCopyTensor(graph, {debug_name_and_id}, output_shapes[i]));
    }
  }

  return result;
}

class RecvFromHostOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "RecvFromHostOp");
    DriverProgramSequence seq(graph, {debug_info});

    const auto* recv = Cast<HloRecvFromHostInstruction>(inst);
    const int64 num_outputs = recv->RendezvousKeys().size();
    CHECK_GT(num_outputs, 0);

    // Either none or all inputs must be provided.
    if (inst->operand_count() > 0) {
      CHECK_EQ(inst->operand_count(), num_outputs);
    }

    // Get the individual output shapes.
    const std::vector<xla::Shape> output_shapes =
        GetOutputShapes(output_shape, num_outputs);
    CHECK_EQ(output_shapes.size(), num_outputs);

    // Get/allocate output tensors according to in-placeness.
    TF_ASSIGN_OR_RETURN(
        TensorVector output_tensors,
        GetOutputTensors(graph, res, inst, output_shapes, num_outputs,
                         tensor_map, seq, {debug_info}));
    CHECK_EQ(output_tensors.size(), num_outputs);

    // As long as the stream copies are scheduled right after each other,
    // Poplar will attempt to merge them according to `opt.maxCopyMergeSize`.
    for (int64 i = 0; i < num_outputs; ++i) {
      const std::string& rendezvous_key = recv->RendezvousKeys()[i];
      const xla::Shape& shape = output_shapes[i];
      poplar::Tensor& tensor = output_tensors[i];

      // Use the rendezvous key also for the Poplar stream handle.
      const poplar::DataStream stream = graph.addHostToDeviceFIFO(
          rendezvous_key, tensor.elementType(), tensor.numElements(),
          poplar::ReplicatedStreamMode::BROADCAST);

      seq.add(poplar::program::Copy(
          stream, tensor, res.always_rearrange_copies_on_host, {debug_info}));

      TF_CHECK_OK(
          AddOutputTensor(tensor_map, inst, i, DriverTensor(tensor, graph)));
      res.annotations.recv_infos.emplace_back(stream.handle(), rendezvous_key,
                                              shape);
    }

    return seq;
  }

  StatusOr<poplar::Tensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "RecvFromHostOp");
    const int64 input_index = tensor_target.input_index;
    const Shape& input_shape = tensor_target.tgt->operand(input_index)->shape();
    return AddHostCopyTensor(graph, {debug_info}, input_shape);
  }
};

REGISTER_POPLAR_OP(RecvFromHost, RecvFromHostOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
