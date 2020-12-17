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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"

#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {
// TODO(T28772): Work around to make sure remote buffers can be rearranged on
// host.
std::pair<poplar::program::Sequence, poplar::program::Sequence>
AddRemoteBufferStoreCopy(
    poplar::Graph& graph, CompilerResources& res, poplar::Tensor source,
    poplar::RemoteBuffer remote_buffer,
    absl::optional<poplar::Tensor> offset = absl::nullopt) {
  poplar::program::Sequence temporary_copy_seq;
  poplar::program::Sequence stream_copy_seq;

  const auto& handle = remote_buffer.handle();
  poplar::Tensor layout_tensor;
  if (res.remote_buffer_layouts.contains(handle)) {
    layout_tensor = res.remote_buffer_layouts.at(handle);
  } else {
    layout_tensor = source;
    res.remote_buffer_layouts[handle] = layout_tensor;
  }

  poplar::Tensor copy_tensor = graph.clone(layout_tensor);

  temporary_copy_seq.add(
      poplar::program::Copy(source.flatten(), copy_tensor.flatten()));

  if (offset) {
    stream_copy_seq.add(
        poplar::program::Copy(copy_tensor, remote_buffer, *offset));
  } else {
    stream_copy_seq.add(poplar::program::Copy(copy_tensor, remote_buffer));
  }
  return {temporary_copy_seq, stream_copy_seq};
}

class RemoteParameterLoadOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    // Should have been handled by the deferred visitor.
    return FailedPrecondition("Remote parameter loads cannot be generated.");
  }
};
REGISTER_POPLAR_OP(RemoteParameterLoad, RemoteParameterLoadOp);

class RemoteParameterStoreOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    VLOG(1) << "Processing " << GetDebugName(inst);
    poplar::program::Sequence seq;

    const auto* store_inst = Cast<HloRemoteParameterStore>(inst);
    const int64 num_outputs = store_inst->RemoteBuffers().size();

    const auto shapes = FlattenedXlaShape(output_shape);
    CHECK_EQ(shapes.size(), num_outputs);

    TF_ASSIGN_OR_RETURN(TensorOrRemoteBufferVectors outputs,
                        FindInplaceOutputs(tensor_map, res, inst, seq));
    CHECK_EQ(outputs.size(), num_outputs);

    poplar::program::Sequence temporary_copies_seq;
    poplar::program::Sequence stream_copies_seq;
    for (int64 i = 0; i < num_outputs; ++i) {
      poplar::Graph& shard_graph = GetGraphWithOutputIndex(res, inst, i);
      if (store_inst->GetReplicationFactor(i) != res.replication_factor &&
          store_inst->GetReplicationFactor(i) != 1) {
        return xla::FailedPrecondition(
            "RemoteBuffer store instruction replication factor doesn't match "
            "graph replication factor.");
      }

      CHECK_EQ(outputs[i].size(), 1);

      if (!outputs[i][0].IsRemoteBuffer()) {
        return xla::FailedPrecondition(
            "Expected a Poplar RemoteBuffer as operand %d to %s", i,
            GetDebugName(inst));
      }

      poplar::RemoteBuffer remote_buffer = outputs[i][0].AsRemoteBuffer();

      if (!UseSyntheticData()) {
        TF_ASSIGN_OR_RETURN(
            poplar::Tensor tensor,
            FindInstructionInput(tensor_map, res, inst, num_outputs + i, seq));

        auto pair_seq =
            AddRemoteBufferStoreCopy(shard_graph, res, tensor, remote_buffer);
        temporary_copies_seq.add(pair_seq.first);
        stream_copies_seq.add(pair_seq.second);
      }
      TF_CHECK_OK(AddOutput(tensor_map, inst, i, outputs[i][0]));
    }
    seq.add(temporary_copies_seq);
    seq.add(stream_copies_seq);

    return seq;
  }
};
REGISTER_POPLAR_OP(RemoteParameterStore, RemoteParameterStoreOp);

class BufferLoadSliceOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    // Should have been handled by the deferred visitor.
    return FailedPrecondition("BufferLoadSlice cannot be generated.");
  }
};
REGISTER_POPLAR_OP(BufferLoadSlice, BufferLoadSliceOp);

class BufferStoreSliceOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    const auto* store_inst = Cast<HloBufferStoreSlice>(inst);
    const int64 num_outputs = store_inst->RemoteBuffers().size();

    const auto shapes = FlattenedXlaShape(output_shape);
    CHECK_EQ(shapes.size(), num_outputs);

    poplar::program::Sequence seq;
    TF_ASSIGN_OR_RETURN(TensorOrRemoteBufferVectors outputs,
                        FindInplaceOutputs(tensor_map, res, inst, seq));
    CHECK_EQ(outputs.size(), num_outputs);

    poplar::program::Sequence temporary_copies_seq;
    poplar::program::Sequence stream_copies_seq;
    for (int64 i = 0; i < num_outputs; ++i) {
      poplar::Graph& shard_graph = GetGraphWithOutputIndex(res, inst, i);
      CHECK_EQ(outputs[i].size(), 1);
      TensorOrRemoteBuffer& output = outputs[i][0];

      if (!UseSyntheticData()) {
        poplar::RemoteBuffer remote_buffer = output.AsRemoteBuffer();
        const auto value_index = num_outputs + i;
        const auto offset_index = 2 * num_outputs + i;

        TF_ASSIGN_OR_RETURN(
            poplar::Tensor value,
            FindInstructionInput(tensor_map, res, inst, value_index, seq));

        TF_ASSIGN_OR_RETURN(
            poplar::Tensor offset,
            FindInstructionInput(tensor_map, res, inst, offset_index, seq));

        auto pair_seq = AddRemoteBufferStoreCopy(shard_graph, res, value,
                                                 remote_buffer, offset);
        temporary_copies_seq.add(pair_seq.first);
        stream_copies_seq.add(pair_seq.second);
      }

      TF_CHECK_OK(AddOutput(tensor_map, inst, i, output));
    }
    seq.add(temporary_copies_seq);
    seq.add(stream_copies_seq);

    return seq;
  }
};
REGISTER_POPLAR_OP(BufferStoreSlice, BufferStoreSliceOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
