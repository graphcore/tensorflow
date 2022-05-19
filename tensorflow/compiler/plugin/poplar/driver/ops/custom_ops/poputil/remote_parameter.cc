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

#include <poplar/DebugContext.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"

#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
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
    DriverGraph& graph, CompilerResources& res, poplar::Tensor source,
    poplar::RemoteBuffer remote_buffer,
    const poplar::DebugNameAndId& debug_name_and_id,
    absl::optional<poplar::Tensor> offset = absl::nullopt) {
  poplar::program::Sequence temporary_copy_seq({}, debug_name_and_id);
  poplar::program::Sequence stream_copy_seq({}, debug_name_and_id);

  const auto& handle = remote_buffer.handle();
  poplar::Tensor layout_tensor;
  if (res.remote_buffer_layouts.contains(handle)) {
    layout_tensor = res.remote_buffer_layouts.at(handle);
  } else {
    layout_tensor = source;
    res.remote_buffer_layouts[handle] = DriverTensor(layout_tensor, graph);
  }

  poplar::Tensor copy_tensor = graph.clone(layout_tensor, {debug_name_and_id});

  temporary_copy_seq.add(poplar::program::Copy(
      source.flatten(), copy_tensor.flatten(), false, {debug_name_and_id}));

  if (offset) {
    stream_copy_seq.add(poplar::program::Copy(copy_tensor, remote_buffer,
                                              *offset, debug_name_and_id));
  } else {
    stream_copy_seq.add(
        poplar::program::Copy(copy_tensor, remote_buffer, debug_name_and_id));
  }
  return {temporary_copy_seq, stream_copy_seq};
}

class RemoteParameterLoadOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    // Should have been handled by the deferred visitor.
    return FailedPrecondition("Remote parameter loads cannot be generated.");
  }
};
REGISTER_POPLAR_OP(RemoteParameterLoad, RemoteParameterLoadOp);

class RemoteParameterStoreOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "RemoteParameterStoreOp");
    VLOG(1) << "Processing " << GetDebugName(inst);
    DriverProgramSequence seq(graph, debug_info);

    const auto* store_inst = Cast<HloRemoteParameterStore>(inst);
    const int64_t num_outputs = store_inst->RemoteBuffers().size();

    const auto shapes = FlattenedXlaShape(output_shape);
    CHECK_EQ(shapes.size(), num_outputs);

    TF_ASSIGN_OR_RETURN(
        TensorOrRemoteBufferVectors outputs,
        FindInplaceOutputs(tensor_map, res, inst, seq, debug_info));
    CHECK_EQ(outputs.size(), num_outputs);

    poplar::program::Sequence temporary_copies_seq;
    poplar::program::Sequence stream_copies_seq;
    for (int64_t i = 0; i < num_outputs; ++i) {
      auto& shard_graph = GetGraphWithOutputIndex(res, inst, i);
      CHECK_EQ(outputs[i].size(), 1);
      TensorOrRemoteBuffer& output = outputs[i][0];

      const uint64 store_replication_factor =
          output.IsReplicaPartitioned() ? res.partition_replication_factor : 1;
      CHECK_EQ(store_inst->GetReplicationFactor(i), store_replication_factor)
          << store_inst->ToString();

      if (!output.IsRemoteBuffer()) {
        return xla::FailedPrecondition(
            "Expected a Poplar RemoteBuffer as operand %d to %s", i,
            GetDebugName(inst));
      }

      poplar::RemoteBuffer remote_buffer = output.AsRemoteBuffer();

      if (!UseSyntheticDataFor(SyntheticDataCategory::Parameters)) {
        TF_ASSIGN_OR_RETURN(
            poplar::Tensor tensor,
            FindInstructionInput(tensor_map, res, inst, num_outputs + i, seq,
                                 {debug_info}));

        auto pair_seq = AddRemoteBufferStoreCopy(shard_graph, res, tensor,
                                                 remote_buffer, {debug_info});
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
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    // Should have been handled by the deferred visitor.
    return FailedPrecondition("BufferLoadSlice cannot be generated.");
  }
};
REGISTER_POPLAR_OP(BufferLoadSlice, BufferLoadSliceOp);

class BufferStoreSliceOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "BufferStoreSliceOp");
    const auto* store_inst = Cast<HloBufferStoreSlice>(inst);
    const int64_t num_outputs = store_inst->RemoteBuffers().size();

    const auto shapes = FlattenedXlaShape(output_shape);
    CHECK_EQ(shapes.size(), num_outputs);

    DriverProgramSequence seq(graph, debug_info);
    TF_ASSIGN_OR_RETURN(
        TensorOrRemoteBufferVectors outputs,
        FindInplaceOutputs(tensor_map, res, inst, seq, debug_info));
    CHECK_EQ(outputs.size(), num_outputs);

    poplar::program::Sequence temporary_copies_seq;
    poplar::program::Sequence stream_copies_seq;
    for (int64_t i = 0; i < num_outputs; ++i) {
      auto& shard_graph = GetGraphWithOutputIndex(res, inst, i);
      CHECK_EQ(outputs[i].size(), 1);
      TensorOrRemoteBuffer& output = outputs[i][0];

      const uint64 store_replication_factor =
          output.IsReplicaPartitioned() ? res.partition_replication_factor : 1;
      CHECK_EQ(store_inst->GetReplicationFactor(i), store_replication_factor)
          << store_inst->ToString();

      if (!UseSyntheticDataFor(SyntheticDataCategory::Parameters)) {
        poplar::RemoteBuffer remote_buffer = output.AsRemoteBuffer();
        const auto value_index = num_outputs + i;
        const auto offset_index = 2 * num_outputs + i;

        TF_ASSIGN_OR_RETURN(
            poplar::Tensor value,
            FindInstructionInput(tensor_map, res, inst, value_index, seq,
                                 {debug_info}));

        TF_ASSIGN_OR_RETURN(
            poplar::Tensor offset,
            FindInstructionInput(tensor_map, res, inst, offset_index, seq,
                                 {debug_info}));

        auto pair_seq = AddRemoteBufferStoreCopy(
            shard_graph, res, value, remote_buffer, {debug_info}, offset);
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
