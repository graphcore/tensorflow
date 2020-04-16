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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/host_embedding.h"

#include <popops/Zero.hpp>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

class HostEmbeddingLookupOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    if (res.use_verified_transfers) {
      return FailedPrecondition(
          "Verified transfers cannot be used with Host embeddings");
    }

    poplar::program::Sequence seq;
    TensorVector indices =
        FindInstructionInputs(tensor_map, res, inst, 0, seq, false);

    TF_ASSIGN_OR_RETURN(poplar::Tensor output,
                        AddTensor(graph, TensorLocation{inst, 0}, output_shape,
                                  res, tensor_map));

    if (UseSyntheticData()) {
      seq.add(poplar::program::WriteUndef(output));
    } else {
      const HloHostEmbeddingLookupInstruction* host_embedding_inst =
          Cast<HloHostEmbeddingLookupInstruction>(inst);

      res.annotations.host_embedding_lookup_infos.push_back(
          {inst->name(), host_embedding_inst->EmbeddingId(),
           inst->operand(0)->shape(), output_shape});

      auto index_buffer = graph.addDeviceToHostFIFO(
          inst->name() + host_embedding_inst->EmbeddingId() + "_indices",
          indices[0].elementType(), indices[0].numElements());

      auto activation_fifo = graph.addHostToDeviceFIFO(
          inst->name() + host_embedding_inst->EmbeddingId() + "_activations",
          output.elementType(), output.numElements());

      seq.add(poplar::program::Copy(indices[0], index_buffer));
      seq.add(poplar::program::Sync(poplar::SyncType::INTERNAL));
      seq.add(poplar::program::Copy(activation_fifo, output));
    }

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));

    return seq;
  }
};

REGISTER_POPLAR_OP(HostEmbeddingLookup, HostEmbeddingLookupOp);

class HostEmbeddingUpdateOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    if (res.use_verified_transfers) {
      return FailedPrecondition(
          "Verified transfers cannot be used with Host embeddings");
    }

    poplar::program::Sequence seq;
    if (!UseSyntheticData()) {
      TensorVector grads =
          FindInstructionInputs(tensor_map, res, inst, 0, seq, false);

      TensorVector indices =
          FindInstructionInputs(tensor_map, res, inst, 1, seq, false);

      const HloHostEmbeddingUpdateInstruction* host_embedding_inst =
          Cast<HloHostEmbeddingUpdateInstruction>(inst);

      res.annotations.host_embedding_update_infos.push_back(
          {inst->name(), host_embedding_inst->EmbeddingId(),
           inst->operand(1)->shape(), inst->operand(0)->shape()});

      auto index_buffer = graph.addDeviceToHostFIFO(
          inst->name() + host_embedding_inst->EmbeddingId() + "_indices",
          indices[0].elementType(), indices[0].numElements());

      auto grad_fifo = graph.addDeviceToHostFIFO(
          inst->name() + host_embedding_inst->EmbeddingId() + "_grads",
          grads[0].elementType(), grads[0].numElements());

      seq.add(poplar::program::Copy(indices[0], index_buffer));
      seq.add(poplar::program::Copy(grads[0], grad_fifo));
    }

    return seq;
  }
};

REGISTER_POPLAR_OP(HostEmbeddingUpdate, HostEmbeddingUpdateOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
