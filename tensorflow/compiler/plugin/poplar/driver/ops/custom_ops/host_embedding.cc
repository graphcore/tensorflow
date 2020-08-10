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

#include <popnn/Loss.hpp>
#include <popops/Cast.hpp>
#include <popops/Collectives.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/HostSliceTensor.hpp>
#include <popops/Pad.hpp>
#include <popops/Reduce.hpp>
#include <popops/Zero.hpp>
#include <poputil/TileMapping.hpp>

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

StatusOr<poplar::RemoteBuffer> GetOrCreateRemoteBuffer(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    const std::string& embedding_id, const xla::Shape& embedding_shape,
    HostEmbeddingSplittingStrategy splitting_strategy,
    const xla::Shape& output_shape) {
  if (!res.remote_memory_supported) {
    return FailedPrecondition(
        "Poplar remote buffers are not supported on this machine. They are "
        "required to support experimental remote buffer embeddings. Consider "
        "either configuring this machine to support remote buffers or "
        "setting enable_experimental_remote_buffer_embedding to false.");
  }

  auto itr = res.remote_buffers.find(embedding_id);

  // If we didn't find the remote buffer, create it.
  if (itr == res.remote_buffers.end()) {
    std::vector<std::size_t> dim = PoplarShapeFromXlaShape(embedding_shape);
    TF_ASSIGN_OR_RETURN(poplar::Type poplar_type,
                        PoplarDataType(embedding_shape));

    // In a replicated graph, so we need to consider splitting strategy.
    if (res.replication_factor > 1) {
      if (splitting_strategy == HostEmbeddingSplittingStrategy::Token) {
        dim[0] = tensorflow::MathUtil::CeilOfRatio<int>(dim[0],
                                                        res.replication_factor);
      } else if (splitting_strategy ==
                 HostEmbeddingSplittingStrategy::Encoding) {
        dim[1] = tensorflow::MathUtil::CeilOfRatio<int>(dim[1],
                                                        res.replication_factor);
      }
    }

    // Create a remote buffer, and add an extra "junk" token for invalid
    // writes.
    poplar::RemoteBuffer result = graph.addRemoteBuffer(
        embedding_id, poplar_type, dim[1], dim[0] + 1, true, true);
    res.remote_buffers.insert({embedding_id, result});

    // Keep track of the embedding information.
    res.annotations.host_embedding_lookup_infos.push_back(
        {inst->name(), embedding_id, inst->operand(0)->shape(), output_shape,
         splitting_strategy});

    return result;
  }

  return itr->second;
}

class HostEmbeddingLookupOp : public PoplarOpDef {
  // Synthetic embedding lookup.
  StatusOr<poplar::program::Program> SyntheticImpl(
      poplar::Graph& graph, poplar::Tensor indices,
      poplar::program::Sequence seq, CompilerResources& res,
      const HloHostEmbeddingLookupInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map) {
    TF_ASSIGN_OR_RETURN(poplar::Tensor output,
                        AddTensor(graph, TensorLocation{inst, 0}, output_shape,
                                  res, tensor_map));
    seq.add(poplar::program::WriteUndef(output));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));

    return seq;
  }

  // Host embedding using a poplar callback.
  StatusOr<poplar::program::Program> CallbackImpl(
      poplar::Graph& graph, poplar::Tensor indices,
      poplar::program::Sequence seq, CompilerResources& res,
      const HloHostEmbeddingLookupInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map) {
    TF_ASSIGN_OR_RETURN(poplar::Tensor output,
                        AddTensor(graph, TensorLocation{inst, 0}, output_shape,
                                  res, tensor_map));

    res.annotations.host_embedding_lookup_infos.push_back(
        {inst->name(), inst->EmbeddingId(), inst->operand(0)->shape(),
         output_shape, inst->SplittingStrategy()});

    auto index_buffer = graph.addDeviceToHostFIFO(
        inst->name() + inst->EmbeddingId() + "_indices", indices.elementType(),
        indices.numElements());

    auto activation_fifo = graph.addHostToDeviceFIFO(
        inst->name() + inst->EmbeddingId() + "_activations",
        output.elementType(), output.numElements());

    // Send the indices to the host.
    seq.add(poplar::program::Copy(indices, index_buffer));

    // Sync to avoid any stream merging due to host-side data dependecy.
    seq.add(poplar::program::Sync(poplar::SyncType::INTERNAL));

    // Read the values from the host.
    seq.add(poplar::program::Copy(activation_fifo, output));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));

    return seq;
  }

  // Single replica remote buffer implementation.
  StatusOr<poplar::program::Program> RemoteBufferImpl(
      poplar::Graph& graph, poplar::RemoteBuffer& remote_buffer,
      poplar::Tensor indices, poplar::program::Sequence seq,
      CompilerResources& res, const HloHostEmbeddingLookupInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map) {
    // Create the output tensor.
    TF_ASSIGN_OR_RETURN(poplar::Tensor output,
                        AddTensor(graph, TensorLocation{inst, 0}, output_shape,
                                  res, tensor_map));

    // Create the host sliceable tensor.
    auto host_sliceable = popops::createHostSliceableTensor(
        graph, output.elementType(), output.shape(), false,
        GetDebugName(inst) + "/tmp");

    // Copy the indices into the host sliceable indices.
    seq.add(poplar::program::Copy(indices, host_sliceable.indices));

    // Copy from the remote buffer.
    seq.add(poplar::program::Copy(remote_buffer, host_sliceable.tensor,
                                  host_sliceable.indices));

    // Copy the values into the output tensor.
    seq.add(poplar::program::Copy(host_sliceable.tensor, output));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));
    return seq;
  }

  // Replicated remote buffer embedding lookup using the token splitting
  // strategy.
  StatusOr<poplar::program::Program> RemoteBufferSplitTokensImpl(
      poplar::Graph& graph, poplar::RemoteBuffer& remote_buffer,
      poplar::Tensor indices, poplar::program::Sequence seq,
      CompilerResources& res, const HloHostEmbeddingLookupInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map) {
    // Create the output tensor.
    TF_ASSIGN_OR_RETURN(poplar::Tensor output,
                        AddTensor(graph, TensorLocation{inst, 0}, output_shape,
                                  res, tensor_map));

    // All-Gather the indices from all replicas.
    indices = popops::replicatedAllGather(graph, indices, seq,
                                          GetDebugName(inst) + "/indices",
                                          GetReplicatedCollectiveOptions(res));

    // Create a replication factor constant tensor.
    poplar::Tensor rep =
        graph.addConstant(poplar::UNSIGNED_INT, {}, res.replication_factor,
                          GetDebugName(inst) + "/replication_factor");
    MappingHelper::MapTensorLinearly(res.linear_mapping_state, graph, rep);

    // Divide the indices by the replication factor to transform the global
    // address space indices to the replica-local address space. Given we can
    // currently assume there is a power of two replication factor, this could
    // be rewritten as bitshift right log_2(rep) bits.
    poplar::Tensor ind = popops::div(graph, indices.flatten(), rep, seq,
                                     GetDebugName(inst) + "/shift_indices");

    // Create the host sliceable temporary tensor.
    auto host_sliceable = popops::createHostSliceableTensor(
        graph, output.elementType(),
        {indices.dim(0) * indices.dim(1), output.dim(1)}, false,
        GetDebugName(inst) + "/tmp");

    // Copy the replica-local indices to the host sliceable indices.
    seq.add(poplar::program::Copy(ind, host_sliceable.indices));

    // Read from the remote buffer into the host sliceable tensor.
    seq.add(poplar::program::Copy(remote_buffer, host_sliceable.tensor,
                                  host_sliceable.indices));

    // The current replica index.
    poplar::Tensor replica_id = graph.addReplicationIndexConstant();
    MappingHelper::MapTensorLinearly(res.linear_mapping_state, graph,
                                     replica_id);

    // Take the modulus of the global indices with the replication factor.
    // Given we can currently assume that the replication factor is a power of
    // two, this could be rewritten as a bitwise-and with rep-1.
    poplar::Tensor rem = popops::rem(graph, indices, rep, seq,
                                     GetDebugName(inst) + "/mask_indices");

    // Check whether (i mod r) is equal to this replica id. This tells us
    // whether the element refered to by the global index is "owned" by this
    // replca.
    poplar::Tensor mask =
        popops::eq(graph, rem.reinterpret(replica_id.elementType()), replica_id,
                   seq, GetDebugName(inst) + "/mask");

    // Create a constant zero fo masking.
    poplar::Tensor zero =
        graph.addConstant(host_sliceable.tensor.elementType(), {}, 0,
                          GetDebugName(inst) + "/zero");
    MappingHelper::MapTensorLinearly(res.linear_mapping_state, graph, zero);

    // Use the computed mask, based on whether this replica "owns" the requested
    // element, to zero the invalid elements of host_sliceable.tensor.
    popops::selectInPlace(graph, host_sliceable.tensor, zero,
                          mask.flatten().expand({1}), seq,
                          GetDebugName(inst) + "/mask_output");
    host_sliceable.tensor = host_sliceable.tensor.reshape(
        {indices.dim(0), indices.dim(1), output.dim(1)});

    // Given that invalid elements will be zero and valid elements will contain
    // the correct value, we can replicatedReduceScatter sum to distribute the
    // results to the correct replica.
    host_sliceable.tensor = popops::replicatedReduceScatter(
        graph, host_sliceable.tensor.flatten(), popops::Operation::ADD, seq,
        GetDebugName(inst) + "/reduce_scatter",
        GetReplicatedCollectiveOptions(res));

    // Copy the result to the output tensor.
    seq.add(poplar::program::Copy(host_sliceable.tensor.reshape(output.shape()),
                                  output));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));
    return seq;
  }

  // Replicated remote buffer embedding lookup using the encoding splitting
  // strategy.
  StatusOr<poplar::program::Program> RemoteBufferSplitEncodingImpl(
      poplar::Graph& graph, poplar::RemoteBuffer& remote_buffer,
      poplar::Tensor indices, poplar::program::Sequence seq,
      CompilerResources& res, const HloHostEmbeddingLookupInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map) {
    // Create the output tensor.
    TF_ASSIGN_OR_RETURN(poplar::Tensor output,
                        AddTensor(graph, TensorLocation{inst, 0}, output_shape,
                                  res, tensor_map));

    // All-Gather the indices from all replicas.
    indices = popops::replicatedAllGather(graph, indices, seq,
                                          GetDebugName(inst) + "/indices",
                                          GetReplicatedCollectiveOptions(res));

    // Create the host sliceable temporary tensor.
    auto host_sliceable = popops::createHostSliceableTensor(
        graph, output.elementType(),
        {indices.dim(0) * indices.dim(1),
         tensorflow::MathUtil::CeilOfRatio<unsigned>(output.dim(1),
                                                     res.replication_factor)},
        false, inst->name() + "/tmp");

    // Copy the indices to the host sliceable indices.
    seq.add(poplar::program::Copy(indices.flatten(), host_sliceable.indices));

    // Copy from the remote buffer into the host sliceable tensor.
    seq.add(poplar::program::Copy(remote_buffer, host_sliceable.tensor,
                                  host_sliceable.indices));
    host_sliceable.tensor = host_sliceable.tensor.reshapePartial(
        0, 1, {indices.dim(0), indices.dim(1)});

    // Exchange the columns from this replica back to their respective replicas.
    // We also recieve the columns we requested from the other replicas.
    host_sliceable.tensor = popops::allToAllPersonalizedExchange(
        graph, host_sliceable.tensor, seq,
        GetDebugName(inst) + "/exchange_columns",
        GetReplicatedCollectiveOptions(res));

    // Dimshuffle and reshape back to the output shape.
    host_sliceable.tensor = host_sliceable.tensor.dimShuffle({1, 0, 2});
    host_sliceable.tensor = host_sliceable.tensor.flatten(1, 3);
    host_sliceable.tensor = host_sliceable.tensor.slice(0, output.dim(1), 1);

    // Copy to the output tensor.
    seq.add(poplar::program::Copy(host_sliceable.tensor, output));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));
    return seq;
  }

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
    TF_ASSIGN_OR_RETURN(
        TensorVector indices,
        FindInstructionInputTensors(tensor_map, res, inst, 0, seq, false));

    const HloHostEmbeddingLookupInstruction* host_embedding_inst =
        Cast<HloHostEmbeddingLookupInstruction>(inst);

    if (UseSyntheticData()) {
      return SyntheticImpl(graph, indices[0].reinterpret(poplar::UNSIGNED_INT),
                           seq, res, host_embedding_inst, output_shape,
                           tensor_map);
    }

    if (res.enable_experimental_remote_buffer_embedding) {
      VLOG(1) << "Using experimental remote buffer embedding lookup";

      TF_ASSIGN_OR_RETURN(
          poplar::RemoteBuffer rbuffer,
          GetOrCreateRemoteBuffer(
              graph, res, inst, host_embedding_inst->EmbeddingId(),
              host_embedding_inst->EmbeddingShape(),
              host_embedding_inst->SplittingStrategy(), output_shape));

      if (res.replication_factor > 1) {
        if (host_embedding_inst->SplittingStrategy() ==
            HostEmbeddingSplittingStrategy::Token) {
          return RemoteBufferSplitTokensImpl(
              graph, rbuffer, indices[0].reinterpret(poplar::UNSIGNED_INT), seq,
              res, host_embedding_inst, output_shape, tensor_map);
        }

        if (host_embedding_inst->SplittingStrategy() ==
            HostEmbeddingSplittingStrategy::Encoding) {
          return RemoteBufferSplitEncodingImpl(
              graph, rbuffer, indices[0].reinterpret(poplar::UNSIGNED_INT), seq,
              res, host_embedding_inst, output_shape, tensor_map);
        }
      }

      return RemoteBufferImpl(
          graph, rbuffer, indices[0].reinterpret(poplar::UNSIGNED_INT), seq,
          res, host_embedding_inst, output_shape, tensor_map);
    }

    return CallbackImpl(graph, indices[0].reinterpret(poplar::UNSIGNED_INT),
                        seq, res, host_embedding_inst, output_shape,
                        tensor_map);
  }
};  // namespace

REGISTER_POPLAR_OP(HostEmbeddingLookup, HostEmbeddingLookupOp);

class HostEmbeddingUpdateOp : public PoplarOpDef {
  // Synthetic embedding update.
  StatusOr<poplar::program::Program> SyntheticImpl(
      poplar::program::Sequence seq) {
    return seq;
  }

  // Host embedding using a poplar callback.
  StatusOr<poplar::program::Program> CallbackImpl(
      poplar::Graph& graph, poplar::Tensor grads, poplar::Tensor indices,
      poplar::program::Sequence seq, CompilerResources& res,
      const HloHostEmbeddingUpdateInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map) {
    res.annotations.host_embedding_update_infos.push_back(
        {inst->name(), inst->EmbeddingId(), inst->operand(2)->shape(),
         inst->operand(1)->shape()});

    auto index_buffer = graph.addDeviceToHostFIFO(
        inst->name() + inst->EmbeddingId() + "_indices", indices.elementType(),
        indices.numElements());

    auto grad_fifo =
        graph.addDeviceToHostFIFO(inst->name() + inst->EmbeddingId() + "_grads",
                                  grads.elementType(), grads.numElements());

    seq.add(poplar::program::Copy(indices, index_buffer));
    seq.add(poplar::program::Copy(grads, grad_fifo));

    return seq;
  }

  // Single replica remote buffer implementation.
  StatusOr<poplar::program::Program> RemoteBufferImpl(
      poplar::Graph& graph, poplar::RemoteBuffer& remote_buffer,
      poplar::Tensor grads, poplar::Tensor indices,
      poplar::program::Sequence seq, CompilerResources& res,
      const HloHostEmbeddingUpdateInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map) {
    // Create the temporary host sliceable tensor.
    auto host_sliceable = popops::createHostSliceableTensor(
        graph, grads.elementType(), grads.shape(), true,
        GetDebugName(inst) + "/tmp");

    // Read the weights from the host.
    seq.add(poplar::program::Copy(indices, host_sliceable.indices));
    seq.add(poplar::program::Copy(remote_buffer, host_sliceable.tensor,
                                  host_sliceable.indices));

    // Apply the gradients.
    popops::addInPlace(graph, host_sliceable.tensor, grads, seq,
                       GetDebugName(inst) + "/apply_grads");

    // Send the updated weights back to the host.
    seq.add(poplar::program::Copy(host_sliceable.tensor, remote_buffer,
                                  host_sliceable.indices));

    return seq;
  }

  // Replicated remote buffer embedding lookup using the token splitting
  // strategy.
  StatusOr<poplar::program::Program> RemoteBufferSplitTokensImpl(
      poplar::Graph& graph, poplar::RemoteBuffer& remote_buffer,
      poplar::Tensor grads, poplar::Tensor indices,
      poplar::program::Sequence seq, CompilerResources& res,
      const HloHostEmbeddingUpdateInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map) {
    // All-Gather the indices from all replicas.
    indices = popops::replicatedAllGather(graph, indices, seq,
                                          GetDebugName(inst) + "/indices",
                                          GetReplicatedCollectiveOptions(res));
    indices = indices.flatten(0, 2);

    // All-Gather the grads from all replicas.
    grads = popops::replicatedAllGather(graph, grads, seq,
                                        GetDebugName(inst) + "/grads",
                                        GetReplicatedCollectiveOptions(res));
    grads = grads.flatten(0, 2);

    // Create the host sliceable temporary tensor.
    auto host_sliceable = popops::createHostSliceableTensor(
        graph, grads.elementType(), grads.shape(), true,
        GetDebugName(inst) + "/tmp");

    // Copy the indices to the host sliceable indices.
    seq.add(poplar::program::Copy(indices, host_sliceable.indices));

    // Create a replication factor constant tensor.
    poplar::Tensor rep =
        graph.addConstant(poplar::UNSIGNED_INT, {}, res.replication_factor,
                          GetDebugName(inst) + "/replication_factor");
    MappingHelper::MapTensorLinearly(res.linear_mapping_state, graph, rep);

    // Divide the indices by the replication factor to transform the global
    // address space indices to the replica-local address space. Given we can
    // currently assume there is a power of two replication factor, this could
    // be rewritten as bitshift right log_2(rep) bits.
    popops::divInPlace(graph, host_sliceable.indices, rep, seq,
                       GetDebugName(inst) + "/shift_indices");

    // Create an invalid constant index that is used to avoid clobbering valid
    // values in the remote buffer. Currently implemented as one-past-the-end
    // index. Might be replaced with another sentinel value later.
    poplar::Tensor invalid = graph.addConstant(
        poplar::UNSIGNED_INT, {},
        tensorflow::MathUtil::CeilOfRatio<unsigned>(
            inst->EmbeddingShape().dimensions(0), res.replication_factor),
        GetDebugName(inst) + "/invalid_index");
    MappingHelper::MapTensorLinearly(res.linear_mapping_state, graph, invalid);

    // The current replica index.
    poplar::Tensor replica_id = graph.addReplicationIndexConstant();
    MappingHelper::MapTensorLinearly(res.linear_mapping_state, graph,
                                     replica_id);

    // Take the modulus of the global indices with the replication factor.
    // Given we can currently assume that the replication factor is a power of
    // two, this could be rewritten as a bitwise-and with rep-1.
    poplar::Tensor rem = popops::rem(graph, host_sliceable.indices, rep, seq,
                                     GetDebugName(inst) + "/mask_indices");

    // Check whether (i mod r) is equal to this replica id. This tells us
    // whether the element refered to by the global index is "owned" by this
    // replca.
    poplar::Tensor mask =
        popops::eq(graph, rem.reinterpret(replica_id.elementType()), replica_id,
                   seq, GetDebugName(inst) + "/mask");

    // Use the computed mask, based on whether this replica "owns" the requested
    // element, to invalidate the indices and avoid overwritting valid regions
    // of the replica-local table.
    popops::selectInPlace(graph,
                          host_sliceable.indices.reinterpret(poplar::INT),
                          invalid.reinterpret(poplar::INT), mask, seq,
                          GetDebugName(inst) + "/mask_indices");

    // Read from the remote buffer into the host sliceable tensor.
    seq.add(poplar::program::Copy(remote_buffer, host_sliceable.tensor,
                                  host_sliceable.indices));

    // Apply the gradients to the values read from the remote buffer.
    popops::addInPlace(graph, host_sliceable.tensor, grads, seq,
                       GetDebugName(inst) + "/apply_grads");

    // Read from the host sliceable tensor into the remote buffer.
    seq.add(poplar::program::Copy(host_sliceable.tensor, remote_buffer,
                                  host_sliceable.indices));

    return seq;
  }

  // Replicated remote buffer embedding lookup using the encoding splitting
  // strategy.
  StatusOr<poplar::program::Program> RemoteBufferSplitEncodingImpl(
      poplar::Graph& graph, poplar::RemoteBuffer& remote_buffer,
      poplar::Tensor grads, poplar::Tensor indices,
      poplar::program::Sequence seq, CompilerResources& res,
      const HloHostEmbeddingUpdateInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map) {
    // All-Gather the indices from all replicas.
    indices = popops::replicatedAllGather(graph, indices, seq,
                                          GetDebugName(inst) + "/indices",
                                          GetReplicatedCollectiveOptions(res));

    // Check whether we need to pad the gradients because the replication factor
    // isn't a factor of the encoding width.
    if (grads.dim(1) % res.replication_factor > 0) {
      grads = popops::pad(graph, grads, {0, 0},
                          {0, grads.dim(1) % res.replication_factor});
    }

    // Dimshuffle and reshape to move the column slices to the outermost
    // dimension.
    grads = grads.reshapePartial(
        1, 2, {indices.dim(0), grads.dim(1) / indices.dim(0)});
    grads = grads.dimShuffle({1, 0, 2});

    // All-To-All exchange the grad columns with their respective replicas.
    grads = popops::allToAllPersonalizedExchange(
        graph, grads, seq, GetDebugName(inst) + "/exchange_columns",
        GetReplicatedCollectiveOptions(res));
    grads = grads.flatten(0, 2);

    // Create the host sliceable temporary tensor.
    auto host_sliceable = popops::createHostSliceableTensor(
        graph, grads.elementType(),
        {indices.dim(0) * indices.dim(1), grads.dim(1)}, true,
        inst->name() + "/tmp");

    // Copy the replica-local indices to the host sliceable indices.
    seq.add(poplar::program::Copy(indices.flatten(), host_sliceable.indices));

    // Copy from the remote buffer into the host sliceable tensor.
    seq.add(poplar::program::Copy(remote_buffer, host_sliceable.tensor,
                                  host_sliceable.indices));

    // Apply the gradients to the read values.
    popops::addInPlace(graph, host_sliceable.tensor, grads, seq,
                       GetDebugName(inst) + "/apply_grads");

    // Copy from the host sliceable tensor into the remote buffer.
    seq.add(poplar::program::Copy(host_sliceable.tensor, remote_buffer,
                                  host_sliceable.indices));

    return seq;
  }

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

    TF_ASSIGN_OR_RETURN(
        TensorVector grads,
        FindInstructionInputTensors(tensor_map, res, inst, 1, seq, false));

    TF_ASSIGN_OR_RETURN(
        TensorVector indices,
        FindInstructionInputTensors(tensor_map, res, inst, 2, seq, false));

    const HloHostEmbeddingUpdateInstruction* host_embedding_inst =
        Cast<HloHostEmbeddingUpdateInstruction>(inst);
    if (UseSyntheticData()) {
      return SyntheticImpl(seq);
    } else if (res.enable_experimental_remote_buffer_embedding) {
      VLOG(1) << "Using experimental remote buffer embedding update";

      TF_ASSIGN_OR_RETURN(
          poplar::RemoteBuffer rbuffer,
          GetOrCreateRemoteBuffer(
              graph, res, inst, host_embedding_inst->EmbeddingId(),
              host_embedding_inst->EmbeddingShape(),
              host_embedding_inst->SplittingStrategy(), output_shape));

      if (res.replication_factor > 1) {
        if (host_embedding_inst->SplittingStrategy() ==
            HostEmbeddingSplittingStrategy::Token) {
          return RemoteBufferSplitTokensImpl(
              graph, rbuffer, grads[0],
              indices[0].reinterpret(poplar::UNSIGNED_INT), seq, res,
              host_embedding_inst, output_shape, tensor_map);
        }

        if (host_embedding_inst->SplittingStrategy() ==
            HostEmbeddingSplittingStrategy::Encoding) {
          return RemoteBufferSplitEncodingImpl(
              graph, rbuffer, grads[0],
              indices[0].reinterpret(poplar::UNSIGNED_INT), seq, res,
              host_embedding_inst, output_shape, tensor_map);
        }
      }

      return RemoteBufferImpl(graph, rbuffer, grads[0],
                              indices[0].reinterpret(poplar::UNSIGNED_INT), seq,
                              res, host_embedding_inst, output_shape,
                              tensor_map);
    } else {
      return CallbackImpl(graph, grads[0],
                          indices[0].reinterpret(poplar::UNSIGNED_INT), seq,
                          res, host_embedding_inst, output_shape, tensor_map);
    }

    return seq;
  }
};

REGISTER_POPLAR_OP(HostEmbeddingUpdate, HostEmbeddingUpdateOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
