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

#include <gcl/Collectives.hpp>
#include <popnn/Loss.hpp>
#include <popops/Cast.hpp>
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
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
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

StatusOr<RemoteBufferHolder*> GetOrCreateRemoteBuffer(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    const std::string& embedding_id, const xla::Shape& embedding_shape,
    HostEmbeddingSplittingStrategy splitting_strategy,
    const xla::Shape& output_shape) {
  if (!res.remote_memory_supported) {
    return FailedPrecondition(
        "Poplar remote buffers are not supported on this machine. They are "
        "required to support experimental remote buffer embeddings. Consider "
        "either configuring this machine to support remote buffers or "
        "setting experimental.enable_remote_buffer_embedding to False on an "
        "IPUConfig instance.");
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

    // Create a remote buffer proxy, and add an extra "junk" token for invalid
    // writes.
    auto result =
        res.remote_buffers
            .emplace(embedding_id, absl::make_unique<RemoteBufferHolder>(
                                       graph, embedding_id, poplar_type, dim[1],
                                       dim[0] + 1, true, true))
            .first->second.get();

    // Keep track of the embedding information.
    res.annotations.host_embedding_lookup_infos.push_back(
        {inst->name(), embedding_id, inst->operand(0)->shape(), output_shape,
         splitting_strategy});

    return result;
  }

  return itr->second.get();
}

class HostEmbeddingLookupOp : public PoplarOpDef {
  // Synthetic embedding lookup.
  StatusOr<poplar::program::Program> SyntheticImpl(
      poplar::Graph& graph, poplar::Tensor indices,
      poplar::program::Sequence seq, CompilerResources& res,
      const HloHostEmbeddingLookupInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugNameAndId& debug_name_and_id) {
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor output,
        AddTensor(graph, TensorLocation{inst, 0}, output_shape, res, tensor_map,
                  {debug_name_and_id, "output"}));
    seq.add(poplar::program::WriteUndef(output, {debug_name_and_id}));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));

    return seq;
  }

  // Host embedding using a poplar callback.
  StatusOr<poplar::program::Program> CallbackImpl(
      poplar::Graph& graph, poplar::Tensor indices,
      poplar::program::Sequence seq, CompilerResources& res,
      const HloHostEmbeddingLookupInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugNameAndId& debug_name_and_id) {
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor output,
        AddTensor(graph, TensorLocation{inst, 0}, output_shape, res, tensor_map,
                  {debug_name_and_id, "output"}));

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
    seq.add(poplar::program::Copy(indices, index_buffer, false,
                                  {debug_name_and_id}));

    // Sync to avoid any stream merging due to host-side data dependecy.
    seq.add(
        poplar::program::Sync(poplar::SyncType::INTERNAL, {debug_name_and_id}));

    // Read the values from the host.
    seq.add(poplar::program::Copy(activation_fifo, output, false,
                                  {debug_name_and_id}));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));

    return seq;
  }

  // Single replica remote buffer implementation.
  StatusOr<poplar::program::Program> RemoteBufferImpl(
      poplar::Graph& graph, poplar::RemoteBuffer& remote_buffer,
      poplar::Tensor indices, poplar::program::Sequence seq,
      CompilerResources& res, const HloHostEmbeddingLookupInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugNameAndId& debug_name_and_id) {
    // Create the output tensor.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor output,
        AddTensor(graph, TensorLocation{inst, 0}, output_shape, res, tensor_map,
                  {debug_name_and_id, "output"}));

    // Create the host sliceable tensor.
    auto host_sliceable = popops::createHostSliceableTensor(
        graph, output.elementType(), output.shape(), false,
        {debug_name_and_id, "tmp"});

    // Copy the indices into the host sliceable indices.
    seq.add(poplar::program::Copy(indices, host_sliceable.indices, false,
                                  {debug_name_and_id}));

    // Copy from the remote buffer.
    seq.add(poplar::program::Copy(remote_buffer, host_sliceable.tensor,
                                  host_sliceable.indices, {debug_name_and_id}));

    // Copy the values into the output tensor.
    seq.add(poplar::program::Copy(host_sliceable.tensor, output, false,
                                  {debug_name_and_id}));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));
    return seq;
  }

  // Replicated remote buffer embedding lookup using the token splitting
  // strategy.
  StatusOr<poplar::program::Program> RemoteBufferSplitTokensImpl(
      poplar::Graph& graph, poplar::RemoteBuffer& remote_buffer,
      poplar::Tensor indices, poplar::program::Sequence seq,
      CompilerResources& res, const HloHostEmbeddingLookupInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugNameAndId& debug_name_and_id) {
    // Create the output tensor.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor output,
        AddTensor(graph, TensorLocation{inst, 0}, output_shape, res, tensor_map,
                  {debug_name_and_id, "output"}));

    // All-Gather the indices from all replicas.
    indices =
        gcl::allGather(graph, indices, seq, {debug_name_and_id, "indices"},
                       GetReplicatedCollectiveOptions(res));

    // Create a replication factor constant tensor.
    poplar::Tensor rep =
        graph.addConstant(poplar::UNSIGNED_INT, {}, res.replication_factor,
                          {debug_name_and_id, "replication_factor"});
    MappingHelper::MapTensorLinearly(res.linear_mapping_state, graph, rep);

    // Divide the indices by the replication factor to transform the global
    // address space indices to the replica-local address space. Given we can
    // currently assume there is a power of two replication factor, this could
    // be rewritten as bitshift right log_2(rep) bits.
    poplar::Tensor ind = popops::div(graph, indices.flatten(), rep, seq,
                                     {debug_name_and_id, "shift_indices"});

    // Create the host sliceable temporary tensor.
    auto host_sliceable = popops::createHostSliceableTensor(
        graph, output.elementType(),
        {indices.dim(0) * indices.dim(1), output.dim(1)}, false,
        {debug_name_and_id, "tmp"});

    // Copy the replica-local indices to the host sliceable indices.
    seq.add(poplar::program::Copy(ind, host_sliceable.indices, false,
                                  {debug_name_and_id}));

    // Read from the remote buffer into the host sliceable tensor.
    seq.add(poplar::program::Copy(remote_buffer, host_sliceable.tensor,
                                  host_sliceable.indices, {debug_name_and_id}));

    // The current replica index.
    poplar::Tensor replica_id = graph.addReplicationIndexConstant();
    MappingHelper::MapTensorLinearly(res.linear_mapping_state, graph,
                                     replica_id);

    // Take the modulus of the global indices with the replication factor.
    // Given we can currently assume that the replication factor is a power of
    // two, this could be rewritten as a bitwise-and with rep-1.
    poplar::Tensor rem = popops::rem(graph, indices, rep, seq,
                                     {debug_name_and_id, "mask_indices"});

    // Check whether (i mod r) is equal to this replica id. This tells us
    // whether the element refered to by the global index is "owned" by this
    // replca.
    poplar::Tensor mask =
        popops::eq(graph, rem.reinterpret(replica_id.elementType()), replica_id,
                   seq, {debug_name_and_id, "mask"});

    // Create a constant zero fo masking.
    poplar::Tensor zero = graph.addConstant(host_sliceable.tensor.elementType(),
                                            {}, 0, {debug_name_and_id, "zero"});
    MappingHelper::MapTensorLinearly(res.linear_mapping_state, graph, zero);

    // Use the computed mask, based on whether this replica "owns" the requested
    // element, to zero the invalid elements of host_sliceable.tensor.
    popops::selectInPlace(graph, host_sliceable.tensor, zero,
                          mask.flatten().expand({1}), seq,
                          {debug_name_and_id, "mask_output"});
    host_sliceable.tensor = host_sliceable.tensor.reshape(
        {indices.dim(0), indices.dim(1), output.dim(1)});

    // Given that invalid elements will be zero and valid elements will contain
    // the correct value, we can replicatedReduceScatter sum to distribute the
    // results to the correct replica.
    host_sliceable.tensor = gcl::reduceScatter(
        graph, host_sliceable.tensor.flatten(), popops::CollectiveOperator::ADD,
        seq, {debug_name_and_id, "reduce_scatter"},
        GetReplicatedCollectiveOptions(res));

    // Copy the result to the output tensor.
    seq.add(poplar::program::Copy(host_sliceable.tensor.reshape(output.shape()),
                                  output, false, {debug_name_and_id}));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));
    return seq;
  }

  // Replicated remote buffer embedding lookup using the encoding splitting
  // strategy.
  StatusOr<poplar::program::Program> RemoteBufferSplitEncodingImpl(
      poplar::Graph& graph, poplar::RemoteBuffer& remote_buffer,
      poplar::Tensor indices, poplar::program::Sequence seq,
      CompilerResources& res, const HloHostEmbeddingLookupInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugNameAndId& debug_name_and_id) {
    // Create the output tensor.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor output,
        AddTensor(graph, TensorLocation{inst, 0}, output_shape, res, tensor_map,
                  {debug_name_and_id, "output"}));

    // All-Gather the indices from all replicas.
    indices =
        gcl::allGather(graph, indices, seq, {debug_name_and_id, "indices"},
                       GetReplicatedCollectiveOptions(res));

    // Create the host sliceable temporary tensor.
    auto host_sliceable = popops::createHostSliceableTensor(
        graph, output.elementType(),
        {indices.dim(0) * indices.dim(1),
         tensorflow::MathUtil::CeilOfRatio<unsigned>(output.dim(1),
                                                     res.replication_factor)},
        false, {debug_name_and_id, "tmp"});

    // Copy the indices to the host sliceable indices.
    seq.add(poplar::program::Copy(indices.flatten(), host_sliceable.indices,
                                  false, {debug_name_and_id}));

    // Copy from the remote buffer into the host sliceable tensor.
    seq.add(poplar::program::Copy(remote_buffer, host_sliceable.tensor,
                                  host_sliceable.indices, {debug_name_and_id}));
    host_sliceable.tensor = host_sliceable.tensor.reshapePartial(
        0, 1, {indices.dim(0), indices.dim(1)});

    // Exchange the columns from this replica back to their respective replicas.
    // We also recieve the columns we requested from the other replicas.
    host_sliceable.tensor =
        gcl::allToAll(graph, host_sliceable.tensor, seq,
                      {debug_name_and_id, "exchange_columns"},
                      GetReplicatedCollectiveOptions(res));

    // Dimshuffle and reshape back to the output shape.
    host_sliceable.tensor = host_sliceable.tensor.dimShuffle({1, 0, 2});
    host_sliceable.tensor = host_sliceable.tensor.flatten(1, 3);
    host_sliceable.tensor = host_sliceable.tensor.slice(0, output.dim(1), 1);

    // Copy to the output tensor.
    seq.add(poplar::program::Copy(host_sliceable.tensor, output, false,
                                  {debug_name_and_id}));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));
    return seq;
  }

  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "HostEmbeddingLookupOp");
    poplar::program::Sequence seq({}, debug_info);

    TF_ASSIGN_OR_RETURN(TensorVector indices,
                        FindInstructionInputTensors(tensor_map, res, inst, 0,
                                                    seq, {debug_info}, false));

    const HloHostEmbeddingLookupInstruction* host_embedding_inst =
        Cast<HloHostEmbeddingLookupInstruction>(inst);

    if (UseSyntheticDataFor(SyntheticDataCategory::HostEmbedding)) {
      return SyntheticImpl(graph, indices[0].reinterpret(poplar::UNSIGNED_INT),
                           seq, res, host_embedding_inst, output_shape,
                           tensor_map, {debug_info});
    }

    if (res.enable_experimental_remote_buffer_embedding) {
      VLOG(1) << "Using experimental remote buffer embedding lookup";

      TF_ASSIGN_OR_RETURN(
          auto rbuffer_holder,
          GetOrCreateRemoteBuffer(
              graph, res, inst, host_embedding_inst->EmbeddingId(),
              host_embedding_inst->EmbeddingShape(),
              host_embedding_inst->SplittingStrategy(), output_shape));

      auto rbuffer = rbuffer_holder->Get();
      if (res.replication_factor > 1) {
        if (host_embedding_inst->SplittingStrategy() ==
            HostEmbeddingSplittingStrategy::Token) {
          return RemoteBufferSplitTokensImpl(
              graph, rbuffer, indices[0].reinterpret(poplar::UNSIGNED_INT), seq,
              res, host_embedding_inst, output_shape, tensor_map, {debug_info});
        }

        if (host_embedding_inst->SplittingStrategy() ==
            HostEmbeddingSplittingStrategy::Encoding) {
          return RemoteBufferSplitEncodingImpl(
              graph, rbuffer, indices[0].reinterpret(poplar::UNSIGNED_INT), seq,
              res, host_embedding_inst, output_shape, tensor_map, {debug_info});
        }
      }

      return RemoteBufferImpl(
          graph, rbuffer, indices[0].reinterpret(poplar::UNSIGNED_INT), seq,
          res, host_embedding_inst, output_shape, tensor_map, {debug_info});
    }

    return CallbackImpl(graph, indices[0].reinterpret(poplar::UNSIGNED_INT),
                        seq, res, host_embedding_inst, output_shape, tensor_map,
                        {debug_info});
  }
};

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
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugNameAndId& debug_name_and_id) {
    res.annotations.host_embedding_update_infos.push_back(
        {inst->name(), inst->EmbeddingId(), inst->operand(2)->shape(),
         inst->operand(1)->shape()});

    auto index_buffer = graph.addDeviceToHostFIFO(
        inst->name() + inst->EmbeddingId() + "_indices", indices.elementType(),
        indices.numElements());

    auto grad_fifo =
        graph.addDeviceToHostFIFO(inst->name() + inst->EmbeddingId() + "_grads",
                                  grads.elementType(), grads.numElements());

    seq.add(poplar::program::Copy(indices, index_buffer, false,
                                  {debug_name_and_id}));
    seq.add(
        poplar::program::Copy(grads, grad_fifo, false, {debug_name_and_id}));

    return seq;
  }

  // Single replica remote buffer implementation.
  StatusOr<poplar::program::Program> RemoteBufferImpl(
      poplar::Graph& graph, poplar::RemoteBuffer& remote_buffer,
      poplar::Tensor grads, poplar::Tensor indices,
      poplar::program::Sequence seq, CompilerResources& res,
      const HloHostEmbeddingUpdateInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugNameAndId& debug_name_and_id) {
    // Create the temporary host sliceable tensor.
    auto host_sliceable = popops::createHostSliceableTensor(
        graph, grads.elementType(), grads.shape(), true,
        {debug_name_and_id, "tmp"});

    // Read the weights from the host.
    seq.add(poplar::program::Copy(indices, host_sliceable.indices, false,
                                  {debug_name_and_id}));
    seq.add(poplar::program::Copy(remote_buffer, host_sliceable.tensor,
                                  host_sliceable.indices, {debug_name_and_id}));

    // Apply the gradients.
    popops::addInPlace(graph, host_sliceable.tensor, grads, seq,
                       {debug_name_and_id, "apply_grads"});

    // Send the updated weights back to the host.
    seq.add(poplar::program::Copy(host_sliceable.tensor, remote_buffer,
                                  host_sliceable.indices, {debug_name_and_id}));

    return seq;
  }

  // Replicated remote buffer embedding lookup using the token splitting
  // strategy.
  StatusOr<poplar::program::Program> RemoteBufferSplitTokensImpl(
      poplar::Graph& graph, poplar::RemoteBuffer& remote_buffer,
      poplar::Tensor grads, poplar::Tensor indices,
      poplar::program::Sequence seq, CompilerResources& res,
      const HloHostEmbeddingUpdateInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugNameAndId& debug_name_and_id) {
    // All-Gather the indices from all replicas.
    indices =
        gcl::allGather(graph, indices, seq, {debug_name_and_id, "indices"},
                       GetReplicatedCollectiveOptions(res));
    indices = indices.flatten(0, 2);

    // All-Gather the grads from all replicas.
    grads = gcl::allGather(graph, grads, seq, {debug_name_and_id, "grads"},
                           GetReplicatedCollectiveOptions(res));
    grads = grads.flatten(0, 2);

    // Create the host sliceable temporary tensor.
    auto host_sliceable = popops::createHostSliceableTensor(
        graph, grads.elementType(), grads.shape(), true,
        {debug_name_and_id, "tmp"});

    // Copy the indices to the host sliceable indices.
    seq.add(poplar::program::Copy(indices, host_sliceable.indices, false,
                                  {debug_name_and_id}));

    // Create a replication factor constant tensor.
    poplar::Tensor rep =
        graph.addConstant(poplar::UNSIGNED_INT, {}, res.replication_factor,
                          {debug_name_and_id, "replication_factor"});
    MappingHelper::MapTensorLinearly(res.linear_mapping_state, graph, rep);

    // Divide the indices by the replication factor to transform the global
    // address space indices to the replica-local address space. Given we can
    // currently assume there is a power of two replication factor, this could
    // be rewritten as bitshift right log_2(rep) bits.
    popops::divInPlace(graph, host_sliceable.indices, rep, seq,
                       {debug_name_and_id, "shift_indices"});

    // Create an invalid constant index that is used to avoid clobbering valid
    // values in the remote buffer. Currently implemented as one-past-the-end
    // index. Might be replaced with another sentinel value later.
    poplar::Tensor invalid = graph.addConstant(
        poplar::UNSIGNED_INT, {},
        tensorflow::MathUtil::CeilOfRatio<unsigned>(
            inst->EmbeddingShape().dimensions(0), res.replication_factor),
        {debug_name_and_id, "invalid_index"});
    MappingHelper::MapTensorLinearly(res.linear_mapping_state, graph, invalid);

    // The current replica index.
    poplar::Tensor replica_id = graph.addReplicationIndexConstant();
    MappingHelper::MapTensorLinearly(res.linear_mapping_state, graph,
                                     replica_id);

    // Take the modulus of the global indices with the replication factor.
    // Given we can currently assume that the replication factor is a power of
    // two, this could be rewritten as a bitwise-and with rep-1.
    poplar::Tensor rem = popops::rem(graph, host_sliceable.indices, rep, seq,
                                     {debug_name_and_id, "mask_indices"});

    // Check whether (i mod r) is equal to this replica id. This tells us
    // whether the element refered to by the global index is "owned" by this
    // replca.
    poplar::Tensor mask =
        popops::eq(graph, rem.reinterpret(replica_id.elementType()), replica_id,
                   seq, {debug_name_and_id, "mask"});

    // Use the computed mask, based on whether this replica "owns" the requested
    // element, to invalidate the indices and avoid overwritting valid regions
    // of the replica-local table.
    popops::selectInPlace(graph,
                          host_sliceable.indices.reinterpret(poplar::INT),
                          invalid.reinterpret(poplar::INT), mask, seq,
                          {debug_name_and_id, "mask_indices"});

    // Read from the remote buffer into the host sliceable tensor.
    seq.add(poplar::program::Copy(remote_buffer, host_sliceable.tensor,
                                  host_sliceable.indices, {debug_name_and_id}));

    // Apply the gradients to the values read from the remote buffer.
    popops::addInPlace(graph, host_sliceable.tensor, grads, seq,
                       {debug_name_and_id, "apply_grads"});

    // Read from the host sliceable tensor into the remote buffer.
    seq.add(poplar::program::Copy(host_sliceable.tensor, remote_buffer,
                                  host_sliceable.indices, debug_name_and_id));

    return seq;
  }

  // Replicated remote buffer embedding lookup using the encoding splitting
  // strategy.
  StatusOr<poplar::program::Program> RemoteBufferSplitEncodingImpl(
      poplar::Graph& graph, poplar::RemoteBuffer& remote_buffer,
      poplar::Tensor grads, poplar::Tensor indices,
      poplar::program::Sequence seq, CompilerResources& res,
      const HloHostEmbeddingUpdateInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugNameAndId& debug_name_and_id) {
    // All-Gather the indices from all replicas.
    indices =
        gcl::allGather(graph, indices, seq, {debug_name_and_id, "indices"},
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
    grads = gcl::allToAll(graph, grads, seq,
                          {debug_name_and_id, "exchange_columns"},
                          GetReplicatedCollectiveOptions(res));
    grads = grads.flatten(0, 2);

    // Create the host sliceable temporary tensor.
    auto host_sliceable = popops::createHostSliceableTensor(
        graph, grads.elementType(),
        {indices.dim(0) * indices.dim(1), grads.dim(1)}, true,
        {debug_name_and_id, "tmp"});

    // Copy the replica-local indices to the host sliceable indices.
    seq.add(poplar::program::Copy(indices.flatten(), host_sliceable.indices,
                                  false, {debug_name_and_id}));

    // Copy from the remote buffer into the host sliceable tensor.
    seq.add(poplar::program::Copy(remote_buffer, host_sliceable.tensor,
                                  host_sliceable.indices, {debug_name_and_id}));

    // Apply the gradients to the read values.
    popops::addInPlace(graph, host_sliceable.tensor, grads, seq,
                       {debug_name_and_id, "apply_grads"});

    // Copy from the host sliceable tensor into the remote buffer.
    seq.add(poplar::program::Copy(host_sliceable.tensor, remote_buffer,
                                  host_sliceable.indices, {debug_name_and_id}));

    return seq;
  }

  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "HostEmbeddingUpdateOp");
    poplar::program::Sequence seq({}, debug_info);

    TF_ASSIGN_OR_RETURN(TensorVector grads,
                        FindInstructionInputTensors(tensor_map, res, inst, 1,
                                                    seq, {debug_info}, false));

    TF_ASSIGN_OR_RETURN(TensorVector indices,
                        FindInstructionInputTensors(tensor_map, res, inst, 2,
                                                    seq, {debug_info}, false));

    const HloHostEmbeddingUpdateInstruction* host_embedding_inst =
        Cast<HloHostEmbeddingUpdateInstruction>(inst);
    if (UseSyntheticDataFor(SyntheticDataCategory::HostEmbedding)) {
      return SyntheticImpl(seq);
    } else if (res.enable_experimental_remote_buffer_embedding) {
      VLOG(1) << "Using experimental remote buffer embedding update";

      TF_ASSIGN_OR_RETURN(
          auto rbuffer_holder,
          GetOrCreateRemoteBuffer(
              graph, res, inst, host_embedding_inst->EmbeddingId(),
              host_embedding_inst->EmbeddingShape(),
              host_embedding_inst->SplittingStrategy(), output_shape));

      poplar::RemoteBuffer rbuffer = rbuffer_holder->Get();
      if (res.replication_factor > 1) {
        if (host_embedding_inst->SplittingStrategy() ==
            HostEmbeddingSplittingStrategy::Token) {
          return RemoteBufferSplitTokensImpl(
              graph, rbuffer, grads[0],
              indices[0].reinterpret(poplar::UNSIGNED_INT), seq, res,
              host_embedding_inst, output_shape, tensor_map, {debug_info});
        }

        if (host_embedding_inst->SplittingStrategy() ==
            HostEmbeddingSplittingStrategy::Encoding) {
          return RemoteBufferSplitEncodingImpl(
              graph, rbuffer, grads[0],
              indices[0].reinterpret(poplar::UNSIGNED_INT), seq, res,
              host_embedding_inst, output_shape, tensor_map, {debug_info});
        }
      }

      return RemoteBufferImpl(graph, rbuffer, grads[0],
                              indices[0].reinterpret(poplar::UNSIGNED_INT), seq,
                              res, host_embedding_inst, output_shape,
                              tensor_map, {debug_info});
    } else {
      return CallbackImpl(
          graph, grads[0], indices[0].reinterpret(poplar::UNSIGNED_INT), seq,
          res, host_embedding_inst, output_shape, tensor_map, {debug_info});
    }

    return seq;
  }
};

REGISTER_POPLAR_OP(HostEmbeddingUpdate, HostEmbeddingUpdateOp);

class HostEmbeddingNotifyOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    const HloHostEmbeddingNotifyInstruction* host_embedding_inst =
        Cast<HloHostEmbeddingNotifyInstruction>(inst);

    PoplarOpDefDebugInfo debug_info(debug_context, "HostEmbeddingNotifyOp");
    poplar::program::Sequence seq({}, debug_info);

    // For synthetic data or remote buffers, there's no communication with the
    // host.
    if (UseSyntheticDataFor(SyntheticDataCategory::HostEmbedding) ||
        res.enable_experimental_remote_buffer_embedding) {
      return seq;
    }

    // Add the notify information to the annotations
    res.annotations.host_embedding_notify_infos.push_back(
        {inst->name(), host_embedding_inst->EmbeddingId(), {}, {}});

    // Create a poplar FIFO
    auto notify = graph.addDeviceToHostFIFO(
        inst->name() + host_embedding_inst->EmbeddingId() + "_notify",
        poplar::INT, 0);

    // Use a dummy tensor to "copy" to the FIFO
    poplar::Tensor t = graph.addVariable(poplar::INT, {0}, {debug_info});
    graph.setTileMapping(t, 0);
    seq.add(poplar::program::Copy(t, notify, false, debug_info));

    return seq;
  }
};

REGISTER_POPLAR_OP(HostEmbeddingNotify, HostEmbeddingNotifyOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
