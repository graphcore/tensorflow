/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <poplar/Graph.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {

namespace {
Status CreatePoplarH2DFIFO(
    CompilerResources& res, const HloInstruction* inst, int64 tuple_index,
    const xla::poplarplugin::PoplarFeedConfig& infeed_config,
    const std::string& handle, poplar::Graph& graph,
    poplar::Tensor& tensor_to_update, poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id) {
  TF_RETURN_IF_ERROR(res.streams_indices.InitializeFeedStream(
      infeed_config.feed_id(), tuple_index, handle, seq, inst,
      debug_name_and_id));

  poplar::OptionFlags fifo_options =
      res.streams_indices.GraphFeedOptions(handle);
  fifo_options.set("bufferingDepth",
                   std::to_string(infeed_config.prefetch_depth()));

  auto fifo = graph.addHostToDeviceFIFO(
      handle, tensor_to_update.elementType(), tensor_to_update.numElements(),
      poplar::ReplicatedStreamMode::REPLICATE, fifo_options);
  if (res.use_verified_transfers) {
    TF_ASSIGN_OR_RETURN(poplar::Tensor index,
                        res.streams_indices.IndexTensor(handle, inst, seq));
    seq.add(poplar::program::Copy(fifo, tensor_to_update, index, false,
                                  res.streams_indices.CopyOptions(),
                                  {debug_name_and_id}));
    // Increment the index by one.
    popops::mapInPlace(graph, pe::Add(pe::_1, pe::Const(1)),
                       {index.slice(0, 1)}, seq,
                       {debug_name_and_id, std::string("InfeedIndexInc/") +
                                               std::to_string(tuple_index)});
  } else {
    seq.add(poplar::program::Copy(fifo, tensor_to_update, false,
                                  debug_name_and_id));
  }

  return Status::OK();
}

Status CreateReusablePoplarH2DFIFO(
    CompilerResources& res, const HloInstruction* inst, int64 tuple_index,
    const xla::poplarplugin::PoplarFeedConfig& infeed_config,
    const std::string& handle, poplar::Graph& graph,
    poplar::Tensor& tensor_to_update, poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id) {
  // Is the stream registered in the cache?
  auto itr = res.infeed_cache.find(handle);
  if (itr != res.infeed_cache.end()) {
    // Reuse the cache program and copy the result into the tensor.
    seq.add(itr->second.first);
    seq.add(poplar::program::Copy(itr->second.second, tensor_to_update, false,
                                  debug_name_and_id));

    return Status::OK();
  }

  // Wasn't in the cache, so we'll create one.
  poplar::Tensor tmp = graph.clone(tensor_to_update, debug_name_and_id);
  TF_RETURN_IF_ERROR(CreatePoplarH2DFIFO(res, inst, tuple_index, infeed_config,
                                         handle, graph, tmp, seq,
                                         debug_name_and_id));

  // Add to the cache.
  res.infeed_cache[handle] = std::make_pair(seq, tmp);
  seq.add(
      poplar::program::Copy(tmp, tensor_to_update, false, debug_name_and_id));

  return Status::OK();
}

Status CreatePoplarD2HFIFO(
    CompilerResources& res, const HloInstruction* inst, int64 tuple_index,
    const xla::poplarplugin::PoplarFeedConfig& outfeed_config,
    const std::string& handle, poplar::Graph& graph, poplar::Tensor& in,
    poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id) {
  TF_RETURN_IF_ERROR(res.streams_indices.InitializeFeedStream(
      outfeed_config.feed_id(), tuple_index, handle, seq, inst,
      debug_name_and_id));
  auto fifo =
      graph.addDeviceToHostFIFO(handle, in.elementType(), in.numElements(),
                                res.streams_indices.GraphFeedOptions(handle));
  if (res.use_verified_transfers) {
    TF_ASSIGN_OR_RETURN(poplar::Tensor index,
                        res.streams_indices.IndexTensor(handle, inst, seq));

    seq.add(poplar::program::Copy(in, fifo, index, false,
                                  res.streams_indices.CopyOptions(),
                                  {debug_name_and_id}));
    // Increment the index by one.
    popops::mapInPlace(graph, pe::Add(pe::_1, pe::Const(1)),
                       {index.slice(0, 1)}, seq,
                       {debug_name_and_id, "OutfeedIndexInc"});
  } else {
    seq.add(poplar::program::Copy(in, fifo, false, {debug_name_and_id}));
  }

  return Status::OK();
}

Status CreateReusablePoplarD2HFIFO(
    CompilerResources& res, const HloInstruction* inst, int64 tuple_index,
    const xla::poplarplugin::PoplarFeedConfig& outfeed_config,
    const std::string& handle, poplar::Graph& graph, poplar::Tensor& in,
    poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id) {
  TF_RETURN_IF_ERROR(res.streams_indices.InitializeFeedStream(
      outfeed_config.feed_id(), tuple_index, handle, seq, inst,
      debug_name_and_id));

  // Is the stream registered in the cache?
  auto itr = res.outfeed_cache.find(handle);
  if (itr != res.outfeed_cache.end()) {
    // Reuse the cache program and copy the input into the tensor.
    seq.add(poplar::program::Copy(in, itr->second.second, false,
                                  {debug_name_and_id}));
    seq.add(itr->second.first);

    return Status::OK();
  }

  // Wasn't in the cache, so we'll create one.
  poplar::Tensor tmp = graph.clone(in, {debug_name_and_id});
  poplar::program::Sequence out_copy({}, debug_name_and_id);
  TF_RETURN_IF_ERROR(CreatePoplarD2HFIFO(res, inst, tuple_index, outfeed_config,
                                         handle, graph, tmp, out_copy,
                                         {debug_name_and_id}));

  // Add to the cache.
  seq.add(poplar::program::Copy(in, tmp, false, {debug_name_and_id}));
  seq.add(out_copy);
  res.outfeed_cache[handle] = std::make_pair(out_copy, tmp);

  return Status::OK();
}
}  // namespace

StatusOr<poplar::program::Program> CreateInfeed(
    CompilerResources& res, const HloInstruction* inst, int64 tuple_index,
    const xla::Shape& shape, poplar::Tensor tensor,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::program::Sequence seq({}, debug_name_and_id);
  const HloInfeedInstruction* infeed = Cast<HloInfeedInstruction>(inst);

  poplar::Graph& graph = GetGraph(res, inst);

  // Parse the infeed config to find out how much data to prefetch if at all.
  xla::poplarplugin::PoplarFeedConfig infeed_config;
  infeed_config.ParseFromString(infeed->infeed_config());

  // The amount of data the user has specified to be prefetched on each host
  // sync.
  size_t io_batch_size = std::max<size_t>(1, infeed_config.io_batch_size());
  // Only batch size 1 is supported for verified transfers as we need to
  // increment the index between each read.
  if (res.use_verified_transfers && io_batch_size != 1) {
    return InvalidArgument(
        "Only io_batch_size = 1 is supported for verified transfers.");
  }

  // A functor wrapper to either use synthetic data or copy from the host,
  // depending on the global synthetic flags.
  auto init_synthetic_or_copy = [&](poplar::program::Sequence& seq,
                                    const Shape& data_shape,
                                    poplar::Tensor& tensor_to_update) {
    const auto use_synthetic_data =
        UseSyntheticDataFor(SyntheticDataCategory::Infeed);
    if (!use_synthetic_data) {
      const std::string handle =
          GetInfeedCopyHandle(infeed_config.feed_id(), tuple_index);

      if (infeed_config.reusable()) {
        return CreateReusablePoplarH2DFIFO(
            res, inst, tuple_index, infeed_config, handle, graph,
            tensor_to_update, seq, debug_name_and_id);
      }

      return CreatePoplarH2DFIFO(res, inst, tuple_index, infeed_config, handle,
                                 graph, tensor_to_update, seq,
                                 debug_name_and_id);
    } else if (use_synthetic_data && UseSyntheticDataInitializer()) {
      // Initialize the tensor with a synthetic initalizer.
      auto& initializer = DataInitializer::GetSyntheticDataInitializer();
      TF_ASSIGN_OR_RETURN(auto literal, initializer.GetData(data_shape));
      TF_RETURN_IF_ERROR(
          SetInitialTensorValue(graph, tensor_to_update, literal));
    }
    // If neither case then we want synthetic data but don't want it initalized
    // so we just return the empty tensor unchanged.
    return Status::OK();
  };

  if (io_batch_size != 1) {
    // If the tensor is a scalar then we need to make sure the buffer is created
    // with the right shape.
    const bool is_scalar = tensor.rank() == 0;

    // Extend the old shape to add a new dimension for the batches of memory.
    std::vector<size_t> buffer_shape = tensor.shape();
    buffer_shape.insert(buffer_shape.begin(), io_batch_size);

    // Buffer slice shape - depends on whether the tensor is scalar.
    std::vector<size_t> slice_shape =
        is_scalar ? std::vector<size_t>({1}) : tensor.shape();

    std::vector<poplar::Tensor> cloned_tensors(io_batch_size);
    for (size_t i = 0; i < io_batch_size; ++i) {
      if (res.always_rearrange_copies_on_host) {
        // When rearranging on the host, it is better to keep the layout of the
        // slices in the output tensor layout, in order to minimise on-device
        // rearrangement.
        cloned_tensors[i] =
            graph.clone(tensor, {debug_name_and_id}).reshape(slice_shape);
      } else {
        // When rearranging on the device, it is better to rearrange after the
        // dynamic slice, so that the rearrangement only takes place on the
        // slice, not the whole incoming pegged_memory buffer.
        cloned_tensors[i] = graph.addVariable(
            tensor.elementType(), slice_shape,
            poplar::VariableMappingMethod::LINEAR, {debug_name_and_id});
      }
    }
    // Concatenate all the cloned tensors then reshape to make sure we are
    // in the shape [io_batch_size][original_shape].
    poplar::Tensor pegged_memory =
        poplar::concat(cloned_tensors).reshape(buffer_shape);

    // A counter for tracking the number of entries in the buffer
    poplar::Tensor counter =
        graph.addVariable(poplar::UNSIGNED_INT, {},
                          {debug_name_and_id, std::string("InfeedCtr/") +
                                                  std::to_string(tuple_index)});
    // Map counter to the next tile.
    MappingHelper::MapTensorLinearly(res.linear_mapping_state, graph, counter);
    AddZeroTensorToPreamble(res, counter, {debug_name_and_id});

    // The body for copying from host and zeroing the counter.
    poplar::program::Sequence true_body({}, {debug_name_and_id, "true_body"});

    // If we are using synthetic data, init pegged_memory with it otherwise host
    // copy. Either way we will have a tensor with some number of prefetched
    // batches and we will dynamic slice the actual batch from that. This is to
    // ensure that we can benchmark synthetic vs non-synthetic without changing
    // the graph too much.
    TF_RETURN_IF_ERROR(init_synthetic_or_copy(
        true_body,
        XlaShapeFromPoplarShape(shape.element_type(), pegged_memory.shape()),
        pegged_memory));

    // The NOP body.
    poplar::program::Sequence false_body({}, {debug_name_and_id, "false_body"});

    // Predicate for fetching the next batch
    poplar::Tensor predicate =
        popops::map(graph, pe::Equal(pe::_1, pe::Const(0)), {counter}, seq,
                    {debug_name_and_id, std::string("InfeedCtrCmp/") +
                                            std::to_string(tuple_index)});

    // The main body which contains the control flow for copy from host and
    // the dynamic slice.
    seq.add(poplar::program::If(predicate, true_body, false_body,
                                {debug_name_and_id}));

    // Use dynamic slice to extract the slices from the buffer
    poplar::Tensor slice = popops::dynamicSlice(
        graph, pegged_memory, counter.reshape({1}), {0}, {1}, seq,
        {debug_name_and_id,
         std::string("Slice/") + std::to_string(tuple_index)});
    seq.add(poplar::program::Copy(slice, tensor.reshape(slice_shape), false,
                                  {debug_name_and_id}));

    // Increment the counter by one.
    popops::mapInPlace(
        graph, pe::Rem(pe::Add(pe::_1, pe::Const(1)), pe::Const(io_batch_size)),
        {counter}, seq,
        {debug_name_and_id,
         std::string("InfeedCtrInc/") + std::to_string(tuple_index)});

  } else {
    // Just an normal copy from host->tensor or init tensor with synthetic data.
    TF_RETURN_IF_ERROR(init_synthetic_or_copy(seq, shape, tensor));
  }
  return seq;
}

StatusOr<poplar::program::Program> CreateOutfeed(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::program::Sequence seq({}, debug_name_and_id);
  poplar::Graph& graph = GetGraph(res, inst);

  const HloOutfeedInstruction* outfeed = Cast<HloOutfeedInstruction>(inst);
  xla::poplarplugin::PoplarFeedConfig outfeed_config;
  outfeed_config.ParseFromString(outfeed->outfeed_config());
  outfeed_config.set_replication_factor(res.local_replication_factor);

  size_t io_batch_size = std::max<size_t>(1, outfeed_config.io_batch_size());
  // Only batch size 1 is supported for verified transfers as we need to
  // increment the index between each write.
  if (res.use_verified_transfers && io_batch_size != 1) {
    return InvalidArgument(
        "Only io_batch_size = 1 is supported for verified transfers.");
  }

  FeedInfo info(outfeed_config.feed_id(), outfeed_config,
                outfeed->operands()[0]->shape());
  TF_RETURN_IF_ERROR(AddOutfeedInfo(res.annotations, info));

  if (UseSyntheticDataFor(SyntheticDataCategory::Outfeed)) {
    return seq;
  }

  HloInstruction* operand = outfeed->operands()[0];
  const Shape& shape = operand->shape();
  if (ShapeUtil::IsNestedTuple(shape)) {
    return InvalidArgument("Nested tuple shapes are not supported for outfeed");
  }

  const bool expand_aliasing = true;
  TF_ASSIGN_OR_RETURN(
      TensorVector input_tensors,
      FindInstructionInputTensors(tensor_map, res, inst, 0, seq,
                                  debug_name_and_id, expand_aliasing));

  for (unsigned i = 0; i < input_tensors.size(); ++i) {
    poplar::Tensor& in = input_tensors[i];
    const std::string handle =
        GetOutfeedCopyHandle(outfeed_config.feed_id(), i);

    if (io_batch_size == 1) {
      // Simply copy to the stream
      if (outfeed_config.reusable()) {
        TF_RETURN_IF_ERROR(
            CreateReusablePoplarD2HFIFO(res, inst, i, outfeed_config, handle,
                                        graph, in, seq, debug_name_and_id));
      } else {
        TF_RETURN_IF_ERROR(CreatePoplarD2HFIFO(res, inst, i, outfeed_config,
                                               handle, graph, in, seq,
                                               debug_name_and_id));
      }
    } else {
      if (outfeed_config.reusable()) {
        return UnimplementedStrCat(
            "Trying to reuse", inst->ToString(),
            " outfeed with an io batch size greater than 1.");
      }

      // Batch multiple writes, and then write as a block.

      // If the tensor is a scalar then we need to make sure the buffer is
      // created with the right shape.
      const bool is_scalar = in.rank() == 0;

      // Extend the old shape to add a new dimension for the batches of memory
      std::vector<size_t> buffer_shape = in.shape();
      buffer_shape.insert(buffer_shape.begin(), io_batch_size);

      // Buffer slice shape - depends on whether the tensor is scalar.
      std::vector<size_t> slice_shape =
          is_scalar ? std::vector<size_t>({1}) : in.shape();

      std::vector<poplar::Tensor> cloned_tensors(io_batch_size);
      for (size_t i = 0; i < io_batch_size; ++i) {
        if (res.always_rearrange_copies_on_host) {
          // When rearranging on the host it is better to have the slices of the
          // buffer laid out in the same form as the 'in' tensor so that there
          // is no cost of rearrangement.
          cloned_tensors[i] =
              graph.clone(in, {debug_name_and_id}).reshape(slice_shape);
        } else {
          // When the data is rearranged on the device, it is beter to have the
          // slices arranged in the standard order of the host buffer, and then
          // to have the rearragement done only once, during the dynamicUpdate.
          cloned_tensors[i] = graph.addVariable(
              in.elementType(), slice_shape,
              poplar::VariableMappingMethod::LINEAR, {debug_name_and_id});
        }
      }
      poplar::Tensor batched =
          poplar::concat(cloned_tensors).reshape(buffer_shape);

      //  A counter for counting slots
      poplar::Tensor counter = graph.addVariable(
          poplar::UNSIGNED_INT, {},
          {debug_name_and_id, std::string("OutfeedCtr/") + std::to_string(i)});
      // Map counter to the next tile.
      MappingHelper::MapTensorLinearly(res.linear_mapping_state, graph,
                                       counter);
      AddZeroTensorToPreamble(res, counter, debug_name_and_id);

      // Use dynamic slice update to put the slices into the buffer
      popops::dynamicUpdate(
          graph, batched, in.expand({0}), counter.reshape({1}), {0}, {1}, seq,
          {debug_name_and_id, std::string("Slice/") + std::to_string(i)});

      // Increment the counter by one.
      popops::mapInPlace(
          graph,
          pe::Rem(pe::Add(pe::_1, pe::Const(1)), pe::Const(io_batch_size)),
          {counter}, seq,
          {debug_name_and_id,
           std::string("OutfeedCtrInc/") + std::to_string(i)});

      // The body for copying to host and zeroing the counter.
      poplar::program::Sequence true_body({}, debug_name_and_id);

      // Copy the data to the host
      auto fifo = graph.addDeviceToHostFIFO(handle, batched.elementType(),
                                            batched.numElements());
      true_body.add(
          poplar::program::Copy(batched, fifo, false, {debug_name_and_id}));

      // The NOP body.
      poplar::program::Sequence false_body({}, debug_name_and_id);

      // Check the counter doesn't equal
      poplar::Tensor predicate =
          popops::map(graph, pe::Equal(pe::_1, pe::Const(0)), {counter}, seq,
                      {debug_name_and_id,
                       std::string("OutfeedCtrCmp/") + std::to_string(i)});

      // The main body which contains the control flow for copy from host and
      // the dynamic slice.
      seq.add(poplar::program::If(predicate, true_body, false_body,
                                  {debug_name_and_id}));
    }
  }

  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
