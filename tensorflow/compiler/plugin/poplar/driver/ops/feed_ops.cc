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
#include <popops/ElementWise.hpp>
#include <popops/HostSliceTensor.hpp>

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

DriverTensor CreateTemporary(DriverGraph& graph,
                             const xla::poplarplugin::PoplarFeedConfig& config,
                             const DriverTensor& target, const bool is_read,
                             const poplar::DebugNameAndId& debug_name_and_id) {
  if (config.optimise_latency()) {
    return DriverTensor(popops::createHostTransferableTensor(
                            graph, target.elementType(), target.shape(),
                            is_read, {debug_name_and_id}),
                        graph);
  }
  return graph.clone(target, {debug_name_and_id});
}

Status CreatePoplarH2DFIFO(
    CompilerResources& res, const HloInstruction* inst, int64_t tuple_index,
    const Shape& shape,
    const xla::poplarplugin::PoplarFeedConfig& infeed_config,
    const std::string& handle, DriverGraph& graph,
    DriverTensor& tensor_to_update, ExternalAndLocalTransferSequence& seqs,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::OptionFlags fifo_options;
  fifo_options.set("bufferingDepth",
                   std::to_string(infeed_config.prefetch_depth()));

  auto fifo = graph.addHostToDeviceFIFO(
      handle, tensor_to_update.elementType(), tensor_to_update.numElements(),
      poplar::ReplicatedStreamMode::REPLICATE, fifo_options);

  CHECK(!(infeed_config.optimise_latency() && infeed_config.reusable()));
  if (infeed_config.optimise_latency()) {
    // Create a tensor that is well laid out for the host exchange
    // and then copy from that to desired output
    DriverTensor tmp = CreateTemporary(graph, infeed_config, tensor_to_update,
                                       /*is_read=*/false, debug_name_and_id);
    seqs.external_transfer.add(
        poplar::program::Copy(fifo, tmp, false, {debug_name_and_id}));
    seqs.local_transfer.add(
        DriverProgramCopy(tmp, tensor_to_update, false, {debug_name_and_id}));
  } else {
    seqs.external_transfer.add(
        DriverProgramCopy(fifo, tensor_to_update, false, debug_name_and_id));
  }

  InputInfo info = {handle, handle, 0, tuple_index, shape};
  TF_RETURN_IF_ERROR(AddFeedInputInfo(res.annotations, info));

  return Status::OK();
}

Status CreateReusablePoplarH2DFIFO(
    CompilerResources& res, const HloInstruction* inst, int64_t tuple_index,
    const Shape& shape,
    const xla::poplarplugin::PoplarFeedConfig& infeed_config,
    const std::string& handle, DriverGraph& graph,
    DriverTensor& tensor_to_update, ExternalAndLocalTransferSequence& seqs,
    const poplar::DebugNameAndId& debug_name_and_id) {
  // Is the stream registered in the cache?
  auto itr = res.infeed_cache.find(handle);
  if (itr != res.infeed_cache.end()) {
    // Reuse the cache program and copy the result into the tensor.
    seqs.external_transfer.add(itr->second.first);
    seqs.local_transfer.add(poplar::program::Copy(
        itr->second.second, tensor_to_update, false, debug_name_and_id));

    return Status::OK();
  }

  // Wasn't in the cache, so we'll create one.
  DriverTensor tmp = CreateTemporary(graph, infeed_config, tensor_to_update,
                                     /*is_read=*/false, debug_name_and_id);

  xla::poplarplugin::PoplarFeedConfig internal_config = infeed_config;
  // already created the temporary so don't need to create it inside
  internal_config.set_optimise_latency(false);
  TF_RETURN_IF_ERROR(CreatePoplarH2DFIFO(res, inst, tuple_index, shape,
                                         internal_config, handle, graph, tmp,
                                         seqs, debug_name_and_id));

  // Add to the cache.
  res.infeed_cache.insert_or_assign(
      handle, std::make_pair(seqs.external_transfer, tmp));
  seqs.local_transfer.add(
      poplar::program::Copy(tmp, tensor_to_update, false, debug_name_and_id));

  return Status::OK();
}

Status CreatePoplarD2HFIFO(
    CompilerResources& res, const HloInstruction* inst, int64_t tuple_index,
    const xla::poplarplugin::PoplarFeedConfig& outfeed_config,
    const std::string& handle, DriverGraph& graph, DriverTensor& in,
    ExternalAndLocalTransferSequence& seqs,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::OptionFlags fifo_options;
  fifo_options.set("bufferingDepth",
                   std::to_string(outfeed_config.prefetch_depth()));

  auto fifo = graph.addDeviceToHostFIFO(handle, in.elementType(),
                                        in.numElements(), fifo_options);

  CHECK(!(outfeed_config.optimise_latency() && outfeed_config.reusable()));
  if (outfeed_config.optimise_latency()) {
    DriverTensor tmp = CreateTemporary(graph, outfeed_config, in,
                                       /*is_read=*/true, debug_name_and_id);
    seqs.local_transfer.add(
        DriverProgramCopy(in, tmp, false, {debug_name_and_id}));
    seqs.external_transfer.add(
        DriverProgramCopy(tmp, fifo, false, {debug_name_and_id}));
  } else {
    seqs.external_transfer.add(
        DriverProgramCopy(in, fifo, false, {debug_name_and_id}));
  }
  auto* op = inst->operand(0);
  const auto& op_shape = op->shape();

  auto fifo_shape =
      op_shape.IsTuple() ? op_shape.tuple_shapes(tuple_index) : op_shape;
  OutputInfo info = {handle, handle, tuple_index, fifo_shape};
  TF_RETURN_IF_ERROR(AddFeedOutputInfo(res.annotations, info));

  return Status::OK();
}

Status CreateReusablePoplarD2HFIFO(
    CompilerResources& res, const HloInstruction* inst, int64_t tuple_index,
    const xla::poplarplugin::PoplarFeedConfig& outfeed_config,
    const std::string& handle, DriverGraph& graph, DriverTensor& in,
    ExternalAndLocalTransferSequence& seqs,
    const poplar::DebugNameAndId& debug_name_and_id) {
  // Is the stream registered in the cache?
  auto itr = res.outfeed_cache.find(handle);
  if (itr != res.outfeed_cache.end()) {
    // Reuse the cache program and copy the input into the tensor.
    seqs.local_transfer.add(poplar::program::Copy(in, itr->second.second, false,
                                                  {debug_name_and_id}));
    seqs.external_transfer.add(itr->second.first);

    return Status::OK();
  }

  // Wasn't in the cache, so we'll create one.
  DriverTensor tmp = CreateTemporary(graph, outfeed_config, in,
                                     /*is_read=*/true, debug_name_and_id);

  ExternalAndLocalTransferSequence external_copy(graph);

  xla::poplarplugin::PoplarFeedConfig internal_config = outfeed_config;
  // already created the temporary so don't need to create it inside
  internal_config.set_optimise_latency(false);

  TF_RETURN_IF_ERROR(CreatePoplarD2HFIFO(res, inst, tuple_index,
                                         internal_config, handle, graph, tmp,
                                         external_copy, {debug_name_and_id}));

  // Add to the cache.
  seqs.local_transfer.add(
      DriverProgramCopy(in, tmp, false, {debug_name_and_id}));
  seqs.external_transfer.add(external_copy.external_transfer);
  res.outfeed_cache.insert_or_assign(
      handle, std::make_pair(external_copy.external_transfer, tmp));

  return Status::OK();
}
}  // namespace

StatusOr<ExternalAndLocalTransferSequence> CreateInfeed(
    CompilerResources& res, const HloInstruction* inst, int64_t tuple_index,
    const xla::Shape& shape, DriverTensor tensor,
    const poplar::DebugNameAndId& debug_name_and_id) {
  auto& graph = GetGraph(res, inst);
  ExternalAndLocalTransferSequence seqs = {
      DriverProgramSequence(graph, {debug_name_and_id, "ExternalSequence"}),
      DriverProgramSequence(graph, {debug_name_and_id, "LocalSequence"})};
  const HloInfeedInstruction* infeed = Cast<HloInfeedInstruction>(inst);

  // Parse the infeed config to find out how much data to prefetch if at all.
  xla::poplarplugin::PoplarFeedConfig infeed_config;
  infeed_config.ParseFromString(infeed->infeed_config());

  const auto use_synthetic_data =
      UseSyntheticDataFor(SyntheticDataCategory::Infeed);
  if (!use_synthetic_data) {
    const std::string handle =
        GetInfeedCopyHandle(infeed_config.feed_id(), tuple_index);

    if (infeed_config.reusable()) {
      TF_RETURN_IF_ERROR(CreateReusablePoplarH2DFIFO(
          res, inst, tuple_index, shape, infeed_config, handle, graph, tensor,
          seqs, debug_name_and_id));
    } else {
      TF_RETURN_IF_ERROR(CreatePoplarH2DFIFO(res, inst, tuple_index, shape,
                                             infeed_config, handle, graph,
                                             tensor, seqs, debug_name_and_id));
    }
  } else if (use_synthetic_data && UseSyntheticDataInitializer()) {
    // Initialize the tensor with a synthetic initalizer.
    auto& initializer = DataInitializer::GetSyntheticDataInitializer();
    TF_ASSIGN_OR_RETURN(auto literal, initializer.GetData(shape));

    DriverTensor d(tensor, graph);
    TF_RETURN_IF_ERROR(SetInitialTensorValue(graph, d, literal));
  }
  return seqs;
}

StatusOr<ExternalAndLocalTransferSequence> CreateOutfeed(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  auto& graph = GetGraph(res, inst);
  ExternalAndLocalTransferSequence seqs = {
      DriverProgramSequence(graph, {debug_name_and_id, "ExternalSequence"}),
      DriverProgramSequence(graph, {debug_name_and_id, "LocalSequence"})};

  const HloOutfeedInstruction* outfeed = Cast<HloOutfeedInstruction>(inst);
  xla::poplarplugin::PoplarFeedConfig outfeed_config;
  outfeed_config.ParseFromString(outfeed->outfeed_config());

  CanonicalFeedInfo info(outfeed_config, outfeed->operands()[0]->shape());
  TF_RETURN_IF_ERROR(AddOutfeedInfo(res.annotations, info));

  if (UseSyntheticDataFor(SyntheticDataCategory::Outfeed)) {
    return seqs;
  }

  HloInstruction* operand = outfeed->operands()[0];
  const Shape& shape = operand->shape();
  if (ShapeUtil::IsNestedTuple(shape)) {
    return InvalidArgument("Nested tuple shapes are not supported for outfeed");
  }

  const bool expand_aliasing = true;
  TF_ASSIGN_OR_RETURN(
      TensorVector input_tensors,
      FindInstructionInputTensors(tensor_map, res, inst, 0, seqs.local_transfer,
                                  debug_name_and_id, expand_aliasing));

  for (unsigned i = 0; i < input_tensors.size(); ++i) {
    DriverTensor& in = input_tensors[i];
    const std::string handle =
        GetOutfeedCopyHandle(outfeed_config.feed_id(), i);
    // Simply copy to the stream
    if (outfeed_config.reusable()) {
      TF_RETURN_IF_ERROR(
          CreateReusablePoplarD2HFIFO(res, inst, i, outfeed_config, handle,
                                      graph, in, seqs, debug_name_and_id));
    } else {
      TF_RETURN_IF_ERROR(CreatePoplarD2HFIFO(res, inst, i, outfeed_config,
                                             handle, graph, in, seqs,
                                             debug_name_and_id));
    }
  }

  return seqs;
}

}  // namespace poplarplugin
}  // namespace xla
