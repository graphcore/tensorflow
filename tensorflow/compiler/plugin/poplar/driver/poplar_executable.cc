/* Copyright 2017 Graphcore Ltd
 */

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/poplar_executable.h"

#include <fstream>
#include <utility>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executable.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/tools/poplar_executable_runner.h"

namespace xla {
namespace poplarplugin {

PoplarExecutable::PoplarExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloProfilePrinterData> profile_printer,
    std::unique_ptr<HloProfileIndexMap> profile_index_map,
    std::unique_ptr<poplar::Engine> engine,
    const InputOutputAliasingMap& input_output_aliasing_map,
    const bool is_constant_graph,
    std::vector<std::vector<Literal>> literal_output, const bool is_remap_graph,
    const bool is_scalar_elementwise_graph, std::vector<uint64> remaped_output,
    uint32 replication_factor, const InfeedInfos& infeed_infos,
    const OutfeedInfos& outfeed_infos, StreamInfos&& stream_infos,
    StreamMetaInfos&& stream_meta_info, SendRecvInfos&& send_infos,
    SendRecvInfos&& recv_infos,
    HostEmbeddingInfos&& host_embedding_lookup_infos,
    HostEmbeddingInfos&& host_embedding_update_infos,
    RemoteParameterInfos&& remote_parameter_infos,
    const VerifiedStreamsIndices::KeyIdMappings& key_id_mappings,
    const std::vector<string>& checkpoint_feeds_order)
    : Executable(std::move(hlo_module), std::move(profile_printer),
                 std::move(profile_index_map)),
      poplar_engine_(std::move(engine)),
      input_output_aliasing_map_(std::move(input_output_aliasing_map)),
      literal_output_(std::move(literal_output)),
      is_constant_graph_(is_constant_graph),
      remaped_output_(std::move(remaped_output)),
      is_remap_graph_(is_remap_graph),
      is_scalar_elementwise_graph_(is_scalar_elementwise_graph),
      execution_count_(0),
      replication_factor_(replication_factor),
      infeed_infos_(std::move(infeed_infos)),
      outfeed_infos_(std::move(outfeed_infos)),
      stream_infos_(std::move(stream_infos)),
      stream_meta_infos_(std::move(stream_meta_info)),
      send_infos_(std::move(send_infos)),
      recv_infos_(std::move(recv_infos)),
      host_embedding_lookup_infos_(std::move(host_embedding_lookup_infos)),
      host_embedding_update_infos_(std::move(host_embedding_update_infos)),
      remote_parameter_infos_(std::move(remote_parameter_infos)),
      loaded_from_cache_(false),
      key_id_mappings_(key_id_mappings),
      checkpoint_feeds_order_(checkpoint_feeds_order) {}

PoplarExecutable::~PoplarExecutable() {
  if (poplar_engine_.get() != nullptr) {
    auto platform =
        se::MultiPlatformManager::PlatformWithName(tensorflow::PLATFORM_NAME);
    if (platform.ok()) {
      auto* p = static_cast<PoplarPlatform*>(platform.ValueOrDie());
      p->AboutToFreeEngine(poplar_engine_.get());
    }
  }
}

StatusOr<ScopedShapedBuffer> PoplarExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  se::Stream* stream = run_options->stream();

  std::vector<se::DeviceMemoryBase> argument_buffers;
  for (size_t i = 0; i < arguments.size(); ++i) {
    argument_buffers.push_back(arguments[i]->buffer(/*index=*/{}));
  }

  VLOG(1) << "Execute " << module().name();
  if (VLOG_IS_ON(2)) {
    for (const auto& a : argument_buffers) {
      VLOG(2) << "-- argument " << a.opaque();
    }
  }

  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  perftools::gputools::StreamExecutor* executor(stream->parent());
  PoplarExecutor* poplarExecutor(
      static_cast<PoplarExecutor*>(executor->implementation()));

  if (!poplarExecutor->PoplarDeviceIsAttached() &&
      poplar_engine_.get() != nullptr) {
    if (poplarExecutor->ConnectionType() == IpuDeviceConnectionType::NEVER) {
      return InvalidArgument(
          "Trying to run an executable on a device that was configured for "
          "compilation only.");
    }

    TF_RETURN_IF_ERROR(poplarExecutor->AttachToPoplarDevice());
  }

  if (poplar_engine_.get() != nullptr &&
      poplarExecutor->UseVerifiedTransfers()) {
    return InvalidArgument(
        "Executables using verified transfers can't be run "
        "in Tensorflow");
  }

  se::DeviceMemoryAllocator* memory_allocator = run_options->allocator();

  se::DeviceMemoryBase result;
  PoplarExecutor::AsPoplarStream(stream)->BlockUntilDone();
  TF_ASSIGN_OR_RETURN(
      result, poplarExecutor->ExecuteEngine(executor, *this, memory_allocator,
                                            argument_buffers));

  execution_count_++;
  if (poplarExecutor->ReportEventNthExecution() > 0 &&
      execution_count_ >= poplarExecutor->ReportEventNthExecution()) {
    execution_count_ = 0;
  }

  uint64 end_micros = tensorflow::Env::Default()->NowMicros();

  if (run_options->run_options().execution_profile()) {
    auto profile = run_options->run_options().execution_profile();
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    profile->set_compute_time_ns(std::max(nanoseconds, 1.0));
    profile->set_compute_cycle_count(1);
  }

  ScopedShapedBuffer result_buffer(result_shape(), result_shape(),
                                   run_options->allocator(),
                                   stream->parent()->device_ordinal());

  // Copy DeviceMemoryBase values which contain the array(s) of the result into
  // the respective location in ShapedBuffer which is returned to the caller.

  TF_RETURN_IF_ERROR(result_buffer.buffers().ForEachMutableElementWithStatus(
      [&result, poplarExecutor](const ShapeIndex& index,
                                se::DeviceMemoryBase* device_memory) {
        se::DeviceMemoryBase buffer = result;
        for (auto i : index) {
          TF_ASSIGN_OR_RETURN(buffer,
                              poplarExecutor->GetTupleBufferByIndex(buffer, i));
        }
        CHECK(!buffer.is_null() || buffer.size() == 0);
        if (VLOG_IS_ON(2)) {
          VLOG(2) << "-- return " << buffer.opaque();
        }
        *device_memory = buffer;
        return Status::OK();
      }));

  return std::move(result_buffer);
}

/*static*/ int64 PoplarExecutable::ShapeSizeBytes(const Shape& shape) {
  if (shape.IsOpaque()) {
    return sizeof(void*);
  }
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}

/*static*/ StatusOr<PoplarExecutable*> PoplarExecutable::Deserialize(
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloProfilePrinterData> profile_printer,
    std::unique_ptr<HloProfileIndexMap> profile_index_map,
    const ModuleFilenames& filenames) {
  PoplarExecutableProto proto;

  TF_RETURN_IF_ERROR(ReadBinaryProto(tensorflow::Env::Default(),
                                     filenames.CachedEngineFilename(), &proto));

  // Load metadata
  int replication_factor = proto.replication_factor();

  InfeedInfos infeeds;
  for (const auto& infeed : proto.infeeds()) {
    infeeds.emplace_back(infeed.stream_prefix(), infeed.config(),
                         Shape(infeed.shape()));
  }

  OutfeedInfos outfeeds;
  for (const auto& outfeed : proto.outfeeds()) {
    outfeeds.emplace_back(outfeed.stream_prefix(), outfeed.config(),
                          Shape(outfeed.shape()));
  }

  SendRecvInfos sends;
  for (const auto& send : proto.sends()) {
    sends.emplace_back(send.stream_handle(), send.rendezvous_key(),
                       Shape(send.shape()), send.concat_replicas());
  }

  SendRecvInfos recvs;
  for (const auto& recv : proto.recvs()) {
    recvs.emplace_back(recv.stream_handle(), recv.rendezvous_key(),
                       Shape(recv.shape()));
  }

  HostEmbeddingInfos lookups;
  for (const auto& lookup : proto.lookups()) {
    lookups.emplace_back(lookup.stream_handle(), lookup.embedding_id(),
                         Shape(lookup.indices_shape()),
                         Shape(lookup.activations_shape()));
  }

  HostEmbeddingInfos updates;
  for (const auto& update : proto.updates()) {
    updates.emplace_back(update.stream_handle(), update.embedding_id(),
                         Shape(update.indices_shape()),
                         Shape(update.activations_shape()));
  }

  RemoteParameterInfos remote_parameter_infos;
  for (const auto& remote_parameter : proto.remote_parameters()) {
    remote_parameter_infos.emplace(
        RemoteParameterInfo{remote_parameter.parameter_number()});
  }

  // Load the poplar compilation options from the serialized executable
  poplar::OptionFlags opts;
  for (const auto& flag : proto.option_flags()) {
    opts.set(flag.option(), flag.value());
  }

  VerifiedStreamsIndices::KeyIdMappings key_id_mappings;
  for (const auto& mapping : proto.key_id_mappings()) {
    key_id_mappings.emplace(
        mapping.handle(),
        VerifiedStreamsIndices::KeyIdPair(mapping.key(), mapping.start_id()));
  }

  std::vector<std::string> checkpoint_feeds_order;
  for (auto feed : proto.checkpoint_feeds_order()) {
    checkpoint_feeds_order.push_back(feed);
  }

  // Load the executable
  std::string poplar_executable_filename = proto.engine();
  if (poplar_executable_filename != filenames.CachedExecutableFilename()) {
    return tensorflow::errors::InvalidArgument(
        "Filename mismatch between module expected filename '",
        filenames.CachedExecutableFilename(), "' and file stored in proto '",
        poplar_executable_filename, "'");
  }
  std::unique_ptr<poplar::Engine> engine;
  try {
    std::ifstream file(poplar_executable_filename, std::ios::binary);
    auto poplar_executable = poplar::Executable::deserialize(file);
    engine.reset(new poplar::Engine(std::move(poplar_executable), opts));
  } catch (const std::exception& e) {
    return PoplarExceptionToTensorflowStatus("[Deserialize] ", e);
  }

  auto iomap = InputOutputAliasingMap(hlo_module.get());

  auto executable = new PoplarExecutable(
      std::move(hlo_module), std::move(profile_printer),
      std::move(profile_index_map), std::move(engine), std::move(iomap), false,
      {}, false, false, {}, replication_factor, std::move(infeeds),
      std::move(outfeeds), {}, {}, std::move(sends), std::move(recvs),
      std::move(lookups), std::move(updates), std::move(remote_parameter_infos),
      key_id_mappings, checkpoint_feeds_order);

  executable->loaded_from_cache_ = true;

  return executable;
}
namespace {
Status ExportInternal(const ModuleFilenames& filenames,
                      const poplar::Executable& executable,
                      const InfeedInfos& infeeds, const OutfeedInfos& outfeeds,
                      const SendRecvInfos& sends, const SendRecvInfos& recvs,
                      const HostEmbeddingInfos& lookups,
                      const HostEmbeddingInfos& updates,
                      const InputOutputAliasingMap& io_map,
                      uint32 replication_count, const poplar::OptionFlags& opts,
                      const poplar::Target& target,
                      const VerifiedStreamsIndices::KeyIdMappings& indices,
                      const std::vector<string> checkpoint_feeds_order) {
  if (!sends.empty()) {
    return tensorflow::errors::FailedPrecondition(
        "Failed to export the PoplarExecutable because it contains Sends "
        "operations.");
  }
  if (!recvs.empty()) {
    return tensorflow::errors::FailedPrecondition(
        "Failed to export the PoplarExecutable because it contains Receives "
        "operations.");
  }
  if (!lookups.empty()) {
    return tensorflow::errors::FailedPrecondition(
        "Failed to export the PoplarExecutable because it contains Host "
        "embedding lookups.");
  }
  if (!updates.empty()) {
    return tensorflow::errors::FailedPrecondition(
        "Failed to export the PoplarExecutable because it contains Host "
        "embedding updates.");
  }

  // Write poplar executable to a file
  try {
    TF_ASSIGN_OR_RETURN(std::string json_metadata,
                        CreateExecutableMetadataJson(
                            io_map, infeeds, outfeeds, replication_count, opts,
                            target, indices, checkpoint_feeds_order));
    TF_ASSIGN_OR_RETURN(std::string json_metadata_no_verif,
                        CreateExecutableMetadataJson(
                            io_map, infeeds, outfeeds, replication_count, opts,
                            target, {}, checkpoint_feeds_order));
    // For security reasons don't store the verification information inside the
    // binary.
    ipu::BinaryWriter writer(filenames.SerializedExecutableFilename());
    writer.WriteMetadata(filenames.Name(), json_metadata_no_verif);
    {
      ipu::ExecutableWriter exec_writer =
          writer.CreateExecutable(filenames.Name());
      executable.serialize(exec_writer.Stream());
    }
    writer.Close();

    std::unique_ptr<tensorflow::WritableFile> file;
    TF_RETURN_IF_ERROR(tensorflow::Env::Default()->NewWritableFile(
        filenames.SerializedMetadataFilename(), &file));
    TF_RETURN_IF_ERROR(file->Append(json_metadata));
    TF_RETURN_IF_ERROR(file->Close());
  } catch (const std::exception& e) {
    return PoplarExceptionToTensorflowStatus("[Serialize] ", e);
  }
  return Status::OK();
}

}  // namespace
/*static*/ Status PoplarExecutable::Export(const ModuleFilenames& filenames,
                                           const poplar::Executable& executable,
                                           const CompilerResources& resources,
                                           uint32 replication_count,
                                           const poplar::OptionFlags& opts,
                                           const poplar::Target& target) {
  return ExportInternal(
      filenames, executable, resources.annotations.infeed_infos,
      resources.annotations.outfeed_infos, resources.annotations.send_infos,
      resources.annotations.recv_infos,
      resources.annotations.host_embedding_lookup_infos,
      resources.annotations.host_embedding_update_infos,
      resources.annotations.input_output_aliasing_map, replication_count, opts,
      target, resources.streams_indices.GetAssignedIds(),
      resources.streams_indices.CheckpointFeedsOrder());
}

/*static*/ Status PoplarExecutable::Export(
    const ModuleFilenames& filenames, const poplar::Executable& executable,
    const PoplarExecutable& poplar_executable, const poplar::OptionFlags& opts,
    const poplar::Target& target) {
  return ExportInternal(
      filenames, executable, poplar_executable.GetInfeedInfos(),
      poplar_executable.GetOutfeedInfos(), poplar_executable.GetSendInfos(),
      poplar_executable.GetRecvInfos(),
      poplar_executable.GetHostEmbeddingLookupInfos(),
      poplar_executable.GetHostEmbeddingUpdateInfos(),
      poplar_executable.GetInputOutputAliasingMap(),
      poplar_executable.GetReplicationFactor(), opts, target,
      poplar_executable.KeyIdMappings(),
      poplar_executable.CheckpointFeedsOrder());
}
/*static*/ Status PoplarExecutable::Serialize(
    const ModuleFilenames& filenames, const poplar::Executable& executable,
    const CompilerAnnotations& annotations, uint32 replication_count,
    const poplar::OptionFlags& opts,
    const VerifiedStreamsIndices::KeyIdMappings& mappings,
    const std::vector<string>& checkpoint_feeds_order) {
  PoplarExecutableProto proto;

  // Write poplar executable to a file
  try {
    auto file =
        std::ofstream(filenames.CachedExecutableFilename(), std::ios::binary);
    executable.serialize(file);
  } catch (const std::exception& e) {
    return PoplarExceptionToTensorflowStatus("[Serialize] ", e);
  }

  proto.set_engine(filenames.CachedExecutableFilename());

  proto.set_replication_factor(replication_count);

  for (const auto& infeed : annotations.infeed_infos) {
    auto* feed = proto.add_infeeds();
    feed->set_stream_prefix(infeed.stream_prefix);
    *(feed->mutable_config()) = infeed.config;
    *(feed->mutable_shape()) = infeed.shape.ToProto();
  }

  for (const auto& outfeed : annotations.outfeed_infos) {
    auto* feed = proto.add_outfeeds();
    feed->set_stream_prefix(outfeed.stream_prefix);
    *(feed->mutable_config()) = outfeed.config;
    *(feed->mutable_shape()) = outfeed.shape.ToProto();
  }

  for (const auto& send : annotations.send_infos) {
    auto* send_proto = proto.add_sends();
    send_proto->set_stream_handle(send.stream_handle);
    send_proto->set_rendezvous_key(send.rendezvous_key);
    send_proto->set_concat_replicas(send.concat_replicas);
    *(send_proto->mutable_shape()) = send.shape.ToProto();
  }

  for (const auto& recv : annotations.recv_infos) {
    auto* recv_proto = proto.add_recvs();
    recv_proto->set_stream_handle(recv.stream_handle);
    recv_proto->set_rendezvous_key(recv.rendezvous_key);
    *(recv_proto->mutable_shape()) = recv.shape.ToProto();
  }

  // write the compilation options into the serialized executable
  for (const auto flag : opts) {
    auto* poplar_opt = proto.add_option_flags();
    poplar_opt->set_option(flag.first);
    poplar_opt->set_value(flag.second);
  }

  for (const auto lookup : annotations.host_embedding_lookup_infos) {
    auto* lookup_proto = proto.add_lookups();
    lookup_proto->set_stream_handle(lookup.stream_handle);
    lookup_proto->set_embedding_id(lookup.embedding_id);
    *lookup_proto->mutable_indices_shape() = lookup.indices_shape.ToProto();
    *lookup_proto->mutable_activations_shape() =
        lookup.activations_shape.ToProto();
  }

  for (const auto update : annotations.host_embedding_update_infos) {
    auto* update_proto = proto.add_updates();
    update_proto->set_stream_handle(update.stream_handle);
    update_proto->set_embedding_id(update.embedding_id);
    *update_proto->mutable_indices_shape() = update.indices_shape.ToProto();
    *update_proto->mutable_activations_shape() =
        update.activations_shape.ToProto();
  }

  for (const auto remote_parameter_info : annotations.remote_parameter_infos) {
    auto* remote_parameter = proto.add_remote_parameters();
    remote_parameter->set_parameter_number(
        remote_parameter_info.parameter_number);
  }

  for (const auto key_id_mapping : mappings) {
    auto* mapping = proto.add_key_id_mappings();
    mapping->set_handle(key_id_mapping.first);
    mapping->set_key(key_id_mapping.second.key);
    mapping->set_start_id(key_id_mapping.second.id);
  }

  for (const auto feed : checkpoint_feeds_order) {
    std::string* proto_feed = proto.add_checkpoint_feeds_order();
    *proto_feed = feed;
  }

  return WriteBinaryProto(tensorflow::Env::Default(),
                          filenames.CachedEngineFilename(), proto);
}

}  // namespace poplarplugin
}  // namespace xla
