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

#include <algorithm>
#include <fstream>
#include <utility>

#include "ipu/poplar_executable_data.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executable.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executable_cache.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_executable_binary_file.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/tracepoint.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/core/public/version.h"

namespace xla {
namespace poplarplugin {

PoplarExecutableCore::PoplarExecutableCore(
    std::unique_ptr<poplar::Engine> engine,
    const InputOutputAliasingMap& input_output_aliasing_map,
    bool is_constant_graph,
    std::vector<std::vector<Literal>> constant_literal_output,
    bool is_remap_graph, bool is_scalar_elementwise_graph,
    bool loaded_from_cache, std::vector<uint64> remaped_output,
    StreamInfos&& stream_infos, StreamMetaInfos&& stream_meta_info,
    PoplarExecutableInfo&& info)
    : poplar_engine_(std::move(engine)),
      input_output_aliasing_map_(std::move(input_output_aliasing_map)),
      constant_literal_output_(std::move(constant_literal_output)),
      is_constant_graph_(is_constant_graph),
      is_remap_graph_(is_remap_graph),
      is_scalar_elementwise_graph_(is_scalar_elementwise_graph),
      loaded_from_cache_(loaded_from_cache),
      remaped_output_(std::move(remaped_output)),
      stream_infos_(std::move(stream_infos)),
      stream_meta_infos_(std::move(stream_meta_info)),
      info_(std::move(info)) {
  TENSORFLOW_TRACEPOINT();
  PopulateCollectiveBalanceReorderHostRerrangements();
}

PoplarExecutableCore::~PoplarExecutableCore() {
  if (poplar_engine_) {
    auto platform =
        se::MultiPlatformManager::PlatformWithName(tensorflow::PLATFORM_NAME);
    if (platform.ok()) {
      auto* p = static_cast<PoplarPlatform*>(platform.ValueOrDie());
      p->AboutToFreeEngine(poplar_engine_.get());
    }
  }
}

void PoplarExecutableCore::PopulateCollectiveBalanceReorderHostRerrangements() {
  cbr_host_rearrangements_.clear();
  for (auto& param_host_rearrangement_entry :
       GetRemoteParameterHostRearrangements()) {
    int64 id = param_host_rearrangement_entry.first;
    auto& param_host_rearrangement = param_host_rearrangement_entry.second;
    gcl::CollectiveBalancedHostRearrangement host_rearrangement;
    host_rearrangement.replicationFactor =
        param_host_rearrangement.replication_factor;
    host_rearrangement.totalElementsPerReplica =
        param_host_rearrangement.total_elements_per_replica;
    host_rearrangement.gatheredToRefSlices.reserve(
        param_host_rearrangement.gathered_to_ref_slice.size());
    for (auto& slice : param_host_rearrangement.gathered_to_ref_slice) {
      host_rearrangement.gatheredToRefSlices.emplace_back(slice.first,
                                                          slice.second);
    }
    host_rearrangement.elementMap = param_host_rearrangement.element_map;
    cbr_host_rearrangements_[id] = std::move(host_rearrangement);
  }
}

namespace {

PoplarExecutableInfo FromProto(const PoplarExecutableProto& proto,
                               poplar::OptionFlags* engine_options) {
  PoplarExecutableInfo info;

  auto& ertc = proto.embedded_runtime_config();
  info.num_IPUs = ertc.num_ipus();
  info.target_type = ertc.target_type();
  info.target_arch = ertc.target_arch();
  info.gateway_mode = ertc.gateway_mode();
  info.supports_remote_buffers = ertc.supports_remote_buffers();

  info.tf_major_version = proto.tf_major_version();
  info.tf_minor_version = proto.tf_minor_version();
  info.tf_git_version = proto.tf_git_version();

  info.replication_factor = proto.replication_factor();

  for (const auto& infeed : proto.infeeds()) {
    info.infeed_infos.emplace(infeed.config(), Shape(infeed.shape()));
  }

  for (const auto& outfeed : proto.outfeeds()) {
    info.outfeed_infos.emplace(outfeed.config(), Shape(outfeed.shape()));
  }

  for (const auto& send : proto.sends()) {
    info.send_infos.emplace_back(send.stream_handle(), send.rendezvous_key(),
                                 Shape(send.shape()));
  }

  for (const auto& recv : proto.recvs()) {
    info.recv_infos.emplace_back(recv.stream_handle(), recv.rendezvous_key(),
                                 Shape(recv.shape()));
  }

  for (const auto& lookup : proto.lookups()) {
    info.host_embedding_lookup_infos.emplace_back(
        lookup.stream_handle(), lookup.embedding_id(),
        Shape(lookup.indices_shape()), Shape(lookup.activations_shape()));
  }

  for (const auto& update : proto.updates()) {
    info.host_embedding_update_infos.emplace_back(
        update.stream_handle(), update.embedding_id(),
        Shape(update.indices_shape()), Shape(update.activations_shape()));
  }

  for (const auto& notification : proto.notifications()) {
    info.host_embedding_notify_infos.emplace_back(
        notification.stream_handle(), notification.embedding_id(),
        Shape(notification.indices_shape()),
        Shape(notification.activations_shape()));
  }

  for (const auto& remote_parameter : proto.remote_parameters()) {
    info.remote_parameter_infos.emplace(RemoteParameterInfo{
        remote_parameter.parameter_number(),
        remote_parameter.is_replica_partitioned(),
        remote_parameter.buffer_name(), remote_parameter.buffer_offset(),
        remote_parameter.num_merged(),
        remote_parameter.host_rearrangement_id()});
  }

  for (const auto& remote_parameter_host_rearrangement :
       proto.collective_balanced_host_rearrangements()) {
    int64 id = remote_parameter_host_rearrangement.id();
    RemoteParameterHostRearrangement host_rearrangement;
    host_rearrangement.replication_factor =
        remote_parameter_host_rearrangement.replication_factor();
    host_rearrangement.total_elements_per_replica =
        remote_parameter_host_rearrangement.total_elements_per_replica();
    auto& gathered_to_ref_slices =
        remote_parameter_host_rearrangement.gathered_to_ref_slices();
    host_rearrangement.gathered_to_ref_slice.reserve(
        gathered_to_ref_slices.size());
    for (const auto& slice : gathered_to_ref_slices) {
      host_rearrangement.gathered_to_ref_slice.emplace_back(slice.begin(),
                                                            slice.end());
    }
    auto& element_map = remote_parameter_host_rearrangement.element_map();
    host_rearrangement.element_map.assign(
        element_map.data(), element_map.data() + element_map.size());
    info.remote_parameter_host_rearrangements.emplace(
        id, std::move(host_rearrangement));
  }

  info.logging_cycle_count = proto.logging_cycle_count();

  for (const auto& mapping : proto.key_id_mappings()) {
    info.key_id_mappings.emplace(
        mapping.handle(),
        VerifiedStreamsIndices::KeyIdPair(mapping.key(), mapping.start_id()));
  }

  std::vector<std::string> checkpoint_feeds_order;
  for (auto feed : proto.checkpoint_feeds_order()) {
    info.checkpoint_feeds_order.push_back(feed);
  }

  // Load the additional Poplar engine options that we need to restore.
  CHECK_NOTNULL(engine_options);
  for (const auto& flag : proto.option_flags()) {
    engine_options->set(flag.option(), flag.value());
  }

  return info;
}

PoplarExecutableProto ToProto(const PoplarExecutableInfo& info,
                              const poplar::OptionFlags& poplar_options = {}) {
  PoplarExecutableProto proto;

  auto ertc = proto.mutable_embedded_runtime_config();
  ertc->set_num_ipus(info.num_IPUs);
  ertc->set_target_type(info.target_type);
  ertc->set_target_arch(info.target_arch);
  ertc->set_gateway_mode(info.gateway_mode);
  ertc->set_supports_remote_buffers(info.supports_remote_buffers);

  proto.set_tf_major_version(info.tf_major_version);
  proto.set_tf_minor_version(info.tf_minor_version);
  proto.set_tf_git_version(info.tf_git_version);

  proto.set_replication_factor(info.replication_factor);

  for (const auto& infeed : info.infeed_infos) {
    auto* feed = proto.add_infeeds();
    *(feed->mutable_config()) = infeed.config;
    *(feed->mutable_shape()) = infeed.shape.ToProto();
  }

  for (const auto& outfeed : info.outfeed_infos) {
    auto* feed = proto.add_outfeeds();
    *(feed->mutable_config()) = outfeed.config;
    *(feed->mutable_shape()) = outfeed.shape.ToProto();
  }

  for (const auto& send : info.send_infos) {
    auto* send_proto = proto.add_sends();
    send_proto->set_stream_handle(send.stream_handle);
    send_proto->set_rendezvous_key(send.rendezvous_key);
    *(send_proto->mutable_shape()) = send.shape.ToProto();
  }

  for (const auto& recv : info.recv_infos) {
    auto* recv_proto = proto.add_recvs();
    recv_proto->set_stream_handle(recv.stream_handle);
    recv_proto->set_rendezvous_key(recv.rendezvous_key);
    *(recv_proto->mutable_shape()) = recv.shape.ToProto();
  }

  for (const auto& lookup : info.host_embedding_lookup_infos) {
    auto* lookup_proto = proto.add_lookups();
    lookup_proto->set_stream_handle(lookup.stream_handle);
    lookup_proto->set_embedding_id(lookup.embedding_id);
    *lookup_proto->mutable_indices_shape() = lookup.indices_shape.ToProto();
    *lookup_proto->mutable_activations_shape() =
        lookup.activations_shape.ToProto();
  }

  for (const auto& update : info.host_embedding_update_infos) {
    auto* update_proto = proto.add_updates();
    update_proto->set_stream_handle(update.stream_handle);
    update_proto->set_embedding_id(update.embedding_id);
    *update_proto->mutable_indices_shape() = update.indices_shape.ToProto();
    *update_proto->mutable_activations_shape() =
        update.activations_shape.ToProto();
  }

  for (const auto& notification : info.host_embedding_notify_infos) {
    auto* update_proto = proto.add_notifications();
    update_proto->set_stream_handle(notification.stream_handle);
    update_proto->set_embedding_id(notification.embedding_id);
  }

  for (const auto& remote_parameter_info : info.remote_parameter_infos) {
    auto* remote_parameter = proto.add_remote_parameters();
    remote_parameter->set_parameter_number(
        remote_parameter_info.parameter_number);
    remote_parameter->set_is_replica_partitioned(
        remote_parameter_info.is_replica_partitioned);
    remote_parameter->set_buffer_name(remote_parameter_info.buffer_name);
    remote_parameter->set_buffer_offset(remote_parameter_info.buffer_offset);
    remote_parameter->set_num_merged(remote_parameter_info.num_merged);
    remote_parameter->set_host_rearrangement_id(
        remote_parameter_info.host_rearrangement_id);
  }

  for (const auto& remote_parameter_host_rearrangement_entry :
       info.remote_parameter_host_rearrangements) {
    int64 id = remote_parameter_host_rearrangement_entry.first;
    auto& src_host_rearrangement =
        remote_parameter_host_rearrangement_entry.second;
    auto* host_rearrangement =
        proto.add_collective_balanced_host_rearrangements();
    host_rearrangement->set_id(id);
    host_rearrangement->set_replication_factor(
        src_host_rearrangement.replication_factor);
    host_rearrangement->set_total_elements_per_replica(
        src_host_rearrangement.total_elements_per_replica);
    for (auto& slice : src_host_rearrangement.gathered_to_ref_slice) {
      auto* interval = host_rearrangement->add_gathered_to_ref_slices();
      interval->set_begin(slice.first);
      interval->set_end(slice.second);
    }
    auto& element_map = src_host_rearrangement.element_map;
    *host_rearrangement->mutable_element_map() = {element_map.begin(),
                                                  element_map.end()};
  }

  for (const auto& key_id_mapping : info.key_id_mappings) {
    auto* mapping = proto.add_key_id_mappings();
    mapping->set_handle(key_id_mapping.first);
    mapping->set_key(key_id_mapping.second.key);
    mapping->set_start_id(key_id_mapping.second.id);
  }

  for (const auto& feed : info.checkpoint_feeds_order) {
    std::string* proto_feed = proto.add_checkpoint_feeds_order();
    *proto_feed = feed;
  }

  proto.set_logging_cycle_count(info.logging_cycle_count);

  // Items that don't need deserialising.
  for (const auto& input_info : info.entry_input_infos) {
    auto input = ertc->mutable_signature()->add_inputs();
    input->set_name(input_info.name);
    input->set_handle(input_info.handle);
    input->set_argument(input_info.argument);
    input->set_tuple_index(input_info.tuple_index);
    (*input->mutable_shape()) = input_info.shape.ToProto();
  }

  for (const auto& streamed_input_info : info.feed_input_infos) {
    auto input = ertc->mutable_signature()->add_streamed_inputs();
    input->set_name(streamed_input_info.name);
    input->set_handle(streamed_input_info.handle);
    input->set_argument(streamed_input_info.argument);
    input->set_tuple_index(streamed_input_info.tuple_index);
    (*input->mutable_shape()) = streamed_input_info.shape.ToProto();
  }

  for (const auto& output_info : info.entry_output_infos) {
    auto output = ertc->mutable_signature()->add_outputs();
    output->set_name(output_info.name);
    output->set_handle(output_info.handle);
    output->set_tuple_index(output_info.tuple_index);
    (*output->mutable_shape()) = output_info.shape.ToProto();
  }

  for (const auto& streamed_output_info : info.feed_output_infos) {
    auto output = ertc->mutable_signature()->add_streamed_outputs();
    output->set_name(streamed_output_info.name);
    output->set_handle(streamed_output_info.handle);
    output->set_tuple_index(streamed_output_info.tuple_index);
    (*output->mutable_shape()) = streamed_output_info.shape.ToProto();
  }

  // Note that Poplar will serialize its own state. Here we can serialize
  // additional engine options (typically run-time options) that Poplar
  // does not consider a part of its own executable state.
  for (const auto& flag : poplar_options) {
    auto* poplar_opt = proto.add_option_flags();
    poplar_opt->set_option(flag.first);
    poplar_opt->set_value(flag.second);
  }

  return proto;
}

}  // namespace

/*static*/ StatusOr<std::unique_ptr<PoplarExecutableCore>>
PoplarExecutableCore::Deserialize(
    const HloModule* module,
    absl::optional<RuntimeReplicaOptions> runtime_replica_options,
    const ModuleFilenames& filenames) {
  TENSORFLOW_TRACEPOINT();
  PoplarExecutableProto proto;
  VLOG(1) << "Trying to deserialize cached file: "
          << filenames.CachedExecutableFilename();

  TF_ASSIGN_OR_RETURN(poplar::Executable executable,
                      PoplarExecutableBinaryFile::Read(
                          filenames.CachedExecutableFilename(), &proto));

  poplar::OptionFlags engine_options;
  PoplarExecutableInfo info = FromProto(proto, &engine_options);

  // Also set run-time replica engine options for multi-replica distribution, as
  // these are not serialized to allow for using the same serialized executable
  // across processes.
  if (runtime_replica_options.has_value()) {
    SetRuntimeReplicaOptions(
        &engine_options, runtime_replica_options->process_index,
        runtime_replica_options->process_count, info.replication_factor);
  }

  // Load the Poplar executable.
  std::unique_ptr<poplar::Engine> engine =
      absl::make_unique<poplar::Engine>(std::move(executable), engine_options);

  InputOutputAliasingMap iomap(module);

  std::unique_ptr<PoplarExecutableCore> executable_core =
      absl::make_unique<PoplarExecutableCore>(
          std::move(engine), std::move(iomap),
          /*is_constant_graph=*/false, std::vector<std::vector<Literal>>{},
          /*is_remap_graph=*/false,
          /*is_scalar_elementwise_graph=*/false,
          /*loaded_from_cache=*/true, std::vector<uint64>{}, StreamInfos{},
          StreamMetaInfos{}, std::move(info));

  return executable_core;
}

/*static*/ Status PoplarExecutableCore::Serialize(
    const ModuleFilenames& filenames, const poplar::Executable& executable,
    const poplar::OptionFlags& opts, const PoplarExecutableInfo& info) {
  TENSORFLOW_TRACEPOINT();

  const PoplarExecutableProto proto = ToProto(info, opts);

  return PoplarExecutableBinaryFile::Write(
      filenames.CachedExecutableFilename(), proto,
      [&executable](std::ostream& out) { executable.serialize(out); });
}

namespace {
Status ExportInternal(
    const ModuleFilenames& filenames, const poplar::Executable& executable,
    const CanonicalInfeedInfos& infeeds, const CanonicalOutfeedInfos& outfeeds,
    const SendRecvInfos& sends, const SendRecvInfos& recvs,
    const HostEmbeddingInfos& lookups, const HostEmbeddingInfos& updates,
    const InputOutputAliasingMap& io_map, uint32 replication_count,
    const poplar::OptionFlags& device_opts,
    const poplar::OptionFlags& engine_opts, const poplar::Target& target,
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
    TF_ASSIGN_OR_RETURN(
        ipu::Metadata metadata,
        CreateExecutableMetadata(io_map, infeeds, outfeeds, replication_count,
                                 device_opts, engine_opts, target, indices,
                                 checkpoint_feeds_order));
    std::string json_metadata = metadata.ToJson();
    VLOG(1) << "Module JSON Metadata: " << json_metadata;
    // For security reasons don't store the verification information inside the
    // binary.
    metadata.verification_info.clear();
    ipu::BinaryWriter writer(filenames.SerializedExecutableFilename());
    writer.WriteMetadata(filenames.Name(), metadata);
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
    return PoplarExceptionToTensorflowStatus("[Serialize]", e);
  }
  return Status::OK();
}
}  // namespace

Status PoplarExecutableCore::Serialize(const std::string& filepath) const {
  if (is_scalar_elementwise_graph_) {
    return FailedPrecondition("Cannot serialize a scalar elementwise graph");
  }
  if (is_constant_graph_) {
    return FailedPrecondition("Cannot serialize a constant graph");
  }
  if (is_remap_graph_) {
    return FailedPrecondition(
        "Cannot serialize a graph in which all outputs are inputs");
  }

  if (!poplar_engine_) {
    return InternalError("Missing Poplar engine");
  }

  return PoplarExecutableBinaryFile::Write(
      filepath, ToProto(info_),
      [this](std::ostream& out) { poplar_engine_->serializeExecutable(out); });
}

/*static*/ Status PoplarExecutableCore::Export(
    const ModuleFilenames& filenames, const poplar::Executable& executable,
    const CompilerResources& resources, uint32 replication_count,
    const poplar::OptionFlags& device_opts,
    const poplar::OptionFlags& engine_opts, const poplar::Target& target) {
  return ExportInternal(
      filenames, executable, resources.annotations.infeed_infos,
      resources.annotations.outfeed_infos, resources.annotations.send_infos,
      resources.annotations.recv_infos,
      resources.annotations.host_embedding_lookup_infos,
      resources.annotations.host_embedding_update_infos,
      resources.annotations.input_output_aliasing_map, replication_count,
      device_opts, engine_opts, target,
      resources.streams_indices.GetAssignedIds(),
      resources.streams_indices.CheckpointFeedsOrder());
}

/*static*/ Status PoplarExecutableCore::Export(
    const ModuleFilenames& filenames, const poplar::Executable& executable,
    const PoplarExecutableCore& executable_core,
    const poplar::OptionFlags& device_opts,
    const poplar::OptionFlags& engine_opts, const poplar::Target& target) {
  return ExportInternal(
      filenames, executable, executable_core.GetInfeedInfos(),
      executable_core.GetOutfeedInfos(), executable_core.GetSendInfos(),
      executable_core.GetRecvInfos(),
      executable_core.GetHostEmbeddingLookupInfos(),
      executable_core.GetHostEmbeddingUpdateInfos(),
      executable_core.GetInputOutputAliasingMap(),
      executable_core.GetReplicationFactor(), device_opts, engine_opts, target,
      executable_core.KeyIdMappings(), executable_core.CheckpointFeedsOrder());
}

PoplarExecutable::PoplarExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloProfilePrinterData> profile_printer,
    std::unique_ptr<HloProfileIndexMap> profile_index_map,
    const TranslatedInfeedInfos& infeed_infos,
    const TranslatedOutfeedInfos& outfeed_infos,
    std::shared_ptr<PoplarExecutableCore> executable_core)
    : Executable(std::move(hlo_module), std::move(profile_printer),
                 std::move(profile_index_map)),
      infeed_infos_(infeed_infos),
      outfeed_infos_(outfeed_infos),
      executable_core_(std::move(executable_core)) {
  TENSORFLOW_TRACEPOINT();
}

Status PoplarExecutable::ExecuteComputeFunction(
    const ExecutableRunOptions* run_options,
    se::DeviceMemoryBase* result_buffer,
    HloExecutionProfile* hlo_execution_profile,
    const std::vector<se::DeviceMemoryBase>& argument_buffers,
    const std::vector<Shape>& argument_shapes,
    const PoplarExecutor::ArgsHandleMap& args_map, uint64 start_time_us) {
  TENSORFLOW_TRACEPOINT();
  VLOG(2) << "Begin asynchronous engine execution " << module().name();
  se::Stream* stream = run_options->stream();
  se::StreamExecutor* executor = stream->parent();
  PoplarExecutor* poplar_executor =
      static_cast<PoplarExecutor*>(executor->implementation());
  se::DeviceMemoryAllocator* memory_allocator = run_options->allocator();

  poplar_executor->ExecuteEngine(result_buffer, executor, *this, args_map,
                                 memory_allocator, argument_buffers);

  execution_count_++;
  if (poplar_executor->ReportEventNthExecution() > 0 &&
      execution_count_ >= poplar_executor->ReportEventNthExecution()) {
    execution_count_ = 0;
  }

  uint64 end_time_us = tensorflow::Env::Default()->NowMicros();

  if (run_options->execution_profile()) {
    auto profile = run_options->execution_profile();
    const double nanoseconds = (end_time_us - start_time_us) * 1000.0;
    profile->set_compute_time_ns(std::max(nanoseconds, 1.0));
    profile->set_compute_cycle_count(1);
  }

  // Decrement the reference counter for all buffers once the execution is
  // completed so that they can be deallocated if required (this applies even
  // if the execution didn't complete successfully).
  TF_RETURN_IF_ERROR(PoplarExecutor::DecrementBufferReferenceCount(
      *result_buffer, result_shape()));
  for (int64 i = 0; i != argument_buffers.size(); ++i) {
    TF_RETURN_IF_ERROR(PoplarExecutor::DecrementBufferReferenceCount(
        argument_buffers[i], argument_shapes[i]));
  }

  VLOG(2) << "End asynchronous engine execution " << module().name();

  return Status::OK();
}

StatusOr<ScopedShapedBuffer> PoplarExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  TENSORFLOW_TRACEPOINT();
  se::Stream* stream = run_options->stream();

  std::vector<se::DeviceMemoryBase> argument_buffers;
  std::vector<Shape> argument_shapes;

  for (size_t i = 0; i < arguments.size(); ++i) {
    const se::DeviceMemoryBase& argument_buffer =
        arguments[i]->buffer(/*index=*/{});
    const Shape& argument_shape = arguments[i]->on_device_shape();
    argument_buffers.push_back(argument_buffer);
    argument_shapes.push_back(argument_shape);
    // Make sure inputs are not deallocated during execution by increasing the
    // reference counter.
    TF_RETURN_IF_ERROR(PoplarExecutor::IncrementBufferReferenceCount(
        argument_buffer, argument_shape));
  }

  VLOG(1) << "Execute " << module().name();
  if (VLOG_IS_ON(2)) {
    for (const auto& a : argument_buffers) {
      VLOG(2) << "-- argument " << a.opaque();
    }
  }

  uint64 start_time_us = tensorflow::Env::Default()->NowMicros();

  se::StreamExecutor* executor = stream->parent();
  PoplarExecutor* poplar_executor =
      static_cast<PoplarExecutor*>(executor->implementation());

  // There is no obvious way to return a failed Status asynchronously when
  // executing an engine, so make sure the previous execution was ok.
  TF_RETURN_IF_ERROR(poplar_executor->GetAndResetExecutorStatus());

  switch (poplar_executor->ConnectionType()) {
    case IpuDeviceConnectionType::NEVER: {
      if (Engine()) {
        return InvalidArgument(
            "Trying to run an executable on a device that was configured for "
            "compilation only.");
      }
      break;
    }
    case IpuDeviceConnectionType::PRE_COMPILE: {
      VLOG(2) << "No device attached for pre compilation of Poplar programs, "
                 "output buffer will be populated with zeros.";
      break;
    }
    default: {
      if (!poplar_executor->PoplarDeviceIsAttached() && Engine()) {
        TF_RETURN_IF_ERROR(poplar_executor->AttachToPoplarDevice());
      }
      break;
    }
  }

  if (Engine() && poplar_executor->UseVerifiedTransfers()) {
    return InvalidArgument(
        "Executables using verified transfers can't be run "
        "in TensorFlow");
  }

  se::DeviceMemoryAllocator* memory_allocator = run_options->allocator();

  // Create the argument map which stores the information about all the inputs
  // and their locations.
  auto args_map = PoplarExecutor::CreateArgsHandleMap(
      argument_buffers, memory_allocator, *this,
      poplar_executor->device_ordinal());

  // Create the output buffer.
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase result,
      PoplarExecutor::AllocateOutputBuffer(*this, memory_allocator, args_map,
                                           poplar_executor->device_ordinal(),
                                           poplar_executor->ConnectionType()));

  // Make sure the result is not deallocated until the execution has finished by
  // increasing the reference counters.
  TF_RETURN_IF_ERROR(
      PoplarExecutor::IncrementBufferReferenceCount(result, result_shape()));

  // Copy DeviceMemoryBase values which contain the array(s) of the result into
  // the respective location in ShapedBuffer which is returned to the caller.
  ScopedShapedBuffer result_buffer(result_shape(), result_shape(),
                                   memory_allocator,
                                   stream->parent()->device_ordinal());

  TF_RETURN_IF_ERROR(result_buffer.buffers().ForEachMutableElementWithStatus(
      [&result](const ShapeIndex& index, se::DeviceMemoryBase* device_memory) {
        TF_ASSIGN_OR_RETURN(
            se::DeviceMemoryBase buffer,
            PoplarExecutor::GetBufferByShapeIndex(result, index));
        CHECK(!buffer.is_null() || buffer.size() == 0);
        if (VLOG_IS_ON(2)) {
          VLOG(2) << "-- return " << buffer.opaque();
        }
        *device_memory = buffer;
        return Status::OK();
      }));

  // Need to make sure to capture all the required resources for the
  // execution. We use a struct instead of a lambda to make this explicit.
  struct AsyncExecuteTask {
    PoplarExecutable* executable;
    ServiceExecutableRunOptions run_options;
    se::DeviceMemoryBase output_buffer;
    HloExecutionProfile* hlo_execution_profile;
    std::vector<se::DeviceMemoryBase> argument_buffers;
    std::vector<Shape> argument_shapes;
    PoplarExecutor::ArgsHandleMap args_map;
    uint64 start_time_us;

    void operator()() {
      TF_CHECK_OK(executable->ExecuteComputeFunction(
          &run_options.run_options(), &output_buffer, hlo_execution_profile,
          argument_buffers, argument_shapes, args_map, start_time_us));
    }
  };

  PoplarExecutor::AsPoplarStream(stream)->EnqueueTask(AsyncExecuteTask{
      this, *run_options, result, hlo_execution_profile, argument_buffers,
      argument_shapes, args_map, start_time_us});

  return std::move(result_buffer);
}

/*static*/ int64 PoplarExecutable::ShapeSizeBytes(const Shape& shape) {
  if (shape.IsOpaque()) {
    return sizeof(void*);
  }
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}

}  // namespace poplarplugin
}  // namespace xla
