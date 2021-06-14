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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPLAR_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPLAR_EXECUTABLE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"

#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/feed_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/input_output_aliasing_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/verified_streams_indices.h"

#include <poplar/Engine.hpp>

namespace xla {
namespace poplarplugin {
struct CompilerAnnotations;
struct CompilerResources;

class PoplarExecutableCore {
 public:
  PoplarExecutableCore(
      std::unique_ptr<poplar::Engine> engine,
      const InputOutputAliasingMap& input_output_aliasing_map,
      bool is_constant_graph,
      std::vector<std::vector<Literal>> constant_literal_output,
      bool is_remap_graph, bool is_scalar_elementwise_graph,
      bool loaded_from_cache, std::vector<uint64> remaped_output,
      uint32 replication_factor, const CanonicalInfeedInfos& infeed_infos,
      const CanonicalOutfeedInfos& outfeed_infos, StreamInfos&& stream_infos,
      StreamMetaInfos&& stream_meta_info, SendRecvInfos&& send_infos,
      SendRecvInfos&& recv_infos,
      HostEmbeddingInfos&& host_embedding_lookup_infos,
      HostEmbeddingInfos&& host_embedding_update_infos,
      HostEmbeddingInfos&& host_embedding_notify_infos,
      RemoteParameterInfos&& remote_parameter_infos, bool logging_cycle_count,
      const VerifiedStreamsIndices::KeyIdMappings& key_id_mappings,
      const std::vector<string>& checkpoint_feeds_order);

  ~PoplarExecutableCore();

  const InputOutputAliasingMap& GetInputOutputAliasingMap() const {
    return input_output_aliasing_map_;
  }

  poplar::Engine* Engine() const { return poplar_engine_.get(); }

  const std::vector<std::vector<Literal>>& ConstantLiteralOutput() const {
    return constant_literal_output_;
  }

  const CanonicalInfeedInfos& GetInfeedInfos() const { return infeed_infos_; }

  const CanonicalOutfeedInfos& GetOutfeedInfos() const {
    return outfeed_infos_;
  }

  const HostEmbeddingInfos& GetHostEmbeddingLookupInfos() const {
    return host_embedding_lookup_infos_;
  }

  const HostEmbeddingInfos& GetHostEmbeddingUpdateInfos() const {
    return host_embedding_update_infos_;
  }

  const HostEmbeddingInfos& GetHostEmbeddingNotifyInfos() const {
    return host_embedding_notify_infos_;
  }

  const RemoteParameterInfos& GetRemoteParameterInfos() const {
    return remote_parameter_infos_;
  }

  const StreamInfos& GetStreamInfos() const { return stream_infos_; }

  const StreamMetaInfos& GetStreamMetaInfos() const {
    return stream_meta_infos_;
  }

  const SendRecvInfos& GetSendInfos() const { return send_infos_; }

  const SendRecvInfos& GetRecvInfos() const { return recv_infos_; }

  const uint32 GetReplicationFactor() const { return replication_factor_; }

  bool IsConstantGraph() const { return is_constant_graph_; }

  const std::vector<uint64>& RemapMap() const { return remaped_output_; }

  bool IsRemapGraph() const { return is_remap_graph_; }

  bool IsLoadedFromCache() const { return loaded_from_cache_; }

  bool IsScalarElementwiseGraph() const { return is_scalar_elementwise_graph_; }

  bool LoggingCycleCount() const { return logging_cycle_count_; }

  const VerifiedStreamsIndices::KeyIdMappings& KeyIdMappings() const {
    return key_id_mappings_;
  }

  const std::vector<string>& CheckpointFeedsOrder() const {
    return checkpoint_feeds_order_;
  }

  struct RuntimeReplicaOptions {
    int64 process_index;
    int64 process_count;
  };

  static StatusOr<std::unique_ptr<PoplarExecutableCore>> Deserialize(
      const HloModule* module,
      absl::optional<RuntimeReplicaOptions> runtime_replica_options,
      const ModuleFilenames& filenames);

  static Status Serialize(const ModuleFilenames& filenames,
                          const poplar::Executable& executable,
                          const CompilerAnnotations& annotations,
                          uint32 replication_count,
                          const poplar::OptionFlags& opts,
                          bool logging_cycle_count,
                          const VerifiedStreamsIndices::KeyIdMappings& mappings,
                          const std::vector<string>& checkpoint_feeds_order);

  static Status Export(const ModuleFilenames& filenames,
                       const poplar::Executable& executable,
                       const CompilerResources& resources,
                       uint32 replication_count,
                       const poplar::OptionFlags& device_opts,
                       const poplar::OptionFlags& engine_opts,
                       const poplar::Target& target);

  static Status Export(const ModuleFilenames& filenames,
                       const poplar::Executable& executable,
                       const PoplarExecutableCore& poplar_executable,
                       const poplar::OptionFlags& device_opts,
                       const poplar::OptionFlags& engine_opts,
                       const poplar::Target& target);

 private:
  // If you add fields which are specific to a compiled engine, then you will
  // need to add them to the poplar_executable.proto, and the serialization
  // code.
  std::unique_ptr<poplar::Engine> poplar_engine_;
  InputOutputAliasingMap input_output_aliasing_map_;
  std::vector<std::vector<Literal>> constant_literal_output_;
  const bool is_constant_graph_;
  std::vector<uint64> remaped_output_;
  const bool is_remap_graph_;
  const bool is_scalar_elementwise_graph_;
  const bool loaded_from_cache_;
  uint32 replication_factor_;
  CanonicalInfeedInfos infeed_infos_;
  CanonicalOutfeedInfos outfeed_infos_;
  StreamInfos stream_infos_;
  StreamMetaInfos stream_meta_infos_;
  SendRecvInfos send_infos_;
  SendRecvInfos recv_infos_;
  HostEmbeddingInfos host_embedding_lookup_infos_;
  HostEmbeddingInfos host_embedding_update_infos_;
  HostEmbeddingInfos host_embedding_notify_infos_;
  RemoteParameterInfos remote_parameter_infos_;
  const bool logging_cycle_count_;
  VerifiedStreamsIndices::KeyIdMappings key_id_mappings_;
  const std::vector<string> checkpoint_feeds_order_;

  TF_DISALLOW_COPY_AND_ASSIGN(PoplarExecutableCore);
};

class PoplarExecutable : public Executable {
 public:
  PoplarExecutable(std::unique_ptr<HloModule> hlo_module,
                   std::unique_ptr<HloProfilePrinterData> hlo_profile_printer,
                   std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map,
                   const TranslatedInfeedInfos& infeed_infos,
                   const TranslatedOutfeedInfos& outfeed_infos,
                   std::shared_ptr<PoplarExecutableCore> executable_core);

  StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  static int64 ShapeSizeBytes(const Shape& shape);

  int64 ExecutionCount() const { return execution_count_; }

  void OnEngineLoaded() { execution_count_ = 0; }

  const InputOutputAliasingMap& GetInputOutputAliasingMap() const {
    return executable_core_->GetInputOutputAliasingMap();
  }

  poplar::Engine* Engine() const { return executable_core_->Engine(); }

  void SetLiteralValue(std::vector<std::vector<Literal>>&& literals) {
    CHECK(IsScalarElementwiseGraph());
    literal_output_ = std::move(literals);
  }

  const std::vector<std::vector<Literal>>& LiteralValue() const {
    if (IsConstantGraph()) {
      return executable_core_->ConstantLiteralOutput();
    }
    return literal_output_;
  }

  const TranslatedInfeedInfos& GetInfeedInfos() const { return infeed_infos_; }

  const TranslatedOutfeedInfos& GetOutfeedInfos() const {
    return outfeed_infos_;
  }

  const HostEmbeddingInfos& GetHostEmbeddingLookupInfos() const {
    return executable_core_->GetHostEmbeddingLookupInfos();
  }

  const HostEmbeddingInfos& GetHostEmbeddingUpdateInfos() const {
    return executable_core_->GetHostEmbeddingUpdateInfos();
  }

  const HostEmbeddingInfos& GetHostEmbeddingNotifyInfos() const {
    return executable_core_->GetHostEmbeddingNotifyInfos();
  }

  const RemoteParameterInfos& GetRemoteParameterInfos() const {
    return executable_core_->GetRemoteParameterInfos();
  }

  const StreamInfos& GetStreamInfos() const {
    return executable_core_->GetStreamInfos();
  }

  const StreamMetaInfos& GetStreamMetaInfos() const {
    return executable_core_->GetStreamMetaInfos();
  }

  const SendRecvInfos& GetSendInfos() const {
    return executable_core_->GetSendInfos();
  }

  const SendRecvInfos& GetRecvInfos() const {
    return executable_core_->GetRecvInfos();
  }

  const uint32 GetReplicationFactor() const {
    return executable_core_->GetReplicationFactor();
  }

  bool IsConstantGraph() const { return executable_core_->IsConstantGraph(); }

  const std::vector<uint64>& RemapMap() const {
    return executable_core_->RemapMap();
  }

  bool IsRemapGraph() const { return executable_core_->IsRemapGraph(); }

  bool IsLoadedFromCache() const {
    return executable_core_->IsLoadedFromCache();
  }

  bool IsScalarElementwiseGraph() const {
    return executable_core_->IsScalarElementwiseGraph();
  }

  bool LoggingCycleCount() const {
    return executable_core_->LoggingCycleCount();
  }

  const VerifiedStreamsIndices::KeyIdMappings& KeyIdMappings() const {
    return executable_core_->KeyIdMappings();
  }

  const std::vector<string>& CheckpointFeedsOrder() const {
    return executable_core_->CheckpointFeedsOrder();
  }

 private:
  Status ExecuteComputeFunction(
      const ExecutableRunOptions* run_options,
      se::DeviceMemoryBase* result_buffer,
      HloExecutionProfile* hlo_execution_profile,
      const std::vector<se::DeviceMemoryBase>& argument_buffers,
      const std::vector<Shape>& argument_shapes,
      const PoplarExecutor::ArgsHandleMap& args_map, uint64 start_time_us);

  friend class GraphCompileIoMapTest;

  const TranslatedInfeedInfos infeed_infos_;
  const TranslatedOutfeedInfos outfeed_infos_;
  std::shared_ptr<PoplarExecutableCore> executable_core_;
  std::vector<std::vector<Literal>> literal_output_;
  int64 execution_count_;

  TF_DISALLOW_COPY_AND_ASSIGN(PoplarExecutable);
};

}  // namespace poplarplugin
}  // namespace xla

#endif
