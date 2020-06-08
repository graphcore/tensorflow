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
#include <vector>

#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"

#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/input_output_aliasing_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/verified_streams_indices.h"

#include <poplar/Engine.hpp>

namespace xla {
namespace poplarplugin {
struct CompilerAnnotations;

class CompilerResources;

// A Poplar executable is a wrapper around an Engine, with
// the execution Sequence program, input tensors and output
// tensor recorded.
class PoplarExecutable : public Executable {
 public:
  PoplarExecutable(std::unique_ptr<HloModule> hlo_module,
                   std::unique_ptr<HloProfilePrinterData> hlo_profile_printer,
                   std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map,
                   std::unique_ptr<poplar::Engine> engine,
                   const InputOutputAliasingMap& input_output_aliasing_map,
                   const bool is_constant_graph,
                   std::vector<std::vector<Literal>> literal_output,
                   const bool is_remap_graph,
                   const bool is_scalar_elementwise_graph,
                   std::vector<uint64> remaped_output,
                   uint32 replication_factor_, const InfeedInfos& infeed_infos,
                   const OutfeedInfos& outfeed_infos, StreamInfos&& stream_info,
                   StreamMetaInfos&& stream_meta_info,
                   SendRecvInfos&& send_infos, SendRecvInfos&& recv_infos,
                   HostEmbeddingInfos&& host_embedding_lookup_infos,
                   HostEmbeddingInfos&& host_embedding_update_infos,
                   RemoteParameterInfos&& remote_parameter_infos,
                   const VerifiedStreamsIndices::KeyIdMappings& key_id_mappings,
                   const std::vector<string>& checkpoint_feeds_order);

  ~PoplarExecutable() override;

  StatusOr<ScopedShapedBuffer> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<const ShapedBuffer* const> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  static int64 ShapeSizeBytes(const Shape& shape);

  int64 ExecutionCount() const { return execution_count_; }

  void OnEngineLoaded() { execution_count_ = 0; }

  const InputOutputAliasingMap& GetInputOutputAliasingMap() const {
    return input_output_aliasing_map_;
  }

  poplar::Engine* Engine() const { return poplar_engine_.get(); }

  const std::vector<std::vector<Literal>>& LiteralValue() const {
    return literal_output_;
  }

  const InfeedInfos& GetInfeedInfos() const { return infeed_infos_; }

  const OutfeedInfos& GetOutfeedInfos() const { return outfeed_infos_; }

  const HostEmbeddingInfos& GetHostEmbeddingLookupInfos() const {
    return host_embedding_lookup_infos_;
  }

  const HostEmbeddingInfos& GetHostEmbeddingUpdateInfos() const {
    return host_embedding_update_infos_;
  }

  const RemoteParameterInfos& GeRemoteParameterInfos() const {
    return remote_parameter_infos_;
  }

  const StreamInfos& GetStreamInfos() const { return stream_infos_; }

  const StreamMetaInfos& GetStreamMetaInfos() const {
    return stream_meta_infos_;
  }

  const SendRecvInfos& GetSendInfos() const { return send_infos_; }

  const SendRecvInfos& GetRecvInfos() const { return recv_infos_; }

  const uint32 GetReplicationFactor() const { return replication_factor_; }

  const bool IsConstantGraph() const { return is_constant_graph_; }

  const std::vector<uint64>& RemapMap() const { return remaped_output_; }

  const bool IsRemapGraph() const { return is_remap_graph_; }

  const bool IsLoadedFromCache() const { return loaded_from_cache_; }

  const bool IsScalarElementwiseGraph() const {
    return is_scalar_elementwise_graph_;
  }

  const VerifiedStreamsIndices::KeyIdMappings& KeyIdMappings() const {
    return key_id_mappings_;
  }

  const std::vector<string>& CheckpointFeedsOrder() const {
    return checkpoint_feeds_order_;
  }

  static StatusOr<PoplarExecutable*> Deserialize(
      std::unique_ptr<HloModule> hlo_module,
      std::unique_ptr<HloProfilePrinterData> hlo_profile_printer,
      std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map,
      const ModuleFilenames& filenames);

  static Status Serialize(const ModuleFilenames& filenames,
                          const poplar::Executable& executable,
                          const CompilerAnnotations& annotations,
                          uint32 replication_count,
                          const poplar::OptionFlags& opts,
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
                       const PoplarExecutable& poplar_executable,
                       const poplar::OptionFlags& device_opts,
                       const poplar::OptionFlags& engine_opts,
                       const poplar::Target& target);

 private:
  friend class GraphCompileIoMapTest;

  // If you add fields which are specific to a compiled engine, then you will
  // need to add them to the poplar_executable.proto, and the serialization code
  // in PoplarExecutable.
  std::unique_ptr<poplar::Engine> poplar_engine_;
  InputOutputAliasingMap input_output_aliasing_map_;
  std::vector<std::vector<Literal>> literal_output_;
  const bool is_constant_graph_;
  std::vector<uint64> remaped_output_;
  const bool is_remap_graph_;
  int64 execution_count_;
  uint32 replication_factor_;
  InfeedInfos infeed_infos_;
  OutfeedInfos outfeed_infos_;
  StreamInfos stream_infos_;
  StreamMetaInfos stream_meta_infos_;
  SendRecvInfos send_infos_;
  SendRecvInfos recv_infos_;
  HostEmbeddingInfos host_embedding_lookup_infos_;
  HostEmbeddingInfos host_embedding_update_infos_;
  RemoteParameterInfos remote_parameter_infos_;
  bool loaded_from_cache_;
  const bool is_scalar_elementwise_graph_;
  VerifiedStreamsIndices::KeyIdMappings key_id_mappings_;
  const std::vector<string> checkpoint_feeds_order_;

  TF_DISALLOW_COPY_AND_ASSIGN(PoplarExecutable);
};

}  // namespace poplarplugin
}  // namespace xla

#endif
