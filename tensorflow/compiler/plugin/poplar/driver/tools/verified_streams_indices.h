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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_VERIFIED_STREAMS_INDICES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_VERIFIED_STREAMS_INDICES_H_

#include <list>
#include <map>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/input_output_aliasing_map.h"

#include <poplar/Program.hpp>

namespace poplar {

class Tensor;
class Graph;
class OptionFlags;

namespace program {

class Sequence;

}  // namespace program
}  // namespace poplar

namespace xla {

namespace poplarplugin {
class CompilerResources;

class VerifiedStreamsIndices {
 public:
  struct KeyIdPair {
    KeyIdPair(uint64 key_value, uint64 id_value)
        : key(key_value), id(id_value) {}
    uint64 key;
    uint64 id;
  };
  using KeyIdMappings = std::map<std::string, KeyIdPair>;

  // Initialize internal resources based on the compiler resources and the
  // IpuOptions and create an index tensor in the master graph for each tensor
  // type.
  Status InitializeIndexTensors(
      CompilerResources& resources, const IpuOptions::VerifiedTransfers& opts,
      const poplar::DebugNameAndId& debug_name_and_id);

  // Return an index tensor from the instruction's graph
  // corresponding to the tensor type of the passed info.
  StatusOr<poplar::Tensor> IndexTensor(
      const InputOutputAliasingMap::InputInfo& info, const HloInstruction* inst,
      poplar::program::Sequence& seq);

  // Return an index tensor from the instruction's graph
  // corresponding to the tensor type of the passed info.
  StatusOr<poplar::Tensor> IndexTensor(
      const InputOutputAliasingMap::OutputInfo& info,
      const HloInstruction* inst, poplar::program::Sequence& seq);
  // Return an index tensor from the instruction's graph
  // corresponding to the passed infeed or outfeed stream handle.
  StatusOr<poplar::Tensor> IndexTensor(const std::string& feed_stream_handle,
                                       const HloInstruction* inst,
                                       poplar::program::Sequence& seq);

  // Return some empty options if verified transfers are disabled otherwise some
  // options containing the key and id associated to the given handle.
  poplar::OptionFlags GraphOptions(const std::string& handle) const;

  // Return some empty options if verified transfers are disabled otherwise some
  // options containing the key and id associated to the given feed stream
  // handle.
  poplar::OptionFlags GraphFeedOptions(const std::string& feed_stream_handle);

  // Return some empty options if verified transfers are disabled otherwise some
  // options with verified transfers enabled.
  poplar::OptionFlags CopyOptions() const;

  // Return a map of all the executable handles and their corresponding key/id
  // pair.
  const KeyIdMappings& GetAssignedIds() const;

  Status InitializeFeedStream(const std::string& feed_name, int64 stream_idx,
                              poplar::program::Sequence& seq,
                              const HloInstruction* inst,
                              const poplar::DebugNameAndId& debug_name_and_id);

  // Program to upload a checkpoint and connect it to the verified streams.
  poplar::program::Sequence LoadCheckpointSequence() const;
  // Program to save the verified streams positions in a checkpoint tensor.
  poplar::program::Sequence SaveCheckpointSequence() const;

  // Return the order in which the feeds indices are stored in the checkpoint
  // tensor.
  const std::vector<std::string>& CheckpointFeedsOrder() const;

 private:
  // Retrieve the user's configuration for a given feed from the IpuOptions.
  StatusOr<IpuOptions::VerifiedInfo> GetFeedInfo(
      const std::string& feed_name) const;

  // Get the verification Id of a given feed stream: feed->start_id + stream_idx
  StatusOr<int64> GetStreamId(const std::string& feed_name, int64 stream_idx,
                              const HloInstruction* inst);
  // Retrieve the number of streams in a feed based on its shape.
  StatusOr<int64> GetFeedNumStreams(const HloInstruction* inst);
  poplar::Tensor GetFeedIndexTensor(const std::string& feed_name);
  // Initialize the key and id values based on the passed IpuOptions.
  Status SetKeysAndStartIds(const IpuOptions::VerifiedTransfers& opts,
                            const poplar::DebugNameAndId& debug_name_and_id);
  // Create a checkpoint tensor containing the positions of all the streams in
  // the graph. It also creates the sequences to load and save this checkpoint
  // tensor which can be accessed by LoadCheckpointSequence and
  // SaveCheckpointSequence.
  Status CreateCheckpointLoadSave(
      const IpuOptions::VerifiedTransfers& opts,
      const poplar::DebugNameAndId& debug_name_and_id);

  // Store index tensors, id, key and number of tensors for a given variable
  // type.
  class Index {
   public:
    explicit Index(const std::string& name);
    // Create an index tensor in the master graph if the object contains any
    // tensor.
    void Initialize(CompilerResources& resources, poplar::Tensor index);
    void IncrementNumTensors();
    uint64 NumTensors() const;
    // Return the main index tensor or a copy of it if the instruction is
    // not in the main graph.
    StatusOr<poplar::Tensor> IndexTensor(const HloInstruction* inst,
                                         poplar::program::Sequence& seq,
                                         CompilerResources& resources);
    // Create KeyIdPair with the current key and id, then increment the id.
    KeyIdPair NextKeyIdPair();
    // Create KeyIdPair with the current key and id.
    KeyIdPair GetKeyIdPair() const;
    void SetKeyAndStartId(uint64 key, uint64 id);
    void SetKey(uint64 key);

   private:
    std::map<poplar::Graph*, poplar::Tensor> base_indices_;
    const std::string name_;
    uint64 id_;
    uint64 key_;
    uint64 num_tensors_;
  };
  Index input_data_{"input"};
  Index output_data_{"output"};
  Index input_parameters_{"input_parameter"};
  Index output_parameters_{"output_parameter"};
  CompilerResources* resources_{nullptr};
  // map[handle] = (key, id)
  KeyIdMappings assigned_ids_;
  std::map<std::string, Index> feeds_streams_;
  std::map<std::string, const IpuOptions::VerifiedInfo> feeds_info_;
  std::map<std::string, int64> feeds_start_ids_;
  // Order in which the feeds are stored inside the checkpoint tensor.
  std::vector<std::string> feeds_;
  // Global index counter used for automatic id assignment.
  int64 next_start_id_{0};
  poplar::Tensor checkpoint_tensor_;
  poplar::program::Sequence load_checkpoint_;
  poplar::program::Sequence save_checkpoint_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_VERIFIED_STREAMS_INDICES_H_
