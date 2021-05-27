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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_COMPILER_ANNOTATIONS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_COMPILER_ANNOTATIONS_H_

#include <set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_feed_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/host_embedding.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/input_output_aliasing_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

namespace xla {
using FlattenedInstMap = absl::flat_hash_map<HloInstruction*, HloInstruction*>;

class HloInfeedInstruction;

namespace poplarplugin {

struct FeedInfo {
  FeedInfo(const std::string& stream_prefix, const PoplarFeedConfig& config,
           const Shape& shape)
      : stream_prefix(stream_prefix), config(config), shape(shape) {}
  FeedInfo() = delete;

  bool operator<(const FeedInfo& rhs) const {
    return stream_prefix < rhs.stream_prefix;
  }

  std::string stream_prefix;
  PoplarFeedConfig config;
  Shape shape;
};

struct SendRecvInfo {
  SendRecvInfo(const std::string& stream_handle,
               const std::string& rendezvous_key, const Shape& shape)
      : stream_handle(stream_handle),
        rendezvous_key(rendezvous_key),
        shape(shape) {}
  SendRecvInfo() = delete;

  std::string stream_handle;
  std::string rendezvous_key;
  Shape shape;
};

struct HostEmbeddingInfo {
  HostEmbeddingInfo(const std::string& stream_handle,
                    const std::string& embedding_id, const Shape& indices_shape,
                    const Shape& activations_shape,
                    HostEmbeddingSplittingStrategy strategy =
                        HostEmbeddingSplittingStrategy::Token)
      : stream_handle(stream_handle),
        embedding_id(embedding_id),
        indices_shape(indices_shape),
        activations_shape(activations_shape),
        strategy(strategy) {}
  HostEmbeddingInfo() = delete;

  std::string stream_handle;
  std::string embedding_id;
  Shape indices_shape;
  Shape activations_shape;

  // Only used with experimental remote buffer embedding
  HostEmbeddingSplittingStrategy strategy;
};

struct RemoteParameterInfo {
  // Constructor used for lookups.
  explicit RemoteParameterInfo(int64 parameter_number)
      : RemoteParameterInfo(parameter_number, false, "", 0, 0) {}

  explicit RemoteParameterInfo(int64 parameter_number,
                               bool is_replica_partitioned,
                               const std::string& buffer_name,
                               int64 buffer_offset, int64 num_merged)
      : parameter_number(parameter_number),
        is_replica_partitioned(is_replica_partitioned),
        buffer_name(buffer_name),
        buffer_offset(buffer_offset),
        num_merged(num_merged) {}

  RemoteParameterInfo() = delete;

  const int64 parameter_number;
  const bool is_replica_partitioned;
  const std::string buffer_name;
  const int64 buffer_offset;
  const int64 num_merged;

  bool operator<(const RemoteParameterInfo& other) const {
    return parameter_number < other.parameter_number;
  }
};

using OutfeedInfos = std::set<FeedInfo>;
using InfeedInfos = std::set<FeedInfo>;
using StreamedInputInfos = std::set<FeedInfo>;
using StreamedOutputInfos = std::set<FeedInfo>;
using SendRecvInfos = std::vector<SendRecvInfo>;
using HostEmbeddingInfos = std::vector<HostEmbeddingInfo>;
using RemoteParameterInfos = std::set<RemoteParameterInfo>;

// We use this structure to communicate data about the DataStreams between the
// UserOp custom operation and the PoplarExecutable so it can link the streams
// to the tensor callbacks.
struct StreamCopyInfo {
  using FunctionTy = std::function<void(
      std::vector<void*>& data, std::vector<std::uint32_t>& number_of_elements,
      std::vector<void*>& outputs)>;

  StreamCopyInfo(const HloInstruction* inst, const std::string& handle,
                 std::uint32_t num_elems, std::uint32_t elem_size,
                 uint32_t operand, FunctionTy functor = nullptr)
      : parent_instruction(inst),
        stream_handle(handle),
        number_of_elements(num_elems),
        size_of_element(elem_size),
        operand_number(operand),
        callback_to_register(functor) {}
  StreamCopyInfo() = delete;

  // The instruction the user op came from. We use this as a unique identifier
  // for the inputs/outputs so we can sort the input/outputs by operation.
  const HloInstruction* parent_instruction;

  // The handle of the DataStream
  std::string stream_handle;

  // Number of elements we are sending.
  std::uint32_t number_of_elements;

  // The size of each element.
  std::uint32_t size_of_element;

  // We need to know what operand this is for outputs so we can map the output
  // to the correct memory location.
  uint32_t operand_number;

  // The call back to add, for inputs we add a call back which will popluate all
  // of the data arrays then call the user provided callback once they have been
  // populated. For outputs we don't add any callback and just use the default
  // copy into a memory location behaviour.
  FunctionTy callback_to_register;
};

// For each operation the user has added track all of the in/out streams
// assosiated with that instruction.
using StreamInfos = std::unordered_map<std::string, std::list<StreamCopyInfo>>;

// Stream meta info contains the information relating to the setup of the output
// streams. We need to know how many outputs there are and how much data to
// allocate in each buffer.
struct StreamCopyMetaInfo {
  StreamCopyMetaInfo() {}
  StreamCopyMetaInfo(const HloInstruction* inst, std::uint32_t input_count)
      : parent_instruction(inst), num_inputs(input_count) {}

  // The instruction the user op came from. We use this as a unique identifier
  // for the inputs/outputs so we can sort the input/outputs by operation.
  const HloInstruction* parent_instruction;

  // Track all of the output streams, we do this so we can allocate them in
  // advance.
  std::list<StreamCopyInfo*> output_stream_info;

  // The number of inputs this operation has.
  std::uint32_t num_inputs;
};

// We track one metainfo struct for each stream copy which the user has added.
using StreamMetaInfos = std::unordered_map<std::string, StreamCopyMetaInfo>;

// This structure contains all information which we generate that pertains
// to the XLA graph, as opposed to the poplar lowering of that graph.
struct CompilerAnnotations {
  CompilerAnnotations(const HloModule* module)
      : input_output_aliasing_map(module) {}

  InputOutputAliasingMap input_output_aliasing_map;

  TensorAllocationMap tensor_allocation_map;

  InfeedInfos infeed_infos;

  OutfeedInfos outfeed_infos;

  StreamInfos stream_infos;

  StreamMetaInfos stream_meta_infos;

  SendRecvInfos send_infos;
  SendRecvInfos recv_infos;

  HostEmbeddingInfos host_embedding_lookup_infos;
  HostEmbeddingInfos host_embedding_update_infos;
  HostEmbeddingInfos host_embedding_notify_infos;

  RemoteParameterInfos remote_parameter_infos;

  std::unique_ptr<HloModule> flattened_module;

  FlattenedInstMap flattened_inst_map_fwd;
  FlattenedInstMap flattened_inst_map_bwd;

  StreamedInputInfos streamed_input_infos;
  StreamedOutputInfos streamed_output_infos;
};

inline Status AddInfeedInfo(CompilerAnnotations& compiler_annotations,
                            const FeedInfo& feed_info) {
  auto other_info_itr = compiler_annotations.infeed_infos.find(feed_info);
  if (other_info_itr != compiler_annotations.infeed_infos.end() &&
      feed_info.shape != other_info_itr->shape) {
    return xla::FailedPrecondition(
        "Infeeds with matching name '%s' have different shapes.",
        feed_info.stream_prefix);
  }

  if (other_info_itr == compiler_annotations.infeed_infos.end()) {
    compiler_annotations.infeed_infos.insert(feed_info);
  }

  return Status::OK();
}

inline Status AddOutfeedInfo(CompilerAnnotations& compiler_annotations,
                             const FeedInfo& feed_info) {
  auto other_info_itr = compiler_annotations.outfeed_infos.find(feed_info);
  if (other_info_itr != compiler_annotations.outfeed_infos.end() &&
      feed_info.shape != other_info_itr->shape) {
    return xla::FailedPrecondition(
        "Outfeeds with matching name '%s' have different shapes.",
        feed_info.stream_prefix);
  }

  if (other_info_itr == compiler_annotations.outfeed_infos.end()) {
    compiler_annotations.outfeed_infos.insert(feed_info);
  }

  return Status::OK();
}

inline Status AddStreamedInputInfo(CompilerAnnotations& compiler_annotations,
                                   const FeedInfo& feed_info) {
  auto other_info_itr =
      compiler_annotations.streamed_input_infos.find(feed_info);
  if (other_info_itr != compiler_annotations.streamed_input_infos.end() &&
      feed_info.shape != other_info_itr->shape) {
    return xla::FailedPrecondition(
        "Streamed input with matching name '%s' have different shapes.",
        feed_info.stream_prefix);
  }

  if (other_info_itr == compiler_annotations.streamed_input_infos.end()) {
    compiler_annotations.streamed_input_infos.insert(feed_info);
  }

  return Status::OK();
}

inline Status AddStreamedOutputInfo(CompilerAnnotations& compiler_annotations,
                                    const FeedInfo& feed_info) {
  auto other_info_itr =
      compiler_annotations.streamed_output_infos.find(feed_info);
  if (other_info_itr != compiler_annotations.streamed_output_infos.end() &&
      feed_info.shape != other_info_itr->shape) {
    return xla::FailedPrecondition(
        "Streamed output with matching name '%s' have different shapes.",
        feed_info.stream_prefix);
  }

  if (other_info_itr == compiler_annotations.streamed_output_infos.end()) {
    compiler_annotations.streamed_output_infos.insert(feed_info);
  }

  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla

#endif
