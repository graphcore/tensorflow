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

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_feed_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/host_embedding.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/feed_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/input_output_aliasing_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

namespace xla {
namespace poplarplugin {
using FlattenedInstMap = absl::flat_hash_map<HloInstruction*, HloInstruction*>;

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

struct RemoteParameterHostRearrangement {
  using GatheredToRefSlice = std::vector<std::pair<int64, int64>>;

  int32 replication_factor = 0;
  int64 total_elements_per_replica = 0;
  GatheredToRefSlice gathered_to_ref_slice;
  std::vector<uint32> element_map;

  RemoteParameterHostRearrangement() = default;
  RemoteParameterHostRearrangement(
      int32 replication_factor, int64 total_elements_per_replica,
      const GatheredToRefSlice& gathered_to_ref_slice,
      const std::vector<uint32>& element_map)
      : replication_factor(replication_factor),
        total_elements_per_replica(total_elements_per_replica),
        gathered_to_ref_slice(gathered_to_ref_slice),
        element_map(element_map) {}
};

struct RemoteParameterInfo {
  // Constructor used for lookups.
  explicit RemoteParameterInfo(int64 parameter_number)
      : RemoteParameterInfo(parameter_number, false, "", 0, 0) {}

  explicit RemoteParameterInfo(int64 parameter_number,
                               bool is_replica_partitioned,
                               const std::string& buffer_name,
                               int64 buffer_offset, int64 num_merged,
                               const std::vector<int64>& merged_params = {},
                               int64 host_rearrangement_id = 0)
      : parameter_number(parameter_number),
        is_replica_partitioned(is_replica_partitioned),
        buffer_name(buffer_name),
        buffer_offset(buffer_offset),
        num_merged(num_merged),
        merged_params(merged_params),
        host_rearrangement_id(host_rearrangement_id) {}

  RemoteParameterInfo() = delete;

  const int64 parameter_number;
  const bool is_replica_partitioned;
  const std::string buffer_name;
  const int64 buffer_offset;
  const int64 num_merged;
  const std::vector<int64> merged_params;
  const int64 host_rearrangement_id;

  bool operator<(const RemoteParameterInfo& other) const {
    return parameter_number < other.parameter_number;
  }
};

struct InputInfo {
  std::string name;
  std::string handle;
  int64 argument;
  int64 tuple_index;
  Shape shape;

  bool operator<(const InputInfo& rhs) const {
    return std::tie(argument, tuple_index) <
           std::tie(rhs.argument, rhs.tuple_index);
  }
};

struct OutputInfo {
  std::string name;
  std::string handle;
  int64 tuple_index;
  Shape shape;

  bool operator<(const OutputInfo& rhs) const {
    return std::tie(tuple_index, handle) <
           std::tie(rhs.tuple_index, rhs.handle);
  }
};

using SendRecvInfos = std::vector<SendRecvInfo>;
using HostEmbeddingInfos = std::vector<HostEmbeddingInfo>;
using RemoteParameterInfos = std::set<RemoteParameterInfo>;
using RemoteParameterHostRearrangements =
    std::map<int64, RemoteParameterHostRearrangement>;
using InputInfos = std::set<InputInfo>;
using OutputInfos = std::set<OutputInfo>;

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

struct HostFunctionInfo {
  using FunctionType = std::function<void(const std::vector<const void*>& input,
                                          const std::vector<void*>& outputs)>;

  const HloInstruction* parent_instruction;
  std::string handle;
  std::vector<Shape> input_shapes;
  std::vector<Shape> output_shapes;
  FunctionType function;
};

using HostFunctionInfos = std::unordered_map<std::string, HostFunctionInfo>;

// This structure contains all information which we generate that pertains
// to the XLA graph, as opposed to the poplar lowering of that graph.
struct CompilerAnnotations {
  CompilerAnnotations(const HloModule* module)
      : input_output_aliasing_map(module) {}

  InputOutputAliasingMap input_output_aliasing_map;

  TensorAllocationMap tensor_allocation_map;

  CanonicalInfeedInfos infeed_infos;
  CanonicalOutfeedInfos outfeed_infos;

  StreamInfos stream_infos;

  SendRecvInfos send_infos;
  SendRecvInfos recv_infos;

  HostEmbeddingInfos host_embedding_lookup_infos;
  HostEmbeddingInfos host_embedding_update_infos;
  HostEmbeddingInfos host_embedding_notify_infos;

  RemoteParameterInfos remote_parameter_infos;
  RemoteParameterHostRearrangements remote_parameter_host_rearrangements;

  std::unique_ptr<HloModule> flattened_module;

  FlattenedInstMap flattened_inst_map_fwd;
  FlattenedInstMap flattened_inst_map_bwd;

  // The functional signature of the graph.
  // This is not used by TF, but is useful for external tools to be able to
  // interpret the content of a TF Poplar binary. Entry computation inputs
  // descriptions.
  InputInfos entry_input_infos;
  // Feed input descriptions.
  InputInfos feed_input_infos;

  // Entry computation output descriptions.
  OutputInfos entry_output_infos;
  // Feed output descriptions.
  OutputInfos feed_output_infos;

  // Host function information.
  HostFunctionInfos host_function_infos;
};

inline Status AddInfeedInfo(CompilerAnnotations& compiler_annotations,
                            const CanonicalFeedInfo& feed_info) {
  auto other_info_itr = compiler_annotations.infeed_infos.find(feed_info);
  if (other_info_itr != compiler_annotations.infeed_infos.end() &&
      feed_info.shape != other_info_itr->shape) {
    return xla::FailedPrecondition(
        "Infeeds with matching name '%s' have different shapes.",
        feed_info.config.feed_id());
  }

  if (other_info_itr == compiler_annotations.infeed_infos.end()) {
    compiler_annotations.infeed_infos.insert(feed_info);
  }

  return Status::OK();
}

inline Status AddOutfeedInfo(CompilerAnnotations& compiler_annotations,
                             const CanonicalFeedInfo& feed_info) {
  auto other_info_itr = compiler_annotations.outfeed_infos.find(feed_info);
  if (other_info_itr != compiler_annotations.outfeed_infos.end() &&
      feed_info.shape != other_info_itr->shape) {
    return xla::FailedPrecondition(
        "Outfeeds with matching name '%s' have different shapes.",
        feed_info.config.feed_id());
  }

  if (other_info_itr == compiler_annotations.outfeed_infos.end()) {
    compiler_annotations.outfeed_infos.insert(feed_info);
  }

  return Status::OK();
}

inline Status AddEntryInputInfo(CompilerAnnotations& compiler_annotations,
                                const InputInfo& input_info) {
  auto other_info_itr = compiler_annotations.entry_input_infos.find(input_info);
  if (other_info_itr != compiler_annotations.entry_input_infos.end() &&
      input_info.shape != other_info_itr->shape) {
    return xla::FailedPrecondition(
        "Input with matching name '%s' and tuple index %d have different "
        "shapes (%s != %s).",
        input_info.name, input_info.tuple_index, input_info.shape.ToString(),
        other_info_itr->shape.ToString());
  }

  if (other_info_itr == compiler_annotations.entry_input_infos.end()) {
    compiler_annotations.entry_input_infos.insert(input_info);
  }

  return Status::OK();
}

inline Status AddFeedInputInfo(CompilerAnnotations& compiler_annotations,
                               const InputInfo& input_info) {
  auto other_info_itr = compiler_annotations.feed_input_infos.find(input_info);
  if (other_info_itr != compiler_annotations.feed_input_infos.end() &&
      input_info.shape != other_info_itr->shape) {
    return xla::FailedPrecondition(
        "Streamed input with matching name '%s' and tuple index %d have "
        "different shapes (%s != %s).",
        input_info.name, input_info.tuple_index, input_info.shape.ToString(),
        other_info_itr->shape.ToString());
  }

  if (other_info_itr == compiler_annotations.feed_input_infos.end()) {
    compiler_annotations.feed_input_infos.insert(input_info);
  }

  return Status::OK();
}

inline Status AddEntryOutputInfo(CompilerAnnotations& compiler_annotations,
                                 const OutputInfo& output_info) {
  auto other_info_itr =
      compiler_annotations.entry_output_infos.find(output_info);
  if (other_info_itr != compiler_annotations.entry_output_infos.end() &&
      output_info.shape != other_info_itr->shape) {
    return xla::FailedPrecondition(
        "Output with matching handle '%s'=='%s' have different shapes (%s != "
        "%s).",
        output_info.handle, other_info_itr->handle,
        output_info.shape.ToString(), other_info_itr->shape.ToString());
  }

  if (other_info_itr == compiler_annotations.entry_output_infos.end()) {
    compiler_annotations.entry_output_infos.insert(output_info);
  }

  return Status::OK();
}

inline Status AddFeedOutputInfo(CompilerAnnotations& compiler_annotations,
                                const OutputInfo& output_info) {
  auto other_info_itr =
      compiler_annotations.feed_output_infos.find(output_info);
  if (other_info_itr != compiler_annotations.feed_output_infos.end() &&
      output_info.shape != other_info_itr->shape) {
    return xla::FailedPrecondition(
        "Streamed output with matching handle '%s' have different shapes (%s "
        "!= %s).",
        output_info.handle, output_info.shape.ToString(),
        other_info_itr->shape.ToString());
  }

  if (other_info_itr == compiler_annotations.feed_output_infos.end()) {
    compiler_annotations.feed_output_infos.insert(output_info);
  }

  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla

#endif
