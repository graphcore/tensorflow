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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_COMPILER_ANNOTATIONS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_COMPILER_ANNOTATIONS_H_

#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_feed_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/input_output_aliasing_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

namespace xla {
using FlattenedInstMap = absl::flat_hash_map<HloInstruction*, HloInstruction*>;

class HloInfeedInstruction;

namespace poplarplugin {

struct FeedInfo {
  FeedInfo(const std::string& stream_prefix, const PoplarFeedConfig& config,
           const Shape& shape)
      : stream_prefix(stream_prefix), config(config), shape(shape) {}
  FeedInfo() = delete;

  std::string stream_prefix;
  PoplarFeedConfig config;
  Shape shape;
};

using OutfeedInfos = std::vector<FeedInfo>;
using InfeedInfos = std::vector<FeedInfo>;

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

  DeferredAllocations deferred_allocations;

  InfeedInfos infeed_infos;

  OutfeedInfos outfeed_infos;

  StreamInfos stream_infos;

  StreamMetaInfos stream_meta_infos;

  TensorsWithLayouts tensors_with_layout;

  std::unique_ptr<HloModule> flattened_module;

  FlattenedInstMap flattened_inst_map_fwd;
  FlattenedInstMap flattened_inst_map_bwd;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
