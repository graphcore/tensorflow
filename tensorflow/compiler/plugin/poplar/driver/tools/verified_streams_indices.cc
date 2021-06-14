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
#include "tensorflow/compiler/plugin/poplar/driver/tools/verified_streams_indices.h"

#include <string>
#include <utility>
#include <vector>

#include "ipu/poplar_executable_data.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"

#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <poputil/TileMapping.hpp>

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {

namespace {
std::vector<std::string> GetFeedsNames(
    const std::map<std::string, const IpuOptions::VerifiedInfo>& m) {
  std::vector<std::string> keys;
  keys.reserve(m.size());
  absl::c_transform(
      m, std::back_inserter(keys),
      [](const std::pair<std::string, const IpuOptions::VerifiedInfo>& pair) {
        return pair.first;
      });
  return keys;
}

}  // namespace

StatusOr<IpuOptions::VerifiedInfo> VerifiedStreamsIndices::GetFeedInfo(
    const std::string& feed_name) const {
  auto it = feeds_info_.find(feed_name);
  if (it == feeds_info_.end()) {
    return tensorflow::errors::InvalidArgument(
        "Feed '", feed_name, "' not found in the IpuOptions list: [",
        absl::StrJoin(GetFeedsNames(feeds_info_), ", "), "]");
  }
  return it->second;
}

StatusOr<int64> VerifiedStreamsIndices::GetFeedNumStreams(
    const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kInfeed: {
      return inst->shape().tuple_shapes(0).tuple_shapes_size();
    }
    case HloOpcode::kOutfeed: {
      return inst->shape().IsTuple() ? inst->shape().tuple_shapes_size() : 1;
    }
    default: {
      return FailedPrecondition(
          "Expected instruction to either be an Infeed or an Outfeed");
    }
  }
}

StatusOr<int64> VerifiedStreamsIndices::GetStreamId(
    const std::string& feed_name, int64 stream_idx,
    const HloInstruction* inst) {
  auto it = feeds_start_ids_.find(feed_name);
  if (it == feeds_start_ids_.end()) {
    TF_ASSIGN_OR_RETURN(int64 num_streams, GetFeedNumStreams(inst));
    int64 start_id = next_start_id_;
    feeds_start_ids_[feed_name] = start_id;
    next_start_id_ += num_streams;
    return start_id + stream_idx;
  }
  return it->second + stream_idx;
}

poplar::Tensor VerifiedStreamsIndices::GetFeedIndexTensor(
    const std::string& feed_name) {
  int i = absl::c_find(feeds_, feed_name) - feeds_.begin();
  if (i == feeds_.size()) {
    feeds_.push_back(feed_name);
  }
  // Each index is one U64 value made of 2x U32.
  return checkpoint_tensor_.slice(i * 2, i * 2 + 2);
}

const std::vector<std::string>& VerifiedStreamsIndices::CheckpointFeedsOrder()
    const {
  return feeds_;
}

Status VerifiedStreamsIndices::InitializeFeedStream(
    const std::string& feed_name, int64 stream_idx,
    poplar::program::Sequence& seq, const HloInstruction* inst,
    const poplar::DebugNameAndId& debug_name_and_id) {
  // Only needed for verified streams.
  if (!resources_->use_verified_transfers) {
    return Status::OK();
  }
  if (feeds_streams_.find(feed_name) == feeds_streams_.end()) {
    TF_ASSIGN_OR_RETURN(IpuOptions::VerifiedInfo info, GetFeedInfo(feed_name));
    if (info.start_id() >= 0 && next_start_id_ > 0) {
      return xla::FailedPrecondition(
          "The ids must either all be generated automatically or all set "
          "manually");
    }
    if (info.start_id() >= 0) {
      info.set_start_id(info.start_id() + stream_idx);
    } else {
      TF_ASSIGN_OR_RETURN(int64 stream_id,
                          GetStreamId(feed_name, stream_idx, inst));
      info.set_start_id(stream_id);
    }
    Index new_index{feed_name};
    new_index.IncrementNumTensors();
    poplar::Tensor feed_index = GetFeedIndexTensor(feed_name);
    poplar::Tensor stream_index;
    if (stream_idx > 0) {
      // Create a copy to avoid conflicts with other streams.
      TF_ASSIGN_OR_RETURN(
          stream_index,
          AddPlainTensor(GetMasterGraph(*resources_),
                         {debug_name_and_id, absl::StrCat(feed_name, "Index")},
                         XlaShapeFromPoplarShape(xla::PrimitiveType::U32, {2}),
                         *resources_));
      load_checkpoint_.add(poplar::program::Copy(feed_index, stream_index,
                                                 false, {debug_name_and_id}));
    } else {
      stream_index = feed_index;
    }
    new_index.Initialize(*resources_, stream_index);
    new_index.SetKeyAndStartId(info.key(), info.start_id());
    feeds_streams_.insert({feed_name, new_index});
  }
  return Status::OK();
}

Status VerifiedStreamsIndices::InitializeIndexTensors(
    CompilerResources& resources, const IpuOptions::VerifiedTransfers& opts,
    const poplar::DebugNameAndId& debug_name_and_id) {
  if (resources_ != nullptr) {
    return xla::FailedPrecondition("Indices already allocated");
  }
  resources_ = &resources;

  // Indices are only needed for verified streams.
  if (!resources_->use_verified_transfers) {
    return Status::OK();
  }

  const InputOutputAliasingMap& io_map =
      resources.annotations.input_output_aliasing_map;
  const auto& inputs = io_map.GetEntryInputInfos();
  const auto& outputs = io_map.GetEntryOutputInfos();
  std::list<std::pair<std::string, Index*>> handles;
  for (auto input : inputs) {
    Index* index = &input_data_;
    if (!input.IsStreaming()) {
      if (!input.IsResource()) {
        return xla::FailedPrecondition(
            "Input should be either a stream or a resource");
      }
      index = &input_parameters_;
    }
    index->IncrementNumTensors();
    for (auto handle : input.Handles()) {
      handles.emplace_back(handle, index);
    }
  }

  for (auto output : outputs) {
    Index* index = &output_data_;
    if (!output.IsStreaming()) {
      if (!output.IsResource()) {
        return xla::FailedPrecondition(
            "Output should be either a stream or a resource");
      }
      if (output.IsResourceModified()) {
        output_parameters_.IncrementNumTensors();
        continue;
      } else {
        index = &output_parameters_;
      }
    }
    index->IncrementNumTensors();
    for (auto handle : output.Handles()) {
      handles.emplace_back(handle, index);
    }
  }

  if (io_map.GetNumStreamingInputs() != input_data_.NumTensors()) {
    return xla::FailedPrecondition(
        "Number of inputs not matching the one from the aliasing map");
  }
  if (io_map.GetEntryInputInfos().size() - io_map.GetNumStreamingInputs() !=
      input_parameters_.NumTensors()) {
    return xla::FailedPrecondition(
        "Number of input parameters not matching the one from the aliasing "
        "map");
  }

  if (io_map.GetNumStreamingOutputs() != output_data_.NumTensors()) {
    return xla::FailedPrecondition(
        "Number of outputs not matching the one from the aliasing map");
  }
  if (io_map.GetEntryOutputInfos().size() - io_map.GetNumStreamingOutputs() !=
      output_parameters_.NumTensors()) {
    return xla::FailedPrecondition(
        "Number of output parameters not matching the one from the aliasing "
        "map");
  }

  xla::Literal zeroes = LiteralUtil::CreateR1<uint32>({0, 0});
  poplar::Graph& graph = GetMasterGraph(*resources_);
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor zero_tensor,
      CreateConstantTensor(graph, zeroes, zeroes.shape(), poplar::UNSIGNED_INT,
                           {debug_name_and_id, "Index0"}));

  input_data_.Initialize(*resources_, zero_tensor);
  output_data_.Initialize(*resources_, zero_tensor);
  input_parameters_.Initialize(*resources_, zero_tensor);
  output_parameters_.Initialize(*resources_, zero_tensor);

  TF_RETURN_IF_ERROR(SetKeysAndStartIds(opts, debug_name_and_id));

  // Generate the key / id pairs for all the handles
  for (auto handle : handles) {
    KeyIdPair new_pair = handle.second->NextKeyIdPair();
    assigned_ids_.insert({handle.first, new_pair});
  }

  // For modified resources use the same id as the matching input.
  for (auto output : outputs) {
    Index* index = &output_data_;
    if (!output.IsStreaming() && output.IsResourceModified()) {
      int index = 0;
      for (auto handle : output.Handles()) {
        const std::string input_handle =
            GetInputCopyHandle(output.GetInputIndex(), index);
        auto it = assigned_ids_.find(input_handle);
        if (it == assigned_ids_.end()) {
          return xla::FailedPrecondition(
              "Couldn't find output parameter %s's matching input: %s",
              output.Handles().at(index), input_handle);
        }
        KeyIdPair new_pair{output_parameters_.GetKeyIdPair().key,
                           it->second.id};
        assigned_ids_.insert({handle, new_pair});
        index++;
      }
    }
  }
  return Status::OK();
}

StatusOr<poplar::Tensor> VerifiedStreamsIndices::IndexTensor(
    const InputOutputAliasingMap::InputInfo& info, const HloInstruction* inst,
    poplar::program::Sequence& seq) {
  if (info.IsStreaming()) {
    return input_data_.IndexTensor(inst, seq, *resources_);
  }
  return input_parameters_.IndexTensor(inst, seq, *resources_);
}

StatusOr<poplar::Tensor> VerifiedStreamsIndices::IndexTensor(
    const InputOutputAliasingMap::OutputInfo& info, const HloInstruction* inst,
    poplar::program::Sequence& seq) {
  if (info.IsStreaming()) {
    return output_data_.IndexTensor(inst, seq, *resources_);
  }
  return output_parameters_.IndexTensor(inst, seq, *resources_);
}

StatusOr<poplar::Tensor> VerifiedStreamsIndices::IndexTensor(
    const std::string& feed_stream_handle, const HloInstruction* inst,
    poplar::program::Sequence& seq) {
  return feeds_streams_.at(feed_stream_handle)
      .IndexTensor(inst, seq, *resources_);
}

const VerifiedStreamsIndices::KeyIdMappings&
VerifiedStreamsIndices::GetAssignedIds() const {
  return assigned_ids_;
}

VerifiedStreamsIndices::Index::Index(const std::string& name)
    : name_(name), id_(0), num_tensors_(0) {}

StatusOr<poplar::Tensor> VerifiedStreamsIndices::Index::IndexTensor(
    const HloInstruction* inst, poplar::program::Sequence& seq,
    CompilerResources& resources) {
  poplar::Graph& graph = GetGraph(resources, inst);
  auto it_tensor = base_indices_.find(&graph);
  // If we don't have a version of the index tensor in that graph then copy it.
  if (it_tensor == base_indices_.end()) {
    if (inst->has_sharding() && !inst->sharding().HasUniqueDevice()) {
      return FailedPrecondition("Only single device sharding is supported.");
    }
    unsigned dst_device_id = GetShardForOutputIndex(inst, 0);
    poplar::Graph& master_graph = GetMasterGraph(resources);
    poplar::Tensor tensor = poputil::copyToIpu(
        master_graph, base_indices_.at(&master_graph), seq, dst_device_id,
        absl::StrCat("Index_", name_, "_", dst_device_id),
        poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    base_indices_.insert({&graph, tensor});
    return tensor;
  }
  // Otherwise return the existing tensor
  return it_tensor->second;
}

void VerifiedStreamsIndices::Index::Initialize(CompilerResources& resources,
                                               poplar::Tensor index) {
  poplar::Graph& graph = GetMasterGraph(resources);
  base_indices_.insert({&graph, index});
}

VerifiedStreamsIndices::KeyIdPair
VerifiedStreamsIndices::Index::NextKeyIdPair() {
  KeyIdPair pair{key_, id_};
  id_++;
  return pair;
}

VerifiedStreamsIndices::KeyIdPair VerifiedStreamsIndices::Index::GetKeyIdPair()
    const {
  return {key_, id_};
}

void VerifiedStreamsIndices::Index::IncrementNumTensors() { num_tensors_++; }

uint64 VerifiedStreamsIndices::Index::NumTensors() const {
  return num_tensors_;
}

void VerifiedStreamsIndices::Index::SetKeyAndStartId(uint64 key, uint64 id) {
  id_ = id;
  SetKey(key);
}

void VerifiedStreamsIndices::Index::SetKey(uint64 key) { key_ = key; }

Status VerifiedStreamsIndices::CreateCheckpointLoadSave(
    const IpuOptions::VerifiedTransfers& opts,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::Graph& graph = GetMasterGraph(*resources_);
  TF_ASSIGN_OR_RETURN(
      checkpoint_tensor_,
      AddPlainTensor(graph,
                     {debug_name_and_id, ipu::Metadata::CheckpointName()},
                     XlaShapeFromPoplarShape(xla::PrimitiveType::U32,
                                             {2 * feeds_info_.size()}),
                     *resources_));

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor checkpoint_idx,
      AddPlainTensor(
          graph, {debug_name_and_id, ipu::Metadata::InputCheckpointIndexName()},
          XlaShapeFromPoplarShape(xla::PrimitiveType::U32, {2}), *resources_));

  auto fifo_index = graph.addHostToDeviceFIFO(
      ipu::Metadata::InputCheckpointIndexHandle(), checkpoint_idx.elementType(),
      checkpoint_idx.numElements());
  load_checkpoint_.add(poplar::program::Copy(fifo_index, checkpoint_idx, false,
                                             {debug_name_and_id}));

  auto fifo_in = graph.addHostToDeviceFIFO(
      ipu::Metadata::InputCheckpointHandle(), checkpoint_tensor_.elementType(),
      checkpoint_tensor_.numElements(), poplar::ReplicatedStreamMode::BROADCAST,
      {{"streamVerification", "true"},
       {"key", absl::StrCat(opts.checkpoint_in().key())},
       {"id", absl::StrCat(opts.checkpoint_in().start_id())}});
  load_checkpoint_.add(poplar::program::Copy(
      fifo_in, checkpoint_tensor_, checkpoint_idx, false,
      {{"streamVerification", "true"}}, {debug_name_and_id}));
  assigned_ids_.insert(
      {fifo_in.handle(),
       {opts.checkpoint_in().key(), opts.checkpoint_in().start_id()}});

  // Increment the index by one.
  popops::mapInPlace(graph, pe::Add(pe::_1, pe::Const(1)),
                     {checkpoint_idx.slice(0, 1)}, load_checkpoint_,
                     {debug_name_and_id, "CheckpointIndexInc"});

  auto fifo_out = graph.addDeviceToHostFIFO(
      ipu::Metadata::OutputCheckpointHandle(), checkpoint_tensor_.elementType(),
      checkpoint_tensor_.numElements(),
      {{"streamVerification", "true"},
       {"key", absl::StrCat(opts.checkpoint_out().key())},
       {"id", absl::StrCat(opts.checkpoint_out().start_id())}});
  save_checkpoint_.add(poplar::program::Copy(
      checkpoint_tensor_, fifo_out, checkpoint_idx, false,
      {{"streamVerification", "true"}}, {debug_name_and_id}));
  assigned_ids_.insert(
      {fifo_out.handle(),
       {opts.checkpoint_out().key(), opts.checkpoint_out().start_id()}});
  auto fifo_out_clear = graph.addDeviceToHostFIFO(
      ipu::Metadata::OutputClearCheckpointHandle(),
      checkpoint_tensor_.elementType(), checkpoint_tensor_.numElements());
  save_checkpoint_.add(poplar::program::Copy(checkpoint_tensor_, fifo_out_clear,
                                             false, {debug_name_and_id}));

  return Status::OK();
}

Status VerifiedStreamsIndices::SetKeysAndStartIds(
    const IpuOptions::VerifiedTransfers& opts,
    const poplar::DebugNameAndId& debug_name_and_id) {
  std::vector<std::pair<Index*, IpuOptions::VerifiedInfo>> indices;

  absl::c_transform(
      opts.infeeds(), std::inserter(feeds_info_, feeds_info_.end()),
      [](const std::pair<std::string, IpuOptions::VerifiedInfo>& pair) {
        return std::make_pair(pair.first, pair.second);
      });
  absl::c_transform(
      opts.outfeeds(), std::inserter(feeds_info_, feeds_info_.end()),
      [](const std::pair<std::string, IpuOptions::VerifiedInfo>& pair) {
        return std::make_pair(pair.first, pair.second);
      });
  if (feeds_info_.size() > 0) {
    TF_RETURN_IF_ERROR(CreateCheckpointLoadSave(opts, debug_name_and_id));
  }

  if (input_data_.NumTensors() > 0) {
    indices.emplace_back(&input_data_, opts.inputs());
  }
  if (input_parameters_.NumTensors() > 0) {
    indices.emplace_back(&input_parameters_, opts.input_parameters());
  }
  if (output_data_.NumTensors() > 0) {
    indices.emplace_back(&output_data_, opts.outputs());
  }
  if (output_parameters_.NumTensors() > 0) {
    // If there are more output parameters than just the modified ones.
    if (output_parameters_.NumTensors() > input_parameters_.NumTensors()) {
      indices.emplace_back(&output_parameters_, opts.output_parameters());
    } else {
      output_parameters_.SetKey(opts.output_parameters().key());
    }
  }

  // Generate Ids automatically if needed.
  // Use a global counter (next_start_id_ ) and starting from 0 assign a
  // unique id to each input data, input parameter and output data tensor.
  // Output parameters will be assigned the same IDs as the input parameters
  // (This is needed in order to be able to use the output of a
  // run as the input of another one.
  if (absl::c_any_of(
          indices, [](const std::pair<Index*, IpuOptions::VerifiedInfo>& pair) {
            return pair.second.start_id() < 0;
          })) {
    if (!absl::c_all_of(
            indices,
            [](const std::pair<Index*, IpuOptions::VerifiedInfo>& pair) {
              return pair.second.start_id() < 0;
            })) {
      return xla::FailedPrecondition(
          "The ids must either all be generated automatically or all set "
          "manually");
    }
    for (auto& pair : indices) {
      pair.second.set_start_id(next_start_id_);
      next_start_id_ += pair.first->NumTensors();
    }
  }

  for (auto& pair : indices) {
    pair.first->SetKeyAndStartId(pair.second.key(), pair.second.start_id());
  }

  return Status::OK();
}

poplar::OptionFlags VerifiedStreamsIndices::GraphOptions(
    const std::string& handle) const {
  if (resources_->use_verified_transfers) {
    auto key_id = assigned_ids_.at(handle);
    return {{"streamVerification", "true"},
            {"key", absl::StrCat(key_id.key)},
            {"id", absl::StrCat(key_id.id)}};
  }
  return poplar::OptionFlags();
}

poplar::OptionFlags VerifiedStreamsIndices::GraphFeedOptions(
    const std::string& feed_stream_handle) {
  if (resources_->use_verified_transfers) {
    const Index& idx = feeds_streams_.at(feed_stream_handle);
    KeyIdPair new_pair = idx.GetKeyIdPair();
    assigned_ids_.insert({feed_stream_handle, new_pair});
    return {{"streamVerification", "true"},
            {"key", absl::StrCat(new_pair.key)},
            {"id", absl::StrCat(new_pair.id)}};
  }
  return poplar::OptionFlags();
}

poplar::OptionFlags VerifiedStreamsIndices::CopyOptions() const {
  if (resources_->use_verified_transfers) {
    return {{"streamVerification", "true"}};
  }
  return poplar::OptionFlags();
}

poplar::program::Sequence VerifiedStreamsIndices::LoadCheckpointSequence()
    const {
  return load_checkpoint_;
}

poplar::program::Sequence VerifiedStreamsIndices::SaveCheckpointSequence()
    const {
  return save_checkpoint_;
}

}  // namespace poplarplugin
}  // namespace xla
