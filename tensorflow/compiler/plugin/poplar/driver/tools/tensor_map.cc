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
#include "tensorflow/compiler/plugin/poplar/driver/tools/tensor_map.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "include/json/json.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/core/lib/core/errors.h"

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {

const TensorMap& TensorMaps::GetTensorMapForComputation(
    const std::string& computation_name) const {
  return _map.at(computation_name);
}
void TensorMaps::AddTensorMapForComputation(const std::string& computation_name,
                                            TensorMap tensor_map) {
  _map[computation_name] = std::move(tensor_map);
}

Status TensorMap::AddOutputTensor(const HloInstruction* inst,
                                  int64 output_index, poplar::Tensor tensor) {
  VLOG(2) << "Adding output tensor for instruction " << inst->name()
          << " at output index " << output_index;

  TensorLocation location(inst, output_index);
  auto it = _map.find(location);
  if (it != _map.end()) {
    return tensorflow::errors::Unknown(
        StrCat("[Poplar] Output Tensor ", location.flattened_output_tuple_index,
               " for ", GetDebugName(inst), " already exists"));
  }
  _map[location].tensor = tensor;
  _map[location].name = inst->metadata().op_name();
  return Status::OK();
}

Status TensorMap::AddOutputRemoteBuffer(const HloInstruction* inst,
                                        int64 output_index,
                                        poplar::RemoteBuffer rbuffer) {
  return AddOutputRemoteBufferImpl(inst, output_index, rbuffer, false,
                                   absl::nullopt);
}

Status TensorMap::AddOutputRemoteBuffer(const HloInstruction* inst,
                                        int64 output_index,
                                        poplar::RemoteBuffer rbuffer,
                                        bool is_replica_partitioned) {
  return AddOutputRemoteBufferImpl(inst, output_index, rbuffer,
                                   is_replica_partitioned, absl::nullopt);
}

Status TensorMap::AddOutputRemoteBuffer(const HloInstruction* inst,
                                        int64 output_index,
                                        poplar::RemoteBuffer rbuffer,
                                        int64 slice_dimension) {
  return AddOutputRemoteBufferImpl(inst, output_index, rbuffer, false,
                                   slice_dimension);
}

Status TensorMap::AddOutputRemoteBuffer(const HloInstruction* inst,
                                        int64 output_index,
                                        poplar::RemoteBuffer rbuffer,
                                        bool is_replica_partitioned,
                                        int64 slice_dimension) {
  return AddOutputRemoteBufferImpl(inst, output_index, rbuffer,
                                   is_replica_partitioned, slice_dimension);
}

Status TensorMap::AddOutput(const HloInstruction* inst, int64 output_index,
                            TensorOrRemoteBuffer torb) {
  VLOG(1) << "Adding output for instruction " << inst->name()
          << " at output index " << output_index;

  TensorLocation location(inst, output_index);
  auto it = _map.find(location);
  if (it != _map.end()) {
    return tensorflow::errors::Unknown(
        StrCat("[Poplar] Output ", location.flattened_output_tuple_index,
               " for ", GetDebugName(inst), " already exists"));
  }

  if (!torb.IsTensor() && !torb.IsRemoteBuffer()) {
    return tensorflow::errors::Unknown(StrCat(
        "[Poplar] Output ", location.flattened_output_tuple_index, " for ",
        GetDebugName(inst), " is neither a tensor or a remote buffer."));
  }

  _map[location].tensor = torb;
  _map[location].name = inst->metadata().op_name();
  return Status::OK();
}

Status TensorMap::AddOutputRemoteBufferImpl(
    const HloInstruction* inst, int64 output_index,
    poplar::RemoteBuffer rbuffer, bool is_replica_partitioned,
    absl::optional<int64> slice_dimension) {
  VLOG(2) << "Adding output remote buffer for instruction " << inst->name()
          << " at output index " << output_index;

  TensorLocation location(inst, output_index);
  auto it = _map.find(location);
  if (it != _map.end()) {
    return tensorflow::errors::Unknown(StrCat(
        "[Poplar] Output RemoteBuffer ", location.flattened_output_tuple_index,
        " for ", GetDebugName(inst), " already exists"));
  }
  if (slice_dimension) {
    return tensorflow::errors::Unknown(StrCat(
        "[Poplar] Output RemoteBuffer ", location.flattened_output_tuple_index,
        " for ", GetDebugName(inst),
        " has a slice dimension, but this is not currently supported."));
  }

  _map[location].tensor =
      TensorOrRemoteBuffer(rbuffer, is_replica_partitioned, slice_dimension);
  _map[location].name = inst->metadata().op_name();

  return Status::OK();
}

poplar::Tensor TensorMap::GetTensor(TensorLocation location) const {
  return _map.at(location).tensor;
}

Status TensorMap::UpdateTensor(TensorLocation location, poplar::Tensor tensor) {
  auto it = _map.find(location);
  if (it == _map.end()) {
    return tensorflow::errors::Unknown(
        StrCat("[Poplar] Output Tensor ", location.instruction->name(), ":",
               location.flattened_output_tuple_index, " not found"));
  }
  it->second.tensor = tensor;
  return Status::OK();
}

void TensorMap::Clear() { _map.clear(); }

TensorOrRemoteBufferVector TensorMap::FindInstructionOutputs(
    const HloInstruction* inst, absl::optional<int64> opt_tensors_start,
    absl::optional<int64> opt_tensors_end) const {
  NamedTensorLocationVector tensor_vector = FindInstructionNamedTensorLocations(
      inst, opt_tensors_start, opt_tensors_end);
  TensorOrRemoteBufferVector outputs;
  absl::c_transform(
      tensor_vector, std::back_inserter(outputs),
      [](const NamedTensorLocation& value) { return value.tensor; });

  return outputs;
}

StatusOr<TensorVector> TensorMap::FindInstructionOutputTensors(
    const HloInstruction* inst, absl::optional<int64> opt_tensors_start,
    absl::optional<int64> opt_tensors_end) const {
  TensorOrRemoteBufferVector outputs =
      FindInstructionOutputs(inst, opt_tensors_start, opt_tensors_end);

  TensorVector result;

  for (int i = 0; i < outputs.size(); ++i) {
    if (!outputs[i].IsTensor()) {
      return tensorflow::errors::FailedPrecondition(
          "Expected all outputs of " + inst->name() +
          " to be poplar tensors, but output " + std::to_string(i) + " is not");
    }

    result.push_back(outputs[i]);
  }

  return result;
}

TensorMap::NamedTensorLocationVector
TensorMap::FindInstructionNamedTensorLocations(
    const HloInstruction* inst, absl::optional<int64> opt_tensors_start,
    absl::optional<int64> opt_tensors_end) const {
  TensorLocation lower(inst, DefaultToFirst(opt_tensors_start));
  TensorLocation upper(inst, DefaultToLast(opt_tensors_end) - 1);

  NamedTensorLocationVector outputs;
  std::transform(_map.lower_bound(lower), _map.upper_bound(upper),
                 std::back_inserter(outputs),
                 [](const std::pair<TensorLocation, NamedTensor>& pair) {
                   return NamedTensorLocation(pair.first, pair.second);
                 });

  return outputs;
}
poplar::Tensor TensorMap::FindTensorByName(const std::string& name,
                                           int64 output_index) const {
  for (auto it : _map) {
    if (it.first.instruction->name() == name &
        it.first.flattened_output_tuple_index == output_index) {
      return it.second.tensor;
    }
  }
}

TensorOrRemoteBufferVectors CastTensorVectors(
    const TensorVectors& tensor_vectors) {
  TensorOrRemoteBufferVectors result;
  result.reserve(tensor_vectors.size());

  for (auto& tensor_vector : tensor_vectors) {
    TensorOrRemoteBufferVector trb_vector;
    trb_vector.reserve(tensor_vector.size());

    for (auto& tensor : tensor_vector) {
      trb_vector.emplace_back(tensor);
    }
    result.push_back(trb_vector);
  }

  return result;
}
}  // namespace poplarplugin
}  // namespace xla
