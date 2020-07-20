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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_TENSOR_MAP_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_TENSOR_MAP_H_

/*
 * These functions are related to poplar, and cannot be used within the
 * optimizers target in the BUILD file.
 */

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/tensor_location.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"

#include <poplar/DataStream.hpp>
#include <poplar/Tensor.hpp>

namespace xla {
class HloInstruction;

namespace poplarplugin {

/**
 * A struct that can hold either a poplar tensor or a poplar remote buffer.
 *
 * The operator overloads are design so that it implicitely casts to the
 * appropriate type, when the context is unambiguous. This should minimise the
 * need for code changes where poplar tensors are assumed.
 */
struct TensorOrRemoteBuffer {
  /**
   * The inner storage is a variable of a tensor or a remote buffer.
   */
  absl::variant<poplar::Tensor, poplar::RemoteBuffer> element;

  /**
   * Default construct the element as an empty tensor.
   *
   * This preserves the existing behaviour where we would always default
   * construct the tensor, because it could only be a tensor.
   */
  TensorOrRemoteBuffer() : element(poplar::Tensor()) {}

  /**
   * Construct with a poplar tensor.
   */
  explicit TensorOrRemoteBuffer(poplar::Tensor tensor) : element(tensor) {}

  /**
   * Construct with a poplar remote buffer.
   */
  explicit TensorOrRemoteBuffer(poplar::RemoteBuffer rbuffer)
      : element(rbuffer) {}

  /**
   * Helper function to test whether a tensor is stored in the element.
   */
  bool IsTensor() const { return element.index() == 0; }

  /**
   * Helper function to test whether a remote buffer is stored in the element.
   */
  bool IsRemoteBuffer() const { return element.index() == 1; }

  /**
   * Helper functions to force the cast to a poplar tensor when it is
   * unambiguous.
   */
  poplar::Tensor AsTensor() const { return *this; }

  /**
   * Helper functions to force the cast to a poplar remote buffer when it is
   * unambiguous.
   */
  poplar::RemoteBuffer AsRemoteBuffer() const { return *this; }

  /**
   * Operator overloads to support implicit casts.
   */
  operator poplar::Tensor() const { return absl::get<0>(element); }
  operator poplar::RemoteBuffer() const { return absl::get<1>(element); }

  /**
   * Support assignment, like this is a poplar tensor.
   */
  TensorOrRemoteBuffer& operator=(poplar::Tensor tensor) {
    element = tensor;
    return *this;
  }

  /**
   * Support assignment, like this is a poplar remote buffer.
   */
  TensorOrRemoteBuffer& operator=(poplar::RemoteBuffer rbuffer) {
    element = rbuffer;
    return *this;
  }

  /**
   * In a few places, tensor equality is checked. This operator overload allows
   * that code to continue working.
   */
  inline bool operator==(const TensorOrRemoteBuffer& rhs) const {
    if (IsRemoteBuffer() && rhs.IsRemoteBuffer()) {
      return AsRemoteBuffer() == rhs.AsRemoteBuffer();
    }

    if (IsTensor() && rhs.IsTensor()) {
      return AsTensor() == rhs.AsTensor();
    }

    return false;
  }

  inline bool operator!=(const TensorOrRemoteBuffer& rhs) const {
    return !(*this == rhs);
  }
};

using TensorOrRemoteBufferVector = std::vector<TensorOrRemoteBuffer>;
using TensorOrRemoteBufferVectors = std::vector<TensorOrRemoteBufferVector>;

using TensorVector = std::vector<poplar::Tensor>;
using TensorVectors = std::vector<TensorVector>;

/**
 * Cast a TensorVectors to a TensorOrRemoteBufferVectors.
 *
 * None of the elements are "converted" to remote buffers, instead this helps
 * use API mismatches.
 */
TensorOrRemoteBufferVectors CastTensorVectors(
    const TensorVectors& tensor_vectors);

class TensorMap {
 public:
  struct NamedTensor {
    std::string name;
    TensorOrRemoteBuffer tensor;
  };
  struct NamedTensorLocation {
    NamedTensorLocation(const TensorLocation& loc,
                        const NamedTensor& named_tensor)
        : location(loc), name(named_tensor.name), tensor(named_tensor.tensor) {}
    explicit NamedTensorLocation(
        const std::pair<TensorLocation, NamedTensor>& pair)
        : NamedTensorLocation(pair.first, pair.second) {}
    TensorLocation location;
    std::string name;
    TensorOrRemoteBuffer tensor;
  };
  using Iterator =
      MapIterator<TensorLocation, NamedTensor, NamedTensorLocation>;
  using ConstIterator =
      ConstMapIterator<TensorLocation, NamedTensor, NamedTensorLocation>;
  Iterator begin() { return Iterator(_map.begin()); }
  Iterator end() { return Iterator(_map.end()); }
  ConstIterator begin() const { return ConstIterator(_map.begin()); }
  ConstIterator end() const { return ConstIterator(_map.end()); }

  // Status AddOutputTensor( TensorLocation key, const std::string& tensor_name,
  // poplar::Tensor tensor);
  Status AddOutputTensor(const HloInstruction* inst, int64 output_index,
                         poplar::Tensor tensor);
  Status AddOutputRemoteBuffer(const HloInstruction* inst, int64 output_index,
                               poplar::RemoteBuffer rbuffer);
  Status UpdateTensor(TensorLocation key, poplar::Tensor tensor);
  poplar::Tensor GetTensor(TensorLocation key) const;
  poplar::Tensor FindTensorByName(const std::string& name,
                                  int64 output_index) const;
  void Clear();
  /* This returns a vector of poplar tensors or remote buffers which are all of
   * the outputs from the given instruction
   */
  TensorOrRemoteBufferVector FindInstructionOutputs(
      const HloInstruction* inst,
      absl::optional<int64> opt_tensors_start = absl::nullopt,
      absl::optional<int64> opt_tensors_end = absl::nullopt) const;
  StatusOr<TensorVector> FindInstructionOutputTensors(
      const HloInstruction* inst,
      absl::optional<int64> opt_tensors_start = absl::nullopt,
      absl::optional<int64> opt_tensors_end = absl::nullopt) const;
  using NamedTensorLocationVector = std::vector<NamedTensorLocation>;
  NamedTensorLocationVector FindInstructionNamedTensorLocations(
      const HloInstruction* inst,
      absl::optional<int64> opt_tensors_start = absl::nullopt,
      absl::optional<int64> opt_tensors_end = absl::nullopt) const;

 private:
  std::map<TensorLocation, NamedTensor> _map;
};

struct ComputationTensorMap {
  ComputationTensorMap(const std::string& computation_name,
                       const TensorMap& map)
      : computation(computation_name), tensor_map(map) {}
  const std::string& computation;
  const TensorMap& tensor_map;
};

class TensorMaps {
 public:
  void AddTensorMapForComputation(const std::string& computation_name,
                                  TensorMap tensor_map);
  const TensorMap& GetTensorMapForComputation(
      const std::string& computation_name) const;
  using Iterator = MapIterator<std::string, TensorMap, ComputationTensorMap>;
  using ConstIterator =
      ConstMapIterator<std::string, TensorMap, ComputationTensorMap>;
  Iterator begin() { return Iterator(_map.begin()); }
  Iterator end() { return Iterator(_map.end()); }
  ConstIterator begin() const { return ConstIterator(_map.begin()); }
  ConstIterator end() const { return ConstIterator(_map.end()); }

 private:
  std::map<std::string, TensorMap> _map;
};
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_TENSOR_MAP_H_
