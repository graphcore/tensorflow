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

#include "absl/types/any.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "tensorflow/compiler/plugin/poplar/driver/driver_types.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/tensor_location.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
class HloInstruction;

namespace poplarplugin {

/**
 * A drop-in replacement for poplar RemoteBuffer structure.
 *
 * RemoteBufferHolder holds all information required for RemoteBuffer creation,
 * and creates it on-demand. This allows deferring of remote buffer creation and
 * optionally changing parameters before it's created.
 */

class RemoteBufferHolder {
 public:
  explicit RemoteBufferHolder(const DriverRemoteBuffer& buffer)
      : graph_(nullptr),
        handle_(buffer.handle()),
        element_type_(buffer.elementType()),
        num_elements_(buffer.numElements()),
        repeats_(buffer.getRepeats()),
        rearrange_on_host_(buffer.isRearrangeOnHost()),
        optimise_memory_(buffer.isOptimisedForMemory()),
        remote_buffer_(buffer) {}

  RemoteBufferHolder(DriverGraph& graph, const std::string& handle,
                     const poplar::Type& element_type, std::size_t num_elements,
                     std::size_t repeats = 1, bool rearrange_on_host = false,
                     bool optimise_memory = false)
      : graph_(&graph),
        handle_(handle),
        element_type_(element_type),
        num_elements_(num_elements),
        repeats_(repeats),
        rearrange_on_host_(rearrange_on_host),
        optimise_memory_(optimise_memory) {}

  /**
   * Creates poplar RemoteBuffer or returns it if it's already created.
   */
  DriverRemoteBuffer Get();
  Status SetNumElements(std::size_t num_elements);

  const std::string& GetHandle() const { return handle_; }
  poplar::Type GetElementType() const { return element_type_; }
  std::size_t GetNumElements() const { return num_elements_; }
  std::size_t GetRepeats() const { return repeats_; }

  bool operator==(const RemoteBufferHolder& other) const {
    return handle_ == other.handle_;
  }

 private:
  DriverGraph* graph_;
  std::string handle_;
  poplar::Type element_type_;
  std::size_t num_elements_;
  std::size_t repeats_;
  bool rearrange_on_host_;
  bool optimise_memory_;

  absl::optional<DriverRemoteBuffer> remote_buffer_;

  TF_DISALLOW_COPY_AND_ASSIGN(RemoteBufferHolder);
};

/**
 * A struct that can hold either a poplar tensor, a poplar remote buffer, or an
 * opaque absl::any.
 *
 * The operator overloads are design so that it implicitely casts to the
 * appropriate type, when the context is unambiguous. This should minimise the
 * need for code changes where poplar tensors are assumed.
 */
struct TensorOrRemoteBuffer {
  /**
   * Default construct the element as an empty tensor.
   *
   * This preserves the existing behaviour where we would always default
   * construct the tensor, because it could only be a tensor.
   */
  TensorOrRemoteBuffer() = default;
  TensorOrRemoteBuffer(TensorOrRemoteBuffer&&) = default;

  /**
   * Construct with a poplar tensor.
   */
  explicit TensorOrRemoteBuffer(DriverTensor tensor)
      : tensor(tensor), content_type(ContentType::Tensor) {}

  /**
   * Construct with a poplar remote buffer.
   */
  explicit TensorOrRemoteBuffer(RemoteBufferHolder* rbuffer,
                                bool is_replica_partitioned, int64_t num_merged)
      : remote_buffer_holder(rbuffer),
        is_replica_partitioned(is_replica_partitioned),
        num_merged(num_merged),
        content_type(ContentType::RemoteBuffer) {}

  /**
   * Construct with an opaque absl::any.
   */
  explicit TensorOrRemoteBuffer(absl::any opaque)
      : opaque_(opaque), content_type(ContentType::Opaque) {}

  /**
   * Construct with a remote buffer, tensor, or opaque.
   */
  TensorOrRemoteBuffer(const TensorOrRemoteBuffer& rhs) { *this = rhs; }

  /**
   * Helper function to test whether a tensor is stored in the element.
   */
  bool IsTensor() const { return content_type == ContentType::Tensor; }

  /**
   * Helper function to test whether a remote buffer is stored in the element.
   */
  bool IsRemoteBuffer() const {
    return content_type == ContentType::RemoteBuffer;
  }

  bool IsReplicaPartitioned() const {
    return IsRemoteBuffer() && is_replica_partitioned;
  }

  int64_t NumMerged() const { return num_merged; }

  /**
   * Helper function to test whether an opaque absl::any is stored in the
   * element.
   */
  bool IsOpaque() const { return content_type == ContentType::Opaque; }

  /**
   * Helper function to force the cast to a poplar tensor when it is
   * unambiguous.
   */
  DriverTensor AsTensor() const {
    CHECK(content_type == ContentType::Tensor);
    return tensor;
  }

  /**
   * Helper function to force the cast to a poplar remote buffer when it is
   * unambiguous.
   */
  RemoteBufferHolder& AsRemoteBufferHolder() const {
    CHECK(content_type == ContentType::RemoteBuffer);
    CHECK_NOTNULL(remote_buffer_holder);
    return *remote_buffer_holder;
  }

  DriverRemoteBuffer AsRemoteBuffer() const {
    return AsRemoteBufferHolder().Get();
  }

  /**
   * Helper function to force the cast to an opaque absl::any when it is
   * unambiguous.
   */
  absl::any AsOpaque() const {
    CHECK(content_type == ContentType::Opaque);
    return opaque_;
  }

  /**
   * Operator overloads to support implicit casts.
   */
  operator DriverTensor() const { return AsTensor(); }
  operator DriverRemoteBuffer() const { return AsRemoteBuffer(); }

  TensorOrRemoteBuffer& operator=(const TensorOrRemoteBuffer& rhs) {
    switch (rhs.content_type) {
      case ContentType::Empty:
        content_type = ContentType::Empty;
        break;
      case ContentType::Tensor:
        content_type = ContentType::Tensor;
        tensor = rhs.tensor;
        break;
      case ContentType::RemoteBuffer:
        content_type = ContentType::RemoteBuffer;
        remote_buffer_holder = rhs.remote_buffer_holder;
        is_replica_partitioned = rhs.is_replica_partitioned;
        num_merged = rhs.num_merged;
        break;
      case ContentType::Opaque:
        content_type = ContentType::Opaque;
        opaque_ = rhs.opaque_;
        break;
    }

    return *this;
  }

  /**
   * Support assignment, like this is a poplar tensor.
   */
  TensorOrRemoteBuffer& operator=(DriverTensor t) {
    content_type = ContentType::Tensor;
    tensor = t;
    return *this;
  }

  /**
   * Support assignment, like this is an opaque absl::any.
   */
  TensorOrRemoteBuffer& operator=(absl::any opaque) {
    content_type = ContentType::Opaque;
    opaque_ = opaque;
    return *this;
  }

  /**
   * In a few places, tensor equality is checked. This operator overload allows
   * that code to continue working.
   */
  inline bool operator==(const TensorOrRemoteBuffer& rhs) const {
    if (IsRemoteBuffer() && rhs.IsRemoteBuffer()) {
      return AsRemoteBufferHolder() == rhs.AsRemoteBufferHolder();
    }

    if (IsTensor() && rhs.IsTensor()) {
      return AsTensor() == rhs.AsTensor();
    }

    if (IsOpaque() && rhs.IsOpaque()) {
      // We can only do address comparison of absl::any objects.
      return this == &rhs;
    }

    return false;
  }

  inline bool operator!=(const TensorOrRemoteBuffer& rhs) const {
    return !(*this == rhs);
  }

 private:
  /**
   * The inner storage is a variable of a tensor, a remote buffer, or neither.
   */
  DriverTensor tensor;
  RemoteBufferHolder* remote_buffer_holder = nullptr;
  absl::any opaque_;

  /**
   * Additional meta-data is stored here
   */
  bool is_replica_partitioned = false;
  int64_t num_merged = 1;

  enum class ContentType { Empty, Tensor, RemoteBuffer, Opaque };

  ContentType content_type = ContentType::Empty;
};

using TensorOrRemoteBufferVector = std::vector<TensorOrRemoteBuffer>;
using TensorOrRemoteBufferVectors = std::vector<TensorOrRemoteBufferVector>;

using TensorVector = std::vector<DriverTensor>;
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
  // DriverTensor tensor);
  Status AddOutputTensor(const HloInstruction* inst, int64_t output_index,
                         DriverTensor tensor);
  Status AddOutputOpaque(const HloInstruction* inst, int64_t output_index,
                         absl::any opaque);
  Status AddOutput(const HloInstruction* inst, int64_t output_index,
                   TensorOrRemoteBuffer torb);

  Status UpdateTensor(TensorLocation key, DriverTensor tensor);
  DriverTensor GetTensor(TensorLocation key) const;
  DriverTensor FindTensorByName(const std::string& name,
                                int64_t output_index) const;
  void Clear();
  /* This returns a vector of poplar tensors or remote buffers which are all of
   * the outputs from the given instruction
   */
  TensorOrRemoteBufferVector FindInstructionOutputs(
      const HloInstruction* inst,
      absl::optional<int64_t> opt_tensors_start = absl::nullopt,
      absl::optional<int64_t> opt_tensors_end = absl::nullopt) const;
  StatusOr<TensorVector> FindInstructionOutputTensors(
      const HloInstruction* inst,
      absl::optional<int64_t> opt_tensors_start = absl::nullopt,
      absl::optional<int64_t> opt_tensors_end = absl::nullopt) const;
  using NamedTensorLocationVector = std::vector<NamedTensorLocation>;
  NamedTensorLocationVector FindInstructionNamedTensorLocations(
      const HloInstruction* inst,
      absl::optional<int64_t> opt_tensors_start = absl::nullopt,
      absl::optional<int64_t> opt_tensors_end = absl::nullopt) const;

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
  std::size_t size() const { return _map.size(); }

 private:
  std::map<std::string, TensorMap> _map;
};
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_TENSOR_MAP_H_
