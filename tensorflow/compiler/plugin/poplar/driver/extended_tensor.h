/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXTENDED_TENSOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXTENDED_TENSOR_H_

#include <string>
#include <utility>
#include <vector>

#include <snap/DataStream.hpp>
#include <snap/RemoteBuffer.hpp>
#include <snap/Tensor.hpp>

namespace xla {
namespace poplarplugin {

// Wrapper class to abstract migration from poplar to snap
class ExtendedTensor : public snap::Tensor {
 public:
  using snap::Tensor::Tensor;
  ExtendedTensor(snap::Tensor&& tensor)  // NOLINT
      : snap::Tensor(std::move(tensor)) {}

  ExtendedTensor(snap::Graph& graph)  // NOLINT
      : snap::Tensor(poplar::Tensor(), graph) {}

  ExtendedTensor reshape(poplar::ArrayRef<std::size_t> shape) const;

  ExtendedTensor reshapePartial(unsigned beginIndex, unsigned endIndex,
                                poplar::ArrayRef<std::size_t> newDims) const;

  ExtendedTensor flatten() const;

  ExtendedTensor flatten(unsigned dimBegin, unsigned dimEnd) const;

  ExtendedTensor slice(const poplar::Interval& region,
                       unsigned dimension = 0) const;

  ExtendedTensor slice(std::size_t begin, std::size_t end,
                       unsigned dimension = 0) const&;

  std::vector<ExtendedTensor> slices(
      poplar::ArrayRef<poplar::Interval> intervals,
      unsigned dimension = 0) const;

  std::vector<ExtendedTensor> slices(
      const std::vector<std::vector<poplar::Interval>>& intervals,
      unsigned dimension = 0) const;

  std::vector<ExtendedTensor> slices(const poplar::ArrayRef<unsigned>& indices,
                                     unsigned dimension = 0) const;

  bool containsAliases() const { return getPoplarTensor().containsAliases(); }

  bool containsConstant() const { return getPoplarTensor().containsConstant(); }

  ExtendedTensor dimShuffle(poplar::ArrayRef<unsigned> permutation) const;

  ExtendedTensor dimShufflePartial(
      poplar::ArrayRef<unsigned> source,
      poplar::ArrayRef<unsigned> destination) const;

  ExtendedTensor dimRoll(unsigned dimIdx, unsigned newIdx = 0) const;

  ExtendedTensor reinterpret(const poplar::Type& type) const;

  const std::vector<poplar::Interval> getContiguousRegions() const {
    return getPoplarTensor().getContiguousRegions();
  }

  std::string shapeToString() const {
    return getPoplarTensor().shapeToString();
  }

  bool intersectsWith(const poplar::Tensor& other) const {
    return getPoplarTensor().intersectsWith(other);
  }

  bool operator!=(const Tensor& o) const { return !(*this == o); }

  operator poplar::Tensor&() { return getPoplarTensor(); }
  operator const poplar::Tensor&() const { return getPoplarTensor(); }
};

std::ostream& operator<<(std::ostream& os, const ExtendedTensor& tensor);

std::vector<snap::Tensor> GetSnapTensors(std::vector<ExtendedTensor>& tensors);

// Wrapper class to abstract migration from poplar to snap
class ExtendedDataStream : public snap::DataStream {
 public:
  ExtendedDataStream() = default;
  ExtendedDataStream(snap::DataStream&& data_stream)  // NOLINT
      : snap::DataStream(std::move(data_stream)) {}

  operator poplar::DataStream&() { return getPoplarDataStream(); }
  operator const poplar::DataStream&() const { return getPoplarDataStream(); }
};

// Wrapper class to abstract migration from poplar to snap
class ExtendedRemoteBuffer : public snap::RemoteBuffer {
 public:
  ExtendedRemoteBuffer() = default;
  ExtendedRemoteBuffer(snap::RemoteBuffer&& remote_buffer)  // NOLINT
      : snap::RemoteBuffer(std::move(remote_buffer)) {}

  operator poplar::RemoteBuffer&() { return getPoplarRemoteBuffer(); }
  operator const poplar::RemoteBuffer&() const {
    return getPoplarRemoteBuffer();
  }
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXTENDED_TENSOR_H_
