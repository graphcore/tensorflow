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

#include <poplar/DataStream.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

namespace xla {
namespace poplarplugin {

class ExtendedGraph;

// Wrapper class for poplar (will be removed in T67791)
class ExtendedTensor : public poplar::Tensor {
 public:
  using poplar::Tensor::Tensor;
  ExtendedTensor(poplar::Tensor&& tensor)  // NOLINT
      : poplar::Tensor(std::move(tensor)) {}
  ExtendedTensor(const poplar::Tensor& tensor)  // NOLINT
      : poplar::Tensor(tensor) {}

  ExtendedTensor(const poplar::Tensor& tensor,
                 const ExtendedGraph& graph)  // NOLINT
      : poplar::Tensor(tensor) {}

  ExtendedTensor(poplar::Graph& graph)  // NOLINT
      : poplar::Tensor() {}

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

  ExtendedTensor dimShuffle(poplar::ArrayRef<unsigned> permutation) const;

  ExtendedTensor dimShufflePartial(
      poplar::ArrayRef<unsigned> source,
      poplar::ArrayRef<unsigned> destination) const;

  ExtendedTensor dimRoll(unsigned dimIdx, unsigned newIdx = 0) const;

  ExtendedTensor reinterpret(const poplar::Type& type) const;

  bool operator!=(const Tensor& o) const { return !(*this == o); }

  poplar::Tensor& getPoplarTensor() { return *this; }
  const poplar::Tensor& getPoplarTensor() const { return *this; }
};

std::ostream& operator<<(std::ostream& os, const ExtendedTensor& tensor);

std::vector<poplar::Tensor> GetPoplarTensors(
    std::vector<ExtendedTensor>& tensors);

using ExtendedDataStream = poplar::DataStream;
using ExtendedRemoteBuffer = poplar::RemoteBuffer;

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXTENDED_TENSOR_H_
