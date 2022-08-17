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

#include <iostream>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/plugin/poplar/driver/extended_tensor.h"

namespace xla {
namespace poplarplugin {

ExtendedTensor ExtendedTensor::reshape(
    poplar::ArrayRef<std::size_t> shape) const {
  auto t = poplar::Tensor::reshape(shape);
  return {std::move(t)};
}

ExtendedTensor ExtendedTensor::reshapePartial(
    unsigned beginIndex, unsigned endIndex,
    poplar::ArrayRef<std::size_t> newDims) const {
  auto t = poplar::Tensor::reshapePartial(beginIndex, endIndex, newDims);
  return {std::move(t)};
}

ExtendedTensor ExtendedTensor::flatten() const {
  auto t = poplar::Tensor::flatten();
  return {std::move(t)};
}

ExtendedTensor ExtendedTensor::flatten(unsigned dimBegin,
                                       unsigned dimEnd) const {
  auto t = poplar::Tensor::flatten(dimBegin, dimEnd);
  return {std::move(t)};
}

ExtendedTensor ExtendedTensor::slice(const poplar::Interval& region,
                                     unsigned dimension) const {
  auto t = poplar::Tensor::slice(region, dimension);
  return {std::move(t)};
}

ExtendedTensor ExtendedTensor::slice(std::size_t begin, std::size_t end,
                                     unsigned dimension) const& {
  auto t = poplar::Tensor::slice(begin, end, dimension);
  return {std::move(t)};
}

std::vector<ExtendedTensor> ExtendedTensor::slices(
    poplar::ArrayRef<poplar::Interval> intervals, unsigned dimension) const {
  auto ts = poplar::Tensor::slices(intervals, dimension);
  std::vector<ExtendedTensor> ets(ts.size());
  absl::c_transform(ts, ets.begin(), [this](const poplar::Tensor& t) {
    return ExtendedTensor(t);
  });
  return ets;
}

std::vector<ExtendedTensor> ExtendedTensor::slices(
    const std::vector<std::vector<poplar::Interval>>& intervals,
    unsigned dimension) const {
  auto ts = poplar::Tensor::slices(intervals, dimension);
  std::vector<ExtendedTensor> ets(ts.size());
  absl::c_transform(ts, ets.begin(), [this](const poplar::Tensor& t) {
    return ExtendedTensor(t);
  });
  return ets;
}

std::vector<ExtendedTensor> ExtendedTensor::slices(
    const poplar::ArrayRef<unsigned>& indices, unsigned dimension) const {
  auto ts = poplar::Tensor::slices(indices, dimension);
  std::vector<ExtendedTensor> ets(ts.size());
  absl::c_transform(ts, ets.begin(), [this](const poplar::Tensor& t) {
    return ExtendedTensor(t);
  });
  return ets;
}

ExtendedTensor ExtendedTensor::dimShuffle(
    poplar::ArrayRef<unsigned> permutation) const {
  auto t = poplar::Tensor::dimShuffle(permutation);
  return {std::move(t)};
}

ExtendedTensor ExtendedTensor::dimShufflePartial(
    poplar::ArrayRef<unsigned> source,
    poplar::ArrayRef<unsigned> destination) const {
  auto t = poplar::Tensor::dimShufflePartial(source, destination);
  return {std::move(t)};
}

ExtendedTensor ExtendedTensor::dimRoll(unsigned dimIdx, unsigned newIdx) const {
  auto t = poplar::Tensor::dimRoll(dimIdx, newIdx);
  return {std::move(t)};
}

ExtendedTensor ExtendedTensor::reinterpret(const poplar::Type& type) const {
  auto t = poplar::Tensor::reinterpret(type);
  return {std::move(t)};
}

std::ostream& operator<<(std::ostream& os, const ExtendedTensor& tensor) {
  os << tensor;
  return os;
}

std::vector<poplar::Tensor> GetPoplarTensors(std::vector<ExtendedTensor>& ets) {
  std::vector<poplar::Tensor> ts(ets.size());
  absl::c_copy(ets, ts.begin());
  return ts;
}

}  // namespace poplarplugin
}  // namespace xla
