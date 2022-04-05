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

#include <utility>

#include <snap/DataStream.hpp>
#include <snap/Tensor.hpp>

namespace xla {
namespace poplarplugin {

// Wrapper class to abstract migration from poplar to snap
class ExtendedTensor : public snap::Tensor {
 public:
  ExtendedTensor() = default;
  ExtendedTensor(snap::Tensor&& tensor)  // NOLINT
      : snap::Tensor(std::move(tensor)) {}
  ExtendedTensor(poplar::Tensor tensor, snap::Graph& graph)
      : snap::Tensor(tensor, graph) {}

  operator poplar::Tensor&() { return getPoplarTensor(); }
  operator const poplar::Tensor&() const { return getPoplarTensor(); }
};

// Wrapper class to abstract migration from poplar to snap
class ExtendedDataStream : public snap::DataStream {
 public:
  ExtendedDataStream() = default;
  ExtendedDataStream(snap::DataStream&& data_stream)  // NOLINT
      : snap::DataStream(std::move(data_stream)) {}

  operator poplar::DataStream&() { return getPoplarDataStream(); }
  operator const poplar::DataStream&() const { return getPoplarDataStream(); }
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXTENDED_TENSOR_H_
