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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXTENDED_GRAPH_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXTENDED_GRAPH_H_

#include <utility>
#include <vector>

#include <poplar/Graph.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/extended_tensor.h"

namespace xla {
namespace poplarplugin {

// Wrapper class to abstract migration from poplar to snap
class ExtendedGraph {
 public:
  ExtendedGraph(const poplar::Target& target,
                poplar::replication_factor replication_factor);
  ExtendedGraph(poplar::Graph&& graph);  // NOLINT

  operator poplar::Graph&() { return getPoplarGraph(); }
  operator const poplar::Graph&() const { return getPoplarGraph(); }

  const poplar::Target& getTarget() const { return graph_.getTarget(); }

  ExtendedGraph createVirtualGraph(unsigned lowerTile, unsigned upperTile);

  ExtendedGraph createVirtualGraph(const std::vector<unsigned>& perIpuTiles);

  ExtendedTensor clone(const poplar::Tensor& t);

  void addPoplibsCodelets() {
    poplin::addCodelets(graph_);
    popnn::addCodelets(graph_);
    popops::addCodelets(graph_);
    poprand::addCodelets(graph_);
  }

  ExtendedTensor addVariable(const poplar::Type& type,
                             poplar::ArrayRef<std::size_t> shape,
                             const poplar::DebugContext& debugContext = {});

  ExtendedTensor addVariable(const poplar::Type& type,
                             poplar::ArrayRef<std::size_t> shape,
                             poplar::VariableMappingMethod mappingMethod,
                             const poplar::DebugContext& debugContext = {});

  template <typename T>
  ExtendedTensor addConstant(const poplar::Type& type,
                             poplar::ArrayRef<std::size_t> shape,
                             poplar::ArrayRef<T> values,
                             const poplar::DebugContext& debugContext = {
                                 "<const>"}) {  // NOLINT
    auto tensor =
        getPoplarGraph().addConstant(type, shape, values, debugContext);
    return {std::move(tensor), *this};
  }

  template <typename T>
  ExtendedTensor addConstant(const poplar::Type& type,
                             poplar::ArrayRef<std::size_t> shape, T val,
                             const poplar::DebugContext& debugContext = {
                                 "<const>"}) {  // NOLINT
    auto tensor = getPoplarGraph().addConstant(type, shape, val, debugContext);
    return {std::move(tensor), *this};
  }

  poplar::Graph::TileToTensorMapping getTileMapping(poplar::Tensor t);

  poplar::Function addFunction(const poplar::program::Program& program);

  void createHostWrite(poplar::StringRef handle, const ExtendedTensor& t,
                       bool rearrangeOnHost = false);

  void createHostRead(poplar::StringRef handle, const ExtendedTensor& t,
                      bool rearrangeOnHost = false);

  template <typename T>
  void setInitialValue(const ExtendedTensor& t, poplar::ArrayRef<T> values) {
    getPoplarGraph().setInitialValue<T>(t, values);
  }

  void setInitialValueHalf(const poplar::Tensor& t,
                           poplar::ArrayRef<uint16_t> values);

  void addCodelets(
      std::stringstream& stream, poplar::StringRef compileFlags,
      std::ostream& compileOutput,
      poplar::CodeletFileType type = poplar::CodeletFileType::CppSource,
      poplar::StringRef targetName = "");

  void setTileMapping(const ExtendedTensor& t, unsigned tileNum);

  poplar::DataStream addHostToDeviceFIFO(
      poplar::StringRef handle, const poplar::Type& elementType,
      std::size_t numElements,
      poplar::ReplicatedStreamMode replicatedMode =
          poplar::ReplicatedStreamMode::REPLICATE,
      const poplar::OptionFlags& options = {}) {
    return graph_.addHostToDeviceFIFO(handle, elementType, numElements,
                                      replicatedMode, options);
  }

  poplar::DataStream addDeviceToHostFIFO(
      poplar::StringRef handle, const poplar::Type& elementType,
      std::size_t numElements, const poplar::OptionFlags& options = {}) {
    return graph_.addDeviceToHostFIFO(handle, elementType, numElements,
                                      options);
  }

  poplar::Graph& getPoplarGraph() { return graph_; }
  const poplar::Graph& getPoplarGraph() const { return graph_; }

 private:
  poplar::Graph graph_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXTENDED_GRAPH_H_
