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

#include <popfloat/experimental/codelets.hpp>
#include <poplar/Graph.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>
#include <popsparse/codelets.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/extended_tensor.h"

namespace xla {
namespace poplarplugin {

using ExtendedFunction = poplar::Function;

// Wrapper class for poplar (will be removed in T67791)
class ExtendedGraph : public poplar::Graph {
 public:
  ExtendedGraph(const poplar::Target& target,
                poplar::replication_factor replication_factor);
  ExtendedGraph(poplar::Graph&& graph);  // NOLINT

  ExtendedGraph createVirtualGraph(unsigned lowerTile, unsigned upperTile);

  ExtendedGraph createVirtualGraph(const std::vector<unsigned>& perIpuTiles);

  ExtendedGraph getTopLevelGraph();

  ExtendedTensor addReplicationIndexConstant(
      const poplar::DebugContext& debugContext = {});

  ExtendedTensor clone(
      const poplar::Type& type, const poplar::Tensor& t,
      const poplar::DebugContext& debugContext = {},
      poplar::TensorCloneMethod method =
          poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);
  ExtendedTensor clone(
      const poplar::Tensor& t, const poplar::DebugContext& debugContext = {},
      poplar::TensorCloneMethod method =
          poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);

  void addPoplibsCodelets() {
    poplin::addCodelets(*this);
    popnn::addCodelets(*this);
    popops::addCodelets(*this);
    poprand::addCodelets(*this);
    popsparse::addCodelets(*this);
    popfloat::experimental::addCodelets(*this);
  }

  ExtendedTensor addVariable(const poplar::Type& type,
                             poplar::ArrayRef<std::size_t> shape,
                             const poplar::DebugContext& debugContext = {});

  ExtendedTensor addVariable(const poplar::Type& type,
                             poplar::ArrayRef<std::size_t> shape,
                             poplar::VariableMappingMethod mappingMethod,
                             const poplar::DebugContext& debugContext = {});

  ExtendedTensor addLinearlyMappedVariable(
      const poplar::Type& type, poplar::ArrayRef<std::size_t> shape,
      const poplar::DebugContext& debugContext = {});

  ExtendedTensor addLinearlyMappedVariable(
      const poplar::Type& type, poplar::ArrayRef<std::size_t> shape,
      unsigned minElementsPerTile, unsigned grainSize,
      const poplar::DebugContext& debugContext = {});

  template <typename T>
  ExtendedTensor addConstant(const poplar::Type& type,
                             poplar::ArrayRef<std::size_t> shape,
                             poplar::ArrayRef<T> values,
                             const poplar::DebugContext& debugContext = {
                                 "<const>"}) {
    auto tensor = poplar::Graph::addConstant(type, shape, values, debugContext);
    return {std::move(tensor)};
  }

  template <typename T>
  ExtendedTensor addConstant(
      const poplar::Type& type, poplar::ArrayRef<std::size_t> shape,
      const T* values, const poplar::DebugContext& debugContext = {"<const>"}) {
    auto tensor = poplar::Graph::addConstant(type, shape, values, debugContext);
    return {std::move(tensor), *this};
  }

  template <typename T>
  ExtendedTensor addConstant(const poplar::Type& type,
                             poplar::ArrayRef<std::size_t> shape, T val,
                             const poplar::DebugContext& debugContext = {
                                 "<const>"}) {
    auto tensor = poplar::Graph::addConstant(type, shape, val, debugContext);
    return {tensor, *this};
  }

  ExtendedTensor addConstantHalf(
      const poplar::Type& type, poplar::ArrayRef<std::size_t> shape,
      uint16_t val, const poplar::DebugContext& debugContext = {"<const>"});

  ExtendedTensor addConstantHalf(const poplar::Type& type,
                                 poplar::ArrayRef<std::size_t> shape,
                                 const uint16_t* val,
                                 const poplar::DebugContext& debugContext = {
                                     "<const>"});

  void setTileMapping(poplar::VertexRef v, unsigned tileNum);

  void setTileMapping(const ExtendedTensor& t, unsigned tileNum);

  void setTileMapping(const poplar::Tensor& t, unsigned tileNum);

  void setTileMapping(const ExtendedTensor& t,
                      const poplar::Graph::TileToTensorMapping& mapping);

  poplar::Graph::TileToTensorMapping getTileMapping(ExtendedTensor t) const;
  poplar::Graph::TileToTensorMapping getTileMapping(poplar::Tensor t) const;

  std::vector<std::vector<poplar::Interval>> getSortedContiguousRegions(
      const ExtendedTensor& t, poplar::ArrayRef<poplar::Interval> regions,
      bool removeAliasedIntervals = false,
      std::vector<std::size_t>* aliases = nullptr) const {
    return poplar::Graph::getSortedContiguousRegions(
        t, regions, removeAliasedIntervals, aliases);
  }

  poplar::HostFunction addHostFunction(
      poplar::StringRef handle,
      poplar::ArrayRef<poplar::Graph::HostFunctionArgument> inputs,
      poplar::ArrayRef<poplar::Graph::HostFunctionArgument> outputs);

  ExtendedDataStream addHostToDeviceFIFO(
      poplar::StringRef handle, const poplar::Type& elementType,
      std::size_t numElements,
      poplar::ReplicatedStreamMode replicatedMode =
          poplar::ReplicatedStreamMode::REPLICATE,
      const poplar::OptionFlags& options = {});

  ExtendedDataStream addDeviceToHostFIFO(
      poplar::StringRef handle, const poplar::Type& elementType,
      std::size_t numElements, const poplar::OptionFlags& options = {});

  void createHostWrite(poplar::StringRef handle, const ExtendedTensor& t,
                       bool rearrangeOnHost = false);

  void createHostRead(poplar::StringRef handle, const ExtendedTensor& t,
                      bool rearrangeOnHost = false);

  template <typename T>
  void setInitialValue(const ExtendedTensor& t, poplar::ArrayRef<T> values) {
    poplar::Graph::setInitialValue<T>(t, values);
  }

  template <typename T>
  void setInitialValue(const poplar::FieldRef field, T value) {
    poplar::Graph::setInitialValue<T>(field, value);
  }

  void setInitialValueHalf(const poplar::Tensor& t,
                           poplar::ArrayRef<uint16_t> values);

  bool addCodelets(poplar::StringRef src,
                   poplar::CodeletFileType type = poplar::CodeletFileType::Auto,
                   poplar::StringRef compileFlags = "",
                   poplar::StringRef targetName = "");

  void addCodelets(
      std::stringstream& stream, poplar::StringRef compileFlags = "",
      poplar::CodeletFileType type = poplar::CodeletFileType::CppSource,
      poplar::StringRef targetName = "");

  void addCodelets(
      std::stringstream& stream, poplar::StringRef compileFlags,
      std::ostream& compileOutput,
      poplar::CodeletFileType type = poplar::CodeletFileType::CppSource,
      poplar::StringRef targetName = "");

  poplar::ComputeSet addComputeSet(
      const poplar::DebugContext& debugContext = {}) {
    return poplar::Graph::addComputeSet(debugContext);
  }

  poplar::VertexRef addVertex(poplar::ComputeSet cs,
                              poplar::StringRef vertexType);

  poplar::VertexRef addVertex(
      poplar::ComputeSet cs, poplar::StringRef vertexType,
      poplar::ArrayRef<poplar::Graph::ConnectionDesc> connections);

  void connect(poplar::FieldRef field, const poplar::Tensor& tensor);

  std::vector<std::vector<poplar::Interval>> getSortedContiguousRegions(
      const poplar::Tensor& t, poplar::ArrayRef<poplar::Interval> regions,
      bool removeAliasedIntervals = false,
      std::vector<std::size_t>* aliases = nullptr) const;

  void setPerfEstimate(const poplar::VertexRef& v, std::uint64_t cycles,
                       std::uint64_t flops = 0);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXTENDED_GRAPH_H_
