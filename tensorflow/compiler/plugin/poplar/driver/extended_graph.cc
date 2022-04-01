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

#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/extended_graph.h"
#include "tensorflow/compiler/plugin/poplar/driver/extended_program.h"
#include "tensorflow/compiler/plugin/poplar/driver/extended_tensor.h"

namespace xla {
namespace poplarplugin {

ExtendedGraph::ExtendedGraph(const poplar::Target& target,
                             poplar::replication_factor replication_factor)
    : graph_(target, replication_factor) {}

ExtendedGraph::ExtendedGraph(poplar::Graph&& graph)
    : graph_(std::move(graph)) {}

ExtendedGraph ExtendedGraph::createVirtualGraph(unsigned lowerTile,
                                                unsigned upperTile) {
  auto graph = getPoplarGraph().createVirtualGraph(lowerTile, upperTile);
  return {std::move(graph)};
}

ExtendedGraph ExtendedGraph::createVirtualGraph(
    const std::vector<unsigned>& perIpuTiles) {
  auto graph = getPoplarGraph().createVirtualGraph(perIpuTiles);
  return {std::move(graph)};
}

ExtendedTensor ExtendedGraph::clone(const poplar::Tensor& t) {
  auto clone = getPoplarGraph().clone(t);
  return {std::move(clone), *this};
}

ExtendedTensor ExtendedGraph::addVariable(
    const poplar::Type& type, poplar::ArrayRef<std::size_t> shape,
    const poplar::DebugContext& debugContext) {
  auto tensor = getPoplarGraph().addVariable(type, shape, debugContext);
  return {std::move(tensor), *this};
}

ExtendedTensor ExtendedGraph::addVariable(
    const poplar::Type& type, poplar::ArrayRef<std::size_t> shape,
    poplar::VariableMappingMethod mappingMethod,
    const poplar::DebugContext& debugContext) {
  auto tensor =
      getPoplarGraph().addVariable(type, shape, mappingMethod, debugContext);
  return {std::move(tensor), *this};
}

poplar::Graph::TileToTensorMapping ExtendedGraph::getTileMapping(
    poplar::Tensor t) {
  return getPoplarGraph().getTileMapping(t);
}

poplar::Function ExtendedGraph::addFunction(
    const poplar::program::Program& program) {
  return getPoplarGraph().addFunction(program);
}

void ExtendedGraph::createHostWrite(poplar::StringRef handle,
                                    const ExtendedTensor& t,
                                    bool rearrangeOnHost) {
  getPoplarGraph().createHostWrite(handle, t, rearrangeOnHost);
}

void ExtendedGraph::createHostRead(poplar::StringRef handle,
                                   const ExtendedTensor& t,
                                   bool rearrangeOnHost) {
  getPoplarGraph().createHostRead(handle, t, rearrangeOnHost);
}

void ExtendedGraph::setInitialValueHalf(const poplar::Tensor& t,
                                        poplar::ArrayRef<uint16_t> values) {
  getPoplarGraph().setInitialValueHalf(t, values);
}

void ExtendedGraph::addCodelets(std::stringstream& stream,
                                poplar::StringRef compileFlags,
                                std::ostream& compileOutput,
                                poplar::CodeletFileType type,
                                poplar::StringRef targetName) {
  getPoplarGraph().addCodelets(stream, compileFlags, compileOutput, type,
                               targetName);
}

void ExtendedGraph::setTileMapping(const ExtendedTensor& t, unsigned tileNum) {
  getPoplarGraph().setTileMapping(t.getPoplarTensor(), tileNum);
}

}  // namespace poplarplugin
}  // namespace xla
