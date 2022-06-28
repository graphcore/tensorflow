/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/mapping_helper.h"

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/TileMapping.hpp>

#include <algorithm>

namespace xla {
namespace poplarplugin {
namespace {
template <typename T>
void rotate_right(T& t, std::size_t spaces) {
  std::rotate(t.rbegin(), t.rbegin() + spaces, t.rend());
}

}  // namespace

void MappingHelper::RotateMapping(
    DriverGraph& graph, std::vector<std::vector<poplar::Interval>>& mapping,
    uint64 offset) {
  auto tile_count = graph.getPoplarGraph().getTarget().getNumTiles();

  // Move the tile mapping cyclically by the offset.
  mapping.resize(tile_count);
  offset %= tile_count;
  rotate_right(mapping, offset);
}

void MappingHelper::MapTensorLinearlyImpl(
    LinearMapperState& state, DriverGraph& graph, poplar::Tensor& tensor,
    std::vector<std::vector<poplar::Interval>>& mapping) {
  uint64& next_tile_to_map_from = state[&graph];

  // The number of tiles the mapping is across.
  auto mapping_tile_count = GetMappingWidth(mapping);
  auto tile_count = graph.getTarget().getNumTiles();
  if (mapping_tile_count == tile_count) {
    // Do not rotate tensors scattered across all tiles
    graph.getPoplarGraph().setTileMapping(tensor, mapping);
    return;
  }

  RotateMapping(graph, mapping, mapping_tile_count);
  graph.getPoplarGraph().setTileMapping(tensor, mapping);

  // Update offset.
  next_tile_to_map_from += mapping_tile_count;
  next_tile_to_map_from %= tile_count;
}

size_t MappingHelper::GetMappingWidth(
    const std::vector<std::vector<poplar::Interval>>& mapping) {
  auto non_empty = [](const std::vector<poplar::Interval>& intervals) {
    return !intervals.empty();
  };
  auto first = std::find_if(mapping.begin(), mapping.end(), non_empty);
  std::size_t first_idx = std::distance(mapping.begin(), first);
  auto last = std::find_if(mapping.rbegin(), mapping.rend(), non_empty);
  std::size_t last_idx = std::distance(last, mapping.rend());

  return last_idx - first_idx;
}

void MappingHelper::RemapTensor(LinearMapperState& state, DriverGraph& graph,
                                DriverTensor& tensor) {
  auto mapping = graph.getTileMapping(tensor);
  MapTensorLinearlyImpl(state, graph, tensor, mapping);
}

void MappingHelper::MapTensorLinearly(LinearMapperState& state,
                                      DriverGraph& graph,
                                      DriverTensor& tensor) {
  auto mapping = poputil::calcLinearTileMapping(graph, tensor);
  MapTensorLinearlyImpl(state, graph, tensor, mapping);
}

void MappingHelper::MapTensorLinearly(LinearMapperState& state,
                                      DriverGraph& graph, DriverTensor& tensor,
                                      uint32 min_elements_per_tile,
                                      uint32 grain_size) {
  auto mapping = poputil::calcLinearTileMapping(
      graph, tensor.shape(), min_elements_per_tile, grain_size);
  MapTensorLinearlyImpl(state, graph, tensor, mapping);
}

const uint64 MappingHelper::YieldNextTile(LinearMapperState& state,
                                          DriverGraph& graph) {
  uint64& next_tile = state[&graph];
  const uint64 tile = next_tile;
  next_tile += 1;
  next_tile = next_tile % graph.getTarget().getNumTiles();
  return tile;
}

}  // namespace poplarplugin
}  // namespace xla
