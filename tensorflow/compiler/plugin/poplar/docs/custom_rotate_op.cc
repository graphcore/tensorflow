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

#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

// Export the API level symbol
extern "C" {
int32_t custom_op_api_level = 5;
}

extern "C" void Build_metadata(
    std::vector<std::int64_t>& allocating_indices,
    std::vector<std::int64_t>& replica_identical_output_indices,
    std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
    bool& is_elementwise, bool& is_stateless, bool& is_hashable,
    std::uint32_t num_inputs) {
  allocating_indices.clear();
  is_elementwise = true;
}

extern "C" poplar::program::Program Build(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes,
    const std::string& debugPrefix) {
  if (inputs.size() != 3) {
    throw poputil::poplibs_error("Rotate requires 3 inputs");
  }

  if (inputs[0].numElements() == 0) {
    return poplar::program::Sequence();
  }

  if (inputs[0].rank() != 1 || inputs[1].rank() != 1 || inputs[2].rank() != 1) {
    throw poputil::poplibs_error("All inputs must be rank 1");
  }

  if (inputs[0].dim(0) != inputs[1].dim(0) ||
      inputs[0].dim(0) != inputs[2].dim(0)) {
    throw poputil::poplibs_error(
        "Length of rotate vector and data vectors must match");
  }

  if (inputs[0].elementType() != inputs[1].elementType() ||
      inputs[0].elementType() != inputs[2].elementType()) {
    throw poputil::poplibs_error(
        "Data types of angle vector and data vectors must match");
  }

  auto dType = inputs[0].elementType();

  /*
   * Create a ComputeSet which will be executed, and contains the vertices
   */
  auto cs = graph.addComputeSet(debugPrefix + "/rotate");

  /*
   * Get the tile mapping for the complete tensor.  We will map the vertices so
   * that they match the layout of the 'x' input tensor (input[0]).  If the 'x'
   * tensor was layed out differently to the other ones, then Poplar will
   * insert code to move the data in the other tensors to the mapped tile. So
   * ideally we would choose the best mapping for the vertices by analysing
   * all of the tensor mappings.
   */
  auto tileMapping = graph.getTileMapping(inputs[0]);

  /*
   * Get the target, which descibes properties of the hardware.
   */
  auto target = graph.getTarget();

  /*
   * Get the vector width of the particular data type, so that later we can
   * divide the tensor up between workers in an appropriate way.
   */
  const auto vectorWidth = target.getVectorWidth(dType);

  /*
   * Create the output tensors
   */
  outputs.push_back(graph.clone(inputs[0]));
  outputs.push_back(graph.clone(inputs[1]));

  auto xFlat = inputs[0].flatten();
  auto yFlat = inputs[1].flatten();
  auto aFlat = inputs[2].flatten();
  auto xOutputFlat = outputs[0].flatten();
  auto yOutputFlat = outputs[1].flatten();

  for (unsigned tile = 0; tile != tileMapping.size(); ++tile) {
    /*
     * If a tile contains no elements of the tensor then do not create any
     * vertices for it.
     */
    if (tileMapping[tile].empty()) {
      continue;
    }

    /*
     * Split up the regions of the inputs tensors so that they are evenly
     * distributed between the workers on the tile.
     */
    auto vertexRegions = poputil::splitRegionsBetweenWorkers(
        target, tileMapping[tile], vectorWidth, 2 * vectorWidth);

    for (const auto& regions : vertexRegions) {
      /*
       * If a region has no elements, then there is no need to add a vertex for
       * it.
       */
      if (regions.empty()) {
        continue;
      }

      /*
       * Add codelets to tiles which work over the regions in the input
       * tensors.
       */
      auto v = graph.addVertex(cs, poputil::templateVertex("Rotate", dType),
                               {{"x_out", xOutputFlat.slices(regions)},
                                {"y_out", yOutputFlat.slices(regions)},
                                {"x_in", xFlat.slices(regions)},
                                {"y_in", yFlat.slices(regions)},
                                {"angle", aFlat.slices(regions)}});

      /* Map the vertex onto the appropriate tile. */
      graph.setTileMapping(v, tile);

      /* Provide a bogus cycle count estimate for the profiler. */
      graph.setPerfEstimate(v, 1);
    }
  }

  return poplar::program::Execute(cs);
}
