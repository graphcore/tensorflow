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

#include <iostream>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

extern "C" {
int32_t custom_op_api_level = 5;
}

// If an operation takes one or more tensors of the same shape,
// and performs an expression on only corresponding elements in
// the input tensors, and produces a tensor of the same shape,
// then it is elementwise.
extern "C" void Build_metadata(
    std::vector<std::int64_t>& allocating_indices,
    std::vector<std::int64_t>& replica_identical_output_indices,
    std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
    bool& is_elementwise, bool& is_stateless, bool& is_hashable,
    std::uint32_t num_inputs) {
  is_elementwise = true;
}

// The Build function constructs the Poplar graph that computes the custom op.
extern "C" poplar::program::Program Build(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes,
    const std::string& debugPrefix) {
  if (inputs.size() != 5) {
    throw poputil::poplibs_error("ScaledVectorAdd requires 5 inputs");
  }

  if (inputs[0].numElements() == 0) {
    return poplar::program::Sequence();
  }

  if (inputs[0].rank() != 1 || inputs[2].rank() != 1) {
    throw poputil::poplibs_error("All inputs must be vectors");
  }

  if (inputs[1].rank() != 0) {
    throw poputil::poplibs_error("scale must be a scalar");
  }

  if (inputs[0].dim(0) != inputs[2].dim(0)) {
    throw poputil::poplibs_error("Length of input vectors must match");
  }

  if (inputs[0].elementType() != inputs[2].elementType()) {
    throw poputil::poplibs_error("Data types of inputs must match");
  }

  auto dType = inputs[0].elementType();

  // Create a ComputeSet which will be executed, and contains the vertices
  auto cs = graph.addComputeSet(debugPrefix + "/ScaledVectorAdd");

  // Get the tile mapping for the complete tensor.  We will map the vertices so
  // that they match the layout of the 'x' input tensor (input[0]).  If the 'x'
  // tensor was layed out differently to the other ones, then Poplar will
  // insert code to move the data in the other tensors to the mapped tile. So
  // ideally we would choose the best mapping for the vertices by analysing
  // all of the tensor mappings.
  auto tileMapping = graph.getTileMapping(inputs[0]);

  // Get the target, which descibes properties of the hardware.
  auto target = graph.getTarget();

  // Get the vector width of the particular data type, so that later we can
  // divide the tensor up between workers in an appropriate way.
  const auto vectorWidth = target.getVectorWidth(dType);

  // Create the output tensors
  outputs.push_back(graph.clone(inputs[0]));

  auto xFlat = inputs[0].flatten();
  auto yFlat = inputs[2].flatten();
  auto xOutputFlat = outputs[0].flatten();

  for (unsigned tile = 0; tile != tileMapping.size(); ++tile) {
    // If a tile contains no elements of the tensor then do not create any
    // vertices for it.
    if (tileMapping[tile].empty()) {
      continue;
    }

    // Split up the regions of the inputs tensors so that they are evenly
    // distributed between the workers on the tile.
    auto vertexRegions = poputil::splitRegionsBetweenWorkers(
        target, tileMapping[tile], vectorWidth, 2 * vectorWidth);

    for (const auto& regions : vertexRegions) {
      // If a region has no elements, then there is no need to add a vertex for
      // it.
      if (regions.empty()) {
        continue;
      }

      // Add codelets to tiles which work over the regions in the input
      // tensors.
      auto v =
          graph.addVertex(cs, poputil::templateVertex("ScaledVectorAdd", dType),
                          {{"z", xOutputFlat.slices(regions)},
                           {"x", xFlat.slices(regions)},
                           {"y", yFlat.slices(regions)},
                           {"scale", inputs[1].reshape({1})}});

      // Map the vertex onto the appropriate tile.
      graph.setTileMapping(v, tile);

      // Provide a bogus cycle count estimate for the profiler.
      graph.setPerfEstimate(v, 1);
    }
  }

  return poplar::program::Execute(cs);
}

extern "C" poplar::program::Program Build_grad(
    poplar::Graph& graph, int input_grad_index,
    const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& fwd_inputs,
    const std::vector<poplar::Tensor>& fwd_outputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes,
    const std::string& debug_prefix) {
  std::cerr << outputs.size() << "\n";
  outputs.resize(2);
  poplar::program::Sequence seq;
  // The forward op had inputs, x, scale y. Let's say their gradients are:
  // * grad of x -> scaled add (y, scale, grad)
  {
    outputs[0] = graph.clone(fwd_inputs[0]);
    std::vector<poplar::Tensor> ins = {fwd_inputs[2], fwd_inputs[1],
                                       gradients[0], fwd_inputs[3],
                                       fwd_inputs[4]};
    std::vector<poplar::Tensor> outs = {outputs[0]};
    seq.add(Build(graph, ins, outs, attributes, "gradX"));
  }
  // We don't expect scale gradient here

  // * grad of y -> scaled add (x, scale, grad)
  {
    outputs[1] = graph.clone(fwd_inputs[2]);
    std::vector<poplar::Tensor> ins = {fwd_inputs[0], fwd_inputs[1],
                                       gradients[0], fwd_inputs[3],
                                       fwd_inputs[4]};
    std::vector<poplar::Tensor> outs = {outputs[1]};
    seq.add(Build(graph, ins, outs, attributes, "gradY"));
  }

  return seq;
}
