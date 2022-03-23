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

#include <poplar/Graph.hpp>
#include <popops/Cast.hpp>
#include <popops/ScaledAdd.hpp>
#include <poputil/exceptions.hpp>

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
  input_to_output_tensor_aliasing = {
      {/*input tensor index*/ 0, /*output tensor index=*/0}};
  is_elementwise = true;
}

extern "C" poplar::program::Program Build(poplar::Graph& graph,
                                          std::vector<poplar::Tensor>& inputs,
                                          std::vector<poplar::Tensor>& outputs,
                                          const std::string& attributes,
                                          const std::string& debug_prefix) {
  if (inputs.size() != 2) {
    throw poputil::poplibs_error("add requires 2 inputs.");
  }

  auto left = inputs[0];
  auto right = inputs[1];

  if (left.shape() != right.shape()) {
    throw poputil::poplibs_error("Inputs must have identical shapes.");
  }

  poplar::program::Sequence prog;
  popops::scaledAddTo(graph, left, right, 1.0, prog,
                      debug_prefix + "/custom_add_inplace");
  outputs.push_back(left);
  return prog;
}
