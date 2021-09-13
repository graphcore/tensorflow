/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>

extern "C" {
int32_t custom_op_api_level = 5;
}

namespace pe = popops::expr;

// Unhashable test op

extern "C" poplar::program::Program UnhashableTest(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes,
    const std::string& debugPrefix) {
  poplar::program::Sequence seq;

  outputs.resize(1);

  outputs[0] =
      popops::map(graph, pe::Add(pe::_1, pe::_2), {inputs[0], inputs[1]}, seq);
  return seq;
}

extern "C" void UnhashableTest_metadata(
    std::vector<std::int64_t>& allocating_indices,
    std::vector<std::int64_t>& replica_identical_output_indices,
    std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
    bool& is_elementwise, bool& is_stateless, bool& is_hashable,
    std::uint32_t num_inputs) {
  // default value must be unhashable
}

// Hashable test op

extern "C" poplar::program::Program HashableTest(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes,
    const std::string& debugPrefix) {
  poplar::program::Sequence seq;

  outputs.resize(1);

  outputs[0] =
      popops::map(graph, pe::Add(pe::_1, pe::_2), {inputs[0], inputs[1]}, seq);
  return seq;
}

extern "C" void HashableTest_metadata(
    std::vector<std::int64_t>& allocating_indices,
    std::vector<std::int64_t>& replica_identical_output_indices,
    std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
    bool& is_elementwise, bool& is_stateless, bool& is_hashable,
    std::uint32_t num_inputs) {
  is_hashable = true;
}
