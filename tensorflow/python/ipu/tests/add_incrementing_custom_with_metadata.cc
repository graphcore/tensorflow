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

#include <algorithm>

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>

#include <unordered_map>
#include <unordered_set>

namespace pe = popops::expr;

// Custom poplar kernel.
extern "C" poplar::program::Program Build(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& debugPrefix) {
  poplar::program::Sequence seq;

  outputs.resize(inputs.size());

  float i = 1.0f;
  std::transform(inputs.begin(), inputs.end(), outputs.begin(),
                 [&](const poplar::Tensor& input) {
                   return popops::map(graph, pe::Add(pe::_1, pe::Const(i++)),
                                      {input}, seq);
                 });

  return seq;
}

// Custom poplar kernel.
extern "C" poplar::program::Program Build_grad(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& old_outputs,
    const std::vector<poplar::Tensor>& old_inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& debugPrefix) {
  poplar::program::Sequence seq;

  outputs.resize(gradients.size());

  float i = 1.0f;
  std::transform(
      gradients.begin(), gradients.end(), outputs.begin(),
      [&](const poplar::Tensor& input) {
        seq.add(poplar::program::PrintTensor(std::to_string(i), input));
        return popops::map(graph, pe::Mul(pe::_1, pe::Const(i++)), {input},
                           seq);
      });
  return seq;
}

extern "C" void Build_metadata(std::unordered_set<std::int64_t>& allocating_indices,
    std::unordered_map<std::int64_t,std::int64_t>& layout_dependencies,
    std::uint32_t& num_inplace, bool& is_elementwise,
    std::uint32_t num_inputs) {
  num_inplace = num_inputs;
  is_elementwise = num_inputs < 2;

  if (num_inputs == 2) {
    allocating_indices.insert(0);
    allocating_indices.insert(1);
    layout_dependencies[0] = 1;
  }
}

extern "C" poplar::Tensor Build_allocator(std::uint32_t operand) {
  return poplar::Tensor{};  
}
