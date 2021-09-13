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
#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>

extern "C" {
int32_t custom_op_api_level = 5;
}

namespace pe = popops::expr;

// Custom poplar kernel.
extern "C" poplar::program::Program Build(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes,
    const std::string& debugPrefix) {
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
    poplar::Graph& graph, int input_grad_index,
    const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& fwd_inputs,
    const std::vector<poplar::Tensor>& fwd_outputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes,
    const std::string& debugPrefix) {
  poplar::program::Sequence seq;

  outputs.resize(gradients.size());

  float i = 1.0f;
  std::transform(gradients.begin(), gradients.end(), outputs.begin(),
                 [&](const poplar::Tensor& input) {
                   return popops::map(graph, pe::Mul(pe::_1, pe::Const(i++)),
                                      {input}, seq);
                 });
  return seq;
}

extern "C" void Build_metadata(
    std::vector<std::int64_t>& allocating_indices,
    std::vector<std::int64_t>& replica_identical_output_indices,
    std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
    bool& is_elementwise, bool& is_stateless, bool& is_hashable,
    std::uint32_t num_inputs) {
  for (std::uint32_t i = 0; i != num_inputs; ++i) {
    input_to_output_tensor_aliasing.emplace(i, i);
  }
  is_elementwise = num_inputs < 2;
}

extern "C" poplar::Tensor Build_allocator(poplar::Graph& graph,
                                          std::uint32_t operand,
                                          const std::vector<size_t>& shape,
                                          poplar::Type type,
                                          const std::string& attributes,
                                          const std::string& debugPrefix) {
  return poplar::Tensor{};
}

// An alternative name for the same operation
extern "C" poplar::program::Program SepGrad(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes,
    const std::string& debugPrefix) {
  return Build(graph, inputs, outputs, attributes, debugPrefix);
}

extern "C" poplar::program::Program SepGrad_grad(
    poplar::Graph& graph, int input_grad_index,
    const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& fwd_outputs,
    const std::vector<poplar::Tensor>& fwd_inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes,
    const std::string& debugPrefix) {
  poplar::program::Sequence seq;

  outputs.resize(1);

  float i = static_cast<float>(input_grad_index + 1);
  outputs[0] = popops::map(graph, pe::Mul(pe::_1, pe::Const(i)),
                           {gradients[input_grad_index]}, seq);

  return seq;
}

// Custom poplar Add kernel with allocator
extern "C" poplar::program::Program AllocTest(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes,
    const std::string& debugPrefix) {
  poplar::program::Sequence seq;

  outputs.resize(1);

  outputs[0] =
      popops::map(graph, pe::Add(pe::_1, pe::_2), {inputs[0], inputs[1]}, seq);
  return seq;
}

extern "C" void AllocTest_metadata(
    std::vector<std::int64_t>& allocating_indices,
    std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
    bool& is_elementwise, std::uint32_t num_inputs) {
  allocating_indices.push_back(0);
  is_elementwise = false;
}

extern "C" poplar::Tensor AllocTest_allocator(poplar::Graph& graph,
                                              std::uint32_t operand,
                                              const std::vector<size_t>& shape,
                                              poplar::Type type,
                                              const std::string& attributes,
                                              const std::string& debugPrefix) {
  auto t = graph.addVariable(type, shape);
  graph.setTileMapping(t, 0);
  return t;
}

// Stateful test op

extern "C" poplar::program::Program Stateful(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes,
    const std::string& debugPrefix) {
  poplar::program::Sequence seq;

  outputs.resize(1);

  outputs[0] = popops::map(graph, pe::Add(pe::_1, pe::_2),
                           {inputs[0], inputs[1]}, seq, "Stateful");
  return seq;
}

extern "C" void Stateful_metadata(
    std::vector<std::int64_t>& allocating_indices,
    std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
    bool& is_elementwise, bool& is_stateless, bool& is_hashable,
    std::uint32_t num_inputs) {
  // default value must indicate stateful op
}

// Stateless test op

extern "C" poplar::program::Program Stateless(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes,
    const std::string& debugPrefix) {
  poplar::program::Sequence seq;

  outputs.resize(1);

  outputs[0] = popops::map(graph, pe::Add(pe::_1, pe::_2),
                           {inputs[0], inputs[1]}, seq, "Stateless");
  return seq;
}

extern "C" void Stateless_metadata(
    std::vector<std::int64_t>& allocating_indices,
    std::vector<std::int64_t>& replica_identical_output_indices,
    std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
    bool& is_elementwise, bool& is_stateless, bool& is_hashable,
    std::uint32_t num_inputs) {
  is_stateless = true;
}
