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
#include <iostream>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <unordered_map>
#include <unordered_set>

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
  std::transform(
      gradients.begin(), gradients.end(), outputs.begin(),
      [&](const poplar::Tensor& input) {
        seq.add(poplar::program::PrintTensor(std::to_string(i), input));
        return popops::map(graph, pe::Mul(pe::_1, pe::Const(i++)), {input},
                           seq);
      });
  return seq;
}

// Custom host program.
extern "C" void Callback(const std::vector<void*>& data,
                         const std::vector<std::uint32_t>& number_of_elements,
                         std::vector<void*>& outputs,
                         const std::string& attributes,
                         const std::string& name) {
  float acc = 0.0f;
  float* out_ptr = (float*)outputs[0];
  float* in_ptr = (float*)data[0];
  for (std::uint32_t i = 0; i < number_of_elements[0]; ++i) {
    out_ptr[i] = in_ptr[i] + 6.0f;
    acc += out_ptr[i];
  }

  std::int32_t* out_ptr2 = (std::int32_t*)outputs[1];
  std::int32_t* in_ptr2 = (std::int32_t*)data[1];
  for (std::uint32_t i = 0; i < number_of_elements[1]; ++i) {
    out_ptr2[i] = in_ptr2[i] / 2;
    acc += out_ptr2[i];
  }

  *((float*)outputs[2]) = acc;
}
