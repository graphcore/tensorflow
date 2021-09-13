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
#include <poplin/MatMul.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/exceptions.hpp>

// Use the https://github.com/open-source-parsers/jsoncpp JsonCpp parser
#include "include/json/json.h"

extern "C" {
int32_t custom_op_api_level = 5;
}

namespace {
Json::Value ParseAttributes(const std::string& attributes) {
  // Parse Json.
  Json::CharReaderBuilder builder;
  std::string errs;
  Json::Value parsed_json;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  bool parsed =
      reader->parse(attributes.c_str(), attributes.c_str() + attributes.size(),
                    &parsed_json, &errs);
  assert(parsed && errs);
  return parsed_json;
}

std::vector<size_t> GetVectorFromJson(Json::Value& val) {
  std::vector<size_t> result;
  result.reserve(val.size());
  for (auto a : val) {
    result.push_back(a.asUInt64());
  }
  return result;
}
}  // namespace

extern "C" void Build_metadata(
    std::vector<std::int64_t>& allocating_indices,
    std::vector<std::int64_t>& replica_identical_output_indices,
    std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
    bool& is_elementwise, bool& is_hashable, std::uint32_t num_inputs) {
  allocating_indices = {0, 1};
  is_elementwise = false;
}

extern "C" poplar::Tensor Build_allocator(poplar::Graph& graph,
                                          std::uint32_t operand,
                                          const std::vector<size_t>& shape,
                                          poplar::Type type,
                                          const std::string& attributes,
                                          const std::string& debug_prefix) {
  assert(operand < 2);
  // Parse JSON and get the expected attributes.
  Json::Value json = ParseAttributes(attributes);
  const int serialization_factor = json["serialization_factor"].asInt();
  std::vector<std::size_t> lhs_shape = GetVectorFromJson(json["lhs_shape"]);
  std::vector<std::size_t> rhs_shape = GetVectorFromJson(json["rhs_shape"]);

  // Verify shapes and adjust them to be slice shapes.
  assert(lhs_shape.size() == 2);
  assert(rhs_shape.size() == 2);

  assert(lhs_shape[1] % serialization_factor == 0 &&
         "serialization_factor must divide the dimension of LHS shape");
  lhs_shape[1] /= serialization_factor;

  assert(rhs_shape[0] % serialization_factor == 0 &&
         "serialization_factor must divide the dimension of RHS shape");
  rhs_shape[0] /= serialization_factor;

  // Allocate the slice.
  poplar::Tensor slice;
  if (operand == 0) {
    // Allocating for lhs - allocate the slice.
    slice = poplin::createMatMulInputLHS(graph, type, lhs_shape, rhs_shape,
                                         debug_prefix + "/LHS");
  } else {
    assert(operand == 1);
    slice = poplin::createMatMulInputRHS(graph, type, lhs_shape, rhs_shape,
                                         debug_prefix + "/RHS");
  }

  // Clone the slice for each serialized matrix multiply.
  std::vector<poplar::Tensor> slices(serialization_factor);
  slices[0] = slice;
  for (int i = 1; i != serialization_factor; ++i) {
    slices[i] = graph.clone(slice);
  }

  // Concatenate the slices into a single tensor - the concatentation dimension
  // depends on the operand which is being allocated.
  poplar::Tensor t = poplar::concat(slices, operand == 0 ? 1 : 0);
  return t;
}

extern "C" poplar::program::Program Build(poplar::Graph& graph,
                                          std::vector<poplar::Tensor>& inputs,
                                          std::vector<poplar::Tensor>& outputs,
                                          const std::string& attributes,
                                          const std::string& debug_prefix) {
  if (inputs.size() != 2) {
    throw poputil::poplibs_error("add requires 2 inputs.");
  }
  Json::Value json = ParseAttributes(attributes);
  poplar::program::Sequence seq;
  poplar::Tensor lhs = inputs[0];
  poplar::Tensor rhs = inputs[1];
  poplar::Tensor output;

  const int serialization_factor = json["serialization_factor"].asInt();
  const int slice_size = lhs.dim(1) / serialization_factor;
  for (int i = 0; i != serialization_factor; ++i) {
    // Slice out the parts of the matmul.
    poplar::Tensor lhs_slice =
        lhs.slice(i * slice_size, (i + 1) * slice_size, 1);
    poplar::Tensor rhs_slice =
        rhs.slice(i * slice_size, (i + 1) * slice_size, 0);
    // Do the partial matmul.
    poplar::Tensor partial_matmul = poplin::matMul(
        graph, lhs_slice, rhs_slice, seq, debug_prefix + "/Slice");

    // Accumulate the results from partial matmuls.
    if (i == 0) {
      output = partial_matmul;
    } else {
      popops::addInPlace(graph, output, partial_matmul, seq,
                         debug_prefix + "/Add");
    }
  }
  outputs = {output};
  return seq;
}
