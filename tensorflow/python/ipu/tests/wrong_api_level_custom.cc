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

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>

namespace pe = popops::expr;

// This program simulates custom op with different API Level.
// Function prototypes have been changed to test that they are not executed.

extern "C" {
int32_t custom_op_api_level = 0x0bad0ab1;
}

extern "C" poplar::program::Program Build(poplar::Graph& graph) {
  poplar::program::Sequence seq;
  return seq;
}

extern "C" poplar::program::Program Build_grad(poplar::Graph& graph) {
  poplar::program::Sequence seq;

  return seq;
}
