/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Sort.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/test.h"

#include <dlfcn.h>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>

using namespace poplar;
using namespace poplar::program;

namespace xla {
namespace poplarplugin {

static poplar::program::Sequence Sort(poplar::Graph& graph,
                                      const poplar::Tensor& t, unsigned dim) {
  poplar::program::Sequence result;

  popops::sortInPlace(graph, t, dim, result);

  return result;
}

static poplar::program::Sequence Sort(poplar::Graph& graph,
                                      const poplar::Tensor& k,
                                      const poplar::Tensor& v, unsigned dim) {
  poplar::program::Sequence result;

  popops::sortKeyValueInPlace(graph, k, v, dim, result);

  return result;
}

template <typename T>
static std::vector<T> iota(std::size_t count) {
  std::vector<T> result(count);

  std::iota(result.begin(), result.end(), 0);

  return result;
}

template <typename T>
static std::vector<T> random(std::size_t count) {
  std::vector<T> result(count);

  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_int_distribution<int> uniform_dist(100, 999);
  std::generate(result.begin(), result.end(), std::bind(uniform_dist, e1));

  return result;
}

template <typename T>
static std::vector<T> zeros(std::size_t count) {
  std::vector<T> result(count);

  std::fill(result.begin(), result.end(), 0.0f);

  return result;
}

TEST(Sort, OneDimension) {
  IPUModel ipuModel;
  ipuModel.tilesPerIPU = 12;
  Device device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

  const std::size_t tensor_size = 256;

  Tensor a = graph.addVariable(FLOAT, {tensor_size}, "input");
  poputil::mapTensorLinearly(graph, a);
  graph.createHostWrite("a-write", a);
  graph.createHostRead("a-read", a);

  auto prog = Sort(graph, a, 0);

  Engine engine(graph, prog);
  engine.load(device);

  const auto input_buffer = random<float>(tensor_size);
  const std::size_t input_buffer_size = input_buffer.size() * sizeof(float);
  engine.writeTensor("a-write", input_buffer.data(),
                     input_buffer.data() + input_buffer_size);

  engine.run();

  std::vector<float> output_buffer = zeros<float>(tensor_size);
  const std::size_t output_buffer_size = output_buffer.size() * sizeof(float);
  engine.readTensor("a-read", output_buffer.data(),
                    output_buffer.data() + output_buffer_size);

  EXPECT_TRUE(std::is_sorted(output_buffer.begin(), output_buffer.end()));
  EXPECT_TRUE(std::is_permutation(output_buffer.begin(), output_buffer.end(),
                                  input_buffer.begin()));
}

TEST(SortInt, OneDimension) {
  IPUModel ipuModel;
  ipuModel.tilesPerIPU = 12;
  Device device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

  const std::size_t tensor_size = 256;

  Tensor a = graph.addVariable(INT, {tensor_size}, "input");
  poputil::mapTensorLinearly(graph, a);
  graph.createHostWrite("a-write", a);
  graph.createHostRead("a-read", a);

  auto prog = Sort(graph, a, 0);

  Engine engine(graph, prog);
  engine.load(device);

  const auto input_buffer = random<int>(tensor_size);
  const std::size_t input_buffer_size = input_buffer.size() * sizeof(float);
  engine.writeTensor("a-write", input_buffer.data(),
                     input_buffer.data() + input_buffer_size);

  engine.run();

  auto output_buffer = zeros<int>(tensor_size);
  const std::size_t output_buffer_size = output_buffer.size() * sizeof(float);
  engine.readTensor("a-read", output_buffer.data(),
                    output_buffer.data() + output_buffer_size);

  EXPECT_TRUE(std::is_sorted(output_buffer.begin(), output_buffer.end()));
  EXPECT_TRUE(std::is_permutation(output_buffer.begin(), output_buffer.end(),
                                  input_buffer.begin()));
}

TEST(SortKV, OneDimension) {
  IPUModel ipuModel;
  ipuModel.tilesPerIPU = 12;
  Device device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

  const std::size_t tensor_size = 256;

  Tensor k = graph.addVariable(FLOAT, {tensor_size}, "key");
  Tensor v = graph.addVariable(FLOAT, {tensor_size}, "value");
  poputil::mapTensorLinearly(graph, k);
  poputil::mapTensorLinearly(graph, v);
  graph.createHostWrite("a-write", k);
  graph.createHostWrite("b-write", v);
  graph.createHostRead("b-read", v);

  auto prog = Sort(graph, k, v, 0);

  Engine engine(graph, prog);
  engine.load(device);

  auto input_buffer = iota<float>(tensor_size);
  const std::size_t input_buffer_size = input_buffer.size() * sizeof(float);
  std::reverse(input_buffer.begin(), input_buffer.end());
  engine.writeTensor("a-write", input_buffer.data(),
                     input_buffer.data() + input_buffer_size);
  std::reverse(input_buffer.begin(), input_buffer.end());
  engine.writeTensor("b-write", input_buffer.data(),
                     input_buffer.data() + input_buffer_size);

  engine.run();

  std::vector<float> output_buffer = zeros<float>(tensor_size);
  const std::size_t output_buffer_size = output_buffer.size() * sizeof(float);
  engine.readTensor("b-read", output_buffer.data(),
                    output_buffer.data() + output_buffer_size);

  EXPECT_TRUE(std::is_sorted(output_buffer.rbegin(), output_buffer.rend()));
  EXPECT_TRUE(std::is_permutation(output_buffer.begin(), output_buffer.end(),
                                  input_buffer.begin()));
}

TEST(Sort, TwoDimension) {
  IPUModel ipuModel;
  ipuModel.tilesPerIPU = 12;
  Device device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

  const std::size_t tensor_size = 8;

  Tensor a = graph.addVariable(FLOAT, {tensor_size, tensor_size}, "input");
  poputil::mapTensorLinearly(graph, a);
  graph.createHostWrite("a-write", a);
  graph.createHostRead("a-read", a);

  auto prog = Sort(graph, a, 1);

  Engine engine(graph, prog);
  engine.load(device);

  const auto input_buffer = random<float>(tensor_size * tensor_size);
  const std::size_t input_buffer_size = input_buffer.size() * sizeof(float);
  engine.writeTensor("a-write", input_buffer.data(),
                     input_buffer.data() + input_buffer_size);

  engine.run();

  std::vector<float> output_buffer = zeros<float>(tensor_size * tensor_size);
  const std::size_t output_buffer_size = output_buffer.size() * sizeof(float);
  engine.readTensor("a-read", output_buffer.data(),
                    output_buffer.data() + output_buffer_size);
  for (int i = 0; i < tensor_size; ++i) {
    const auto begin_idx = i * tensor_size;
    const auto end_idx = begin_idx + tensor_size;

    const auto out_begin = std::next(output_buffer.begin(), begin_idx);
    const auto out_end = std::next(output_buffer.begin(), end_idx);
    const auto in_begin = std::next(input_buffer.begin(), begin_idx);

    EXPECT_TRUE(std::is_sorted(out_begin, out_end));
    EXPECT_TRUE(std::is_permutation(out_begin, out_end, in_begin));
  }
}

TEST(Sort, ThreeDimension) {
  IPUModel ipuModel;
  ipuModel.tilesPerIPU = 12;
  Device device = ipuModel.createDevice();
  Graph graph(device);
  popops::addCodelets(graph);

  const std::size_t tensor_size = 16;

  Tensor a =
      graph.addVariable(FLOAT, {tensor_size, tensor_size, tensor_size}, "key");
  poputil::mapTensorLinearly(graph, a);
  graph.createHostWrite("a-write", a);
  graph.createHostRead("a-read", a);

  auto prog = Sort(graph, a, 2);

  Engine engine(graph, prog);
  engine.load(device);

  const auto input_buffer =
      random<float>(tensor_size * tensor_size * tensor_size);
  const std::size_t input_buffer_size = input_buffer.size() * sizeof(float);
  engine.writeTensor("a-write", input_buffer.data(),
                     input_buffer.data() + input_buffer_size);

  engine.run();

  std::vector<float> output_buffer =
      zeros<float>(tensor_size * tensor_size * tensor_size);
  const std::size_t output_buffer_size = output_buffer.size() * sizeof(float);
  engine.readTensor("a-read", output_buffer.data(),
                    output_buffer.data() + output_buffer_size);
  for (int i = 0; i < tensor_size; ++i) {
    for (int k = 0; k < tensor_size; ++k) {
      const auto begin_idx = i * tensor_size * tensor_size + k * tensor_size;
      const auto end_idx = begin_idx + tensor_size;

      const auto out_begin = std::next(output_buffer.begin(), begin_idx);
      const auto out_end = std::next(output_buffer.begin(), end_idx);
      const auto in_begin = std::next(input_buffer.begin(), begin_idx);

      EXPECT_TRUE(std::is_sorted(out_begin, out_end));
      EXPECT_TRUE(std::is_permutation(out_begin, out_end, in_begin));
    }
  }
}

}  // namespace poplarplugin
}  // namespace xla
