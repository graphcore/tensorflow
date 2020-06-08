/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_CONV_GRAPH_CACHING_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_CONV_GRAPH_CACHING_H_

#include "tensorflow/compiler/plugin/poplar/driver/ops/graph_caching_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/convolution_classifier.h"

#include <poplar/Tensor.hpp>
#include <poplin/ConvUtil.hpp>
#include <poputil/GraphFunction.hpp>

namespace xla {
namespace poplarplugin {
struct CompilerResources;

namespace conv_graph_caching {

// Fwd and bwd convolution caches

// The convolution key is:
// * Shape of the input tensor
// * Shape of the weights tensor
// * ConvolutionDimensionNumbers for the given convolution
// * poplin ConvParams for the given convolution
// * Enum for the type of convolution
// * bool indicating whether to do weightsTransposeChansFlipXY
// * sharding device ID
using ConvolutionCacheKey =
    std::tuple<PoplarTensorSignature, PoplarTensorSignature, poplin::ConvParams,
               MLType, bool, uint64>;
using ConvolutionGraphCache =
    std::map<ConvolutionCacheKey, poputil::graphfn::TensorFunction>;

using BwdWeightCacheKey =
    std::tuple<PoplarTensorSignature, PoplarTensorSignature, uint64>;
using BwdWeightGraphCache =
    std::map<BwdWeightCacheKey, poputil::graphfn::VoidFunction>;

StatusOr<poplar::Tensor> DoCachedConvolution(
    poplar::Graph& graph, CompilerResources& res, const poplar::Tensor& in,
    const poplar::Tensor& weights, const poplin::ConvParams& params,
    const HloInstruction* inst, bool transpose_and_flip_weights,
    poplar::program::Sequence& prog);
}  // namespace conv_graph_caching
}  // namespace poplarplugin
}  // namespace xla

#endif
