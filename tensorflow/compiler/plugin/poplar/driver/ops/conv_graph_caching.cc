/* Copyright 2017 Graphcore Ltd
 */

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

#include "tensorflow/compiler/plugin/poplar/driver/ops/conv_graph_caching.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/graph_caching_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/ml_type_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include <poplar/Tensor.hpp>
#include <poplin/ConvUtil.hpp>
#include <poplin/Convolution.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <poputil/GraphFunction.hpp>

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {

namespace conv_graph_caching {
namespace {

BwdWeightCacheKey GetBwdWeightCacheKey(const poplar::Tensor& weights,
                                       const poplar::Tensor& bwd_weights,
                                       const uint64 device_id) {
  return std::make_tuple(
      graph_caching_util::GetPoplarTensorSignature(weights),
      graph_caching_util::GetPoplarTensorSignature(bwd_weights), device_id);
}

void CreateCachedBwdWeights(poplar::Graph& graph, CompilerResources& res,
                            const poplar::Tensor& weights,
                            const poplar::Tensor& bwd_weights,
                            const uint64 device_id,
                            poplar::program::Sequence& prog,
                            const std::string& debug_prefix) {
  auto key = GetBwdWeightCacheKey(weights, bwd_weights, device_id);
  std::vector<poplar::Tensor> args = {weights, bwd_weights};
  auto it = res.bwd_weight_graph_cache.find(key);
  if (it != res.bwd_weight_graph_cache.end()) {
    auto& f = it->second;
    f(args, prog);
    return;
  }
  using namespace poputil::graphfn;
  auto f = VoidFunction(
      graph, {input(weights, "weights"), output(bwd_weights, "bwd_weights")},
      [&](std::vector<poplar::Tensor>& args, poplar::program::Sequence& prog) {
        poplin::weightsTransposeChansFlipXY(graph, args[0], args[1], prog,
                                            debug_prefix);
        return prog;
      });
  res.bwd_weight_graph_cache.emplace(key, f);
  f(args, prog);
}

ConvolutionCacheKey GetConvolutionCacheKey(const poplin::ConvParams& params,
                                           const MLType& conv_type,
                                           bool transpose_and_flip_weights,
                                           const uint64 device_id) {
  // Create signature for the convolution input
  std::vector<std::size_t> in_shape = {params.getBatchSize(),
                                       params.getNumInputChans()};
  in_shape.insert(in_shape.end(), params.inputFieldShape.begin(),
                  params.inputFieldShape.end());
  PoplarTensorSignature in_sig(params.inputType, std::move(in_shape));

  // Create signature for the weights
  std::vector<std::size_t> weights_shape = {
      params.getNumConvGroups(), params.getNumOutputChansPerConvGroup(),
      params.getNumInputChansPerConvGroup()};
  weights_shape.insert(weights_shape.end(), params.kernelShape.begin(),
                       params.kernelShape.end());
  PoplarTensorSignature weights_sig(params.inputType, std::move(weights_shape));
  return std::make_tuple(in_sig, weights_sig, params.canonicalize(), conv_type,
                         transpose_and_flip_weights, device_id);
}
}  // namespace

StatusOr<poplar::Tensor> DoCachedConvolution(
    poplar::Graph& graph, CompilerResources& res, const poplar::Tensor& in,
    const poplar::Tensor& input_weights, const poplin::ConvParams& params,
    const MLType& input_conv_type, bool input_transpose_and_flip_weights,
    const uint64 device_id, poplar::program::Sequence& prog,
    const std::string& debug_prefix) {
  // If this is a pass bwd convolution, turn it into a
  // weightsTransposeChansFlipXY and a fwd pass convolution - this allows us to
  // reuse the graph for the convolution and save code space.
  MLType conv_type = input_conv_type;
  poplar::Tensor weights = input_weights;
  bool transpose_and_flip_weights = input_transpose_and_flip_weights;
  // If this is a backprop input convolution perform the
  // weightsTransposeChansFlipXY on weights.
  if (conv_type == MLType::TRAINING_BWD &&
      !res.disable_graph_convolution_caching && transpose_and_flip_weights) {
    conv_type = MLType::TRAINING_FWD;
    transpose_and_flip_weights = false;
    auto fwd_opts = GetConvolutionOptionsForType(res, conv_type);
    auto bwd_weights = poplin::createWeights(graph, params, "bwd_weights",
                                             fwd_opts, &res.convolution_cache);
    CreateCachedBwdWeights(graph, res, weights, bwd_weights, device_id, prog,
                           debug_prefix);
    weights = bwd_weights;
  }
  // Perform the convolution.
  std::vector<poplar::Tensor> args = {in, weights};
  auto key = GetConvolutionCacheKey(params.canonicalize(), conv_type,
                                    transpose_and_flip_weights, device_id);
  auto it = res.conv_graph_cache.find(key);
  if (it != res.conv_graph_cache.end() &&
      !res.disable_graph_convolution_caching) {
    auto& f = it->second;
    return f(args, prog);
  }
  auto opts = GetConvolutionOptionsForType(res, conv_type);

  if (VLOG_IS_ON(2)) {
    std::stringstream stream;
    poplin::reportPlanInfo(stream, graph, params, opts, &res.convolution_cache);
    VLOG(2) << "Convolution " << debug_prefix << ". Type "
            << MLType_Name(conv_type) << ". Plan " << stream.str();
    for (auto opt : opts) {
      VLOG(2) << "- option: " << opt.first << " = " << opt.second;
    }
  }

  poplar::Tensor sig_in = in;
  if (sig_in.containsAliases()) {
    VLOG(1) << "Reallocating input to cached convolution for " << debug_prefix;

    sig_in = poplin::createInput(graph, params, debug_prefix + "/ReallocateIn",
                                 opts, &res.convolution_cache);

    if (sig_in.shape() != in.shape()) {
      return InternalErrorStrCat(
          "Mismatch of convolution input shape, expected: ",
          absl::StrJoin(in.shape(), ", "),
          ", got: ", absl::StrJoin(sig_in.shape(), ", "));
    }
  }

  poplar::Tensor sig_weights = weights;
  if (sig_weights.containsAliases()) {
    VLOG(1) << "Reallocating weights to cached convolution for "
            << debug_prefix;
    sig_weights = poplin::createWeights(graph, params,
                                        debug_prefix + "/ReallocateWeights",
                                        opts, &res.convolution_cache);
    if (sig_weights.shape() != weights.shape()) {
      return InternalErrorStrCat(
          "Mismatch of convolution weight shape, expected: ",
          absl::StrJoin(weights.shape(), ", "),
          ", got: ", absl::StrJoin(sig_weights.shape(), ", "));
    }
  }

  using namespace poputil::graphfn;
  auto f = TensorFunction(
      graph, {input(sig_in, "in"), input(sig_weights, "weights")},
      [&](std::vector<poplar::Tensor>& args, poplar::program::Sequence& prog) {
        return convolution(graph, args[0], args[1], params,
                           transpose_and_flip_weights, prog, debug_prefix, opts,
                           &res.convolution_cache);
      });
  res.conv_graph_cache.emplace(key, f);
  return f(args, prog);
}
}  // namespace conv_graph_caching
}  // namespace poplarplugin
}  // namespace xla
