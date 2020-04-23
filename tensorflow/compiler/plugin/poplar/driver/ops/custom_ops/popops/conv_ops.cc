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
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"

#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_poplar_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

#include "absl/strings/str_cat.h"

#include <poplar/Graph.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/Collectives.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {
namespace {

// This function operates on the poplibs format weights (GOI...)
poplar::Tensor RemoveGroupsDimensionFromWeights(const poplin::ConvParams& p,
                                                const poplar::Tensor& t,
                                                bool flipped) {
  poplar::Tensor out = t;
  return out.reshapePartial(0, 2, {out.dim(0) * out.dim(1)});
}

// This function operates on the poplibs format weights (GOI...)
poplar::Tensor AddGroupsDimensionToWeights(const poplin::ConvParams& p,
                                           const poplar::Tensor& t,
                                           bool flipped) {
  poplar::Tensor out = t;

  unsigned int out_dim = flipped ? 1 : 0;
  unsigned int in_dim = 1 - out_dim;

  if (p.getNumConvGroups() == 1) {
    // Non-grouped case
    return out.reshapePartial(0, 0, {1});
  } else {
    unsigned int chan_div[2];
    chan_div[in_dim] = out.dim(in_dim) / p.getNumInputChansPerConvGroup();
    chan_div[out_dim] = out.dim(out_dim) / p.getNumOutputChansPerConvGroup();

    // OI... ->(GO)(GI)...
    out = out.reshapePartial(0, 2,
                             {chan_div[0], out.dim(0) / chan_div[0],
                              chan_div[1], out.dim(1) / chan_div[1]});

    // (GO)(GI)... -> (GG)OI...
    out = out.dimShufflePartial({2}, {1});

    // (GG)OI... -> GOI...
    return out.reshapePartial(0, 2, {out.dim(0) * out.dim(1)});
  }
}

StatusOr<poplar::Tensor> AddConvolutionInput(poplar::Graph& graph,
                                             const std::string& debug_name,
                                             const HloInstruction* target,
                                             CompilerResources& resources) {
  TF_ASSIGN_OR_RETURN(poplin::ConvParams params,
                      GetConvolutionParameters(target, 0, 1));

  TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                      GetConvolutionOptionsForInst(target, resources));

  auto name = StrCat(debug_name, "_input");
  poplar::Tensor out = poplin::createInput(graph, params, name, opts,
                                           &resources.convolution_cache);

  auto o = ShuffleConvolutionInputToTensorflow(target, out);

  return o;
}

StatusOr<poplar::Tensor> AddConvolutionWeights(poplar::Graph& graph,
                                               const std::string& debug_name,
                                               const HloInstruction* target,
                                               CompilerResources& resources) {
  TF_ASSIGN_OR_RETURN(poplin::ConvParams params,
                      GetConvolutionParameters(target, 0, 1));

  TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                      GetConvolutionOptionsForInst(target, resources));

  auto name = StrCat(debug_name, "_weights");
  poplar::Tensor out = poplin::createWeights(graph, params, name, opts,
                                             &resources.convolution_cache);

  out = RemoveGroupsDimensionFromWeights(params, out, false);

  return ShuffleConvolutionWeightsToTensorflow(target, out);
}

class Conv2DOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence prog;

    // Find the input tensor
    TF_ASSIGN_OR_RETURN(poplar::Tensor in,
                        FindInstructionInput(tensor_map, res, inst, 0, prog));

    // Find the kernel tensor
    TF_ASSIGN_OR_RETURN(poplar::Tensor kernel,
                        FindInstructionInput(tensor_map, res, inst, 1, prog));

    TF_ASSIGN_OR_RETURN(poplin::ConvParams params,
                        GetConvolutionParameters(inst, 0, 1));

    in = ShuffleConvolutionInputToPoplar(inst, in);

    kernel = ShuffleConvolutionWeightsToPoplar(inst, kernel, false);

    kernel = AddGroupsDimensionToWeights(params, kernel, false);

    TF_ASSIGN_OR_RETURN(poplar::Tensor out,
                        conv_graph_caching::DoCachedConvolution(
                            graph, res, in, kernel, params, inst, false, prog));

    out = ShuffleConvolutionOutputToTensorflow(inst, out);

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

    return prog;
  }

  // We want the accumulation tensor to be the same layout as the input tensor.
  StatusOr<poplar::Tensor> Allocator(poplar::Graph& graph,
                                     CompilerResources& res,
                                     const std::string& name,
                                     const TensorTarget& tensor_target,
                                     const TensorMap& tensor_map) override {
    const int64 input_index = tensor_target.input_index;

    const HloInstruction* inst = tensor_target.tgt;

    poplar::Tensor out;
    switch (input_index) {
      case 0: {
        TF_ASSIGN_OR_RETURN(
            out, AddConvolutionInput(graph, GetDebugName(inst), inst, res));
        break;
      }
      case 1: {
        TF_ASSIGN_OR_RETURN(
            out, AddConvolutionWeights(graph, GetDebugName(inst), inst, res));
        break;
      }
      default:
        return xla::FailedPrecondition(
            "Input index %d of convolution shouldn't be allocating",
            input_index);
    }

    return out;
  }
};

REGISTER_HLO_OP(kConvolution, Conv2DOp);
REGISTER_POPLAR_OP(Depthwise_conv, Conv2DOp);

class Conv2DReverseOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence prog;

    // Find the input tensor
    TF_ASSIGN_OR_RETURN(poplar::Tensor in,
                        FindInstructionInput(tensor_map, res, inst, 0, prog));

    // Find the kernel tensor
    TF_ASSIGN_OR_RETURN(poplar::Tensor kernel,
                        FindInstructionInput(tensor_map, res, inst, 1, prog));

    TF_ASSIGN_OR_RETURN(poplin::ConvParams params,
                        GetConvolutionParameters(inst, 0, 1));

    in = ShuffleConvolutionInputToPoplar(inst, in);

    kernel = ShuffleConvolutionWeightsToPoplar(inst, kernel, true);

    kernel = AddGroupsDimensionToWeights(params, kernel, true);

    TF_ASSIGN_OR_RETURN(poplar::Tensor out,
                        conv_graph_caching::DoCachedConvolution(
                            graph, res, in, kernel, params, inst, true, prog));

    out = ShuffleConvolutionOutputToTensorflow(inst, out);

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

    return prog;
  }
};

REGISTER_POPLAR_OP(Conv_with_reverse, Conv2DReverseOp);

class DepthwiseBackpropFilterOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence prog;

    // Find the input tensor
    TF_ASSIGN_OR_RETURN(poplar::Tensor in,
                        FindInstructionInput(tensor_map, res, inst, 0, prog));

    // Find the kernel tensor
    TF_ASSIGN_OR_RETURN(poplar::Tensor kernel,
                        FindInstructionInput(tensor_map, res, inst, 1, prog));

    TF_ASSIGN_OR_RETURN(poplin::ConvParams params,
                        GetConvolutionParameters(inst, 0, 1));

    in = ShuffleConvolutionInputToPoplar(inst, in);

    // Move 'G' parts of the I to B (because B is the reducing dimension)
    unsigned n_g = params.getNumConvGroups();
    in = in.reshapePartial(0, 1, {n_g, in.dim(0) / n_g});
    in = in.dimShufflePartial({0}, {1});
    in = in.reshapePartial(1, 3, {in.dim(1) * in.dim(2)});

    kernel = ShuffleConvolutionWeightsToPoplar(inst, kernel, false);

    kernel = AddGroupsDimensionToWeights(params, kernel, false);

    TF_ASSIGN_OR_RETURN(poplar::Tensor out,
                        conv_graph_caching::DoCachedConvolution(
                            graph, res, in, kernel, params, inst, false, prog));

    // Move 'G' parts of the B back to I
    out = out.reshapePartial(1, 2, {n_g, out.dim(1) / n_g});
    out = out.dimShufflePartial({1}, {0});
    out = out.reshapePartial(0, 2, {out.dim(0) * out.dim(1)});

    out = ShuffleConvolutionOutputToTensorflow(inst, out);

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

    return prog;
  }
};

REGISTER_POPLAR_OP(Depthwise_filter, DepthwiseBackpropFilterOp);

class ConvScaledInplaceOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;

    // Find the weights tensor
    TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                        FindInplaceOutputTensors(tensor_map, res, inst, seq));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor arg_weights = inputs[0][0];

    // Find the input tensor
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_in,
                        FindInstructionInput(tensor_map, res, inst, 1, seq));

    // Find the deltas tensor
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_deltas,
                        FindInstructionInput(tensor_map, res, inst, 2, seq));

    // Find the scale.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_scale,
        FindInstructionInput(tensor_map, res, inst, 3, seq, false));

    TF_ASSIGN_OR_RETURN(poplin::ConvParams params,
                        GetConvolutionParameters(inst, 1, 2));

    TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                        GetConvolutionOptionsForInst(inst, res));

    const ConvolutionDimensionNumbers& conv_dims = GetConvolutionDims(inst);
    // Get the root of the fusion - it indicates whether this is add or
    // subtract.
    const auto* root_inst = inst->fused_expression_root();
    auto op_type = root_inst->opcode();

    const std::string debug_prefix = GetDebugName(inst);

    auto func = [&graph, &res, params, opts, conv_dims, op_type, debug_prefix](
                    std::vector<poplar::Tensor>& args,
                    poplar::program::Sequence& prog) {
      poplar::Tensor weights = args[0];
      poplar::Tensor in = args[1];
      poplar::Tensor deltas = args[2];
      poplar::Tensor scale = args[3];

      weights = ShuffleConvolutionOutputToPoplar(conv_dims, weights);
      in = ShuffleConvolutionInputToPoplar(conv_dims, in);
      deltas = ShuffleConvolutionWeightsToPoplar(conv_dims, deltas, false);
      deltas = AddGroupsDimensionToWeights(params, deltas, false);

      auto c_out =
          poplin::convolution(graph, in, deltas, params, false, prog,
                              debug_prefix, opts, &res.convolution_cache);

      TF_CHECK_OK(ScaledInplaceConstantOrTensor(graph, weights, c_out, scale,
                                                prog, op_type, debug_prefix));

      args[0] = ShuffleConvolutionOutputToTensorflow(conv_dims, weights);
    };

    std::vector<poplar::Tensor> args = {arg_weights, arg_in, arg_deltas,
                                        arg_scale};
    poputil::graphfn::Signature signature = {
        poputil::graphfn::inout(arg_weights, "w"),
        poputil::graphfn::input(arg_in, "in"),
        poputil::graphfn::input(arg_deltas, "deltas"),
        poputil::graphfn::input(arg_scale, "scale"),
    };

    TF_RETURN_IF_ERROR(res.graph_cache.ExecuteCached(inst, graph, res, seq,
                                                     func, signature, args));

    arg_weights = args[0];

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, arg_weights));

    return seq;
  }
};
REGISTER_POPLAR_OP(Conv_scaled_inplace, ConvScaledInplaceOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
