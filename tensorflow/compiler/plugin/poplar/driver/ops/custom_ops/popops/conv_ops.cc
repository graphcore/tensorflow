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
#include <poplin/MultiConvolution.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/Collectives.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_conv.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {
namespace {

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

  out = RemoveGroupsDimensionFromWeights(params, out);

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
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor in,
        FindInstructionInput(tensor_map, res, inst, 0, prog, false));

    // Find the kernel tensor
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor kernel,
        FindInstructionInput(tensor_map, res, inst, 1, prog, false));

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
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor in,
        FindInstructionInput(tensor_map, res, inst, 0, prog, false));

    // Find the kernel tensor
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor kernel,
        FindInstructionInput(tensor_map, res, inst, 1, prog, false));

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

poplar::Tensor DepthwiseFilterShuffleInput(const poplin::ConvParams& params,
                                           const poplar::Tensor& in) {
  // Move 'G' parts of the I to B (because B is the reducing dimension)
  const unsigned n_g = params.getNumConvGroups();
  poplar::Tensor out = in.reshapePartial(0, 1, {n_g, in.dim(0) / n_g});
  out = out.dimShufflePartial({0}, {1});
  out = out.reshapePartial(1, 3, {out.dim(1) * out.dim(2)});
  return out;
}

poplar::Tensor DepthwiseFilterShuffleOutput(const poplin::ConvParams& params,
                                            const poplar::Tensor& in) {
  // Move 'G' parts of the B back to I
  const unsigned n_g = params.getNumConvGroups();
  poplar::Tensor out = in.reshapePartial(1, 2, {n_g, in.dim(1) / n_g});
  out = out.dimShufflePartial({1}, {0});
  out = out.reshapePartial(0, 2, {out.dim(0) * out.dim(1)});
  return out;
}

class DepthwiseBackpropFilterOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence prog;

    // Find the input tensor
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor in,
        FindInstructionInput(tensor_map, res, inst, 0, prog, false));

    // Find the kernel tensor
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor kernel,
        FindInstructionInput(tensor_map, res, inst, 1, prog, false));

    TF_ASSIGN_OR_RETURN(poplin::ConvParams params,
                        GetConvolutionParameters(inst, 0, 1));

    in = ShuffleConvolutionInputToPoplar(inst, in);

    in = DepthwiseFilterShuffleInput(params, in);

    kernel = ShuffleConvolutionWeightsToPoplar(inst, kernel, false);

    kernel = AddGroupsDimensionToWeights(params, kernel, false);

    TF_ASSIGN_OR_RETURN(poplar::Tensor out,
                        conv_graph_caching::DoCachedConvolution(
                            graph, res, in, kernel, params, inst, false, prog));

    out = DepthwiseFilterShuffleOutput(params, out);

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

StatusOr<std::vector<poplin::multiconv::CreateTensorArgs>>
GetMultiConvCreateArgs(const HloMultiConvInstruction* inst,
                       CompilerResources& res, const std::string& name) {
  TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                      GetConvolutionOptionsForInst(inst, res));

  const auto& convolution_specs = inst->GetConvolutionSpecs();

  std::vector<poplin::multiconv::CreateTensorArgs> conv_args(
      convolution_specs.size());

  TF_ASSIGN_OR_RETURN(std::vector<poplin::ConvParams> conv_params,
                      GetConvolutionParametersForMultiConv(inst));

  for (int64 i = 0; i != convolution_specs.size(); ++i) {
    conv_args[i] = {conv_params[i], opts, absl::StrCat(name, "/SubConv", i)};
  }

  return conv_args;
}

class MultiConvOp : public PoplarOpDef {
  StatusOr<poplar::Tensor> Allocator(poplar::Graph& graph,
                                     CompilerResources& res,
                                     const std::string& name,
                                     const TensorTarget& tensor_target,
                                     const TensorMap& tensor_map) override {
    const int64 input_index = tensor_target.input_index;
    const HloMultiConvInstruction* inst =
        Cast<HloMultiConvInstruction>(tensor_target.tgt);

    const auto& convolution_specs = inst->GetConvolutionSpecs();
    // Operands [0, n) are inputs and [n, 2n) are kernels.
    const bool is_conv_input = input_index < convolution_specs.size();
    const int64 conv_index =
        input_index - (is_conv_input ? 0 : convolution_specs.size());
    CHECK_LT(conv_index, convolution_specs.size());
    auto convolution_spec = convolution_specs[conv_index];

    TF_ASSIGN_OR_RETURN(
        std::vector<poplin::multiconv::CreateTensorArgs> create_args,
        GetMultiConvCreateArgs(inst, res, name));

    poplar::Tensor out;
    switch (convolution_spec.type) {
      case ConvType::Conv:
      case ConvType::DepthwiseConv: {
        if (is_conv_input) {
          out = poplin::multiconv::createInput(graph, create_args, conv_index,
                                               &res.convolution_cache);
          out = ShuffleConvolutionInputToTensorflow(convolution_spec.dims, out);
        } else {
          out = poplin::multiconv::createWeights(graph, create_args, conv_index,
                                                 &res.convolution_cache);
          out = RemoveGroupsDimensionFromWeights(create_args[conv_index].params,
                                                 out);
          out =
              ShuffleConvolutionWeightsToTensorflow(convolution_spec.dims, out);
        }
        break;
      }
      default: {
        return FailedPrecondition(
            "Cannot allocate a tensor for multi conv %d from %s.", conv_index,
            inst->ToString().c_str());
      }
    }
    return out;
  }

  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;
    const HloMultiConvInstruction* multi_conv_inst =
        Cast<HloMultiConvInstruction>(inst);
    const std::string debug_name = GetDebugName(inst);

    TF_ASSIGN_OR_RETURN(
        std::vector<poplin::multiconv::CreateTensorArgs> create_args,
        GetMultiConvCreateArgs(multi_conv_inst, res, debug_name));

    const auto& convolution_specs = multi_conv_inst->GetConvolutionSpecs();

    auto func = [&graph, &res, create_args, convolution_specs, debug_name](
                    std::vector<poplar::Tensor>& args,
                    poplar::program::Sequence& prog) -> void {
      // Check whether we can set transpose_and_flip_weights for all
      // convolutions, and if not, any `ConvWithReverse` needs to do it before
      // the multi conv individually.
      const bool all_transpose_and_flip_weights = absl::c_all_of(
          convolution_specs,
          [](const HloMultiConvInstruction::ConvolutionSpec& convolution_spec) {
            return convolution_spec.type == ConvType::ConvWithReverse;
          });

      std::vector<poplin::multiconv::ConvolutionArgs> conv_args(
          convolution_specs.size());
      for (int64 i = 0; i != convolution_specs.size(); ++i) {
        const auto& convolution_spec = convolution_specs[i];
        poplar::Tensor input = args[i];
        poplar::Tensor kernel = args[i + convolution_specs.size()];

        // Process the inputs, which is dependent on the convolution type.
        switch (convolution_spec.type) {
          case ConvType::Conv:
          case ConvType::DepthwiseConv: {
            input =
                ShuffleConvolutionInputToPoplar(convolution_spec.dims, input);

            kernel = ShuffleConvolutionWeightsToPoplar(convolution_spec.dims,
                                                       kernel, false);
            kernel = AddGroupsDimensionToWeights(create_args[i].params, kernel,
                                                 false);
            break;
          }
          case ConvType::ConvWithReverse: {
            input =
                ShuffleConvolutionInputToPoplar(convolution_spec.dims, input);

            kernel = ShuffleConvolutionWeightsToPoplar(convolution_spec.dims,
                                                       kernel, true);
            kernel = AddGroupsDimensionToWeights(create_args[i].params, kernel,
                                                 true);
            if (!all_transpose_and_flip_weights) {
              // Transpose the individual kernel.
              poplar::Tensor new_kernel = poplin::createWeights(
                  graph, create_args[i].params,
                  absl::StrCat(debug_name, "/", i, "/BwdWeights"),
                  create_args[i].options, &res.convolution_cache);

              poplin::weightsTransposeChansFlipXY(
                  graph, kernel, new_kernel, prog,
                  absl::StrCat(debug_name, "/", i));
              kernel = new_kernel;
            }
            break;
          }
          case ConvType::DepthwiseFilter: {
            input =
                ShuffleConvolutionInputToPoplar(convolution_spec.dims, input);
            input = DepthwiseFilterShuffleInput(create_args[i].params, input);

            kernel = ShuffleConvolutionWeightsToPoplar(convolution_spec.dims,
                                                       kernel, false);
            kernel = AddGroupsDimensionToWeights(create_args[i].params, kernel,
                                                 false);
            break;
          }
          default: { LOG(FATAL) << "Unknown convolution type."; }
        }
        conv_args[i] = {input, kernel, create_args[i].params,
                        create_args[i].options};
      }

      std::vector<poplar::Tensor> outputs = poplin::multiconv::convolution(
          graph, conv_args, all_transpose_and_flip_weights, prog, debug_name,
          &res.convolution_cache);

      for (int64 i = 0; i != convolution_specs.size(); ++i) {
        poplar::Tensor output = outputs[i];
        const auto& convolution_spec = convolution_specs[i];
        // Process the outputs, which is dependent on the convolution type.
        switch (convolution_spec.type) {
          case ConvType::Conv:
          case ConvType::ConvWithReverse:
          case ConvType::DepthwiseConv: {
            output = ShuffleConvolutionOutputToTensorflow(convolution_spec.dims,
                                                          output);
            break;
          }
          case ConvType::DepthwiseFilter: {
            output =
                DepthwiseFilterShuffleOutput(create_args[i].params, output);
            output = ShuffleConvolutionOutputToTensorflow(convolution_spec.dims,
                                                          output);
            break;
          }
          default: { LOG(FATAL) << "Unknown convolution type."; }
        }

        args[2 * convolution_specs.size() + i] = output;
      }
    };

    // Get the inputs and the function signature.
    std::vector<poplar::Tensor> args(3 * convolution_specs.size());
    poputil::graphfn::Signature inputs_signature;
    poputil::graphfn::Signature kernels_signature;
    poputil::graphfn::Signature outputs_signature;
    for (int64 i = 0; i != convolution_specs.size(); ++i) {
      // Find the input tensor.
      TF_ASSIGN_OR_RETURN(
          args[i], FindInstructionInput(tensor_map, res, inst, i, seq, false));
      inputs_signature.push_back(
          poputil::graphfn::input(args[i], absl::StrCat("Input", i)));

      // Find the kernels tensor.
      TF_ASSIGN_OR_RETURN(
          args[convolution_specs.size() + i],
          FindInstructionInput(tensor_map, res, inst,
                               convolution_specs.size() + i, seq, false));
      kernels_signature.push_back(poputil::graphfn::input(
          args[convolution_specs.size() + i], absl::StrCat("Kernel", i)));

      outputs_signature.push_back(
          poputil::graphfn::created(absl::StrCat("Output", i)));
    }

    // Combine the signatures.
    poputil::graphfn::Signature signature = inputs_signature;
    signature.insert(signature.end(), kernels_signature.begin(),
                     kernels_signature.end());
    signature.insert(signature.end(), outputs_signature.begin(),
                     outputs_signature.end());

    TF_RETURN_IF_ERROR(res.graph_cache.ExecuteCached(
        inst, graph, res, seq, func, signature, args,
        multi_conv_inst->AllocatingIndices()));

    // Set the outputs.
    for (int64 i = 0; i != convolution_specs.size(); ++i) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i,
                                  args[2 * convolution_specs.size() + i]));
    }

    return seq;
  }
};
REGISTER_POPLAR_OP(MultiConv, MultiConvOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
