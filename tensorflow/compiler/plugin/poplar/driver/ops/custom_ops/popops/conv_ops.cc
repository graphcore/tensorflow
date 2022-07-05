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
#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <poplin/MultiConvolution.hpp>
#include <popnn/NonLinearity.hpp>
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
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {
namespace {

StatusOr<DriverTensor> AddConvolutionInput(
    DriverGraph& graph, const HloInstruction* target,
    CompilerResources& resources, const poplar::DebugInfo& debug_info) {
  TF_ASSIGN_OR_RETURN(poplin::ConvParams params,
                      GetConvolutionParameters(target, 0, 1));

  TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                      GetConvolutionOptionsForInst(target, resources));

  auto out =
      DriverTensor(poplin::createInput(graph, params, {debug_info, "input"},
                                       opts, &resources.planning_cache),
                   graph);

  auto o = ShuffleConvolutionInputToTensorflow(target, out);

  return o;
}

StatusOr<DriverTensor> AddConvolutionWeights(
    DriverGraph& graph, const HloInstruction* target,
    CompilerResources& resources, const poplar::DebugInfo& debug_info) {
  TF_ASSIGN_OR_RETURN(poplin::ConvParams params,
                      GetConvolutionParameters(target, 0, 1));

  TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                      GetConvolutionOptionsForInst(target, resources));

  auto out =
      DriverTensor(poplin::createWeights(graph, params, {debug_info, "weights"},
                                         opts, &resources.planning_cache),
                   graph);

  out = RemoveGroupsDimensionFromWeights(params, out);

  return ShuffleConvolutionWeightsToTensorflow(target, out);
}

class Conv2DOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "Conv2DOp");
    DriverProgramSequence seq(graph, {debug_info});

    // Find the input tensor
    TF_ASSIGN_OR_RETURN(poplar::Tensor in,
                        FindInstructionInput(tensor_map, res, inst, 0, seq,
                                             {debug_info}, false));

    // Find the kernel tensor
    TF_ASSIGN_OR_RETURN(poplar::Tensor kernel,
                        FindInstructionInput(tensor_map, res, inst, 1, seq,
                                             {debug_info}, false));

    TF_ASSIGN_OR_RETURN(poplin::ConvParams params,
                        GetConvolutionParameters(inst, 0, 1));

    TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                        GetConvolutionOptionsForInst(inst, res));

    TF_ASSIGN_OR_RETURN(int64_t group_count, GetBatchGroupCount(inst));
    TF_ASSIGN_OR_RETURN(ConvolutionDimensionNumbers conv_dims,
                        GetConvolutionDims(inst));

    in = ShuffleConvolutionInputToPoplar(group_count, conv_dims, in);

    kernel = ShuffleConvolutionWeightsToPoplar(conv_dims, kernel, false);

    kernel = AddGroupsDimensionToWeights(params, kernel, false);

    poplar::Tensor out =
        poplin::convolution(graph, in, kernel, params, false, seq, {debug_info},
                            opts, &res.planning_cache);

    out = ShuffleConvolutionOutputToTensorflow(conv_dims, out);

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(out, graph)));

    return seq;
  }

  StatusOr<DriverTensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "Conv2DOp");
    const int64_t input_index = tensor_target.input_index;

    const HloInstruction* inst = tensor_target.tgt;

    DriverTensor out;
    switch (input_index) {
      case 0: {
        TF_ASSIGN_OR_RETURN(out,
                            AddConvolutionInput(graph, inst, res, debug_info));
        break;
      }
      case 1: {
        TF_ASSIGN_OR_RETURN(
            out, AddConvolutionWeights(graph, inst, res, debug_info));
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

class Conv2DReverseOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "Conv2DReverseOp");
    DriverProgramSequence seq(graph, debug_info);

    // Find the input tensor
    TF_ASSIGN_OR_RETURN(poplar::Tensor in,
                        FindInstructionInput(tensor_map, res, inst, 0, seq,
                                             {debug_info}, false));

    // Find the kernel tensor
    TF_ASSIGN_OR_RETURN(poplar::Tensor kernel,
                        FindInstructionInput(tensor_map, res, inst, 1, seq,
                                             {debug_info}, false));

    TF_ASSIGN_OR_RETURN(poplin::ConvParams params,
                        GetConvolutionParameters(inst, 0, 1));

    TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                        GetConvolutionOptionsForInst(inst, res));

    TF_ASSIGN_OR_RETURN(int64_t group_count, GetBatchGroupCount(inst));
    TF_ASSIGN_OR_RETURN(ConvolutionDimensionNumbers conv_dims,
                        GetConvolutionDims(inst));

    in = ShuffleConvolutionInputToPoplar(group_count, conv_dims, in);

    kernel = ShuffleConvolutionWeightsToPoplar(conv_dims, kernel, true);

    kernel = AddGroupsDimensionToWeights(params, kernel, true);

    poplar::Tensor out =
        poplin::convolution(graph, in, kernel, params, true, seq, {debug_info},
                            opts, &res.planning_cache);

    out = ShuffleConvolutionOutputToTensorflow(conv_dims, out);

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(out, graph)));

    return seq;
  }
};

REGISTER_POPLAR_OP(ConvWithReverse, Conv2DReverseOp);

class ConvScaledInplaceOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "ConvScaledInplaceOp");
    DriverProgramSequence seq(graph, debug_info);

    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, seq, {debug_info}));
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor weights = inputs[0][0];

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor in,
        FindInstructionInput(tensor_map, res, inst, 1, seq, {debug_info}));

    // Find the deltas tensor
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor deltas,
        FindInstructionInput(tensor_map, res, inst, 2, seq, {debug_info}));

    // Find the scale.
    TF_ASSIGN_OR_RETURN(poplar::Tensor scale,
                        FindInstructionInput(tensor_map, res, inst, 3, seq,
                                             {debug_info}, false));

    TF_ASSIGN_OR_RETURN(poplin::ConvParams params,
                        GetConvolutionParameters(inst, 1, 2));

    TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                        GetConvolutionOptionsForInst(inst, res));

    TF_ASSIGN_OR_RETURN(ConvolutionDimensionNumbers conv_dims,
                        GetConvolutionDims(inst));
    // Get the root of the fusion - it indicates whether this is add or
    // subtract.
    const auto* root_inst = inst->fused_expression_root();
    auto op_type = root_inst->opcode();

    TF_ASSIGN_OR_RETURN(int64_t group_count, GetBatchGroupCount(inst));

    weights = ShuffleConvolutionOutputToPoplar(conv_dims, weights);
    in = ShuffleConvolutionInputToPoplar(group_count, conv_dims, in);

    deltas = ShuffleConvolutionWeightsToPoplar(conv_dims, deltas, false);
    deltas = AddGroupsDimensionToWeights(params, deltas, false);

    auto c_out = poplin::convolution(graph, in, deltas, params, false, seq,
                                     {debug_info}, opts, &res.planning_cache);

    TF_CHECK_OK(ScaledInplaceConstantOrTensor(graph, weights, c_out, scale, seq,
                                              op_type, {debug_info}));

    weights = ShuffleConvolutionOutputToTensorflow(conv_dims, weights);

    TF_CHECK_OK(
        AddOutputTensor(tensor_map, inst, 0, DriverTensor(weights, graph)));

    return seq;
  }
};
REGISTER_POPLAR_OP(Conv_scaled_inplace, ConvScaledInplaceOp);

poplar::OptionFlags GetMultiConvOptions(const HloMultiConvInstruction* inst) {
  poplar::OptionFlags poplar_flags;
  for (const auto& flag : inst->GetOptionFlags()) {
    poplar_flags.set(flag.key, flag.value);
  }
  return poplar_flags;
}

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

  for (int64_t i = 0; i != convolution_specs.size(); ++i) {
    conv_args[i] = {conv_params[i], opts, absl::StrCat(name, "/SubConv", i)};
  }

  return conv_args;
}

class MultiConvOp : public PoplarOpDef {
  StatusOr<DriverTensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "MultiConvOp");
    const int64_t input_index = tensor_target.input_index;
    const HloMultiConvInstruction* inst =
        Cast<HloMultiConvInstruction>(tensor_target.tgt);

    const auto& convolution_specs = inst->GetConvolutionSpecs();
    // Operands [0, n) are inputs and [n, 2n) are kernels.
    const bool is_conv_input = input_index < convolution_specs.size();
    const int64_t conv_index =
        input_index - (is_conv_input ? 0 : convolution_specs.size());
    CHECK_LT(conv_index, convolution_specs.size());
    auto convolution_spec = convolution_specs[conv_index];

    TF_ASSIGN_OR_RETURN(
        std::vector<poplin::multiconv::CreateTensorArgs> create_args,
        GetMultiConvCreateArgs(inst, res, debug_info.getPathName()));

    const poplar::OptionFlags multi_conv_options = GetMultiConvOptions(inst);
    DriverTensor out;
    switch (convolution_spec.type) {
      case ConvType::Conv: {
        if (is_conv_input) {
          out = DriverTensor(poplin::multiconv::createInput(
                                 graph, create_args, conv_index,
                                 multi_conv_options, &res.planning_cache),
                             graph);  /// T32699 add missing debug info
          out = ShuffleConvolutionInputToTensorflow(
              convolution_spec.batch_group_count, convolution_spec.dims, out);
        } else {
          out = DriverTensor(poplin::multiconv::createWeights(
                                 graph, create_args, conv_index,
                                 multi_conv_options, &res.planning_cache),
                             graph);  ///  T32699 add missing debug info
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

  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "MultiConvOp");
    DriverProgramSequence seq(graph, debug_info);

    const HloMultiConvInstruction* multi_conv_inst =
        Cast<HloMultiConvInstruction>(inst);

    TF_ASSIGN_OR_RETURN(
        std::vector<poplin::multiconv::CreateTensorArgs> create_args,
        GetMultiConvCreateArgs(multi_conv_inst, res, debug_info.getPathName()));

    const auto& convolution_specs = multi_conv_inst->GetConvolutionSpecs();
    const poplar::OptionFlags multi_conv_options =
        GetMultiConvOptions(multi_conv_inst);

    std::vector<poplar::Tensor> inputs(convolution_specs.size());
    std::vector<poplar::Tensor> kernels(convolution_specs.size());
    for (int64_t i = 0; i != convolution_specs.size(); ++i) {
      // Find the input tensor.
      TF_ASSIGN_OR_RETURN(inputs[i],
                          FindInstructionInput(tensor_map, res, inst, i, seq,
                                               {debug_info}, false));

      // Find the kernels tensor.
      TF_ASSIGN_OR_RETURN(
          kernels[i], FindInstructionInput(tensor_map, res, inst,
                                           convolution_specs.size() + i, seq,
                                           {debug_info}, false));
    }

    // Check whether we can set transpose_and_flip_weights for all
    // convolutions, and if not, any `ConvWithReverse` needs to do it before
    // the multi conv individually.
    const bool all_transpose_and_flip_weights = absl::c_all_of(
        convolution_specs,
        [](const HloMultiConvInstruction::ConvolutionSpec& convolution_spec) {
          return convolution_spec.type == ConvType::ConvWithReverse;
        });

    std::vector<poplin::multiconv::ConvolutionArgs> flip_args;
    std::vector<poplar::Tensor> flip_weights;
    flip_args.reserve(convolution_specs.size());
    flip_weights.reserve(convolution_specs.size());

    std::vector<poplin::multiconv::ConvolutionArgs> conv_args(
        convolution_specs.size());
    for (std::size_t i = 0; i != convolution_specs.size(); ++i) {
      const auto& convolution_spec = convolution_specs[i];
      poplar::Tensor input = inputs[i];
      poplar::Tensor kernel = kernels[i];

      // Process the inputs, which is dependent on the convolution type.
      switch (convolution_spec.type) {
        case ConvType::Conv: {
          input = ShuffleConvolutionInputToPoplar(
              convolution_spec.batch_group_count, convolution_spec.dims, input);

          kernel = ShuffleConvolutionWeightsToPoplar(convolution_spec.dims,
                                                     kernel, false);
          kernel =
              AddGroupsDimensionToWeights(create_args[i].params, kernel, false);
          break;
        }
        case ConvType::ConvWithReverse: {
          input = ShuffleConvolutionInputToPoplar(
              convolution_spec.batch_group_count, convolution_spec.dims, input);

          kernel = ShuffleConvolutionWeightsToPoplar(convolution_spec.dims,
                                                     kernel, true);
          kernel =
              AddGroupsDimensionToWeights(create_args[i].params, kernel, true);
          if (!all_transpose_and_flip_weights) {
            // Create new kernel and pass it to the
            // multiconv::weightsTransposeChansFlipXY later.
            poplar::Tensor new_kernel = poplin::createWeights(
                graph, create_args[i].params,
                {debug_info, absl::StrCat(i, "/BwdWeights")},
                create_args[i].options, &res.planning_cache);
            flip_args.push_back({input, new_kernel, create_args[i].params,
                                 create_args[i].options});
            flip_weights.push_back(kernel);
          }
          break;
        }
        default: { LOG(FATAL) << "Unknown convolution type."; }
      }
      conv_args[i] = {input, kernel, create_args[i].params,
                      create_args[i].options};
    }

    if (!flip_args.empty()) {
      CHECK_EQ(flip_args.size(), flip_weights.size());
      poplin::multiconv::weightsTransposeChansFlipXY(
          graph, flip_args, flip_weights, seq, {}, debug_info,
          &res.planning_cache);

      for (std::size_t i = 0, flip_index = 0; i != convolution_specs.size();
           ++i) {
        const auto& convolution_spec = convolution_specs[i];
        if (convolution_spec.type == ConvType::ConvWithReverse) {
          // Update weights for the future multiconvolution.
          conv_args[i].weights = flip_args.at(flip_index++).weights;
        }
      }
    }

    std::vector<poplar::Tensor> outputs = poplin::multiconv::convolution(
        graph, conv_args, all_transpose_and_flip_weights, seq, {debug_info},
        multi_conv_options, &res.planning_cache);

    for (int64_t i = 0; i != convolution_specs.size(); ++i) {
      poplar::Tensor output = outputs[i];
      const auto& convolution_spec = convolution_specs[i];
      // Process the outputs, which is dependent on the convolution type.
      switch (convolution_spec.type) {
        case ConvType::Conv:
        case ConvType::ConvWithReverse: {
          output = ShuffleConvolutionOutputToTensorflow(convolution_spec.dims,
                                                        output);
          break;
        }
        default: { LOG(FATAL) << "Unknown convolution type."; }
      }

      outputs[i] = output;
    }

    // Set the outputs.
    for (int64_t i = 0; i != convolution_specs.size(); ++i) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i,
                                  DriverTensor(outputs[i], graph)));
    }

    return seq;
  }
};
REGISTER_POPLAR_OP(MultiConv, MultiConvOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
