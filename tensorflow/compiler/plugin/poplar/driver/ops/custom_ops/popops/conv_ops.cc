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

    TF_ASSIGN_OR_RETURN(int64 group_count, GetBatchGroupCount(inst));
    TF_ASSIGN_OR_RETURN(ConvolutionDimensionNumbers conv_dims,
                        GetConvolutionDims(inst));

    poplar::DebugNameAndId debug_name_and_id(debug_info);
    auto func = [&graph, &res, params, opts, group_count, conv_dims,
                 debug_name_and_id](std::vector<poplar::Tensor>& args,
                                    poplar::program::Sequence& prog) {
      poplar::Tensor in_f = args[0];
      poplar::Tensor kernel_f = args[1];

      in_f = ShuffleConvolutionInputToPoplar(group_count, conv_dims, in_f);

      kernel_f = ShuffleConvolutionWeightsToPoplar(conv_dims, kernel_f, false);

      kernel_f = AddGroupsDimensionToWeights(params, kernel_f, false);

      poplar::Tensor out_f =
          poplin::convolution(graph, in_f, kernel_f, params, false, prog,
                              {debug_name_and_id}, opts, &res.planning_cache);

      out_f = ShuffleConvolutionOutputToTensorflow(conv_dims, out_f);

      args[2] = out_f;
    };

    poplar::Tensor out;
    std::vector<poplar::Tensor> args = {in, kernel, out};
    poputil::graphfn::Signature signature = {
        poputil::graphfn::input(in, "in"),
        poputil::graphfn::input(kernel, "kernel"),
        poputil::graphfn::created("out"),
    };

    TF_RETURN_IF_ERROR(res.graph_cache.ExecuteCached(
        inst, graph, res, seq, func, signature, args, {0, 1}, {}));

    out = args[2];

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(out, graph)));

    return seq;
  }

  StatusOr<DriverTensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "Conv2DOp");
    const int64 input_index = tensor_target.input_index;

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

    TF_ASSIGN_OR_RETURN(int64 group_count, GetBatchGroupCount(inst));
    TF_ASSIGN_OR_RETURN(ConvolutionDimensionNumbers conv_dims,
                        GetConvolutionDims(inst));

    poplar::DebugNameAndId debug_name_and_id(debug_info);
    auto func = [&graph, &res, params, opts, group_count, conv_dims,
                 debug_name_and_id](std::vector<poplar::Tensor>& args,
                                    poplar::program::Sequence& prog) {
      poplar::Tensor in_f = args[0];
      poplar::Tensor kernel_f = args[1];

      in_f = ShuffleConvolutionInputToPoplar(group_count, conv_dims, in_f);

      kernel_f = ShuffleConvolutionWeightsToPoplar(conv_dims, kernel_f, true);

      kernel_f = AddGroupsDimensionToWeights(params, kernel_f, true);

      poplar::Tensor out_f =
          poplin::convolution(graph, in_f, kernel_f, params, true, prog,
                              {debug_name_and_id}, opts, &res.planning_cache);

      out_f = ShuffleConvolutionOutputToTensorflow(conv_dims, out_f);

      args[2] = out_f;
    };

    poplar::Tensor out;
    std::vector<poplar::Tensor> args = {in, kernel, out};
    poputil::graphfn::Signature signature = {
        poputil::graphfn::input(in, "in"),
        poputil::graphfn::input(kernel, "kernel"),
        poputil::graphfn::created("out"),
    };

    TF_RETURN_IF_ERROR(res.graph_cache.ExecuteCached(inst, graph, res, seq,
                                                     func, signature, args));

    out = args[2];

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
    poplar::Tensor arg_weights = inputs[0][0];

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_in,
        FindInstructionInput(tensor_map, res, inst, 1, seq, {debug_info}));

    // Find the deltas tensor
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_deltas,
        FindInstructionInput(tensor_map, res, inst, 2, seq, {debug_info}));

    // Find the scale.
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_scale,
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

    TF_ASSIGN_OR_RETURN(int64 group_count, GetBatchGroupCount(inst));
    poplar::DebugNameAndId debug_name_and_id(debug_info);
    auto func = [&graph, &res, params, opts, group_count, conv_dims, op_type,
                 inst, debug_name_and_id](std::vector<poplar::Tensor>& args,
                                          poplar::program::Sequence& prog) {
      poplar::Tensor weights = args[0];
      poplar::Tensor in = args[1];
      poplar::Tensor deltas = args[2];
      poplar::Tensor scale = args[3];

      weights = ShuffleConvolutionOutputToPoplar(conv_dims, weights);
      in = ShuffleConvolutionInputToPoplar(group_count, conv_dims, in);

      deltas = ShuffleConvolutionWeightsToPoplar(conv_dims, deltas, false);
      deltas = AddGroupsDimensionToWeights(params, deltas, false);

      auto c_out =
          poplin::convolution(graph, in, deltas, params, false, prog,
                              {debug_name_and_id}, opts, &res.planning_cache);

      TF_CHECK_OK(ScaledInplaceConstantOrTensor(
          graph, weights, c_out, scale, prog, op_type, {debug_name_and_id}));

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

    TF_CHECK_OK(
        AddOutputTensor(tensor_map, inst, 0, DriverTensor(arg_weights, graph)));

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

  for (int64 i = 0; i != convolution_specs.size(); ++i) {
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

    poplar::DebugNameAndId debug_name_and_id(debug_info);
    auto func = [&graph, &res, create_args, convolution_specs,
                 multi_conv_options,
                 debug_name_and_id](std::vector<poplar::Tensor>& args,
                                    poplar::program::Sequence& prog) -> void {
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
        poplar::Tensor input = args[i];
        poplar::Tensor kernel = args[i + convolution_specs.size()];

        // Process the inputs, which is dependent on the convolution type.
        switch (convolution_spec.type) {
          case ConvType::Conv: {
            input = ShuffleConvolutionInputToPoplar(
                convolution_spec.batch_group_count, convolution_spec.dims,
                input);

            kernel = ShuffleConvolutionWeightsToPoplar(convolution_spec.dims,
                                                       kernel, false);
            kernel = AddGroupsDimensionToWeights(create_args[i].params, kernel,
                                                 false);
            break;
          }
          case ConvType::ConvWithReverse: {
            input = ShuffleConvolutionInputToPoplar(
                convolution_spec.batch_group_count, convolution_spec.dims,
                input);

            kernel = ShuffleConvolutionWeightsToPoplar(convolution_spec.dims,
                                                       kernel, true);
            kernel = AddGroupsDimensionToWeights(create_args[i].params, kernel,
                                                 true);
            if (!all_transpose_and_flip_weights) {
              // Create new kernel and pass it to the
              // multiconv::weightsTransposeChansFlipXY later.
              poplar::Tensor new_kernel = poplin::createWeights(
                  graph, create_args[i].params,
                  {debug_name_and_id, absl::StrCat(i, "/BwdWeights")},
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
            graph, flip_args, flip_weights, prog, {}, debug_name_and_id,
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
          graph, conv_args, all_transpose_and_flip_weights, prog,
          {debug_name_and_id}, multi_conv_options, &res.planning_cache);

      for (int64 i = 0; i != convolution_specs.size(); ++i) {
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
      TF_ASSIGN_OR_RETURN(args[i],
                          FindInstructionInput(tensor_map, res, inst, i, seq,
                                               {debug_info}, false));
      inputs_signature.push_back(
          poputil::graphfn::input(args[i], absl::StrCat("Input", i)));

      // Find the kernels tensor.
      TF_ASSIGN_OR_RETURN(args[convolution_specs.size() + i],
                          FindInstructionInput(tensor_map, res, inst,
                                               convolution_specs.size() + i,
                                               seq, {debug_info}, false));
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
        multi_conv_inst->AllocatingIndices(),
        multi_conv_inst->LayoutDependencies()));

    // Set the outputs.
    for (int64 i = 0; i != convolution_specs.size(); ++i) {
      TF_CHECK_OK(AddOutputTensor(
          tensor_map, inst, i,
          DriverTensor(args[2 * convolution_specs.size() + i], graph)));
    }

    return seq;
  }
};
REGISTER_POPLAR_OP(MultiConv, MultiConvOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
