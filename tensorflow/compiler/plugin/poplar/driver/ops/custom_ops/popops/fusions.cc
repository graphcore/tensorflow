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
#include <poplin/MatMul.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Pad.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <poputil/TileMapping.hpp>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/all_gather.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/reduction_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_arithmetic_expr.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"
using ::absl::StrCat;

namespace xla {
namespace poplarplugin {

static StatusOr<poplar::Tensor> ReversePathTransform(
    poplar::Graph& graph, poplar::Tensor in,
    const std::vector<const HloInstruction*>& path,
    const poplar::DebugNameAndId& debug_name_and_id) {
  // Now apply any transformations required by the path from the source to
  // the target

  for (auto i = path.rbegin(); i != path.rend(); ++i) {
    auto& inst = *i;
    switch (inst->opcode()) {
      case HloOpcode::kTranspose: {
        auto optional_permutation =
            convert_array<std::vector<unsigned>>(inst->dimensions());
        if (!optional_permutation) {
          return xla::FailedPrecondition(
              "PathTransform - cannot cast permutation.");
        }
        std::vector<unsigned> permutation = *optional_permutation;
        std::vector<unsigned> shuffle(permutation.size());
        for (unsigned int d = 0; d < permutation.size(); d++) {
          shuffle[d] = permutation[d];
        }
        in = in.dimShuffle(shuffle);
        break;
      }
      case HloOpcode::kReshape: {
        std::vector<size_t> dims(PoplarShapeFromXlaShape(inst->shape()));
        in = in.reshape(dims);
        break;
      }
      case HloOpcode::kConvert: {
        TF_ASSIGN_OR_RETURN(auto poplar_type, PoplarDataType(inst->shape()));
        in = graph.clone(poplar_type, in, {debug_name_and_id});
        break;
      }
      case HloOpcode::kConcatenate: {
        return xla::FailedPrecondition(
            "ReversePathTransform - Concatenante not supported");
      }
      case HloOpcode::kPad: {
        return xla::FailedPrecondition(
            "ReversePathTransform - Pad not supported.");
      }
      case HloOpcode::kFusion: {
        if (IsPopOpsFusion(inst, "zero_pad")) {
          return xla::FailedPrecondition(
              "ReversePathTransform - 'zero_pad' Fusion not supported.");
        }
      }
      default: {
        // All other instructions in the path do not modify the shape
        break;
      }
    }
  }

  return in;
}

class ConvBiasAddOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "ConvBiasAddOp");
    DriverProgramSequence prog(graph, {debug_info});

    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, prog, debug_info));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    auto in = inputs[0][0];

    TF_ASSIGN_OR_RETURN(poplar::Tensor bias,
                        FindInstructionInput(tensor_map, res, inst, 1, prog,
                                             {debug_info}, false));

    const auto* conv_op = LookThroughCopies(inst->operand(0));
    TF_ASSIGN_OR_RETURN(poplar::Tensor shuffled_in,
                        ShuffleConvolutionOutputToPoplar(conv_op, in));

    poplin::addBias(graph, shuffled_in, bias, prog, {debug_info});

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, in));
    return prog;
  }

  // We want the accumulation tensor to be the same layout as the input tensor.
  StatusOr<DriverTensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "ConvBiasAddOp");
    const int64 input_index = tensor_target.input_index;
    const HloInstruction* layout = *tensor_target.layout;

    const auto layout_output_idx = *tensor_target.layout_output_idx;
    const auto forward_path = tensor_target.forward_path;
    const HloInstruction* inst = tensor_target.tgt;

    TF_ASSIGN_OR_RETURN(TensorVector outputs,
                        FindInstructionOutputTensors(tensor_map, res, layout));

    if (layout_output_idx < 0 || outputs.size() <= layout_output_idx) {
      return xla::FailedPrecondition("Convolution %s output not found for %s",
                                     layout->name(), GetDebugName(layout));
    }

    poplar::Tensor acts = outputs[layout_output_idx];

    TF_ASSIGN_OR_RETURN(acts, ShuffleConvolutionOutputToPoplar(layout, acts));
    TF_ASSIGN_OR_RETURN(
        acts, ReversePathTransform(graph, acts, forward_path, {debug_info}));

    return DriverTensor(
        poplin::createBiases(graph, acts, {debug_info, "biases"}), graph);
  }
};

REGISTER_POPLAR_OP(Conv_biasadd, ConvBiasAddOp);

class MatMulBiasAddOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "MatMulBiasAddOp");

    // Get the broadcast instruction which is required to get the bias size.
    const HloInstruction* root = inst->fused_expression_root();
    const HloInstruction* broadcast = root->operand(1);
    CHECK_EQ(broadcast->opcode(), HloOpcode::kBroadcast);

    DriverProgramSequence prog(graph, debug_info);

    TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                        FindInplaceOutputTensors(tensor_map, res, inst, prog,
                                                 debug_info, false));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    auto in = inputs[0][0];

    TF_ASSIGN_OR_RETURN(auto bias,
                        FindInstructionInput(tensor_map, res, inst, 1, prog,
                                             {debug_info}, false));

    TF_ASSIGN_OR_RETURN(bias, BroadcastTensor(bias, broadcast->shape(),
                                              broadcast->dimensions()));
    popops::addInPlace(graph, in, bias.getPoplarTensor(), prog, {debug_info});

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(in, graph)));
    return prog;
  }

  // We want the accumulation tensor to be the same layout as the input tensor.
  StatusOr<DriverTensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "MatMulBiasAddOp");
    const HloInstruction* layout = *tensor_target.layout;

    const auto layout_output_idx = *tensor_target.layout_output_idx;
    const auto forward_path = tensor_target.forward_path;
    const HloInstruction* inst = tensor_target.tgt;

    TF_ASSIGN_OR_RETURN(TensorVector outputs,
                        FindInstructionOutputTensors(tensor_map, res, layout));

    if (layout_output_idx < 0 || outputs.size() <= layout_output_idx) {
      return xla::FailedPrecondition("Matmul %s output not found for %s",
                                     layout->name(), name);
    }

    poplar::Tensor acts = outputs[layout_output_idx];
    TF_ASSIGN_OR_RETURN(
        acts, ReversePathTransform(graph, acts, forward_path, {debug_info}));

    // Flatten activations into 2D.
    acts = acts.flatten(0, acts.rank() - 1);
    return DriverTensor(poputil::createBroadcastOperand(
                            graph, acts, acts.elementType(), acts.rank() - 1,
                            /*ditherMapping*/ false, {debug_info, "biases"}),
                        graph);
  }
};

REGISTER_POPLAR_OP(Matmul_biasadd, MatMulBiasAddOp);

class BiasApplyOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "BiasApplyOp");
    DriverProgramSequence seq(graph, debug_info);

    const HloInstruction* root = inst->fused_expression_root();

    // Find the biases.
    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, seq, debug_info));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor biases = inputs[0][0];

    // Find the deltas.
    TF_ASSIGN_OR_RETURN(poplar::Tensor deltas,
                        FindInstructionInput(tensor_map, res, inst, 1, seq,
                                             {debug_info}, false));
    // Find the scale.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor scale,
        FindInstructionInput(tensor_map, res, inst, 2, seq, {debug_info}));

    // Find reduction dimensions
    const auto* reduce = root->operand(1)->operand(0);
    std::vector<std::size_t> reduction_dims;
    for (auto d : reduce->dimensions()) {
      reduction_dims.push_back(d);
    }

    poplar::DebugNameAndId debug_name_and_id(debug_info);
    auto func = [&graph, reduction_dims, debug_name_and_id](
                    std::vector<poplar::Tensor>& args,
                    poplar::program::Sequence& prog) {
      poplar::Tensor scale_float = args[2];
      if (scale_float.elementType() != poplar::FLOAT) {
        scale_float = popops::cast(graph, scale_float, poplar::FLOAT, prog,
                                   {debug_name_and_id, "ScaleToFloat"});
      }
      // Reduce with scale and update in place
      popops::mapInPlace(graph, popops::expr::UnaryOpType::NEGATE, scale_float,
                         prog, {debug_name_and_id, "negate"});
      popops::reduceWithOutput(graph, args[1], args[0], reduction_dims,
                               {popops::Operation::ADD, true, scale_float},
                               prog, {debug_name_and_id});
    };

    // Depending on whether this is performed inplace or not, the output could
    // be a new tensor or the biases tensor.
    std::vector<poplar::Tensor> args = {biases, deltas, scale};
    poputil::graphfn::Signature signature = {
        poputil::graphfn::inout(biases, "biases"),
        poputil::graphfn::input(deltas, "deltas"),
        poputil::graphfn::input(scale, "scale")};

    TF_RETURN_IF_ERROR(res.graph_cache.ExecuteCached(inst, graph, res, seq,
                                                     func, signature, args));

    TF_CHECK_OK(
        AddOutputTensor(tensor_map, inst, 0, DriverTensor(args[0], graph)));

    return seq;
  }
};

REGISTER_POPLAR_OP(Bias_apply, BiasApplyOp);

class ZeroPadOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "ZeroPadOp");
    DriverProgramSequence seq(graph, debug_info);
    const HloInstruction* root = inst->fused_expression_root();
    const PaddingConfig& cfg(root->padding_config());

    TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                        FindInplaceOutputTensors(tensor_map, res, inst, seq,
                                                 debug_info, false));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor in = inputs[0][0];

    std::vector<std::ptrdiff_t> paddingLower;
    std::vector<std::ptrdiff_t> paddingUpper;
    for (auto& d : cfg.dimensions()) {
      paddingLower.push_back(d.edge_padding_low());
      paddingUpper.push_back(d.edge_padding_high());
    }
    poplar::Tensor zero =
        graph.addConstant(in.elementType(), {}, 0, {debug_info, "ZeroPad"});
    graph.setTileMapping(zero, 0);
    poplar::Tensor out =
        popops::pad(graph, in, paddingLower, paddingUpper, zero);

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(out, graph)));
    return seq;
  }
};

REGISTER_POPLAR_OP(Zero_pad, ZeroPadOp);

class PaddingReduceWindowOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "PaddingReduceWindowOp");
    DriverProgramSequence seq(graph, debug_info);

    const HloInstruction* root = inst->fused_expression_root();
    const Window& window(root->window());

    TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                        FindInplaceOutputTensors(tensor_map, res, inst, seq,
                                                 debug_info, false));
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor in = inputs[0][0];
    CHECK_EQ(inputs[1].size(), 1);
    poplar::Tensor init_val = inputs[1][0];

    std::vector<std::ptrdiff_t> paddingLower;
    std::vector<std::ptrdiff_t> paddingUpper;
    for (auto& d : window.dimensions()) {
      paddingLower.push_back(d.padding_low());
      paddingUpper.push_back(d.padding_high());
    }

    poplar::Tensor out =
        popops::pad(graph, in, paddingLower, paddingUpper, init_val);

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(out, graph)));
    return seq;
  }
};

REGISTER_POPLAR_OP(Padding_reduce_window, PaddingReduceWindowOp)

class ReductionFp16InputOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "ReductionFp16InputOp");
    const HloInstruction* reduce_inst = inst->fused_expression_root();
    TF_ASSIGN_OR_RETURN(
        DriverProgramSequence prog,
        CreateSimpleReduction(res, inst, reduce_inst, output_shape, tensor_map,
                              /*with_scale=*/false, {debug_info}));
    return prog;
  }
};

REGISTER_POPLAR_OP(Reduction_fp16_input, ReductionFp16InputOp)

class ReductionSquareAddOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "ReductionSquareAddOp");
    const bool with_scale = inst->operand_count() == 3;
    TF_ASSIGN_OR_RETURN(const HloInstruction* reduce_inst,
                        GetReduceInstruction(inst));
    TF_ASSIGN_OR_RETURN(
        DriverProgramSequence prog,
        CreateSimpleReduction(res, popops::Operation::SQUARE_ADD, inst,
                              reduce_inst, output_shape, tensor_map, with_scale,
                              {debug_info}));
    return prog;
  }
};

REGISTER_POPLAR_OP(Reduction_square_add, ReductionSquareAddOp)

class ReductionScaledOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "ReductionScaledOp");
    TF_ASSIGN_OR_RETURN(const HloInstruction* reduce_inst,
                        GetReduceInstruction(inst));
    TF_ASSIGN_OR_RETURN(
        DriverProgramSequence prog,
        CreateSimpleReduction(res, inst, reduce_inst, output_shape, tensor_map,
                              /*with_scale=*/true, {debug_info}));
    return prog;
  }
};

REGISTER_POPLAR_OP(Reduction_scaled, ReductionScaledOp)

class ArithemticExpressionOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "ArithemticExpressionOp");
    DriverProgramSequence seq(graph, debug_info);

    // Get all the inputs.
    TensorVectors args;
    for (int64 i = 0; i < inst->operand_count(); i++) {
      TF_ASSIGN_OR_RETURN(
          TensorVector t,
          FindInstructionInputTensors(tensor_map, res, inst, i, seq, debug_info,
                                      /*expand_aliasing=*/false));
      args.push_back(t);
    }

    // Evaluate the expression.
    const HloComputation* comp = inst->fused_instructions_computation();
    ArithmeticExprVisitor arithmetic_visitor(res, args, inst, debug_info);
    TF_RETURN_IF_ERROR(comp->Accept(&arithmetic_visitor));
    seq.add(arithmetic_visitor.GetSequence(graph));

    for (size_t i = 0; i < arithmetic_visitor.outputs().size(); i++) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i,
                                  arithmetic_visitor.outputs()[i]));
    }
    return seq;
  }
};

REGISTER_POPLAR_OP(Arithmetic_expression, ArithemticExpressionOp)

}  // namespace poplarplugin
}  // namespace xla
