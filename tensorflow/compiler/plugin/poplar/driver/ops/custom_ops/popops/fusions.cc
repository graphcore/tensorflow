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
#include <popnn/NonLinearity.hpp>
#include <popops/Cast.hpp>
#include <popops/Collectives.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Pad.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/all_gather.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {

static StatusOr<poplar::Tensor> ReversePathTransform(
    poplar::Graph& graph, poplar::Tensor in,
    const std::vector<const HloInstruction*>& path) {
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
        in = graph.clone(poplar_type, in, GetDebugName(inst));
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

class WideConstOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;

    const HloInstruction* root = inst->fused_expression_root();

    const HloInstruction* constant = root->operand(0);
    CHECK_EQ(constant->opcode(), HloOpcode::kConstant);
    const Literal& constant_literal = constant->literal();

    // Allocate the constant first.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor constant_tensor,
        AddConstantTensor(graph, TensorLocation{constant, 0}, constant->shape(),
                          constant_literal, res, tensor_map));

    // Broadcast the tensor to the right shape.
    TF_ASSIGN_OR_RETURN(poplar::Tensor out,
                        BroadcastTensor(constant_tensor, output_shape, {}));
    // For wide constants, check if they have an allocation target, if so then
    // allocate the tensor with that target and copy the constant to that
    // layout.
    TensorLocation src = TensorLocation{inst, 0};
    if (HasTensorAllocationTarget(src, res)) {
      // Doing this copy rather than allocating a big constant and calling
      // setInitialValue is a trade off between having a large tensor always
      // live and a copy + a scalar constant always being live.
      TF_ASSIGN_OR_RETURN(poplar::Tensor layout,
                          AddTensor(graph, src, output_shape, res, tensor_map));
      seq.add(poplar::program::Copy(out, layout));
      out = layout;
    }
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

    return seq;
  }
};

REGISTER_POPLAR_OP(Wide_const, WideConstOp);

class ConvBiasAddOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence prog;

    TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                        FindInplaceOutputTensors(tensor_map, res, inst, prog));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor in = inputs[0][0];

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor bias,
        FindInstructionInput(tensor_map, res, inst, 1, prog, false));

    const auto* conv_op = GetOperandLookThroughInterIpuCopy(inst, 0);
    poplar::Tensor shuffled_in = ShuffleConvolutionOutputToPoplar(conv_op, in);

    poplin::addBias(graph, shuffled_in, bias, prog, GetDebugName(inst));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, in));
    return prog;
  }

  // We want the accumulation tensor to be the same layout as the input tensor.
  StatusOr<poplar::Tensor> Allocator(poplar::Graph& graph,
                                     CompilerResources& res,
                                     const std::string& name,
                                     const TensorTarget& tensor_target,
                                     const TensorMap& tensor_map) override {
    const int64 input_index = tensor_target.input_index;
    const HloInstruction* layout = *tensor_target.layout;

    const auto layout_output_idx = *tensor_target.layout_output_idx;
    const auto forward_path = tensor_target.forward_path;
    const HloInstruction* inst = tensor_target.tgt;

    TensorVector outputs = FindInstructionOutputs(tensor_map, res, layout);

    if (layout_output_idx < 0 || outputs.size() <= layout_output_idx) {
      return xla::FailedPrecondition("Convolution %s output not found for %s",
                                     layout->name(), GetDebugName(layout));
    }

    poplar::Tensor acts = outputs[layout_output_idx];

    acts = ShuffleConvolutionOutputToPoplar(layout, acts);
    TF_ASSIGN_OR_RETURN(acts, ReversePathTransform(graph, acts, forward_path));

    return poplin::createBiases(graph, acts, GetDebugName(inst));
  }
};

REGISTER_POPLAR_OP(Conv_biasadd, ConvBiasAddOp);

class MatMulBiasAddOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    // Get the broadcast instruction which is required to get the bias size.
    const HloInstruction* root = inst->fused_expression_root();
    const HloInstruction* broadcast = root->operand(1);
    CHECK_EQ(broadcast->opcode(), HloOpcode::kBroadcast);

    poplar::program::Sequence prog;

    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, prog, false));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor in = inputs[0][0];

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor bias,
        FindInstructionInput(tensor_map, res, inst, 1, prog, false));

    TF_ASSIGN_OR_RETURN(bias, BroadcastTensor(bias, broadcast->shape(),
                                              broadcast->dimensions()));
    popops::addInPlace(graph, in, bias, prog, GetDebugName(inst));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, in));
    return prog;
  }

  // We want the accumulation tensor to be the same layout as the input tensor.
  StatusOr<poplar::Tensor> Allocator(poplar::Graph& graph,
                                     CompilerResources& res,
                                     const std::string& name,
                                     const TensorTarget& tensor_target,
                                     const TensorMap& tensor_map) override {
    const HloInstruction* layout = *tensor_target.layout;

    const auto layout_output_idx = *tensor_target.layout_output_idx;
    const auto forward_path = tensor_target.forward_path;
    const HloInstruction* inst = tensor_target.tgt;

    TensorVector outputs = FindInstructionOutputs(tensor_map, res, layout);

    if (layout_output_idx < 0 || outputs.size() <= layout_output_idx) {
      return xla::FailedPrecondition("Matmul %s output not found for %s",
                                     layout->name(), name);
    }

    poplar::Tensor acts = outputs[layout_output_idx];
    TF_ASSIGN_OR_RETURN(acts, ReversePathTransform(graph, acts, forward_path));

    // Get a slice representing the last dimension.
    const std::vector<size_t> index(acts.rank() - 1);
    return graph.clone(acts.index(index), StrCat(name, "/biases"));
  }
};

REGISTER_POPLAR_OP(Matmul_biasadd, MatMulBiasAddOp);

class BiasApplyOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;

    const HloInstruction* root = inst->fused_expression_root();

    // Find the biases.
    TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                        FindInplaceOutputTensors(tensor_map, res, inst, seq));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor biases = inputs[0][0];

    // Find the deltas.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor deltas,
        FindInstructionInput(tensor_map, res, inst, 1, seq, false));
    // Find the scale.
    TF_ASSIGN_OR_RETURN(poplar::Tensor scale,
                        FindInstructionInput(tensor_map, res, inst, 2, seq));

    // Find reduction dimensions
    const auto* reduce = root->operand(1)->operand(0);
    std::vector<std::size_t> reduction_dims;
    for (auto d : reduce->dimensions()) {
      reduction_dims.push_back(d);
    }

    const std::string debug_prefix = GetDebugName(inst);
    auto func = [&graph, reduction_dims, debug_prefix](
                    std::vector<poplar::Tensor>& args,
                    poplar::program::Sequence& prog) {
      poplar::Tensor scale_float = args[2];
      if (scale_float.elementType() != poplar::FLOAT) {
        scale_float = popops::cast(graph, scale_float, poplar::FLOAT, prog,
                                   debug_prefix + "/ScaleToFloat");
      }
      // Reduce with scale and update in place
      popops::mapInPlace(graph, popops::expr::UnaryOpType::NEGATE, scale_float,
                         prog, debug_prefix + "/negate");
      popops::reduceWithOutput(graph, args[1], args[0], reduction_dims,
                               {popops::Operation::ADD, true, scale_float},
                               prog, debug_prefix);
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

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, args[0]));

    return seq;
  }
};

REGISTER_POPLAR_OP(Bias_apply, BiasApplyOp);

class ZeroPadOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;
    const HloInstruction* root = inst->fused_expression_root();
    const PaddingConfig& cfg(root->padding_config());

    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, seq, false));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor in = inputs[0][0];

    std::vector<std::ptrdiff_t> paddingLower;
    std::vector<std::ptrdiff_t> paddingUpper;
    for (auto& d : cfg.dimensions()) {
      paddingLower.push_back(d.edge_padding_low());
      paddingUpper.push_back(d.edge_padding_high());
    }
    poplar::Tensor zero = graph.addConstant(in.elementType(), {}, 0,
                                            GetDebugName(inst) + "/ZeroPad");
    graph.setTileMapping(zero, 0);
    poplar::Tensor out =
        popops::pad(graph, in, paddingLower, paddingUpper, zero);

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
    return seq;
  }
};

REGISTER_POPLAR_OP(Zero_pad, ZeroPadOp);

class ScaledInplaceabYOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;
    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, seq, false));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor in0 = inputs[0][0];

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor in1,
        FindInstructionInput(tensor_map, res, inst, 1, seq, false));

    const auto* root_inst = inst->fused_expression_root();

    if (inst->operand_count() == 2) {
      const auto* const_inst = root_inst->operand(1)->operand(1)->operand(0);
      CHECK_EQ(const_inst->opcode(), HloOpcode::kConstant);
      // Get the scalar multiplier
      TF_ASSIGN_OR_RETURN(double scale, LiteralScalarToNativeType<double>(
                                            const_inst->literal()));

      TF_CHECK_OK(ScaledInplaceConstantOrTensor(graph, in0, in1, scale, seq,
                                                root_inst->opcode(),
                                                GetDebugName(inst)));
    } else if (inst->operand_count() == 3) {
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor scale,
          FindInstructionInput(tensor_map, res, inst, 2, seq, false));
      TF_CHECK_OK(ScaledInplaceConstantOrTensor(graph, in0, in1, scale, seq,
                                                root_inst->opcode(),
                                                GetDebugName(inst)));
    } else {
      return xla::FailedPrecondition("Unsupported use of scaled inplace op: %s",
                                     root_inst->name().c_str());
    }

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, in0));
    return seq;
  }
};
REGISTER_POPLAR_OP(Scaled_inplace_aby, ScaledInplaceabYOp);

class ScaledInplaceaXbOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;
    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, seq, false));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor in0 = inputs[0][0];

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor in1,
        FindInstructionInput(tensor_map, res, inst, 1, seq, false));

    const auto* root_inst = inst->fused_expression_root();

    if (inst->operand_count() == 2) {
      const auto* const_inst = root_inst->operand(0)->operand(1)->operand(0);
      CHECK_EQ(const_inst->opcode(), HloOpcode::kConstant);
      // Get the scalar multiplier
      TF_ASSIGN_OR_RETURN(double scale, LiteralScalarToNativeType<double>(
                                            const_inst->literal()));

      TF_CHECK_OK(ScaledInplaceConstantOrTensor(graph, in0, scale, in1, 1.0,
                                                seq, root_inst->opcode(),
                                                GetDebugName(inst)));
    } else if (inst->operand_count() == 3) {
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor scale,
          FindInstructionInput(tensor_map, res, inst, 2, seq, false));

      const Shape& scalar_shape = inst->operand(2)->shape();
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor one,
          CreateConstantTensor(
              graph, LiteralUtil::One(scalar_shape.element_type()),
              scalar_shape, in1.elementType(), GetDebugName(inst) + "/One"));

      TF_CHECK_OK(ScaledInplaceConstantOrTensor(graph, in0, scale, in1, one,
                                                seq, root_inst->opcode(),
                                                GetDebugName(inst)));
    } else {
      return xla::FailedPrecondition("Unsupported use of scaled inplace op: %s",
                                     root_inst->name().c_str());
    }

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, in0));
    return seq;
  }
};
REGISTER_POPLAR_OP(Scaled_inplace_axb, ScaledInplaceaXbOp);

class ScaledInplaceaXbYOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;
    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, seq, true));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor in0 = inputs[0][0];

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor in1,
        FindInstructionInput(tensor_map, res, inst, 1, seq, false));

    const auto* root_inst = inst->fused_expression_root();

    if (inst->operand_count() == 2) {
      const auto* const_inst_a = root_inst->operand(0)->operand(1)->operand(0);
      CHECK_EQ(const_inst_a->opcode(), HloOpcode::kConstant);
      // Get the scalar multiplier
      TF_ASSIGN_OR_RETURN(double scale_a, LiteralScalarToNativeType<double>(
                                              const_inst_a->literal()));

      const auto* const_inst_b = root_inst->operand(1)->operand(1)->operand(0);
      CHECK_EQ(const_inst_b->opcode(), HloOpcode::kConstant);
      // Get the scalar multiplier
      TF_ASSIGN_OR_RETURN(double scale_b, LiteralScalarToNativeType<double>(
                                              const_inst_b->literal()));

      TF_CHECK_OK(ScaledInplaceConstantOrTensor(
          graph, in0, scale_a, in1, scale_b, seq, root_inst->opcode(),
          GetDebugName(inst)));
    } else if (inst->operand_count() == 4) {
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor scale_a,
          FindInstructionInput(tensor_map, res, inst, 2, seq, false));

      TF_ASSIGN_OR_RETURN(
          poplar::Tensor scale_b,
          FindInstructionInput(tensor_map, res, inst, 3, seq, false));

      TF_CHECK_OK(ScaledInplaceConstantOrTensor(
          graph, in0, scale_a, in1, scale_b, seq, root_inst->opcode(),
          GetDebugName(inst)));
    } else {
      return xla::FailedPrecondition("Unsupported, aXbY scaled inplace op: %s",
                                     root_inst->name().c_str());
    }

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, in0));
    return seq;
  }
};
REGISTER_POPLAR_OP(Scaled_inplace_axby, ScaledInplaceaXbYOp);

class PaddingReduceWindowOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;

    const HloInstruction* root = inst->fused_expression_root();
    const Window& window(root->window());

    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, seq, false));
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

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
    return seq;
  }
};

REGISTER_POPLAR_OP(Padding_reduce_window, PaddingReduceWindowOp)

class ReductionFp16InputOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    const HloInstruction* reduce_inst = inst->fused_expression_root();
    TF_ASSIGN_OR_RETURN(poplar::program::Program prog,
                        CreateSimpleReduction(res, inst, reduce_inst,
                                              output_shape, tensor_map));
    return prog;
  }
};

REGISTER_POPLAR_OP(Reduction_fp16_input, ReductionFp16InputOp)

}  // namespace poplarplugin
}  // namespace xla
