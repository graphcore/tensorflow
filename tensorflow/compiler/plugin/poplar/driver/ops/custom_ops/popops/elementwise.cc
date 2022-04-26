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
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/TileMapping.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops/expression_helpers.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_instruction_extensions.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {
class UnaryElementwiseOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "UnaryElementwiseOp");
    DriverProgramSequence seq(graph, debug_info);
    TF_ASSIGN_OR_RETURN(auto expression_inputs,
                        helper::GetElementwiseInputs(res, inst, {0}, tensor_map,
                                                     seq, {debug_info}));
    auto input_tensors =
        helper::GetTensorsFromExpressionInputs(expression_inputs);

    auto operation = helper::GetElementwiseOp(inst);
    TF_ASSIGN_OR_RETURN(popops::expr::UnaryOpType op, LookupUnaryFn(operation));
    auto expr = popops::expr::UnaryOp(op, *expression_inputs[0].expr);

    poplar::Tensor out;
    const bool is_inplace =
        AreInplaceOutputTensorsWritable(tensor_map, res, inst);
    if (is_inplace) {
      popops::mapInPlace(graph, expr, input_tensors, seq, {debug_info});
      out = input_tensors[0];
    } else {
      out = popops::map(graph, expr, input_tensors, seq, {debug_info});
    }

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(out, graph)));

    return seq;
  }
};

void RegisterInplaceExtension(HloOpcode opcode) {
  RegisterHloInstructionExtension<InplaceExtension>(
      opcode, [](const HloInstruction*) {
        return HloPoplarInplaceDescription(
            HloInstructionType::kInplaceReadWrite, /*inplace_operands=*/{0},
            /*allow_non_inplace=*/true);
      });
}

void RegisterInplaceOnOperand0IfTypesMatch(HloOpcode opcode) {
  RegisterHloInstructionExtension<InplaceExtension>(
      opcode, [](const HloInstruction* inst) {
        if (inst->shape().element_type() ==
            inst->operand(0)->shape().element_type()) {
          return HloPoplarInplaceDescription(
              HloInstructionType::kInplaceReadWrite, /*inplace_operands=*/{0},
              /*allow_non_inplace=*/true);
        } else {
          return HloPoplarInplaceDescription();
        }
      });
}

#define REGISTER_INPLACE_HLO_OP(OPCODE, CLS)                     \
  REGISTER_HLO_INST_EXTENSIONS(OPCODE, RegisterInplaceExtension) \
  REGISTER_HLO_OP(OPCODE, CLS)

REGISTER_INPLACE_HLO_OP(kAbs, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kRoundNearestAfz, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kCeil, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kClz, UnaryElementwiseOp);
REGISTER_HLO_OP(kConvert, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kBitcastConvert, UnaryElementwiseOp);
REGISTER_HLO_OP(kCopy, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kCos, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kExp, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kExpm1, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kFloor, UnaryElementwiseOp);
REGISTER_HLO_OP(kImag, UnaryElementwiseOp);
REGISTER_HLO_OP(kIsFinite, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kLog, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kLog1p, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kNot, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kNegate, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kPopulationCount, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kReal, UnaryElementwiseOp);
REGISTER_HLO_OP(kReducePrecision, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kRsqrt, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kSign, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kSin, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kSqrt, UnaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kTanh, UnaryElementwiseOp);
REGISTER_POPLAR_OP(Inverse, UnaryElementwiseOp);
REGISTER_POPLAR_OP(Square, UnaryElementwiseOp);
REGISTER_POPLAR_OP(Erf, UnaryElementwiseOp);
REGISTER_POPLAR_OP(GeluErf, UnaryElementwiseOp);

struct NaryOutput {
  DriverProgramSequence sequence;
  DriverTensor result;
};

class BinaryElementwiseOp : public PoplarOpDef {
 protected:
  static StatusOr<NaryOutput> Compute(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugNameAndId& debug_name_and_id) {
    DriverProgramSequence seq(graph, debug_name_and_id);
    TF_ASSIGN_OR_RETURN(
        auto expression_inputs,
        helper::GetElementwiseInputs(res, inst, {0, 1}, tensor_map, seq,
                                     {debug_name_and_id}));
    auto input_tensors =
        helper::GetTensorsFromExpressionInputs(expression_inputs);

    auto operation = helper::GetElementwiseOp(inst);
    TF_ASSIGN_OR_RETURN(popops::expr::BinaryOpType op,
                        LookupBinaryFn(operation));
    auto expr = popops::expr::BinaryOp(op, *expression_inputs[0].expr,
                                       *expression_inputs[1].expr);

    poplar::Tensor out;
    const bool is_inplace =
        AreInplaceOutputTensorsWritable(tensor_map, res, inst);
    if (is_inplace) {
      popops::mapInPlace(graph, expr, input_tensors, seq, {debug_name_and_id});
      out = input_tensors[0];
    } else {
      out = popops::map(graph, expr, input_tensors, seq, {debug_name_and_id});
    }
    return NaryOutput{seq, DriverTensor(out, graph)};
  }

 public:
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "BinaryElementwiseOp");
    TF_ASSIGN_OR_RETURN(auto output, Compute(graph, res, inst, output_shape,
                                             tensor_map, debug_info));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0,
                                DriverTensor(output.result, graph)));
    return output.sequence;
  }
};

REGISTER_INPLACE_HLO_OP(kAdd, BinaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kAtan2, BinaryElementwiseOp);
REGISTER_HLO_OP(kCompare, BinaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kComplex, BinaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kDivide, BinaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kMaximum, BinaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kMinimum, BinaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kMultiply, BinaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kPower, BinaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kRemainder, BinaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kSubtract, BinaryElementwiseOp);
REGISTER_HLO_OP(kAnd, BinaryElementwiseOp);
REGISTER_HLO_INST_EXTENSIONS(kAnd, RegisterInplaceOnOperand0IfTypesMatch);
REGISTER_HLO_OP(kOr, BinaryElementwiseOp);
REGISTER_HLO_INST_EXTENSIONS(kOr, RegisterInplaceOnOperand0IfTypesMatch);
REGISTER_HLO_OP(kXor, BinaryElementwiseOp);
REGISTER_HLO_INST_EXTENSIONS(kXor, RegisterInplaceOnOperand0IfTypesMatch);
REGISTER_INPLACE_HLO_OP(kShiftLeft, BinaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kShiftRightArithmetic, BinaryElementwiseOp);
REGISTER_INPLACE_HLO_OP(kShiftRightLogical, BinaryElementwiseOp);

StatusOr<DriverTensor> BroadcastImplicitNaryOutputTensor(
    const DriverTensor& in, const HloInstruction* inst,
    const Shape& output_shape) {
  auto output = in;
  // Handle special case where all the inputs to the operation were broadcasts
  // of a scalar.
  if (!PoplarShapeMatchesXLAShape(output, output_shape)) {
    const HloInstruction* root = inst->fused_expression_root();
    if (absl::c_all_of(root->operands(), [](const HloInstruction* operand) {
          return operand->opcode() == HloOpcode::kBroadcast &&
                 ShapeUtil::ElementsIn(operand->operand(0)->shape()) == 1;
        })) {
      TF_ASSIGN_OR_RETURN(
          output,
          BroadcastTensor(output.reshape({}), output_shape, /*dimensions=*/{}));
    }
  }
  return output;
}

class ImplicitBinaryElementwiseOp : public BinaryElementwiseOp {
  StatusOr<poplar::Tensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context,
                                    "ImplicitBinaryElementwiseOp");
    const HloInstruction* inst = tensor_target.tgt;
    CHECK_EQ(inst->operand_count(), 2);
    const int64 input_index = tensor_target.input_index;

    const HloInstruction* layout = *tensor_target.layout;
    const auto layout_output_idx = *tensor_target.layout_output_idx;

    const Shape allocation_shape = inst->operand(input_index)->shape();
    const Shape layout_shape = layout->shape();

    TF_ASSIGN_OR_RETURN(auto type, PoplarDataType(allocation_shape));

    // Get the tensor.
    TF_ASSIGN_OR_RETURN(TensorVector outputs,
                        FindInstructionOutputTensors(tensor_map, res, layout));
    if (layout_output_idx < 0 || outputs.size() <= layout_output_idx) {
      return xla::FailedPrecondition("Binary %s input not found for %s",
                                     layout->name(), name);
    }
    poplar::Tensor other_side = outputs[layout_output_idx];

    // Get the broadcast instruction.
    const HloInstruction* broadcast =
        inst->fused_expression_root()->operand(input_index);
    CHECK_EQ(broadcast->opcode(), HloOpcode::kBroadcast);

    const std::vector<int64> non_broadcast_dimensions = broadcast->dimensions();
    absl::flat_hash_set<int64> non_broadcast_dimensions_set{
        non_broadcast_dimensions.begin(), non_broadcast_dimensions.end()};

    // Create a permutation of the other side tensor, which collapses all
    // non-broadcastable dimensions into dimension 0.
    std::vector<uint32> permutation(layout_shape.rank());
    {
      absl::c_copy(non_broadcast_dimensions, permutation.begin());
      int64 next_non_broadcast_dim = non_broadcast_dimensions.size();
      for (int64 i = 0; i != layout_shape.rank(); ++i) {
        if (!non_broadcast_dimensions_set.contains(i)) {
          permutation[next_non_broadcast_dim++] = i;
        }
      }
      CHECK_EQ(next_non_broadcast_dim, layout_shape.rank());
    }

    // Apply the permutation.
    other_side = other_side.dimShuffle(permutation);
    // Collapse all the non-broadcastable dimensions.
    other_side = other_side.flatten(0, non_broadcast_dimensions.size());

    // Allocate the tensor.
    poplar::Tensor output =
        poputil::createBroadcastOperand(graph, other_side, type, 0,
                                        /*ditherMapping*/ false, {debug_info});

    // Reshape back for all the non-broadcasted dimensions.
    output = output.reshape(PoplarShapeFromXlaShape(allocation_shape));
    return output;
  }

  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context,
                                    "ImplicitBinaryElementwiseOp");
    TF_ASSIGN_OR_RETURN(auto output, Compute(graph, res, inst, output_shape,
                                             tensor_map, debug_info));
    TF_ASSIGN_OR_RETURN(auto result, BroadcastImplicitNaryOutputTensor(
                                         output.result, inst, output_shape));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, result));
    return output.sequence;
  }
};

REGISTER_POPLAR_OP(Implicit_binary_inplace, ImplicitBinaryElementwiseOp);
REGISTER_POPLAR_OP(Implicit_binary, ImplicitBinaryElementwiseOp);

class TernaryElementwiseOp : public PoplarOpDef {
 protected:
  static StatusOr<NaryOutput> Compute(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugNameAndId& debug_name_and_id) {
    DriverProgramSequence seq(graph, debug_name_and_id);

    // Get the ternary operation.
    auto operation = helper::GetElementwiseOp(inst);
    // Create the permutation of the inputs expected by popops.
    std::vector<int64> permutation;
    switch (operation->opcode()) {
      case HloOpcode::kClamp: {
        permutation = {1, 0, 2};
        break;
      }
      case HloOpcode::kSelect: {
        permutation = {1, 2, 0};
        break;
      }
      default: {
        return xla::FailedPrecondition(
            "Trying to process %s as a ternary operation.",
            operation->ToString().c_str());
      }
    }

    TF_ASSIGN_OR_RETURN(
        auto expression_inputs,
        helper::GetElementwiseInputs(res, inst, permutation, tensor_map, seq,
                                     {debug_name_and_id}));
    auto input_tensors =
        helper::GetTensorsFromExpressionInputs(expression_inputs);

    TF_ASSIGN_OR_RETURN(popops::expr::TernaryOpType op,
                        LookupTernaryFn(operation));
    auto expr = popops::expr::TernaryOp(op, *expression_inputs[0].expr,
                                        *expression_inputs[1].expr,
                                        *expression_inputs[2].expr);

    poplar::Tensor out;
    const bool is_inplace =
        AreInplaceOutputTensorsWritable(tensor_map, res, inst);

    if (is_inplace) {
      popops::mapInPlace(graph, expr, input_tensors, seq, {debug_name_and_id});
      out = input_tensors[0];
    } else {
      out = popops::map(graph, expr, input_tensors, seq, {debug_name_and_id});
    }
    return NaryOutput{seq, DriverTensor(out, graph)};
  }

 public:
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "TernaryElementwiseOp");
    TF_ASSIGN_OR_RETURN(auto output, Compute(graph, res, inst, output_shape,
                                             tensor_map, debug_info));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output.result));
    return output.sequence;
  }
};

// TODO(T20398): Clamp and Select could be inplace on operand index 1.
REGISTER_HLO_OP(kClamp, TernaryElementwiseOp);
REGISTER_HLO_OP(kSelect, TernaryElementwiseOp);

class ImplicitTernaryElementwiseOp : public TernaryElementwiseOp {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context,
                                    "ImplicitTernaryElementwiseOp");
    TF_ASSIGN_OR_RETURN(auto output, Compute(graph, res, inst, output_shape,
                                             tensor_map, debug_info));
    TF_ASSIGN_OR_RETURN(auto result, BroadcastImplicitNaryOutputTensor(
                                         output.result, inst, output_shape));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, result));
    return output.sequence;
  }
};
REGISTER_POPLAR_OP(Implicit_ternary_inplace, ImplicitTernaryElementwiseOp);
REGISTER_POPLAR_OP(Implicit_ternary, ImplicitTernaryElementwiseOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
