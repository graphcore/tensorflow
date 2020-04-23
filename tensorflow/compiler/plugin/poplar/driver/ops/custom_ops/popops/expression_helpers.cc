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

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops/expression_helpers.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matmul_util.h"
#include "tensorflow/core/util/bcast.h"

namespace xla {
namespace poplarplugin {
namespace helper {

std::vector<poplar::Tensor> GetTensorsFromExpressionInputs(
    ExpressionInputs& expression_inputs) {
  std::vector<poplar::Tensor> tensors;
  for (auto& expression_input : expression_inputs) {
    if (expression_input.tensor) {
      tensors.push_back(*expression_input.tensor);
    }
  }
  return tensors;
}

// Get the input tensor and create a PlaceHolder Expression.
StatusOr<ExpressionInput> GetTensorInput(CompilerResources& res,
                                         const HloInstruction* inst,
                                         TensorMap& tensor_map,
                                         int64 operand_idx, int64 input_idx,
                                         poplar::program::Sequence& seq) {
  // For elementwise, operand 0 might be inplace.
  poplar::Tensor tensor;
  if (operand_idx == 0 &&
      AreInplaceOutputTensorsWritable(tensor_map, res, inst)) {
    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, seq, false));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    tensor = inputs[0][0];
  } else {
    TF_ASSIGN_OR_RETURN(tensor, FindInstructionInput(tensor_map, res, inst,
                                                     input_idx, seq, false));
  }
  // Poplar PlaceHolders start at 1
  auto expr = absl::make_unique<popops::expr::PlaceHolder>(input_idx + 1);
  return ExpressionInput(std::move(expr), tensor);
}

StatusOr<ExpressionInput> GetConstantInput(const HloInstruction* inst) {
  auto type = inst->shape().element_type();
  switch (type) {
#define GET_CONST_EXPRESSION(XLA_TYPE, NATIVE_TYPE)                         \
  case XLA_TYPE: {                                                          \
    TF_ASSIGN_OR_RETURN(                                                    \
        auto val, LiteralScalarToNativeType<NATIVE_TYPE>(inst->literal())); \
    return ExpressionInput(absl::make_unique<popops::expr::Const>(val));    \
  }
    GET_CONST_EXPRESSION(PRED, bool)
    GET_CONST_EXPRESSION(S8, int8)
    GET_CONST_EXPRESSION(U8, uint8)
    GET_CONST_EXPRESSION(S16, int16)
    GET_CONST_EXPRESSION(U16, uint16)
    GET_CONST_EXPRESSION(S32, int32)
    GET_CONST_EXPRESSION(U32, uint32)
    GET_CONST_EXPRESSION(S64, int64)
    GET_CONST_EXPRESSION(U64, uint64)
    GET_CONST_EXPRESSION(F32, float)
#undef GET_CONST_EXPRESSION
    case F16: {
      // Poplar doesn't support half as a native type, use the ConstHalf
      // expression.
      TF_ASSIGN_OR_RETURN(auto val,
                          LiteralScalarToNativeType<float>(inst->literal()));
      return ExpressionInput(absl::make_unique<popops::expr::ConstHalf>(val));
    }
    default:
      return xla::FailedPrecondition(
          "Unsupported primitive type %s.",
          primitive_util::LowercasePrimitiveTypeName(type).c_str());
  }
}

StatusOr<ExpressionInput> GetElementwiseInput(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    int64 operand_idx, int64 input_idx, poplar::program::Sequence& seq) {
  if (inst->opcode() == HloOpcode::kFusion) {
    // Fusion indicates implicit broadcasting.
    const auto* root_inst = inst->fused_expression_root();
    const auto* input = root_inst->operand(operand_idx);
    if (input->opcode() == HloOpcode::kBroadcast) {
      // We either have a broadcast of a constant or another tensor.
      if (input->operand(0)->opcode() == HloOpcode::kConstant) {
        // Input is a constant, create a constant popops expression.
        return GetConstantInput(input->operand(0));
      } else {
        // Input is not constant.
        CHECK_EQ(input->operand(0)->opcode(), HloOpcode::kParameter);
        TF_ASSIGN_OR_RETURN(
            auto expr_input,
            GetTensorInput(res, inst, tensor_map, operand_idx, input_idx, seq));
        // Broadcast the tensor internally to the right shape.
        TF_ASSIGN_OR_RETURN(expr_input.tensor,
                            BroadcastTensor(*expr_input.tensor, input->shape(),
                                            input->dimensions()));
        return expr_input;
      }
    } else {
      // The input is not broadcasted - just get the tensor.
      CHECK_EQ(input->opcode(), HloOpcode::kParameter);
      return GetTensorInput(res, inst, tensor_map, operand_idx, input_idx, seq);
    }
  } else {
    // Explicit version - just get the tensor.
    return GetTensorInput(res, inst, tensor_map, operand_idx, input_idx, seq);
  }
}

// Get the elementwise instruction when the instruction can be a fused
// instruction indicating implicit broadcasting op.
const HloInstruction* GetElementwiseOp(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kFusion ? inst->fused_expression_root()
                                              : inst;
}

// Get all the elementwise input expression and tensors.
StatusOr<ExpressionInputs> GetElementwiseInputs(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    poplar::program::Sequence& seq) {
  std::vector<ExpressionInput> expression_inputs;

  auto operation = GetElementwiseOp(inst);

  int64 input_idx = 0;
  // Go over all the inputs to the operation, and figure out what type they are.
  for (int64 operand_idx = 0; operand_idx < operation->operand_count();
       operand_idx++) {
    TF_ASSIGN_OR_RETURN(auto expression_input,
                        GetElementwiseInput(res, inst, tensor_map, operand_idx,
                                            input_idx, seq));
    expression_inputs.push_back(expression_input);
    if (expression_input.tensor) {
      input_idx++;
    }
  }
  return expression_inputs;
}

}  // namespace helper

}  // namespace poplarplugin

}  // namespace xla
