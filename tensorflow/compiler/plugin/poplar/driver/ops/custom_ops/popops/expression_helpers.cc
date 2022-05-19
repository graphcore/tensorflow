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

#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
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
namespace {
// Get the input tensor and create a PlaceHolder Expression.
StatusOr<ExpressionInput> GetTensorInput(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    int64_t operand_idx, int64_t input_idx, DriverProgramSequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id) {
  DriverTensor tensor;

  // Check whether this is the inplace input to the elementwise operation.
  bool inplace_input = false;
  auto description = GetInplaceDescription(inst);
  if (description.IsInplaceType()) {
    const auto inplace_operands = description.GetInplaceOperandIndices();
    CHECK_EQ(inplace_operands.size(), 1);
    inplace_input = inplace_operands[0] == input_idx;
  }

  if (inplace_input && AreInplaceOutputTensorsWritable(tensor_map, res, inst)) {
    TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                        FindInplaceOutputTensors(tensor_map, res, inst, seq,
                                                 debug_name_and_id, false));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    tensor = inputs[0][0];
  } else {
    TF_ASSIGN_OR_RETURN(
        tensor, FindInstructionInput(tensor_map, res, inst, input_idx, seq,
                                     debug_name_and_id, false));
  }
  return ExpressionInput(tensor);
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
    GET_CONST_EXPRESSION(S64, int64_t)
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
    int64_t operand_idx, int64_t input_idx, DriverProgramSequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id) {
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
        TF_ASSIGN_OR_RETURN(auto expr_input,
                            GetTensorInput(res, inst, tensor_map, operand_idx,
                                           input_idx, seq, debug_name_and_id));

        // Allow passing scalars to the underlying op and let poplibs broadcast.
        if (expr_input.tensor->numElements() > 1) {
          // Broadcast the tensor internally to the right shape.
          TF_ASSIGN_OR_RETURN(
              expr_input.tensor,
              BroadcastTensor(*expr_input.tensor, input->shape(),
                              input->dimensions()));
        }
        return expr_input;
      }
    } else {
      // The input is not broadcasted - just get the tensor.
      CHECK_EQ(input->opcode(), HloOpcode::kParameter);
      return GetTensorInput(res, inst, tensor_map, operand_idx, input_idx, seq,
                            debug_name_and_id);
    }
  } else {
    // Explicit version - just get the tensor.
    return GetTensorInput(res, inst, tensor_map, operand_idx, input_idx, seq,
                          debug_name_and_id);
  }
}
}  // namespace

// Get the elementwise instruction when the instruction can be a fused
// instruction indicating implicit broadcasting op.
const HloInstruction* GetElementwiseOp(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kFusion ? inst->fused_expression_root()
                                              : inst;
}

// Get all the elementwise input expression and tensors.
StatusOr<ExpressionInputs> GetElementwiseInputs(
    CompilerResources& res, const HloInstruction* inst,
    const std::vector<int64_t>& inputs_permutation, TensorMap& tensor_map,
    DriverProgramSequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id) {
  auto operation = GetElementwiseOp(inst);

  // Go over all the inputs to the operation, and figure out what type they are.
  std::vector<ExpressionInput> expression_inputs;
  for (int64_t operand_idx = 0, input_idx = 0;
       operand_idx != operation->operand_count(); ++operand_idx) {
    TF_ASSIGN_OR_RETURN(auto expression_input,
                        GetElementwiseInput(res, inst, tensor_map, operand_idx,
                                            input_idx, seq, debug_name_and_id));
    expression_inputs.push_back(expression_input);
    if (expression_input.tensor) {
      input_idx++;
    }
  }

  // Now permute the expression and create placeholder expressions.
  // Note that popops placeholders start from 1.
  std::vector<ExpressionInput> permuted_expression_inputs;
  for (int64_t i = 0, placeholder_idx = 1; i != operation->operand_count();
       ++i) {
    ExpressionInput& input = expression_inputs[inputs_permutation[i]];
    if (input.tensor) {
      input.expr =
          absl::make_unique<popops::expr::PlaceHolder>(placeholder_idx++);
    }
    permuted_expression_inputs.push_back(input);
  }
  return permuted_expression_inputs;
}

}  // namespace helper
}  // namespace poplarplugin
}  // namespace xla
