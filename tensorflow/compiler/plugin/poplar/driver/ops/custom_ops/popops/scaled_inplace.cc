/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/scaled_inplace.h"

#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <popops/ElementWise.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

class ScaledInplaceXbYOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "ScaledInplaceXbYOp");
    poplar::program::Sequence seq({}, debug_info);
    TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                        FindInplaceOutputTensors(tensor_map, res, inst, seq,
                                                 debug_info, false));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor in0 = inputs[0][0];

    TF_ASSIGN_OR_RETURN(poplar::Tensor in1,
                        FindInstructionInput(tensor_map, res, inst, 1, seq,
                                             {debug_info}, false));

    auto* scaled_inplace = Cast<HloScaledInplaceXbY>(inst);
    const HloInstruction* scale = inst->operand(2);

    if (scale->opcode() == HloOpcode::kConstant) {
      // Get the scalar multiplier
      TF_ASSIGN_OR_RETURN(double scale_val,
                          LiteralScalarToNativeType<double>(scale->literal()));

      TF_RETURN_IF_ERROR(ScaledInplaceConstantOrTensor(
          graph, in0, in1, scale_val, seq, scaled_inplace->GetOperation(),
          {debug_info}));
    } else {
      TF_ASSIGN_OR_RETURN(poplar::Tensor scale_tensor,
                          FindInstructionInput(tensor_map, res, inst, 2, seq,
                                               {debug_info}, false));
      TF_RETURN_IF_ERROR(ScaledInplaceConstantOrTensor(
          graph, in0, in1, scale_tensor, seq, scaled_inplace->GetOperation(),
          {debug_info}));
    }

    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, in0));
    return seq;
  }
};
REGISTER_POPLAR_OP(ScaledInplaceXbY, ScaledInplaceXbYOp);

class ScaledInplaceaXbYOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "ScaledInplaceaXbYOp");
    poplar::program::Sequence seq({}, debug_info);
    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, seq, debug_info, true));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor in0 = inputs[0][0];

    TF_ASSIGN_OR_RETURN(poplar::Tensor in1,
                        FindInstructionInput(tensor_map, res, inst, 1, seq,
                                             {debug_info}, false));

    auto* scaled_inplace = Cast<HloScaledInplaceaXbY>(inst);
    const HloInstruction* a = inst->operand(2);
    const HloInstruction* b = inst->operand(3);

    if (a->opcode() == HloOpcode::kConstant &&
        b->opcode() == HloOpcode::kConstant) {
      TF_ASSIGN_OR_RETURN(double a_val,
                          LiteralScalarToNativeType<double>(a->literal()));
      TF_ASSIGN_OR_RETURN(double b_val,
                          LiteralScalarToNativeType<double>(b->literal()));

      TF_RETURN_IF_ERROR(ScaledInplaceConstantOrTensor(
          graph, in0, a_val, in1, b_val, seq, scaled_inplace->GetOperation(),
          {debug_info}));
    } else {
      TF_ASSIGN_OR_RETURN(poplar::Tensor a_tensor,
                          FindInstructionInput(tensor_map, res, inst, 2, seq,
                                               {debug_info}, false));

      TF_ASSIGN_OR_RETURN(poplar::Tensor b_tensor,
                          FindInstructionInput(tensor_map, res, inst, 3, seq,
                                               {debug_info}, false));

      TF_RETURN_IF_ERROR(ScaledInplaceConstantOrTensor(
          graph, in0, a_tensor, in1, b_tensor, seq,
          scaled_inplace->GetOperation(), {debug_info}));
    }

    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, in0));
    return seq;
  }
};
REGISTER_POPLAR_OP(ScaledInplaceaXbY, ScaledInplaceaXbYOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
