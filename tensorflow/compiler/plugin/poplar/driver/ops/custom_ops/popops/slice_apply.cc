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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/slice_apply.h"

#include <poplar/Graph.hpp>
#include <popops/ElementWise.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

poplar::Tensor SliceInputForBinaryApply(const HloSliceApplyBase* inst,
                                        const poplar::Tensor& input) {
  const int64 slice_dimension = inst->GetApplyDimension();
  const int64 slice_start = inst->GetStartIndex();
  const int64 slice_end =
      slice_start + inst->operand(1)->shape().dimensions(slice_dimension);
  return input.slice(slice_start, slice_end, slice_dimension);
}

poplar::Tensor CreateSliceFromInput(poplar::Graph& graph,
                                    const HloSliceApplyBase* inst,
                                    const poplar::Tensor& input,
                                    CompilerResources& res,
                                    const std::string& name) {
  poplar::Tensor input_slice = SliceInputForBinaryApply(inst, input);
  return TensorCloneAndRebalanceAliasing(graph, res, input_slice, name);
}

poplar::Tensor CreateInputFromSlice(poplar::Graph& graph,
                                    const HloSliceApplyBase* inst,
                                    const poplar::Tensor& update,
                                    CompilerResources& res,
                                    const std::string& name) {
  // Allocate the input tensor from the update.
  const int64 slice_dimension = inst->GetApplyDimension();
  const int64 inputs_size =
      inst->operand(0)->shape().dimensions(slice_dimension);
  const int64 slice_size =
      inst->operand(1)->shape().dimensions(slice_dimension);
  return CreateTensorFromSlice(graph, update, slice_dimension, inputs_size, res,
                               name);
}

// All the slice apply ops allocate on the first two operands.
class SliceApplyAllocatorOp : public PoplarOpDef {
  StatusOr<poplar::Tensor> Allocator(poplar::Graph& graph,
                                     CompilerResources& res,
                                     const std::string& name,
                                     const TensorTarget& tensor_target,
                                     const TensorMap& tensor_map) override {
    const HloInstruction* inst = tensor_target.tgt;
    const int64 input_index = tensor_target.input_index;
    if (input_index != 0 && input_index != 1) {
      return InternalErrorStrCat("SliceApply op ", inst->name().c_str(),
                                 " should not be allocating on index ",
                                 input_index, ".");
    }
    const HloInstruction* layout = *tensor_target.layout;
    int64 layout_output_idx = *tensor_target.layout_output_idx;
    TensorOrRemoteBufferVector outputs =
        FindInstructionOutputs(tensor_map, res, layout);
    if (layout_output_idx < 0 || outputs.size() <= layout_output_idx) {
      return xla::FailedPrecondition(
          "Elementwise %s layout input not found for %s", layout->name(), name);
    }
    poplar::Tensor other_side = outputs[layout_output_idx];
    const HloSliceApplyBase* slice_apply = Cast<HloSliceApplyBase>(inst);

    return input_index == 0
               ? CreateInputFromSlice(graph, slice_apply, other_side, res, name)
               : CreateSliceFromInput(graph, slice_apply, other_side, res,
                                      name);
  }
};

class SliceApplyaXbYOp : public SliceApplyAllocatorOp {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;
    // Get the inputs.
    TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                        FindInplaceOutputTensors(tensor_map, res, inst, seq));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor input = inputs[0][0];
    TF_ASSIGN_OR_RETURN(poplar::Tensor update,
                        FindInstructionInput(tensor_map, res, inst, 1, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor scale_input,
                        FindInstructionInput(tensor_map, res, inst, 2, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor scale_update,
                        FindInstructionInput(tensor_map, res, inst, 3, seq));

    // Slice the input into the right shape.
    const HloSliceApplyaXbY* slice_apply = Cast<HloSliceApplyaXbY>(inst);
    poplar::Tensor input_slice = SliceInputForBinaryApply(slice_apply, input);

    // Apply the aXbY.
    TF_RETURN_IF_ERROR(ScaledInplaceConstantOrTensor(
        graph, input_slice, scale_input, update, scale_update, seq,
        slice_apply->GetOperation(), GetDebugName(inst)));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, input));
    return seq;
  }
};
REGISTER_POPLAR_OP(SliceApplyaXbY, SliceApplyaXbYOp);

class SliceApplyabYOp : public SliceApplyAllocatorOp {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;
    // Get the inputs.
    TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                        FindInplaceOutputTensors(tensor_map, res, inst, seq));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor input = inputs[0][0];
    TF_ASSIGN_OR_RETURN(poplar::Tensor update,
                        FindInstructionInput(tensor_map, res, inst, 1, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor scale_update,
                        FindInstructionInput(tensor_map, res, inst, 2, seq));

    // Slice the input into the right shape.
    const HloSliceApplyabY* slice_apply = Cast<HloSliceApplyabY>(inst);
    poplar::Tensor input_slice = SliceInputForBinaryApply(slice_apply, input);

    // Apply the abY.
    TF_RETURN_IF_ERROR(ScaledInplaceConstantOrTensor(
        graph, input_slice, update, scale_update, seq,
        slice_apply->GetOperation(), GetDebugName(inst)));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, input));
    return seq;
  }
};
REGISTER_POPLAR_OP(SliceApplyabY, SliceApplyabYOp);

class SliceApplyaXbOp : public SliceApplyAllocatorOp {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;
    // Get the inputs.
    TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                        FindInplaceOutputTensors(tensor_map, res, inst, seq));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor input = inputs[0][0];
    TF_ASSIGN_OR_RETURN(poplar::Tensor update,
                        FindInstructionInput(tensor_map, res, inst, 1, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor scale_input,
                        FindInstructionInput(tensor_map, res, inst, 2, seq));

    // Slice the input into the right shape.
    const HloSliceApplyaXb* slice_apply = Cast<HloSliceApplyaXb>(inst);
    poplar::Tensor input_slice = SliceInputForBinaryApply(slice_apply, input);

    const Shape& scalar_shape = inst->operand(2)->shape();
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor one,
        CreateConstantTensor(
            graph, LiteralUtil::One(scalar_shape.element_type()), scalar_shape,
            update.elementType(), GetDebugName(inst) + "/One"));

    // Apply the aXb.
    TF_RETURN_IF_ERROR(ScaledInplaceConstantOrTensor(
        graph, input_slice, scale_input, update, one, seq,
        slice_apply->GetOperation(), GetDebugName(inst)));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, input));
    return seq;
  }
};
REGISTER_POPLAR_OP(SliceApplyaXb, SliceApplyaXbOp);

class SliceApplyOp : public SliceApplyAllocatorOp {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;
    // Get the inputs.
    TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                        FindInplaceOutputTensors(tensor_map, res, inst, seq));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor input = inputs[0][0];
    TF_ASSIGN_OR_RETURN(poplar::Tensor update,
                        FindInstructionInput(tensor_map, res, inst, 1, seq));

    // Slice the input into the right shape.
    const HloSliceApply* slice_apply = Cast<HloSliceApply>(inst);
    poplar::Tensor input_slice = SliceInputForBinaryApply(slice_apply, input);

    // Apply the binary operation on the slice.
    TF_ASSIGN_OR_RETURN(popops::expr::BinaryOpType op, LookupBinaryFn(inst));
    auto expr = popops::expr::BinaryOp(op, popops::expr::_1, popops::expr::_2);
    popops::mapInPlace(graph, expr, {input_slice, update}, seq,
                       GetDebugName(inst));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, input));
    return seq;
  }
};
REGISTER_POPLAR_OP(SliceApply, SliceApplyOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
