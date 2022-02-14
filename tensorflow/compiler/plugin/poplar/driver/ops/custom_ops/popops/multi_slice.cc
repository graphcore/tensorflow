/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"

#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

#include <poplar/DebugContext.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/GraphFunction.hpp>

namespace pgf = poputil::graphfn;

namespace xla {
namespace poplarplugin {
namespace {
absl::optional<std::vector<unsigned>> GetConstantIndices(
    HloInstruction* const indices) {
  // For a given input Literal, this lambda reshapes the literal to
  // be of rank-1, if possible.
  auto get_val = [](const Literal& val) -> absl::optional<Literal> {
    if (val.shape().rank() == 1) {
      return absl::optional<Literal>(val.Clone());
    }

    if (val.shape().rank() == 0) {
      return absl::optional<Literal>(val.Reshape({1}).ValueOrDie());
    }

    // If the product of the dims is equal to the number of elements,
    // then the Literal can be reshaped to be rank-1.
    const auto dims = val.shape().dimensions();
    const auto max_dim = absl::c_max_element(dims);
    if (max_dim == dims.end()) {
      return absl::nullopt;
    }

    const auto dim_product =
        absl::c_accumulate(dims, 1, std::multiplies<int64>());

    if (*max_dim != dim_product) {
      return absl::nullopt;
    }

    return absl::optional<Literal>(val.Reshape({*max_dim}).ValueOrDie());
  };

  if (indices->IsConstant()) {
    auto val_opt = get_val(indices->literal());
    if (!val_opt.has_value()) {
      return absl::nullopt;
    }

    auto value = LiteralVectorToNativeType<unsigned>(val_opt.value());
    return absl::optional<std::vector<unsigned>>(value.ValueOrDie());
  }

  auto cloned_indices = indices->Clone();
  Literal result;
  HloEvaluator evaluator(/*max_loop_iterations=*/0);

  if (!evaluator.TryEvaluate(cloned_indices.get(), &result)) {
    return absl::nullopt;
  }

  auto val_opt = get_val(result);
  if (!val_opt.has_value()) {
    return absl::nullopt;
  }

  auto value = LiteralVectorToNativeType<unsigned>(val_opt.value());
  return absl::optional<std::vector<unsigned>>(value.ValueOrDie());
}

StatusOr<poplar::Tensor> CreateInputTensor(
    poplar::Graph& graph, const popops::SlicePlan& plan,
    const Shape& xla_input_shape,
    const poplar::DebugNameAndId& debug_name_and_id) {
  TF_ASSIGN_OR_RETURN(poplar::Type type, PoplarDataType(xla_input_shape));
  return popops::createSliceableTensor(graph, type,
                                       PoplarShapeFromXlaShape(xla_input_shape),
                                       {0}, {1}, plan, {}, {debug_name_and_id});
}

StatusOr<poplar::Tensor> CreateReallocatedInputTensor(
    poplar::Graph& graph, const popops::SlicePlan& plan,
    const poplar::Tensor& tensor,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return popops::createSliceableTensor(graph, tensor.elementType(),
                                       tensor.shape(), {0}, {1}, plan, {},
                                       debug_name_and_id);
}

StatusOr<poplar::Tensor> CreateUpdatesTensor(
    poplar::Graph& graph, const popops::SlicePlan& plan,
    const Shape& xla_input_shape, const Shape& xla_updates_shape,
    const Shape& xla_indices_shape,
    const poplar::DebugNameAndId& debug_name_and_id) {
  TF_ASSIGN_OR_RETURN(poplar::Type type, PoplarDataType(xla_updates_shape));
  std::vector<size_t> indices_shape =
      PoplarShapeFromXlaShape(xla_indices_shape);
  const auto num_indices = absl::c_accumulate(indices_shape, std::size_t(1),
                                              std::multiplies<std::size_t>());
  poplar::Tensor out = popops::createSliceTensor(
      graph, type, PoplarShapeFromXlaShape(xla_input_shape), {0}, {1},
      num_indices, plan, {}, {debug_name_and_id});
  out = out.reshape(PoplarShapeFromXlaShape(xla_updates_shape));
  return out;
}

class MultiSliceOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "MultiSliceOp");
    poplar::program::Sequence seq({}, debug_info);

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor input,
        FindInstructionInput(tensor_map, res, inst, 0, seq, {debug_info}));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor indices,
        FindInstructionInput(tensor_map, res, inst, 1, seq, {debug_info}));

    TF_ASSIGN_OR_RETURN(const popops::SlicePlan* plan, GetSlicePlan(res, inst));
    // Check whether the plan was use to allocate the input tensor.
    // If it was not then we need to allocate a new input tensor.
    TF_ASSIGN_OR_RETURN(bool plan_used, SlicePlanHasAllocation(res, inst));
    if (!plan_used) {
      VLOG(3) << "Creating a new tensor for input 0 of " << inst->name()
              << " because it was not allocated using the slice plan.";
      // Create a copy of the input and allocate it using the plan.
      poplar::Tensor original_input = input;
      TF_ASSIGN_OR_RETURN(
          input, CreateReallocatedInputTensor(
                     graph, *plan, input, {debug_info, "inputReallocated"}));
      seq.add(
          poplar::program::Copy(original_input, input, false, {debug_info}));
    }

    TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                        GetSliceOptionsForInst(inst, res));

    const auto constant_indices = GetConstantIndices(inst->operands().at(1));

    poplar::Tensor output;
    if (!constant_indices.has_value()) {
      output = popops::multiSlice(
          graph, input,
          indices.flatten().expand({1}).reinterpret(poplar::UNSIGNED_INT), {0},
          {1}, seq, *plan, opts, {debug_info, "output"});
    } else {
      output = popops::multiSlice(graph, input, constant_indices.value(), {0},
                                  seq, {debug_info, "output"});
    }

    auto poplar_output_shape = PoplarShapeFromXlaShape(output_shape);

    // Unflatten the output:
    output = output.reshape(poplar_output_shape);

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));
    return seq;
  }

  StatusOr<poplar::Tensor> Allocator(
      poplar::Graph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "MultiSliceOp");
    const HloInstruction* inst = tensor_target.tgt;
    const int64 input_index = tensor_target.input_index;
    TF_ASSIGN_OR_RETURN(const popops::SlicePlan* plan, GetSlicePlan(res, inst));
    switch (input_index) {
      case 0: {
        NotifySlicePlanAllocation(res, tensor_target);
        return CreateInputTensor(graph, *plan, inst->operand(0)->shape(),
                                 {debug_info, "input"});
      }
      case 1: {
        return CreateIndicesTensor(graph, *plan, inst->operand(1)->shape(),
                                   {debug_info, "indices"});
      }
      default: {
        return FailedPrecondition(
            "Invalid allocation index %d for instruction ", input_index,
            inst->ToString());
      }
    }
  }
};
REGISTER_POPLAR_OP(MultiSlice, MultiSliceOp);

enum class UpdateMode { Replace, Accumulate };
Status MultiUpdateInternal(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    const popops::SlicePlan& plan, poplar::Tensor operand,
    const poplar::Tensor& indices, const poplar::Tensor& updates,
    poplar::program::Sequence& prog,
    const HloMultiUpdateInstruction* multi_update, UpdateMode mode,
    const poplar::DebugNameAndId& debug_name_and_id,
    absl::optional<poplar::Tensor> scale = absl::nullopt) {
  // If the updates tensor is empty, there is no need to update the operand. We
  // can return the operand as is.
  if (updates.numElements() == 0) {
    return Status::OK();
  }
  if (indices.shape().size() != 2 || indices.shape()[1] != 1) {
    return FailedPrecondition(
        "Indices should be 2D with the second dimension set to 1.");
  }

  const std::size_t num_updates = updates.shape()[0];

  TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                      GetSliceOptionsForInst(inst, res));

  poplar::Tensor unsigned_indices = indices.reinterpret(poplar::UNSIGNED_INT);
  poplar::Tensor expanded_updates = updates.expand({1});

  if (mode == UpdateMode::Replace) {
    popops::multiUpdate(graph, operand, expanded_updates, unsigned_indices, {0},
                        {1}, prog, plan, opts, {debug_name_and_id});
  } else {
    const auto constant_indices = GetConstantIndices(inst->operands().at(1));

    if (constant_indices.has_value()) {
      poplar::Tensor scale_casted = *scale;
      if (operand.elementType() == poplar::HALF &&
          scale_casted.elementType() == poplar::HALF) {
        VLOG(2) << "Casting static multi update scale to F32";
        scale_casted = popops::cast(graph, scale_casted, poplar::FLOAT, prog,
                                    {debug_name_and_id, "ScaleCast"});
      }
      popops::multiUpdateAdd(graph, operand, expanded_updates,
                             constant_indices.value(), scale_casted, 0, prog,
                             {debug_name_and_id});
    } else {
      popops::multiUpdateAdd(graph, operand, expanded_updates, unsigned_indices,
                             *scale, {0}, {1}, prog, plan, opts,
                             {debug_name_and_id});
    }
  }

  return Status::OK();
}

class MultiUpdateOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "MultiUpdateOp");
    const HloMultiUpdateInstruction* multi_update =
        Cast<HloMultiUpdateInstruction>(inst);
    poplar::program::Sequence prog({}, debug_info);
    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, prog, debug_info));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor operand = inputs[0][0];
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor indices,
        FindInstructionInput(tensor_map, res, inst, 1, prog, {debug_info}));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor updates,
        FindInstructionInput(tensor_map, res, inst, 2, prog, {debug_info}));

    TF_ASSIGN_OR_RETURN(const popops::SlicePlan* plan, GetSlicePlan(res, inst));
    // Check whether the plan was use to allocate the input tensor.
    // If it was not then we need to allocate a new input tensor.
    TF_ASSIGN_OR_RETURN(bool plan_used, SlicePlanHasAllocation(res, inst));
    poplar::Tensor operand_reallocated;
    if (!plan_used) {
      VLOG(3) << "Creating a new tensor for input 0 of " << inst->name()
              << " because it was not allocated using the slice plan.";
      // Create a copy of the input and allocate it using the plan.
      TF_ASSIGN_OR_RETURN(
          operand_reallocated,
          CreateReallocatedInputTensor(graph, *plan, operand,
                                       {debug_info, "inputReallocated"}));
      prog.add(poplar::program::Copy(operand, operand_reallocated, false,
                                     {debug_info}));
    }
    TF_RETURN_IF_ERROR(MultiUpdateInternal(
        graph, res, inst, *plan, plan_used ? operand : operand_reallocated,
        indices, updates, prog, multi_update, UpdateMode::Replace,
        {debug_info}));

    if (!plan_used) {
      // Copy the results back into the original input tensor.
      prog.add(poplar::program::Copy(operand_reallocated, operand, false,
                                     {debug_info}));
    }
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand));

    return prog;
  }

  StatusOr<poplar::Tensor> Allocator(
      poplar::Graph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "MultiUpdateOp");
    const HloInstruction* inst = tensor_target.tgt;
    const int64 input_index = tensor_target.input_index;
    TF_ASSIGN_OR_RETURN(const popops::SlicePlan* plan, GetSlicePlan(res, inst));
    switch (input_index) {
      case 0: {
        NotifySlicePlanAllocation(res, tensor_target);
        return CreateInputTensor(graph, *plan, inst->operand(0)->shape(),
                                 {debug_info, "input"});
      }
      case 1: {
        return CreateIndicesTensor(graph, *plan, inst->operand(1)->shape(),
                                   {debug_info, "indices"});
      }
      case 2: {
        return CreateUpdatesTensor(
            graph, *plan, inst->operand(0)->shape(), inst->operand(2)->shape(),
            inst->operand(1)->shape(), {debug_info, "updates"});
      }
      default: {
        return FailedPrecondition(
            "Invalid allocation index %d for instruction ", input_index,
            inst->ToString());
      }
    }
  }
};
REGISTER_POPLAR_OP(MultiUpdate, MultiUpdateOp);

class MultiUpdateAddOp : public MultiUpdateOp {
  StatusOr<poplar::program::Sequence> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "MultiUpdateAddOp");
    const HloMultiUpdateAddInstruction* multi_update_add =
        Cast<HloMultiUpdateAddInstruction>(inst);
    poplar::program::Sequence prog({}, debug_info);

    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, prog, debug_info));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor operand = inputs[0][0];
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor indices,
        FindInstructionInput(tensor_map, res, inst, 1, prog, {debug_info}));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor updates,
        FindInstructionInput(tensor_map, res, inst, 2, prog, {debug_info}));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor scale,
        FindInstructionInput(tensor_map, res, inst, 3, prog, {debug_info}));

    TF_ASSIGN_OR_RETURN(const popops::SlicePlan* plan, GetSlicePlan(res, inst));
    // Check whether the plan was use to allocate the input tensor.
    // If it was not then we need to allocate a new input tensor.
    TF_ASSIGN_OR_RETURN(bool plan_used, SlicePlanHasAllocation(res, inst));
    poplar::Tensor operand_reallocated;
    if (!plan_used) {
      VLOG(3) << "Creating a new tensor for input 0 of " << inst->name()
              << " because it was not allocated using the slice plan.";
      // Create a copy of the input and allocate it using the plan.
      TF_ASSIGN_OR_RETURN(
          operand_reallocated,
          CreateReallocatedInputTensor(graph, *plan, operand,
                                       {debug_info, "inputReallocated"}));
      prog.add(poplar::program::Copy(operand, operand_reallocated, false,
                                     {debug_info}));
    }
    TF_RETURN_IF_ERROR(MultiUpdateInternal(
        graph, res, inst, *plan, plan_used ? operand : operand_reallocated,
        indices, updates, prog, multi_update_add, UpdateMode::Accumulate,
        {debug_info}, scale));

    if (!plan_used) {
      // Copy the results back into the original input tensor.
      prog.add(poplar::program::Copy(operand_reallocated, operand, false,
                                     {debug_info}));
    }

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand));
    return prog;
  }
};
REGISTER_POPLAR_OP(MultiUpdateAdd, MultiUpdateAddOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
