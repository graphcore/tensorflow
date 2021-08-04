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
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

#include <poplar/DebugContext.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/GraphFunction.hpp>

namespace pgf = poputil::graphfn;

namespace xla {
namespace poplarplugin {
namespace {

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
  StatusOr<poplar::program::Program> Creator(
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
      VLOG(1) << "Creating a new tensor for input 0 of " << inst->name()
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

    poplar::Tensor output = popops::multiSlice(
        graph, input,
        indices.flatten().expand({1}).reinterpret(poplar::UNSIGNED_INT), {0},
        {1}, seq, *plan, opts, {debug_info, "output"});
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
  const uint32 serialization_factor = multi_update->GetSerializationFactor();

  if (serialization_factor == 0 || serialization_factor > num_updates) {
    return FailedPrecondition("Invalid serialization factor %u.",
                              serialization_factor);
  }

  TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                      GetSliceOptionsForInst(inst, res));

  auto update_fn = [&graph, &opts, &plan, &operand, &debug_name_and_id, &mode,
                    &scale](poplar::Tensor slice_indices,
                            poplar::Tensor slice_updates,
                            poplar::program::Sequence& seq) -> void {
    slice_indices = slice_indices.reinterpret(poplar::UNSIGNED_INT);
    slice_updates = slice_updates.expand({1});

    if (mode == UpdateMode::Replace) {
      popops::multiUpdate(graph, operand, slice_updates, slice_indices, {0},
                          {1}, seq, plan, opts, {debug_name_and_id});
    } else {
      popops::multiUpdateAdd(graph, operand, slice_updates, slice_indices,
                             *scale, {0}, {1}, seq, plan, opts,
                             {debug_name_and_id});
    }
  };

  if (serialization_factor == 1) {
    update_fn(indices, updates, prog);
  } else {
    // Do the updates serially and reuse the code to do so.
    const std::size_t slice_size = num_updates / serialization_factor;

    // Allocate the indices with an ideal layout.
    Shape indices_shape = multi_update->operand(1)->shape();
    indices_shape.set_dimensions(0, slice_size);
    TF_ASSIGN_OR_RETURN(poplar::Tensor allocated_slice_indices,
                        CreateIndicesTensor(graph, plan, indices_shape,
                                            {debug_name_and_id, "indices"}));

    // Allocate the updates with an ideal layout.
    Shape update_shape = multi_update->operand(2)->shape();
    update_shape.set_dimensions(0, slice_size);
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor allocated_slice_updates,
        CreateUpdatesTensor(graph, plan, multi_update->operand(0)->shape(),
                            update_shape, indices_shape,
                            {debug_name_and_id, "updates"}));

    // Wrap the update function in a poplar function so that we can reuse it
    // between the slices.
    auto f = pgf::VoidFunction(
        graph,
        {pgf::input(allocated_slice_indices, "allocated_slice_indices"),
         pgf::input(allocated_slice_updates, "allocated_slice_updates")},
        [&](std::vector<poplar::Tensor>& args, poplar::program::Sequence& seq) {
          update_fn(args[0], args[1], seq);
        });

    for (std::size_t i = 0; i != serialization_factor; ++i) {
      // Slice the indices and updates.
      const std::size_t slice_begin = i * slice_size;
      poplar::Tensor slice_indices =
          indices.slice(slice_begin, slice_begin + slice_size);
      poplar::Tensor slice_updates =
          updates.slice(slice_begin, slice_begin + slice_size);
      // Reuse the multi update function.
      std::vector<poplar::Tensor> args = {slice_indices, slice_updates};
      f(args, prog);
    }
    if (slice_size * serialization_factor != num_updates) {
      const std::size_t slice_begin = serialization_factor * slice_size;
      // Do the remainder.
      poplar::Tensor slice_indices = indices.slice(slice_begin, num_updates);
      poplar::Tensor slice_updates = updates.slice(slice_begin, num_updates);
      update_fn(slice_indices, slice_updates, prog);
    }
  }

  return Status::OK();
}

class MultiUpdateOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(
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
      VLOG(1) << "Creating a new tensor for input 0 of " << inst->name()
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
  StatusOr<poplar::program::Program> Creator(
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
      VLOG(1) << "Creating a new tensor for input 0 of " << inst->name()
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
