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
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"

#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>

namespace xla {
namespace poplarplugin {
namespace {

StatusOr<poplar::Tensor> CreateInputTensor(poplar::Graph& graph,
                                           const popops::SlicePlan& plan,
                                           const xla::Shape& xla_input_shape,
                                           const std::string& name) {
  TF_ASSIGN_OR_RETURN(poplar::Type type, PoplarDataType(xla_input_shape));
  return popops::createSliceableTensor(graph, type,
                                       PoplarShapeFromXlaShape(xla_input_shape),
                                       {0}, {1}, plan, {}, name);
}

StatusOr<poplar::Tensor> CreateUpdatesTensor(
    poplar::Graph& graph, const popops::SlicePlan& plan,
    const xla::Shape& xla_input_shape, const xla::Shape& xla_updates_shape,
    const xla::Shape& xla_indices_shape, const std::string& name) {
  TF_ASSIGN_OR_RETURN(poplar::Type type, PoplarDataType(xla_updates_shape));
  std::vector<size_t> indices_shape =
      PoplarShapeFromXlaShape(xla_indices_shape);
  const auto num_indices = absl::c_accumulate(indices_shape, std::size_t(1),
                                              std::multiplies<std::size_t>());
  poplar::Tensor out = popops::createSliceTensor(
      graph, type, PoplarShapeFromXlaShape(xla_input_shape), {0}, {1},
      num_indices, plan, {}, name);
  out = out.reshape(PoplarShapeFromXlaShape(xla_updates_shape));
  return out;
}

class MultiSliceOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;

    TF_ASSIGN_OR_RETURN(poplar::Tensor input,
                        FindInstructionInput(tensor_map, res, inst, 0, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor indices,
                        FindInstructionInput(tensor_map, res, inst, 1, seq));

    TF_ASSIGN_OR_RETURN(const popops::SlicePlan* plan, GetSlicePlan(res, inst));

    poplar::Tensor output = popops::multiSlice(
        graph, input,
        indices.flatten().expand({1}).reinterpret(poplar::UNSIGNED_INT), {0},
        {1}, seq, *plan, {}, absl::StrCat(GetDebugName(inst), "/output"));
    auto poplar_output_shape = PoplarShapeFromXlaShape(output_shape);

    // Unflatten the output:
    output = output.reshape(poplar_output_shape);

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));
    return seq;
  }

  StatusOr<poplar::Tensor> Allocator(poplar::Graph& graph,
                                     CompilerResources& res,
                                     const std::string& name,
                                     const TensorTarget& tensor_target,
                                     const TensorMap& tensor_map) override {
    const HloInstruction* inst = tensor_target.tgt;
    const int64 input_index = tensor_target.input_index;
    TF_ASSIGN_OR_RETURN(const popops::SlicePlan* plan, GetSlicePlan(res, inst));
    switch (input_index) {
      case 0: {
        return CreateInputTensor(graph, *plan, inst->operand(0)->shape(),
                                 GetDebugName(inst) + "/input");
      }
      case 1: {
        return CreateIndicesTensor(graph, *plan, inst->operand(1)->shape(),
                                   GetDebugName(inst) + "/indices");
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
    poplar::Graph& graph, const popops::SlicePlan& plan, poplar::Tensor operand,
    const poplar::Tensor& indices, const poplar::Tensor& updates,
    poplar::program::Sequence& prog, const std::string& debug_prefix,
    UpdateMode mode, absl::optional<poplar::Tensor> scale = absl::nullopt) {
  // If the updates tensor is empty, there is no need to update the operand. We
  // can return the operand as is.
  if (updates.numElements() == 0) {
    return Status::OK();
  }
  if (indices.shape().size() != 2 || indices.shape()[1] != 1) {
    return xla::FailedPrecondition(
        "Indices should be 2D with the second dimension set to 1.");
  }
  poplar::Tensor expanded_updates = updates.expand({1});

  if (mode == UpdateMode::Replace) {
    popops::multiUpdate(graph, operand, expanded_updates,
                        indices.reinterpret(poplar::UNSIGNED_INT), {0}, {1},
                        prog, plan, poplar::OptionFlags(), debug_prefix);
  } else {
    popops::multiUpdateAdd(graph, operand, expanded_updates,
                           indices.reinterpret(poplar::UNSIGNED_INT), *scale,
                           {0}, {1}, prog, plan, poplar::OptionFlags(),
                           debug_prefix);
  }
  return Status::OK();
}

class MultiUpdateOp : public PoplarOpDef {
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
    poplar::Tensor operand = inputs[0][0];
    TF_ASSIGN_OR_RETURN(poplar::Tensor indices,
                        FindInstructionInput(tensor_map, res, inst, 1, prog));
    TF_ASSIGN_OR_RETURN(poplar::Tensor updates,
                        FindInstructionInput(tensor_map, res, inst, 2, prog));

    TF_ASSIGN_OR_RETURN(const popops::SlicePlan* plan, GetSlicePlan(res, inst));
    TF_RETURN_IF_ERROR(MultiUpdateInternal(graph, *plan, operand, indices,
                                           updates, prog, GetDebugName(inst),
                                           UpdateMode::Replace));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand));

    return prog;
  }

  StatusOr<poplar::Tensor> Allocator(poplar::Graph& graph,
                                     CompilerResources& res,
                                     const std::string& name,
                                     const TensorTarget& tensor_target,
                                     const TensorMap& tensor_map) override {
    const HloInstruction* inst = tensor_target.tgt;
    const int64 input_index = tensor_target.input_index;
    TF_ASSIGN_OR_RETURN(const popops::SlicePlan* plan, GetSlicePlan(res, inst));
    switch (input_index) {
      case 0: {
        return CreateInputTensor(graph, *plan, inst->operand(0)->shape(),
                                 GetDebugName(inst) + "/input");
      }
      case 1: {
        return CreateIndicesTensor(graph, *plan, inst->operand(1)->shape(),
                                   GetDebugName(inst) + "/indices");
      }
      case 2: {
        return CreateUpdatesTensor(
            graph, *plan, inst->operand(0)->shape(), inst->operand(2)->shape(),
            inst->operand(1)->shape(), GetDebugName(inst) + "/updates");
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
    poplar::Tensor operand = inputs[0][0];
    TF_ASSIGN_OR_RETURN(poplar::Tensor indices,
                        FindInstructionInput(tensor_map, res, inst, 1, prog));
    TF_ASSIGN_OR_RETURN(poplar::Tensor updates,
                        FindInstructionInput(tensor_map, res, inst, 2, prog));
    TF_ASSIGN_OR_RETURN(poplar::Tensor scale,
                        FindInstructionInput(tensor_map, res, inst, 3, prog));

    TF_ASSIGN_OR_RETURN(const popops::SlicePlan* plan, GetSlicePlan(res, inst));
    TF_RETURN_IF_ERROR(MultiUpdateInternal(graph, *plan, operand, indices,
                                           updates, prog, GetDebugName(inst),
                                           UpdateMode::Accumulate, scale));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand));
    return prog;
  }
};
REGISTER_POPLAR_OP(MultiUpdateAdd, MultiUpdateAddOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
