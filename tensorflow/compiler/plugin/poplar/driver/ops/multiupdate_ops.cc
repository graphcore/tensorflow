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
#include <algorithm>
#include <limits>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Scatter.hpp>

namespace xla {
namespace poplarplugin {
namespace {

StatusOr<popops::SlicePlan*> GetSlicePlan(const HloInstruction* inst,
                                          const CompilerResources& res) {
  auto plan = res.slice_plan_mappings.find(inst->operand(0));
  if (plan == res.slice_plan_mappings.end()) {
    return xla::FailedPrecondition(
        "Could not find input in slice plan mappings");
  }
  return plan->second;
}

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
}  // namespace

StatusOr<poplar::program::Program> CreateMultiUpdate(CompilerResources& res,
                                                     const HloInstruction* inst,
                                                     TensorMap& tensor_map) {
  VLOG(1) << "Processing " << inst->name() << " as multiUpdate";
  poplar::program::Sequence prog;
  poplar::Graph& graph = GetGraph(res, inst);

  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
                      FindInplaceOutputTensors(tensor_map, res, inst, prog));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor operand = inputs[0][0];
  TF_ASSIGN_OR_RETURN(poplar::Tensor indices,
                      FindInstructionInput(tensor_map, res, inst, 1, prog));
  TF_ASSIGN_OR_RETURN(poplar::Tensor updates,
                      FindInstructionInput(tensor_map, res, inst, 2, prog));

  TF_ASSIGN_OR_RETURN(popops::SlicePlan * plan, GetSlicePlan(inst, res));
  TF_RETURN_IF_ERROR(MultiUpdateInternal(graph, *plan, operand, indices,
                                         updates, prog, GetDebugName(inst),
                                         UpdateMode::Replace));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand));

  return prog;
}

StatusOr<poplar::program::Program> CreateMultiUpdateAdd(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map) {
  VLOG(1) << "Processing " << inst->name() << " as multiUpdateAdd";
  poplar::program::Sequence prog;
  poplar::Graph& graph = GetGraph(res, inst);

  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
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

  TF_ASSIGN_OR_RETURN(popops::SlicePlan * plan, GetSlicePlan(inst, res));
  TF_RETURN_IF_ERROR(MultiUpdateInternal(graph, *plan, operand, indices,
                                         updates, prog, GetDebugName(inst),
                                         UpdateMode::Accumulate, scale));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand));

  return prog;
}

StatusOr<poplar::program::Program> CreateFusedMultiUpdateAddOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  auto fusion_root = inst->fused_instructions_computation()->root_instruction();

  const bool negate_scale = fusion_root->opcode() == HloOpcode::kSubtract;
  const bool has_reshape =
      fusion_root->operand(1)->opcode() == HloOpcode::kReshape;
  // Get the rhs of the root operation, looking through the reshape.
  auto rhs_inst = has_reshape ? fusion_root->operand(1)->operand(0)
                              : fusion_root->operand(1);
  CHECK_EQ(rhs_inst->opcode(), HloOpcode::kMultiply);

  poplar::program::Sequence prog;
  poplar::Graph& graph = GetGraph(res, inst);

  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
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
  scale = negate_scale ? popops::neg(graph, scale, prog,
                                     GetDebugName(inst) + "/NegateScale")
                       : scale;

  // A tensor view of the operand which depends on whether there is a reshape or
  // not.
  poplar::Tensor operand_reshaped = operand;
  if (has_reshape) {
    // Reshape the operand so that the values are scattered into it.
    std::vector<size_t> dims(PoplarShapeFromXlaShape(rhs_inst->shape()));
    operand_reshaped = operand_reshaped.reshape(dims);
  }

  TF_ASSIGN_OR_RETURN(popops::SlicePlan * plan, GetSlicePlan(inst, res));
  TF_RETURN_IF_ERROR(MultiUpdateInternal(
      graph, *plan, operand_reshaped, indices, updates, prog,
      GetDebugName(inst), UpdateMode::Accumulate, scale));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand));
  return prog;
}

}  // namespace poplarplugin
}  // namespace xla
