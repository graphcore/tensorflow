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
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"

#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

#include <popops/DynamicSlice.hpp>

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
  const auto num_indices =
      std::accumulate(indices_shape.begin(), indices_shape.end(),
                      std::size_t(1), std::multiplies<std::size_t>());
  poplar::Tensor out = popops::createSliceTensor(
      graph, type, PoplarShapeFromXlaShape(xla_input_shape), {0}, {1},
      num_indices, plan, {}, name);
  out = out.reshape(PoplarShapeFromXlaShape(xla_updates_shape));
  return out;
}

class MultiSliceOp : public PoplibsOpDef {
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
REGISTER_POPLIBS_OP(Popops, MultiSlice, MultiSliceOp);

class MultiUpdateOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    TF_ASSIGN_OR_RETURN(poplar::program::Sequence seq,
                        CreateMultiUpdate(res, inst, tensor_map));

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
REGISTER_POPLIBS_OP(Popops, MultiUpdate, MultiUpdateOp);

class MultiUpdateAddOp : public MultiUpdateOp {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    TF_ASSIGN_OR_RETURN(poplar::program::Sequence seq,
                        CreateMultiUpdateAdd(res, inst, tensor_map));
    return seq;
  }
};
REGISTER_POPLIBS_OP(Popops, MultiUpdateAdd, MultiUpdateAddOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
