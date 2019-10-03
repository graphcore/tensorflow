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
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"

#include <popops/DynamicSlice.hpp>
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

StatusOr<poplar::Tensor> CreateIndicesTensor(
    poplar::Graph& graph, const popops::SlicePlan& plan,
    const xla::Shape& xla_indices_shape, const std::string& name) {
  std::vector<size_t> indices_shape =
      PoplarShapeFromXlaShape(xla_indices_shape);
  const auto num_indices =
      std::accumulate(indices_shape.begin(), indices_shape.end(), 1,
                      std::multiplies<std::size_t>());
  return popops::createIndicesTensor(graph, {0}, num_indices, plan, {}, name)
      .reshape(indices_shape)
      .reinterpret(poplar::INT);
}

StatusOr<poplar::Tensor> CreateInputTensor(poplar::Graph& graph,
                                           const xla::Shape& xla_input_shape,
                                           const std::string& name) {
  TF_ASSIGN_OR_RETURN(poplar::Type type, PoplarDataType(xla_input_shape));
  return popops::createSliceableTensor(
      graph, type, PoplarShapeFromXlaShape(xla_input_shape), {0}, {1}, 0, name);
}

StatusOr<poplar::Tensor> CreateGradientTensor(
    poplar::Graph& graph, const popops::SlicePlan& plan,
    const xla::Shape& xla_input_shape, const xla::Shape& xla_gradient_shape,
    const xla::Shape& xla_indices_shape, const std::string& name) {
  TF_ASSIGN_OR_RETURN(poplar::Type type, PoplarDataType(xla_gradient_shape));
  std::vector<size_t> indices_shape =
      PoplarShapeFromXlaShape(xla_indices_shape);
  const auto num_indices =
      std::accumulate(indices_shape.begin(), indices_shape.end(), 1,
                      std::multiplies<std::size_t>());
  poplar::Tensor out = popops::createSliceTensor(
      graph, type, PoplarShapeFromXlaShape(xla_input_shape), {0}, {1},
      num_indices, plan, {}, name);
  out = out.reshape(PoplarShapeFromXlaShape(xla_gradient_shape));
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
    popops::SlicePlan plan;  // TODO: Get it from res
    poplar::Tensor output = popops::multiSlice(
        graph, input,
        indices.flatten().expand({1}).reinterpret(poplar::UNSIGNED_INT), {0},
        {1}, seq, plan, {}, absl::StrCat(GetDebugName(inst), "/output"));
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
    switch (input_index) {
      case 0: {
        return CreateInputTensor(graph, inst->operand(0)->shape(),
                                 GetDebugName(inst) + "/input");
      }
      case 1: {
        popops::SlicePlan plan;  // TODO: Get it from res
        return CreateIndicesTensor(graph, plan, inst->operand(1)->shape(),
                                   GetDebugName(inst) + "/indices");
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
    poplar::program::Sequence seq;

    TF_ASSIGN_OR_RETURN(ArgVectors inputs,
                        FindInplaceOutputTensors(tensor_map, res, inst, seq));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor input = inputs[0][0];
    TF_ASSIGN_OR_RETURN(poplar::Tensor gradient,
                        FindInstructionInput(tensor_map, res, inst, 1, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor indices,
                        FindInstructionInput(tensor_map, res, inst, 2, seq));
    popops::SlicePlan plan;  // TODO: Get it from res

    gradient = gradient.expand({1});
    popops::multiUpdate(
        graph, input, gradient,
        indices.flatten().expand({1}).reinterpret(poplar::UNSIGNED_INT), {0},
        {1}, seq, plan, {}, absl::StrCat(GetDebugName(inst), "/multiUpdate"));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, input));

    return seq;
  }
  StatusOr<poplar::Tensor> Allocator(poplar::Graph& graph,
                                     CompilerResources& res,
                                     const std::string& name,
                                     const TensorTarget& tensor_target,
                                     const TensorMap& tensor_map) override {
    const HloInstruction* inst = tensor_target.tgt;
    const int64 input_index = tensor_target.input_index;
    switch (input_index) {
      case 0: {
        return CreateInputTensor(graph, inst->operand(0)->shape(),
                                 GetDebugName(inst) + "/input");
      }
      case 1: {
        popops::SlicePlan plan;  // TODO: Get it from res
        return CreateGradientTensor(
            graph, plan, inst->operand(0)->shape(), inst->operand(1)->shape(),
            inst->operand(2)->shape(), GetDebugName(inst) + "/gradient");
      }
      case 2: {
        popops::SlicePlan plan;  // TODO: Get it from res
        return CreateIndicesTensor(graph, plan, inst->operand(2)->shape(),
                                   GetDebugName(inst) + "/indices");
      }
    }
  }
};
REGISTER_POPLIBS_OP(Popops, MultiUpdate, MultiUpdateOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
