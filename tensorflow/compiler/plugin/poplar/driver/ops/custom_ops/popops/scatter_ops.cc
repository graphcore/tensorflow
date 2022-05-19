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

#include <algorithm>
#include <limits>
#include <poplar/DebugContext.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Scatter.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {
namespace poplarplugin {
namespace {

StatusOr<DriverTensor> AddIndicesTensor(
    DriverGraph& graph, const poplar::DebugNameAndId& debug_name_and_id,
    const xla::Shape& shape, CompilerResources& resources) {
  return CreateIndicesTensor(graph, popops::SlicePlan(), shape,
                             {debug_name_and_id});
}

class ScatterOp : public PoplarOpDef {
  StatusOr<DriverTensor> Allocator(
      DriverGraph& graph, CompilerResources& resources, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "ScatterOp");
    const auto* target = tensor_target.tgt;
    const auto input_index = tensor_target.input_index;
    const auto shape = target->operand(input_index)->shape();
    DriverTensor out;

    switch (input_index) {
      case 0: {
        const auto inserted_window_dims =
            target->scatter_dimension_numbers().inserted_window_dims();
        xla::Shape slice_shape = target->operand(0)->shape();
        for (int i = 0; i < shape.rank(); ++i) {
          if (absl::c_binary_search(inserted_window_dims, i)) {
            slice_shape.set_dimensions(i, 1);
          }
        }

        TF_ASSIGN_OR_RETURN(
            out, AddScatterTensor(graph, {debug_info}, shape, slice_shape));
        break;
      }
      case 1: {
        TF_ASSIGN_OR_RETURN(
            out, AddIndicesTensor(graph, {debug_info}, shape, resources));
        break;
      }
      case 2: {
        const auto update_window_dims =
            target->scatter_dimension_numbers().update_window_dims();
        xla::Shape slice_shape = target->operand(2)->shape();
        for (int i = 0; i < shape.rank(); ++i) {
          if (!absl::c_binary_search(update_window_dims, i)) {
            slice_shape.set_dimensions(i, 1);
          }
        }

        TF_ASSIGN_OR_RETURN(
            out, AddScatterTensor(graph, {debug_info}, shape, slice_shape));
        break;
      }
      default:
        return xla::FailedPrecondition(
            "%s",
            absl::StrCat("Invalid operand for scatter instruction on ", name));
    }
    return out;
  }

  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "ScatterOp");
    const auto update_computation = inst->to_apply();
    const auto dim_numbers = inst->scatter_dimension_numbers();

    const auto update_window_dims = dim_numbers.update_window_dims();
    const auto inserted_window_dims = dim_numbers.inserted_window_dims();
    const auto scatter_dims_to_operand_dims =
        dim_numbers.scatter_dims_to_operand_dims();
    const auto index_vector_dim = dim_numbers.index_vector_dim();

    DriverProgramSequence prog(graph, debug_info);

    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, prog, debug_info));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    auto operand = inputs[0][0];

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor indices,
        FindInstructionInput(tensor_map, res, inst, 1, prog, {debug_info}));

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor updates,
        FindInstructionInput(tensor_map, res, inst, 2, prog, {debug_info}));

    popops::UpdateComputationFunc update_computation_func;
    auto root_inst = update_computation->root_instruction();

    auto tmp = graph.addVariable(operand.elementType(), {});
    graph.setTileMapping(tmp, 0);
    TensorOrRemoteBufferVectors args = {
        TensorOrRemoteBufferVector{TensorOrRemoteBuffer{tmp}},
        TensorOrRemoteBufferVector{TensorOrRemoteBuffer{graph.clone(tmp)}}};

    std::shared_ptr<DeferredVisitor> update_comp_visitor;
    // Fast path the gradient accumulation case
    if (root_inst->opcode() == HloOpcode::kAdd &&
        root_inst->operand_count() == 2 &&
        root_inst->operand(0)->opcode() == HloOpcode::kParameter &&
        root_inst->operand(1)->opcode() == HloOpcode::kParameter) {
      update_computation_func =
          [&](poplar::Graph& g, poplar::Tensor& a, poplar::Tensor& b,
              poplar::program::Sequence& p) -> poplar::Tensor {
        popops::addInPlace(g, b, a, p, {debug_info});

        return b;
      };
    } else {
      TF_ASSIGN_OR_RETURN(update_comp_visitor,
                          res.subcomputation_cache.GetOrCompileSubcomputation(
                              res, args, update_computation));

      // Handle the general case
      update_computation_func =
          [&](poplar::Graph& g, poplar::Tensor& a, poplar::Tensor& b,
              poplar::program::Sequence& p) -> poplar::Tensor {
        auto result = g.clone(b);
        for (size_t i = 0; i < a.numElements(); ++i) {
          auto a_elem = a.flatten()[i];
          auto b_elem = b.flatten()[i];
          auto o_elem = result.flatten()[i];

          // Copy the inputs in if they were used.
          if (update_comp_visitor->InputIsUsed(0, 0)) {
            p.add(poplar::program::Copy(
                a_elem, update_comp_visitor->inputs()[0][0].AsTensor(), false,
                {debug_info}));
          }

          if (update_comp_visitor->InputIsUsed(1, 0)) {
            p.add(poplar::program::Copy(
                b_elem, update_comp_visitor->inputs()[1][0].AsTensor(), false,
                {debug_info}));
          }

          // Add the sequence.
          p.add(update_comp_visitor->GetSequence(graph));

          // Copy the output out
          p.add(poplar::program::Copy(
              update_comp_visitor->outputs()[0].AsTensor(), o_elem, false,
              {debug_info}));
        }

        return result;
      };
    }

    popops::scatter(graph, operand, indices, updates, index_vector_dim,
                    {update_window_dims.begin(), update_window_dims.end()},
                    {inserted_window_dims.begin(), inserted_window_dims.end()},
                    {scatter_dims_to_operand_dims.begin(),
                     scatter_dims_to_operand_dims.end()},
                    update_computation_func, prog, {debug_info});

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand));
    return prog;
  }
};
REGISTER_HLO_OP(kScatter, ScatterOp);
}  // anonymous namespace
}  // namespace poplarplugin
}  // namespace xla
