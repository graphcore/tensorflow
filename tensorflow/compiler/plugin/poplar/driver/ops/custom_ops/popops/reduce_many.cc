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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/reduce_many.h"

#include <poplar/DebugContext.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <poputil/TileMapping.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

namespace xla {
namespace poplarplugin {
namespace {

StatusOr<poplar::Tensor> GetOutputTensor(
    poplar::Graph& graph, TensorMap& tensor_map, CompilerResources& res,
    const HloInstruction* inst, int64 output_index,
    const poplar::DebugNameAndId& debug_name_and_id) {
  // If output is scalar, map it linearly with res.linear_mapping_state.
  const Shape& output_shape = inst->shape().tuple_shapes(output_index);
  if (ShapeUtil::ElementsIn(output_shape) == 1) {
    return AddPlainTensor(
        graph, {debug_name_and_id, "out_" + std::to_string(output_index)},
        output_shape, res, /*offset=*/true);
  }
  return AddTensor(graph, TensorLocation{inst, output_index}, output_shape, res,
                   tensor_map,
                   {debug_name_and_id, "out_" + std::to_string(output_index)});
}

class ReduceManyOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "ReduceManyOp");
    poplar::program::Sequence seq({}, debug_info);

    const auto* reduce_many_inst = Cast<HloReduceManyInstruction>(inst);
    const int64 num_reductions = reduce_many_inst->ReductionsInfo().size();

    std::vector<popops::SingleReduceOp> popops_reductions;
    std::vector<poplar::Tensor> output_tensors;
    popops_reductions.reserve(num_reductions);
    output_tensors.reserve(num_reductions);

    for (size_t i = 0; i != num_reductions; ++i) {
      // Find tensors and construct popops reductions.
      const ReductionInfo& info = reduce_many_inst->ReductionsInfo()[i];
      // Get input 2 * i as each reduction has 2 inputs
      // (tensor_to_reduce and init_value).
      TF_ASSIGN_OR_RETURN(poplar::Tensor in,
                          FindInstructionInput(tensor_map, res, inst, 2 * i,
                                               seq, {debug_info}));
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor out,
          GetOutputTensor(graph, tensor_map, res, inst, i, {debug_info}));
      output_tensors.push_back(out);

      TF_ASSIGN_OR_RETURN(popops::Operation popops_reduction_op,
                          ToPopopsReductionOp(info.reduction_op));
      popops::ReduceParams params(popops_reduction_op);
      popops_reductions.emplace_back(in, info.reduction_dims, params,
                                     out.elementType());
    }
    popops::reduceMany(graph, popops_reductions, output_tensors, seq,
                       {debug_info}, {});

    for (size_t i = 0; i != num_reductions; ++i) {
      // Apply initial value.
      const ReductionInfo& info = reduce_many_inst->ReductionsInfo()[i];
      poplar::Tensor out = output_tensors[i];

      auto* init_inst = inst->operand(2 * i + 1);
      if (!(init_inst->IsConstant() &&
            init_inst->literal() == info.identity_literal)) {
        // Get input 2 * i + 1 as each reduction has 2 inputs
        // (tensor_to_reduce and init_value).
        TF_ASSIGN_OR_RETURN(poplar::Tensor init_val,
                            FindInstructionInput(tensor_map, res, inst,
                                                 2 * i + 1, seq, debug_info));
        TF_ASSIGN_OR_RETURN(
            init_val, BroadcastTensor(init_val, inst->shape().tuple_shapes(i)));
        TF_ASSIGN_OR_RETURN(popops::expr::BinaryOpType binary_op_type,
                            ToBinaryOpType(info.reduction_op));
        popops::mapInPlace(graph, binary_op_type, out, init_val, seq,
                           {debug_info, "initval"});
      }
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, out));
    }

    // Return the sequence.
    return seq;
  }
};

REGISTER_POPLAR_OP(ReduceMany, ReduceManyOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
