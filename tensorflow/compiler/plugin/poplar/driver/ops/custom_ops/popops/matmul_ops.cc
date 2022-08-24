/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <poplin/MultiConvolution.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_conv.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matmul_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

StatusOr<DriverTensor> AddLeftMatMul(
    DriverGraph& graph, CompilerResources& resources,
    const poplar::DebugNameAndId& debug_name_and_id,
    const TensorTarget& tensor_target, const xla::Shape& shape) {
  TF_ASSIGN_OR_RETURN(poplar::Type type, PoplarDataType(shape));
  const HloInstruction* target = tensor_target.tgt;
  auto dot_dims = target->dot_dimension_numbers();

  // Find the permutations
  std::vector<int64_t> permutations;
  Shape shuffled_shape, a_shape;

  // Collapse the LHS dimensions down to [Batch, M, Contracting]
  std::tie(a_shape, shuffled_shape, permutations) =
      LeftMatMulPrepare(target->operand(0)->shape(), dot_dims);
  auto b_shape = std::get<0>(RightMatMulPrepare(
      target->operand(1)->shape(), target->dot_dimension_numbers()));

  TF_ASSIGN_OR_RETURN(const poplar::OptionFlags opts,
                      GetMatMulOptionsForInst(target, resources));

  DriverTensor result = poplin::createMatMulGroupedInputLHS(
      graph, type, type, PoplarShapeFromXlaShape(a_shape),
      PoplarShapeFromXlaShape(b_shape), {debug_name_and_id, "lhs"}, opts,
      &resources.planning_cache);

  // Unpack matrix
  result = result.reshape(PoplarShapeFromXlaShape(shuffled_shape));
  // Permute the matrix dimensions back to the XLA shape
  // Note: the permutations vector was generated for an XLA shape
  // therefore it is already inverted.
  result = result.dimShuffle(ToUnsignedVector(permutations));
  return result;
}

StatusOr<DriverTensor> AddRightMatMul(
    DriverGraph& graph, CompilerResources& resources,
    const poplar::DebugNameAndId& debug_name_and_id,
    const TensorTarget& tensor_target, const xla::Shape& shape) {
  TF_ASSIGN_OR_RETURN(poplar::Type type, PoplarDataType(shape));
  const HloInstruction* target = tensor_target.tgt;
  auto dot_dims = target->dot_dimension_numbers();

  // Find the permutations
  std::vector<int64_t> permutations;
  Shape shuffled_shape, b_shape;

  // Collapse the LHS dimensions down to [Batch, Contracting, N]
  std::tie(b_shape, shuffled_shape, permutations) =
      RightMatMulPrepare(target->operand(1)->shape(), dot_dims);
  auto a_shape = std::get<0>(LeftMatMulPrepare(
      target->operand(0)->shape(), target->dot_dimension_numbers()));

  TF_ASSIGN_OR_RETURN(const poplar::OptionFlags opts,
                      GetMatMulOptionsForInst(target, resources));

  // Support RHS being sliceable on the output dimension.
  bool make_output_dimension_sliceable = false;
  if (tensor_target.permutation && tensor_target.sliceable_dimension) {
    const std::vector<int64_t> permutation = *tensor_target.permutation;
    const int64_t sliceable_dimension = *tensor_target.sliceable_dimension;

    // Get the dimension in the output which is sliced.
    const int64_t sliceable_output_dimension = permutation[sliceable_dimension];

    // We want the matmul to be sliceable if sliceable_output_dimension is one
    // of the output dimensions - the output dimensions are all the dimensions
    // after the batch and contracting dimensions.
    const int64_t first_output_dimension =
        dot_dims.rhs_batch_dimensions_size() +
        dot_dims.rhs_contracting_dimensions_size();

    for (int64_t d = first_output_dimension; d != permutations.size(); ++d) {
      make_output_dimension_sliceable |=
          permutations[d] == sliceable_output_dimension;
    }
  }

  DriverTensor result;
  std::vector<size_t> poplar_a_shape = PoplarShapeFromXlaShape(a_shape);
  std::vector<size_t> poplar_b_shape = PoplarShapeFromXlaShape(b_shape);
  if (make_output_dimension_sliceable) {
    VLOG(2) << "Allocating the tensor " << debug_name_and_id.getPathName()
            << " as sliceable.";
    // Swap the contracting dimension and output dimension on b.
    std::swap(poplar_b_shape[1], poplar_b_shape[2]);
    // Set the correct contracting dimension in a as well.
    poplar_a_shape[2] = poplar_b_shape[1];

    result = poplin::createMatMulGroupedInputRHS(
        graph, type, type, poplar_a_shape, poplar_b_shape,
        {debug_name_and_id, "rhs"}, opts, &resources.planning_cache);

    // Move the contracting dimension and output dimension into the right
    // locations.
    result = result.dimShuffle({0, 2, 1});
  } else {
    result = poplin::createMatMulGroupedInputRHS(
        graph, type, type, poplar_a_shape, poplar_b_shape,
        {debug_name_and_id, "rhs"}, opts, &resources.planning_cache);
  }

  // Unpack matrix
  result = result.reshape(PoplarShapeFromXlaShape(shuffled_shape));
  // Permute the matrix dimensions back to the XLA shape
  // Note: the permutations vector was generated for an XLA shape
  // therefore it is already inverted.
  result = result.dimShuffle(ToUnsignedVector(permutations));
  return result;
}

class MatMulOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "MatMulOp");
    DriverProgramSequence seq(debug_info);
    // Do not expand aliasing when creating a cached op - the input will be
    // reallocated if required.
    // Find matmul lhs tensor
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor lhs,
        FindInstructionInput(tensor_map, res, inst, 0, seq, {debug_info},
                             /*expand_aliasing*/ false));
    // Find matmul rhs tensor
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor rhs,
        FindInstructionInput(tensor_map, res, inst, 1, seq, {debug_info},
                             /*expand_aliasing*/ false));

    const DotDimensionNumbers dot_dims = inst->dot_dimension_numbers();
    TF_ASSIGN_OR_RETURN(const std::string dot_type_s, GetMLTypeAsString(inst));
    TF_ASSIGN_OR_RETURN(const poplar::OptionFlags opts,
                        GetMatMulOptionsForInst(inst, res));

    poplar::DebugNameAndId debug_name_and_id(debug_info);

    auto lhs_reduction_dimensions = dot_dims.lhs_contracting_dimensions();
    auto rhs_reduction_dimensions = dot_dims.rhs_contracting_dimensions();
    auto lhs_batch_dimensions = dot_dims.lhs_batch_dimensions();
    auto rhs_batch_dimensions = dot_dims.rhs_batch_dimensions();

    // DimShuffle the LHS to [Batch..., M..., Contracting...]
    std::vector<unsigned> lhs_permutation =
        LeftMatMulPermutations(lhs.shape(), dot_dims);
    lhs = lhs.dimShuffle(lhs_permutation);

    // DimShuffle the RHS to [Batch..., Contracting..., N...]
    std::vector<unsigned> rhs_permutation =
        RightMatMulPermutations(rhs.shape(), dot_dims);
    rhs = rhs.dimShuffle(rhs_permutation);

    // Collapse the LHS dimensions down to [Batch, M, Contracting]
    lhs = lhs.reshape(LeftMatMulPackShape(lhs.shape(), dot_dims));

    // Collapse the RHS dimensions down to [Batch, Contracting, N]
    rhs = rhs.reshape(RightMatMulPackShape(rhs.shape(), dot_dims));

    if (VLOG_IS_ON(2)) {
      std::stringstream stream;
      poplin::matMulGroupedReportPlan(stream, graph, lhs.elementType(),
                                      lhs.elementType(), lhs.shape(),
                                      rhs.shape(), opts, &res.planning_cache);
      VLOG(2) << "MatMul " << debug_name_and_id.getPathName() << ". Type "
              << dot_type_s << (res.clear_matmul_pass_type ? " (cleared)" : "")
              << ". Plan " << stream.str();
      for (auto opt : opts) {
        VLOG(2) << "- option: " << opt.first << " = " << opt.second;
      }
    }

    auto output =
        poplin::matMulGrouped(graph, lhs, rhs, seq, lhs.elementType(),
                              {debug_name_and_id}, opts, &res.planning_cache);
    // Reshape to XLA shape
    output = output.reshape(PoplarShapeFromXlaShape(output_shape));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(output)));

    return seq;
  }

  StatusOr<DriverTensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "MatMulOp");
    const int64_t input_index = tensor_target.input_index;

    const HloInstruction* inst = tensor_target.tgt;

    DriverTensor out;
    switch (input_index) {
      case 0: {
        TF_ASSIGN_OR_RETURN(out, AddLeftMatMul(graph, res, debug_info,
                                               tensor_target, inst->shape()));
        break;
      }
      case 1: {
        TF_ASSIGN_OR_RETURN(out, AddRightMatMul(graph, res, debug_info,
                                                tensor_target, inst->shape()));
        break;
      }
      default:
        return xla::FailedPrecondition(
            "Input index %d of convolution shouldn't be allocating",
            input_index);
    }

    return out;
  }
};

REGISTER_HLO_OP(kDot, MatMulOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
