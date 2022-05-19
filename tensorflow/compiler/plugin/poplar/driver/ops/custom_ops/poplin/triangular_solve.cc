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

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

#include "poplin/TriangularSolve.hpp"

namespace xla {
namespace poplarplugin {
namespace {

class TriangularSolveOp : public PoplarOpDef {
  static StatusOr<bool> TransposeA(const TriangularSolveOptions& options) {
    switch (options.transpose_a()) {
      case xla::TriangularSolveOptions::NO_TRANSPOSE:
        return false;
      case xla::TriangularSolveOptions::TRANSPOSE:
      case xla::TriangularSolveOptions::ADJOINT:
        return true;
      default:
        return xla::InvalidArgument("invalid transpose_a value");
    }
  }

  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "TriangularSolve");
    poplar::DebugNameAndId debug_name_and_id(debug_info);
    // Create the control program.
    DriverProgramSequence seq(graph, debug_name_and_id);

    // Get the input.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor a,
        FindInstructionInput(tensor_map, res, inst, 0, seq,
                             {debug_name_and_id, "input_a"}, false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor b,
        FindInstructionInput(tensor_map, res, inst, 1, seq,
                             {debug_name_and_id, "input_b"}, false));

    const HloTriangularSolveInstruction* as_solve =
        Cast<HloTriangularSolveInstruction>(inst);

    auto rank = a.rank();
    if (rank < 2 || rank != b.rank()) {
      return InvalidArgument("Invalid rank of the input matrices.");
    }

    auto& options = as_solve->triangular_solve_options();
    TF_ASSIGN_OR_RETURN(const bool transpose_a, TransposeA(options));

    bool lower = options.lower();
    if (transpose_a) {
      // TF allows transposed or adjoint matrix to be passed to solver.
      // Over the field of real numbers, adjoint and transposed matrix
      // are equivalent.
      // See
      // https://www.tensorflow.org/xla/operation_semantics#triangularsolve
      // for details.

      // Transposing A considering all possible batch dimensions.
      a = a.dimShufflePartial({rank - 1, rank - 2}, {rank - 2, rank - 1});
      // If we transpose lower triangular matrix we get upper triangular,
      // and vice versa. Flip `lower` flag before solving for transposed
      // matrix.
      lower = !lower;
    }

    TF_ASSIGN_OR_RETURN(poplar::OptionFlags poplar_options,
                        GetTriangularSolveOptionsForInst(inst, res));

    auto func = [&graph, &res, options, poplar_options, lower, rank, options,
                 debug_name_and_id](std::vector<poplar::Tensor>& args,
                                    poplar::program::Sequence& prog) {
      poplar::Tensor a_f = args[0];
      poplar::Tensor b_f = args[1];

      auto x_f = poplin::triangularSolve(graph, a_f, b_f, options.left_side(),
                                         lower, options.unit_diagonal(), prog,
                                         {debug_name_and_id, "TriangularSolve"},
                                         poplar_options, &res.planning_cache);
      args[2] = x_f;
    };

    poplar::Tensor x;
    std::vector<poplar::Tensor> args = {a, b, x};
    poputil::graphfn::Signature signature = {
        poputil::graphfn::input(a, "a_in"),
        poputil::graphfn::input(b, "b_in"),
        poputil::graphfn::created("out"),
    };

    TF_RETURN_IF_ERROR(res.graph_cache.ExecuteCached(
        inst, graph, res, seq, func, signature, args, {0, 1}, {},
        /*always_allocate=*/true));

    x = args[2];

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(x, graph)));
    return seq;
  }

  StatusOr<poplar::Tensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "TriangularSolveAllocator");
    const int64_t input_index = tensor_target.input_index;

    const HloInstruction* inst = tensor_target.tgt;
    const Shape& aShape = inst->operand(0)->shape();
    const Shape& bShape = inst->operand(1)->shape();
    TF_ASSIGN_OR_RETURN(auto type_a, PoplarDataType(aShape));
    TF_ASSIGN_OR_RETURN(auto type_b, PoplarDataType(bShape));

    const HloTriangularSolveInstruction* as_solve =
        Cast<HloTriangularSolveInstruction>(inst);

    auto& options = as_solve->triangular_solve_options();
    bool left_side = options.left_side();

    TF_ASSIGN_OR_RETURN(const bool transpose_a, TransposeA(options));

    auto poplar_shape_a = PoplarShapeFromXlaShape(aShape);
    auto poplar_shape_b = PoplarShapeFromXlaShape(bShape);

    auto rank = poplar_shape_b.size();
    if (rank < 2 || rank != poplar_shape_a.size()) {
      return InvalidArgument("Invalid rank of the input matrices.");
    }

    if (transpose_a && input_index == 0) {
      left_side = !left_side;
      std::swap(poplar_shape_b[rank - 1], poplar_shape_b[rank - 2]);
    }

    VLOG(2) << "Allocating input " << input_index << " for " << aShape << " "
            << bShape << " solver.";

    TF_ASSIGN_OR_RETURN(poplar::OptionFlags poplar_options,
                        GetTriangularSolveOptionsForInst(inst, res));

    poplar::Tensor out;
    switch (input_index) {
      case 0: {
        out = poplin::createTriangularSolveInputLHS(
            graph, type_a, type_b, poplar_shape_a, poplar_shape_b, left_side,
            {debug_info, "lhs"}, poplar_options, &res.planning_cache);
        break;
      }
      case 1: {
        out = poplin::createTriangularSolveInputRHS(
            graph, type_a, type_b, poplar_shape_a, poplar_shape_b, left_side,
            {debug_info, "rhs"}, poplar_options, &res.planning_cache);
        break;
      }
      default:
        return xla::FailedPrecondition(
            "Input index %d of triangular-solve shouldn't be allocating",
            input_index);
    }
    MappingHelper::RemapTensor(res.linear_mapping_state, graph, out);
    return out;
  }
};

REGISTER_HLO_OP(kTriangularSolve, TriangularSolveOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
