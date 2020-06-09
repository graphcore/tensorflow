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
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    // Create the control program.
    poplar::program::Sequence seq;

    // Get the input.
    TF_ASSIGN_OR_RETURN(poplar::Tensor a,
                        FindInstructionInput(tensor_map, res, inst, 0, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor b,
                        FindInstructionInput(tensor_map, res, inst, 1, seq));

    const HloTriangularSolveInstruction* as_solve =
        Cast<HloTriangularSolveInstruction>(inst);

    auto& options = as_solve->triangular_solve_options();
    bool transpose_a;
    bool lower = options.lower();

    switch (options.transpose_a()) {
      case xla::TriangularSolveOptions::NO_TRANSPOSE:
        transpose_a = false;
        break;
      case xla::TriangularSolveOptions::TRANSPOSE:
      case xla::TriangularSolveOptions::ADJOINT:
        transpose_a = true;
        break;
      default:
        return xla::InvalidArgument("invalid transpose_a value");
    }

    if (transpose_a) {
      // TF allows transposed or adjoint matrix to be passed to solver.
      // Over the field of real numbers, adjoint and transposed matrix
      // are equivalent.
      // See https://www.tensorflow.org/xla/operation_semantics#triangularsolve
      // for details.

      // Transposing A flattening all batch dimensions:
      a = a.flatten(0, a.rank() - 2).dimShuffle({0, 2, 1}).reshape(a.shape());
      // If we transpose lower triangular matrix we get upper triangular,
      // and vice versa. Flip `lower` flag before solving for transposed matrix.
      lower = !lower;
    }

    auto x = poplin::triangularSolve(
        graph, a, b, options.left_side(), lower, options.unit_diagonal(),
        res.triangular_solve_expander_block_size, seq,
        GetDebugName(inst) + "/TriangularSolve", {}, &res.matmul_cache);

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, x));
    return seq;
  }
};

REGISTER_HLO_OP(kTriangularSolve, TriangularSolveOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
