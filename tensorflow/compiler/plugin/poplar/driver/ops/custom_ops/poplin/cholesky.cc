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
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

#include "poplin/Cholesky.hpp"

namespace xla {
namespace poplarplugin {
namespace {

class CholeskyOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "Cholesky");

    // Create the control program.
    DriverProgramSequence seq(graph, debug_context);

    // Get the input.
    TF_ASSIGN_OR_RETURN(poplar::Tensor a,
                        FindInstructionInput(tensor_map, res, inst, 0, seq,
                                             {debug_info, "input"}, false));

    const HloCholeskyInstruction* as_solve = Cast<HloCholeskyInstruction>(inst);

    auto& options = as_solve->cholesky_options();
    bool lower = options.lower();
    auto rank = a.rank();
    if (rank < 2) {
      return InvalidArgument("Invalid rank for the input matrix.");
    }

    TF_ASSIGN_OR_RETURN(poplar::OptionFlags poplar_options,
                        GetCholeskyOptionsForInst(inst, res));

    poplar::DebugNameAndId dnai(debug_info);

    auto out = poplin::cholesky(graph, a, lower, seq, {dnai, "Cholesky"},
                                poplar_options, &res.planning_cache);

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(out, graph)));

    return seq;
  }

  StatusOr<DriverTensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "CholeskyAllocator");

    const HloInstruction* inst = tensor_target.tgt;
    const Shape& xla_shape = inst->operand(0)->shape();
    TF_ASSIGN_OR_RETURN(auto type, PoplarDataType(xla_shape));

    const HloCholeskyInstruction* as_solve = Cast<HloCholeskyInstruction>(inst);
    auto& options = as_solve->cholesky_options();
    bool lower = options.lower();

    auto shape = PoplarShapeFromXlaShape(xla_shape);

    TF_ASSIGN_OR_RETURN(poplar::OptionFlags poplar_options,
                        GetCholeskyOptionsForInst(inst, res));

    auto out = DriverTensor(
        poplin::createCholeskyInput(graph, type, shape, lower,
                                    {debug_context, "createInput"},
                                    poplar_options, &res.planning_cache),
        graph);

    MappingHelper::RemapTensor(res.linear_mapping_state, graph, out);

    return out;
  }
};

REGISTER_HLO_OP(kCholesky, CholeskyOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
