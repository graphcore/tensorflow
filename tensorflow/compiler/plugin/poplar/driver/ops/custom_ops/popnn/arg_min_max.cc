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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/arg_min_max.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

#include <poplar/DebugContext.hpp>
#include <popnn/Loss.hpp>
#include "absl/container/flat_hash_map.h"

namespace xla {
namespace poplarplugin {
namespace {

class ArgMinMaxOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "ArgMinMaxOp");
    // Create the control program.
    poplar::program::Sequence seq({}, {debug_info});

    // Get the input.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor input,
        FindInstructionInput(tensor_map, res, inst, 0, seq, debug_info));

    const bool is_max = IsPoplarInstruction(PoplarOp::ArgMax)(inst);
    const bool is_min = IsPoplarInstruction(PoplarOp::ArgMin)(inst);

    if (!is_max && !is_min) {
      return xla::FailedPrecondition(
          "Expected HLO instruction to be one of HloArgMax or HloArgMin!");
    }
    const int64 axis = Cast<HloArgMinMax>(inst)->Axis();

    std::vector<std::size_t> index_shape;

    if (inst->operand(0)->shape().rank() > 1) {
      // Roll the axis dim to the end.
      input = input.dimRoll(axis, input.rank() - 1);

      // Use the remaining dims as the dims of the output.
      index_shape = input.shape();

      // Remove the last element.
      index_shape.pop_back();

      std::size_t sum = std::accumulate(index_shape.begin(), index_shape.end(),
                                        1, std::multiplies<std::size_t>());

      // Flatten the remaining dims as popnn expects a 2d input.
      input = input.reshapePartial(0, input.rank() - 1, {sum});
    } else {
      // Special case for vectors.
      input = input.reshape({1, input.numElements()});
      index_shape = {};
    }

    // Call into the
    poplar::Tensor output;
    if (is_max) {
      output = popnn::argMax(graph, input, seq, {debug_info});
    } else {
      output = popnn::argMin(graph, input, seq, {debug_info});
    }
    output = output.reinterpret(poplar::INT);

    // Reshape the output back.
    output = output.reshape(index_shape);

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));

    return seq;
  }
};

REGISTER_POPLAR_OP(ArgMax, ArgMinMaxOp);
REGISTER_POPLAR_OP(ArgMin, ArgMinMaxOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
