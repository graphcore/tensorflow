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
  virtual Status LowerToPoplar(
      DriverGraph& graph, poplar::Tensor& input, DriverProgramSequence& seq,
      CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
      const std::vector<std::size_t>& output_dimensions,
      const poplar::DebugNameAndId& debug_name_and_id) {
    poplar::Tensor indices;
    if (IsPoplarInstruction(PoplarOp::ArgMax)(inst)) {
      indices = popnn::argMax(graph, input, seq, {debug_name_and_id});
    } else {
      CHECK(IsPoplarInstruction(PoplarOp::ArgMin)(inst));
      indices = popnn::argMin(graph, input, seq, {debug_name_and_id});
    }

    TF_ASSIGN_OR_RETURN(poplar::Type output_type,
                        PoplarDataType(inst->shape().element_type()));
    indices = indices.reinterpret(output_type);
    indices = indices.reshape(output_dimensions);

    TF_RETURN_IF_ERROR(
        AddOutputTensor(tensor_map, inst, 0, DriverTensor(indices, graph)));
    return Status::OK();
  }

  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "ArgMinMaxOp");
    // Create the control program.
    DriverProgramSequence seq(graph, {debug_info});

    // Get the input.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor input,
        FindInstructionInput(tensor_map, res, inst, 0, seq, debug_info));

    const int64 axis = Cast<HloArgMinMaxBase>(inst)->Axis();
    std::vector<std::size_t> output_dimensions;
    if (inst->operand(0)->shape().rank() > 1) {
      // Roll the axis dim to the end.
      input = input.dimRoll(axis, input.rank() - 1);

      // Use the remaining dims as the dims of the output.
      output_dimensions = input.shape();

      // Remove the last element.
      output_dimensions.pop_back();

      const std::size_t sum = absl::c_accumulate(
          output_dimensions, 1, std::multiplies<std::size_t>());

      // Flatten the remaining dims as popnn expects a 2d input.
      input = input.reshapePartial(0, input.rank() - 1, {sum});
    } else {
      // Special case for vectors.
      input = input.reshape({1, input.numElements()});
      output_dimensions = {};
    }
    TF_RETURN_IF_ERROR(LowerToPoplar(graph, input, seq, res, inst, tensor_map,
                                     output_dimensions, debug_info));

    return seq;
  }
};
REGISTER_POPLAR_OP(ArgMax, ArgMinMaxOp);
REGISTER_POPLAR_OP(ArgMin, ArgMinMaxOp);

class MaxMinAndArgMinMaxOp : public ArgMinMaxOp {
  virtual Status LowerToPoplar(
      DriverGraph& graph, poplar::Tensor& input, DriverProgramSequence& seq,
      CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
      const std::vector<std::size_t>& output_dimensions,
      const poplar::DebugNameAndId& debug_name_and_id) {
    poplar::Tensor values;
    poplar::Tensor indices;
    if (IsPoplarInstruction(PoplarOp::MaxAndArgMax)(inst)) {
      std::tie(values, indices) =
          popnn::maxAndArgMax(graph, input, seq, {debug_name_and_id});
    } else {
      CHECK(IsPoplarInstruction(PoplarOp::MinAndArgMin)(inst));
      std::tie(values, indices) =
          popnn::minAndArgMin(graph, input, seq, {debug_name_and_id});
    }

    TF_ASSIGN_OR_RETURN(
        poplar::Type output_type,
        PoplarDataType(ShapeUtil::GetSubshape(inst->shape(), ShapeIndexView{1})
                           .element_type()));
    indices = indices.reinterpret(output_type);
    indices = indices.reshape(output_dimensions);
    values = values.reshape(output_dimensions);

    TF_RETURN_IF_ERROR(
        AddOutputTensor(tensor_map, inst, 0, DriverTensor(values, graph)));
    TF_RETURN_IF_ERROR(
        AddOutputTensor(tensor_map, inst, 1, DriverTensor(indices, graph)));
    return Status::OK();
  }
};
REGISTER_POPLAR_OP(MaxAndArgMax, MaxMinAndArgMinMaxOp);
REGISTER_POPLAR_OP(MinAndArgMin, MaxMinAndArgMinMaxOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
