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

#include <popops/Cast.hpp>
#include <poputil/Util.hpp>
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {

namespace {

poplar::Tensor GetMetaData(const poplar::Tensor& t, poplar::Graph& graph) {
  if (t.hasMetadata()) {
    return t.getMetadata();
  }
  return poputil::createVariableMetadataTensor(
             graph, poplar::QuarterMetadata::Format::F143, 0)
      .reshape({});
}

poplar::OptionFlags GetDefaultOptions() {
  poplar::OptionFlags result;
  result.set("partialsType", poplar::HALF.toString());
  return result;
}

poplar::Tensor MaybeGetAsU8(const poplar::Tensor& input) {
  return (input.elementType() == poplar::QUARTER ||
          input.elementType() == poplar::QUARTER_METADATA)
             ? input.reinterpret(poplar::UNSIGNED_CHAR)
             : input;
}

std::pair<poplar::Tensor, poplar::Tensor> MaybeGetAsF8(
    const poplar::Tensor& input, const poplar::Tensor& metadata,
    poplar::Graph& graph, DriverProgramSequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id) {
  auto new_metadata = graph.clone(poplar::QUARTER_METADATA, metadata);
  if (input.elementType() != poplar::UNSIGNED_CHAR) {
    auto new_input = input;
    if (input.elementType() != poplar::QUARTER) {
      new_input = popops::cast(graph, input, poplar::QUARTER, new_metadata, seq,
                               {debug_name_and_id, "MaybeGetAsF8Cast"});
    }
    return {input, new_metadata};
  }
  // We can't reinterpret to neither QUARTER_METADATA nor QUARTER type.
  // Instead, clone them and copy raw unsigned char data over.
  // This copy will be elided by poplar.
  auto output = graph.clone(poplar::QUARTER, new_metadata, input,
                            {debug_name_and_id, "fp8matmul"});
  seq.add(poplar::program::Copy(
      metadata, new_metadata.reinterpret(poplar::UNSIGNED_CHAR)));
  seq.add(
      poplar::program::Copy(input, output.reinterpret(poplar::UNSIGNED_CHAR)));
  return {output, new_metadata};
}

class QuarterMatMulOp : public PoplarOpDef {
  static StatusOr<poplar::Type> GetPoplarType(const Shape& shape) {
    TF_ASSIGN_OR_RETURN(auto type, PoplarDataType(shape));
    return type == poplar::UNSIGNED_CHAR ? poplar::QUARTER : type;
  }

  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    // Want to create program
    // maybe_copy_metadata: input[1] --> input[0].getMetadata()
    // maybe_copy_metadata: input[3] --> input[2].getMetadata()
    // matmul: (input[0], input[2]) --> output[0]

    PoplarOpDefDebugInfo debug_info(debug_context, "QuarterMatMulOp");
    poplar::DebugNameAndId debug_name_and_id(debug_info);

    // Create the control program
    DriverProgramSequence seq(debug_name_and_id);

    std::array<poplar::Tensor, 4> inputs;
    for (int i = 0; i < 4; ++i) {
      TF_ASSIGN_OR_RETURN(
          inputs[i],
          FindInstructionInput(tensor_map, res, inst, i, seq,
                               {debug_name_and_id, absl::StrCat("input_", i)},
                               false));
    }

    auto new_inputs =
        MaybeGetAsF8(inputs[0], inputs[1], graph, seq, debug_name_and_id);
    inputs[0] = new_inputs.first;
    inputs[1] = new_inputs.second;
    new_inputs =
        MaybeGetAsF8(inputs[2], inputs[3], graph, seq, debug_name_and_id);
    inputs[2] = new_inputs.first;
    inputs[3] = new_inputs.second;

    TF_ASSIGN_OR_RETURN(
        auto out_type,
        GetPoplarType(ShapeUtil::GetTupleElementShape(inst->shape(), 0)));

    auto output =
        poplin::matMulGrouped(graph, inputs[0], inputs[2], seq, out_type,
                              {debug_name_and_id, "fp8matmul"},
                              GetDefaultOptions(), &res.planning_cache);
    TF_RETURN_IF_ERROR(
        AddOutputTensor(tensor_map, inst, 0, MaybeGetAsU8(output)));
    TF_RETURN_IF_ERROR(AddOutputTensor(
        tensor_map, inst, 1, MaybeGetAsU8(GetMetaData(output, graph))));
    return seq;
  }

  StatusOr<DriverTensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "QuarterMatMulOp");
    poplar::DebugNameAndId debug_name_and_id(debug_info);
    const HloInstruction* inst = tensor_target.tgt;
    const int64_t input_index = tensor_target.input_index;

    const Shape& a_shape = inst->operand(0)->shape();
    const Shape& b_shape = inst->operand(2)->shape();
    TF_ASSIGN_OR_RETURN(auto type_a, GetPoplarType(a_shape));
    TF_ASSIGN_OR_RETURN(auto type_b, GetPoplarType(b_shape));
    TF_ASSIGN_OR_RETURN(
        auto output_type,
        GetPoplarType(ShapeUtil::GetTupleElementShape(inst->shape(), 0)));
    switch (input_index) {
      case 0: {
        return DriverTensor(MaybeGetAsU8(poplin::createMatMulGroupedInputLHS(
            graph, type_a, output_type, PoplarShapeFromXlaShape(a_shape),
            PoplarShapeFromXlaShape(b_shape), {debug_name_and_id, "LHS"},
            GetDefaultOptions(), &res.planning_cache)));
        break;
      }
      case 2: {
        return DriverTensor(MaybeGetAsU8(poplin::createMatMulGroupedInputRHS(
            graph, type_b, output_type, PoplarShapeFromXlaShape(a_shape),
            PoplarShapeFromXlaShape(b_shape), {debug_name_and_id, "RHS"},
            GetDefaultOptions(), &res.planning_cache)));
      }
      default: {
        std::string err_string = absl::StrCat(std::string("Input index "),
                                              std::to_string(input_index),
                                              " shouldn't be allocating");
        return xla::FailedPrecondition("Allocating at invalid index");
      }
    }
  }
};

REGISTER_POPLAR_OP(F8MatMul, QuarterMatMulOp);

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
