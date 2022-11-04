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
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops/conv_ops_helpers.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/core/lib/core/errors.h"

using xla::poplarplugin::helper::AddConvolutionInput;
using xla::poplarplugin::helper::AddConvolutionWeights;

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

StatusOr<std::pair<poplar::Tensor, poplar::Tensor>> MaybeGetAsF8(
    const poplar::Tensor& input, const poplar::Tensor& metadata,
    poplar::Graph& graph, DriverProgramSequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id) {
  auto new_metadata = graph.clone(poplar::QUARTER_METADATA, metadata);
  if (input.elementType() != poplar::UNSIGNED_CHAR) {
    if (input.elementType() != poplar::QUARTER) {
      return tensorflow::errors::InvalidArgument(
          "MaybeGetAsF8 received an input tensor with a type ",
          input.elementType(),
          "; the allowed types are poplar::UNSIGNED_CHAR and poplar::QUARTER");
    }
    return std::make_pair(input, new_metadata);
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
  return std::make_pair(output, new_metadata);
}

StatusOr<poplar::Type> GetPoplarType(const Shape& shape) {
  TF_ASSIGN_OR_RETURN(auto type, PoplarDataType(shape));
  return type == poplar::UNSIGNED_CHAR ? poplar::QUARTER : type;
}

template <int64_t N>
StatusOr<std::array<poplar::Tensor, N>> GetInstructionInputs(
    TensorMap& tensor_map, CompilerResources& res, const HloInstruction* inst,
    DriverProgramSequence& seq, poplar::DebugNameAndId& debug_name_and_id) {
  std::array<poplar::Tensor, 4> inputs;
  for (int64_t i = 0; i < N; ++i) {
    TF_ASSIGN_OR_RETURN(
        inputs[i], FindInstructionInput(
                       tensor_map, res, inst, i, seq,
                       {debug_name_and_id, absl::StrCat("input_", i)}, false));
  }
  return inputs;
}

// (tensor index, metadata index)
template <int64_t N>
using MetadataIndices = std::array<std::pair<int64_t, int64_t>, N>;

template <int64_t N>
Status WriteNewInputMetadata(std::array<poplar::Tensor, N>& inputs,
                             const MetadataIndices<N / 2>& indices,
                             DriverGraph& graph, DriverProgramSequence& seq,
                             poplar::DebugNameAndId& debug_name_and_id) {
  for (const auto& [t_idx, m_idx] : indices) {
    TF_ASSIGN_OR_RETURN(auto new_inputs,
                        MaybeGetAsF8(inputs[t_idx], inputs[m_idx], graph, seq,
                                     debug_name_and_id));
    inputs[t_idx] = new_inputs.first;
    inputs[m_idx] = new_inputs.second;
  }

  return Status::OK();
}

Status AddOutputTensors(poplar::Tensor& output, TensorMap& tensor_map,
                        const HloInstruction* inst, DriverGraph& graph) {
  TF_RETURN_IF_ERROR(
      AddOutputTensor(tensor_map, inst, 0, MaybeGetAsU8(output)));

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 1,
                                     MaybeGetAsU8(GetMetaData(output, graph))));

  return Status::OK();
}

Status InvalidAllocatingIndex(int64_t input_index) {
  std::string err_string =
      absl::StrCat(std::string("Input index "), std::to_string(input_index),
                   " shouldn't be allocating");
  return xla::FailedPrecondition("Allocating at invalid index");
}

class QuarterMatMulOp : public PoplarOpDef {
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

    TF_ASSIGN_OR_RETURN(
        auto inputs,
        GetInstructionInputs<4>(tensor_map, res, inst, seq, debug_name_and_id));

    const MetadataIndices<2> meta_indices = {{{0, 1}, {2, 3}}};

    TF_RETURN_IF_ERROR(WriteNewInputMetadata<4>(inputs, meta_indices, graph,
                                                seq, debug_name_and_id));

    TF_ASSIGN_OR_RETURN(
        auto out_type,
        GetPoplarType(ShapeUtil::GetTupleElementShape(inst->shape(), 0)));

    auto output =
        poplin::matMulGrouped(graph, inputs[0], inputs[2], seq, out_type,
                              {debug_name_and_id, "fp8matmul"},
                              GetDefaultOptions(), &res.planning_cache);

    TF_RETURN_IF_ERROR(AddOutputTensors(output, tensor_map, inst, graph));

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
      default: { return InvalidAllocatingIndex(input_index); }
    }
  }
};

REGISTER_POPLAR_OP(F8MatMul, QuarterMatMulOp);

template <uint16_t D>
class QuarterConvOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    // Want to create program
    // maybe_copy_metadata: input[2] --> input[0].getMetadata()
    // maybe_copy_metadata: input[3] --> input[1].getMetadata()
    // convNd: (input[0], input[1]) --> output[0]

    PoplarOpDefDebugInfo debug_info(debug_context, "QuarterConvOp");
    poplar::DebugNameAndId debug_name_and_id(debug_info);

    // Create the control program
    DriverProgramSequence seq(debug_name_and_id);

    TF_ASSIGN_OR_RETURN(
        auto inputs,
        GetInstructionInputs<4>(tensor_map, res, inst, seq, debug_name_and_id));

    const MetadataIndices<2> meta_indices = {{{0, 2}, {1, 3}}};

    TF_RETURN_IF_ERROR(WriteNewInputMetadata<4>(inputs, meta_indices, graph,
                                                seq, debug_name_and_id));

    auto in = inputs[0];
    auto kernel = inputs[1];

    TF_ASSIGN_OR_RETURN(auto params, GetConvolutionParameters(inst, 0, 1));
    TF_ASSIGN_OR_RETURN(auto opts, GetConvolutionOptionsForInst(inst, res));
    TF_ASSIGN_OR_RETURN(auto group_count, GetBatchGroupCount(inst));
    TF_ASSIGN_OR_RETURN(auto conv_dims, GetConvolutionDims(inst));

    in = ShuffleConvolutionInputToPoplar(group_count, conv_dims, in);

    kernel = ShuffleConvolutionWeightsToPoplar(conv_dims, kernel, false);
    kernel = AddGroupsDimensionToWeights(params, kernel, false);

    auto out =
        poplin::convolution(graph, in, kernel, params, false, seq,
                            {debug_info, "fp8conv"}, opts, &res.planning_cache);
    out = ShuffleConvolutionOutputToTensorflow(conv_dims, out);

    TF_RETURN_IF_ERROR(AddOutputTensors(out, tensor_map, inst, graph));

    return seq;
  }

  StatusOr<DriverTensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "QuarterMatMulOp");
    poplar::DebugNameAndId debug_name_and_id(debug_info);

    const auto* inst = tensor_target.tgt;
    const auto input_index = tensor_target.input_index;

    switch (input_index) {
      case 0: {
        TF_ASSIGN_OR_RETURN(auto input_tensor,
                            AddConvolutionInput(graph, inst, res, debug_info));
        return DriverTensor(MaybeGetAsU8(input_tensor));
        break;
      }
      case 1: {
        TF_ASSIGN_OR_RETURN(
            auto kernel_tensor,
            AddConvolutionWeights(graph, inst, res, debug_info));
        return DriverTensor(MaybeGetAsU8(kernel_tensor));
        break;
      }
      default: { return InvalidAllocatingIndex(input_index); }
    }
  }
};

REGISTER_POPLAR_OP(F8Conv2D, QuarterConvOp<2>);
REGISTER_POPLAR_OP(F8Conv3D, QuarterConvOp<3>);

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
