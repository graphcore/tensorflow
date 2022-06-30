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

#include <poplar/DebugContext.hpp>
#include <popops/Gather.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {
namespace poplarplugin {
namespace {

StatusOr<DriverTensor> AddGatherTensor(
    DriverGraph& graph, const poplar::DebugNameAndId& debug_name_and_id,
    const xla::Shape& shape_xla, std::vector<std::size_t> slice_sizes,
    std::vector<unsigned> start_index_map) {
  const auto shape = PoplarShapeFromXlaShape(shape_xla);

  TF_ASSIGN_OR_RETURN(poplar::Type poplar_type, PoplarDataType(shape_xla));

  return DriverTensor(
      popops::createGatherInput(graph, poplar_type, shape, slice_sizes,
                                start_index_map, {debug_name_and_id}),
      graph);
}

StatusOr<DriverTensor> AddIndicesTensor(
    DriverGraph& graph, const poplar::DebugNameAndId& debug_name_and_id,
    const xla::Shape& shape, CompilerResources& resources) {
  return CreateIndicesTensor(graph, popops::SlicePlan(), shape,
                             {debug_name_and_id});
}

class GatherOp : public PoplarOpDef {
  StatusOr<DriverTensor> Allocator(
      DriverGraph& graph, CompilerResources& resources, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "GatherOp");
    const auto* target = tensor_target.tgt;
    const auto input_index = tensor_target.input_index;
    const auto shape = target->operand(input_index)->shape();
    DriverTensor out;

    switch (input_index) {
      case 0: {
        const auto dim_numbers = target->gather_dimension_numbers();
        const auto slice_sizes = target->gather_slice_sizes();
        const auto start_index_map = dim_numbers.start_index_map();

        TF_ASSIGN_OR_RETURN(
            out,
            AddGatherTensor(graph, {debug_info}, shape,
                            {slice_sizes.begin(), slice_sizes.end()},
                            {start_index_map.begin(), start_index_map.end()}));
        break;
      }
      case 1: {
        TF_ASSIGN_OR_RETURN(
            out, AddIndicesTensor(graph, {debug_info}, shape, resources));
        break;
      }
      default:
        return xla::FailedPrecondition(
            "%s",
            absl::StrCat("Invalid operand for gather instruction on ", name));
    }
    return out;
  }

  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "GatherOp");
    const auto slice_sizes = inst->gather_slice_sizes();
    const auto dim_numbers = inst->gather_dimension_numbers();

    const auto index_vector_dim = dim_numbers.index_vector_dim();
    const auto offset_dims = dim_numbers.offset_dims();
    const auto collapsed_slice_dims = dim_numbers.collapsed_slice_dims();
    const auto start_index_map = dim_numbers.start_index_map();

    DriverProgramSequence prog(graph, debug_info);

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor operand,
        FindInstructionInput(tensor_map, res, inst, 0, prog, {debug_info}));

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor indices,
        FindInstructionInput(tensor_map, res, inst, 1, prog, {debug_info}));

    auto result = popops::gather(
        graph, operand, indices.reinterpret(poplar::UNSIGNED_INT),
        index_vector_dim, {offset_dims.begin(), offset_dims.end()},
        {slice_sizes.begin(), slice_sizes.end()},
        {collapsed_slice_dims.begin(), collapsed_slice_dims.end()},
        {start_index_map.begin(), start_index_map.end()}, prog, {debug_info},
        GetDefaultSlicingOptions());

    TF_CHECK_OK(
        AddOutputTensor(tensor_map, inst, 0, DriverTensor(result, graph)));
    return prog;
  }
};
REGISTER_HLO_OP(kGather, GatherOp);
}  // anonymous namespace
}  // namespace poplarplugin
}  // namespace xla
