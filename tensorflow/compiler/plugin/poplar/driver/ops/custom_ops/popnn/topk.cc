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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/topk.h"

#include <poplar/DebugContext.hpp>
#include <popnn/Loss.hpp>
#include <popops/SortOrder.hpp>
#include <popops/TopK.hpp>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {
popops::TopKParams GetParams(const HloInstruction* inst) {
  const HloTopK* top_k = Cast<HloTopK>(inst);
  const auto sort_order = top_k->ShouldBeSorted()
                              ? popops::SortOrder::DESCENDING
                              : popops::SortOrder::NONE;
  return {top_k->NumK(), top_k->ShouldBeSorted(), sort_order};
}

std::vector<std::size_t> Get2DDimensions(const Shape& shape) {
  CHECK_GE(shape.rank(), 1);
  const int64_t rank = shape.rank();
  return {ShapeUtil::ElementsIn(shape) / shape.dimensions(rank - 1),
          shape.dimensions(rank - 1)};
}

class TopKOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "TopKOp");
    // Create the control program.
    DriverProgramSequence seq(graph, debug_info);
    const auto& xla_shape = inst->operand(0)->shape();

    // Get the input.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor input,
        FindInstructionInput(tensor_map, res, inst, 0, seq, {debug_info}));
    input = input.reshape(Get2DDimensions(xla_shape));

    poplar::Tensor value_output, index_output;
    const auto params = GetParams(inst);
    switch (xla_shape.element_type()) {
      case F16:
      case F32: {
        std::tie(value_output, index_output) = popops::topKWithPermutation(
            graph, seq, input, params, {debug_info, "value_output"});
        break;
      }
      default: {
        value_output =
            popnn::topK(graph, input, index_output, params.k,
                        params.sortOrder == popops::SortOrder::DESCENDING, seq,
                        {debug_info, "value_output"});

        break;
      }
    }
    index_output = index_output.reinterpret(poplar::INT);

    // Reshape back.
    value_output = value_output.reshape(
        PoplarShapeFromXlaShape(output_shape.tuple_shapes(0)));
    index_output = index_output.reshape(
        PoplarShapeFromXlaShape(output_shape.tuple_shapes(1)));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0,
                                DriverTensor(value_output, graph)));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1,
                                DriverTensor(index_output, graph)));
    return seq;
  }

 public:
  StatusOr<DriverTensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    const auto& xla_shape =
        tensor_target.tgt->operand(tensor_target.input_index)->shape();
    TF_ASSIGN_OR_RETURN(poplar::Type type,
                        PoplarDataType(xla_shape.element_type()));

    PoplarOpDefDebugInfo debug_info(debug_context, "TopKAllocator");

    switch (tensor_target.input_index) {
      case 0: {
        // Create the tensor in 2D.
        auto tensor = DriverTensor(
            popops::createTopKInput(graph, type, Get2DDimensions(xla_shape),
                                    GetParams(tensor_target.tgt), debug_info),
            graph);

        // Unflatten the tensor.
        return tensor.reshape(PoplarShapeFromXlaShape(xla_shape));
      }
      default: {
        return FailedPrecondition(
            "Trying to allocate TopKOp tensor for an index out of range %d.",
            tensor_target.input_index);
      }
    }
  }
};

REGISTER_POPLAR_OP(TopK, TopKOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
