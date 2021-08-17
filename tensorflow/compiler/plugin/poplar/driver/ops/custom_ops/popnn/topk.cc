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

class TopKOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "TopKOp");
    // Create the control program.
    poplar::program::Sequence seq({}, debug_info);

    // Get the input.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor input,
        FindInstructionInput(tensor_map, res, inst, 0, seq, {debug_info}));

    const HloTopK* as_top_k = Cast<HloTopK>(inst);
    int64 num_k = as_top_k->NumK();
    bool sorted = as_top_k->ShouldBeSorted();

    std::vector<std::size_t> original_shape = input.shape();
    std::vector<std::size_t> index_shape = original_shape;

    std::size_t sum =
        std::accumulate(index_shape.begin(), index_shape.end() - 1, 1,
                        std::multiplies<std::size_t>());

    // Flatten the remaining dims as popnn expects a 2d input.
    input = input.reshapePartial(0, input.rank() - 1, {sum});

    // The output will be in the form of [sum][num_k].
    index_shape[index_shape.size() - 1] = num_k;

    poplar::Tensor index_output;

    poplar::Tensor value_output =
        popnn::topK(graph, input, index_output, num_k, sorted, seq,
                    {debug_info, "value_output"});

    // Reshape the input to be in the original form with the last dimension
    // replaced with K.
    original_shape[original_shape.size() - 1] = num_k;

    // Add the values to the tuple.
    value_output = value_output.reshape(original_shape);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, value_output));

    index_output = index_output.reinterpret(poplar::INT);

    // Add the indices to the tuple.
    index_output = index_output.reshape(original_shape);

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, index_output));
    return seq;
  }

 public:
  StatusOr<poplar::Tensor> Allocator(
      poplar::Graph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    const auto* topk_instruction = Cast<HloTopK>(tensor_target.tgt);
    const auto sort_order = topk_instruction->ShouldBeSorted()
                                ? popops::SortOrder::DESCENDING
                                : popops::SortOrder::NONE;

    const popops::TopKParams params = {topk_instruction->NumK(),
                                       topk_instruction->ShouldBeSorted(),
                                       sort_order};

    const auto xla_shape =
        tensor_target.tgt->operand(tensor_target.input_index)->shape();
    const auto shape = PoplarShapeFromXlaShape(xla_shape);

    TF_ASSIGN_OR_RETURN(poplar::Type type,
                        PoplarDataType(xla_shape.element_type()));

    PoplarOpDefDebugInfo debug_info(debug_context, "TopKAllocator");

    switch (tensor_target.input_index) {
      case 0: {
        return popops::createTopKInput(graph, type, shape, params, debug_info);
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
