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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/all_gather.h"

#include <gcl/Collectives.hpp>
#include <poplar/DebugContext.hpp>
#include <popops/DynamicSlice.hpp>
#include <poputil/Util.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

// Return the number of elements in the given tensor.
int64 ElementCount(const poplar::Tensor& tensor) {
  return tensor.numElements();
}

// Reshape the tensor to the given shape, excluding the outermost dimension.
poplar::Tensor Reshape(const poplar::Tensor& tensor,
                       const std::vector<std::size_t>& shape) {
  return tensor.reshapePartial(1, 2, shape);
}

// Return a functor that slices the elements from the given tensor.
std::function<poplar::Tensor(int64, int64)> OffsetSlice(
    const poplar::Tensor& tensor, unsigned axis) {
  return [&tensor, axis](int64 offset, int64 count) -> poplar::Tensor {
    return tensor.slice(offset, offset + count, axis);
  };
}

class AllGatherOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "AllGatherOp");
    DriverProgramSequence seq(graph, debug_info);
    const int64 num_inputs = inst->operand_count();

    // If there is no replication, then we can just duplicate the inputs.
    if (res.replication_factor < 2) {
      for (int64 i = 0; i < num_inputs; ++i) {
        TF_ASSIGN_OR_RETURN(
            poplar::Tensor input,
            FindInstructionInput(tensor_map, res, inst, i, seq, {debug_info}));

        poplar::Tensor output_tensor = poputil::duplicate(
            graph, input, seq, {debug_info},
            poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);

        TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i,
                                    DriverTensor(output_tensor, graph)));
      }

      return seq;
    }

    const auto* all_gather_inst = Cast<HloPoplarAllGatherInstruction>(inst);
    TF_ASSIGN_OR_RETURN(
        const auto gcl_comm_group,
        ToGclCommGroup(all_gather_inst->GetPoplarReplicaGroups(), res));

    // Keeps track of the input tensors.
    std::vector<poplar::Tensor> inputs;

    // Collect up all the inputs
    for (int64 i = 0; i < num_inputs; ++i) {
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor input,
          FindInstructionInput(tensor_map, res, inst, i, seq, {debug_info}));

      inputs.push_back(input);
    }

    // all gather the concatenated tensor.
    std::vector<poplar::Tensor> outputs = gcl::allGatherCrossReplica(
        GetMasterGraph(res), inputs, seq, gcl_comm_group, {debug_info},
        GetReplicatedCollectiveOptions(res));

    // Add output tensors.
    for (int64 i = 0; i != outputs.size(); ++i) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i,
                                  DriverTensor(outputs[i], graph)));
    }

    // Return the sequence.
    return seq;
  }
};

REGISTER_POPLAR_OP(AllGather, AllGatherOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
