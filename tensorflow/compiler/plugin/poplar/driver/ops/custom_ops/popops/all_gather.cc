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
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "AllGatherOp");
    poplar::program::Sequence seq({}, debug_info);
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

        TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, output_tensor));
      }

      return seq;
    }

    const auto* all_gather_inst = Cast<HloPoplarAllGatherInstruction>(inst);
    const auto replica_groups = all_gather_inst->GetPoplarReplicaGroups();
    TF_ASSIGN_OR_RETURN(
        const auto gcl_comm_group,
        ToGclCommGroup(all_gather_inst->GetPoplarReplicaGroups(), res));

    // Keeps track of what types we have seen in a deterministic ordering.
    std::vector<poplar::Type> seen_types;

    // Keeps track of the input tensors, grouped by type.
    absl::flat_hash_map<poplar::Type, std::vector<poplar::Tensor>,
                        PoplarTypeHasher>
        typed_inputs;

    // Keeps track of the output position, grouped by type.
    absl::flat_hash_map<poplar::Type, std::vector<int64>, PoplarTypeHasher>
        output_tracking;

    // Keeps track of the input shape, grouped by type.
    absl::flat_hash_map<poplar::Type, std::vector<std::vector<std::size_t>>,
                        PoplarTypeHasher>
        shape_tracking;

    // Collect up all the inputs
    for (int64 i = 0; i < num_inputs; ++i) {
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor input,
          FindInstructionInput(tensor_map, res, inst, i, seq, {debug_info}));

      // If we haven't seen the type before, add it to the list of seen types.
      if (!typed_inputs.contains(input.elementType())) {
        seen_types.push_back(input.elementType());
      }
      typed_inputs[input.elementType()].push_back(input.flatten());
      shape_tracking[input.elementType()].push_back(input.shape());
      output_tracking[input.elementType()].push_back(i);
    }

    // Visit each seen type in the provided order.
    for (auto type : seen_types) {
      auto& typed_input = typed_inputs[type];

      // Concatenate all the inputs of the same type.
      poplar::Tensor input = poplar::concat(typed_input);

      // all gather the concatenated tensor.
      poplar::Tensor output = gcl::allGatherCrossReplica(
          GetMasterGraph(res), input, seq, gcl_comm_group, {debug_info},
          GetReplicatedCollectiveOptions(res));

      // Work out what each sub-tensor's element count is.
      std::vector<int64> element_count;
      element_count.reserve(typed_input.size());
      absl::c_transform(typed_input, std::back_inserter(element_count),
                        ElementCount);

      // Use the sub-tensor element count to work out the offset.
      std::vector<int64> element_offset;
      element_offset.reserve(typed_input.size() + 1);
      element_offset.push_back(0);
      absl::c_partial_sum(element_count, std::back_inserter(element_offset));
      element_offset.resize(element_count.size());

      // Slice each of the sub-tensors from the output.
      std::vector<poplar::Tensor> output_tensors;
      element_offset.reserve(element_count.size());
      absl::c_transform(element_offset, element_count,
                        std::back_inserter(output_tensors),
                        OffsetSlice(output, 1));

      // Reshape to the expected output.
      auto& shape_track = shape_tracking[type];
      absl::c_transform(output_tensors, shape_track, output_tensors.begin(),
                        Reshape);

      // Add output tensors.
      auto& output_track = output_tracking[type];
      for (int i = 0; i < output_tensors.size(); ++i) {
        TF_CHECK_OK(AddOutputTensor(tensor_map, inst, output_track[i],
                                    output_tensors[i]));
      }
    }

    // Return the sequence.
    return seq;
  }
};

REGISTER_POPLAR_OP(AllGather, AllGatherOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
