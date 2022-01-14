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
#include <gcl/Collectives.hpp>
#include <poplar/DebugContext.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

class AllGatherWithinReplicaOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "AllGatherWithinReplicaOp");
    poplar::program::Sequence seq({}, debug_info);

    auto& master_graph = GetMasterGraph(res);
    const auto ipu_count = master_graph.getTarget().getNumIPUs();

    TF_RETURN_IF_ERROR(ValidateInputSharding(inst, ipu_count));
    TF_ASSIGN_OR_RETURN(
        auto chunks, BuildInputChunks(inst, res, tensor_map, seq, debug_info));

    poplar::Tensor output =
        gcl::allGatherWithinReplica(master_graph, chunks, seq, {debug_info},
                                    GetReplicatedCollectiveOptions(res));

    CHECK_EQ(ipu_count, output.dim(0))
        << "Expecting the gathered tensor to have an output on each IPU.";
    for (auto ipu = 0; ipu < ipu_count; ++ipu) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, ipu, output[ipu]));
    }

    return seq;
  }

 private:
  static Status ValidateInputSharding(const HloInstruction* inst,
                                      unsigned int ipu_count) {
    const auto operand_count = inst->operand_count();
    if (operand_count == ipu_count) {
      for (auto i = 0u; i < ipu_count; ++i) {
        const auto* operand = inst->operand(i);

        if (operand->has_sharding()) {
          const auto& sharding = operand->sharding();

          const auto sharding_vector =
              GetShardingDeviceIdVector(operand->sharding());
          const bool valid_sharding =
              sharding_vector.size() == 1 && sharding_vector.front() == i;
          if (!valid_sharding) {
            return FailedPrecondition(
                "'%s' has sharding '%s' but expected it to be on shard %i. "
                "allGatherWithinReplica operands can only be sharded on 1 "
                "device "
                "and must be provided in incrementing shard order.",
                operand->name(), sharding.ToString(), i);
          }
        } else {
          return FailedPrecondition("Missing shard information on %s",
                                    operand->name());
        }
      }
    } else {
      return FailedPrecondition(
          "'%s' should have an operand for each ipu, but got %i operands for "
          "%i IPUs.",
          inst->name(), operand_count, ipu_count);
    }

    return Status::OK();
  }

  static StatusOr<gcl::Chunks> BuildInputChunks(
      const HloInstruction* inst, CompilerResources& res, TensorMap& tensor_map,
      poplar::program::Sequence& seq, const PoplarOpDefDebugInfo& debug_info) {
    gcl::Chunks chunks;

    std::vector<poplar::Tensor> shards;
    for (auto i = 0u; i < inst->operand_count(); ++i) {
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor input,
          FindInstructionInput(tensor_map, res, inst, i, seq, {debug_info}));

      // Offset is always 0 since we always have 1 ipu per chunk, since
      // allGatherWithinReplica is limited to 1 ipu per rank (where rank refers
      // to the subtensor after indexing the outermost dimension). Having
      // index=i means that the chunks in our gathered tensor will be in the
      // same order as the inputs.
      chunks.chunks.push_back({input, /*index*/ i, /*offset*/ 0});

      shards.push_back(input);
    }

    auto original_input = poplar::concat(shards);
    chunks.originalInput =
        original_input.expand({0}).broadcast(inst->operand_count(), 0);
    return chunks;
  }
};

REGISTER_POPLAR_OP(AllGatherWithinReplica, AllGatherWithinReplicaOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
