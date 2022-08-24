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
#include <popops/Pad.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/within_replicas.h"

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/reduction_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

Status ValidateInputSharding(const HloInstruction* inst,
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
              "Operands can only be sharded on 1 device and must be "
              "must be provided in incrementing shard order.",
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

StatusOr<poplar::Tensor> BuildReductionInputTensor(
    const HloInstruction* inst, CompilerResources& res, TensorMap& tensor_map,
    DriverProgramSequence& seq, const PoplarOpDefDebugInfo& debug_info) {
  std::vector<poplar::Tensor> input_shards;

  for (auto i = 0u; i < inst->operand_count(); ++i) {
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor input,
        FindInstructionInput(tensor_map, res, inst, i, seq, {debug_info}));

    CHECK_EQ(input.rank(), 1);
    input_shards.push_back(input.expand({0}));
  }

  auto sharded_input = poplar::concat(input_shards, 0);
  return sharded_input;
}

class AllGatherWithinReplicaOp : public PoplarOpDef {
 public:
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "AllGatherWithinReplicaOp");
    DriverProgramSequence seq(debug_info);

    const auto ipu_count = GetNumIPUs(res);
    TF_RETURN_IF_ERROR(ValidateInputSharding(inst, ipu_count));
    TF_ASSIGN_OR_RETURN(
        auto chunks, BuildInputChunks(inst, res, tensor_map, seq, debug_info));

    poplar::Tensor output = gcl::allGatherWithinReplica(
        GetMasterGraph(res), chunks, seq, {debug_info},
        GetReplicatedCollectiveOptions(res));

    CHECK_EQ(ipu_count, output.dim(0))
        << "Expecting the gathered tensor to have an output on each IPU.";
    for (auto ipu = 0; ipu < ipu_count; ++ipu) {
      TF_CHECK_OK(
          AddOutputTensor(tensor_map, inst, ipu, DriverTensor(output[ipu])));
    }

    return seq;
  }

 private:
  static StatusOr<gcl::Chunks> BuildInputChunks(
      const HloInstruction* inst, CompilerResources& res, TensorMap& tensor_map,
      DriverProgramSequence& seq, const PoplarOpDefDebugInfo& debug_info) {
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

class ReduceScatterWithinReplicaOp : public PoplarOpDef {
 public:
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context,
                                    "ReduceScatterWithinReplicaOp");
    DriverProgramSequence seq(debug_info);

    const auto ipu_count = GetNumIPUs(res);
    TF_RETURN_IF_ERROR(ValidateInputSharding(inst, ipu_count));

    TF_ASSIGN_OR_RETURN(
        auto sharded_input,
        BuildReductionInputTensor(inst, res, tensor_map, seq, debug_info));

    const auto* reduce_scatter_inst =
        Cast<HloReduceScatterWithinReplicaInstruction>(inst);
    TF_ASSIGN_OR_RETURN(auto op,
                        ToPoplarCollectiveOperator(
                            reduce_scatter_inst->GetCollectiveOperator()));

    auto chunks = gcl::reduceScatterWithinReplica(
        GetMasterGraph(res), sharded_input, op, seq,
        {debug_info, "ReduceScatterWithinReplica"},
        GetReplicatedCollectiveOptions(res));

    CHECK_EQ(ipu_count, chunks.chunks.size())
        << "Expecting to have a chunk for each IPU.";
    TF_CHECK_OK(SetOutputs(chunks.chunks, inst, res, tensor_map));

    return seq;
  }

 private:
  Status SetOutputs(std::vector<gcl::Chunk>& output_chunks,
                    const HloInstruction* inst, CompilerResources& res,
                    TensorMap& tensor_map) {
    const auto output_tensor_shape =
        ShapeUtil::GetTupleElementShape(inst->shape(), 0);
    CHECK_EQ(output_tensor_shape.rank(), 1);
    // Each tuple element is a rank 1 tensor of the same size.
    const auto output_tensor_size = output_tensor_shape.dimensions(0);

    for (auto i = 0; i < output_chunks.size(); ++i) {
      auto tensor = output_chunks[i].tensor;
      CHECK_EQ(tensor.rank(), 1);

      // Pad everything to a consistent shape. We don't know how
      // reduceScatterWithinReplica will distribute the results, so to generate
      // the HLO with the real sizes we'd have to know some implementation
      // detail of gcl. It's easier to say everything will be size
      // Ceil(num_elements, num_replicas), like reduceScatterCrossReplica, and
      // pad the difference.
      const auto size = tensor.dim(0);
      const auto missing_elements = size != output_tensor_size;
      if (missing_elements) {
        CHECK_LT(size, output_tensor_size);

        const auto pad_count = output_tensor_size - size;
        auto& graph_shard = GetGraphWithOutputIndex(res, inst, i);
        tensor = popops::pad(graph_shard, tensor, /*paddingLower*/ 0,
                             /*paddingUpper*/ pad_count, /*dim*/ 0, /*val*/ 0);
      }

      TF_RETURN_IF_ERROR(
          AddOutputTensor(tensor_map, inst, i, DriverTensor(tensor)));
    }

    return Status::OK();
  }
};

REGISTER_POPLAR_OP(ReduceScatterWithinReplica, ReduceScatterWithinReplicaOp);

class AllReduceWithinReplicaOp : public PoplarOpDef {
 public:
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "AllReduceWithinReplicaOp");
    DriverProgramSequence seq(debug_info);

    const auto ipu_count = GetNumIPUs(res);
    TF_RETURN_IF_ERROR(ValidateInputSharding(inst, ipu_count));

    TF_ASSIGN_OR_RETURN(
        auto sharded_input,
        BuildReductionInputTensor(inst, res, tensor_map, seq, debug_info));

    const auto* all_reduce_inst =
        Cast<HloAllReduceWithinReplicaInstruction>(inst);
    TF_ASSIGN_OR_RETURN(auto op, ToPoplarCollectiveOperator(
                                     all_reduce_inst->GetCollectiveOperator()));

    auto output =
        gcl::allReduceWithinReplica(GetMasterGraph(res), sharded_input, op, seq,
                                    {debug_info, "AllReduceWithinReplica"},
                                    GetReplicatedCollectiveOptions(res));

    CHECK_EQ(ipu_count, output.dim(0))
        << "Expecting the reduced tensor to have an output on each IPU.";
    for (auto ipu = 0; ipu < ipu_count; ++ipu) {
      TF_CHECK_OK(
          AddOutputTensor(tensor_map, inst, ipu, DriverTensor(output[ipu])));
    }

    return seq;
  }
};

REGISTER_POPLAR_OP(AllReduceWithinReplica, AllReduceWithinReplicaOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
