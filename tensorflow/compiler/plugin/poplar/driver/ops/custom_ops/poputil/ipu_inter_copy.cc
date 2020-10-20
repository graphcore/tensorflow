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
#include <poputil/TileMapping.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
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

class IpuInterCopyOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const Shape& output_shape,
                                             TensorMap& tensor_map) override;
};

namespace {
struct TensorCopyInfo {
  // The input tensor with the correct shape and aliasing.
  poplar::Tensor input;
  // The input tensor which has been flattened and aliasing might have been
  // removed.
  poplar::Tensor input_for_copy;
  // The output tensor with the correct shape and aliasing.
  poplar::Tensor output;
  // The output tensor which has been flattened and aliasing might have been
  // removed.
  poplar::Tensor output_for_copy;
};

StatusOr<TensorCopyInfo> GetTensorCopyInfo(
    CompilerResources& res, poplar::Tensor input, const HloInstruction* inst,
    int64 output_flat_tuple_index, int64 dst_shard, const Shape& output_shape,
    TensorMap& tensor_map) {
  const unsigned dst_device_id = res.shard_to_ipu_id[dst_shard];
  TensorLocation output_location{inst, output_flat_tuple_index};
  if (HasTensorAllocationTarget(output_location, res)) {
    // Allocate the new tensor on the destination device.
    poplar::Graph& graph =
        GetGraphWithOutputIndex(res, inst, output_flat_tuple_index);
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor output,
        AddTensor(graph, output_location, output_shape, res, tensor_map));
    return TensorCopyInfo{input, input.flatten(), output, output.flatten()};
  }

  // No tensor target, so just reuse the tensor, preserving aliasing.
  poplar::Graph& master_graph = GetMasterGraph(res);
  poplar::Tensor output = poputil::cloneToIpu(
      master_graph, input, dst_device_id,
      absl::StrCat(GetDebugName(inst), "/", output_flat_tuple_index),
      poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);

  poplar::Tensor input_dealised = input.flatten();
  poplar::Tensor output_dealised = output.flatten();
  if (input_dealised.containsAliases()) {
    input_dealised = input_dealised.flatten();
    output_dealised = output_dealised.flatten();

    auto flat_regions = master_graph.getSortedContiguousRegions(
        input_dealised, {{0, input_dealised.numElements()}}, true);

    input_dealised = poplar::concat(input_dealised.slices(flat_regions));
    output_dealised = poplar::concat(output_dealised.slices(flat_regions));
  }
  return TensorCopyInfo{input, input_dealised, output, output_dealised};
}
}  // namespace

StatusOr<poplar::program::Program> IpuInterCopyOp::Creator(
    poplar::Graph&, CompilerResources& res, const HloInstruction* inst,
    const Shape& output_shape, TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  if (!inst->has_sharding()) {
    return FailedPrecondition("Missing shard information on %s", inst->name());
  }
  const auto& dst_sharding = GetShardingDeviceIdVector(inst->sharding());

  // For each destination IPU, store the information about the copies.
  using DestinationShardCopyInfos =
      std::map<int64, std::vector<TensorCopyInfo>>;
  // For each source Shard, store information about copies it is about to
  // perform.
  using ShardCopyInfos = std::map<int64, DestinationShardCopyInfos>;
  ShardCopyInfos copy_informations;

  // Construct the infomartion about the source and destination copies.
  for (int64 i = 0, flat_tuple_index = 0; i < inst->operand_count(); ++i) {
    const auto src = inst->operand(i);
    if (!src->has_sharding()) {
      return FailedPrecondition("Missing shard information on %s", src->name());
    }
    const auto& src_sharding = GetShardingDeviceIdVector(src->sharding());

    if (src_sharding.size() != dst_sharding.size()) {
      return FailedPrecondition("Mismatched sharding info on %s", inst->name());
    }

    TF_ASSIGN_OR_RETURN(
        auto operand_tensors,
        FindInstructionInputTensors(tensor_map, res, inst, i, seq, false));
    const auto shapes = FlattenedXlaShape(src->shape());
    CHECK_EQ(src_sharding.size(), shapes.size());
    for (int64 index = 0; index < src_sharding.size();
         ++index, ++flat_tuple_index) {
      const int64 src_shard = src_sharding[index];
      const int64 dst_shard = dst_sharding[index];

      TF_ASSIGN_OR_RETURN(
          TensorCopyInfo copy_info,
          GetTensorCopyInfo(res, operand_tensors[index], inst, flat_tuple_index,
                            dst_shard, shapes[index], tensor_map));

      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, flat_tuple_index,
                                  copy_info.output));
      copy_informations[src_shard][dst_shard].push_back(copy_info);
    }
  }

  // Insert the copies.
  poplar::program::Sequence inter_ipu_copies;
  poplar::program::Sequence intra_ipu_copies;
  for (auto& src_to_dst : copy_informations) {
    const int64 src_shard = src_to_dst.first;

    for (auto& dst_to_copy_infos : src_to_dst.second) {
      const int64 dst_shard = dst_to_copy_infos.first;
      auto& infos = dst_to_copy_infos.second;

      VLOG(2) << "Adding copies from shard " << src_shard << " to "
              << dst_shard;
      poplar::program::Sequence& prog =
          src_shard == dst_shard ? intra_ipu_copies : inter_ipu_copies;

      std::vector<poplar::Tensor> inputs(infos.size());
      std::vector<poplar::Tensor> outputs(infos.size());
      for (int64 i = 0; i != infos.size(); ++i) {
        inputs[i] = infos[i].input_for_copy;
        outputs[i] = infos[i].output_for_copy;
      }
      prog.add(poplar::program::Copy(poplar::concat(inputs),
                                     poplar::concat(outputs)));
    }
  }

  seq.add(inter_ipu_copies);
  seq.add(intra_ipu_copies);
  return seq;
}

REGISTER_POPLAR_OP(IpuInterCopy, IpuInterCopyOp);

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
