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
#include <poputil/TileMapping.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
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

class InterIpuCopyOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override;
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

StatusOr<bool> SrcAndDstGraphsCompatible(CompilerResources& res,
                                         const HloInstruction* inst,
                                         const HloInstruction* src,
                                         const int64 dst_shard) {
  TF_ASSIGN_OR_RETURN(const Tileset src_tileset, GetTileset(src));
  TF_ASSIGN_OR_RETURN(const Tileset dst_tileset, GetTileset(inst));

  if (src_tileset != dst_tileset) {
    return false;
  }

  const int64 src_shard = GetShardForOutputIndex(src, 0);

  // multi-device sharding source is not compatible with any other sharding.
  if (src_shard == Devices::All && dst_shard != Devices::All) {
    return false;
  }

  // The multi-device compute shard is a compatible destination for all other
  // sharding.
  if (dst_shard == Devices::All && src_tileset == TILESET_COMPUTE_TILES) {
    return true;
  }

  if (src_tileset == TILESET_COMPUTE_TILES) {
    CHECK_GE(src_shard, 0) << src->ToString();
    CHECK_LT(src_shard, res.shard_compute_graphs.size()) << src->ToString();
    CHECK_GE(dst_shard, 0) << inst->ToString();
    CHECK_LT(dst_shard, res.shard_compute_graphs.size()) << inst->ToString();
    const poplar::Graph& src_compute_graph =
        res.shard_compute_graphs[src_shard];
    const poplar::Graph& dst_compute_graph =
        res.shard_compute_graphs[dst_shard];
    if (src_compute_graph.getTarget().getNumTiles() !=
        dst_compute_graph.getTarget().getNumTiles()) {
      return false;
    }
  }
  if (!res.io_graph.has_value()) {
    CHECK_NE(src_tileset, TILESET_IO_TILES)
        << "IO tiles not allocated, but requested by " << src->ToString();
  } else {
    // If I/O tiles are being used, the number of I/O tiles on each device must
    // match for the compute or io graphs to be compatible.
    CHECK_GE(src_shard, 0) << src->ToString();
    CHECK_LT(src_shard, res.shard_io_graphs.size()) << src->ToString();
    CHECK_GE(dst_shard, 0) << inst->ToString();
    CHECK_LT(dst_shard, res.shard_io_graphs.size()) << inst->ToString();
    const poplar::Graph& src_io_graph = res.shard_io_graphs[src_shard];
    const poplar::Graph& dst_io_graph = res.shard_io_graphs[dst_shard];
    if (src_io_graph.getTarget().getNumTiles() !=
        dst_io_graph.getTarget().getNumTiles()) {
      return false;
    }
  }
  return true;
}

StatusOr<TensorCopyInfo> GetTensorCopyInfo(
    CompilerResources& res, poplar::Tensor input, const HloInstruction* inst,
    const HloInstruction* src, int64 output_flat_tuple_index, int64 dst_shard,
    const Shape& output_shape, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  TensorLocation output_location{inst, output_flat_tuple_index};

  TF_ASSIGN_OR_RETURN(const bool tiles_match,
                      SrcAndDstGraphsCompatible(res, inst, src, dst_shard));

  if (HasTensorAllocationTarget(output_location, res) || !tiles_match) {
    // Allocate the new tensor on the destination device if there is an
    // allocation target, or if the tiles available on the src and dst devices
    // do not match. This can happen if one device has I/O tiles allocated,
    // and the other does not.
    poplar::Graph& dst_graph =
        GetGraphWithOutputIndex(res, inst, output_flat_tuple_index);
    TF_ASSIGN_OR_RETURN(poplar::Tensor output,
                        AddTensor(dst_graph, output_location, output_shape, res,
                                  tensor_map, {debug_name_and_id, "output"}));
    return TensorCopyInfo{input, input.flatten(), output, output.flatten()};
  }

  CHECK_GE(dst_shard, static_cast<int64>(Devices::All));
  CHECK_LT(dst_shard, static_cast<int64>(res.shard_to_ipu_id.size()));

  // No tensor target and src and dst graphs have equivalent tiles available so
  // reuse the src tensor preserving aliasing.
  poplar::Graph& master_graph = GetMasterGraph(res);

  poplar::Tensor output;
  // When the destination is all the compute tiles, we can directly clone it.
  if (dst_shard == Devices::All) {
    output = master_graph.clone(
        input, {debug_name_and_id, std::to_string(output_flat_tuple_index)},
        poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
  } else {
    const unsigned dst_device_id = res.shard_to_ipu_id[dst_shard];
    output = poputil::cloneToIpu(
        master_graph, input, dst_device_id,
        {debug_name_and_id, std::to_string(output_flat_tuple_index)},
        poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
  }

  poplar::Tensor input_dealiased = input.flatten();
  poplar::Tensor output_dealiased = output.flatten();
  if (input_dealiased.containsAliases()) {
    input_dealiased = input_dealiased.flatten();
    output_dealiased = output_dealiased.flatten();

    auto flat_regions = master_graph.getSortedContiguousRegions(
        input_dealiased, {{0, input_dealiased.numElements()}}, true);

    input_dealiased = poplar::concat(input_dealiased.slices(flat_regions));
    output_dealiased = poplar::concat(output_dealiased.slices(flat_regions));
  }
  return TensorCopyInfo{input, input_dealiased, output, output_dealiased};
}
}  // namespace

StatusOr<poplar::program::Sequence> InterIpuCopyOp::Creator(
    poplar::Graph&, CompilerResources& res, const HloInstruction* inst,
    const Shape& output_shape, TensorMap& tensor_map,
    const poplar::DebugContext& debug_context) {
  PoplarOpDefDebugInfo debug_info(debug_context, "InterIpuCopyOp");
  poplar::program::Sequence seq({}, debug_info);

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
  // For each source element type, store information about shard copies it is
  // about to perform.
  using TypeShardCopyInfos = std::map<poplar::Type, ShardCopyInfos>;
  TypeShardCopyInfos copy_informations;
  std::vector<int64> operand_shardings;

  // Construct the information about the source and destination copies.
  for (int64 i = 0, flat_tuple_index = 0; i < inst->operand_count(); ++i) {
    const auto src = inst->operand(i);
    if (!src->has_sharding()) {
      return FailedPrecondition("Missing shard information on %s", src->name());
    }
    const auto& src_sharding = GetShardingDeviceIdVector(src->sharding());
    operand_shardings.insert(operand_shardings.end(), src_sharding.begin(),
                             src_sharding.end());

    auto operand_tensors = FindInstructionInputs(tensor_map, res, inst, i, seq,
                                                 {debug_info}, false);
    const auto shapes = FlattenedXlaShape(src->shape());
    CHECK_EQ(src_sharding.size(), shapes.size());
    for (int64 index = 0; index < src_sharding.size();
         ++index, ++flat_tuple_index) {
      const int64 src_shard = src_sharding[index];
      const int64 dst_shard = dst_sharding[flat_tuple_index];
      if (operand_tensors[index].IsTensor()) {
        TF_ASSIGN_OR_RETURN(
            TensorCopyInfo copy_info,
            GetTensorCopyInfo(res, operand_tensors[index], inst, src,
                              flat_tuple_index, dst_shard, shapes[index],
                              tensor_map, {debug_info}));

        TF_CHECK_OK(AddOutputTensor(tensor_map, inst, flat_tuple_index,
                                    copy_info.output));
        auto dtype = copy_info.input.elementType();
        copy_informations[dtype][src_shard][dst_shard].push_back(copy_info);
      } else if (operand_tensors[index].IsOpaque()) {
        TF_CHECK_OK(AddOutput(tensor_map, inst, flat_tuple_index,
                              operand_tensors[index]));
      } else {
        return FailedPrecondition(
            "Cannot perform inter-ipu-copy on a RemoteBuffer for instruction "
            "%s on operand %d tuple index %d",
            inst->name(), i, index);
      }
    }
  }

  if (operand_shardings.size() != dst_sharding.size()) {
    return FailedPrecondition("Mismatched sharding info on %s",
                              inst->ToString());
  }

  // Insert the copies.
  poplar::program::Sequence inter_ipu_copies({}, debug_info);
  poplar::program::Sequence intra_ipu_copies({}, debug_info);
  for (auto& type_to_src_to_dst : copy_informations) {
    for (auto& src_to_dst : type_to_src_to_dst.second) {
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
                                       poplar::concat(outputs), false,
                                       {debug_info}));
      }
    }
  }

  seq.add(inter_ipu_copies);
  seq.add(intra_ipu_copies);
  return seq;
}

REGISTER_POPLAR_OP(InterIpuCopy, InterIpuCopyOp);

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
