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
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override;
};

StatusOr<poplar::program::Program> IpuInterCopyOp::Creator(
    poplar::Graph&, CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  if (!inst->has_sharding()) {
    return xla::FailedPrecondition("Missing shard information on %s",
                                   inst->name());
  }
  const auto& dst_sharding = GetShardingDeviceIdVector(inst->sharding());

  absl::flat_hash_set<int64> dst_devices;
  TensorVector tensors;
  // Go over all the operands and find all the tensors to copy.
  std::vector<poplar::Tensor> tensors_to_copy;
  // Keep track of indexes these tensors are in the tuples.
  std::vector<int64> tensors_to_copy_indexes;
  for (int64 i = 0; i < inst->operand_count(); ++i) {
    const auto src = inst->operand(i);
    if (!src->has_sharding()) {
      return xla::FailedPrecondition("Missing shard information on %s",
                                     src->name());
    }
    const auto& src_sharding = GetShardingDeviceIdVector(src->sharding());

    if (src_sharding.size() != dst_sharding.size()) {
      return xla::FailedPrecondition("Mismatched sharding info on %s",
                                     inst->name());
    }

    TF_ASSIGN_OR_RETURN(
        auto operand_tensors,
        FindInstructionInputTensors(tensor_map, res, inst, i, seq, false));

    // Only copy the tensors where the sharding doesn't match.
    for (size_t index = 0; index < src_sharding.size(); ++index) {
      if (src_sharding[index] != dst_sharding[index]) {
        tensors_to_copy_indexes.push_back(tensors.size() + index);
        tensors_to_copy.push_back(operand_tensors[index]);
        dst_devices.insert(dst_sharding[index]);
      }
    }
    tensors.insert(tensors.end(), operand_tensors.begin(),
                   operand_tensors.end());
  }

  // Make sure there is only one destination device.
  if (dst_devices.size() != 1) {
    return xla::InternalErrorStrCat(
        "Unsupported inter IPU copy - trying to copy to ", dst_devices.size(),
        " devices.");
  }

  // Create a concatenated and flattened tensor of the input tensors.
  auto t = FlattenAndConcatenateTensors(tensors_to_copy);

  // Get the destination device id by looking through the virtual graph to IPU
  // mapping.
  CHECK_EQ(res.shard_to_ipu_id.size(), res.shard_graphs.size());
  unsigned dst_device_id = res.shard_to_ipu_id[*std::begin(dst_devices)];
  t = poputil::copyToIpu(GetMasterGraph(res), t, seq, dst_device_id,
                         GetDebugName(inst),
                         poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);

  auto copied_tensors = SliceTensorIntoTensorsLike(t, tensors_to_copy);
  // Move the copied tensors into the right indexes.
  for (size_t i = 0; i < tensors_to_copy_indexes.size(); ++i) {
    tensors[tensors_to_copy_indexes[i]] = copied_tensors[i];
  }

  for (size_t tensor_idx = 0; tensor_idx < tensors.size(); ++tensor_idx) {
    TF_CHECK_OK(
        AddOutputTensor(tensor_map, inst, tensor_idx, tensors[tensor_idx]));
  }

  return seq;
}

REGISTER_POPLAR_OP(IpuInterCopy, IpuInterCopyOp);

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
