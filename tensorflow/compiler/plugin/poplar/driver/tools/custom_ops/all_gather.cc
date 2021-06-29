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

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

// Constructor.
HloPoplarAllGatherInstruction::HloPoplarAllGatherInstruction(
    std::vector<HloInstruction*> inputs, const Shape& output_shape,
    PoplarReplicaGroups replica_groups)
    : HloPoplarInstruction(output_shape, inputs, PoplarOp::AllGather,
                           replica_groups),
      replica_groups_(replica_groups) {}

absl::flat_hash_set<int64> HloPoplarAllGatherInstruction::AllocatingIndices()
    const {
  return {};
}

bool HloPoplarAllGatherInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64>
HloPoplarAllGatherInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloPoplarAllGatherInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloPoplarAllGatherInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults
HloPoplarAllGatherInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloPoplarAllGatherInstruction::IsPopOpsElementwise() const {
  return false;
}

PoplarReplicaGroups HloPoplarAllGatherInstruction::GetPoplarReplicaGroups()
    const {
  return replica_groups_;
}

// Creates an instance of a HloPoplarAllGatherInstruction
std::unique_ptr<HloInstruction> CreatePoplarAllGather(
    std::vector<HloInstruction*> inputs, const Shape& output_shape,
    PoplarReplicaGroups replica_groups) {
  return absl::make_unique<HloPoplarAllGatherInstruction>(inputs, output_shape,
                                                          replica_groups);
}

std::unique_ptr<HloInstruction>
HloPoplarAllGatherInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreatePoplarAllGather({operands.begin(), operands.end()}, shape,
                               replica_groups_);
}

std::vector<std::string>
HloPoplarAllGatherInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {"replica_groups=" + replica_groups_.ToString()};
}

namespace {

static HloPoplarInstructionFactory allgather_factory(
    PoplarOp::AllGather,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

      TF_ASSIGN_OR_RETURN(
          const auto replica_group_size,
          attribute_map.GetAttributeAsInt64("replica_group_size"));
      const auto replica_groups =
          PoplarReplicaGroups::Consecutive(replica_group_size);

      CHECK_EQ(call->operand_count(), 1);
      return CreatePoplarAllGather({call->mutable_operand(0)}, call->shape(),
                                   replica_groups);
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
