/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/reduce_scatter.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloReduceScatterInstruction::HloReduceScatterInstruction(
    const Shape shape, absl::Span<HloInstruction* const> inputs,
    CollectiveOperator op, PoplarReplicaGroups replica_groups)
    : HloPoplarInstruction(shape, inputs, PoplarOp::ReduceScatter, op,
                           replica_groups),
      op_(op),
      replica_groups_(replica_groups) {}

absl::flat_hash_set<int64_t> HloReduceScatterInstruction::AllocatingIndices()
    const {
  return {};
}

bool HloReduceScatterInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64_t, int64_t>
HloReduceScatterInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloReduceScatterInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloReduceScatterInstruction::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloReduceScatterInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloReduceScatterInstruction::AllowNonInplaceLowering() const {
  return false;
}

bool HloReduceScatterInstruction::IsPopOpsElementwise() const { return false; }

PoplarReplicaGroups HloReduceScatterInstruction::GetPoplarReplicaGroups()
    const {
  return replica_groups_;
}

std::unique_ptr<HloInstruction> CreateReduceScatter(
    const Shape& shape, absl::Span<HloInstruction* const> inputs,
    CollectiveOperator op, PoplarReplicaGroups replica_groups) {
  return absl::make_unique<HloReduceScatterInstruction>(shape, inputs, op,
                                                        replica_groups);
}

std::unique_ptr<HloInstruction>
HloReduceScatterInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateReduceScatter(shape, operands, op_, replica_groups_);
}

std::vector<std::string>
HloReduceScatterInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back(absl::StrCat("op=", CollectiveOperator_Name(op_)));
  attributes.push_back("replica_groups=" + replica_groups_.ToString());
  return attributes;
}

CollectiveOperator HloReduceScatterInstruction::GetCollectiveOperator() const {
  return op_;
}

namespace {

static HloPoplarInstructionFactory reduce_scatter_factory(
    PoplarOp::ReduceScatter,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

      CollectiveOperator op;
      TF_ASSIGN_OR_RETURN(std::string op_string,
                          attribute_map.GetAttributeAsString("op"));
      if (!CollectiveOperator_Parse(op_string, &op)) {
        return InternalError("Failed to parse reduce-scatter `op` attribute.");
      }

      TF_ASSIGN_OR_RETURN(
          const auto replica_group_size,
          attribute_map.GetAttributeAsInt64("replica_group_size"));
      const auto replica_groups =
          PoplarReplicaGroups::Consecutive(replica_group_size);

      return CreateReduceScatter(call->shape(), call->operands(), op,
                                 replica_groups);
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
