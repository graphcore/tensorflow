/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/within_replicas.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

// Constructor.
HloWithinReplicaInstruction::HloWithinReplicaInstruction(
    absl::Span<HloInstruction* const> inputs, const Shape& output_shape,
    PoplarOp op)
    : HloPoplarInstruction(output_shape, inputs, op) {}

absl::flat_hash_set<int64> HloWithinReplicaInstruction::AllocatingIndices()
    const {
  return {};
}

bool HloWithinReplicaInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64>
HloWithinReplicaInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloWithinReplicaInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloWithinReplicaInstruction::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloWithinReplicaInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloWithinReplicaInstruction::AllowNonInplaceLowering() const {
  return false;
}

bool HloWithinReplicaInstruction::IsPopOpsElementwise() const { return false; }

HloAllGatherWithinReplicaInstruction::HloAllGatherWithinReplicaInstruction(
    absl::Span<HloInstruction* const> inputs, const Shape& output_shape)
    : HloWithinReplicaInstruction(inputs, output_shape,
                                  PoplarOp::AllGatherWithinReplica) {}

std::vector<std::string>
HloAllGatherWithinReplicaInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction>
HloAllGatherWithinReplicaInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreatePoplarAllGatherWithinReplica(operands, shape);
}

HloReduceScatterWithinReplicaInstruction::
    HloReduceScatterWithinReplicaInstruction(
        absl::Span<HloInstruction* const> inputs, const Shape& output_shape,
        CollectiveOperator op)
    : HloWithinReplicaInstruction(inputs, output_shape,
                                  PoplarOp::ReduceScatterWithinReplica),
      op_(op) {}

std::vector<std::string>
HloReduceScatterWithinReplicaInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back(absl::StrCat("op=", CollectiveOperator_Name(op_)));
  return attributes;
}

std::unique_ptr<HloInstruction>
HloReduceScatterWithinReplicaInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreatePoplarReduceScatterWithinReplica(operands, shape, op_);
}

std::unique_ptr<HloInstruction> CreatePoplarAllGatherWithinReplica(
    absl::Span<HloInstruction* const> inputs, const Shape& output_shape) {
  return absl::make_unique<HloAllGatherWithinReplicaInstruction>(inputs,
                                                                 output_shape);
}

std::unique_ptr<HloInstruction> CreatePoplarReduceScatterWithinReplica(
    absl::Span<HloInstruction* const> inputs, const Shape& output_shape,
    CollectiveOperator op) {
  return absl::make_unique<HloReduceScatterWithinReplicaInstruction>(
      inputs, output_shape, op);
}

namespace {
static HloPoplarInstructionFactory all_gather_factory(
    PoplarOp::AllGatherWithinReplica,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      return CreatePoplarAllGatherWithinReplica(call->operands(),
                                                call->shape());
    });

static HloPoplarInstructionFactory reduce_scatter_factory(
    PoplarOp::ReduceScatterWithinReplica,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

      CollectiveOperator op;
      TF_ASSIGN_OR_RETURN(std::string op_string,
                          attribute_map.GetAttributeAsString("op"));
      if (!CollectiveOperator_Parse(op_string, &op)) {
        return InternalError("Failed to parse reduce-scatter `op` attribute.");
      }

      return CreatePoplarReduceScatterWithinReplica(call->operands(),
                                                    call->shape(), op);
    });
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
