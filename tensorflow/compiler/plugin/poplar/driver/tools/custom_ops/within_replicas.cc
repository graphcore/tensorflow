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
HloPoplarAllGatherWithinReplicaInstruction::
    HloPoplarAllGatherWithinReplicaInstruction(
        absl::Span<HloInstruction* const> inputs, const Shape& output_shape)
    : HloPoplarInstruction(output_shape, inputs,
                           PoplarOp::AllGatherWithinReplica) {}

absl::flat_hash_set<int64>
HloPoplarAllGatherWithinReplicaInstruction::AllocatingIndices() const {
  return {};
}

bool HloPoplarAllGatherWithinReplicaInstruction::AllocatingOutput() const {
  return false;
}

absl::flat_hash_map<int64, int64>
HloPoplarAllGatherWithinReplicaInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions
HloPoplarAllGatherWithinReplicaInstruction::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloPoplarAllGatherWithinReplicaInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults
HloPoplarAllGatherWithinReplicaInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloPoplarAllGatherWithinReplicaInstruction::AllowNonInplaceLowering()
    const {
  return false;
}

bool HloPoplarAllGatherWithinReplicaInstruction::IsPopOpsElementwise() const {
  return false;
}

// Creates an instance of a HloPoplarAllGatherWithinReplicaInstruction
std::unique_ptr<HloInstruction> CreatePoplarAllGatherWithinReplica(
    absl::Span<HloInstruction* const> inputs, const Shape& output_shape) {
  return absl::make_unique<HloPoplarAllGatherWithinReplicaInstruction>(
      inputs, output_shape);
}

std::unique_ptr<HloInstruction>
HloPoplarAllGatherWithinReplicaInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreatePoplarAllGatherWithinReplica(operands, shape);
}

std::vector<std::string>
HloPoplarAllGatherWithinReplicaInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

namespace {

static HloPoplarInstructionFactory factory(
    PoplarOp::AllGatherWithinReplica,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      return CreatePoplarAllGatherWithinReplica(call->operands(),
                                                call->shape());
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
