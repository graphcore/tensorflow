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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/replication_index.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

HloReplicationIndexInstruction::HloReplicationIndexInstruction()
    : HloPoplarInstruction(ShapeUtil::MakeShape(S32, {}), {},
                           PoplarOp::ReplicationIndex) {}

absl::flat_hash_set<int64> HloReplicationIndexInstruction::AllocatingIndices()
    const {
  return {};
}

bool HloReplicationIndexInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64>
HloReplicationIndexInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloReplicationIndexInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloReplicationIndexInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults
HloReplicationIndexInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloReplicationIndexInstruction::AllowNonInplaceLowering() const {
  return false;
}

bool HloReplicationIndexInstruction::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction>
HloReplicationIndexInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloReplicationIndexInstruction>();
}

std::vector<std::string>
HloReplicationIndexInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateReplicationIndex() {
  return absl::make_unique<HloReplicationIndexInstruction>();
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>>
HloReplicationIndexInstructionFactoryFunc(HloCustomCallInstruction* call) {
  return CreateReplicationIndex();
}

static HloPoplarInstructionFactory fifo_factory(
    PoplarOp::ReplicationIndex, HloReplicationIndexInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
