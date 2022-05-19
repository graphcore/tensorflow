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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/replication_factor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloReplicationFactorInstruction::HloReplicationFactorInstruction()
    : HloPoplarInstruction(
          ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<int32>(),
                               {}),
          {}, PoplarOp::ReplicationFactor) {}

absl::flat_hash_set<int64_t>
HloReplicationFactorInstruction::AllocatingIndices() const {
  return {};
}

bool HloReplicationFactorInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64_t, int64_t>
HloReplicationFactorInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloReplicationFactorInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloReplicationFactorInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults
HloReplicationFactorInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloReplicationFactorInstruction::AllowNonInplaceLowering() const {
  return false;
}

bool HloReplicationFactorInstruction::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction>
HloReplicationFactorInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloReplicationFactorInstruction>();
}

std::vector<std::string>
HloReplicationFactorInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateReplicationFactorInstruction() {
  return absl::make_unique<HloReplicationFactorInstruction>();
}

HloReplicationNormaliseInstruction::HloReplicationNormaliseInstruction(
    HloInstruction* operand)
    : HloPoplarInstruction(operand->shape(), {operand},
                           PoplarOp::ReplicationNormalise) {}

const HloInstruction* HloReplicationNormaliseInstruction::input() const {
  return operand(0);
}

absl::flat_hash_set<int64_t>
HloReplicationNormaliseInstruction::AllocatingIndices() const {
  return {};
}

bool HloReplicationNormaliseInstruction::AllocatingOutput() const {
  return false;
}

absl::flat_hash_map<int64_t, int64_t>
HloReplicationNormaliseInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions
HloReplicationNormaliseInstruction::GetUseDescriptions() const {
  return UseDescriptionsSimpleNoTuple0thOperandAliasing(this);
}

HloPoplarBufferDescriptions
HloReplicationNormaliseInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsNoAllocations();
}

const FindConsumersExtensionResults
HloReplicationNormaliseInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloReplicationNormaliseInstruction::AllowNonInplaceLowering() const {
  return false;
}

bool HloReplicationNormaliseInstruction::IsPopOpsElementwise() const {
  return true;
}

std::unique_ptr<HloInstruction>
HloReplicationNormaliseInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateReplicationNormalise(new_operands[0]);
}

std::unique_ptr<HloInstruction> CreateReplicationNormalise(
    HloInstruction* operand) {
  return absl::make_unique<HloReplicationNormaliseInstruction>(operand);
}

std::vector<std::string>
HloReplicationNormaliseInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>>
HloReplicationNormaliseInstructionFactoryFunc(HloCustomCallInstruction* call) {
  return CreateReplicationNormalise(call->mutable_operand(0));
}

StatusOr<std::unique_ptr<HloInstruction>>
HloReplicationFactorInstructionFactoryFunc(HloCustomCallInstruction* call) {
  return CreateReplicationFactorInstruction();
}

static HloPoplarInstructionFactory replication_factor_factory(
    PoplarOp::ReplicationFactor, HloReplicationFactorInstructionFactoryFunc);

static HloPoplarInstructionFactory replication_normalise_factory(
    PoplarOp::ReplicationNormalise,
    HloReplicationNormaliseInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
