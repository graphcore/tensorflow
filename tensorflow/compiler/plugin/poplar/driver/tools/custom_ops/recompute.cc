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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/recompute.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloSuggestRecomputeInstruction::HloSuggestRecomputeInstruction(
    HloInstruction* operand)
    : HloPoplarInstruction(operand->shape(), {operand},
                           PoplarOp::SuggestRecompute) {}

const HloInstruction* HloSuggestRecomputeInstruction::input() const {
  return operand(0);
}

absl::flat_hash_set<int64> HloSuggestRecomputeInstruction::AllocatingIndices()
    const {
  return {};
}

bool HloSuggestRecomputeInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64>
HloSuggestRecomputeInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloSuggestRecomputeInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsForwardsBuffers(this, 1,
                                        BufferUseKind::USE_ALIAS_READ_ONLY);
}

HloPoplarBufferDescriptions
HloSuggestRecomputeInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsNoAllocations();
}

const FindConsumersExtensionResults
HloSuggestRecomputeInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloSuggestRecomputeInstruction::IsPopOpsElementwise() const {
  return true;
}

std::unique_ptr<HloInstruction>
HloSuggestRecomputeInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloSuggestRecomputeInstruction>(new_operands[0]);
}

std::vector<std::string>
HloSuggestRecomputeInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateSuggestRecompute(
    HloInstruction* operand) {
  return absl::make_unique<HloSuggestRecomputeInstruction>(operand);
}

HloBlockRecomputeInstruction::HloBlockRecomputeInstruction(
    HloInstruction* operand)
    : HloPoplarInstruction(operand->shape(), {operand},
                           PoplarOp::BlockRecompute) {}

const HloInstruction* HloBlockRecomputeInstruction::input() const {
  return operand(0);
}

absl::flat_hash_set<int64> HloBlockRecomputeInstruction::AllocatingIndices()
    const {
  return {};
}

bool HloBlockRecomputeInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64>
HloBlockRecomputeInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloBlockRecomputeInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsForwardsBuffers(this, 1,
                                        BufferUseKind::USE_ALIAS_READ_ONLY);
}

HloPoplarBufferDescriptions
HloBlockRecomputeInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsNoAllocations();
}

const FindConsumersExtensionResults HloBlockRecomputeInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloBlockRecomputeInstruction::IsPopOpsElementwise() const { return true; }

std::unique_ptr<HloInstruction>
HloBlockRecomputeInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloBlockRecomputeInstruction>(new_operands[0]);
}

std::vector<std::string>
HloBlockRecomputeInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateBlockRecompute(HloInstruction* operand) {
  return absl::make_unique<HloBlockRecomputeInstruction>(operand);
}

HloRecomputationCheckpointInstruction::HloRecomputationCheckpointInstruction(
    HloInstruction* const operand)
    : HloPoplarInstruction(operand->shape(), {operand},
                           PoplarOp::RecomputationCheckpoint) {}

absl::flat_hash_set<int64>
HloRecomputationCheckpointInstruction::AllocatingIndices() const {
  return {};
}

bool HloRecomputationCheckpointInstruction::AllocatingOutput() const {
  return false;
}

absl::flat_hash_map<int64, int64>
HloRecomputationCheckpointInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions
HloRecomputationCheckpointInstruction::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloRecomputationCheckpointInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults
HloRecomputationCheckpointInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloRecomputationCheckpointInstruction::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction>
HloRecomputationCheckpointInstruction::CloneWithNewOperandsImpl(
    const Shape&, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloRecomputationCheckpointInstruction>(
      new_operands[0]);
}

std::vector<std::string>
HloRecomputationCheckpointInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateRecomputationCheckpoint(
    HloInstruction* const operand) {
  return absl::make_unique<HloRecomputationCheckpointInstruction>(operand);
}

HloRecomputationInputInstruction::HloRecomputationInputInstruction(
    HloInstruction* const checkpointed_input, HloInstruction* const old_input)
    : HloPoplarInstruction(checkpointed_input->shape(),
                           {checkpointed_input, old_input},
                           PoplarOp::RecomputationInput) {}

absl::flat_hash_set<int64> HloRecomputationInputInstruction::AllocatingIndices()
    const {
  return {};
}

bool HloRecomputationInputInstruction::AllocatingOutput() const {
  return false;
}

absl::flat_hash_map<int64, int64>
HloRecomputationInputInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloRecomputationInputInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloRecomputationInputInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults
HloRecomputationInputInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloRecomputationInputInstruction::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction>
HloRecomputationInputInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  CHECK_EQ(new_operands.size(), 2);
  return absl::make_unique<HloRecomputationInputInstruction>(new_operands[0],
                                                             new_operands[1]);
}

std::vector<std::string>
HloRecomputationInputInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateRecomputationInput(
    HloInstruction* const checkpointed_input, HloInstruction* const old_input) {
  return absl::make_unique<HloRecomputationInputInstruction>(checkpointed_input,
                                                             old_input);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>>
HloSuggestRecomputeInstructionFactoryFunc(HloCustomCallInstruction* call) {
  return CreateSuggestRecompute(call->mutable_operand(0));
}

static HloPoplarInstructionFactory recompute_factory(
    PoplarOp::SuggestRecompute, HloSuggestRecomputeInstructionFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>>
HloBlockRecomputeInstructionFactoryFunc(HloCustomCallInstruction* call) {
  return CreateBlockRecompute(call->mutable_operand(0));
}

static HloPoplarInstructionFactory block_factory(
    PoplarOp::BlockRecompute, HloBlockRecomputeInstructionFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloRecomputationCheckpointFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateRecomputationCheckpoint(call->mutable_operand(0));
}

static HloPoplarInstructionFactory recomputation_checkpoint_factory(
    PoplarOp::RecomputationCheckpoint, HloRecomputationCheckpointFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloRecomputationInputFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateRecomputationInput(call->mutable_operand(0),
                                  call->mutable_operand(1));
}

static HloPoplarInstructionFactory recomputation_input_factory(
    PoplarOp::RecomputationInput, HloRecomputationInputFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
