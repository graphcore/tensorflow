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

absl::flat_hash_map<int64, int64>
HloSuggestRecomputeInstruction::LayoutDependencies() const {
  return {};
}

uint64 HloSuggestRecomputeInstruction::NumberOfInplaceOperands() const {
  return 1;
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

absl::flat_hash_map<int64, int64>
HloBlockRecomputeInstruction::LayoutDependencies() const {
  return {};
}

uint64 HloBlockRecomputeInstruction::NumberOfInplaceOperands() const {
  return 1;
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
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
