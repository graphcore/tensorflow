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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remap_deduce.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloRemapDeduceInstruction::HloRemapDeduceInstruction(HloInstruction* operand)
    : HloPoplarInstruction(operand->shape(), {operand}, PoplarOp::RemapDeduce) {
}

const HloInstruction* HloRemapDeduceInstruction::input() const {
  return operand(0);
}

absl::flat_hash_set<int64_t> HloRemapDeduceInstruction::AllocatingIndices()
    const {
  return {};
}

bool HloRemapDeduceInstruction::AllocatingOutput() const { return true; }

absl::flat_hash_map<int64_t, int64_t>
HloRemapDeduceInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloRemapDeduceInstruction::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloRemapDeduceInstruction::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloRemapDeduceInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloRemapDeduceInstruction::AllowNonInplaceLowering() const {
  return false;
}

bool HloRemapDeduceInstruction::IsPopOpsElementwise() const { return true; }

std::unique_ptr<HloInstruction>
HloRemapDeduceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateRemapDeduce(new_operands[0]);
}

std::unique_ptr<HloInstruction> CreateRemapDeduce(HloInstruction* operand) {
  return absl::make_unique<HloRemapDeduceInstruction>(operand);
}

std::vector<std::string>
HloRemapDeduceInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloRemapDeduceInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateRemapDeduce(call->mutable_operand(0));
}

static HloPoplarInstructionFactory remap_deduce_factory(
    PoplarOp::RemapDeduce, HloRemapDeduceInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
