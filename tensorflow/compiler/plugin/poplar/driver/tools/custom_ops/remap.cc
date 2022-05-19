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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remap.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloRemapInstruction::HloRemapInstruction(HloInstruction* operand)
    : HloPoplarInstruction(operand->shape(), {operand}, PoplarOp::Remap) {}

const HloInstruction* HloRemapInstruction::input() const { return operand(0); }

absl::flat_hash_set<int64_t> HloRemapInstruction::AllocatingIndices() const {
  return {};
}

bool HloRemapInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64_t, int64_t> HloRemapInstruction::LayoutDependencies()
    const {
  return {};
}

HloPoplarUseDescriptions HloRemapInstruction::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloRemapInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloRemapInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloRemapInstruction::AllowNonInplaceLowering() const { return false; }

bool HloRemapInstruction::IsPopOpsElementwise() const { return true; }

std::unique_ptr<HloInstruction> HloRemapInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloRemapInstruction>(new_operands[0]);
}

std::vector<std::string> HloRemapInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateRemap(HloInstruction* operand) {
  return absl::make_unique<HloRemapInstruction>(operand);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloRemapInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateRemap(call->mutable_operand(0));
}

static HloPoplarInstructionFactory remap_factory(
    PoplarOp::Remap, HloRemapInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
