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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/collective_reorder.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

namespace xla {
namespace poplarplugin {

HloBaseCollectiveReorderInstruction::HloBaseCollectiveReorderInstruction(
    PoplarOp op, HloInstruction* operand)
    : HloPoplarInstruction(operand->shape(), {operand}, op) {}

std::vector<std::string>
HloBaseCollectiveReorderInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

absl::flat_hash_set<int64>
HloBaseCollectiveReorderInstruction::AllocatingIndices() const {
  return {};
}

bool HloBaseCollectiveReorderInstruction::AllocatingOutput() const {
  return false;
}

absl::flat_hash_map<int64, int64>
HloBaseCollectiveReorderInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions
HloBaseCollectiveReorderInstruction::GetUseDescriptions() const {
  return UseDescriptionsSimpleNoTuple0thOperandAliasing(this);
}

HloPoplarBufferDescriptions
HloBaseCollectiveReorderInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults
HloBaseCollectiveReorderInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloBaseCollectiveReorderInstruction::IsPopOpsElementwise() const {
  return false;
}

HloCollectiveReorderInstruction::HloCollectiveReorderInstruction(
    HloInstruction* operand)
    : HloBaseCollectiveReorderInstruction(PoplarOp::CollectiveReorder,
                                          operand) {}

std::unique_ptr<HloInstruction>
HloCollectiveReorderInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloCollectiveReorderInstruction>(new_operands[0]);
}

std::unique_ptr<HloInstruction> CreateCollectiveReorderInstruction(
    HloInstruction* operand) {
  return absl::make_unique<HloCollectiveReorderInstruction>(operand);
}

HloUndoCollectiveReorderInstruction::HloUndoCollectiveReorderInstruction(
    HloInstruction* operand)
    : HloBaseCollectiveReorderInstruction(PoplarOp::UndoCollectiveReorder,
                                          operand) {}

std::unique_ptr<HloInstruction>
HloUndoCollectiveReorderInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloUndoCollectiveReorderInstruction>(
      new_operands[0]);
}

std::unique_ptr<HloInstruction> CreateUndoCollectiveReorderInstruction(
    HloInstruction* operand) {
  return absl::make_unique<HloUndoCollectiveReorderInstruction>(operand);
}

}  // namespace poplarplugin
}  // namespace xla
