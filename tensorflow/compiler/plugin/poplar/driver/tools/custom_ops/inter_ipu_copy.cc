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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/inter_ipu_copy.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloInterIpuCopy::HloInterIpuCopy(absl::Span<HloInstruction* const> operands)
    : HloPoplarInstruction(GetHloPoplarInstructionShape(operands), operands,
                           PoplarOp::InterIpuCopy) {}

absl::flat_hash_set<int64_t> HloInterIpuCopy::AllocatingIndices() const {
  return {};
}

bool HloInterIpuCopy::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64_t, int64_t> HloInterIpuCopy::LayoutDependencies()
    const {
  return {};
}

HloPoplarUseDescriptions HloInterIpuCopy::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloInterIpuCopy::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloInterIpuCopy::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloInterIpuCopy::AllowNonInplaceLowering() const { return false; }

bool HloInterIpuCopy::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction> HloInterIpuCopy::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return CreateInterIpuCopy(new_operands);
}

std::vector<std::string> HloInterIpuCopy::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateInterIpuCopy(
    absl::Span<HloInstruction* const> operands) {
  return absl::make_unique<HloInterIpuCopy>(operands);
}

static HloPoplarInstructionFactory inter_ipu_copy_factory(
    PoplarOp::InterIpuCopy, [](HloCustomCallInstruction* inst) {
      return CreateInterIpuCopy(inst->operands());
    });
}  // namespace poplarplugin
}  // namespace xla
