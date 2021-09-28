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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/ipu_inter_copy.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloIpuInterCopy::HloIpuInterCopy(absl::Span<HloInstruction* const> operands)
    : HloPoplarInstruction(GetHloPoplarInstructionShape(operands), operands,
                           PoplarOp::IpuInterCopy) {}

absl::flat_hash_set<int64> HloIpuInterCopy::AllocatingIndices() const {
  return {};
}

bool HloIpuInterCopy::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64> HloIpuInterCopy::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloIpuInterCopy::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloIpuInterCopy::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloIpuInterCopy::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloIpuInterCopy::AllowNonInplaceLowering() const { return false; }

bool HloIpuInterCopy::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction> HloIpuInterCopy::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return CreateIpuInterCopy(new_operands);
}

std::vector<std::string> HloIpuInterCopy::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateIpuInterCopy(
    absl::Span<HloInstruction* const> operands) {
  return absl::make_unique<HloIpuInterCopy>(operands);
}

}  // namespace poplarplugin
}  // namespace xla
