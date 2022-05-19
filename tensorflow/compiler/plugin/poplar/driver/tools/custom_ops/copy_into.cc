/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/copy_into.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloCopyInto::HloCopyInto(HloInstruction* const destination,
                         HloInstruction* const value)
    : HloPoplarInstruction(destination->shape(), {destination, value},
                           PoplarOp::CopyInto) {}

absl::flat_hash_set<int64_t> HloCopyInto::AllocatingIndices() const {
  return {};
}

bool HloCopyInto::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64_t, int64_t> HloCopyInto::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloCopyInto::GetUseDescriptions() const {
  return UseDescriptionsSimpleNoTuple0thOperandAliasing(this);
}

HloPoplarBufferDescriptions HloCopyInto::GetBufferDescriptions() const {
  return BufferDescriptionsNoAllocations();
}

const FindConsumersExtensionResults HloCopyInto::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloCopyInto::AllowNonInplaceLowering() const { return false; }
bool HloCopyInto::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction> HloCopyInto::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  CHECK_EQ(new_operands.size(), 2);
  return absl::make_unique<HloCopyInto>(new_operands[0], new_operands[1]);
}

std::vector<std::string> HloCopyInto::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateCopyInto(
    HloInstruction* const destination, HloInstruction* const value) {
  return absl::make_unique<HloCopyInto>(destination, value);
}
}  // namespace poplarplugin
}  // namespace xla
