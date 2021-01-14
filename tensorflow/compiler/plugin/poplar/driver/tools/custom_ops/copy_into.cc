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

#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloCopyInto::HloCopyInto(HloInstruction* const destination,
                         HloInstruction* const value)
    : HloPoplarInstruction(destination->shape(), {destination, value},
                           PoplarOp::CopyInto) {}

absl::flat_hash_set<int64> HloCopyInto::AllocatingIndices() const { return {}; }

absl::flat_hash_map<int64, int64> HloCopyInto::LayoutDependencies() const {
  return {};
}
uint64 HloCopyInto::NumberOfInplaceOperands() const { return 1; }

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
