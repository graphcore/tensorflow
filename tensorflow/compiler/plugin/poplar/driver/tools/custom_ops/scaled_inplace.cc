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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/scaled_inplace.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {
HloScaledInplaceBase::HloScaledInplaceBase(
    absl::Span<HloInstruction* const> operands, HloOpcode operation,
    PoplarOp poplar_op)
    : HloPoplarInstruction(operands[0]->shape(), operands, poplar_op,
                           operation),
      operation_(operation) {}

absl::flat_hash_set<int64> HloScaledInplaceBase::AllocatingIndices() const {
  return {};
}

bool HloScaledInplaceBase::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64> HloScaledInplaceBase::LayoutDependencies()
    const {
  return {{0, 1}, {1, 0}};
}

HloPoplarUseDescriptions HloScaledInplaceBase::GetUseDescriptions() const {
  return UseDescriptionsSimpleNoTuple0thOperandAliasing(this);
}

HloPoplarBufferDescriptions HloScaledInplaceBase::GetBufferDescriptions()
    const {
  return BufferDescriptionsNoAllocations();
}

bool HloScaledInplaceBase::IsPopOpsElementwise() const { return true; }

HloOpcode HloScaledInplaceBase::GetOperation() const { return operation_; }

std::vector<std::string>
HloScaledInplaceBase::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back(absl::StrCat("operation=", HloOpcodeString(operation_)));
  return attributes;
}

HloScaledInplaceXbY::HloScaledInplaceXbY(HloInstruction* const x,
                                         HloInstruction* const y,
                                         HloInstruction* const scale,
                                         HloOpcode operation)
    : HloScaledInplaceBase({x, y, scale}, operation,
                           PoplarOp::ScaledInplaceXbY) {}

std::unique_ptr<HloInstruction> HloScaledInplaceXbY::CloneWithNewOperandsImpl(
    const Shape&, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  CHECK_EQ(new_operands.size(), 3);
  return CreateScaledInplaceXbY(new_operands[0], new_operands[1],
                                new_operands[2], GetOperation());
}

std::unique_ptr<HloInstruction> CreateScaledInplaceXbY(
    HloInstruction* const x, HloInstruction* const y,
    HloInstruction* const scale, HloOpcode operation) {
  return absl::make_unique<HloScaledInplaceXbY>(x, y, scale, operation);
}

HloScaledInplaceaXbY::HloScaledInplaceaXbY(HloInstruction* const x,
                                           HloInstruction* const y,
                                           HloInstruction* const a,
                                           HloInstruction* const b,
                                           HloOpcode operation)
    : HloScaledInplaceBase({x, y, a, b}, operation,
                           PoplarOp::ScaledInplaceaXbY) {}

std::unique_ptr<HloInstruction> HloScaledInplaceaXbY::CloneWithNewOperandsImpl(
    const Shape&, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  CHECK_EQ(new_operands.size(), 4);
  return CreateScaledInplaceaXbY(new_operands[0], new_operands[1],
                                 new_operands[2], new_operands[3],
                                 GetOperation());
}

std::unique_ptr<HloInstruction> CreateScaledInplaceaXbY(HloInstruction* const x,
                                                        HloInstruction* const y,
                                                        HloInstruction* const a,
                                                        HloInstruction* const b,
                                                        HloOpcode operation) {
  return absl::make_unique<HloScaledInplaceaXbY>(x, y, a, b, operation);
}

}  // namespace poplarplugin
}  // namespace xla
