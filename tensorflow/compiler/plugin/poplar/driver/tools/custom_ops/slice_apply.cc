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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/slice_apply.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {
HloSliceApplyBase::HloSliceApplyBase(
    const std::vector<HloInstruction*>& operands, int64 apply_dimension,
    int64 start_index, HloOpcode operation, PoplarOp poplar_op)
    : HloPoplarInstruction(operands[0]->shape(), operands, poplar_op,
                           apply_dimension, start_index, operation),
      apply_dimension_(apply_dimension),
      start_index_(start_index),
      operation_(operation) {}

absl::flat_hash_set<int64> HloSliceApplyBase::AllocatingIndices() const {
  return {};
}

bool HloSliceApplyBase::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64> HloSliceApplyBase::LayoutDependencies()
    const {
  return {{0, 1}, {1, 0}};
}

HloPoplarUseDescriptions HloSliceApplyBase::GetUseDescriptions() const {
  return UseDescriptionsSimpleNoTuple0thOperandAliasing(this);
}

HloPoplarBufferDescriptions HloSliceApplyBase::GetBufferDescriptions() const {
  return BufferDescriptionsNoAllocations();
}

const FindConsumersExtensionResults HloSliceApplyBase::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloSliceApplyBase::IsPopOpsElementwise() const { return false; }

int64 HloSliceApplyBase::GetApplyDimension() const { return apply_dimension_; }

int64 HloSliceApplyBase::GetStartIndex() const { return start_index_; }

HloOpcode HloSliceApplyBase::GetOperation() const { return operation_; }

std::vector<std::string> HloSliceApplyBase::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back(absl::StrCat("apply_dimension=", apply_dimension_));
  attributes.push_back(absl::StrCat("start_index=", start_index_));
  attributes.push_back(absl::StrCat("operation=", HloOpcodeString(operation_)));
  return attributes;
}

// SliceApply
HloSliceApply::HloSliceApply(HloInstruction* const input,
                             HloInstruction* const update,
                             int64 apply_dimension, int64 start_index,
                             HloOpcode operation)
    : HloSliceApplyBase({input, update}, apply_dimension, start_index,
                        operation, PoplarOp::SliceApply) {}

std::unique_ptr<HloInstruction> HloSliceApply::CloneWithNewOperandsImpl(
    const Shape&, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateSliceApply(new_operands[0], new_operands[1], GetApplyDimension(),
                          GetStartIndex(), GetOperation());
}

std::unique_ptr<HloInstruction> CreateSliceApply(HloInstruction* const input,
                                                 HloInstruction* const update,
                                                 int64 apply_dimensions,
                                                 int64 start_index,
                                                 HloOpcode operation) {
  return absl::make_unique<HloSliceApply>(input, update, apply_dimensions,
                                          start_index, operation);
}

// SliceApplyaXbY
HloSliceApplyaXbY::HloSliceApplyaXbY(HloInstruction* const input,
                                     HloInstruction* const update,
                                     HloInstruction* const scale_input,
                                     HloInstruction* const scale_update,
                                     int64 apply_dimension, int64 start_index,
                                     HloOpcode operation)
    : HloSliceApplyBase({input, update, scale_input, scale_update},
                        apply_dimension, start_index, operation,
                        PoplarOp::SliceApplyaXbY) {}

std::unique_ptr<HloInstruction> HloSliceApplyaXbY::CloneWithNewOperandsImpl(
    const Shape&, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateSliceApplyaXbY(new_operands[0], new_operands[1], new_operands[2],
                              new_operands[3], GetApplyDimension(),
                              GetStartIndex(), GetOperation());
}

std::unique_ptr<HloInstruction> CreateSliceApplyaXbY(
    HloInstruction* const input, HloInstruction* const update,
    HloInstruction* const scale_input, HloInstruction* const scale_update,
    int64 apply_dimensions, int64 start_index, HloOpcode operation) {
  return absl::make_unique<HloSliceApplyaXbY>(input, update, scale_input,
                                              scale_update, apply_dimensions,
                                              start_index, operation);
}

// SliceApplyabY
HloSliceApplyabY::HloSliceApplyabY(HloInstruction* const input,
                                   HloInstruction* const update,
                                   HloInstruction* const scale_update,
                                   int64 apply_dimension, int64 start_index,
                                   HloOpcode operation)
    : HloSliceApplyBase({input, update, scale_update}, apply_dimension,
                        start_index, operation, PoplarOp::SliceApplyabY) {}

std::unique_ptr<HloInstruction> HloSliceApplyabY::CloneWithNewOperandsImpl(
    const Shape&, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateSliceApplyabY(new_operands[0], new_operands[1], new_operands[2],
                             GetApplyDimension(), GetStartIndex(),
                             GetOperation());
}

std::unique_ptr<HloInstruction> CreateSliceApplyabY(
    HloInstruction* const input, HloInstruction* const update,
    HloInstruction* const scale_update, int64 apply_dimensions,
    int64 start_index, HloOpcode operation) {
  return absl::make_unique<HloSliceApplyabY>(
      input, update, scale_update, apply_dimensions, start_index, operation);
}

// SliceApplyaXb
HloSliceApplyaXb::HloSliceApplyaXb(HloInstruction* const input,
                                   HloInstruction* const update,
                                   HloInstruction* const scale_input,
                                   int64 apply_dimension, int64 start_index,
                                   HloOpcode operation)
    : HloSliceApplyBase({input, update, scale_input}, apply_dimension,
                        start_index, operation, PoplarOp::SliceApplyaXb) {}

std::unique_ptr<HloInstruction> HloSliceApplyaXb::CloneWithNewOperandsImpl(
    const Shape&, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateSliceApplyaXb(new_operands[0], new_operands[1], new_operands[2],
                             GetApplyDimension(), GetStartIndex(),
                             GetOperation());
}

std::unique_ptr<HloInstruction> CreateSliceApplyaXb(
    HloInstruction* const input, HloInstruction* const update,
    HloInstruction* const scale_input, int64 apply_dimensions,
    int64 start_index, HloOpcode operation) {
  return absl::make_unique<HloSliceApplyaXb>(
      input, update, scale_input, apply_dimensions, start_index, operation);
}

}  // namespace poplarplugin
}  // namespace xla
