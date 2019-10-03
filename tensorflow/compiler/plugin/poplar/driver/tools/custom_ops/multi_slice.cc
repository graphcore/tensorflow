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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"

#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

HloMultiSliceInstruction::HloMultiSliceInstruction(
    const Shape& shape, HloInstruction* const input,
    HloInstruction* const indices)
    : HloPoplarInstruction(shape, {input, indices},
                           GetPoplibsCustomOpTargetString(
                               PoplibsOp::Popops, PoplibsOp::MultiSlice)) {}

absl::flat_hash_set<int64> HloMultiSliceInstruction::AllocatingIndices() const {
  return {0, 1};
}

absl::flat_hash_map<int64, int64> HloMultiSliceInstruction::LayoutDependencies()
    const {
  return {};
}

uint64 HloMultiSliceInstruction::NumberOfInplaceOperands() const { return 0; }

bool HloMultiSliceInstruction::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction>
HloMultiSliceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateMultiSlice(shape, new_operands[0], new_operands[1]);
}

std::vector<std::string>
HloMultiSliceInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateMultiSlice(
    const Shape& shape, HloInstruction* const input,
    HloInstruction* const indices) {
  return absl::make_unique<HloMultiSliceInstruction>(shape, input, indices);
}

HloMultiUpdateInstruction::HloMultiUpdateInstruction(
    const Shape& shape, HloInstruction* const input,
    HloInstruction* const gradient, HloInstruction* const indices)
    : HloPoplarInstruction(shape, {input, gradient, indices},
                           GetPoplibsCustomOpTargetString(
                               PoplibsOp::Popops, PoplibsOp::MultiUpdate)) {}

absl::flat_hash_set<int64> HloMultiUpdateInstruction::AllocatingIndices()
    const {
  return {0, 1, 2};
}

absl::flat_hash_map<int64, int64>
HloMultiUpdateInstruction::LayoutDependencies() const {
  return {};
}

uint64 HloMultiUpdateInstruction::NumberOfInplaceOperands() const { return 1; }

bool HloMultiUpdateInstruction::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction>
HloMultiUpdateInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateMultiUpdate(shape, new_operands[0], new_operands[1],
                           new_operands[2]);
}

std::vector<std::string>
HloMultiUpdateInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateMultiUpdate(
    const Shape& shape, HloInstruction* const input,
    HloInstruction* const gradient, HloInstruction* const indices) {
  return absl::make_unique<HloMultiUpdateInstruction>(shape, input, gradient,
                                                      indices);
}
namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloMultiSliceInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateMultiSlice(call->shape(), call->mutable_operand(0),
                          call->mutable_operand(1));
}

StatusOr<std::unique_ptr<HloInstruction>> HloMultiUpdateInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateMultiUpdate(call->shape(), call->mutable_operand(0),
                           call->mutable_operand(1), call->mutable_operand(2));
}
static HloPoplarInstructionFactory multi_slice_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popops, PoplibsOp::MultiSlice),
    HloMultiSliceInstructionFactoryFunc);

static HloPoplarInstructionFactory multi_update_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popops, PoplibsOp::MultiUpdate),
    HloMultiUpdateInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
