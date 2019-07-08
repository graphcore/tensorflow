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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/sparse.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"

namespace xla {
namespace poplarplugin {

HloSelectScalarFromRowsInstruction::HloSelectScalarFromRowsInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands)
    : HloPoplarInstruction(
          shape, operands,
          GetPoplibsCustomOpTargetString(PoplibsOp::Popops,
                                         PoplibsOp::SelectScalarFromRows)) {}

absl::flat_hash_set<int64>
HloSelectScalarFromRowsInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64>
HloSelectScalarFromRowsInstruction::LayoutDependencies() const {
  return {};
}

uint64 HloSelectScalarFromRowsInstruction::NumberOfInplaceOperands() const {
  return 0;
}

bool HloSelectScalarFromRowsInstruction::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction>
HloSelectScalarFromRowsInstruction::CloneWithNewOperandsImpl(
    const Shape& new_shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloSelectScalarFromRowsInstruction>(new_shape,
                                                               new_operands);
}

std::unique_ptr<HloInstruction> CreateSelectScalarFromRows(
    const Shape& shape, absl::Span<HloInstruction* const> operands) {
  return absl::make_unique<HloSelectScalarFromRowsInstruction>(shape, operands);
}

HloUpdateScalarInRowsInstruction::HloUpdateScalarInRowsInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands)
    : HloPoplarInstruction(
          shape, operands,
          GetPoplibsCustomOpTargetString(PoplibsOp::Popops,
                                         PoplibsOp::UpdateScalarInRows)) {}

absl::flat_hash_set<int64> HloUpdateScalarInRowsInstruction::AllocatingIndices()
    const {
  return {};
}

absl::flat_hash_map<int64, int64>
HloUpdateScalarInRowsInstruction::LayoutDependencies() const {
  return {};
}

uint64 HloUpdateScalarInRowsInstruction::NumberOfInplaceOperands() const {
  return 1;
}

bool HloUpdateScalarInRowsInstruction::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction>
HloUpdateScalarInRowsInstruction::CloneWithNewOperandsImpl(
    const Shape& new_shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloUpdateScalarInRowsInstruction>(new_shape,
                                                             new_operands);
}

std::unique_ptr<HloInstruction> CreateUpdateScalarInRows(
    const Shape& shape, absl::Span<HloInstruction* const> operands) {
  return absl::make_unique<HloUpdateScalarInRowsInstruction>(shape, operands);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>>
HloSelectScalarFromRowsInstructionFactoryFunc(HloCustomCallInstruction* call) {
  return CreateSelectScalarFromRows(
      call->shape(), {call->mutable_operand(0), call->mutable_operand(1)});
}

static HloPoplarInstructionFactory select_scalar_from_rows_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popops,
                                   PoplibsOp::SelectScalarFromRows),
    HloSelectScalarFromRowsInstructionFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>>
HloUpdateScalarInRowsInstructionFactoryFunc(HloCustomCallInstruction* call) {
  return CreateUpdateScalarInRows(
      call->shape(), {call->mutable_operand(0), call->mutable_operand(1)});
}

static HloPoplarInstructionFactory update_scalar_in_rows_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popops,
                                   PoplibsOp::UpdateScalarInRows),
    HloUpdateScalarInRowsInstructionFactoryFunc);

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
