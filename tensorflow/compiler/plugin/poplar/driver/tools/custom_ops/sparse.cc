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
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloSelectScalarFromRowsInstruction::HloSelectScalarFromRowsInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands)
    : HloPoplarInstruction(shape, operands, PoplarOp::SelectScalarFromRows) {}

absl::flat_hash_set<int64>
HloSelectScalarFromRowsInstruction::AllocatingIndices() const {
  return {};
}

bool HloSelectScalarFromRowsInstruction::AllocatingOutput() const {
  return false;
}

absl::flat_hash_map<int64, int64>
HloSelectScalarFromRowsInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions
HloSelectScalarFromRowsInstruction::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloSelectScalarFromRowsInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults
HloSelectScalarFromRowsInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloSelectScalarFromRowsInstruction::AllowNonInplaceLowering() const {
  return false;
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

std::vector<std::string>
HloSelectScalarFromRowsInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateSelectScalarFromRows(
    const Shape& shape, absl::Span<HloInstruction* const> operands) {
  return absl::make_unique<HloSelectScalarFromRowsInstruction>(shape, operands);
}

HloUpdateScalarInRowsInstruction::HloUpdateScalarInRowsInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands)
    : HloPoplarInstruction(shape, operands, PoplarOp::UpdateScalarInRows) {}

absl::flat_hash_set<int64> HloUpdateScalarInRowsInstruction::AllocatingIndices()
    const {
  return {};
}

bool HloUpdateScalarInRowsInstruction::AllocatingOutput() const {
  return false;
}

absl::flat_hash_map<int64, int64>
HloUpdateScalarInRowsInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloUpdateScalarInRowsInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsSimpleNoTuple0thOperandAliasing(this);
}

HloPoplarBufferDescriptions
HloUpdateScalarInRowsInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsNoAllocations();
}

const FindConsumersExtensionResults
HloUpdateScalarInRowsInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloUpdateScalarInRowsInstruction::AllowNonInplaceLowering() const {
  return false;
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

std::vector<std::string>
HloUpdateScalarInRowsInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>>
HloSelectScalarFromRowsInstructionFactoryFunc(HloCustomCallInstruction* call) {
  return CreateSelectScalarFromRows(
      call->shape(), {call->mutable_operand(0), call->mutable_operand(1)});
}

static HloPoplarInstructionFactory select_scalar_from_rows_factory(
    PoplarOp::SelectScalarFromRows,
    HloSelectScalarFromRowsInstructionFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>>
HloUpdateScalarInRowsInstructionFactoryFunc(HloCustomCallInstruction* call) {
  return CreateUpdateScalarInRows(
      call->shape(), {call->mutable_operand(0), call->mutable_operand(1)});
}

static HloPoplarInstructionFactory update_scalar_in_rows_factory(
    PoplarOp::UpdateScalarInRows, HloUpdateScalarInRowsInstructionFactoryFunc);

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
