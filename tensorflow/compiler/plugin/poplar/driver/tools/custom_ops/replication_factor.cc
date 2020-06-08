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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/replication_factor.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloReplicationFactorInstruction::HloReplicationFactorInstruction()
    : HloPoplarInstruction(
          ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<int32>(),
                               {}),
          {}, PoplarOp::ReplicationFactor) {}

absl::flat_hash_set<int64> HloReplicationFactorInstruction::AllocatingIndices()
    const {
  return {};
}

absl::flat_hash_map<int64, int64>
HloReplicationFactorInstruction::LayoutDependencies() const {
  return {};
}

uint64 HloReplicationFactorInstruction::NumberOfInplaceOperands() const {
  return 0;
}

bool HloReplicationFactorInstruction::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction>
HloReplicationFactorInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloReplicationFactorInstruction>();
}

std::vector<std::string>
HloReplicationFactorInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateReplicationFactorInstruction() {
  return absl::make_unique<HloReplicationFactorInstruction>();
}

HloReplicationNormaliseInstruction::HloReplicationNormaliseInstruction(
    HloInstruction* operand)
    : HloPoplarInstruction(operand->shape(), {operand},
                           PoplarOp::ReplicationNormalise) {}

const HloInstruction* HloReplicationNormaliseInstruction::input() const {
  return operand(0);
}

absl::flat_hash_set<int64>
HloReplicationNormaliseInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64>
HloReplicationNormaliseInstruction::LayoutDependencies() const {
  return {};
}

uint64 HloReplicationNormaliseInstruction::NumberOfInplaceOperands() const {
  return 1;
}

bool HloReplicationNormaliseInstruction::IsPopOpsElementwise() const {
  return true;
}

std::unique_ptr<HloInstruction>
HloReplicationNormaliseInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateReplicationNormalise(new_operands[0]);
}

std::unique_ptr<HloInstruction> CreateReplicationNormalise(
    HloInstruction* operand) {
  return absl::make_unique<HloReplicationNormaliseInstruction>(operand);
}

std::vector<std::string>
HloReplicationNormaliseInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>>
HloReplicationNormaliseInstructionFactoryFunc(HloCustomCallInstruction* call) {
  return CreateReplicationNormalise(call->mutable_operand(0));
}

StatusOr<std::unique_ptr<HloInstruction>>
HloReplicationFactorInstructionFactoryFunc(HloCustomCallInstruction* call) {
  return CreateReplicationFactorInstruction();
}

static HloPoplarInstructionFactory replication_factor_factory(
    PoplarOp::ReplicationFactor, HloReplicationFactorInstructionFactoryFunc);

static HloPoplarInstructionFactory replication_normalise_factory(
    PoplarOp::ReplicationNormalise,
    HloReplicationNormaliseInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
