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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/gelu.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloGeluInstruction::HloGeluInstruction(HloInstruction* operand)
    : HloPoplarInstruction(operand->shape(), {operand}, PoplarOp::Gelu) {}

const HloInstruction* HloGeluInstruction::input() const { return operand(0); }

absl::flat_hash_set<int64> HloGeluInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloGeluInstruction::LayoutDependencies()
    const {
  return {};
}

uint64 HloGeluInstruction::NumberOfInplaceOperands() const { return 1; }

bool HloGeluInstruction::IsPopOpsElementwise() const { return true; }

std::unique_ptr<HloInstruction> HloGeluInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloGeluInstruction>(new_operands[0]);
}

std::vector<std::string> HloGeluInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateGelu(HloInstruction* operand) {
  return absl::make_unique<HloGeluInstruction>(operand);
}

HloGeluGradInstruction::HloGeluGradInstruction(HloInstruction* out,
                                               HloInstruction* grad)
    : HloPoplarInstruction(out->shape(), {out, grad}, PoplarOp::GeluGrad) {}

const HloInstruction* HloGeluGradInstruction::out() const { return operand(0); }

const HloInstruction* HloGeluGradInstruction::grad() const {
  return operand(1);
}

absl::flat_hash_set<int64> HloGeluGradInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloGeluGradInstruction::LayoutDependencies()
    const {
  return {};
}

uint64 HloGeluGradInstruction::NumberOfInplaceOperands() const { return 0; }

bool HloGeluGradInstruction::IsPopOpsElementwise() const { return true; }

std::unique_ptr<HloInstruction>
HloGeluGradInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloGeluGradInstruction>(new_operands[0],
                                                   new_operands[1]);
}

std::vector<std::string>
HloGeluGradInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateGeluGrad(HloInstruction* out,
                                               HloInstruction* grad) {
  return absl::make_unique<HloGeluGradInstruction>(out, grad);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloGeluInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateGelu(call->mutable_operand(0));
}

static HloPoplarInstructionFactory gelu_factory(PoplarOp::Gelu,
                                                HloGeluInstructionFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloGeluGradInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateGeluGrad(call->mutable_operand(0), call->mutable_operand(1));
}

static HloPoplarInstructionFactory gelu_grad_factory(
    PoplarOp::GeluGrad, HloGeluGradInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
