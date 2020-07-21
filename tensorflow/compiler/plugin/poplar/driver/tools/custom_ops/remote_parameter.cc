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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

HloRemoteParameterLoad::HloRemoteParameterLoad(const Shape& shape,
                                               int64 param_number)
    : HloPoplarInstruction(shape, {}, PoplarOp::RemoteParameterLoad,
                           param_number),
      param_number_(param_number) {
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64> HloRemoteParameterLoad::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloRemoteParameterLoad::LayoutDependencies()
    const {
  return {};
}

uint64 HloRemoteParameterLoad::NumberOfInplaceOperands() const { return 0; }

bool HloRemoteParameterLoad::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction>
HloRemoteParameterLoad::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  CHECK(operands.empty());
  return CreateHloRemoteParameterLoad(shape, param_number_);
}

std::vector<std::string>
HloRemoteParameterLoad::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("param_number=" + std::to_string(param_number_));
  return attributes;
}

std::unique_ptr<HloInstruction> CreateHloRemoteParameterLoad(
    const Shape& shape, int64 param_number) {
  return absl::make_unique<HloRemoteParameterLoad>(shape, param_number);
}

HloRemoteParameterStore::HloRemoteParameterStore(HloInstruction* const output,
                                                 int64 output_idx)
    : HloPoplarInstruction(ShapeUtil::MakeTokenShape(), {output},
                           PoplarOp::RemoteParameterStore, output_idx),
      output_idx_(output_idx) {
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64> HloRemoteParameterStore::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloRemoteParameterStore::LayoutDependencies()
    const {
  return {};
}

uint64 HloRemoteParameterStore::NumberOfInplaceOperands() const { return 1; }

bool HloRemoteParameterStore::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction>
HloRemoteParameterStore::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  CHECK_EQ(operands.size(), 1);
  return CreateHloRemoteParameterStore(operands[0], output_idx_);
}

std::vector<std::string>
HloRemoteParameterStore::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("output_idx=" + std::to_string(output_idx_));
  return attributes;
}

std::unique_ptr<HloInstruction> CreateHloRemoteParameterStore(
    HloInstruction* const output, int64 output_idx) {
  return absl::make_unique<HloRemoteParameterStore>(output, output_idx);
}

HloRemoteParameterDummyOutput::HloRemoteParameterDummyOutput(const Shape& shape,
                                                             int64 output_idx)
    : HloPoplarInstruction(shape, {}, PoplarOp::RemoteParameterDummyOutput,
                           output_idx),
      output_idx_(output_idx) {
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64> HloRemoteParameterDummyOutput::AllocatingIndices()
    const {
  return {};
}

absl::flat_hash_map<int64, int64>
HloRemoteParameterDummyOutput::LayoutDependencies() const {
  return {};
}

uint64 HloRemoteParameterDummyOutput::NumberOfInplaceOperands() const {
  return 0;
}

bool HloRemoteParameterDummyOutput::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction>
HloRemoteParameterDummyOutput::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  CHECK(operands.empty());
  return CreateHloRemoteParameterDummyOutput(shape, output_idx_);
}

std::vector<std::string>
HloRemoteParameterDummyOutput::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("output_idx=" + std::to_string(output_idx_));
  return attributes;
}

std::unique_ptr<HloInstruction> CreateHloRemoteParameterDummyOutput(
    const Shape& shape, int64 output_idx) {
  return absl::make_unique<HloRemoteParameterDummyOutput>(shape, output_idx);
}

HloCreateBuffer::HloCreateBuffer(const Shape& shape, bool is_remote)
    : HloPoplarInstruction(shape, {}, PoplarOp::CreateBuffer, is_remote),
      is_remote_(is_remote) {
  CHECK(!shape.IsTuple());
  // Set the instruction to have side effect to prevent it from being merged
  // with other similarly shaped buffers.
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64> HloCreateBuffer::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloCreateBuffer::LayoutDependencies() const {
  return {};
}

uint64 HloCreateBuffer::NumberOfInplaceOperands() const { return 0; }

bool HloCreateBuffer::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction> HloCreateBuffer::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  CHECK_EQ(operands.size(), 0);
  return CreateHloCreateBuffer(shape, IsRemoteBuffer());
}

std::vector<std::string> HloCreateBuffer::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("is_remote=" + std::to_string(is_remote_));
  return attributes;
}

std::unique_ptr<HloInstruction> CreateHloCreateBuffer(const Shape& shape,
                                                      bool is_remote) {
  return absl::make_unique<HloCreateBuffer>(shape, is_remote);
}

}  // namespace poplarplugin
}  // namespace xla
