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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/truncated_normal.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloTruncatedNormalInstruction::HloTruncatedNormalInstruction(const Shape& shape)
    : HloPoplarInstruction(shape, {}, PoplarOp::TruncatedNormal) {
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64> HloTruncatedNormalInstruction::AllocatingIndices()
    const {
  return {};
}

bool HloTruncatedNormalInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64>
HloTruncatedNormalInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloTruncatedNormalInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloTruncatedNormalInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults
HloTruncatedNormalInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloTruncatedNormalInstruction::AllowNonInplaceLowering() const {
  return false;
}

bool HloTruncatedNormalInstruction::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction>
HloTruncatedNormalInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const>,
    HloCloneContext*) const {
  return absl::make_unique<HloTruncatedNormalInstruction>(shape);
}

std::vector<std::string>
HloTruncatedNormalInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateTruncatedNormal(const Shape& shape) {
  return absl::make_unique<HloTruncatedNormalInstruction>(shape);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>>
HloTruncatedNormalInstructionFactoryFunc(HloCustomCallInstruction* call) {
  // Decode opaque here...
  return CreateTruncatedNormal(call->shape());
}

static HloPoplarInstructionFactory truncated_normal_factory(
    PoplarOp::TruncatedNormal, HloTruncatedNormalInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
