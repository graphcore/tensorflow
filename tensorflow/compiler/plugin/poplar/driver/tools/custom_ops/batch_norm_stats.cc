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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/norm.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/tf2xla/type_util.h"

namespace xla {
namespace poplarplugin {

HloBatchNormStatsInstruction::HloBatchNormStatsInstruction(
    const Shape& shape, HloInstruction* const operand, float epsilon,
    int feature_index)
    : HloNormInstruction(shape, {operand}, PoplarOp::BatchNormStatistics,
                         epsilon, feature_index) {}

absl::flat_hash_set<int64> HloBatchNormStatsInstruction::AllocatingIndices()
    const {
  return {};
}

bool HloBatchNormStatsInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64>
HloBatchNormStatsInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloBatchNormStatsInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloBatchNormStatsInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloBatchNormStatsInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloBatchNormStatsInstruction::AllowNonInplaceLowering() const {
  return false;
}
bool HloBatchNormStatsInstruction::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction>
HloBatchNormStatsInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateBatchNormStats(shape, new_operands[0], epsilon(),
                              feature_index());
}

std::unique_ptr<HloInstruction> CreateBatchNormStats(
    const Shape& shape, HloInstruction* const operand, float epsilon,
    int feature_index) {
  return absl::make_unique<HloBatchNormStatsInstruction>(
      shape, operand, epsilon, feature_index);
}

}  // namespace poplarplugin
}  // namespace xla
