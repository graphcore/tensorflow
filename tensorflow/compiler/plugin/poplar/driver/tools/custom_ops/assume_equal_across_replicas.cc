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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/assume_equal_across_replicas.h"
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"

namespace xla {
namespace poplarplugin {

HloAssumeEqualAcrossReplicas::HloAssumeEqualAcrossReplicas(
    const Shape& shape, HloInstruction* operand)
    : HloPoplarInstruction(shape, {operand},
                           PoplarOp::AssumeEqualAcrossReplicas) {}

absl::flat_hash_set<int64> HloAssumeEqualAcrossReplicas::AllocatingIndices()
    const {
  return {};
}
absl::flat_hash_map<int64, int64>
HloAssumeEqualAcrossReplicas::LayoutDependencies() const {
  return {};
}

const FindConsumersExtensionResults HloAssumeEqualAcrossReplicas::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloAssumeEqualAcrossReplicas::AllowNonInplaceLowering() const {
  return false;
}
bool HloAssumeEqualAcrossReplicas::IsPopOpsElementwise() const { return false; }
bool HloAssumeEqualAcrossReplicas::AllocatingOutput() const { return false; }

HloPoplarUseDescriptions HloAssumeEqualAcrossReplicas::GetUseDescriptions()
    const {
  return UseDescriptionsSimpleNoTuple0thOperandAliasing(
      this, BufferUseKind::USE_ALIAS_READ_ONLY);
}

HloPoplarBufferDescriptions
HloAssumeEqualAcrossReplicas::GetBufferDescriptions() const {
  return BufferDescriptionsNoAllocations();
}

std::unique_ptr<HloInstruction> CreateHloAssumeEqualAcrossReplicas(
    const Shape& shape, HloInstruction* operand) {
  return absl::make_unique<HloAssumeEqualAcrossReplicas>(shape, operand);
}

std::unique_ptr<HloInstruction>
HloAssumeEqualAcrossReplicas::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateHloAssumeEqualAcrossReplicas(shape, operands[0]);
}

std::vector<std::string>
HloAssumeEqualAcrossReplicas::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

namespace {

static HloPoplarInstructionFactory factory(
    PoplarOp::AssumeEqualAcrossReplicas,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      return CreateHloAssumeEqualAcrossReplicas(call->shape(),
                                                call->mutable_operand(0));
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
