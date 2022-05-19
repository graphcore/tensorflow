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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/barrier.h"

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

HloBarrier::HloBarrier(const Shape& shape,
                       absl::Span<HloInstruction* const> operands)
    : HloPoplarInstruction(shape, operands, PoplarOp::Barrier) {}

absl::flat_hash_set<int64_t> HloBarrier::AllocatingIndices() const {
  return {};
}
absl::flat_hash_map<int64_t, int64_t> HloBarrier::LayoutDependencies() const {
  return {};
}

const FindConsumersExtensionResults HloBarrier::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloBarrier::AllowNonInplaceLowering() const { return false; }
bool HloBarrier::IsPopOpsElementwise() const { return false; }
bool HloBarrier::AllocatingOutput() const { return false; }

HloPoplarUseDescriptions HloBarrier::GetUseDescriptions() const {
  return UseDescriptionsForwardsBuffers(this, operand_count(),
                                        BufferUseKind::USE_ALIAS_READ_ONLY);
}

HloPoplarBufferDescriptions HloBarrier::GetBufferDescriptions() const {
  return BufferDescriptionsNoAllocations();
}

std::unique_ptr<HloInstruction> CreateHloBarrier(
    const Shape& shape, absl::Span<HloInstruction* const> operands) {
  return absl::make_unique<HloBarrier>(shape, operands);
}

std::unique_ptr<HloInstruction> HloBarrier::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateHloBarrier(shape, operands);
}

std::vector<std::string> HloBarrier::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

namespace {
static HloPoplarInstructionFactory factory(
    PoplarOp::Barrier,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      return CreateHloBarrier(call->shape(), call->operands());
    });
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
