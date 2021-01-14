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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/reduce_scatter.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloReduceScatterInstruction::HloReduceScatterInstruction(
    absl::Span<HloInstruction* const> inputs, const Shape shape)
    : HloPoplarInstruction(shape, inputs, PoplarOp::ReduceScatter) {}

absl::flat_hash_set<int64> HloReduceScatterInstruction::AllocatingIndices()
    const {
  return {};
}

absl::flat_hash_map<int64, int64>
HloReduceScatterInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloReduceScatterInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloReduceScatterInstruction::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

bool HloReduceScatterInstruction::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction> CreateReduceScatter(
    absl::Span<HloInstruction* const> inputs, const Shape& shape) {
  return absl::make_unique<HloReduceScatterInstruction>(inputs, shape);
}

std::unique_ptr<HloInstruction>
HloReduceScatterInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateReduceScatter(operands, shape);
}

std::vector<std::string>
HloReduceScatterInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

namespace {

static HloPoplarInstructionFactory reduce_scatter_factory(
    PoplarOp::ReduceScatter,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      return CreateReduceScatter(call->operands(), call->shape());
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
