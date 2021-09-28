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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/reduce_many.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

// Constructor.
HloReduceManyInstruction::HloReduceManyInstruction(
    std::vector<HloInstruction*> inputs, const Shape& output_shape,
    const std::vector<ReductionInfo>& reductions_info)
    : HloPoplarInstruction(output_shape, inputs, PoplarOp::ReduceMany),
      reductions_info_(reductions_info) {}

absl::flat_hash_set<int64> HloReduceManyInstruction::AllocatingIndices() const {
  return {};
}

bool HloReduceManyInstruction::AllocatingOutput() const { return true; }

absl::flat_hash_map<int64, int64> HloReduceManyInstruction::LayoutDependencies()
    const {
  return {};
}

HloPoplarUseDescriptions HloReduceManyInstruction::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloReduceManyInstruction::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloReduceManyInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloReduceManyInstruction::AllowNonInplaceLowering() const { return false; }
bool HloReduceManyInstruction::IsPopOpsElementwise() const { return false; }

const std::vector<ReductionInfo>& HloReduceManyInstruction::ReductionsInfo()
    const {
  return reductions_info_;
}

// Creates an instance of a HloReduceManyInstruction
std::unique_ptr<HloInstruction> CreatePoplarReduceMany(
    std::vector<HloInstruction*> inputs, const Shape& output_shape,
    const std::vector<ReductionInfo>& reductions_info) {
  return absl::make_unique<HloReduceManyInstruction>(inputs, output_shape,
                                                     reductions_info);
}

std::unique_ptr<HloInstruction>
HloReduceManyInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreatePoplarReduceMany({operands.begin(), operands.end()}, shape,
                                reductions_info_);
}

std::vector<std::string>
HloReduceManyInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

}  // namespace poplarplugin
}  // namespace xla
