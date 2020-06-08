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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/all_gather.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

// Constructor.
HloAllGatherInstruction::HloAllGatherInstruction(HloInstruction* input,
                                                 const Shape shape)
    : HloPoplarInstruction(shape, {input}, PoplarOp::AllGather) {}

absl::flat_hash_set<int64> HloAllGatherInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloAllGatherInstruction::LayoutDependencies()
    const {
  return {};
}

uint64 HloAllGatherInstruction::NumberOfInplaceOperands() const { return 0; }

bool HloAllGatherInstruction::IsPopOpsElementwise() const { return false; }

// Creates an instance of a HloAllGatherInstruction
std::unique_ptr<HloInstruction> CreateAllGather(HloInstruction* input,
                                                const Shape& shape) {
  return absl::make_unique<HloAllGatherInstruction>(input, shape);
}

std::unique_ptr<HloInstruction>
HloAllGatherInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateAllGather(operands[0], shape);
}

std::vector<std::string>
HloAllGatherInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

namespace {

static HloPoplarInstructionFactory allgather_factory(
    PoplarOp::AllGather,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      return CreateAllGather(call->mutable_operand(0), call->shape());
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
