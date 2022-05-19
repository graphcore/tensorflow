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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/assert.h"
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

// Constructor.
HloAssert::HloAssert(HloInstruction* predicate)
    : HloPoplarInstruction(ShapeUtil::MakeTokenShape(), {predicate},
                           PoplarOp::Assert) {
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64_t> HloAssert::AllocatingIndices() const { return {}; }

absl::flat_hash_map<int64_t, int64_t> HloAssert::LayoutDependencies() const {
  return {};
}

const FindConsumersExtensionResults HloAssert::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloAssert::AllowNonInplaceLowering() const { return false; }

bool HloAssert::IsPopOpsElementwise() const { return false; }

bool HloAssert::AllocatingOutput() const { return false; }

HloPoplarUseDescriptions HloAssert::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloAssert::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

// Creates an instance of a HloOneHotInstruction
std::unique_ptr<HloInstruction> CreateHloAssert(HloInstruction* predicate) {
  return absl::make_unique<HloAssert>(predicate);
}

std::unique_ptr<HloInstruction> HloAssert::CloneWithNewOperandsImpl(
    const Shape&, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateHloAssert(operands[0]);
}

std::vector<std::string> HloAssert::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  return attributes;
}

namespace {

static HloPoplarInstructionFactory assert_factory(
    PoplarOp::Assert,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      return CreateHloAssert(call->mutable_operand(0));
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
