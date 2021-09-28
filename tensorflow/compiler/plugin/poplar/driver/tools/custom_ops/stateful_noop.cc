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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_noop.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloStatefulNoop::HloStatefulNoop()
    : HloPoplarInstruction(ShapeUtil::MakeTupleShape({}), {},
                           PoplarOp::StatefulNoop) {
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64> HloStatefulNoop::AllocatingIndices() const {
  return {};
}

bool HloStatefulNoop::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64> HloStatefulNoop::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloStatefulNoop::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloStatefulNoop::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloStatefulNoop::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloStatefulNoop::AllowNonInplaceLowering() const { return false; }

bool HloStatefulNoop::IsPopOpsElementwise() const { return false; }

std::vector<std::string> HloStatefulNoop::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> HloStatefulNoop::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const>,
    HloCloneContext*) const {
  return CreateStatefulNoop();
}

std::unique_ptr<HloInstruction> CreateStatefulNoop() {
  return absl::make_unique<HloStatefulNoop>();
}

}  // namespace poplarplugin
}  // namespace xla
