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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/send_recv_barrier.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloSendRecvBarrierInstruction::HloSendRecvBarrierInstruction()
    : HloPoplarInstruction(ShapeUtil::MakeTokenShape(), {},
                           PoplarOp::SendRecvBarrier) {
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64> HloSendRecvBarrierInstruction::AllocatingIndices()
    const {
  return {};
}

bool HloSendRecvBarrierInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64>
HloSendRecvBarrierInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloSendRecvBarrierInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloSendRecvBarrierInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults
HloSendRecvBarrierInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloSendRecvBarrierInstruction::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction> CreateSendRecvBarrier() {
  return absl::make_unique<HloSendRecvBarrierInstruction>();
}

std::unique_ptr<HloInstruction>
HloSendRecvBarrierInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateSendRecvBarrier();
}

std::vector<std::string>
HloSendRecvBarrierInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

namespace {

static HloPoplarInstructionFactory send_recv_barrier_factory(
    PoplarOp::SendRecvBarrier,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      return CreateSendRecvBarrier();
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
