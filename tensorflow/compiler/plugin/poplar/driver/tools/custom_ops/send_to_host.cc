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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/send_to_host.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "include/json/json.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloSendToHostInstruction::HloSendToHostInstruction(
    absl::Span<HloInstruction* const> inputs, const Shape shape,
    const std::vector<std::string>& rendezvous_keys)
    : HloPoplarInstruction(shape, inputs, PoplarOp::SendToHost),
      rendezvous_keys_(rendezvous_keys) {
  CHECK_EQ(inputs.size(), rendezvous_keys.size());
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64> HloSendToHostInstruction::AllocatingIndices() const {
  absl::flat_hash_set<int64> result;
  for (int64 i = 0; i < operand_count(); ++i) {
    result.insert(i);
  }
  return result;
}

bool HloSendToHostInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64> HloSendToHostInstruction::LayoutDependencies()
    const {
  return {};
}

HloPoplarUseDescriptions HloSendToHostInstruction::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloSendToHostInstruction::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloSendToHostInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloSendToHostInstruction::AllowNonInplaceLowering() const { return false; }

bool HloSendToHostInstruction::IsPopOpsElementwise() const { return false; }

const std::vector<std::string>& HloSendToHostInstruction::RendezvousKeys()
    const {
  return rendezvous_keys_;
}

std::unique_ptr<HloInstruction> CreateSendToHost(
    absl::Span<HloInstruction* const> inputs, const Shape& shape,
    const std::vector<std::string> rendezvous_keys) {
  return absl::make_unique<HloSendToHostInstruction>(inputs, shape,
                                                     rendezvous_keys);
}

std::unique_ptr<HloInstruction>
HloSendToHostInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateSendToHost(operands, shape, rendezvous_keys_);
}

std::unique_ptr<HloInstruction>
HloSendToHostInstruction::CloneWithNewOperandsAndRendezvousKeys(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const std::vector<std::string>& rendezvous_keys) const {
  auto cloned = CreateSendToHost(operands, shape, rendezvous_keys);
  SetupDerivedInstruction(cloned.get());
  cloned->set_raw_backend_config_string(raw_backend_config_string());
  return cloned;
}

std::vector<std::string>
HloSendToHostInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  Json::Value rendezvous_keys_val;
  for (const std::string& key : rendezvous_keys_) {
    rendezvous_keys_val.append(Json::Value(key));
  }

  Json::FastWriter writer;
  writer.omitEndingLineFeed();
  return {absl::StrCat("rendezvous_keys=", writer.write(rendezvous_keys_val))};
}

namespace {

static HloPoplarInstructionFactory send_to_host_factory(
    PoplarOp::SendToHost,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attributes = IPUCustomKernelsUtil::AttributeMap(call);

      TF_ASSIGN_OR_RETURN(const std::string rendezvous_key,
                          attributes.GetAttributeAsString("rendezvous_key"));

      return CreateSendToHost(call->operands(), call->shape(),
                              {rendezvous_key});
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
