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
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloSendToHostInstruction::HloSendToHostInstruction(
    absl::Span<HloInstruction* const> inputs, const Shape shape,
    const std::vector<std::string>& rendezvous_keys, bool concat_replicas)
    : HloPoplarInstruction(shape, inputs, PoplarOp::SendToHost),
      rendezvous_keys_(rendezvous_keys),
      concat_replicas_(concat_replicas) {
  CHECK_EQ(inputs.size(), rendezvous_keys.size());
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64> HloSendToHostInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloSendToHostInstruction::LayoutDependencies()
    const {
  return {};
}

uint64 HloSendToHostInstruction::NumberOfInplaceOperands() const { return 0; }

bool HloSendToHostInstruction::IsPopOpsElementwise() const { return false; }

const std::vector<std::string>& HloSendToHostInstruction::RendezvousKeys()
    const {
  return rendezvous_keys_;
}

bool HloSendToHostInstruction::ConcatReplicas() const {
  return concat_replicas_;
}

std::unique_ptr<HloInstruction> CreateSendToHost(
    absl::Span<HloInstruction* const> inputs, const Shape& shape,
    const std::vector<std::string> rendezvous_keys, bool concat_replicas) {
  return absl::make_unique<HloSendToHostInstruction>(
      inputs, shape, rendezvous_keys, concat_replicas);
}

std::unique_ptr<HloInstruction>
HloSendToHostInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateSendToHost(operands, shape, rendezvous_keys_, concat_replicas_);
}

std::unique_ptr<HloInstruction>
HloSendToHostInstruction::CloneWithNewOperandsAndRendezvousKeys(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const std::vector<std::string>& rendezvous_keys) const {
  auto cloned =
      CreateSendToHost(operands, shape, rendezvous_keys, concat_replicas_);
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

  Json::Value concat_replicas_val(concat_replicas_);

  Json::FastWriter writer;
  writer.omitEndingLineFeed();
  return {absl::StrCat("rendezvous_keys=", writer.write(rendezvous_keys_val)),
          absl::StrCat("concat_replicas=", writer.write(concat_replicas_val))};
}

namespace {

static HloPoplarInstructionFactory send_to_host_factory(
    PoplarOp::SendToHost,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attributes = IPUCustomKernelsUtil::AttributeMap(call);

      TF_ASSIGN_OR_RETURN(const std::string rendezvous_key,
                          attributes.GetAttributeAsString("rendezvous_key"));

      const auto concat_replicas_attr =
          attributes.GetAttributeAsBool("concat_replicas");
      const bool concat_replicas =
          concat_replicas_attr.ok() && concat_replicas_attr.ValueOrDie();

      return CreateSendToHost(call->operands(), call->shape(), {rendezvous_key},
                              concat_replicas);
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
