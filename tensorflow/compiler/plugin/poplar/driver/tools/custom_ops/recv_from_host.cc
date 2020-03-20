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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/recv_from_host.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "include/json/json.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloRecvFromHostInstruction::HloRecvFromHostInstruction(
    absl::Span<HloInstruction* const> inputs, const Shape shape,
    const std::string& rendezvous_key)
    : HloPoplarInstruction(shape, inputs, PoplarOp::RecvFromHost),
      rendezvous_key_(rendezvous_key) {}

absl::flat_hash_set<int64> HloRecvFromHostInstruction::AllocatingIndices()
    const {
  return {};
}

absl::flat_hash_map<int64, int64>
HloRecvFromHostInstruction::LayoutDependencies() const {
  return {};
}

uint64 HloRecvFromHostInstruction::NumberOfInplaceOperands() const {
  return operand_count();
}

bool HloRecvFromHostInstruction::IsPopOpsElementwise() const { return false; }

const std::string& HloRecvFromHostInstruction::RendezvousKey() const {
  return rendezvous_key_;
}

std::unique_ptr<HloInstruction> CreateRecvFromHost(
    absl::Span<HloInstruction* const> inputs, const Shape& shape,
    const std::string& rendezvous_key) {
  return absl::make_unique<HloRecvFromHostInstruction>(inputs, shape,
                                                       rendezvous_key);
}

std::unique_ptr<HloInstruction>
HloRecvFromHostInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  CHECK_LE(operands.size(), 1);
  return CreateRecvFromHost(operands, shape, rendezvous_key_);
}

std::vector<std::string>
HloRecvFromHostInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  Json::FastWriter writer;
  writer.omitEndingLineFeed();
  return {absl::StrCat("rendezvous_key=",
                       writer.write(Json::Value(rendezvous_key_)))};
}

namespace {

static HloPoplarInstructionFactory recv_from_host_factory(
    PoplarOp::RecvFromHost,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attributes = IPUCustomKernelsUtil::AttributeMap(call);

      TF_ASSIGN_OR_RETURN(const std::string rendezvous_key,
                          attributes.GetAttributeAsString("rendezvous_key"));

      return CreateRecvFromHost(call->operands(), call->shape(),
                                rendezvous_key);
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
