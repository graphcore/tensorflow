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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/fifo.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloFifoInstruction::HloFifoInstruction(HloInstruction* operand, int64_t depth,
                                       bool offload)
    : HloPoplarInstruction(operand->shape(), {operand}, PoplarOp::Fifo, depth,
                           offload),
      depth_(depth),
      offload_(offload) {}

const HloInstruction* HloFifoInstruction::input() const { return operand(0); }

int64_t HloFifoInstruction::depth() const { return depth_; }

bool HloFifoInstruction::offload() const { return offload_; }

absl::flat_hash_set<int64_t> HloFifoInstruction::AllocatingIndices() const {
  return {};
}

bool HloFifoInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64_t, int64_t> HloFifoInstruction::LayoutDependencies()
    const {
  return {};
}

HloPoplarUseDescriptions HloFifoInstruction::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloFifoInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloFifoInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloFifoInstruction::AllowNonInplaceLowering() const { return false; }
bool HloFifoInstruction::IsPopOpsElementwise() const { return true; }

std::unique_ptr<HloInstruction> HloFifoInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloFifoInstruction>(new_operands[0], depth_,
                                               offload_);
}

std::vector<std::string> HloFifoInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back(absl::StrCat("depth=", depth_));
  attributes.push_back(absl::StrCat("offload=", offload_));

  return attributes;
}

std::unique_ptr<HloInstruction> CreateFifo(HloInstruction* operand,
                                           int64_t depth, bool offload) {
  return absl::make_unique<HloFifoInstruction>(operand, depth, offload);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloFifoInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  TF_ASSIGN_OR_RETURN(int64_t depth, attribute_map.GetAttributeAsInt("depth"));
  TF_ASSIGN_OR_RETURN(bool offload,
                      attribute_map.GetAttributeAsBool("offload"));

  return CreateFifo(call->mutable_operand(0), depth, offload);
}

static HloPoplarInstructionFactory fifo_factory(PoplarOp::Fifo,
                                                HloFifoInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
