/* Copyright 2019 Graphcore Ltd

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

#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloFifoInstruction::HloFifoInstruction(HloInstruction* operand, int64 depth)
    : HloPoplarInstruction(operand->shape(), {operand}, PoplarOp::Fifo, depth),
      depth_(depth) {}

const HloInstruction* HloFifoInstruction::input() const { return operand(0); }

int64 HloFifoInstruction::depth() const { return depth_; }

absl::flat_hash_set<int64> HloFifoInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloFifoInstruction::LayoutDependencies()
    const {
  return {};
}

uint64 HloFifoInstruction::NumberOfInplaceOperands() const { return 0; }

bool HloFifoInstruction::IsPopOpsElementwise() const { return true; }

std::unique_ptr<HloInstruction> HloFifoInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloFifoInstruction>(new_operands[0], depth_);
}

std::vector<std::string> HloFifoInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back(absl::StrCat("depth=", depth_));

  return attributes;
}

std::unique_ptr<HloInstruction> CreateFifo(HloInstruction* operand,
                                           int64 depth) {
  return absl::make_unique<HloFifoInstruction>(operand, depth);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloFifoInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  TF_ASSIGN_OR_RETURN(int64 depth, attribute_map.GetAttributeAsInt("depth"));

  return CreateFifo(call->mutable_operand(0), depth);
}

static HloPoplarInstructionFactory fifo_factory(PoplarOp::Fifo,
                                                HloFifoInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
