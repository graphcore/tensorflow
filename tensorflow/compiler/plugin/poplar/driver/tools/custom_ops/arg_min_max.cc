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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/arg_min_max.h"

#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

#include "absl/container/flat_hash_map.h"

namespace xla {
namespace poplarplugin {

HloArgMinMaxBase::HloArgMinMaxBase(HloInstruction* input,
                                   const Shape& output_shape, int64 axis,
                                   const PoplarOp& opcode)
    : HloPoplarInstruction(output_shape, {input}, opcode, axis), axis(axis) {}

absl::flat_hash_set<int64> HloArgMinMaxBase::AllocatingIndices() const {
  return {};
}

bool HloArgMinMaxBase::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64> HloArgMinMaxBase::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloArgMinMaxBase::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloArgMinMaxBase::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloArgMinMaxBase::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloArgMinMaxBase::IsPopOpsElementwise() const { return false; }

std::vector<std::string> HloArgMinMaxBase::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<string> attributes;
  attributes.push_back(absl::StrCat("axis=", axis));

  return attributes;
}

std::unique_ptr<HloInstruction> HloArgMax::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  CHECK_EQ(operands.size(), 1);
  return CreateHloArgMax(operands[0], shape, Axis());
}

std::unique_ptr<HloInstruction> CreateHloArgMax(HloInstruction* input,
                                                const Shape& shape,
                                                int64 axis) {
  return absl::make_unique<HloArgMax>(input, shape, axis);
}

std::unique_ptr<HloInstruction> HloArgMin::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  CHECK_EQ(operands.size(), 1);
  return CreateHloArgMin(operands[0], shape, Axis());
}

std::unique_ptr<HloInstruction> CreateHloArgMin(HloInstruction* input,
                                                const Shape& shape,
                                                int64 axis) {
  return absl::make_unique<HloArgMin>(input, shape, axis);
}

std::unique_ptr<HloInstruction> HloMaxAndArgMax::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  CHECK_EQ(operands.size(), 1);
  return CreateHloMaxAndArgMax(operands[0], shape, Axis());
}

std::unique_ptr<HloInstruction> CreateHloMaxAndArgMax(HloInstruction* input,
                                                      const Shape& shape,
                                                      int64 axis) {
  return absl::make_unique<HloMaxAndArgMax>(input, shape, axis);
}

std::unique_ptr<HloInstruction> HloMinAndArgMin::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  CHECK_EQ(operands.size(), 1);
  return CreateHloMinAndArgMin(operands[0], shape, Axis());
}

std::unique_ptr<HloInstruction> CreateHloMinAndArgMin(HloInstruction* input,
                                                      const Shape& shape,
                                                      int64 axis) {
  return absl::make_unique<HloMinAndArgMin>(input, shape, axis);
}

namespace {

static HloPoplarInstructionFactory argmax_factory(
    PoplarOp::ArgMax,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
      TF_ASSIGN_OR_RETURN(int64 axis, attribute_map.GetAttributeAsInt("axis"));

      return CreateHloArgMax(call->mutable_operand(0), call->shape(), axis);
    });

static HloPoplarInstructionFactory argmin_factory(
    PoplarOp::ArgMin,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
      TF_ASSIGN_OR_RETURN(int64 axis, attribute_map.GetAttributeAsInt("axis"));

      return CreateHloArgMin(call->mutable_operand(0), call->shape(), axis);
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
