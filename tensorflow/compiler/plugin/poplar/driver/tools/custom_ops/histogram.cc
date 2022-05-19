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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/histogram.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

/*
 * Histogram.
 */
HloHistogramInstruction::HloHistogramInstruction(const Shape& shape,
                                                 HloInstruction* const input,
                                                 HloInstruction* const levels,
                                                 bool absolute_of_input)
    : HloPoplarInstruction(shape, {input, levels}, PoplarOp::Histogram,
                           absolute_of_input),
      absolute_of_input_(absolute_of_input) {}

absl::flat_hash_set<int64_t> HloHistogramInstruction::AllocatingIndices()
    const {
  return {};
}

absl::flat_hash_map<int64_t, int64_t>
HloHistogramInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloHistogramInstruction::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloHistogramInstruction::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

bool HloHistogramInstruction::AllocatingOutput() const { return false; }

const FindConsumersExtensionResults HloHistogramInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloHistogramInstruction::AllowNonInplaceLowering() const { return false; }
bool HloHistogramInstruction::IsPopOpsElementwise() const { return false; }

bool HloHistogramInstruction::AbsoluteOfInput() const {
  return absolute_of_input_;
}

std::unique_ptr<HloInstruction>
HloHistogramInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloHistogramInstruction>(
      shape, new_operands[0], new_operands[1], AbsoluteOfInput());
}

std::vector<std::string>
HloHistogramInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {"absolute_of_input=" + AbsoluteOfInput()};
}

namespace {

static HloPoplarInstructionFactory histogram_factory(
    PoplarOp::Histogram,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloHistogramInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

      TF_ASSIGN_OR_RETURN(
          bool absolute_of_input,
          attribute_map.GetAttributeAsBool("absolute_of_input"));

      return absl::make_unique<HloHistogramInstruction>(
          call->shape(), call->mutable_operand(0), call->mutable_operand(1),
          absolute_of_input);
    });

}  // namespace

/*
 * Histogram update.
 */
HloHistogramUpdateInstruction::HloHistogramUpdateInstruction(
    const Shape& shape, HloInstruction* const hist, HloInstruction* const input,
    HloInstruction* const levels, bool absolute_of_input)
    : HloPoplarInstruction(shape, {hist, input, levels},
                           PoplarOp::HistogramUpdate, absolute_of_input),
      absolute_of_input_(absolute_of_input) {}

absl::flat_hash_set<int64_t> HloHistogramUpdateInstruction::AllocatingIndices()
    const {
  return {};
}

absl::flat_hash_map<int64_t, int64_t>
HloHistogramUpdateInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloHistogramUpdateInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsSimpleNoTuple0thOperandAliasing(this);
}

HloPoplarBufferDescriptions
HloHistogramUpdateInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsNoAllocations();
}

bool HloHistogramUpdateInstruction::AllocatingOutput() const { return false; }

const FindConsumersExtensionResults
HloHistogramUpdateInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloHistogramUpdateInstruction::AllowNonInplaceLowering() const {
  return false;
}

bool HloHistogramUpdateInstruction::IsPopOpsElementwise() const {
  return false;
}

bool HloHistogramUpdateInstruction::AbsoluteOfInput() const {
  return absolute_of_input_;
}

std::unique_ptr<HloInstruction>
HloHistogramUpdateInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloHistogramUpdateInstruction>(
      shape, new_operands[0], new_operands[1], new_operands[2],
      AbsoluteOfInput());
}

std::vector<std::string>
HloHistogramUpdateInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {"absolute_of_input=" + AbsoluteOfInput()};
}

namespace {

static HloPoplarInstructionFactory histogram_update_factory(
    PoplarOp::HistogramUpdate,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloHistogramUpdateInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

      TF_ASSIGN_OR_RETURN(
          bool absolute_of_input,
          attribute_map.GetAttributeAsBool("absolute_of_input"));

      return absl::make_unique<HloHistogramUpdateInstruction>(
          call->shape(), call->mutable_operand(0), call->mutable_operand(1),
          call->mutable_operand(2), absolute_of_input);
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
