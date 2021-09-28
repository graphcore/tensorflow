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
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/sequence_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

/*
 * HloSequenceSliceInstruction
 */
HloSequenceSliceInstruction::HloSequenceSliceInstruction(
    const Shape& shape, HloInstruction* const dst, HloInstruction* const src,
    HloInstruction* const num_elems, HloInstruction* const src_offsets,
    HloInstruction* const dst_offsets, bool zero_unused)
    : HloPoplarInstruction(shape,
                           {dst, src, num_elems, src_offsets, dst_offsets},
                           PoplarOp::SequenceSlice),
      zero_unused(zero_unused) {}

HloSequenceSliceInstruction::HloSequenceSliceInstruction(
    const Shape& shape, HloInstruction* const src,
    HloInstruction* const num_elems, HloInstruction* const src_offsets,
    HloInstruction* const dst_offsets, bool zero_unused, PoplarOp op)
    : HloPoplarInstruction(shape, {src, num_elems, src_offsets, dst_offsets},
                           op),
      zero_unused(zero_unused) {}

absl::flat_hash_set<int64> HloSequenceSliceInstruction::AllocatingIndices()
    const {
  return {};
}

absl::flat_hash_map<int64, int64>
HloSequenceSliceInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloSequenceSliceInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsSimpleNoTuple0thOperandAliasing(this);
}

HloPoplarBufferDescriptions HloSequenceSliceInstruction::GetBufferDescriptions()
    const {
  return BufferDescriptionsNoAllocations();
}

bool HloSequenceSliceInstruction::AllocatingOutput() const { return false; }

const FindConsumersExtensionResults HloSequenceSliceInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  auto op_index = params.op_index;

  if ((op_index == 0) || (IsAnyScaledInplace(this) && op_index < 2)) {
    FindConsumersExtensionResults result{true, this, params.index,
                                         params.permutation};
    return result;
  }
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloSequenceSliceInstruction::AllowNonInplaceLowering() const {
  return false;
}

bool HloSequenceSliceInstruction::IsPopOpsElementwise() const { return false; }

bool HloSequenceSliceInstruction::ZeroUnused() const { return zero_unused; }

std::unique_ptr<HloInstruction>
HloSequenceSliceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloSequenceSliceInstruction>(
      shape, new_operands[0], new_operands[1], new_operands[2], new_operands[3],
      new_operands[4], ZeroUnused());
}

std::vector<std::string>
HloSequenceSliceInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {"zero_unused=" + ZeroUnused()};
}

namespace {

StatusOr<std::unique_ptr<HloSequenceSliceInstruction>>
MakeSequenceSliceInstruction(HloCustomCallInstruction* call, bool zero_unused) {
  return absl::make_unique<HloSequenceSliceInstruction>(
      call->shape(), call->mutable_operand(0), call->mutable_operand(1),
      call->mutable_operand(2), call->mutable_operand(3),
      call->mutable_operand(4), zero_unused);
}

StatusOr<std::unique_ptr<HloSequenceSliceInstruction>>
HloSequenceSliceInstructionFactoryFunc(HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

  TF_ASSIGN_OR_RETURN(bool zero_unused,
                      attribute_map.GetAttributeAsBool("zero_unused"));

  return MakeSequenceSliceInstruction(call, zero_unused);
}

static HloPoplarInstructionFactory sequence_slice_factory(
    PoplarOp::SequenceSlice, HloSequenceSliceInstructionFactoryFunc);

}  // namespace

/*
 * HloSequenceSliceUnpackInstruction
 */
HloSequenceSliceUnpackInstruction::HloSequenceSliceUnpackInstruction(
    const Shape& shape, HloInstruction* const src,
    HloInstruction* const num_elems, HloInstruction* const src_offsets,
    HloInstruction* const dst_offsets, bool zero_unused, int64 total_elements)
    : HloSequenceSliceInstruction(shape, src, num_elems, src_offsets,
                                  dst_offsets, zero_unused,
                                  PoplarOp::SequenceSliceUnpack),
      total_elements(total_elements) {}

HloPoplarUseDescriptions HloSequenceSliceUnpackInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloSequenceSliceUnpackInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

bool HloSequenceSliceUnpackInstruction::AllocatingOutput() const {
  return true;
}

int64 HloSequenceSliceUnpackInstruction::TotalElements() const {
  return total_elements;
}

std::unique_ptr<HloInstruction>
HloSequenceSliceUnpackInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloSequenceSliceUnpackInstruction>(
      shape, new_operands[0], new_operands[1], new_operands[2], new_operands[3],
      ZeroUnused(), TotalElements());
}

namespace {
StatusOr<std::unique_ptr<HloSequenceSliceUnpackInstruction>>
MakeSequenceSliceUnpackInstruction(HloCustomCallInstruction* call,
                                   bool zero_unused, int64 total_elements) {
  return absl::make_unique<HloSequenceSliceUnpackInstruction>(
      call->shape(), call->mutable_operand(0), call->mutable_operand(1),
      call->mutable_operand(2), call->mutable_operand(3), zero_unused,
      total_elements);
}

StatusOr<std::unique_ptr<HloSequenceSliceUnpackInstruction>>
HloSequenceSliceUnpackInstructionFactoryFunc(HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

  TF_ASSIGN_OR_RETURN(bool zero_unused,
                      attribute_map.GetAttributeAsBool("zero_unused"));

  TF_ASSIGN_OR_RETURN(int64 total_elements,
                      attribute_map.GetAttributeAsBool("total_elements"));

  return MakeSequenceSliceUnpackInstruction(call, zero_unused, total_elements);
}

static HloPoplarInstructionFactory sequence_slice_unpack_factory(
    PoplarOp::SequenceSliceUnpack,
    HloSequenceSliceUnpackInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
