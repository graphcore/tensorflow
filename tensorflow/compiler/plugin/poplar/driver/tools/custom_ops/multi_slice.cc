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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

// MultiSlice
HloMultiSliceInstruction::HloMultiSliceInstruction(
    const Shape& shape, HloInstruction* const input,
    HloInstruction* const indices, bool indices_are_sorted)
    : HloPoplarInstruction(shape, {input, indices}, PoplarOp::MultiSlice,
                           indices_are_sorted),
      indices_are_sorted_(indices_are_sorted) {}

absl::flat_hash_set<int64> HloMultiSliceInstruction::AllocatingIndices() const {
  return {0, 1};
}

bool HloMultiSliceInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64> HloMultiSliceInstruction::LayoutDependencies()
    const {
  return {};
}

HloPoplarUseDescriptions HloMultiSliceInstruction::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloMultiSliceInstruction::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloMultiSliceInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloMultiSliceInstruction::AllowNonInplaceLowering() const { return false; }

bool HloMultiSliceInstruction::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction>
HloMultiSliceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateMultiSlice(shape, new_operands[0], new_operands[1],
                          indices_are_sorted_);
}

std::vector<std::string>
HloMultiSliceInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateMultiSlice(const Shape& shape,
                                                 HloInstruction* const input,
                                                 HloInstruction* const indices,
                                                 bool indices_are_sorted) {
  return absl::make_unique<HloMultiSliceInstruction>(shape, input, indices,
                                                     indices_are_sorted);
}

// StaticMultiSlice
HloStaticMultiSliceInstruction::HloStaticMultiSliceInstruction(
    const Shape& shape, HloInstruction* const input,
    absl::Span<const int64> indices)
    : HloPoplarInstruction(shape, {input}, PoplarOp::StaticMultiSlice, indices),
      indices_(indices.begin(), indices.end()) {}

absl::flat_hash_set<int64> HloStaticMultiSliceInstruction::AllocatingIndices()
    const {
  return {};
}

bool HloStaticMultiSliceInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64>
HloStaticMultiSliceInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloStaticMultiSliceInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloStaticMultiSliceInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults
HloStaticMultiSliceInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return {true, this, params.index, params.permutation};
}

bool HloStaticMultiSliceInstruction::AllowNonInplaceLowering() const {
  return false;
}

bool HloStaticMultiSliceInstruction::IsPopOpsElementwise() const {
  return false;
}

std::vector<std::string>
HloStaticMultiSliceInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {absl::StrCat("indices={", absl::StrJoin(GetIndices(), ","), "}")};
}

std::unique_ptr<HloInstruction>
HloStaticMultiSliceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateStaticMultiSlice(shape, new_operands[0], indices_);
}

std::unique_ptr<HloInstruction> CreateStaticMultiSlice(
    const Shape& shape, HloInstruction* const input,
    absl::Span<const int64> indices) {
  return absl::make_unique<HloStaticMultiSliceInstruction>(shape, input,
                                                           indices);
}

// MultiUpdate
HloMultiUpdateInstruction::HloMultiUpdateInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_update, bool indices_are_sorted)
    : HloPoplarInstruction(
          shape, operands,
          is_update ? PoplarOp::MultiUpdateAdd : PoplarOp::MultiUpdate,
          indices_are_sorted),

      indices_are_sorted_(indices_are_sorted) {
  CHECK_EQ(shape.rank(), 2);
  CHECK_EQ(operands[0]->shape().rank(), 2);
  CHECK_EQ(operands[1]->shape().rank(), 2);
  CHECK_EQ(operands[1]->shape().dimensions(1), 1);
  CHECK_EQ(operands[2]->shape().rank(), 2);
}

absl::flat_hash_set<int64> HloMultiUpdateInstruction::AllocatingIndices()
    const {
  return {0, 1, 2};
}

bool HloMultiUpdateInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64>
HloMultiUpdateInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloMultiUpdateInstruction::GetUseDescriptions() const {
  return UseDescriptionsSimpleNoTuple0thOperandAliasing(this);
}

HloPoplarBufferDescriptions HloMultiUpdateInstruction::GetBufferDescriptions()
    const {
  return BufferDescriptionsNoAllocations();
}

const FindConsumersExtensionResults HloMultiUpdateInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  auto allocating_indexes = AllocatingIndices();
  auto op_index = params.op_index;

  if (allocating_indexes.count(op_index)) {
    if (op_index == 0) {
      // In order to use slice plan for MultiSlice and MultiUpdate
      // instructions, we have to ensure that tensor allocated for
      // specific instruction has been allocated with equivalent slice
      // plan. MultiUpdate/MultiUpdateAdd operand 0 is in-place, so
      // it's possible to look through its consumers and find all
      // instruction which may use this tensor later.
      FindConsumersExtensionResults result{true, this, params.index,
                                           params.permutation};
      return result;
    }
  }
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloMultiUpdateInstruction::AllowNonInplaceLowering() const {
  return false;
}

bool HloMultiUpdateInstruction::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction>
HloMultiUpdateInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateMultiUpdate(shape, new_operands, indices_are_sorted_);
}

std::vector<std::string>
HloMultiUpdateInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("indices_are_sorted=" +
                       std::string(indices_are_sorted_ ? "true" : "false"));
  return attributes;
}

std::unique_ptr<HloInstruction> CreateMultiUpdate(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool indices_are_sorted) {
  return absl::make_unique<HloMultiUpdateInstruction>(shape, operands,
                                                      indices_are_sorted);
}

// MultiUpdateAdd
HloMultiUpdateAddInstruction::HloMultiUpdateAddInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool indices_are_sorted)
    : HloMultiUpdateInstruction(shape, operands, true, indices_are_sorted) {}

std::unique_ptr<HloInstruction>
HloMultiUpdateAddInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateMultiUpdateAdd(shape, new_operands, indices_are_sorted_);
}

std::unique_ptr<HloInstruction> CreateMultiUpdateAdd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool indices_are_sorted) {
  return absl::make_unique<HloMultiUpdateAddInstruction>(shape, operands,
                                                         indices_are_sorted);
}

// StaticMultiUpdateAdd
HloStaticMultiUpdateAddInstruction::HloStaticMultiUpdateAddInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::Span<const int64> indices)
    : HloPoplarInstruction(shape, operands, PoplarOp::StaticMultiUpdateAdd,
                           indices),
      indices_(indices.begin(), indices.end()) {
  CHECK_EQ(operands[0]->shape().rank(), 2);
  CHECK_EQ(operands[1]->shape().rank(), 2);
  CHECK_EQ(operands[2]->shape().rank(), 0);
  CHECK_EQ(operands[0]->shape().dimensions()[1],
           operands[1]->shape().dimensions()[1]);
  CHECK_EQ(operands[1]->shape().dimensions()[0], indices_.size());
  CHECK_EQ(shape, operands[0]->shape());
  for (const auto& index : indices_) {
    CHECK_LT(index, operands[0]->shape().dimensions()[0]);
  }
}

absl::flat_hash_set<int64>
HloStaticMultiUpdateAddInstruction::AllocatingIndices() const {
  return {};
}

bool HloStaticMultiUpdateAddInstruction::AllocatingOutput() const {
  return false;
}

absl::flat_hash_map<int64, int64>
HloStaticMultiUpdateAddInstruction::LayoutDependencies() const {
  return {{0, 1}, {1, 0}};
}

HloPoplarUseDescriptions
HloStaticMultiUpdateAddInstruction::GetUseDescriptions() const {
  return UseDescriptionsSimpleNoTuple0thOperandAliasing(this);
}

HloPoplarBufferDescriptions
HloStaticMultiUpdateAddInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsNoAllocations();
}

const FindConsumersExtensionResults
HloStaticMultiUpdateAddInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  if (params.op_index == 0) {
    return {true, this, params.index, params.permutation};
  }
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloStaticMultiUpdateAddInstruction::AllowNonInplaceLowering() const {
  return false;
}

bool HloStaticMultiUpdateAddInstruction::IsPopOpsElementwise() const {
  return false;
}

std::vector<std::string>
HloStaticMultiUpdateAddInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {absl::StrCat("indices={", absl::StrJoin(GetIndices(), ","), "}")};
}

std::unique_ptr<HloInstruction>
HloStaticMultiUpdateAddInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateStaticMultiUpdateAdd(shape, new_operands, indices_);
}

std::unique_ptr<HloInstruction> CreateStaticMultiUpdateAdd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::Span<const int64> indices) {
  return absl::make_unique<HloStaticMultiUpdateAddInstruction>(shape, operands,
                                                               indices);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloMultiSliceInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  TF_ASSIGN_OR_RETURN(bool indices_are_sorted,
                      attribute_map.GetAttributeAsBool("indices_are_sorted"));
  return CreateMultiSlice(call->shape(), call->mutable_operand(0),
                          call->mutable_operand(1), indices_are_sorted);
}

StatusOr<std::unique_ptr<HloInstruction>>
HloStaticMultiSliceInstructionFactoryFunc(HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  TF_ASSIGN_OR_RETURN(std::vector<int64> indices,
                      attribute_map.GetAttributeInt64Vector("indices"));
  return CreateStaticMultiSlice(call->shape(), call->mutable_operand(0),
                                indices);
}

StatusOr<std::unique_ptr<HloInstruction>> HloMultiUpdateInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  TF_ASSIGN_OR_RETURN(bool indices_are_sorted,
                      attribute_map.GetAttributeAsBool("indices_are_sorted"));
  return CreateMultiUpdate(call->shape(), call->operands(), indices_are_sorted);
}

StatusOr<std::unique_ptr<HloInstruction>>
HloMultiUpdateAddInstructionFactoryFunc(HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  TF_ASSIGN_OR_RETURN(bool indices_are_sorted,
                      attribute_map.GetAttributeAsBool("indices_are_sorted"));
  return CreateMultiUpdateAdd(call->shape(), call->operands(),
                              indices_are_sorted);
}

StatusOr<std::unique_ptr<HloInstruction>>
HloStaticMultiUpdateAddInstructionFactoryFunc(HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  TF_ASSIGN_OR_RETURN(std::vector<int64> indices,
                      attribute_map.GetAttributeInt64Vector("indices"));
  return CreateStaticMultiUpdateAdd(call->shape(), call->operands(), indices);
}

static HloPoplarInstructionFactory multi_slice_factory(
    PoplarOp::MultiSlice, HloMultiSliceInstructionFactoryFunc);

static HloPoplarInstructionFactory static_multi_slice_factory(
    PoplarOp::StaticMultiSlice, HloStaticMultiSliceInstructionFactoryFunc);

static HloPoplarInstructionFactory multi_update_factory(
    PoplarOp::MultiUpdate, HloMultiUpdateInstructionFactoryFunc);

static HloPoplarInstructionFactory multi_update_add_factory(
    PoplarOp::MultiUpdateAdd, HloMultiUpdateAddInstructionFactoryFunc);

static HloPoplarInstructionFactory static_multi_update_add_factory(
    PoplarOp::StaticMultiUpdateAdd,
    HloStaticMultiUpdateAddInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
