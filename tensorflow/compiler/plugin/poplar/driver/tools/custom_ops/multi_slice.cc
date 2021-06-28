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
    HloInstruction* const indices)
    : HloPoplarInstruction(shape, {input, indices}, PoplarOp::MultiSlice) {}

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

bool HloMultiSliceInstruction::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction>
HloMultiSliceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateMultiSlice(shape, new_operands[0], new_operands[1]);
}

std::vector<std::string>
HloMultiSliceInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateMultiSlice(
    const Shape& shape, HloInstruction* const input,
    HloInstruction* const indices) {
  return absl::make_unique<HloMultiSliceInstruction>(shape, input, indices);
}

// MultiUpdate
HloMultiUpdateInstruction::HloMultiUpdateInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    std::size_t index_vector_dim, std::size_t update_dim,
    uint32 serialization_factor, bool is_update)
    : HloPoplarInstruction(
          shape, operands,
          is_update ? PoplarOp::MultiUpdateAdd : PoplarOp::MultiUpdate,
          index_vector_dim, update_dim, serialization_factor),
      index_vector_dim_(index_vector_dim),
      update_dim_(update_dim),
      serialization_factor_(serialization_factor) {}

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

bool HloMultiUpdateInstruction::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction>
HloMultiUpdateInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateMultiUpdate(shape, new_operands, index_vector_dim_, update_dim_,
                           serialization_factor_);
}

std::vector<std::string>
HloMultiUpdateInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("index_vector_dim=" + std::to_string(index_vector_dim_));
  attributes.push_back("update_dim=" + std::to_string(update_dim_));
  attributes.push_back("serialization_factor=" +
                       std::to_string(serialization_factor_));
  return attributes;
}

std::unique_ptr<HloInstruction> CreateMultiUpdate(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    std::size_t index_vector_dim, std::size_t update_dim,
    uint32 serialization_factor) {
  return absl::make_unique<HloMultiUpdateInstruction>(
      shape, operands, index_vector_dim, update_dim, serialization_factor);
}

// MultiUpdateAdd
HloMultiUpdateAddInstruction::HloMultiUpdateAddInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    std::size_t index_vector_dim, std::size_t update_dim,
    uint32 serialization_factor)
    : HloMultiUpdateInstruction(shape, operands, index_vector_dim, update_dim,
                                serialization_factor, true) {}

std::unique_ptr<HloInstruction>
HloMultiUpdateAddInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateMultiUpdateAdd(shape, new_operands, index_vector_dim_,
                              update_dim_, serialization_factor_);
}

std::unique_ptr<HloInstruction> CreateMultiUpdateAdd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    std::size_t index_vector_dim, std::size_t update_dim,
    uint32 serialization_factor) {
  return absl::make_unique<HloMultiUpdateAddInstruction>(
      shape, operands, index_vector_dim, update_dim, serialization_factor);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloMultiSliceInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateMultiSlice(call->shape(), call->mutable_operand(0),
                          call->mutable_operand(1));
}

StatusOr<std::unique_ptr<HloInstruction>> HloMultiUpdateInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  TF_ASSIGN_OR_RETURN(uint64 index_vector_dim,
                      attribute_map.GetAttributeAsUInt64("index_vector_dim"));
  TF_ASSIGN_OR_RETURN(uint64 update_dim,
                      attribute_map.GetAttributeAsUInt64("update_dim"));
  return CreateMultiUpdate(call->shape(), call->operands(), index_vector_dim,
                           update_dim, 1);
}

StatusOr<std::unique_ptr<HloInstruction>>
HloMultiUpdateAddInstructionFactoryFunc(HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  TF_ASSIGN_OR_RETURN(uint64 index_vector_dim,
                      attribute_map.GetAttributeAsUInt64("index_vector_dim"));
  TF_ASSIGN_OR_RETURN(uint64 update_dim,
                      attribute_map.GetAttributeAsUInt64("update_dim"));
  return CreateMultiUpdateAdd(call->shape(), call->operands(), index_vector_dim,
                              update_dim, 1);
}

static HloPoplarInstructionFactory multi_slice_factory(
    PoplarOp::MultiSlice, HloMultiSliceInstructionFactoryFunc);

static HloPoplarInstructionFactory multi_update_factory(
    PoplarOp::MultiUpdate, HloMultiUpdateInstructionFactoryFunc);

static HloPoplarInstructionFactory multi_update_add_factory(
    PoplarOp::MultiUpdateAdd, HloMultiUpdateAddInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
