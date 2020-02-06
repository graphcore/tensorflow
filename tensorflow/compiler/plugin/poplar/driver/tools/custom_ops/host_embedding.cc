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

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/host_embedding.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloHostEmbeddingLookupInstruction::HloHostEmbeddingLookupInstruction(
    HloInstruction* indices, const std::string& embedding_id, const Shape shape)
    : HloPoplarInstruction(shape, {indices}, PoplarOp::HostEmbeddingLookup,
                           embedding_id),
      embedding_id_(embedding_id) {
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64>
HloHostEmbeddingLookupInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64>
HloHostEmbeddingLookupInstruction::LayoutDependencies() const {
  return {};
}

uint64 HloHostEmbeddingLookupInstruction::NumberOfInplaceOperands() const {
  return 0;
}

bool HloHostEmbeddingLookupInstruction::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction> CreateHostEmbeddingLookup(
    HloInstruction* indices, const std::string& embedding_id,
    const Shape shape) {
  return absl::make_unique<HloHostEmbeddingLookupInstruction>(
      indices, embedding_id, shape);
}

std::unique_ptr<HloInstruction>
HloHostEmbeddingLookupInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateHostEmbeddingLookup(operands[0], embedding_id_, shape);
}

std::vector<std::string>
HloHostEmbeddingLookupInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;

  attributes.push_back(absl::StrCat("embedding_id=", embedding_id_));

  return attributes;
}

HloHostEmbeddingUpdateInstruction::HloHostEmbeddingUpdateInstruction(
    HloInstruction* grads, HloInstruction* indices,
    const std::string& embedding_id, const Shape shape)
    : HloPoplarInstruction(shape, {grads, indices},
                           PoplarOp::HostEmbeddingUpdate, embedding_id),
      embedding_id_(embedding_id) {
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64>
HloHostEmbeddingUpdateInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64>
HloHostEmbeddingUpdateInstruction::LayoutDependencies() const {
  return {};
}

uint64 HloHostEmbeddingUpdateInstruction::NumberOfInplaceOperands() const {
  return 0;
}

bool HloHostEmbeddingUpdateInstruction::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction> CreateHloHostEmbeddingUpdate(
    HloInstruction* grads, HloInstruction* indices,
    const std::string& embedding_id, const Shape shape) {
  return absl::make_unique<HloHostEmbeddingUpdateInstruction>(
      grads, indices, embedding_id, shape);
}

std::unique_ptr<HloInstruction>
HloHostEmbeddingUpdateInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateHloHostEmbeddingUpdate(operands[0], operands[1], embedding_id_,
                                      shape);
}

std::vector<std::string>
HloHostEmbeddingUpdateInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;

  attributes.push_back(absl::StrCat("embedding_id=", embedding_id_));

  return attributes;
}

namespace {

static HloPoplarInstructionFactory host_embedding_lookup_factory(
    PoplarOp::HostEmbeddingLookup,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

      TF_ASSIGN_OR_RETURN(std::string embedding_id,
                          attribute_map.GetAttributeAsString("embedding_id"));

      return CreateHostEmbeddingLookup(call->mutable_operand(0), embedding_id,
                                       call->shape());
    });

static HloPoplarInstructionFactory host_embedding_update_factory(
    PoplarOp::HostEmbeddingUpdate,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

      TF_ASSIGN_OR_RETURN(std::string embedding_id,
                          attribute_map.GetAttributeAsString("embedding_id"));

      return CreateHloHostEmbeddingUpdate(call->mutable_operand(0),
                                          call->mutable_operand(1),
                                          embedding_id, call->shape());
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
