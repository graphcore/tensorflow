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
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloHostEmbeddingLookupInstruction::HloHostEmbeddingLookupInstruction(
    HloInstruction* indices, const std::string& embedding_id,
    const xla::Shape& embedding_shape,
    HostEmbeddingSplittingStrategy splitting_strategy, const Shape shape)
    : HloPoplarInstruction(shape, {indices}, PoplarOp::HostEmbeddingLookup,
                           embedding_id, embedding_shape.ToString(),
                           static_cast<int>(splitting_strategy)),
      embedding_id_(embedding_id),
      embedding_shape_(embedding_shape),
      splitting_strategy_(splitting_strategy) {
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64>
HloHostEmbeddingLookupInstruction::AllocatingIndices() const {
  return {};
}

bool HloHostEmbeddingLookupInstruction::AllocatingOutput() const {
  return true;
}

absl::flat_hash_map<int64, int64>
HloHostEmbeddingLookupInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloHostEmbeddingLookupInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloHostEmbeddingLookupInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults
HloHostEmbeddingLookupInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloHostEmbeddingLookupInstruction::AllowNonInplaceLowering() const {
  return false;
}

bool HloHostEmbeddingLookupInstruction::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction> CreateHostEmbeddingLookup(
    HloInstruction* indices, const std::string& embedding_id,
    const xla::Shape& embedding_shape,
    HostEmbeddingSplittingStrategy splitting_strategy, const Shape shape) {
  return absl::make_unique<HloHostEmbeddingLookupInstruction>(
      indices, embedding_id, embedding_shape, splitting_strategy, shape);
}

std::unique_ptr<HloInstruction>
HloHostEmbeddingLookupInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateHostEmbeddingLookup(operands[0], embedding_id_, embedding_shape_,
                                   splitting_strategy_, shape);
}

std::vector<std::string>
HloHostEmbeddingLookupInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;

  attributes.push_back(absl::StrCat("embedding_id=", embedding_id_));
  attributes.push_back(
      absl::StrCat("embedding_shape=", embedding_shape_.ToString()));

  return attributes;
}

HloHostEmbeddingUpdateInstruction::HloHostEmbeddingUpdateInstruction(
    HloInstruction* in, HloInstruction* grads, HloInstruction* indices,
    const std::string& embedding_id, const xla::Shape& embedding_shape,
    HostEmbeddingSplittingStrategy splitting_strategy, const Shape shape)
    : HloPoplarInstruction(shape, {in, grads, indices},
                           PoplarOp::HostEmbeddingUpdate, embedding_id,
                           embedding_shape.ToString(),
                           static_cast<int>(splitting_strategy)),
      embedding_id_(embedding_id),
      embedding_shape_(embedding_shape),
      splitting_strategy_(splitting_strategy) {
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64>
HloHostEmbeddingUpdateInstruction::AllocatingIndices() const {
  return {};
}

bool HloHostEmbeddingUpdateInstruction::AllocatingOutput() const {
  return false;
}

absl::flat_hash_map<int64, int64>
HloHostEmbeddingUpdateInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloHostEmbeddingUpdateInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloHostEmbeddingUpdateInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults
HloHostEmbeddingUpdateInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloHostEmbeddingUpdateInstruction::AllowNonInplaceLowering() const {
  return false;
}

bool HloHostEmbeddingUpdateInstruction::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction> CreateHloHostEmbeddingUpdate(
    HloInstruction* in, HloInstruction* grads, HloInstruction* indices,
    const std::string& embedding_id, const xla::Shape& embedding_shape,
    HostEmbeddingSplittingStrategy splitting_strategy, const Shape shape) {
  return absl::make_unique<HloHostEmbeddingUpdateInstruction>(
      in, grads, indices, embedding_id, embedding_shape, splitting_strategy,
      shape);
}

std::unique_ptr<HloInstruction>
HloHostEmbeddingUpdateInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateHloHostEmbeddingUpdate(operands[0], operands[1], operands[2],
                                      embedding_id_, embedding_shape_,
                                      splitting_strategy_, shape);
}

std::vector<std::string>
HloHostEmbeddingUpdateInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;

  attributes.push_back(absl::StrCat("embedding_id=", embedding_id_));
  attributes.push_back(
      absl::StrCat("embedding_shape=", embedding_shape_.ToString()));

  return attributes;
}

HloHostEmbeddingNotifyInstruction::HloHostEmbeddingNotifyInstruction(

    const std::string& embedding_id)
    : HloPoplarInstruction(ShapeUtil::MakeTokenShape(), {},
                           PoplarOp::HostEmbeddingNotify, embedding_id),
      embedding_id_(embedding_id) {
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64>
HloHostEmbeddingNotifyInstruction::AllocatingIndices() const {
  return {};
}

bool HloHostEmbeddingNotifyInstruction::AllocatingOutput() const {
  return false;
}

absl::flat_hash_map<int64, int64>
HloHostEmbeddingNotifyInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloHostEmbeddingNotifyInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloHostEmbeddingNotifyInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults
HloHostEmbeddingNotifyInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloHostEmbeddingNotifyInstruction::AllowNonInplaceLowering() const {
  return false;
}

bool HloHostEmbeddingNotifyInstruction::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction> CreateHloHostEmbeddingNotify(
    const std::string& embedding_id) {
  return absl::make_unique<HloHostEmbeddingNotifyInstruction>(embedding_id);
}

std::unique_ptr<HloInstruction>
HloHostEmbeddingNotifyInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateHloHostEmbeddingNotify(embedding_id_);
}

std::vector<std::string>
HloHostEmbeddingNotifyInstruction::ExtraPoplarAttributesToStringImpl(
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
      TF_ASSIGN_OR_RETURN(xla::Shape embedding_shape,
                          attribute_map.GetAttributeAsShape("embedding_shape"));
      TF_ASSIGN_OR_RETURN(
          std::string partition_strategy,
          attribute_map.GetAttributeAsString("partition_strategy"));

      HostEmbeddingSplittingStrategy splitting_strategy;
      if (partition_strategy == "ENCODING") {
        splitting_strategy = HostEmbeddingSplittingStrategy::Encoding;
      } else if (partition_strategy == "TOKEN") {
        splitting_strategy = HostEmbeddingSplittingStrategy::Token;
      } else {
        return FailedPrecondition("Unknown embedding splitting strategy {}",
                                  partition_strategy);
      }

      return CreateHostEmbeddingLookup(call->mutable_operand(0), embedding_id,
                                       embedding_shape, splitting_strategy,
                                       call->shape());
    });

static HloPoplarInstructionFactory host_embedding_update_factory(
    PoplarOp::HostEmbeddingUpdate,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

      TF_ASSIGN_OR_RETURN(std::string embedding_id,
                          attribute_map.GetAttributeAsString("embedding_id"));
      TF_ASSIGN_OR_RETURN(xla::Shape embedding_shape,
                          attribute_map.GetAttributeAsShape("embedding_shape"));
      TF_ASSIGN_OR_RETURN(
          std::string partition_strategy,
          attribute_map.GetAttributeAsString("partition_strategy"));

      HostEmbeddingSplittingStrategy splitting_strategy;
      if (partition_strategy == "ENCODING") {
        splitting_strategy = HostEmbeddingSplittingStrategy::Encoding;
      } else if (partition_strategy == "TOKEN") {
        splitting_strategy = HostEmbeddingSplittingStrategy::Token;
      } else {
        return FailedPrecondition("Unknown embedding splitting strategy {}",
                                  partition_strategy);
      }

      return CreateHloHostEmbeddingUpdate(
          call->mutable_operand(0), call->mutable_operand(1),
          call->mutable_operand(2), embedding_id, embedding_shape,
          splitting_strategy, call->shape());
    });

static HloPoplarInstructionFactory host_embedding_notify_factory(
    PoplarOp::HostEmbeddingNotify,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

      TF_ASSIGN_OR_RETURN(std::string embedding_id,
                          attribute_map.GetAttributeAsString("embedding_id"));

      return CreateHloHostEmbeddingNotify(embedding_id);
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
