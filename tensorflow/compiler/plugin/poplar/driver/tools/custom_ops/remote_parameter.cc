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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"

#include <algorithm>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

namespace {
Shape ComputePerReplicaLoadShape(Shape remote_buffer_shape,
                                 uint64 replication_factor) {
  if (replication_factor < 2) {
    return remote_buffer_shape;
  }

  const int64 element_count = PartitionedElementCountPerReplica(
      ShapeUtil::ElementsIn(remote_buffer_shape), replication_factor);

  return ShapeUtil::MakeShape(remote_buffer_shape.element_type(),
                              {element_count});
}

Shape ComputePerReplicaLoadShape(
    absl::Span<HloInstruction* const> rbuffers,
    const std::vector<uint64>& replication_factors) {
  CHECK_EQ(rbuffers.size(), replication_factors.size());
  std::vector<Shape> result_shape(rbuffers.size());
  for (int64 i = 0; i != rbuffers.size(); ++i) {
    result_shape[i] = ComputePerReplicaLoadShape(rbuffers[i]->shape(),
                                                 replication_factors[i]);
  }

  return rbuffers.size() == 1 ? result_shape[0]
                              : ShapeUtil::MakeTupleShape(result_shape);
}

Shape ComputePerReplicaStoreShape(
    absl::Span<HloInstruction* const> rbuffers_and_values,
    const std::vector<uint64>& replication_factors) {
  auto rbuffers =
      rbuffers_and_values.subspan(0, rbuffers_and_values.size() / 2);

  std::vector<Shape> result_shape(rbuffers.size());
  absl::c_transform(
      rbuffers, result_shape.begin(),
      [](HloInstruction* const rbuffer) { return rbuffer->shape(); });

  return rbuffers.size() == 1 ? result_shape[0]
                              : ShapeUtil::MakeTupleShape(result_shape);
}
}  // namespace

HloAbstractRemoteLoadStore::HloAbstractRemoteLoadStore(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const std::vector<uint64>& replication_factors, PoplarOp op)
    : HloPoplarInstruction(shape, operands, op,
                           absl::StrJoin(replication_factors, ".")),
      replication_factors_(replication_factors) {}

uint64 HloAbstractRemoteLoadStore::GetReplicationFactor(int64 index) const {
  CHECK_GE(index, 0);
  CHECK_LT(static_cast<std::size_t>(index), replication_factors_.size());
  return replication_factors_[index];
}

std::size_t HloAbstractRemoteLoadStore::GetReplicationFactorCount() const {
  return replication_factors_.size();
}

std::vector<std::string>
HloAbstractRemoteLoadStore::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions&) const {
  return {"replication_factors=" + absl::StrJoin(replication_factors_, ", ")};
}

HloRemoteParameterLoad::HloRemoteParameterLoad(
    absl::Span<HloInstruction* const> rbuffers,
    const std::vector<uint64>& replication_factors)
    : HloAbstractRemoteLoadStore(
          ComputePerReplicaLoadShape(rbuffers, replication_factors), rbuffers,
          replication_factors, PoplarOp::RemoteParameterLoad) {
  CHECK_EQ(rbuffers.size(), replication_factors.size());
}

absl::Span<HloInstruction* const> HloRemoteParameterLoad::RemoteBuffers()
    const {
  return operands();
}

absl::flat_hash_set<int64> HloRemoteParameterLoad::AllocatingIndices() const {
  return {};
}

bool HloRemoteParameterLoad::AllocatingOutput() const { return true; }

absl::flat_hash_map<int64, int64> HloRemoteParameterLoad::LayoutDependencies()
    const {
  return {};
}

HloPoplarUseDescriptions HloRemoteParameterLoad::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloRemoteParameterLoad::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloRemoteParameterLoad::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloRemoteParameterLoad::AllowNonInplaceLowering() const { return false; }

bool HloRemoteParameterLoad::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction>
HloRemoteParameterLoad::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateHloRemoteParameterLoad(operands, replication_factors_);
}

std::unique_ptr<HloInstruction> CreateHloRemoteParameterLoad(
    absl::Span<HloInstruction* const> rbuffers,
    const std::vector<uint64>& replication_factors) {
  return absl::make_unique<HloRemoteParameterLoad>(rbuffers,
                                                   replication_factors);
}

HloRemoteParameterStore::HloRemoteParameterStore(
    absl::Span<HloInstruction* const> rbuffers_and_values,
    const std::vector<uint64>& replication_factors)
    : HloAbstractRemoteLoadStore(
          ComputePerReplicaStoreShape(rbuffers_and_values, replication_factors),
          rbuffers_and_values, replication_factors,
          PoplarOp::RemoteParameterStore) {
  // The first half of the operands are the remote buffers, the second half
  // are the corresponding values to store in the buffers.
  CHECK_GE(rbuffers_and_values.size(), 2);
  CHECK_EQ(rbuffers_and_values.size() % 2, 0);
  CHECK_EQ(rbuffers_and_values.size() / 2, replication_factors.size());
  const int64 half_size = rbuffers_and_values.size() / 2;
  for (int64 i = 0; i != half_size; ++i) {
    CHECK_EQ(rbuffers_and_values[i]->shape().element_type(),
             rbuffers_and_values[i + half_size]->shape().element_type());
  }
}

absl::flat_hash_set<int64> HloRemoteParameterStore::AllocatingIndices() const {
  return {};
}

bool HloRemoteParameterStore::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64> HloRemoteParameterStore::LayoutDependencies()
    const {
  return {};
}

HloPoplarUseDescriptions HloRemoteParameterStore::GetUseDescriptions() const {
  // The remote buffers are in-place, but only on the remote buffers.
  return UseDescriptionsForwardsBuffers(this, RemoteBuffers().size(),
                                        BufferUseKind::USE_ALIAS_READ_WRITE);
}

HloPoplarBufferDescriptions HloRemoteParameterStore::GetBufferDescriptions()
    const {
  return BufferDescriptionsNoAllocations();
}

const FindConsumersExtensionResults HloRemoteParameterStore::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloRemoteParameterStore::AllowNonInplaceLowering() const { return false; }

bool HloRemoteParameterStore::IsPopOpsElementwise() const { return false; }

absl::Span<HloInstruction* const> HloRemoteParameterStore::RemoteBuffers()
    const {
  return absl::MakeSpan(operands()).first(operand_count() / 2);
}

absl::Span<HloInstruction* const> HloRemoteParameterStore::ValuesToStore()
    const {
  return absl::MakeSpan(operands()).last(operand_count() / 2);
}

std::unique_ptr<HloInstruction>
HloRemoteParameterStore::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateHloRemoteParameterStore(operands, replication_factors_);
}

std::unique_ptr<HloInstruction> CreateHloRemoteParameterStore(
    absl::Span<HloInstruction* const> rbuffers_and_values,
    const std::vector<uint64>& replication_factors) {
  return absl::make_unique<HloRemoteParameterStore>(rbuffers_and_values,
                                                    replication_factors);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloRemoteParameterLoadFactoryFunc(
    HloCustomCallInstruction* call) {
  if (call->operand_count() != 1) {
    return FailedPrecondition(
        "Expected remote buffer load to have one operand, but got %d",
        call->operand_count());
  }
  if (call->mutable_operand(0)->opcode() != HloOpcode::kParameter) {
    return FailedPrecondition("Can only remote buffer load from a parameter");
  }
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  TF_ASSIGN_OR_RETURN(uint64 replication_factor,
                      attribute_map.GetAttributeAsUInt64("replication_factor"));

  return CreateHloRemoteParameterLoad(call->operands(), {replication_factor});
}

static HloPoplarInstructionFactory remote_parameter_load_factory(
    PoplarOp::RemoteParameterLoad, HloRemoteParameterLoadFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloRemoteParameterStoreFactoryFunc(
    HloCustomCallInstruction* call) {
  if (call->operand_count() != 2) {
    return FailedPrecondition(
        "Expected remote buffer store to have two operands, but got %d",
        call->operand_count());
  }
  if (call->mutable_operand(0)->opcode() != HloOpcode::kParameter) {
    return FailedPrecondition("Can only remote buffer store to a parameter");
  }
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  TF_ASSIGN_OR_RETURN(uint64 replication_factor,
                      attribute_map.GetAttributeAsUInt64("replication_factor"));

  return CreateHloRemoteParameterStore(
      {call->mutable_operand(0), call->mutable_operand(1)},
      {replication_factor});
}

static HloPoplarInstructionFactory remote_parameter_store_factory(
    PoplarOp::RemoteParameterStore, HloRemoteParameterStoreFactoryFunc);
}  // namespace

HloCreateBuffer::HloCreateBuffer(
    const Shape& shape, bool is_remote,
    absl::optional<HloRemoteBufferInfo> remote_buffer_info)
    : HloPoplarInstruction(shape, {}, PoplarOp::CreateBuffer, is_remote,
                           remote_buffer_info),
      is_remote_(is_remote),
      remote_buffer_info_(remote_buffer_info) {
  CHECK(!shape.IsTuple());
  // Set the instruction to have side effect to prevent it from being merged
  // with other similarly shaped buffers.
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64> HloCreateBuffer::AllocatingIndices() const {
  return {};
}

bool HloCreateBuffer::AllocatingOutput() const { return !IsRemoteBuffer(); }

absl::flat_hash_map<int64, int64> HloCreateBuffer::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloCreateBuffer::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloCreateBuffer::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(
      this, is_remote_ ? BufferLocality::kRemoteMemory
                       : BufferLocality::kDeviceMemory);
}

const FindConsumersExtensionResults HloCreateBuffer::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloCreateBuffer::AllowNonInplaceLowering() const { return false; }

bool HloCreateBuffer::IsPopOpsElementwise() const { return false; }

absl::optional<HloRemoteBufferInfo> HloCreateBuffer::RemoteBufferInfo() const {
  CHECK(is_remote_);
  return remote_buffer_info_;
}

std::unique_ptr<HloInstruction> HloCreateBuffer::CloneWithRemoteBufferInfo(
    const HloRemoteBufferInfo& info) const {
  CHECK(is_remote_);
  auto clone = absl::make_unique<HloCreateBuffer>(shape(), is_remote_, info);
  SetupDerivedInstruction(clone.get());
  clone->set_raw_backend_config_string(raw_backend_config_string());
  return clone;
}

std::unique_ptr<HloInstruction> HloCreateBuffer::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  CHECK_EQ(operands.size(), 0);
  return CreateHloCreateBuffer(shape, IsRemoteBuffer());
}

std::vector<std::string> HloCreateBuffer::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("is_remote=" + std::to_string(is_remote_));

  if (remote_buffer_info_.has_value()) {
    attributes.push_back("remote_buffer_name=" + remote_buffer_info_->name);
    attributes.push_back("remote_buffer_num_merged=" +
                         std::to_string(remote_buffer_info_->num_merged));
    attributes.push_back("remote_buffer_merge_offset=" +
                         std::to_string(remote_buffer_info_->merge_offset));
  }

  return attributes;
}

std::unique_ptr<HloInstruction> CreateHloCreateBuffer(const Shape& shape,
                                                      bool is_remote) {
  return absl::make_unique<HloCreateBuffer>(shape, is_remote, absl::nullopt);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloCreateBufferFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  TF_ASSIGN_OR_RETURN(bool is_remote,
                      attribute_map.GetAttributeAsBool("is_remote"));
  return CreateHloCreateBuffer(call->shape(), is_remote);
}
static HloPoplarInstructionFactory create_buffer_factory(
    PoplarOp::CreateBuffer, HloCreateBufferFactoryFunc);
}  // namespace

HloBufferLoadSlice::HloBufferLoadSlice(
    const Shape& shape, absl::Span<HloInstruction* const> rbuffers_and_offsets,
    const std::vector<uint64>& replication_factors)
    : HloAbstractRemoteLoadStore(shape, rbuffers_and_offsets,
                                 replication_factors,
                                 PoplarOp::BufferLoadSlice) {
  // The first half of the operands are the remote buffers, the second half
  // are the corresponding offsets to load from.
  CHECK_GE(rbuffers_and_offsets.size(), 2);
  CHECK_EQ(rbuffers_and_offsets.size() % 2, 0);
  CHECK_EQ(rbuffers_and_offsets.size() / 2, replication_factors.size());
}

absl::flat_hash_set<int64> HloBufferLoadSlice::AllocatingIndices() const {
  return {};
}

bool HloBufferLoadSlice::AllocatingOutput() const { return true; }

absl::flat_hash_map<int64, int64> HloBufferLoadSlice::LayoutDependencies()
    const {
  return {};
}

HloPoplarUseDescriptions HloBufferLoadSlice::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloBufferLoadSlice::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloBufferLoadSlice::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloBufferLoadSlice::AllowNonInplaceLowering() const { return false; }

bool HloBufferLoadSlice::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction> HloBufferLoadSlice::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return absl::make_unique<HloBufferLoadSlice>(shape, operands,
                                               replication_factors_);
}

absl::Span<HloInstruction* const> HloBufferLoadSlice::RemoteBuffers() const {
  return absl::MakeSpan(operands()).first(operand_count() / 2);
}

absl::Span<HloInstruction* const> HloBufferLoadSlice::Offsets() const {
  return absl::MakeSpan(operands()).last(operand_count() / 2);
}

std::unique_ptr<HloInstruction> CreateBufferLoadSlice(
    const Shape& shape, HloInstruction* const buffer,
    HloInstruction* const offset, uint64 replication_factor) {
  return absl::make_unique<HloBufferLoadSlice>(
      shape, std::vector<HloInstruction*>{buffer, offset},
      std::vector<uint64>{replication_factor});
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloBufferLoadSliceFactoryFunc(
    HloCustomCallInstruction* call) {
  CHECK_EQ(call->operand_count(), 2) << call->ToString();

  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  TF_ASSIGN_OR_RETURN(uint64 replication_factor,
                      attribute_map.GetAttributeAsUInt64("replication_factor"));

  return CreateBufferLoadSlice(call->shape(), call->mutable_operand(0),
                               call->mutable_operand(1), replication_factor);
}
static HloPoplarInstructionFactory buffer_load_slice_factory(
    PoplarOp::BufferLoadSlice, HloBufferLoadSliceFactoryFunc);
}  // namespace

HloBufferStoreSlice::HloBufferStoreSlice(
    const Shape& shape,
    absl::Span<HloInstruction* const> rbuffers_values_and_offsets,
    const std::vector<uint64>& replication_factors)
    : HloAbstractRemoteLoadStore(shape, rbuffers_values_and_offsets,
                                 replication_factors,
                                 PoplarOp::BufferStoreSlice) {
  // The first third of the operands are the remote buffers, the second
  // are the corresponding values to store, and the third are the offsets to
  // store into.
  CHECK_GE(rbuffers_values_and_offsets.size(), 3);
  CHECK_EQ(rbuffers_values_and_offsets.size() % 3, 0);
  CHECK_EQ(rbuffers_values_and_offsets.size() / 3, replication_factors.size());
}

HloPoplarUseDescriptions HloBufferStoreSlice::GetUseDescriptions() const {
  // The remote buffers are in-place, but only on the remote buffers.
  return UseDescriptionsForwardsBuffers(this, RemoteBuffers().size(),
                                        BufferUseKind::USE_ALIAS_READ_WRITE);
}

HloPoplarBufferDescriptions HloBufferStoreSlice::GetBufferDescriptions() const {
  return BufferDescriptionsNoAllocations();
}

std::unique_ptr<HloInstruction> HloBufferStoreSlice::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return absl::make_unique<HloBufferStoreSlice>(shape, operands,
                                                replication_factors_);
}

absl::Span<HloInstruction* const> HloBufferStoreSlice::RemoteBuffers() const {
  return absl::MakeSpan(operands()).first(operand_count() / 3);
}

absl::Span<HloInstruction* const> HloBufferStoreSlice::ValuesToStore() const {
  return absl::MakeSpan(operands())
      .subspan(operand_count() / 3, operand_count() / 3);
}

absl::Span<HloInstruction* const> HloBufferStoreSlice::Offsets() const {
  return absl::MakeSpan(operands()).last(operand_count() / 3);
}

std::unique_ptr<HloInstruction> CreateBufferStoreSlice(
    HloInstruction* const buffer, HloInstruction* const slice,
    HloInstruction* const offset, uint64 replication_factor) {
  return absl::make_unique<HloBufferStoreSlice>(
      buffer->shape(), std::vector<HloInstruction*>{buffer, slice, offset},
      std::vector<uint64>{replication_factor});
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloBufferStoreSliceFactoryFunc(
    HloCustomCallInstruction* call) {
  CHECK_EQ(call->operand_count(), 3) << call->ToString();

  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  TF_ASSIGN_OR_RETURN(uint64 replication_factor,
                      attribute_map.GetAttributeAsUInt64("replication_factor"));

  return CreateBufferStoreSlice(call->mutable_operand(0),
                                call->mutable_operand(1),
                                call->mutable_operand(2), replication_factor);
}
static HloPoplarInstructionFactory buffer_store_slice_factory(
    PoplarOp::BufferStoreSlice, HloBufferStoreSliceFactoryFunc);
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
