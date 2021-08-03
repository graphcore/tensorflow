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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {
namespace {
Shape GetShape(absl::Span<HloInstruction* const> operands, PoplarOp op) {
  switch (op) {
    case PoplarOp::StatefulGradientAccumulateWithMomentum:
    case PoplarOp::StatefulGradientAccumulateWithMomentumAndAllReduceWithNorm: {
      std::vector<HloInstruction*> shape_insts = {operands.begin(),
                                                  operands.end() - 1};
      return GetHloPoplarInstructionShape(shape_insts);
    }
    default: { return GetHloPoplarInstructionShape(operands); }
  }
}
}  // namespace

HloStatefulGradientAccumulate::HloStatefulGradientAccumulate(
    absl::Span<HloInstruction* const> operands, int32 num_mini_batches,
    PoplarOp op)
    : HloPoplarInstruction(GetShape(operands, op), operands, op,
                           num_mini_batches),
      num_mini_batches_(num_mini_batches) {}

absl::flat_hash_set<int64> HloStatefulGradientAccumulate::AllocatingIndices()
    const {
  return {};
}

bool HloStatefulGradientAccumulate::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64>
HloStatefulGradientAccumulate::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloStatefulGradientAccumulate::GetUseDescriptions()
    const {
  return UseDescriptionsForwardsBuffers(this, operand_count(),
                                        BufferUseKind::USE_ALIAS_READ_WRITE);
}

HloPoplarBufferDescriptions
HloStatefulGradientAccumulate::GetBufferDescriptions() const {
  return BufferDescriptionsNoAllocations();
}

const FindConsumersExtensionResults
HloStatefulGradientAccumulate::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloStatefulGradientAccumulate::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction>
HloStatefulGradientAccumulate::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloStatefulGradientAccumulate>(new_operands,
                                                          num_mini_batches_);
}

std::vector<std::string>
HloStatefulGradientAccumulate::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("num_mini_batches=" + std::to_string(num_mini_batches_));

  return attributes;
}

std::unique_ptr<HloInstruction> CreateStatefulGradientAccumulation(
    absl::Span<HloInstruction* const> operands, int32 num_mini_batches) {
  return absl::make_unique<HloStatefulGradientAccumulate>(operands,
                                                          num_mini_batches);
}

HloStatefulGradientAccumulateAndAllReduce::
    HloStatefulGradientAccumulateAndAllReduce(
        absl::Span<HloInstruction* const> operands, int32 num_mini_batches)
    : HloStatefulGradientAccumulate(
          operands, num_mini_batches,
          PoplarOp::StatefulGradientAccumulateAndAllReduce) {}

std::unique_ptr<HloInstruction>
HloStatefulGradientAccumulateAndAllReduce::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloStatefulGradientAccumulateAndAllReduce>(
      new_operands, num_mini_batches_);
}

std::unique_ptr<HloInstruction> CreateStatefulGradientAccumulateAndAllReduce(
    absl::Span<HloInstruction* const> operands, int32 num_mini_batches) {
  return absl::make_unique<HloStatefulGradientAccumulateAndAllReduce>(
      operands, num_mini_batches);
}

HloStatefulGradientAccumulateWithMomentum::
    HloStatefulGradientAccumulateWithMomentum(
        absl::Span<HloInstruction* const> operands, int32 num_mini_batches)
    : HloStatefulGradientAccumulate(
          operands, num_mini_batches,
          PoplarOp::StatefulGradientAccumulateWithMomentum) {}

HloPoplarUseDescriptions
HloStatefulGradientAccumulateWithMomentum::GetUseDescriptions() const {
  return UseDescriptionsForwardsBuffers(this, 2,
                                        BufferUseKind::USE_ALIAS_READ_WRITE);
}

HloPoplarBufferDescriptions
HloStatefulGradientAccumulateWithMomentum::GetBufferDescriptions() const {
  return BufferDescriptionsNoAllocations();
}

absl::flat_hash_map<int64, int64>
HloStatefulGradientAccumulateWithMomentum::LayoutDependencies() const {
  return {{0, 1}};
}

std::unique_ptr<HloInstruction>
HloStatefulGradientAccumulateWithMomentum::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloStatefulGradientAccumulateWithMomentum>(
      new_operands, num_mini_batches_);
}

std::unique_ptr<HloInstruction> CreateStatefulGradientAccumulationWithMomentum(
    absl::Span<HloInstruction* const> operands, int32 num_mini_batches) {
  return absl::make_unique<HloStatefulGradientAccumulateWithMomentum>(
      operands, num_mini_batches);
}

HloStatefulGradientAccumulateWithMomentumAndAllReduceWithNorm::
    HloStatefulGradientAccumulateWithMomentumAndAllReduceWithNorm(
        absl::Span<HloInstruction* const> operands, int32 num_mini_batches)
    : HloStatefulGradientAccumulate(
          operands, num_mini_batches,
          PoplarOp::
              StatefulGradientAccumulateWithMomentumAndAllReduceWithNorm) {}

HloPoplarUseDescriptions
HloStatefulGradientAccumulateWithMomentumAndAllReduceWithNorm::
    GetUseDescriptions() const {
  return UseDescriptionsForwardsBuffers(this, operand_count() - 1,
                                        BufferUseKind::USE_ALIAS_READ_WRITE);
}

HloPoplarBufferDescriptions
HloStatefulGradientAccumulateWithMomentumAndAllReduceWithNorm::
    GetBufferDescriptions() const {
  return BufferDescriptionsNoAllocations();
}

absl::flat_hash_map<int64, int64>
HloStatefulGradientAccumulateWithMomentumAndAllReduceWithNorm::
    LayoutDependencies() const {
  // The layouts of the accumulator depend on the layout of the gradient.
  absl::flat_hash_map<int64, int64> deps;
  const uint64 num_grads = operand_count() / 2;
  for (uint64 i = 0; i != num_grads; ++i) {
    deps[i] = num_grads + i;
  }
  return deps;
}

std::unique_ptr<HloInstruction>
HloStatefulGradientAccumulateWithMomentumAndAllReduceWithNorm::
    CloneWithNewOperandsImpl(const Shape& shape,
                             absl::Span<HloInstruction* const> new_operands,
                             HloCloneContext*) const {
  return absl::make_unique<
      HloStatefulGradientAccumulateWithMomentumAndAllReduceWithNorm>(
      new_operands, num_mini_batches_);
}

std::unique_ptr<HloInstruction>
CreateStatefulGradientAccumulationWithMomentumAndAllReduceWithNorm(
    absl::Span<HloInstruction* const> operands, int32 num_mini_batches) {
  return absl::make_unique<
      HloStatefulGradientAccumulateWithMomentumAndAllReduceWithNorm>(
      operands, num_mini_batches);
}

std::unique_ptr<HloInstruction>
HloGradientAccumulatorCreate::CreateFromShapeOnly(const Shape& shape,
                                                  bool is_remote) {
  return absl::make_unique<HloGradientAccumulatorCreate>(
      shape, std::vector<HloInstruction*>{}, is_remote);
}

std::unique_ptr<HloInstruction>
HloGradientAccumulatorCreate::CreateFromShapeAndVariable(
    const Shape& shape, HloInstruction* const variable, bool is_remote) {
  return absl::make_unique<HloGradientAccumulatorCreate>(
      shape, std::vector<HloInstruction*>{variable}, is_remote);
}

HloGradientAccumulatorCreate::HloGradientAccumulatorCreate(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_remote, absl::optional<HloRemoteBufferInfo> remote_buffer_info)
    : HloPoplarInstruction(shape, operands, PoplarOp::GradientAccumulatorCreate,
                           is_remote, remote_buffer_info),
      is_remote_(is_remote),
      remote_buffer_info_(remote_buffer_info) {
  CHECK_LE(operands.size(), 1);
  if (remote_buffer_info_.has_value()) {
    CHECK(is_remote_);
  }
  // Mark the creator as stateful so that it does not get merged with other same
  // shaped accumulators.
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64> HloGradientAccumulatorCreate::AllocatingIndices()
    const {
  return {};
}

bool HloGradientAccumulatorCreate::AllocatingOutput() const { return true; }

absl::flat_hash_map<int64, int64>
HloGradientAccumulatorCreate::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloGradientAccumulatorCreate::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloGradientAccumulatorCreate::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(
      this, is_remote_ ? BufferLocality::kRemoteMemory
                       : BufferLocality::kDeviceMemory);
}

const FindConsumersExtensionResults HloGradientAccumulatorCreate::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloGradientAccumulatorCreate::IsPopOpsElementwise() const { return false; }

absl::optional<HloRemoteBufferInfo>
HloGradientAccumulatorCreate::RemoteBufferInfo() const {
  CHECK(is_remote_);
  return remote_buffer_info_;
}

std::unique_ptr<HloInstruction>
HloGradientAccumulatorCreate::CloneWithRemoteBufferInfo(
    const HloRemoteBufferInfo& info) const {
  CHECK(is_remote_);
  auto clone = absl::make_unique<HloGradientAccumulatorCreate>(
      shape(), operands(), is_remote_, info);
  SetupDerivedInstruction(clone.get());
  clone->set_raw_backend_config_string(raw_backend_config_string());
  return clone;
}

std::unique_ptr<HloInstruction>
HloGradientAccumulatorCreate::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloGradientAccumulatorCreate>(shape, new_operands);
}

std::vector<std::string>
HloGradientAccumulatorCreate::ExtraPoplarAttributesToStringImpl(
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

std::unique_ptr<HloInstruction> CreateGradientAccumulatorCreate(
    HloInstruction* const variable, bool is_remote) {
  return HloGradientAccumulatorCreate::CreateFromShapeAndVariable(
      variable->shape(), variable, is_remote);
}

std::unique_ptr<HloInstruction> CreateGradientAccumulatorCreate(
    const Shape& shape, bool is_remote) {
  return HloGradientAccumulatorCreate::CreateFromShapeOnly(shape, is_remote);
}

HloGradientAccumulatorAdd::HloGradientAccumulatorAdd(
    HloInstruction* const accumulator, HloInstruction* const gradient)
    : HloPoplarInstruction(accumulator->shape(), {accumulator, gradient},
                           PoplarOp::GradientAccumulatorAdd) {}

absl::flat_hash_set<int64> HloGradientAccumulatorAdd::AllocatingIndices()
    const {
  return {};
}

bool HloGradientAccumulatorAdd::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64>
HloGradientAccumulatorAdd::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloGradientAccumulatorAdd::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloGradientAccumulatorAdd::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloGradientAccumulatorAdd::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloGradientAccumulatorAdd::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction>
HloGradientAccumulatorAdd::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  CHECK_EQ(new_operands.size(), 2);
  return absl::make_unique<HloGradientAccumulatorAdd>(new_operands[0],
                                                      new_operands[1]);
}

std::vector<std::string>
HloGradientAccumulatorAdd::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateGradientAccumulatorAdd(
    HloInstruction* const accumulator, HloInstruction* const gradient) {
  return absl::make_unique<HloGradientAccumulatorAdd>(accumulator, gradient);
}

HloGradientAccumulatorSink::HloGradientAccumulatorSink(
    absl::Span<HloInstruction* const> operands)
    : HloPoplarInstruction(operands[0]->shape(), operands,
                           PoplarOp::GradientAccumulatorSink) {}

absl::flat_hash_set<int64> HloGradientAccumulatorSink::AllocatingIndices()
    const {
  return {};
}

bool HloGradientAccumulatorSink::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64>
HloGradientAccumulatorSink::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloGradientAccumulatorSink::GetUseDescriptions()
    const {
  HloPoplarUseDescriptions descriptions;
  for (int64 i = 0; i != operand_count(); ++i) {
    descriptions.push_back(HloPoplarUseDescription{
        i, ShapeIndex{}, ShapeIndex{}, BufferUseKind::USE_ALIAS_READ_ONLY});
  }
  return descriptions;
}

HloPoplarBufferDescriptions HloGradientAccumulatorSink::GetBufferDescriptions()
    const {
  return BufferDescriptionsNoAllocations();
}

const FindConsumersExtensionResults HloGradientAccumulatorSink::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloGradientAccumulatorSink::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction>
HloGradientAccumulatorSink::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloGradientAccumulatorSink>(new_operands);
}

std::vector<std::string>
HloGradientAccumulatorSink::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return std::vector<std::string>();
}

std::unique_ptr<HloInstruction> CreateGradientAccumulatorSink(
    absl::Span<HloInstruction* const> operands) {
  return absl::make_unique<HloGradientAccumulatorSink>(operands);
}

namespace {

StatusOr<std::unique_ptr<HloInstruction>>
HloStatefulGradientAccumulateFactoryFunc(HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  // Get the attribute values
  TF_ASSIGN_OR_RETURN(int32 num_mini_batches,
                      attribute_map.GetAttributeAsInt("num_mini_batches"));

  return CreateStatefulGradientAccumulation(call->operands(), num_mini_batches);
}

static HloPoplarInstructionFactory stateful_gradient_accumulate_factory(
    PoplarOp::StatefulGradientAccumulate,
    HloStatefulGradientAccumulateFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>>
HloStatefulGradientAccumulateWithMomentumFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  // Get the attribute values
  TF_ASSIGN_OR_RETURN(int32 num_mini_batches,
                      attribute_map.GetAttributeAsInt("num_mini_batches"));

  return CreateStatefulGradientAccumulationWithMomentum(call->operands(),
                                                        num_mini_batches);
}

static HloPoplarInstructionFactory
    stateful_gradient_accumulate_factory_with_momentum(
        PoplarOp::StatefulGradientAccumulateWithMomentum,
        HloStatefulGradientAccumulateWithMomentumFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>>
HloStatefulGradientAccumulateWithMomentumAndAllReduceWithNormFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  // Get the attribute values
  TF_ASSIGN_OR_RETURN(int32 num_mini_batches,
                      attribute_map.GetAttributeAsInt("num_mini_batches"));

  return CreateStatefulGradientAccumulationWithMomentumAndAllReduceWithNorm(
      call->operands(), num_mini_batches);
}

static HloPoplarInstructionFactory
    stateful_gradient_accumulate_factory_with_momentum_and_all_reduce_with_norm(
        PoplarOp::StatefulGradientAccumulateWithMomentumAndAllReduceWithNorm,
        HloStatefulGradientAccumulateWithMomentumAndAllReduceWithNormFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>>
HloGradientAccumulatorCreateFactoryFunc(HloCustomCallInstruction* call) {
  bool is_remote = false;

  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  if (attribute_map.HasAttribute("is_remote")) {
    TF_ASSIGN_OR_RETURN(is_remote,
                        attribute_map.GetAttributeAsBool("is_remote"));
  }

  CHECK_LE(call->operand_count(), 1);
  if (call->operand_count() == 1) {
    return HloGradientAccumulatorCreate::CreateFromShapeAndVariable(
        call->shape(), call->mutable_operand(0), is_remote);
  }
  return HloGradientAccumulatorCreate::CreateFromShapeOnly(call->shape(),
                                                           is_remote);
}

static HloPoplarInstructionFactory gradient_accumulator_creator_factory(
    PoplarOp::GradientAccumulatorCreate,
    HloGradientAccumulatorCreateFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloGradientAccumulatorAddFactoryFunc(
    HloCustomCallInstruction* call) {
  CHECK_EQ(call->operand_count(), 2);
  return CreateGradientAccumulatorAdd(call->mutable_operand(0),
                                      call->mutable_operand(1));
}

static HloPoplarInstructionFactory gradient_accumulator_add_factory(
    PoplarOp::GradientAccumulatorAdd, HloGradientAccumulatorAddFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloGradientAccumulatorSinkFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  return CreateGradientAccumulatorSink(call->operands());
}

static HloPoplarInstructionFactory gradient_accumulator_sink_factory(
    PoplarOp::GradientAccumulatorSink, HloGradientAccumulatorSinkFactoryFunc);

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
