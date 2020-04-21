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

absl::flat_hash_map<int64, int64>
HloStatefulGradientAccumulate::LayoutDependencies() const {
  return {};
}

uint64 HloStatefulGradientAccumulate::NumberOfInplaceOperands() const {
  return operand_count();
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

uint64 HloStatefulGradientAccumulateWithMomentum::NumberOfInplaceOperands()
    const {
  return 2;
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

uint64 HloStatefulGradientAccumulateWithMomentumAndAllReduceWithNorm::
    NumberOfInplaceOperands() const {
  return operand_count() - 1;
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

HloGradientAccumulatorCreate::HloGradientAccumulatorCreate(const Shape& shape)
    : HloPoplarInstruction(shape, {}, PoplarOp::GradientAccumulatorCreate) {
  // Mark the creator as stateful so that it does not get merged with other same
  // shaped accumulators.
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64> HloGradientAccumulatorCreate::AllocatingIndices()
    const {
  return {};
}

absl::flat_hash_map<int64, int64>
HloGradientAccumulatorCreate::LayoutDependencies() const {
  return {};
}

uint64 HloGradientAccumulatorCreate::NumberOfInplaceOperands() const {
  return 0;
}

bool HloGradientAccumulatorCreate::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction>
HloGradientAccumulatorCreate::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  CHECK_EQ(new_operands.size(), 0);
  return absl::make_unique<HloGradientAccumulatorCreate>(shape);
}

std::vector<std::string>
HloGradientAccumulatorCreate::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> CreateGradientAccumulatorCreate(
    const Shape& shape) {
  return absl::make_unique<HloGradientAccumulatorCreate>(shape);
}

HloGradientAccumulatorAdd::HloGradientAccumulatorAdd(
    HloInstruction* const accumulator, HloInstruction* const gradient)
    : HloPoplarInstruction(accumulator->shape(), {accumulator, gradient},
                           PoplarOp::GradientAccumulatorAdd) {}

absl::flat_hash_set<int64> HloGradientAccumulatorAdd::AllocatingIndices()
    const {
  return {};
}

absl::flat_hash_map<int64, int64>
HloGradientAccumulatorAdd::LayoutDependencies() const {
  return {};
}

uint64 HloGradientAccumulatorAdd::NumberOfInplaceOperands() const { return 0; }

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
    absl::Span<HloInstruction* const> operands, int32 num_mini_batches)
    : HloPoplarInstruction(operands[0]->shape(), operands,
                           PoplarOp::GradientAccumulatorSink, num_mini_batches),
      num_mini_batches_(num_mini_batches) {}

absl::flat_hash_set<int64> HloGradientAccumulatorSink::AllocatingIndices()
    const {
  return {};
}

absl::flat_hash_map<int64, int64>
HloGradientAccumulatorSink::LayoutDependencies() const {
  return {};
}

uint64 HloGradientAccumulatorSink::NumberOfInplaceOperands() const {
  return operand_count();
}

bool HloGradientAccumulatorSink::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction>
HloGradientAccumulatorSink::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloGradientAccumulatorSink>(new_operands,
                                                       num_mini_batches_);
}

std::vector<std::string>
HloGradientAccumulatorSink::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("num_mini_batches=" + std::to_string(num_mini_batches_));

  return attributes;
}

std::unique_ptr<HloInstruction> CreateGradientAccumulatorSink(
    absl::Span<HloInstruction* const> operands, int32 num_mini_batches) {
  return absl::make_unique<HloGradientAccumulatorSink>(operands,
                                                       num_mini_batches);
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
  return CreateGradientAccumulatorCreate(call->shape());
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
  // Get the attribute values
  TF_ASSIGN_OR_RETURN(int32 num_mini_batches,
                      attribute_map.GetAttributeAsInt("num_mini_batches"));

  return CreateGradientAccumulatorSink(call->operands(), num_mini_batches);
}

static HloPoplarInstructionFactory gradient_accumulator_sink_factory(
    PoplarOp::GradientAccumulatorSink, HloGradientAccumulatorSinkFactoryFunc);

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
