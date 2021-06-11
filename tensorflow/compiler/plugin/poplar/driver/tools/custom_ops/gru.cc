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

#include <memory>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/gru.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/rnn.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/tf2xla/type_util.h"

namespace xla {
namespace poplarplugin {
namespace rnn_helper {
GRUAttributes::GRUAttributes(int32 num_channels, bool is_training,
                             xla::PrimitiveType partials_xla_type,
                             ActivationType activation,
                             ActivationType recurrent_activation,
                             bool output_full_sequence, bool reset_after)
    : RNNAttributes(num_channels, is_training, partials_xla_type, activation,
                    recurrent_activation, output_full_sequence),
      reset_after(reset_after) {}
// Helper for parsing the attribute map when converting the custom call
// instruction.
StatusOr<GRUAttributes> GRUAttributes::Parse(
    const HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

  TF_ASSIGN_OR_RETURN(RNNAttributes rnn_attibutes, RNNAttributes::Parse(call));

  TF_ASSIGN_OR_RETURN(bool reset_after,
                      attribute_map.GetAttributeAsBool("reset_after"));

  return GRUAttributes(rnn_attibutes.num_channels, rnn_attibutes.is_training,
                       rnn_attibutes.partials_xla_type,
                       rnn_attibutes.activation,
                       rnn_attibutes.recurrent_activation,
                       rnn_attibutes.output_full_sequence, reset_after);
}
}  // namespace rnn_helper

using ActivationType = rnn_helper::ActivationType;

HloGRUInstructionCommon::HloGRUInstructionCommon(bool reset_after)
    : reset_after_(reset_after) {}

bool HloGRUInstructionCommon::reset_after() const { return reset_after_; }

HloGRUFwdInstruction::HloGRUFwdInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, ActivationType activation,
    ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, bool output_full_sequence,
    bool reset_after)
    : HloRNNFwdInstruction(PoplarOp::GRULayerFwd, shape, operands, is_training,
                           activation, recurrent_activation, num_channels,
                           partials_type, output_full_sequence, reset_after),
      HloGRUInstructionCommon(reset_after) {}

absl::flat_hash_set<int64> HloGRUFwdInstruction::AllocatingIndices() const {
  return {0, 1, 2, 3};
}

bool HloGRUFwdInstruction::AllocatingOutput() const { return false; }

std::unique_ptr<HloInstruction> HloGRUFwdInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateGRUFwd(shape, operands, is_training(), activation(),
                      recurrent_activation(), num_channels(), partials_type(),
                      output_full_sequence(), reset_after());
}

std::unique_ptr<HloInstruction> CreateGRUFwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, ActivationType activation,
    ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, bool output_full_sequence,
    bool reset_after) {
  return absl::make_unique<HloGRUFwdInstruction>(
      shape, operands, is_training, activation, recurrent_activation,
      num_channels, partials_type, output_full_sequence, reset_after);
}

HloGRUBwdInstruction::HloGRUBwdInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, ActivationType activation,
    ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, bool output_full_sequence,
    bool reset_after)
    : HloRNNBwdInstruction(PoplarOp::GRULayerBwd, shape, operands, is_training,
                           activation, recurrent_activation, num_channels,
                           partials_type, output_full_sequence, reset_after),
      HloGRUInstructionCommon(reset_after) {}

std::unique_ptr<HloInstruction> HloGRUBwdInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateGRUBwd(shape, operands, is_training(), activation(),
                      recurrent_activation(), num_channels(), partials_type(),
                      output_full_sequence(), reset_after());
}

std::unique_ptr<HloInstruction> CreateGRUBwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, ActivationType activation,
    ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, bool output_full_sequence,
    bool reset_after) {
  return absl::make_unique<HloGRUBwdInstruction>(
      shape, operands, is_training, activation, recurrent_activation,
      num_channels, partials_type, output_full_sequence, reset_after);
}

HloDynamicGRUFwdInstruction::HloDynamicGRUFwdInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, ActivationType activation,
    ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, bool output_full_sequence,
    bool reset_after)
    : HloRNNFwdInstruction(PoplarOp::DynamicGRULayerFwd, shape, operands,
                           is_training, activation, recurrent_activation,
                           num_channels, partials_type, output_full_sequence,
                           reset_after),
      HloGRUInstructionCommon(reset_after) {}

absl::flat_hash_set<int64> HloDynamicGRUFwdInstruction::AllocatingIndices()
    const {
  return {0, 1, 2, 3};
}

bool HloDynamicGRUFwdInstruction::AllocatingOutput() const { return false; }

std::unique_ptr<HloInstruction>
HloDynamicGRUFwdInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateDynamicGRUFwd(
      shape, operands, is_training(), activation(), recurrent_activation(),
      num_channels(), partials_type(), output_full_sequence(), reset_after());
}

std::unique_ptr<HloInstruction> CreateDynamicGRUFwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, ActivationType activation,
    ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, bool output_full_sequence,
    bool reset_after) {
  return absl::make_unique<HloDynamicGRUFwdInstruction>(
      shape, operands, is_training, activation, recurrent_activation,
      num_channels, partials_type, output_full_sequence, reset_after);
}

HloDynamicGRUBwdInstruction::HloDynamicGRUBwdInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, ActivationType activation,
    ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, bool output_full_sequence,
    bool reset_after)
    : HloRNNBwdInstruction(PoplarOp::DynamicGRULayerBwd, shape, operands,
                           is_training, activation, recurrent_activation,
                           num_channels, partials_type, output_full_sequence,
                           reset_after),
      HloGRUInstructionCommon(reset_after) {}

std::unique_ptr<HloInstruction>
HloDynamicGRUBwdInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateDynamicGRUBwd(
      shape, operands, is_training(), activation(), recurrent_activation(),
      num_channels(), partials_type(), output_full_sequence(), reset_after());
}

std::unique_ptr<HloInstruction> CreateDynamicGRUBwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, ActivationType activation,
    ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, bool output_full_sequence,
    bool reset_after) {
  return absl::make_unique<HloDynamicGRUBwdInstruction>(
      shape, operands, is_training, activation, recurrent_activation,
      num_channels, partials_type, output_full_sequence, reset_after);
}

HloAUGRUFwdInstruction::HloAUGRUFwdInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, ActivationType activation,
    ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, bool output_full_sequence,
    bool reset_after)
    : HloRNNFwdInstruction(PoplarOp::AUGRULayerFwd, shape, operands,
                           is_training, activation, recurrent_activation,
                           num_channels, partials_type, output_full_sequence,
                           reset_after),
      HloGRUInstructionCommon(reset_after) {}

absl::flat_hash_set<int64> HloAUGRUFwdInstruction::AllocatingIndices() const {
  return {0, 1, 2, 3, 4, 5};
}

bool HloAUGRUFwdInstruction::AllocatingOutput() const { return false; }

std::unique_ptr<HloInstruction>
HloAUGRUFwdInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateAUGRUFwd(shape, operands, is_training(), activation(),
                        recurrent_activation(), num_channels(), partials_type(),
                        output_full_sequence(), reset_after());
}

std::unique_ptr<HloInstruction> CreateAUGRUFwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, ActivationType activation,
    ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, bool output_full_sequence,
    bool reset_after) {
  return absl::make_unique<HloAUGRUFwdInstruction>(
      shape, operands, is_training, activation, recurrent_activation,
      num_channels, partials_type, output_full_sequence, reset_after);
}

HloAUGRUBwdInstruction::HloAUGRUBwdInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, ActivationType activation,
    ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, bool output_full_sequence,
    bool reset_after)
    : HloRNNBwdInstruction(PoplarOp::AUGRULayerBwd, shape, operands,
                           is_training, activation, recurrent_activation,
                           num_channels, partials_type, output_full_sequence,
                           reset_after),
      HloGRUInstructionCommon(reset_after) {}

std::unique_ptr<HloInstruction>
HloAUGRUBwdInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateAUGRUBwd(shape, operands, is_training(), activation(),
                        recurrent_activation(), num_channels(), partials_type(),
                        output_full_sequence(), reset_after());
}

std::unique_ptr<HloInstruction> CreateAUGRUBwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, ActivationType activation,
    ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, bool output_full_sequence,
    bool reset_after) {
  return absl::make_unique<HloAUGRUBwdInstruction>(
      shape, operands, is_training, activation, recurrent_activation,
      num_channels, partials_type, output_full_sequence, reset_after);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloGRUFwdFactoryFunc(
    HloCustomCallInstruction* call) {
  TF_ASSIGN_OR_RETURN(auto parsed_attributes,
                      rnn_helper::GRUAttributes::Parse(call));

  return CreateGRUFwd(
      call->shape(), call->operands(), parsed_attributes.is_training,
      parsed_attributes.activation, parsed_attributes.recurrent_activation,
      parsed_attributes.num_channels, parsed_attributes.partials_xla_type,
      parsed_attributes.output_full_sequence, parsed_attributes.reset_after);
}

static HloPoplarInstructionFactory gru_fwd_factory(PoplarOp::GRULayerFwd,
                                                   HloGRUFwdFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloGRUBwdFactoryFunc(
    HloCustomCallInstruction* call) {
  TF_ASSIGN_OR_RETURN(auto parsed_attributes,
                      rnn_helper::GRUAttributes::Parse(call));

  return CreateGRUBwd(
      call->shape(), call->operands(), parsed_attributes.is_training,
      parsed_attributes.activation, parsed_attributes.recurrent_activation,
      parsed_attributes.num_channels, parsed_attributes.partials_xla_type,
      parsed_attributes.output_full_sequence, parsed_attributes.reset_after);
}

static HloPoplarInstructionFactory gru_bwd_factory(PoplarOp::GRULayerBwd,
                                                   HloGRUBwdFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloDynamicGRUFwdFactoryFunc(
    HloCustomCallInstruction* call) {
  TF_ASSIGN_OR_RETURN(auto parsed_attributes,
                      rnn_helper::GRUAttributes::Parse(call));

  return CreateDynamicGRUFwd(
      call->shape(), call->operands(), parsed_attributes.is_training,
      parsed_attributes.activation, parsed_attributes.recurrent_activation,
      parsed_attributes.num_channels, parsed_attributes.partials_xla_type,
      parsed_attributes.output_full_sequence, parsed_attributes.reset_after);
}

static HloPoplarInstructionFactory dynamic_gru_fwd_factory(
    PoplarOp::DynamicGRULayerFwd, HloDynamicGRUFwdFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloDynamicGRUBwdFactoryFunc(
    HloCustomCallInstruction* call) {
  TF_ASSIGN_OR_RETURN(auto parsed_attributes,
                      rnn_helper::GRUAttributes::Parse(call));

  return CreateDynamicGRUBwd(
      call->shape(), call->operands(), parsed_attributes.is_training,
      parsed_attributes.activation, parsed_attributes.recurrent_activation,
      parsed_attributes.num_channels, parsed_attributes.partials_xla_type,
      parsed_attributes.output_full_sequence, parsed_attributes.reset_after);
}

static HloPoplarInstructionFactory dynamic_gru_bwd_factory(
    PoplarOp::DynamicGRULayerBwd, HloDynamicGRUBwdFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloAUGRUFwdFactoryFunc(
    HloCustomCallInstruction* call) {
  TF_ASSIGN_OR_RETURN(auto parsed_attributes,
                      rnn_helper::GRUAttributes::Parse(call));

  return CreateAUGRUFwd(
      call->shape(), call->operands(), parsed_attributes.is_training,
      parsed_attributes.activation, parsed_attributes.recurrent_activation,
      parsed_attributes.num_channels, parsed_attributes.partials_xla_type,
      parsed_attributes.output_full_sequence, parsed_attributes.reset_after);
}

static HloPoplarInstructionFactory augru_fwd_factory(PoplarOp::AUGRULayerFwd,
                                                     HloAUGRUFwdFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloAUGRUBwdFactoryFunc(
    HloCustomCallInstruction* call) {
  TF_ASSIGN_OR_RETURN(auto parsed_attributes,
                      rnn_helper::GRUAttributes::Parse(call));

  return CreateAUGRUBwd(
      call->shape(), call->operands(), parsed_attributes.is_training,
      parsed_attributes.activation, parsed_attributes.recurrent_activation,
      parsed_attributes.num_channels, parsed_attributes.partials_xla_type,
      parsed_attributes.output_full_sequence, parsed_attributes.reset_after);
}

static HloPoplarInstructionFactory augru_bwd_factory(PoplarOp::AUGRULayerBwd,
                                                     HloAUGRUBwdFactoryFunc);

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
