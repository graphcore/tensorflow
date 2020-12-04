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
                             bool reset_after)
    : RNNAttributes(num_channels, is_training, partials_xla_type),
      reset_after(reset_after) {}
// Helper for parsing the attribute map when converting the custom call
// instruction.
StatusOr<GRUAttributes> GRUAttributes::Parse(
    const HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

  TF_ASSIGN_OR_RETURN(int32 num_channels,
                      attribute_map.GetAttributeAsInt("num_channels"));

  TF_ASSIGN_OR_RETURN(bool is_training,
                      attribute_map.GetAttributeAsBool("is_training"));

  TF_ASSIGN_OR_RETURN(tensorflow::DataType partials_dtype,
                      attribute_map.GetAttributeAsTFDataType("partials_dtype"));

  TF_ASSIGN_OR_RETURN(bool reset_after,
                      attribute_map.GetAttributeAsBool("reset_after"));

  xla::PrimitiveType partials_xla_type;
  TF_CHECK_OK(DataTypeToPrimitiveType(partials_dtype, &partials_xla_type));
  return GRUAttributes(num_channels, is_training, partials_xla_type,
                       reset_after);
}
}  // namespace rnn_helper

HloGRUInstructionCommon::HloGRUInstructionCommon(bool reset_after)
    : reset_after_(reset_after) {}

bool HloGRUInstructionCommon::reset_after() const { return reset_after_; }

HloGRUFwdInstruction::HloGRUFwdInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type,
    bool reset_after)
    : HloRNNFwdInstruction(PoplarOp::GRULayerFwd, shape, operands, is_training,
                           num_channels, partials_type, reset_after),
      HloGRUInstructionCommon(reset_after) {}

absl::flat_hash_set<int64> HloGRUFwdInstruction::AllocatingIndices() const {
  return {0, 1, 2, 3};
}

std::unique_ptr<HloInstruction> HloGRUFwdInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateGRUFwd(shape, operands, is_training(), num_channels(),
                      partials_type(), reset_after());
}

std::unique_ptr<HloInstruction> CreateGRUFwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type,
    bool reset_after) {
  return absl::make_unique<HloGRUFwdInstruction>(
      shape, operands, is_training, num_channels, partials_type, reset_after);
}

HloGRUBwdInstruction::HloGRUBwdInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type,
    bool reset_after)
    : HloRNNBwdInstruction(PoplarOp::GRULayerBwd, shape, operands, is_training,
                           num_channels, partials_type, reset_after),
      HloGRUInstructionCommon(reset_after) {}

std::unique_ptr<HloInstruction> HloGRUBwdInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateGRUBwd(shape, operands, is_training(), num_channels(),
                      partials_type(), reset_after());
}

std::unique_ptr<HloInstruction> CreateGRUBwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type,
    bool reset_after) {
  return absl::make_unique<HloGRUBwdInstruction>(
      shape, operands, is_training, num_channels, partials_type, reset_after);
}

HloDynamicGRUFwdInstruction::HloDynamicGRUFwdInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type,
    bool reset_after)
    : HloRNNFwdInstruction(PoplarOp::DynamicGRULayerFwd, shape, operands,
                           is_training, num_channels, partials_type,
                           reset_after),
      HloGRUInstructionCommon(reset_after) {}

absl::flat_hash_set<int64> HloDynamicGRUFwdInstruction::AllocatingIndices()
    const {
  return {0, 1, 2, 3};
}

std::unique_ptr<HloInstruction>
HloDynamicGRUFwdInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateDynamicGRUFwd(shape, operands, is_training(), num_channels(),
                             partials_type(), reset_after());
}

std::unique_ptr<HloInstruction> CreateDynamicGRUFwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type,
    bool reset_after) {
  return absl::make_unique<HloDynamicGRUFwdInstruction>(
      shape, operands, is_training, num_channels, partials_type, reset_after);
}

HloDynamicGRUBwdInstruction::HloDynamicGRUBwdInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type,
    bool reset_after)
    : HloRNNBwdInstruction(PoplarOp::DynamicGRULayerBwd, shape, operands,
                           is_training, num_channels, partials_type,
                           reset_after),
      HloGRUInstructionCommon(reset_after) {}

std::unique_ptr<HloInstruction>
HloDynamicGRUBwdInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateDynamicGRUBwd(shape, operands, is_training(), num_channels(),
                             partials_type(), reset_after());
}

std::unique_ptr<HloInstruction> CreateDynamicGRUBwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type,
    bool reset_after) {
  return absl::make_unique<HloDynamicGRUBwdInstruction>(
      shape, operands, is_training, num_channels, partials_type, reset_after);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloGRUFwdFactoryFunc(
    HloCustomCallInstruction* call) {
  TF_ASSIGN_OR_RETURN(auto parsed_attributes,
                      rnn_helper::GRUAttributes::Parse(call));

  return CreateGRUFwd(
      call->shape(), call->operands(), parsed_attributes.is_training,
      parsed_attributes.num_channels, parsed_attributes.partials_xla_type,
      parsed_attributes.reset_after);
}

static HloPoplarInstructionFactory gru_fwd_factory(PoplarOp::GRULayerFwd,
                                                   HloGRUFwdFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloGRUBwdFactoryFunc(
    HloCustomCallInstruction* call) {
  TF_ASSIGN_OR_RETURN(auto parsed_attributes,
                      rnn_helper::GRUAttributes::Parse(call));

  return CreateGRUBwd(
      call->shape(), call->operands(), parsed_attributes.is_training,
      parsed_attributes.num_channels, parsed_attributes.partials_xla_type,
      parsed_attributes.reset_after);
}

static HloPoplarInstructionFactory gru_bwd_factory(PoplarOp::GRULayerBwd,
                                                   HloGRUBwdFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloDynamicGRUFwdFactoryFunc(
    HloCustomCallInstruction* call) {
  TF_ASSIGN_OR_RETURN(auto parsed_attributes,
                      rnn_helper::GRUAttributes::Parse(call));

  return CreateDynamicGRUFwd(
      call->shape(), call->operands(), parsed_attributes.is_training,
      parsed_attributes.num_channels, parsed_attributes.partials_xla_type,
      parsed_attributes.reset_after);
}

static HloPoplarInstructionFactory dynamic_gru_fwd_factory(
    PoplarOp::DynamicGRULayerFwd, HloDynamicGRUFwdFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloDynamicGRUBwdFactoryFunc(
    HloCustomCallInstruction* call) {
  TF_ASSIGN_OR_RETURN(auto parsed_attributes,
                      rnn_helper::GRUAttributes::Parse(call));

  return CreateDynamicGRUBwd(
      call->shape(), call->operands(), parsed_attributes.is_training,
      parsed_attributes.num_channels, parsed_attributes.partials_xla_type,
      parsed_attributes.reset_after);
}

static HloPoplarInstructionFactory dynamic_gru_bwd_factory(
    PoplarOp::DynamicGRULayerBwd, HloDynamicGRUBwdFactoryFunc);

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
