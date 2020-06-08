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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/lstm.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/rnn.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/tf2xla/type_util.h"

namespace xla {
namespace poplarplugin {

HloLSTMFwdInstruction::HloLSTMFwdInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type)
    : HloRNNFwdInstruction(PoplarOp::LstmLayerFwd, shape, operands, is_training,
                           num_channels, partials_type) {}

absl::flat_hash_set<int64> HloLSTMFwdInstruction::AllocatingIndices() const {
  return {0, 1, 2, 3, 4};
}

std::unique_ptr<HloInstruction> HloLSTMFwdInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateLSTMFwd(shape, operands, is_training(), num_channels(),
                       partials_type());
}

std::unique_ptr<HloInstruction> CreateLSTMFwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type) {
  return absl::make_unique<HloLSTMFwdInstruction>(shape, operands, is_training,
                                                  num_channels, partials_type);
}

HloLSTMBwdInstruction::HloLSTMBwdInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type)
    : HloRNNBwdInstruction(PoplarOp::LstmLayerBwd, shape, operands, is_training,
                           num_channels, partials_type) {}

std::unique_ptr<HloInstruction> HloLSTMBwdInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateLSTMBwd(shape, operands, is_training(), num_channels(),
                       partials_type());
}

std::unique_ptr<HloInstruction> CreateLSTMBwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type) {
  return absl::make_unique<HloLSTMBwdInstruction>(shape, operands, is_training,
                                                  num_channels, partials_type);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloLSTMFwdFactoryFunc(
    HloCustomCallInstruction* call) {
  TF_ASSIGN_OR_RETURN(auto parsed_attributes,
                      rnn_helper::RNNAttributes::Parse(call));

  return CreateLSTMFwd(
      call->shape(), call->operands(), parsed_attributes.is_training,
      parsed_attributes.num_channels, parsed_attributes.partials_xla_type);
}

static HloPoplarInstructionFactory lstm_fwd_factory(PoplarOp::LstmLayerFwd,
                                                    HloLSTMFwdFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloLSTMBwdFactoryFunc(
    HloCustomCallInstruction* call) {
  TF_ASSIGN_OR_RETURN(auto parsed_attributes,
                      rnn_helper::RNNAttributes::Parse(call));

  return CreateLSTMBwd(
      call->shape(), call->operands(), parsed_attributes.is_training,
      parsed_attributes.num_channels, parsed_attributes.partials_xla_type);
}

static HloPoplarInstructionFactory lstm_bwd_factory(PoplarOp::LstmLayerBwd,
                                                    HloLSTMBwdFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
