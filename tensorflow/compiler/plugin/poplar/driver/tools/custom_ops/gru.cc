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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/gru.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/rnn.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"
#include "tensorflow/compiler/tf2xla/type_util.h"

namespace xla {
namespace poplarplugin {

HloGRUFwdInstruction::HloGRUFwdInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type)
    : HloRNNFwdInstruction(PoplibsOp::GRULayerFwd, shape, operands, is_training,
                           num_channels, partials_type) {}

absl::flat_hash_set<int64> HloGRUFwdInstruction::AllocatingIndices() const {
  return {0, 1, 2, 3};
}

std::unique_ptr<HloInstruction> HloGRUFwdInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateGRUFwd(shape, operands, is_training(), num_channels(),
                      partials_type());
}

std::unique_ptr<HloInstruction> CreateGRUFwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type) {
  return absl::make_unique<HloGRUFwdInstruction>(shape, operands, is_training,
                                                 num_channels, partials_type);
}

HloGRUBwdInstruction::HloGRUBwdInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type)
    : HloRNNBwdInstruction(PoplibsOp::GRULayerBwd, shape, operands, is_training,
                           num_channels, partials_type) {}

std::unique_ptr<HloInstruction> HloGRUBwdInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateGRUBwd(shape, operands, is_training(), num_channels(),
                      partials_type());
}

std::unique_ptr<HloInstruction> CreateGRUBwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type) {
  return absl::make_unique<HloGRUBwdInstruction>(shape, operands, is_training,
                                                 num_channels, partials_type);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloGRUFwdFactoryFunc(
    HloCustomCallInstruction* call) {
  TF_ASSIGN_OR_RETURN(auto parsed_attributes,
                      rnn_helper::RNNAttributes::Parse(call));

  return CreateGRUFwd(
      call->shape(), call->operands(), parsed_attributes.is_training,
      parsed_attributes.num_channels, parsed_attributes.partials_xla_type);
}

static HloPoplarInstructionFactory gru_fwd_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::GRULayerFwd),
    HloGRUFwdFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloGRUBwdFactoryFunc(
    HloCustomCallInstruction* call) {
  TF_ASSIGN_OR_RETURN(auto parsed_attributes,
                      rnn_helper::RNNAttributes::Parse(call));

  return CreateGRUBwd(
      call->shape(), call->operands(), parsed_attributes.is_training,
      parsed_attributes.num_channels, parsed_attributes.partials_xla_type);
}

static HloPoplarInstructionFactory gru_bwd_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::GRULayerBwd),
    HloGRUBwdFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
