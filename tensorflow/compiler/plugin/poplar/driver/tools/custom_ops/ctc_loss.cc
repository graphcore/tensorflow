/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/ctc_loss.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/tf2xla/type_util.h"

namespace xla {
namespace poplarplugin {

HloCTCLossInstructionBase::HloCTCLossInstructionBase(
    PoplarOp op_type, const Shape& shape,
    absl::Span<HloInstruction* const> operands, PrimitiveType in_dtype,
    PrimitiveType out_dtype, int64 blank_index)
    : HloPoplarInstruction(shape, operands, op_type, in_dtype, out_dtype,
                           blank_index),
      in_dtype_(in_dtype),
      out_dtype_(out_dtype),
      blank_index_(blank_index) {}

HloCTCLossWithLogitsInstruction::HloCTCLossWithLogitsInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    PrimitiveType in_dtype, PrimitiveType out_dtype, int64 blank_index)
    : HloCTCLossInstructionBase(PoplarOp::CTCLossWithLogits, shape, operands,
                                in_dtype, out_dtype, blank_index) {}

HloCTCLossWithLogProbsInstruction::HloCTCLossWithLogProbsInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    PrimitiveType in_dtype, PrimitiveType out_dtype, int64 blank_index)
    : HloCTCLossInstructionBase(PoplarOp::CTCLossWithLogProbs, shape, operands,
                                in_dtype, out_dtype, blank_index) {}

PrimitiveType HloCTCLossInstructionBase::in_dtype() const { return in_dtype_; }
PrimitiveType HloCTCLossInstructionBase::out_dtype() const {
  return out_dtype_;
}
int64 HloCTCLossInstructionBase::blank_index() const { return blank_index_; }

absl::flat_hash_set<int64> HloCTCLossInstructionBase::AllocatingIndices()
    const {
  return {0, 1};
}

bool HloCTCLossInstructionBase::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64>
HloCTCLossInstructionBase::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloCTCLossInstructionBase::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloCTCLossInstructionBase::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

bool HloCTCLossInstructionBase::IsPopOpsElementwise() const { return false; }

std::vector<std::string>
HloCTCLossInstructionBase::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back(absl::StrCat("in_dtype=", in_dtype_));
  attributes.push_back(absl::StrCat("out_dtype=", out_dtype_));
  attributes.push_back(absl::StrCat("blank_index=", blank_index_));

  return attributes;
}

std::unique_ptr<HloInstruction>
HloCTCLossWithLogitsInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateCTCLossWithLogits(shape, operands, in_dtype(), out_dtype(),
                                 blank_index());
}

std::unique_ptr<HloInstruction>
HloCTCLossWithLogProbsInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateCTCLossWithLogProbs(shape, operands, in_dtype(), out_dtype(),
                                   blank_index());
}

std::unique_ptr<HloInstruction> CreateCTCLossWithLogits(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    PrimitiveType in_dtype, PrimitiveType out_dtype, int64 blank_index) {
  return absl::make_unique<HloCTCLossWithLogitsInstruction>(
      shape, operands, in_dtype, out_dtype, blank_index);
}

std::unique_ptr<HloInstruction> CreateCTCLossWithLogProbs(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    PrimitiveType in_dtype, PrimitiveType out_dtype, int64 blank_index) {
  return absl::make_unique<HloCTCLossWithLogProbsInstruction>(
      shape, operands, in_dtype, out_dtype, blank_index);
}

namespace {

static HloPoplarInstructionFactory ctc_loss_with_logits_factory(
    PoplarOp::CTCLossWithLogits,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
      TF_ASSIGN_OR_RETURN(tensorflow::DataType in_dtype,
                          attribute_map.GetAttributeAsTFDataType("in_dtype"));
      TF_ASSIGN_OR_RETURN(tensorflow::DataType out_dtype,
                          attribute_map.GetAttributeAsTFDataType("out_dtype"));
      TF_ASSIGN_OR_RETURN(int64 blank_index,
                          attribute_map.GetAttributeAsInt("blank_index"));

      PrimitiveType in_dtype_xla;
      PrimitiveType out_dtype_xla;
      TF_CHECK_OK(DataTypeToPrimitiveType(in_dtype, &in_dtype_xla));
      TF_CHECK_OK(DataTypeToPrimitiveType(out_dtype, &out_dtype_xla));

      return CreateCTCLossWithLogits(call->shape(), call->operands(),
                                     in_dtype_xla, out_dtype_xla, blank_index);
    });

static HloPoplarInstructionFactory ctc_loss_with_log_probs_factory(
    PoplarOp::CTCLossWithLogProbs,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
      TF_ASSIGN_OR_RETURN(tensorflow::DataType in_dtype,
                          attribute_map.GetAttributeAsTFDataType("in_dtype"));
      TF_ASSIGN_OR_RETURN(tensorflow::DataType out_dtype,
                          attribute_map.GetAttributeAsTFDataType("out_dtype"));
      TF_ASSIGN_OR_RETURN(int64 blank_index,
                          attribute_map.GetAttributeAsInt("blank_index"));

      PrimitiveType in_dtype_xla;
      PrimitiveType out_dtype_xla;
      TF_CHECK_OK(DataTypeToPrimitiveType(in_dtype, &in_dtype_xla));
      TF_CHECK_OK(DataTypeToPrimitiveType(out_dtype, &out_dtype_xla));

      return CreateCTCLossWithLogProbs(call->shape(), call->operands(),
                                       in_dtype_xla, out_dtype_xla,
                                       blank_index);
    });

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
