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

HloCTCInferenceAndLossBase::HloCTCInferenceAndLossBase(
    PoplarOp op_type, const Shape& shape,
    absl::Span<HloInstruction* const> operands, PrimitiveType in_dtype,
    PrimitiveType out_dtype, int64 blank_index)
    : HloPoplarInstruction(shape, operands, op_type, in_dtype, out_dtype,
                           blank_index),
      in_dtype_(in_dtype),
      blank_index_(blank_index) {}

HloCTCLossInstructionBase::HloCTCLossInstructionBase(
    PoplarOp op_type, const Shape& shape,
    absl::Span<HloInstruction* const> operands, PrimitiveType in_dtype,
    PrimitiveType out_dtype, int64 blank_index)
    : HloCTCInferenceAndLossBase(op_type, shape, operands, in_dtype, out_dtype,
                                 blank_index),
      out_dtype_(out_dtype) {}

HloCTCLossWithLogProbsInstruction::HloCTCLossWithLogProbsInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    PrimitiveType in_dtype, PrimitiveType out_dtype, int64 blank_index)
    : HloCTCLossInstructionBase(PoplarOp::CTCLossWithLogits, shape, operands,
                                in_dtype, out_dtype, blank_index) {}

HloCTCLossWithLogitsInstruction::HloCTCLossWithLogitsInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    PrimitiveType in_dtype, PrimitiveType out_dtype, int64 blank_index)
    : HloCTCLossInstructionBase(PoplarOp::CTCLossWithLogits, shape, operands,
                                in_dtype, out_dtype, blank_index) {}

HloCTCInferenceInstructionBase::HloCTCInferenceInstructionBase(
    PoplarOp op_type, const Shape& shape,
    absl::Span<HloInstruction* const> operands, PrimitiveType in_dtype,
    int64 beam_width, int64 blank_index, int64 top_paths)
    : HloCTCInferenceAndLossBase(op_type, shape, operands, in_dtype, in_dtype,
                                 blank_index),
      beam_width_(beam_width),
      top_paths_(top_paths) {}

HloCTCBeamSearchDecoderWithLogProbs::HloCTCBeamSearchDecoderWithLogProbs(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    PrimitiveType in_dtype, int64 beam_width, int64 blank_index,
    int64 top_paths)
    : HloCTCInferenceInstructionBase(PoplarOp::CTCBeamSearchWithLogProbs, shape,
                                     operands, in_dtype, beam_width,
                                     blank_index, top_paths) {}

HloCTCBeamSearchDecoderWithLogits::HloCTCBeamSearchDecoderWithLogits(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    PrimitiveType in_dtype, int64 beam_width, int64 blank_index,
    int64 top_paths)
    : HloCTCInferenceInstructionBase(PoplarOp::CTCBeamSearchWithLogits, shape,
                                     operands, in_dtype, beam_width,
                                     blank_index, top_paths) {}

PrimitiveType HloCTCInferenceAndLossBase::in_dtype() const { return in_dtype_; }
PrimitiveType HloCTCLossInstructionBase::out_dtype() const {
  return out_dtype_;
}
int64 HloCTCInferenceAndLossBase::blank_index() const { return blank_index_; }

int64 HloCTCInferenceInstructionBase::beam_width() const { return beam_width_; }
int64 HloCTCInferenceInstructionBase::top_paths() const { return top_paths_; }

absl::flat_hash_set<int64> HloCTCInferenceAndLossBase::AllocatingIndices()
    const {
  return {0, 1};
}

absl::flat_hash_set<int64> HloCTCInferenceInstructionBase::AllocatingIndices()
    const {
  return {0};
}

bool HloCTCInferenceAndLossBase::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64>
HloCTCInferenceAndLossBase::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloCTCInferenceAndLossBase::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloCTCInferenceAndLossBase::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloCTCInferenceAndLossBase::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloCTCInferenceAndLossBase::IsPopOpsElementwise() const { return false; }

std::vector<std::string>
HloCTCLossInstructionBase::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.emplace_back(absl::StrCat("in_dtype=", in_dtype()));
  attributes.emplace_back(absl::StrCat("out_dtype=", out_dtype()));
  attributes.emplace_back(absl::StrCat("blank_index=", blank_index()));

  return attributes;
}

std::vector<std::string>
HloCTCInferenceInstructionBase::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.emplace_back(absl::StrCat("in_dtype=", in_dtype()));
  attributes.emplace_back(absl::StrCat("beam_width=", beam_width()));
  attributes.emplace_back(absl::StrCat("blank_index=", blank_index()));
  attributes.emplace_back(absl::StrCat("top_paths=", top_paths()));

  return attributes;
}

static std::unique_ptr<HloInstruction> CreateCTCLossWithLogits(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    PrimitiveType in_dtype, PrimitiveType out_dtype, int64 blank_index) {
  return absl::make_unique<HloCTCLossWithLogitsInstruction>(
      shape, operands, in_dtype, out_dtype, blank_index);
}

static std::unique_ptr<HloInstruction> CreateCTCLossWithLogProbs(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    PrimitiveType in_dtype, PrimitiveType out_dtype, int64 blank_index) {
  return absl::make_unique<HloCTCLossWithLogProbsInstruction>(
      shape, operands, in_dtype, out_dtype, blank_index);
}

static std::unique_ptr<HloInstruction> CreateCTCBeamSearchWithLogits(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    PrimitiveType in_dtype, int64 beam_width, int64 blank_index,
    int64 top_paths) {
  return absl::make_unique<HloCTCBeamSearchDecoderWithLogits>(
      shape, operands, in_dtype, beam_width, blank_index, top_paths);
}

static std::unique_ptr<HloInstruction> CreateCTCBeamSearchWithLogProbs(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    PrimitiveType in_dtype, int64 beam_width, int64 blank_index,
    int64 top_paths) {
  return absl::make_unique<HloCTCBeamSearchDecoderWithLogProbs>(
      shape, operands, in_dtype, beam_width, blank_index, top_paths);
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

std::unique_ptr<HloInstruction>
HloCTCBeamSearchDecoderWithLogits::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateCTCBeamSearchWithLogits(
      shape, operands, in_dtype(), beam_width(), blank_index(), top_paths());
}

std::unique_ptr<HloInstruction>
HloCTCBeamSearchDecoderWithLogProbs::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateCTCBeamSearchWithLogProbs(
      shape, operands, in_dtype(), beam_width(), blank_index(), top_paths());
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

static HloPoplarInstructionFactory ctc_beam_search_with_logits_factory(
    PoplarOp::CTCBeamSearchWithLogits,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
      TF_ASSIGN_OR_RETURN(tensorflow::DataType in_dtype,
                          attribute_map.GetAttributeAsTFDataType("in_dtype"));
      TF_ASSIGN_OR_RETURN(int64 beam_width,
                          attribute_map.GetAttributeAsInt("beam_width"));
      TF_ASSIGN_OR_RETURN(int64 blank_index,
                          attribute_map.GetAttributeAsInt("blank_index"));
      TF_ASSIGN_OR_RETURN(int64 top_paths,
                          attribute_map.GetAttributeAsInt("top_paths"));

      PrimitiveType in_dtype_xla;
      TF_CHECK_OK(DataTypeToPrimitiveType(in_dtype, &in_dtype_xla));

      return CreateCTCBeamSearchWithLogits(call->shape(), call->operands(),
                                           in_dtype_xla, beam_width,
                                           blank_index, top_paths);
    });

static HloPoplarInstructionFactory ctc_beam_search_with_log_probs_factory(
    PoplarOp::CTCBeamSearchWithLogProbs,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
      TF_ASSIGN_OR_RETURN(tensorflow::DataType in_dtype,
                          attribute_map.GetAttributeAsTFDataType("in_dtype"));
      TF_ASSIGN_OR_RETURN(int64 beam_width,
                          attribute_map.GetAttributeAsInt("beam_width"));
      TF_ASSIGN_OR_RETURN(int64 blank_index,
                          attribute_map.GetAttributeAsInt("blank_index"));
      TF_ASSIGN_OR_RETURN(int64 top_paths,
                          attribute_map.GetAttributeAsInt("top_paths"));

      PrimitiveType in_dtype_xla;
      TF_CHECK_OK(DataTypeToPrimitiveType(in_dtype, &in_dtype_xla));

      return CreateCTCBeamSearchWithLogProbs(call->shape(), call->operands(),
                                             in_dtype_xla, beam_width,
                                             blank_index, top_paths);
    });

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
