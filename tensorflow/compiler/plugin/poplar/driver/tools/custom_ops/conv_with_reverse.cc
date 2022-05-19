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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/conv_with_reverse.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace poplarplugin {

HloConvWithReverse::HloConvWithReverse(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    int64_t feature_group_count, int64_t batch_group_count,
    const Window& window, const ConvolutionDimensionNumbers& dimension_numbers,
    const PrecisionConfig& precision_config)
    : HloPoplarInstruction(shape, {lhs, rhs}, PoplarOp::ConvWithReverse, window,
                           precision_config, dimension_numbers,
                           feature_group_count, batch_group_count),
      window_(window),
      precision_config_(precision_config) {
  set_convolution_dimension_numbers(dimension_numbers);
  set_feature_group_count(feature_group_count);
  set_batch_group_count(batch_group_count);
}

const Window& HloConvWithReverse::window() const { return window_; }

const PrecisionConfig& HloConvWithReverse::GetPrecisionConfig() const {
  return precision_config_;
}

absl::flat_hash_set<int64_t> HloConvWithReverse::AllocatingIndices() const {
  return {};
}

bool HloConvWithReverse::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64_t, int64_t> HloConvWithReverse::LayoutDependencies()
    const {
  return {};
}

HloPoplarUseDescriptions HloConvWithReverse::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloConvWithReverse::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloConvWithReverse::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloConvWithReverse::AllowNonInplaceLowering() const { return false; }
bool HloConvWithReverse::IsPopOpsElementwise() const { return false; }

std::vector<std::string> HloConvWithReverse::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("batch_group_count=" +
                       std::to_string(batch_group_count()));
  attributes.push_back("feature_group_count=" +
                       std::to_string(feature_group_count()));
  attributes.push_back("dim_labels=" + xla::ConvolutionDimensionNumbersToString(
                                           convolution_dimension_numbers()));
  attributes.push_back("window=" + xla::window_util::ToString(window_));
  attributes.push_back(PrecisionConfigToString(precision_config_));

  return attributes;
}

std::unique_ptr<HloInstruction> HloConvWithReverse::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateConvWithReverse(shape, operands[0], operands[1],
                               feature_group_count(), batch_group_count(),
                               window_, convolution_dimension_numbers(),
                               precision_config_);
}

std::unique_ptr<HloConvWithReverse> CreateConvWithReverse(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    int64_t feature_group_count, int64_t batch_group_count,
    const Window& window, const ConvolutionDimensionNumbers& dimension_numbers,
    const PrecisionConfig& precision_config) {
  return absl::make_unique<HloConvWithReverse>(
      shape, lhs, rhs, feature_group_count, batch_group_count, window,
      dimension_numbers, precision_config);
}

namespace {

static HloPoplarInstructionFactory conv_with_reverse_factory(
    PoplarOp::ConvWithReverse,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloConvWithReverse>> {
      // FIXME: HloCustomCallInstruction can't be serialised with a
      // operand_precision attribute, so we use the default precision for now.
      PrecisionConfig precision_config;
      precision_config.mutable_operand_precision()->Resize(
          call->operand_count(), PrecisionConfig::DEFAULT);

      return CreateConvWithReverse(
          call->shape(), call->mutable_operand(0), call->mutable_operand(1),
          call->feature_group_count(), call->batch_group_count(),
          call->window(), call->convolution_dimension_numbers(),
          precision_config);
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
