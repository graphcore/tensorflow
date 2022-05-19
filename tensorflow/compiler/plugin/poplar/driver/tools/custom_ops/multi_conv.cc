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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_conv.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hash.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace poplarplugin {

HloMultiConvInstruction::HloMultiConvInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const std::vector<ConvolutionSpec>& convolution_specs,
    const std::vector<OptionFlag>& option_flags, bool is_weight_update)
    : HloPoplarInstruction(shape, operands, PoplarOp::MultiConv,
                           convolution_specs, option_flags, is_weight_update),
      convolution_specs_(convolution_specs),
      option_flags_(option_flags),
      is_weight_update_(is_weight_update) {
  // Find indices for which we can allocate.
  for (int64_t i = 0; i != convolution_specs_.size(); ++i) {
    const ConvolutionSpec& convolution_spec = convolution_specs_[i];
    switch (convolution_spec.type) {
      case ConvType::Conv: {
        // The [0, n) inputs are the convolution inputs.
        allocating_indices_.insert(i);
        // The [n, 2n) inputs are the convolution kernels.
        allocating_indices_.insert(convolution_specs_.size() + i);
      }
      default: { break; }
    }
  }
}

absl::flat_hash_set<int64_t> HloMultiConvInstruction::AllocatingIndices()
    const {
  return allocating_indices_;
}

bool HloMultiConvInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64_t, int64_t>
HloMultiConvInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloMultiConvInstruction::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloMultiConvInstruction::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloMultiConvInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloMultiConvInstruction::AllowNonInplaceLowering() const { return false; }

bool HloMultiConvInstruction::IsPopOpsElementwise() const { return false; }

bool HloMultiConvInstruction::IsWeightUpdate() const {
  return is_weight_update_;
}

const std::vector<HloMultiConvInstruction::ConvolutionSpec>&
HloMultiConvInstruction::GetConvolutionSpecs() const {
  return convolution_specs_;
}

const std::vector<HloMultiConvInstruction::OptionFlag>&
HloMultiConvInstruction::GetOptionFlags() const {
  return option_flags_;
}

std::unique_ptr<HloInstruction>
HloMultiConvInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateMultiConv(shape, operands, convolution_specs_, option_flags_,
                         is_weight_update_);
}

std::vector<std::string>
HloMultiConvInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> output(convolution_specs_.size() + 1);
  output[0] = absl::StrCat("is_weight_update=", is_weight_update_);
  for (int64_t i = 0; i != convolution_specs_.size(); ++i) {
    const ConvolutionSpec& convolution_spec = convolution_specs_[i];
    output[i + 1] = absl::StrCat(
        "conv_", i, "={conv_type=", ConvType_Name(convolution_spec.type),
        ", window=", xla::window_util::ToString(convolution_spec.window),
        ", dims=",
        xla::ConvolutionDimensionNumbersToString(convolution_spec.dims),
        ", feature_group_count=", convolution_spec.feature_group_count,
        ", batch_group_count=", convolution_spec.batch_group_count, "}");
  }
  return output;
}

std::unique_ptr<HloInstruction> CreateMultiConv(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const std::vector<HloMultiConvInstruction::ConvolutionSpec>&
        convolution_specs,
    const std::vector<HloMultiConvInstruction::OptionFlag>& option_flags,
    bool is_weight_update) {
  return absl::make_unique<HloMultiConvInstruction>(
      shape, operands, convolution_specs, option_flags, is_weight_update);
}

}  // namespace poplarplugin
}  // namespace xla

// Provide hash functions for the ConvolutionSpec.
namespace std {
using ConvolutionSpec =
    xla::poplarplugin::HloMultiConvInstruction::ConvolutionSpec;
using OptionFlag = xla::poplarplugin::HloMultiConvInstruction::OptionFlag;

template <>
struct hash<ConvolutionSpec> {
  size_t operator()(const ConvolutionSpec& convolution_spec) const {
    return xla::poplarplugin::hash_util::hash(
        convolution_spec.type,
        xla::window_util::ToString(convolution_spec.window),
        xla::ConvolutionDimensionNumbersToString(convolution_spec.dims),
        convolution_spec.feature_group_count,
        convolution_spec.batch_group_count);
  }
};

template <>
struct hash<std::vector<ConvolutionSpec>> {
  size_t operator()(
      const std::vector<ConvolutionSpec>& convolution_specs) const {
    std::size_t hash = 7;
    for (const ConvolutionSpec& convolution_spec : convolution_specs) {
      hash = tensorflow::Hash64Combine(
          hash, std::hash<ConvolutionSpec>()(convolution_spec));
    }
    return hash;
  }
};

template <>
struct hash<OptionFlag> {
  size_t operator()(const OptionFlag& option_flag) const {
    return xla::poplarplugin::hash_util::hash(option_flag.key,
                                              option_flag.value);
  }
};

template <>
struct hash<std::vector<OptionFlag>> {
  size_t operator()(const std::vector<OptionFlag>& option_flags) const {
    std::size_t hash = 7;
    for (const OptionFlag& option_flag : option_flags) {
      hash =
          tensorflow::Hash64Combine(hash, std::hash<OptionFlag>()(option_flag));
    }
    return hash;
  }
};
}  // namespace std
