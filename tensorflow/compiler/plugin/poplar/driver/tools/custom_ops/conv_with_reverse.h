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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_CONV_WITH_REVERSE_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_CONV_WITH_REVERSE_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

// Custom instruction for invoking the ConvWithReverse PoplarOp.
class HloConvWithReverse : public HloPoplarInstruction {
 public:
  explicit HloConvWithReverse(
      const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
      int64 feature_group_count, int64 batch_group_count, const Window& window,
      const ConvolutionDimensionNumbers& dimension_numbers,
      const PrecisionConfig& precision_config);

  // Convolution options.
  //
  // These are named slightly differently to avoid shadowing
  // HloInstruction::precision_config, which can only be called for
  // convolve/dot instructions.
  const PrecisionConfig& GetPrecisionConfig() const;

  const Window& window() const override;

  // HloPoplarInstruction virtuals.
  absl::flat_hash_set<int64> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

  Window window_;
  PrecisionConfig precision_config_;
};

std::unique_ptr<HloConvWithReverse> CreateConvWithReverse(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    int64 feature_group_count, int64 batch_group_count, const Window& window,
    const ConvolutionDimensionNumbers& dimension_numbers,
    const PrecisionConfig& precision_config);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_CONV_WITH_REVERSE_H_
