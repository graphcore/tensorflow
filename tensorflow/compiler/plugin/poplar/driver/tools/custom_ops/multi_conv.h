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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_MULTI_CONV_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_MULTI_CONV_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloMultiConvInstruction : public HloPoplarInstruction {
 public:
  struct ConvolutionSpec {
    ConvType type;
    Window window;
    ConvolutionDimensionNumbers dims;
    int64 feature_group_count;
    int64 batch_group_count;
  };
  struct OptionFlag {
    std::string key;
    std::string value;
  };

  HloMultiConvInstruction(const Shape& shape,
                          absl::Span<HloInstruction* const> operands,
                          const std::vector<ConvolutionSpec>& convolution_specs,
                          const std::vector<OptionFlag>& option_flags,
                          bool is_weight_update);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  bool AllocatingOutput() const override;

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

  const std::vector<ConvolutionSpec>& GetConvolutionSpecs() const;

  const std::vector<OptionFlag>& GetOptionFlags() const;

  bool IsWeightUpdate() const;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext*) const override;

  absl::flat_hash_set<int64> allocating_indices_;
  const std::vector<ConvolutionSpec> convolution_specs_;
  const std::vector<OptionFlag> option_flags_;
  const bool is_weight_update_;
};

std::unique_ptr<HloInstruction> CreateMultiConv(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const std::vector<HloMultiConvInstruction::ConvolutionSpec>&
        convolution_specs,
    const std::vector<HloMultiConvInstruction::OptionFlag>& option_flags,
    bool is_weight_update);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_MULTI_CONV_H_
