/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_FP8_OPS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_FP8_OPS_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace poplarplugin {

class HloF8MatMulInstruction : public HloPoplarInstruction {
 public:
  explicit HloF8MatMulInstruction(const Shape& shape, HloInstruction* lhs,
                                  HloInstruction* lhs_metadata,
                                  HloInstruction* rhs,
                                  HloInstruction* rhs_metadata);

  absl::flat_hash_set<int64_t> AllocatingIndices() const override {
    return {0, 2};
  }
  bool AllocatingOutput() const override { return true; }
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override {
    return {};
  }
  HloPoplarUseDescriptions GetUseDescriptions() const override {
    return UseDescriptionsNoInputOutputAlias();
  }
  HloPoplarBufferDescriptions GetBufferDescriptions() const override {
    return BufferDescriptionsAllocatesAllOutputs(this);
  }
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override {
    return FindConsumersExtensionResults::DoNotFindConsumers();
  }

  bool AllowNonInplaceLowering() const override { return true; }
  bool IsPopOpsElementwise() const override { return false; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override {
    return {};
  }

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext* context) const override;
};

template <uint16_t D>
class HloF8ConvInstruction : public HloPoplarInstruction {
 public:
  explicit HloF8ConvInstruction(
      const Shape& shape, HloInstruction* input, HloInstruction* filter,
      HloInstruction* input_meta, HloInstruction* filter_meta,
      const std::vector<int32>& strides, const std::string& padding,
      const std::vector<int32>& explicit_paddings,
      const std::string& data_format, const std::vector<int32>& dilations);

  const Window& window() const override { return window_; }

  absl::flat_hash_set<int64_t> AllocatingIndices() const override {
    return {0, 1};
  }

  bool AllocatingOutput() const override { return true; }

  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override {
    return {};
  }

  HloPoplarUseDescriptions GetUseDescriptions() const override {
    return UseDescriptionsNoInputOutputAlias();
  }

  HloPoplarBufferDescriptions GetBufferDescriptions() const override {
    return BufferDescriptionsAllocatesAllOutputs(this);
  }

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override {
    return FindConsumersExtensionResults::DoNotFindConsumers();
  }

  bool AllowNonInplaceLowering() const override { return true; }

  bool IsPopOpsElementwise() const override { return false; }

  const std::vector<int32>& Strides() const { return strides; }

  const std::string& Padding() const { return padding; }

  const std::vector<int32> ExplicitPaddings() const {
    return explicit_paddings;
  }

  const std::string& DataFormat() const { return data_format; }

  const std::vector<int32>& Dilations() const { return dilations; }

 protected:
  Window window_;

  const std::vector<int32> strides;
  const std::string padding;
  const std::vector<int32> explicit_paddings;
  const std::string data_format;
  const std::vector<int32> dilations;

  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override {
    return {};
  }

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext* context) const override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_FP8_OPS_H_
