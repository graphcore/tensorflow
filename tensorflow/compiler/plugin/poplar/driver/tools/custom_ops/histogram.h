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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_HISTOGRAM_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_HISTOGRAM_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

/*
 * Histogram.
 */
class HloHistogramInstruction : public HloPoplarInstruction {
 public:
  HloHistogramInstruction(const Shape& shape, HloInstruction* const input,
                          HloInstruction* const levels, bool absolute_of_input);

  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;
  bool AllocatingOutput() const override;
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;
  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

  bool AbsoluteOfInput() const;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

  const bool absolute_of_input_;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

/*
 * Histogram update.
 */
class HloHistogramUpdateInstruction : public HloPoplarInstruction {
 public:
  HloHistogramUpdateInstruction(const Shape& shape, HloInstruction* const hist,
                                HloInstruction* const input,
                                HloInstruction* const levels,
                                bool absolute_of_input);

  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;
  bool AllocatingOutput() const override;
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;
  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

  bool AbsoluteOfInput() const;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

  const bool absolute_of_input_;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_HISTOGRAM_H_
