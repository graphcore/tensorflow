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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_DROPOUT_H
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_DROPOUT_H

#include <memory>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloDropout : public HloPoplarInstruction {
 public:
  HloDropout(HloInstruction* operand, HloInstruction* seed, float rate,
             float scale, const std::vector<int64_t>& noise_shape);

  HloDropout(HloInstruction* operand, HloInstruction* seed,
             HloInstruction* reference, float rate, float scale,
             const std::vector<int64_t>& noise_shape);

  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;
  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

  // Probability of a given element being set to zero.
  float Rate() const { return rate; }

  // Scale to apply to all elements that aren't dropped out.
  float Scale() const { return scale; }

  // For shaped dropout.
  const std::vector<int64_t>& NoiseShape() const { return noise_shape; }

  // If noise_shape is not provided then the default ia an empty list.
  bool HasNoiseShape() const { return noise_shape.size(); }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

  const float scale;
  const float rate;
  const std::vector<int64_t> noise_shape;
};

std::unique_ptr<HloInstruction> CreateDropout(
    HloInstruction* operand, HloInstruction* seed, float rate, float scale,
    const std::vector<int64_t>& noise_shape);

std::unique_ptr<HloInstruction> CreateDropout(
    HloInstruction* operand, HloInstruction* seed, HloInstruction* reference,
    float rate, float scale, const std::vector<int64_t>& noise_shape);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_DROPOUT_H
