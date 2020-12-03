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
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloDropoutBase : public HloPoplarInstruction {
 protected:
  HloDropoutBase(HloInstruction* operand, HloInstruction* seed,
                 PoplarOp dropout_type, float rate, float scale,
                 const std::vector<int64>& noise_shape);

 public:
  absl::flat_hash_set<int64> AllocatingIndices() const override { return {}; }
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override {
    return {};
  }
  uint64 NumberOfInplaceOperands() const override { return 0; }
  bool IsPopOpsElementwise() const override { return false; }

  // Probability of a given element being set to zero.
  float Rate() const { return rate; }

  // Scale to apply to all elements that aren't dropped out.
  float Scale() const { return scale; }

  // For shaped dropout.
  const std::vector<int64>& NoiseShape() const { return noise_shape; }

  // If noise_shape is not provided then the default ia an empty list.
  bool HasNoiseShape() const { return noise_shape.size(); }

 private:
  const float scale;
  const float rate;
  const std::vector<int64> noise_shape;
};

class HloDropout : public HloDropoutBase {
 public:
  HloDropout(HloInstruction* operand, HloInstruction* seed, float rate,
             float scale, const std::vector<int64>& noise_shape,
             bool can_create_reference_tensor);

  // Indicates whether this dropout operation can be converted to a
  // HloDropoutWithReference operation by the DropoutWithReferenceFinder pass.
  bool CanCreateReferenceTensor() const { return can_create_reference_tensor_; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

  const bool can_create_reference_tensor_;
};

std::unique_ptr<HloInstruction> CreateDropout(
    HloInstruction* operand, HloInstruction* seed, float rate, float scale,
    const std::vector<int64>& noise_shape, bool can_create_reference_tensor);

class HloDropoutWithReference : public HloDropoutBase {
 public:
  HloDropoutWithReference(HloInstruction* operand, HloInstruction* seed,
                          float rate, float scale,
                          const std::vector<int64>& noise_shape,
                          const std::string& reference_key);

  const std::string& ReferenceKey() const { return reference_key_; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

  const std::string reference_key_;
};

std::unique_ptr<HloInstruction> CreateDropoutWithReference(
    HloInstruction* operand, HloInstruction* seed, float rate, float scale,
    const std::vector<int64>& noise_shape, const std::string& reference_key);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_DROPOUT_H
