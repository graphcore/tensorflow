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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_STATELESS_RANDOM_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_STATELESS_RANDOM_H_

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloStatelessRandom : public HloPoplarInstruction {
 public:
  explicit HloStatelessRandom(const Shape& shape,
                              absl::Span<HloInstruction* const> operands,
                              PoplarOp op);

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
      HloCloneContext*) const = 0;
};

class HloStatelessRandomUniform : public HloStatelessRandom {
 public:
  explicit HloStatelessRandomUniform(const Shape& shape,
                                     absl::Span<HloInstruction* const> operands,
                                     float min_val, float max_val);

  float GetMin() const { return min_val_; }

  float GetMax() const { return max_val_; }

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext*) const override {
    return absl::make_unique<HloStatelessRandomUniform>(shape, operands,
                                                        min_val_, max_val_);
  }
  // The range of values to generate the random numbers within.
  float min_val_;
  float max_val_;
};

std::unique_ptr<HloInstruction> CreateStatelessRandomUniform(
    const Shape& shape);

class HloStatelessRandomUniformInt : public HloStatelessRandom {
 public:
  explicit HloStatelessRandomUniformInt(
      const Shape& shape, absl::Span<HloInstruction* const> operands);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext*) const override {
    return absl::make_unique<HloStatelessRandomUniformInt>(shape, operands);
  }
};

std::unique_ptr<HloInstruction> CreateStatelessRandomUniformInt(
    const Shape& shape);

class HloStatelessRandomNormal : public HloStatelessRandom {
 public:
  explicit HloStatelessRandomNormal(const Shape& shape,
                                    absl::Span<HloInstruction* const> operands);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext*) const override {
    return absl::make_unique<HloStatelessRandomNormal>(shape, operands);
  }
};

std::unique_ptr<HloInstruction> CreateStatelessRandomNormal(const Shape& shape);

class HloStatelessTruncatedNormal : public HloStatelessRandom {
 public:
  explicit HloStatelessTruncatedNormal(
      const Shape& shape, absl::Span<HloInstruction* const> operands);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext*) const override {
    return absl::make_unique<HloStatelessTruncatedNormal>(shape, operands);
  }
};

std::unique_ptr<HloInstruction> CreateStatelessTruncatedNormal(
    const Shape& shape);

}  // namespace poplarplugin
}  // namespace xla

#endif
