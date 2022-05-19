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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_CTC_LOSS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_CTC_LOSS_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloCTCInferenceAndLossBase : public HloPoplarInstruction {
 protected:
  explicit HloCTCInferenceAndLossBase(
      PoplarOp op_type, const Shape& shape,
      absl::Span<HloInstruction* const> operands, PrimitiveType in_dtype,
      PrimitiveType out_dtype, int64_t blank_index);

 public:
  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;
  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

  PrimitiveType in_dtype() const;
  int64_t blank_index() const;

 private:
  PrimitiveType in_dtype_;
  int64_t blank_index_;
};

class HloCTCLossInstructionBase : public HloCTCInferenceAndLossBase {
 protected:
  explicit HloCTCLossInstructionBase(PoplarOp op_type, const Shape& shape,
                                     absl::Span<HloInstruction* const> operands,
                                     PrimitiveType in_dtype,
                                     PrimitiveType out_dtype,
                                     int64_t blank_index);

 public:
  PrimitiveType out_dtype() const;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  PrimitiveType out_dtype_;
};

class HloCTCInferenceInstructionBase : public HloCTCInferenceAndLossBase {
 protected:
  explicit HloCTCInferenceInstructionBase(
      PoplarOp op_type, const Shape& shape,
      absl::Span<HloInstruction* const> operands, PrimitiveType in_dtype,
      int64_t beam_width, int64_t blank_index, int64_t top_paths);
  absl::flat_hash_set<int64_t> AllocatingIndices() const override;

 public:
  int64_t beam_width() const;
  int64_t top_paths() const;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  int64_t beam_width_;
  int64_t top_paths_;
};

class HloCTCLossWithLogitsInstruction : public HloCTCLossInstructionBase {
 public:
  explicit HloCTCLossWithLogitsInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      PrimitiveType in_dtype, xla::PrimitiveType out_dtype,
      int64_t blank_index);

  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

class HloCTCLossWithLogProbsInstruction : public HloCTCLossInstructionBase {
 public:
  explicit HloCTCLossWithLogProbsInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      PrimitiveType in_dtype, PrimitiveType out_dtype, int64_t blank_index);

  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

class HloCTCBeamSearchDecoderWithLogits
    : public HloCTCInferenceInstructionBase {
 public:
  explicit HloCTCBeamSearchDecoderWithLogits(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      PrimitiveType in_dtype, int64_t beam_width, int64_t blank_index,
      int64_t top_paths);

  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

class HloCTCBeamSearchDecoderWithLogProbs
    : public HloCTCInferenceInstructionBase {
 public:
  explicit HloCTCBeamSearchDecoderWithLogProbs(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      PrimitiveType in_dtype, int64_t beam_width, int64_t blank_index,
      int64_t top_paths);

  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_CTC_LOSS_H_
