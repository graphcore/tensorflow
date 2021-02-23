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

class HloCTCLossInstructionBase : public HloPoplarInstruction {
 protected:
  explicit HloCTCLossInstructionBase(PoplarOp op_type, const Shape& shape,
                                     absl::Span<HloInstruction* const> operands,
                                     xla::PrimitiveType in_dtype,
                                     xla::PrimitiveType out_dtype,
                                     int64 blank_index);

 public:
  absl::flat_hash_set<int64> AllocatingIndices() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;
  bool IsPopOpsElementwise() const override;

  xla::PrimitiveType in_dtype() const;
  xla::PrimitiveType out_dtype() const;
  int64 blank_index() const;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  xla::PrimitiveType in_dtype_;
  xla::PrimitiveType out_dtype_;
  int64 blank_index_;
};

class HloCTCLossInstruction : public HloCTCLossInstructionBase {
 public:
  explicit HloCTCLossInstruction(const Shape& shape,
                                 absl::Span<HloInstruction* const> operands,
                                 xla::PrimitiveType in_dtype,
                                 xla::PrimitiveType out_dtype,
                                 int64 blank_index);

  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

class HloCTCLossWithLogitsInstruction : public HloCTCLossInstructionBase {
 public:
  explicit HloCTCLossWithLogitsInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      xla::PrimitiveType in_dtype, xla::PrimitiveType out_dtype,
      int64 blank_index);

  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

std::unique_ptr<HloInstruction> CreateCTCLoss(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    xla::PrimitiveType in_dtype, xla::PrimitiveType out_dtype,
    int64 blank_index);

std::unique_ptr<HloInstruction> CreateCTCLossWithLogits(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    xla::PrimitiveType in_dtype, xla::PrimitiveType out_dtype,
    int64 blank_index);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_CTC_LOSS_H_
