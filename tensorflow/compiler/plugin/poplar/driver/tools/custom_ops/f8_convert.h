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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_F8_CONVERT_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_F8_CONVERT_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

template <PoplarOp Op>
class HloF8ConvertInstruction : public HloPoplarInstruction {
 public:
  HloF8ConvertInstruction(const Shape& shape,
                          absl::Span<HloInstruction* const> operands)
      : HloPoplarInstruction(shape, operands, Op) {}

  absl::flat_hash_set<int64_t> AllocatingIndices() const override { return {}; }
  bool AllocatingOutput() const override { return false; }

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
    return FindConsumersExtensionResults();
  }

  bool AllowNonInplaceLowering() const override { return true; }

  bool IsPopOpsElementwise() const override { return false; }

  std::vector<string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions&) const override {
    return {};
  }
};

class HloConvertFromF8Instruction
    : public HloF8ConvertInstruction<PoplarOp::ConvertFromF8> {
 public:
  HloConvertFromF8Instruction(const Shape& shape, HloInstruction* data,
                              HloInstruction* metadata);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};
std::unique_ptr<HloInstruction> CreateConvertToF8Instruction(
    HloInstruction* operand);

class HloConvertToF8Instruction
    : public HloF8ConvertInstruction<PoplarOp::ConvertToF8> {
 public:
  HloConvertToF8Instruction(const Shape& shape, HloInstruction* data,
                            HloInstruction* metadata);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};
std::unique_ptr<HloInstruction> CreateConvertFromF8Instruction(
    HloInstruction* operand);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_F8_CONVERT_H_
