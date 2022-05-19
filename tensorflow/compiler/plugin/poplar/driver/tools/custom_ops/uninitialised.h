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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_UNINITIALISED_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_UNINITIALISED_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

std::unique_ptr<HloInstruction> CreateUninitialisedInstruction(
    const Shape& shape, int64_t& identifier);

// Instruction used to create an uninitialised buffer
class HloUninitialisedInstruction : public HloPoplarInstruction {
 public:
  explicit HloUninitialisedInstruction(const Shape& shape, int64_t& identifier);
  absl::flat_hash_set<int64_t> AllocatingIndices() const override { return {}; }
  bool AllocatingOutput() const override { return true; }
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override {
    return {};
  }
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool AllowNonInplaceLowering() const override { return false; }
  bool IsPopOpsElementwise() const override { return false; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override {
    return {absl::StrCat("identifier=", identifier_)};
  }

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override {
    int64_t copy_val = identifier_ - 1;
    return CreateUninitialisedInstruction(shape, copy_val);
  }
  int64_t identifier_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_UNINITIALISED_H_
