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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_SCALED_INPLACE_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_SCALED_INPLACE_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloScaledInplaceBase : public HloPoplarInstruction {
 protected:
  HloScaledInplaceBase(absl::Span<HloInstruction* const> operands,
                       HloOpcode operation, PoplarOp poplar_op);

 public:
  absl::flat_hash_set<int64> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;
  bool IsPopOpsElementwise() const override;
  HloOpcode GetOperation() const;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  const HloOpcode operation_;
};

class HloScaledInplaceXbY : public HloScaledInplaceBase {
 public:
  HloScaledInplaceXbY(HloInstruction* const x, HloInstruction* const y,
                      HloInstruction* const scale, HloOpcode operation);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* context) const override;
};

std::unique_ptr<HloInstruction> CreateScaledInplaceXbY(
    HloInstruction* const x, HloInstruction* const y,
    HloInstruction* const scale, HloOpcode operation);

class HloScaledInplaceaXbY : public HloScaledInplaceBase {
 public:
  HloScaledInplaceaXbY(HloInstruction* const x, HloInstruction* const y,
                       HloInstruction* const a, HloInstruction* const b,
                       HloOpcode operation);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* context) const override;
};

std::unique_ptr<HloInstruction> CreateScaledInplaceaXbY(HloInstruction* const x,
                                                        HloInstruction* const y,
                                                        HloInstruction* const a,
                                                        HloInstruction* const b,
                                                        HloOpcode operation);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_SCALED_INPLACE_H_
