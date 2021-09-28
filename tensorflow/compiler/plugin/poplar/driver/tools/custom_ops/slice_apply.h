/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_SLICE_APPLY_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_SLICE_APPLY_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloSliceApplyBase : public HloPoplarInstruction {
 protected:
  HloSliceApplyBase(const std::vector<HloInstruction*>& operands,
                    int64 apply_dimension, int64 start_index,
                    HloOpcode operation, PoplarOp poplar_op);

 public:
  absl::flat_hash_set<int64> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;
  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

  int64 GetApplyDimension() const;
  int64 GetStartIndex() const;
  HloOpcode GetOperation() const;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  const int64 apply_dimension_;
  const int64 start_index_;
  const HloOpcode operation_;
};

// Slices input into along the `apply_dimension` dimension from `start_index` to
// form `input_slice` and applies the update using the elementwise `operation`.
class HloSliceApply : public HloSliceApplyBase {
 public:
  HloSliceApply(HloInstruction* const input, HloInstruction* const update,
                int64 apply_dimension, int64 start_index, HloOpcode operation);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* context) const override;
};

std::unique_ptr<HloInstruction> CreateSliceApply(HloInstruction* const input,
                                                 HloInstruction* const update,
                                                 int64 apply_dimension,
                                                 int64 start_index,
                                                 HloOpcode operation);

// Slices input into along the `apply_dimension` dimension from `start_index` to
// form `input_slice` and applies the update using the elementwise `operation`,
// where both the `input_slice` and `update` are pre-multiplied by a scalar.
class HloSliceApplyaXbY : public HloSliceApplyBase {
 public:
  HloSliceApplyaXbY(HloInstruction* const input, HloInstruction* const update,
                    HloInstruction* const scale_input,
                    HloInstruction* const scale_update, int64 apply_dimension,
                    int64 start_index, HloOpcode operation);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* context) const override;
};

std::unique_ptr<HloInstruction> CreateSliceApplyaXbY(
    HloInstruction* const input, HloInstruction* const update,
    HloInstruction* const scale_input, HloInstruction* const scale_update,
    int64 apply_dimension, int64 start_index, HloOpcode operation);

// Slices input into along the `apply_dimension` dimension from `start_index` to
// form `input_slice` and applies the update using the elementwise `operation`,
// where the `update` is pre-multiplied by a scalar.
class HloSliceApplyabY : public HloSliceApplyBase {
 public:
  HloSliceApplyabY(HloInstruction* const input, HloInstruction* const update,
                   HloInstruction* const scale_update, int64 apply_dimension,
                   int64 start_index, HloOpcode operation);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* context) const override;
};

std::unique_ptr<HloInstruction> CreateSliceApplyabY(
    HloInstruction* const input, HloInstruction* const update,
    HloInstruction* const scale_update, int64 apply_dimension,
    int64 start_index, HloOpcode operation);

// Slices input into along the `apply_dimension` dimension from `start_index` to
// form `input_slice` and applies the update using the elementwise `operation`,
// where the `input_slice` is pre-multiplied by a scalar.
class HloSliceApplyaXb : public HloSliceApplyBase {
 public:
  HloSliceApplyaXb(HloInstruction* const input, HloInstruction* const update,
                   HloInstruction* const scale_input, int64 apply_dimension,
                   int64 start_index, HloOpcode operation);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* context) const override;
};

std::unique_ptr<HloInstruction> CreateSliceApplyaXb(
    HloInstruction* const input, HloInstruction* const update,
    HloInstruction* const scale_input, int64 apply_dimension, int64 start_index,
    HloOpcode operation);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_SLICE_APPLY_H_
