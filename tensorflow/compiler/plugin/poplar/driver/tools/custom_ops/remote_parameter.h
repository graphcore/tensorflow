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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_REMOTE_PARAMETER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_REMOTE_PARAMETER_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

// HloRemoteParameterLoad represents a parameter to the entry computation being
// loaded from remote memory into device memory.
class HloRemoteParameterLoad : public HloPoplarInstruction {
 public:
  explicit HloRemoteParameterLoad(const Shape& shape, int64 param_number);

  absl::flat_hash_set<int64> AllocatingIndices() const override;

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

  uint64 NumberOfInplaceOperands() const override;

  int64 GetParameterNumber() const { return param_number_; }

  bool IsPopOpsElementwise() const override;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

  const int64 param_number_;
};

std::unique_ptr<HloInstruction> CreateHloRemoteParameterLoad(
    const Shape& shape, int64 param_number);

// HloRemoteParameterStore represents an output to the entry computation being
// stored into remote memory from device memory.
class HloRemoteParameterStore : public HloPoplarInstruction {
 public:
  explicit HloRemoteParameterStore(HloInstruction* const output,
                                   int64 output_idx);

  absl::flat_hash_set<int64> AllocatingIndices() const override;

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

  uint64 NumberOfInplaceOperands() const override;

  int64 GetOutputIndex() const { return output_idx_; }

  bool IsPopOpsElementwise() const override;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

  const int64 output_idx_;
};

std::unique_ptr<HloInstruction> CreateHloRemoteParameterStore(
    HloInstruction* const output, int64 param_number);

// HloRemoteParameterDummyOutput represents a dummy input to the root tuple
// which is used to indicate a lack of Poplar tensor at that location (we cannot
// modify the output shape of the program).
class HloRemoteParameterDummyOutput : public HloPoplarInstruction {
 public:
  explicit HloRemoteParameterDummyOutput(const Shape& shape,
                                         int64 param_number);

  absl::flat_hash_set<int64> AllocatingIndices() const override;

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

  uint64 NumberOfInplaceOperands() const override;

  int64 GetOutputIndex() const { return output_idx_; }

  bool IsPopOpsElementwise() const override;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

  const int64 output_idx_;
};

std::unique_ptr<HloInstruction> CreateHloRemoteParameterDummyOutput(
    const Shape& shape, int64 param_number);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_REMOTE_PARAMETER_H_
