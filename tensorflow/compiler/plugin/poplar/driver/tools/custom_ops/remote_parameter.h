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

/**
 * HloRemoteParameterLoad represents a variable being loaded from remote memory
 * into device memory.
 */
class HloRemoteParameterLoad : public HloPoplarInstruction {
 public:
  HloRemoteParameterLoad(const Shape& shape,
                         absl::Span<HloInstruction* const> rbuffers);

  absl::flat_hash_set<int64> AllocatingIndices() const override;

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

  uint64 NumberOfInplaceOperands() const override;

  bool IsPopOpsElementwise() const override;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateHloRemoteParameterLoad(
    HloInstruction* const rbuffer);

/**
 * HloRemoteParameterStore represents a write to a variable stored in remote
 * memory from device memory.
 */
class HloRemoteParameterStore : public HloPoplarInstruction {
 public:
  explicit HloRemoteParameterStore(
      const xla::Shape& shape,
      absl::Span<HloInstruction* const> rbuffers_and_values);

  absl::flat_hash_set<int64> AllocatingIndices() const override;

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

  uint64 NumberOfInplaceOperands() const override;

  bool IsPopOpsElementwise() const override;

  absl::Span<HloInstruction* const> RemoteBuffers() const;
  absl::Span<HloInstruction* const> ValuesToStore() const;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateHloRemoteParameterStore(
    HloInstruction* const rbuffer, HloInstruction* const value);

// HloCreateBuffer represents a creation of a buffer, where the buffer can be
// stored in either device memory or remote memory.
class HloCreateBuffer : public HloPoplarInstruction {
 public:
  explicit HloCreateBuffer(const Shape& shape, bool is_remote);

  absl::flat_hash_set<int64> AllocatingIndices() const override;

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

  uint64 NumberOfInplaceOperands() const override;

  bool IsPopOpsElementwise() const override;

  bool IsRemoteBuffer() const { return is_remote_; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

  const bool is_remote_;
};

std::unique_ptr<HloInstruction> CreateHloCreateBuffer(const Shape& shape,
                                                      bool is_remote);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_REMOTE_PARAMETER_H_
