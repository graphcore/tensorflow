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
  explicit HloRemoteParameterLoad(absl::Span<HloInstruction* const> rbuffers,
                                  std::vector<uint64> replication_factors = {
                                      1});

  absl::flat_hash_set<int64> AllocatingIndices() const override;

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

  uint64 NumberOfInplaceOperands() const override;

  bool IsPopOpsElementwise() const override;
  uint64 GetReplicationFactor(int64 index) const {
    CHECK_LT(index, replication_factors_.size());
    return replication_factors_[index];
  }
  std::size_t GetReplicationFactorCount() const {
    return replication_factors_.size();
  }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  const std::vector<uint64> replication_factors_;

  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateHloRemoteParameterLoad(
    absl::Span<HloInstruction* const> rbuffers,
    std::vector<uint64> replication_factors = {1});

/**
 * HloRemoteParameterStore represents a write to a variable stored in remote
 * memory from device memory.
 */
class HloRemoteParameterStore : public HloPoplarInstruction {
 public:
  explicit HloRemoteParameterStore(
      absl::Span<HloInstruction* const> rbuffers_and_values,
      std::vector<uint64> replication_factors = {1});

  absl::flat_hash_set<int64> AllocatingIndices() const override;

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

  uint64 NumberOfInplaceOperands() const override;

  bool IsPopOpsElementwise() const override;

  absl::Span<HloInstruction* const> RemoteBuffers() const;
  absl::Span<HloInstruction* const> ValuesToStore() const;
  uint64 GetReplicationFactor(int64 index) const {
    CHECK_LT(index, replication_factors_.size());
    return replication_factors_[index];
  }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  const std::vector<uint64> replication_factors_;

  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateHloRemoteParameterStore(
    absl::Span<HloInstruction* const> rbuffers_and_values,
    std::vector<uint64> replication_factors = {1});

/**
 * HloCreateBuffer represents a creation of a buffer, where the buffer can be
 * stored in either device memory or remote memory.
 * Note that the buffer is sliceable on the outter dimension.
 */
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

/**
 * HloBufferLoadSlice represents an instruction which loads a slice of data
 * from a buffer, which is stored in remote memory, at a given offset.
 */
class HloBufferLoadSlice : public HloPoplarInstruction {
 public:
  HloBufferLoadSlice(const Shape& shape, HloInstruction* const buffer,
                     HloInstruction* const offset);

  absl::flat_hash_set<int64> AllocatingIndices() const override { return {}; }

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override {
    return {};
  }

  uint64 NumberOfInplaceOperands() const override { return 0; }

  bool IsPopOpsElementwise() const override { return false; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateBufferLoadSlice(
    const Shape& shape, HloInstruction* const buffer,
    HloInstruction* const offset);

/**
 * HloBufferStoreSlice represents an instruction which stores a slice of data
 * into a buffer, which is stored in remote memory, at a given offset.
 * Outputs the updated buffer.
 */
class HloBufferStoreSlice : public HloPoplarInstruction {
 public:
  HloBufferStoreSlice(HloInstruction* const buffer, HloInstruction* const slice,
                      HloInstruction* const offset);

  absl::flat_hash_set<int64> AllocatingIndices() const override { return {}; }

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override {
    return {};
  }

  uint64 NumberOfInplaceOperands() const override { return 1; }

  bool IsPopOpsElementwise() const override { return false; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateBufferStoreSlice(
    HloInstruction* const buffer, HloInstruction* const slice,
    HloInstruction* const offset);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_REMOTE_PARAMETER_H_
