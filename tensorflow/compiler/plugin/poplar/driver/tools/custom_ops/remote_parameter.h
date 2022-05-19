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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_remote_buffer_info.h"

namespace xla {
namespace poplarplugin {

/*
 * Abstract base class for all remote load/store instructions.
 */
class HloAbstractRemoteLoadStore : public HloPoplarInstruction {
 public:
  HloAbstractRemoteLoadStore(const Shape& shape,
                             absl::Span<HloInstruction* const> operands,
                             const std::vector<uint64>& replication_factors,
                             PoplarOp op);

  virtual ~HloAbstractRemoteLoadStore() = default;

  virtual absl::Span<HloInstruction* const> RemoteBuffers() const = 0;

  uint64 GetReplicationFactor(int64_t index) const;

  std::size_t GetReplicationFactorCount() const;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

  const std::vector<uint64> replication_factors_;
};

/**
 * HloRemoteParameterLoad represents a variable being loaded from remote memory
 * into device memory.
 */
class HloRemoteParameterLoad : public HloAbstractRemoteLoadStore {
 public:
  HloRemoteParameterLoad(absl::Span<HloInstruction* const> rbuffers,
                         const std::vector<uint64>& replication_factors);

  absl::Span<HloInstruction* const> RemoteBuffers() const override;

  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;

  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;

  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateHloRemoteParameterLoad(
    absl::Span<HloInstruction* const> rbuffers,
    const std::vector<uint64>& replication_factors = {1});

/**
 * HloRemoteParameterStore represents a write to a variable stored in remote
 * memory from device memory.
 */
class HloRemoteParameterStore : public HloAbstractRemoteLoadStore {
 public:
  HloRemoteParameterStore(absl::Span<HloInstruction* const> rbuffers_and_values,
                          const std::vector<uint64>& replication_factors);

  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;

  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;

  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

  absl::Span<HloInstruction* const> RemoteBuffers() const override;
  absl::Span<HloInstruction* const> ValuesToStore() const;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateHloRemoteParameterStore(
    absl::Span<HloInstruction* const> rbuffers_and_values,
    const std::vector<uint64>& replication_factors = {1});

/**
 * HloCreateBuffer represents a creation of a buffer, where the buffer can be
 * stored in either device memory or remote memory.
 * Note that the buffer is sliceable on the outter dimension.
 */
class HloCreateBuffer : public HloPoplarInstruction {
 public:
  explicit HloCreateBuffer(
      const Shape& shape, bool is_remote,
      absl::optional<HloRemoteBufferInfo> remote_buffer_info);

  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;

  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;

  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

  bool IsRemoteBuffer() const { return is_remote_; }

  absl::optional<HloRemoteBufferInfo> RemoteBufferInfo() const;

  std::unique_ptr<HloInstruction> CloneWithRemoteBufferInfo(
      const HloRemoteBufferInfo& info) const;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

  const bool is_remote_;
  const absl::optional<HloRemoteBufferInfo> remote_buffer_info_;
};

std::unique_ptr<HloInstruction> CreateHloCreateBuffer(const Shape& shape,
                                                      bool is_remote);

/**
 * HloBufferLoadSlice represents an instruction which loads a slice of data
 * from a buffer, which is stored in remote memory, at a given offset.
 */
class HloBufferLoadSlice : public HloAbstractRemoteLoadStore {
 public:
  HloBufferLoadSlice(const Shape& shape,
                     absl::Span<HloInstruction* const> rbuffers_and_offsets,
                     const std::vector<uint64>& replication_factors);

  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;
  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

  absl::Span<HloInstruction* const> RemoteBuffers() const override;
  absl::Span<HloInstruction* const> Offsets() const;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateBufferLoadSlice(
    const Shape& shape, HloInstruction* const buffer,
    HloInstruction* const offset, uint64 replication_factor = 1);

/**
 * HloBufferStoreSlice represents an instruction which stores a slice of data
 * into a buffer, which is stored in remote memory, at a given offset.
 * Outputs the updated buffer.
 */
class HloBufferStoreSlice : public HloAbstractRemoteLoadStore {
 public:
  HloBufferStoreSlice(
      const Shape& shape,
      absl::Span<HloInstruction* const> rbuffers_values_and_offsets,
      const std::vector<uint64>& replication_factors);

  absl::flat_hash_set<int64_t> AllocatingIndices() const override { return {}; }
  bool AllocatingOutput() const override { return false; }

  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override {
    return {};
  }

  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override {
    return FindConsumersExtensionResults::DoNotFindConsumers();
  }

  bool AllowNonInplaceLowering() const override { return false; }
  bool IsPopOpsElementwise() const override { return false; }

  absl::Span<HloInstruction* const> RemoteBuffers() const override;
  absl::Span<HloInstruction* const> ValuesToStore() const;
  absl::Span<HloInstruction* const> Offsets() const;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateBufferStoreSlice(
    HloInstruction* const buffer, HloInstruction* const slice,
    HloInstruction* const offset, uint64 replication_factor = 1);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_REMOTE_PARAMETER_H_
