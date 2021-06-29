/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_MULTI_SLICE_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_MULTI_SLICE_H_

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloMultiSliceInstruction : public HloPoplarInstruction {
 public:
  explicit HloMultiSliceInstruction(const Shape& shape,
                                    HloInstruction* const input,
                                    HloInstruction* const indices);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  // Run consumers
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool IsPopOpsElementwise() const override;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

class HloMultiUpdateInstruction : public HloPoplarInstruction {
 public:
  explicit HloMultiUpdateInstruction(const Shape& shape,
                                     absl::Span<HloInstruction* const> operands,
                                     std::size_t index_vector_dim,
                                     std::size_t update_dim,
                                     uint32 serialization_factor,
                                     bool is_update_add = false);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool IsPopOpsElementwise() const override;

  // The dimension in indices which contains the starting indices. If it is
  // equal to the indices tensor rank we implicitly consider that tensor to
  // have a trailing 1 dimension.
  std::size_t GetIndexVectorDimension() const { return index_vector_dim_; }

  // The dimension of the update operand which represents the slice.
  std::size_t GetUpdateSliceDimension() const { return update_dim_; }

  // Factor used for serializing the multi update.
  std::size_t GetSerializationFactor() const { return serialization_factor_; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

  const std::size_t index_vector_dim_;
  const std::size_t update_dim_;
  const uint32 serialization_factor_;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

class HloMultiUpdateAddInstruction : public HloMultiUpdateInstruction {
 public:
  explicit HloMultiUpdateAddInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      std::size_t index_vector_dim, std::size_t update_dim,
      uint32 serialization_factor);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateMultiSlice(const Shape& shape,
                                                 HloInstruction* const input,
                                                 HloInstruction* const indices);

std::unique_ptr<HloInstruction> CreateMultiUpdate(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    std::size_t index_vector_dim, std::size_t update_dim,
    uint32 serialization_factor = 1);

std::unique_ptr<HloInstruction> CreateMultiUpdateAdd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    std::size_t index_vector_dim, std::size_t update_dim,
    uint32 serialization_factor = 1);

}  // namespace poplarplugin
}  // namespace xla

#endif
