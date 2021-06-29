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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_HOST_EMBEDDING_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_HOST_EMBEDDING_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

enum class HostEmbeddingSplittingStrategy { Encoding, Token };

class HloHostEmbeddingLookupInstruction : public HloPoplarInstruction {
 public:
  explicit HloHostEmbeddingLookupInstruction(
      HloInstruction* indices, const std::string& embedding_id,
      const xla::Shape& embedding_shape,
      HostEmbeddingSplittingStrategy splitting_strategy, const Shape shape);

  absl::flat_hash_set<int64> AllocatingIndices() const override;

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool AllocatingOutput() const override;

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  const std::string& EmbeddingId() const { return embedding_id_; }
  const xla::Shape& EmbeddingShape() const { return embedding_shape_; }
  HostEmbeddingSplittingStrategy SplittingStrategy() const {
    return splitting_strategy_;
  }

  bool IsPopOpsElementwise() const override;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

  std::string embedding_id_;
  xla::Shape embedding_shape_;
  HostEmbeddingSplittingStrategy splitting_strategy_ =
      HostEmbeddingSplittingStrategy::Token;
};

std::unique_ptr<HloInstruction> CreateHostEmbeddingLookup(
    HloInstruction* indices, const std::string& embedding_id,
    const xla::Shape& embedding_shape,
    HostEmbeddingSplittingStrategy splitting_strategy, const Shape shape);

class HloHostEmbeddingUpdateInstruction : public HloPoplarInstruction {
 public:
  explicit HloHostEmbeddingUpdateInstruction(
      HloInstruction* in, HloInstruction* grads, HloInstruction* indices,
      const std::string& embedding_id, const xla::Shape& embedding_shape,
      HostEmbeddingSplittingStrategy splitting_strategy, const Shape shape);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  bool AllocatingOutput() const override;

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  const std::string& EmbeddingId() const { return embedding_id_; }
  const xla::Shape& EmbeddingShape() const { return embedding_shape_; }
  HostEmbeddingSplittingStrategy SplittingStrategy() const {
    return splitting_strategy_;
  }

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

  std::string embedding_id_;
  xla::Shape embedding_shape_;
  HostEmbeddingSplittingStrategy splitting_strategy_ =
      HostEmbeddingSplittingStrategy::Token;
};

std::unique_ptr<HloInstruction> CreateHloHostEmbeddingUpdate(
    HloInstruction* grads, HloInstruction* indices,
    const std::string& embedding_id, const xla::Shape& embedding_shape,
    HostEmbeddingSplittingStrategy splitting_strategy, const Shape shape);

class HloHostEmbeddingNotifyInstruction : public HloPoplarInstruction {
 public:
  explicit HloHostEmbeddingNotifyInstruction(const std::string& embedding_id);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  bool AllocatingOutput() const override;

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  const std::string& EmbeddingId() const { return embedding_id_; }

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

  std::string embedding_id_;
};

std::unique_ptr<HloInstruction> CreateHloHostEmbeddingNotify(
    const std::string& embedding_id);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_HOST_EMBEDDING_H_
