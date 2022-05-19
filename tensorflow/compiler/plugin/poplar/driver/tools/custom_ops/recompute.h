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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_RECOMPUTE_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_RECOMPUTE_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloSuggestRecomputeInstruction : public HloPoplarInstruction {
 public:
  explicit HloSuggestRecomputeInstruction(HloInstruction* operand);

  const HloInstruction* input() const;

  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateSuggestRecompute(HloInstruction* operand);

class HloBlockRecomputeInstruction : public HloPoplarInstruction {
 public:
  explicit HloBlockRecomputeInstruction(HloInstruction* operand);

  const HloInstruction* input() const;

  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateBlockRecompute(HloInstruction* operand);

// An instruction used to indicate that the input value should be checkpointed
// instead of recomputed. When used in pipelining, the input operand will be
// passed directly as an input to the recomputation/backward operations without
// the need of recomputing its predecessors.
class HloRecomputationCheckpointInstruction : public HloPoplarInstruction {
 public:
  explicit HloRecomputationCheckpointInstruction(HloInstruction* const operand);

  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;
  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateRecomputationCheckpoint(
    HloInstruction* const operand);

// An instruction used to indicate the recomputation predecessors of a
// checkpointed input. Used in pipelining to make sure any predecessors of
// `old_input` are executed after all possible successors of
// `checkpointed_input`.
class HloRecomputationInputInstruction : public HloPoplarInstruction {
 public:
  explicit HloRecomputationInputInstruction(
      HloInstruction* const checkpointed_input,
      HloInstruction* const old_input);

  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;
  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateRecomputationInput(
    HloInstruction* const checkpointed_input, HloInstruction* const old_input);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_RECOMPUTE_H_
