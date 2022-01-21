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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_WITHIN_REPLICAS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_WITHIN_REPLICAS_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloWithinReplicaInstruction : public HloPoplarInstruction {
 public:
  explicit HloWithinReplicaInstruction(absl::Span<HloInstruction* const> inputs,
                                       const Shape& output_shape, PoplarOp op);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  bool AllocatingOutput() const override;

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;
};

class HloAllGatherWithinReplicaInstruction
    : public HloWithinReplicaInstruction {
 public:
  explicit HloAllGatherWithinReplicaInstruction(
      absl::Span<HloInstruction* const> inputs, const Shape& output_shape);

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

class HloReduceScatterWithinReplicaInstruction
    : public HloWithinReplicaInstruction {
 public:
  explicit HloReduceScatterWithinReplicaInstruction(
      absl::Span<HloInstruction* const> inputs, const Shape& output_shape,
      CollectiveOperator op);

  CollectiveOperator GetCollectiveOperator() const { return op_; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

  CollectiveOperator op_;
};

std::unique_ptr<HloInstruction> CreatePoplarAllGatherWithinReplica(
    absl::Span<HloInstruction* const> inputs, const Shape& output_shape);

std::unique_ptr<HloInstruction> CreatePoplarReduceScatterWithinReplica(
    absl::Span<HloInstruction* const> inputs, const Shape& output_shape,
    CollectiveOperator op);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_WITHIN_REPLICAS_H_
