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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_CANDIDATE_SAMPLER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_CANDIDATE_SAMPLER_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloCandidateSampler : public HloPoplarInstruction {
 public:
  explicit HloCandidateSampler(HloInstruction* input, HloInstruction* seed,
                               const Shape shape, bool unique,
                               const uint64 range_max, const std::string dist)
      : HloPoplarInstruction(shape, {input, seed}, PoplarOp::CandidateSampler,
                             unique, range_max, dist),
        unique_(unique),
        range_max_(range_max),
        dist_(dist) {}

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;
  bool IsPopOpsElementwise() const override;

  bool Unique() const { return unique_; }
  const int64 RangeMax() const { return range_max_; }
  const std::string Distribution() const { return dist_; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

  bool unique_;
  const uint64 range_max_;
  const std::string dist_;
};

std::unique_ptr<HloInstruction> CreateHloCandidateSampler(
    HloInstruction* input, HloInstruction* seed, const Shape& shape,
    bool unique, const uint64 range_max, const std::string dist);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_CANDIDATE_SAMPLER_H_
