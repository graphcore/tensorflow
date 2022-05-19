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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/candidate_sampler.h"

#include "absl/container/flat_hash_map.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

absl::flat_hash_set<int64_t> HloCandidateSampler::AllocatingIndices() const {
  return {};
}

bool HloCandidateSampler::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64_t, int64_t> HloCandidateSampler::LayoutDependencies()
    const {
  return {};
}

HloPoplarUseDescriptions HloCandidateSampler::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloCandidateSampler::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloCandidateSampler::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloCandidateSampler::AllowNonInplaceLowering() const { return false; }
bool HloCandidateSampler::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction> CreateHloCandidateSampler(
    HloInstruction* input, HloInstruction* seed, const Shape& shape,
    bool unique, const uint64 range_max, const std::string dist) {
  return absl::make_unique<HloCandidateSampler>(input, seed, shape, unique,
                                                range_max, dist);
}

std::unique_ptr<HloInstruction> HloCandidateSampler::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateHloCandidateSampler(operands[0], operands[1], shape, unique_,
                                   range_max_, dist_);
}

std::vector<std::string> HloCandidateSampler::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("unique=" + std::to_string(unique_));
  attributes.push_back("range_max=" + std::to_string(range_max_));
  attributes.push_back("distribution=" + dist_);
  return attributes;
}

namespace {

static HloPoplarInstructionFactory candidate_sampler_factory(
    PoplarOp::CandidateSampler,
    [](HloCustomCallInstruction* custom_call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      const auto attribute_map =
          IPUCustomKernelsUtil::AttributeMap(custom_call);
      TF_ASSIGN_OR_RETURN(bool unique,
                          attribute_map.GetAttributeAsBool("unique"));
      TF_ASSIGN_OR_RETURN(const uint64 range_max,
                          attribute_map.GetAttributeAsInt("range_max"));
      TF_ASSIGN_OR_RETURN(std::string dist,
                          attribute_map.GetAttributeAsString("dist"));
      auto args = custom_call->operands();
      return CreateHloCandidateSampler(args[0], args[1], custom_call->shape(),
                                       unique, range_max, dist);
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
