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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/normalise_image.h"

#include "absl/container/flat_hash_map.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

absl::flat_hash_set<int64> HloNormaliseImage::AllocatingIndices() const {
  return {0};
}

bool HloNormaliseImage::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64> HloNormaliseImage::LayoutDependencies()
    const {
  return {};
}

HloPoplarUseDescriptions HloNormaliseImage::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloNormaliseImage::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloNormaliseImage::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloNormaliseImage::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction> CreateHloNormaliseImage(
    HloInstruction* image, HloInstruction* channel_offsets,
    HloInstruction* channel_scales, const Shape& shape, const float scale) {
  return absl::make_unique<HloNormaliseImage>(image, channel_offsets,
                                              channel_scales, shape, scale);
}

std::unique_ptr<HloInstruction> HloNormaliseImage::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateHloNormaliseImage(operands[0], operands[1], operands[2], shape,
                                 scale_);
}

std::vector<std::string> HloNormaliseImage::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("scale=" + std::to_string(scale_));
  return attributes;
}

namespace {

static HloPoplarInstructionFactory normalise_image_factory(
    PoplarOp::NormaliseImage,
    [](HloCustomCallInstruction* custom_call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      const auto attribute_map =
          IPUCustomKernelsUtil::AttributeMap(custom_call);
      TF_ASSIGN_OR_RETURN(float scale,
                          attribute_map.GetAttributeAsFloat("scale"));
      auto args = custom_call->operands();
      return CreateHloNormaliseImage(args[0], args[1], args[2],
                                     custom_call->shape(), scale);
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
