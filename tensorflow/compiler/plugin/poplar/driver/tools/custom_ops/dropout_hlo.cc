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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/dropout_hlo.h"

#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace xla {
namespace poplarplugin {

HloDropout::HloDropout(HloInstruction* operand, HloInstruction* seed,
                       float rate, float scale,
                       const std::vector<int64>& noise_shape)
    : HloPoplarInstruction(
          ShapeUtil::MakeTupleShape(
              {operand->shape(), seed->shape(), ShapeUtil::MakeOpaqueShape()}),
          {operand, seed}, PoplarOp::Dropout, rate, scale, noise_shape),
      scale(scale),
      rate(rate),
      noise_shape(noise_shape) {}

HloDropout::HloDropout(HloInstruction* operand, HloInstruction* seed,
                       HloInstruction* reference, float rate, float scale,
                       const std::vector<int64>& noise_shape)
    : HloPoplarInstruction(
          ShapeUtil::MakeTupleShape(
              {operand->shape(), seed->shape(), ShapeUtil::MakeOpaqueShape()}),
          {operand, seed, reference}, PoplarOp::Dropout, rate, scale,
          noise_shape),
      scale(scale),
      rate(rate),
      noise_shape(noise_shape) {}

absl::flat_hash_set<int64> HloDropout::AllocatingIndices() const { return {}; }

bool HloDropout::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64> HloDropout::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloDropout::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloDropout::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloDropout::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloDropout::IsPopOpsElementwise() const { return false; }

std::vector<std::string> HloDropout::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("scale=" + std::to_string(scale));
  attributes.push_back("rate=" + std::to_string(rate));
  if (HasNoiseShape()) {
    attributes.push_back("noise_shape=" + absl::StrJoin(noise_shape, ","));
  }

  return attributes;
}

std::unique_ptr<HloInstruction> HloDropout::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  CHECK_GT(new_operands.size(), 1);
  CHECK_LT(new_operands.size(), 4);
  if (new_operands.size() == 2) {
    return absl::make_unique<HloDropout>(new_operands[0], new_operands[1],
                                         Rate(), Scale(), NoiseShape());
  } else {
    return absl::make_unique<HloDropout>(new_operands[0], new_operands[1],
                                         new_operands[2], Rate(), Scale(),
                                         NoiseShape());
  }
}

std::unique_ptr<HloInstruction> CreateDropout(
    HloInstruction* operand, HloInstruction* seed, float rate, float scale,
    const std::vector<int64>& noise_shape) {
  return absl::make_unique<HloDropout>(operand, seed, rate, scale, noise_shape);
}

std::unique_ptr<HloInstruction> CreateDropout(
    HloInstruction* operand, HloInstruction* seed, HloInstruction* reference,
    float rate, float scale, const std::vector<int64>& noise_shape) {
  return absl::make_unique<HloDropout>(operand, seed, reference, rate, scale,
                                       noise_shape);
}

namespace {

StatusOr<std::unique_ptr<HloInstruction>> HloDropoutFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  // Get the attribute values
  TF_ASSIGN_OR_RETURN(float rate, attribute_map.GetAttributeAsFloat("rate"));
  TF_ASSIGN_OR_RETURN(float scale, attribute_map.GetAttributeAsFloat("scale"));

  // If the noise_shape attribute is not present (defaults to empty list), we
  // want the corresponding std::vector to also be empty.
  std::vector<int64> noise_shape;
  if (attribute_map.HasAttribute("noise_shape")) {
    TF_ASSIGN_OR_RETURN(noise_shape,
                        attribute_map.GetAttributeInt64Vector("noise_shape"));
  }

  if (call->operand_count() == 2) {
    return CreateDropout(call->mutable_operand(0), call->mutable_operand(1),
                         rate, scale, noise_shape);
  } else {
    return CreateDropout(call->mutable_operand(0), call->mutable_operand(1),
                         call->mutable_operand(2), rate, scale, noise_shape);
  }
}

static HloPoplarInstructionFactory dropout_factory(PoplarOp::Dropout,
                                                   HloDropoutFactoryFunc);
}  // namespace
}  // namespace poplarplugin
}  // namespace xla

// std::vector<long long> does not by default have a std::hash specialization.
namespace std {
template <>
struct hash<std::vector<tensorflow::int64>> {
  size_t operator()(const std::vector<tensorflow::int64>& vec) const {
    return tensorflow::Hash64(absl::StrJoin(vec, ""));
  }
};
}  // namespace std
