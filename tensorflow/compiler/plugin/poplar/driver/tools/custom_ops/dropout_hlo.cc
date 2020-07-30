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

#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace xla {
namespace poplarplugin {

HloDropoutInstruction::HloDropoutInstruction(
    HloInstruction* X, HloInstruction* seed, float rate_, float scale_,
    bool should_use_user_seed, bool modify_seed,
    const std::vector<int64>& noise_shape)
    : HloPoplarInstruction(
          xla::ShapeUtil::MakeTupleShape({X->shape(), seed->shape()}),
          {X, seed}, PoplarOp::Dropout, rate_, scale_, should_use_user_seed,
          modify_seed, noise_shape),
      scale(scale_),
      rate(rate_),
      is_user_seed(should_use_user_seed),
      modify_seed(modify_seed),
      noise_shape(noise_shape) {}

absl::flat_hash_set<int64> HloDropoutInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloDropoutInstruction::LayoutDependencies()
    const {
  return {};
}

uint64 HloDropoutInstruction::NumberOfInplaceOperands() const { return 0; }

bool HloDropoutInstruction::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction> HloDropoutInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloDropoutInstruction>(
      new_operands[0], new_operands[1], Rate(), Scale(), IsUserSeed(),
      ModifySeed(), NoiseShape());
}

std::vector<std::string>
HloDropoutInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("scale=" + std::to_string(scale));
  attributes.push_back("rate=" + std::to_string(rate));
  attributes.push_back("is_user_seed=" + std::to_string(is_user_seed));
  attributes.push_back("modify_seed=" + std::to_string(modify_seed));

  // Noise shape is an optional list attribute.
  if (HasNoiseShape()) {
    attributes.push_back("noise_shape=" + absl::StrJoin(noise_shape, ","));
  }
  return attributes;
}

std::unique_ptr<HloInstruction> CreateDropout(
    HloInstruction* operand, HloInstruction* seed, float rate, float scale,
    bool should_use_user_seed, bool modify_seed,
    const std::vector<int64>& noise_shape) {
  return absl::make_unique<HloDropoutInstruction>(operand, seed, rate, scale,
                                                  should_use_user_seed,
                                                  modify_seed, noise_shape);
}

namespace {

StatusOr<std::unique_ptr<HloInstruction>> HloDropoutInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  // Get the attribute values
  TF_ASSIGN_OR_RETURN(float rate, attribute_map.GetAttributeAsFloat("rate"));
  TF_ASSIGN_OR_RETURN(float scale, attribute_map.GetAttributeAsFloat("scale"));
  TF_ASSIGN_OR_RETURN(bool should_use_user_seed,
                      attribute_map.GetAttributeAsBool("is_using_user_seed"));
  TF_ASSIGN_OR_RETURN(bool modify_seed,
                      attribute_map.GetAttributeAsBool("modify_seed"));

  // If the noise_shape attribute is not present (defaults to empty list), we
  // want the corresponding std::vector to also be empty.
  std::vector<int64> noise_shape;
  if (attribute_map.HasAttribute("noise_shape")) {
    TF_ASSIGN_OR_RETURN(noise_shape,
                        attribute_map.GetAttributeInt64Vector("noise_shape"));
  }

  return CreateDropout(call->mutable_operand(0), call->mutable_operand(1), rate,
                       scale, should_use_user_seed, modify_seed, noise_shape);
}

static HloPoplarInstructionFactory dropout_factory(
    PoplarOp::Dropout, HloDropoutInstructionFactoryFunc);

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
