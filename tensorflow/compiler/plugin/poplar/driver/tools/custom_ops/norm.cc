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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/norm.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/tf2xla/type_util.h"

namespace xla {
namespace poplarplugin {

HloNormInstruction::HloNormInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands, PoplarOp op,
    int32 num_groups, float epsilon, int feature_index)
    : HloPoplarInstruction(shape, operands, op, num_groups, epsilon,
                           feature_index),
      num_groups_(num_groups),
      epsilon_(epsilon),
      feature_index_(feature_index) {}

int32 HloNormInstruction::num_groups() const { return num_groups_; }
int32 HloNormInstruction::feature_index() const { return feature_index_; }
float HloNormInstruction::epsilon() const { return epsilon_; }

std::vector<std::string> HloNormInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("num_groups=" + std::to_string(num_groups_));
  attributes.push_back("epsilon=" + std::to_string(epsilon_));
  attributes.push_back("feature_index=" + std::to_string(feature_index_));

  return attributes;
}

HloGroupNormBaseInstruction::HloGroupNormBaseInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands, PoplarOp op,
    int32 num_groups, bool strided_channel_grouping, float epsilon,
    int feature_index)
    : HloNormInstruction(shape, operands, op, num_groups, epsilon,
                         feature_index),
      strided_channel_grouping_(strided_channel_grouping) {}

bool HloGroupNormBaseInstruction::strided_channel_grouping() const {
  return strided_channel_grouping_;
}

}  // namespace poplarplugin
}  // namespace xla
