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

#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_update_scale_apply.h"

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/all_gather.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/replication_factor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

namespace {
// clang-format off
static const std::vector<HloMatcherPattern> patterns = {
  HloMatcherPattern(
    PatternType("multi_update"),
    PatternMetaTarget(2),
    PatternInputs({5, 6, 7}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({2, 1}), IsMultiplyOrDivide},
      {HloOpcode::kBroadcast, NodeOperands({7})},
      {HloOpcode::kCustomCall, NodeOperands({3, 5, 6}), IsPoplarInstruction(PoplarOp::MultiUpdate)},
      {HloOpcode::kBroadcast, NodeOperands({4})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsScalar},
    })
  ),

  HloMatcherPattern(
    PatternType("multi_update_add"),
    PatternMetaTarget(2),
    PatternInputs({5, 6, 7}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({2, 1}), IsMultiplyOrDivide},
      {HloOpcode::kBroadcast, NodeOperands({7})},
      {HloOpcode::kCustomCall, NodeOperands({3, 5, 6, 8}), IsPoplarInstruction(PoplarOp::MultiUpdateAdd)},
      {HloOpcode::kBroadcast, NodeOperands({4})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsScalar},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantOne},
    })
  ),
};
// clang-format on

Status ReplaceMatch(HloMatcherMatched& match) {
  HloComputation* comp = match.computation;
  HloInstruction* scale = match.instruction_mapping.at(7);
  HloInstruction* root = match.instruction_mapping.at(0);
  HloMultiUpdateInstruction* multi_update =
      Cast<HloMultiUpdateInstruction>(match.instruction_mapping.at(2));

  // If the result is divided by a constant, then turn it into multiply.
  const bool do_reciprocal = root->opcode() == HloOpcode::kDivide;
  if (do_reciprocal) {
    HloInstruction* one = comp->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::One(scale->shape().element_type())));
    scale = comp->AddInstruction(HloInstruction::CreateBinary(
        scale->shape(), HloOpcode::kDivide, one, scale));
  }
  // Create a new multi update add instruction.
  HloInstruction* new_multi_update_add =
      comp->AddInstruction(CreateMultiUpdateAdd(
          multi_update->shape(),
          {multi_update->mutable_operand(0), multi_update->mutable_operand(1),
           multi_update->mutable_operand(2), scale},
          multi_update->GetSerializationFactor()));

  // Replace it.
  multi_update->SetupDerivedInstruction(new_multi_update_add);
  TF_RETURN_IF_ERROR(comp->ReplaceInstruction(root, new_multi_update_add));
  return Status::OK();
}
};  // namespace

MultiUpdateScaleApply::MultiUpdateScaleApply(CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, false, false) {}

StatusOr<bool> MultiUpdateScaleApply::HandleMatch(HloMatcherMatched& match,
                                                  const absl::optional<int64>) {
  TF_RETURN_IF_ERROR(ReplaceMatch(match));
  return true;
}

}  // namespace poplarplugin
}  // namespace xla
