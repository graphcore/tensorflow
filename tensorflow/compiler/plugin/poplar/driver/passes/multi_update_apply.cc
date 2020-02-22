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

#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_update_apply.h"

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
    PatternType("multi_update_rhs"),
    PatternMetaTarget(1),
    PatternInputs({4, 5, 6}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({4, 1}), IsAddOrSubtract},
      {HloOpcode::kCustomCall, NodeOperands({2, 5, 6}), IsPoplarInstruction(PoplarOp::MultiUpdate)},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
    })
  ),

  HloMatcherPattern(
    PatternType("multi_update_add_rhs"),
    PatternMetaTarget(1),
    PatternInputs({4, 5, 6, 7}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({4, 1}), IsAddOrSubtract},
      {HloOpcode::kCustomCall, NodeOperands({2, 5, 6, 7}), IsPoplarInstruction(PoplarOp::MultiUpdateAdd)},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
    })
  ),

  // Note that for LHS we can only support add.
  HloMatcherPattern(
    PatternType("multi_update_lhs"),
    PatternMetaTarget(1),
    PatternInputs({4, 5, 6}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({1, 4})},
      {HloOpcode::kCustomCall, NodeOperands({2, 5, 6}), IsPoplarInstruction(PoplarOp::MultiUpdate)},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
    })
  ),

  HloMatcherPattern(
    PatternType("multi_update_add_lhs"),
    PatternMetaTarget(1),
    PatternInputs({4, 5, 6, 7}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({1, 4})},
      {HloOpcode::kCustomCall, NodeOperands({2, 5, 6, 7}), IsPoplarInstruction(PoplarOp::MultiUpdateAdd)},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
    })
  ),

  HloMatcherPattern(
    PatternType("multi_update_rhs_with_reshape"),
    PatternMetaTarget(2),
    PatternInputs({5, 6, 7}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({5, 1}), IsAddOrSubtract},
      {HloOpcode::kReshape, NodeOperands({2})},
      {HloOpcode::kCustomCall, NodeOperands({3, 6, 7}), IsPoplarInstruction(PoplarOp::MultiUpdate)},
      {HloOpcode::kBroadcast, NodeOperands({4})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
    })
  ),

  HloMatcherPattern(
    PatternType("multi_update_add_rhs_with_reshape"),
    PatternMetaTarget(1),
    PatternInputs({5, 6, 7, 8}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({5, 1}), IsAddOrSubtract},
      {HloOpcode::kReshape, NodeOperands({2})},
      {HloOpcode::kCustomCall, NodeOperands({3, 6, 7, 8}), IsPoplarInstruction(PoplarOp::MultiUpdateAdd)},
      {HloOpcode::kBroadcast, NodeOperands({4})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
    })
  ),

  // Note that for LHS we can only support add.
  HloMatcherPattern(
    PatternType("multi_update_lhs_with_reshape"),
    PatternMetaTarget(2),
    PatternInputs({5, 6, 7}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({1, 5})},
      {HloOpcode::kReshape, NodeOperands({2})},
      {HloOpcode::kCustomCall, NodeOperands({3, 6, 7}), IsPoplarInstruction(PoplarOp::MultiUpdate)},
      {HloOpcode::kBroadcast, NodeOperands({4})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
    })
  ),

  HloMatcherPattern(
    PatternType("multi_update_add_lhs_with_reshape"),
    PatternMetaTarget(1),
    PatternInputs({5, 6, 7, 8}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({1, 5})},
      {HloOpcode::kReshape, NodeOperands({2})},
      {HloOpcode::kCustomCall, NodeOperands({3, 6, 7, 8}), IsPoplarInstruction(PoplarOp::MultiUpdateAdd)},
      {HloOpcode::kBroadcast, NodeOperands({4})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
    })
  ),
};
// clang-format on
};  // namespace

MultiUpdateApply::MultiUpdateApply(CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, false, true) {}

StatusOr<HloInstruction*> CreateNewMultiUpdate(
    HloInstruction* old_inst, HloInstruction* operand, HloInstruction* indices,
    HloInstruction* updates, absl::optional<HloInstruction*> opt_scale,
    bool negate_scale, const absl::optional<int64> shard) {
  HloComputation* comp = old_inst->parent();
  HloMultiUpdateInstruction* multi_update =
      Cast<HloMultiUpdateInstruction>(old_inst);

  if (!opt_scale) {
    // If there isn't a scale, then scale with 1.0 (this is to make multi update
    // work like multi update add).
    opt_scale = comp->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::One(operand->shape().element_type())));
  }

  HloInstruction* scale = *opt_scale;
  if (negate_scale) {
    scale = comp->AddInstruction(
        HloInstruction::CreateUnary(scale->shape(), HloOpcode::kNegate, scale));
  }

  // Create a new multi update add instruction.
  HloInstruction* new_multi_update_add = comp->AddInstruction(
      CreateMultiUpdateAdd(operand->shape(), {operand, indices, updates, scale},
                           multi_update->GetIndexVectorDimension(),
                           multi_update->GetUpdateSliceDimension(),
                           multi_update->GetSerializationFactor()));

  if (shard) {
    new_multi_update_add->set_device_sharding(*shard);
  }
  multi_update->SetupDerivedInstruction(new_multi_update_add);
  return new_multi_update_add;
}

Status HandleNoReshape(HloMatcherMatched& match,
                       const absl::optional<int64> shard) {
  // Get the inputs.
  HloComputation* comp = match.computation;
  HloInstruction* root = match.instruction_mapping.at(0);
  HloInstruction* multi_update = match.instruction_mapping.at(1);
  HloInstruction* operand = match.instruction_mapping.at(4);
  HloInstruction* indicies = match.instruction_mapping.at(5);
  HloInstruction* updates = match.instruction_mapping.at(6);
  absl::optional<HloInstruction*> scale;
  if (IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(multi_update)) {
    scale = match.instruction_mapping.at(7);
  }

  // Create the new instruction and remove the old one.
  const bool negate_scale = root->opcode() == HloOpcode::kSubtract;
  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_multi_update_add,
      CreateNewMultiUpdate(multi_update, operand, indicies, updates, scale,
                           negate_scale, shard));

  return comp->ReplaceInstruction(root, new_multi_update_add);
}

Status HandleReshape(HloMatcherMatched& match,
                     const absl::optional<int64> shard) {
  // Get the inputs.
  HloComputation* comp = match.computation;
  HloInstruction* root = match.instruction_mapping.at(0);
  HloInstruction* reshape = match.instruction_mapping.at(1);
  HloInstruction* multi_update = match.instruction_mapping.at(2);
  HloInstruction* operand = match.instruction_mapping.at(5);
  HloInstruction* indicies = match.instruction_mapping.at(6);
  HloInstruction* updates = match.instruction_mapping.at(7);
  absl::optional<HloInstruction*> scale;
  if (IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(multi_update)) {
    scale = match.instruction_mapping.at(8);
  }

  // Reshape the operand to match the update shape - note that adding reshapes
  // here is fine.
  operand = comp->AddInstruction(
      HloInstruction::CreateReshape(reshape->operand(0)->shape(), operand));

  // Create the new instruction and remove the old one.
  const bool negate_scale = root->opcode() == HloOpcode::kSubtract;
  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_multi_update_add,
      CreateNewMultiUpdate(multi_update, operand, indicies, updates, scale,
                           negate_scale, shard));

  // Reshape the multi update add to match the reshape.
  new_multi_update_add = comp->AddInstruction(
      HloInstruction::CreateReshape(reshape->shape(), new_multi_update_add));

  return comp->ReplaceInstruction(root, new_multi_update_add);
}

bool MultiUpdateApply::HandleMatch(HloMatcherMatched& match,
                                   const absl::optional<int64> shard) {
  Status s;
  switch (match.pattern_idx) {
    case 0:
    case 1:
    case 2:
    case 3: {
      s = HandleNoReshape(match, shard);
      break;
    }
    case 4:
    case 5:
    case 6:
    case 7: {
      s = HandleReshape(match, shard);
      break;
    }
    default: {
      s = InternalError("Invalid pattern index.");
      break;
    }
  }

  if (!s.ok()) {
    LOG(FATAL) << s;
  }
  return true;
}

}  // namespace poplarplugin
}  // namespace xla
