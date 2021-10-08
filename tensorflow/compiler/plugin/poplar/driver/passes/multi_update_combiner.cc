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

#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_update_combiner.h"

#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

namespace {
// clang-format off
static const std::vector<HloMatcherPattern> patterns = {
  // We can combine two multi updates into a multi update add with a scale
  // factor 1.
  HloMatcherPattern(
    PatternType("multi_update_multi_update"),
    PatternMetaTarget(1),
    PatternInputs({5, 6, 7, 8}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({1, 2})},
      {HloOpcode::kCustomCall, NodeOperands({3, 5, 6}), IsPoplarInstruction(PoplarOp::MultiUpdate)},
      {HloOpcode::kCustomCall, NodeOperands({3, 7, 8}), IsPoplarInstruction(PoplarOp::MultiUpdate)},
      {HloOpcode::kBroadcast, NodeOperands({4})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
    })
  ),

  // We can combine a multi update and a multi update add with a scale factor
  // one into a multi update add with a scale factor one.
  HloMatcherPattern(
    PatternType("multi_update_multi_update_add"),
    PatternMetaTarget(1),
    PatternInputs({5, 6, 7, 8}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({1, 2})},
      {HloOpcode::kCustomCall, NodeOperands({3, 5, 6}), IsPoplarInstruction(PoplarOp::MultiUpdate)},
      {HloOpcode::kCustomCall, NodeOperands({3, 7, 8, 9}), IsPoplarInstruction(PoplarOp::MultiUpdateAdd)},
      {HloOpcode::kBroadcast, NodeOperands({4})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantOne},
    })
  ),

  // We can combine a multi update and a multi update add with a scale factor
  // one into a multi update add with a scale factor one.
  HloMatcherPattern(
    PatternType("multi_update_add_multi_update"),
    PatternMetaTarget(1),
    PatternInputs({5, 6, 7, 8}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({1, 2})},
      {HloOpcode::kCustomCall, NodeOperands({3, 5, 6, 9}), IsPoplarInstruction(PoplarOp::MultiUpdateAdd)},
      {HloOpcode::kCustomCall, NodeOperands({3, 7, 8}), IsPoplarInstruction(PoplarOp::MultiUpdate)},
      {HloOpcode::kBroadcast, NodeOperands({4})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantOne},
    })
  ),

  // We can combine two multi update adds with a scale factor one into a multi
  // update add with a scale factor one.
  HloMatcherPattern(
    PatternType("multi_update_add_multi_update_add"),
    PatternMetaTarget(1),
    PatternInputs({6, 7, 8, 9}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({1, 2})},
      {HloOpcode::kCustomCall, NodeOperands({3, 6, 7, 5}), IsPoplarInstruction(PoplarOp::MultiUpdateAdd)},
      {HloOpcode::kCustomCall, NodeOperands({3, 8, 9, 5}), IsPoplarInstruction(PoplarOp::MultiUpdateAdd)},
      {HloOpcode::kBroadcast, NodeOperands({4})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantOne},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
    })
  ),
};
// clang-format on

uint64 GetIndexDimSize(const HloMultiUpdateInstruction* inst) {
  const Shape& shape = inst->operand(1)->shape();
  return shape.dimensions(shape.rank() - 1);
}

uint64 GetUpdateDimSize(const HloMultiUpdateInstruction* inst) {
  const Shape& shape = inst->operand(2)->shape();
  return shape.dimensions(shape.rank() - 1);
}

bool IndicesCastable(const HloInstruction* inst) {
  switch (inst->shape().element_type()) {
    case PrimitiveType::S32:
    case PrimitiveType::S64:
    case PrimitiveType::U32:
    case PrimitiveType::U64:
      return true;
    default:
      return false;
  }
}

bool IndicesCastable(const HloInstruction* inst1, const HloInstruction* inst2) {
  return IndicesCastable(inst1) && IndicesCastable(inst2);
}

}  // namespace

MultiUpdateCombiner::MultiUpdateCombiner(
    struct CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, false, true) {}

StatusOr<bool> MultiUpdateCombiner::HandleMatch(
    HloMatcherMatched& match, const absl::optional<int64> sharding_device) {
  const auto& pattern = match.pattern;
  HloComputation* computation = match.computation;
  HloInstruction* pattern_root =
      match.instruction_mapping[pattern.GetOutputs()[0]];
  // Get the multi updates.
  HloMultiUpdateInstruction* multi_update1 =
      Cast<HloMultiUpdateInstruction>(match.instruction_mapping.at(1));
  const uint64 multi_update1_update_dim_size = GetUpdateDimSize(multi_update1);

  HloMultiUpdateInstruction* multi_update2 =
      Cast<HloMultiUpdateInstruction>(match.instruction_mapping.at(2));
  const uint64 multi_update2_update_dim_size = GetUpdateDimSize(multi_update2);
  // Check that the ops are compatible.
  if (multi_update1_update_dim_size != multi_update2_update_dim_size) {
    return false;
  }

  // Check the indices are the same type, or castable to the same type
  HloInstruction* indices1 = multi_update1->mutable_operand(1);
  HloInstruction* indices2 = multi_update2->mutable_operand(1);
  PrimitiveType indices2_type = indices2->shape().element_type();
  if (indices1->shape().element_type() != indices2_type) {
    if (IndicesCastable(indices1, indices2)) {
      Shape indices1_new_shape = indices1->shape();
      indices1_new_shape.set_element_type(indices2_type);
      indices1 = computation->AddInstruction(
          HloInstruction::CreateConvert(indices1_new_shape, indices1));
    } else {
      return false;
    }
  }

  // Concat the indices.
  std::vector<HloInstruction*> indices = {indices1, indices2};
  HloInstruction* new_indices =
      computation->AddInstruction(HloInstruction::CreateConcatenate(
          GetConcatenatedShape(indices, 0), indices, 0));
  indices1->SetupDerivedInstruction(new_indices);

  // Concat the updates.
  HloInstruction* updates1 = multi_update1->mutable_operand(2);
  HloInstruction* updates2 = multi_update2->mutable_operand(2);
  std::vector<HloInstruction*> updates = {updates1, updates2};
  HloInstruction* new_updates =
      computation->AddInstruction(HloInstruction::CreateConcatenate(
          GetConcatenatedShape(updates, 0), updates, 0));
  updates1->SetupDerivedInstruction(new_updates);

  // Create new multi update add with scale 1.
  HloInstruction* one =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::One(multi_update1->shape().element_type())));
  HloInstruction* new_multi_update =
      computation->AddInstruction(CreateMultiUpdateAdd(
          multi_update1->shape(),
          {multi_update1->mutable_operand(0), new_indices, new_updates, one},
          std::max(multi_update1->GetSerializationFactor(),
                   multi_update2->GetSerializationFactor())));
  multi_update1->SetupDerivedInstruction(new_multi_update);
  computation->ReplaceInstruction(pattern_root, new_multi_update);

  // Set the sharding device if there is one.
  if (sharding_device) {
    new_indices->set_sharding(HloSharding::AssignDevice(*sharding_device));
    new_updates->set_sharding(HloSharding::AssignDevice(*sharding_device));
    new_multi_update->set_sharding(HloSharding::AssignDevice(*sharding_device));
  }
  return true;
}

}  // namespace poplarplugin
}  // namespace xla
