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

#include "tensorflow/compiler/plugin/poplar/driver/passes/redundant_triangular_mask_remover.h"

#include <algorithm>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {
namespace {

bool IsTriangularMask(HloMatcherMatched& match) {
  const auto& pattern = match.pattern;

  // Get the compare op.
  HloCompareInstruction* compare_inst =
      Cast<HloCompareInstruction>(match.instruction_mapping.at(1));
  // The compare op can have different numbers of operands depending on its
  // rank. Instead of matching them in the pattern, we check here that they are
  // all iota ops.
  for (HloInstruction* operand : compare_inst->operands()) {
    if (operand->opcode() != HloOpcode::kIota) {
      return false;
    }
  }

  // Get the triangular shape op.
  const HloInstruction* triangular_shape_inst =
      match.instruction_mapping[pattern.GetMetaTarget()];
  // Get the select op.
  HloInstruction* select_inst =
      match.instruction_mapping[pattern.GetOutputs()[0]];
  // Check whether the triangular shape op is first or second in the select.
  bool first_in_select = false;
  if (select_inst->operand(1) == triangular_shape_inst) {
    first_in_select = true;
  } else {
    // Sanity check that it is second in the select if it isn't first.
    CHECK(select_inst->operand(2) == triangular_shape_inst);
  }
  // Check the comparison direction matches the lower flag on the triangular
  // shape op, and the order of the select operands.
  bool lower = false;
  if (triangular_shape_inst->opcode() == HloOpcode::kCholesky) {
    lower = Cast<HloCholeskyInstruction>(triangular_shape_inst)
                ->cholesky_options()
                .lower();
  } else {
    CHECK(triangular_shape_inst->opcode() == HloOpcode::kTriangularSolve);
    lower = Cast<HloTriangularSolveInstruction>(triangular_shape_inst)
                ->triangular_solve_options()
                .lower();
  }
  switch (compare_inst->comparison_direction()) {
    case ComparisonDirection::kGt:
    case ComparisonDirection::kGe:
      if (lower != first_in_select) {
        return false;
      }
      break;
    case ComparisonDirection::kLt:
    case ComparisonDirection::kLe:
      if (lower == first_in_select) {
        return false;
      }
      break;
    default:
      return false;
  }

  return true;
}
}  // namespace

// clang-format off
static const std::vector<HloMatcherPattern> patterns = {
  HloMatcherPattern(
    PatternType("triangular_mask"),
    PatternMetaTarget(2),
    PatternInputs({}),
    PatternOutputs({0}),
    Pattern({
      // NOLINTNEXTLINE
      {HloOpcode::kSelect, NodeOperands({1, 2, 3})},
      {HloOpcode::kCompare, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsTriangularShapeInst},
      {HloOpcode::kBroadcast, NodeOperands({4})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
    })),

  HloMatcherPattern(
    PatternType("triangular_mask"),
    PatternMetaTarget(2),
    PatternInputs({}),
    PatternOutputs({0}),
    Pattern({
      // NOLINTNEXTLINE
      {HloOpcode::kSelect, NodeOperands({1, 3, 2})},
      {HloOpcode::kCompare, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsTriangularShapeInst},
      {HloOpcode::kBroadcast, NodeOperands({4})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
    })),
};
// clang-format on

RedundantTriangularMaskRemover::RedundantTriangularMaskRemover(
    struct CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, /*root_only=*/false,
                 /*requires_unique_sharding=*/true) {}

StatusOr<bool> RedundantTriangularMaskRemover::HandleMatch(
    HloMatcherMatched& match, const absl::optional<int64_t>) {
  if (!IsTriangularMask(match)) {
    return false;
  }

  const auto& pattern = match.pattern;
  HloComputation* comp = match.computation;

  // Get the select inst which applies the mask.
  HloInstruction* select_inst =
      match.instruction_mapping[pattern.GetOutputs()[0]];
  // Get the triangular shape op.
  HloInstruction* triangular_shape_inst =
      match.instruction_mapping[pattern.GetMetaTarget()];

  // Replace select with unmasked triangular shape op.
  TF_RETURN_IF_ERROR(select_inst->ReplaceAllUsesWith(triangular_shape_inst));
  if (comp->root_instruction() == select_inst) {
    comp->set_root_instruction(triangular_shape_inst);
  }
  TF_RETURN_IF_ERROR(comp->RemoveInstructionAndUnusedOperands(select_inst));

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
