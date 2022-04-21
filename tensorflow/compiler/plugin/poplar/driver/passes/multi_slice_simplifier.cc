/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_slice_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace poplarplugin {

static bool IsVector(const HloInstruction* instr) {
  const Shape& shape = instr->shape();
  const absl::Span<const int64> dimensions = shape.dimensions();
  const int dimensions_size = shape.dimensions_size();
  bool result = false;
  result |= dimensions_size == 1;                        // shape = [N]
  result |= dimensions_size == 2 && dimensions[1] == 1;  // shape = [N, 1]
  return result;
}

// clang-format off
static const std::vector<HloMatcherPattern> patterns = {
  HloMatcherPattern(
    PatternType("multi_slice_const_indices"),
    PatternMetaTarget(0),
    PatternInputs({1, 2}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kCustomCall, NodeOperands({1, 2}), IsMultiSlice},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloOpcode::kConstant, NodeOperands({}), IsVector},
    })),
  HloMatcherPattern(
    PatternType("multi_update_add_const_indices"),
    PatternMetaTarget(0),
    PatternInputs({1, 2, 3, 4}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kCustomCall, NodeOperands({1, 2, 3, 4}), IsMultiUpdateAdd},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloOpcode::kConstant, NodeOperands({}), IsVector},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
    })),
};
// clang-format on

static StatusOr<bool> HandleMultiSliceConstIndices(
    HloMatcherMatched& match, const absl::optional<int64> sharding_device) {
  HloComputation* comp = match.computation;

  HloMultiSliceInstruction* slice =
      Cast<HloMultiSliceInstruction>(match.instruction_mapping.at(0));
  HloInstruction* inputs = match.GetInputs()[0];
  HloInstruction* indices = match.GetInputs()[1];

  // Flatten the indices literal.
  Literal indices_literal = indices->literal().Clone();
  const int64 element_count = indices_literal.element_count();
  TF_ASSIGN_OR_RETURN(indices_literal,
                      indices_literal.Reshape({element_count}));
  TF_ASSIGN_OR_RETURN(std::vector<int64> static_indices,
                      LiteralVectorToNativeType<int64>(indices_literal));

  HloInstruction* new_slice = comp->AddInstruction(CreateStaticMultiSlice(
      slice->shape(), inputs, std::move(static_indices)));

  slice->SetupDerivedInstruction(new_slice);

  TF_RETURN_IF_ERROR(comp->ReplaceInstruction(slice, new_slice));

  return true;
}

static StatusOr<bool> HandleMultiUpdateAddConstIndices(
    HloMatcherMatched& match, const absl::optional<int64> sharding_device) {
  HloComputation* comp = match.computation;

  HloMultiUpdateAddInstruction* update_add =
      Cast<HloMultiUpdateAddInstruction>(match.instruction_mapping.at(0));
  HloInstruction* inputs = match.GetInputs()[0];
  HloInstruction* indices = match.GetInputs()[1];
  HloInstruction* updates = match.GetInputs()[2];
  HloInstruction* scale = match.GetInputs()[3];

  // Flatten the indices literal.
  Literal indices_literal = indices->literal().Clone();
  const int64 element_count = indices_literal.element_count();
  TF_ASSIGN_OR_RETURN(indices_literal,
                      indices_literal.Reshape({element_count}));
  TF_ASSIGN_OR_RETURN(std::vector<int64> static_indices,
                      LiteralVectorToNativeType<int64>(indices_literal));

  HloInstruction* new_update_add = comp->AddInstruction(
      CreateStaticMultiUpdateAdd(update_add->shape(), {inputs, updates, scale},
                                 std::move(static_indices)));

  update_add->SetupDerivedInstruction(new_update_add);

  TF_RETURN_IF_ERROR(comp->ReplaceInstruction(update_add, new_update_add));

  return true;
}

MultiSliceSimplifier::MultiSliceSimplifier(
    struct CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, /*root_only=*/false,
                 /*requires_unique_sharding=*/true) {}

StatusOr<bool> MultiSliceSimplifier::HandleMatch(
    HloMatcherMatched& match, const absl::optional<int64> sharding_device) {
  switch (match.pattern_idx) {
    case 0: {
      return HandleMultiSliceConstIndices(match, sharding_device);
    }
    case 1: {
      return HandleMultiUpdateAddConstIndices(match, sharding_device);
    }
    default: { return xla::FailedPrecondition(""); }
  }
}

}  // namespace poplarplugin
}  // namespace xla
