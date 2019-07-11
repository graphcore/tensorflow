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

#include "tensorflow/compiler/plugin/poplar/driver/passes/scatter_combiner.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

namespace {
bool ScatterSupportsConcats(const HloInstruction* inst) {
  return IsMultiUpdate(inst) || IsMultiUpdateAdd(inst);
}

// clang-format off
static const std::vector<HloMatcherPattern> patterns = {
  HloMatcherPattern(
    PatternType("reduction_no_convert"),
    PatternMetaTarget(1),
    PatternInputs({4}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({1, 2})},
      {HloOpcode::kScatter, NodeOperands({3, 5, 6}), ScatterSupportsConcats},
      {HloOpcode::kScatter, NodeOperands({3, 7, 8}), ScatterSupportsConcats},
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

Shape GetConcatenatedShape(std::vector<HloInstruction*> insts,
                           const int64 dimension) {
  std::vector<const Shape*> inst_shapes;
  absl::c_transform(insts, std::back_inserter(inst_shapes),
                    [](HloInstruction* inst) { return &inst->shape(); });
  auto statusor = ShapeInference::InferConcatOpShape(inst_shapes, dimension);
  if (!statusor.ok()) {
    LOG(FATAL) << "Failed concatentating shapes together.";
  }
  return statusor.ValueOrDie();
}
}  // namespace
ScatterCombiner::ScatterCombiner(struct CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, false, true) {}

bool ScatterCombiner::HandleMatch(HloMatcherMatched& match,
                                  const absl::optional<int64> sharding_device) {
  // We currently only expect a single pattern.
  CHECK_EQ(match.pattern_idx, 0);
  auto pattern = patterns_[match.pattern_idx];
  HloComputation* computation = match.computation;
  HloInstruction* pattern_root =
      match.instruction_mapping[pattern.GetOutputs()[0]];
  // Get the scatters.
  HloScatterInstruction* scatter1 =
      Cast<HloScatterInstruction>(match.instruction_mapping.at(1));
  HloScatterInstruction* scatter2 =
      Cast<HloScatterInstruction>(match.instruction_mapping.at(2));
  // Skip this match if the scatters are not identical ignoring the operands
  // (note that we have already made sure that operand 0 is identical for both).
  auto compare_operands = [](const HloInstruction*, const HloInstruction*) {
    return true;
  };
  auto compare_computations = [](const HloComputation* comp_a,
                                 const HloComputation* comp_b) {
    return comp_a == comp_b || *comp_a == *comp_b;
  };
  if (!scatter1->Identical(*scatter2, compare_operands, compare_computations)) {
    return false;
  }
  // Get the dimenstions to concatentate on.
  auto dim_numbers = scatter1->scatter_dimension_numbers();
  CHECK_EQ(dim_numbers.update_window_dims().size(), 1);
  CHECK(dim_numbers.update_window_dims()[0] < 2);
  const int64 updates_conctat_dim =
      (dim_numbers.update_window_dims()[0] + 1) % 2;

  CHECK_EQ(dim_numbers.inserted_window_dims().size(), 1);
  CHECK_EQ(dim_numbers.inserted_window_dims()[0], 0);
  const int64 indices_concat_dim = 0;

  // Concat the indices and updates.
  std::vector<HloInstruction*> indices = {scatter1->mutable_operand(1),
                                           scatter2->mutable_operand(1)};
  HloInstruction* new_indices =
      computation->AddInstruction(HloInstruction::CreateConcatenate(
          GetConcatenatedShape(indices, indices_concat_dim), indices,
          indices_concat_dim));
  std::vector<HloInstruction*> updates = {scatter1->mutable_operand(2),
                                          scatter2->mutable_operand(2)};
  HloInstruction* new_update =
      computation->AddInstruction(HloInstruction::CreateConcatenate(
          GetConcatenatedShape(updates, updates_conctat_dim), updates,
          updates_conctat_dim));

  // Create scatter with the new operands, but same other options.
  HloInstruction* new_scatter =
      computation->AddInstruction(scatter1->CloneWithNewOperands(
          scatter1->shape(),
          {scatter1->mutable_operand(0), new_indices, new_update}));
  computation->ReplaceInstruction(pattern_root, new_scatter);
  // Set the sharding device if there is one.
  if (sharding_device) {
    new_indices->set_device_sharding(*sharding_device);
    new_update->set_device_sharding(*sharding_device);
    new_scatter->set_device_sharding(*sharding_device);
  }
  return true;
}

}  // namespace poplarplugin
}  // namespace xla
