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

#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_slice_combiner.h"
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
  // We can combine two multi slices sharing the same input into a single multi slice
  HloMatcherPattern(
    PatternType("multi_slice_multi_slice"),
    PatternMetaTarget(0),
    PatternInputs({2,3,4}),
    PatternOutputs({0,1}),
    Pattern({
      {HloOpcode::kCustomCall, NodeOperands({2, 3}), IsPoplarInstruction(PoplarOp::MultiSlice)},
      {HloOpcode::kCustomCall, NodeOperands({2, 4}), IsPoplarInstruction(PoplarOp::MultiSlice)},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
    })
  )
};
// clang-format on

}  // namespace

MultiSliceCombiner::MultiSliceCombiner(struct CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, false, true) {}

StatusOr<bool> MultiSliceCombiner::HandleMatch(
    HloMatcherMatched& match, const absl::optional<int64> sharding_device) {
  const auto& pattern = match.pattern;
  HloComputation* computation = match.computation;
  // Get the multi slices.
  HloMultiSliceInstruction* multi_slice1 =
      Cast<HloMultiSliceInstruction>(match.instruction_mapping.at(0));
  HloMultiSliceInstruction* multi_slice2 =
      Cast<HloMultiSliceInstruction>(match.instruction_mapping.at(1));

  HloInstruction* input = match.instruction_mapping.at(2);

  // Concat the indices.
  HloInstruction* indices1 = multi_slice1->mutable_operand(1);
  HloInstruction* indices2 = multi_slice2->mutable_operand(1);

  // Make sure the indices have been flattened.
  if (indices1->shape().rank() != 2 || indices2->shape().rank() != 2 ||
      indices1->shape().dimensions(1) != 1 ||
      indices2->shape().dimensions(1) != 1) {
    return false;
  }

  std::vector<HloInstruction*> indices = {indices1, indices2};
  HloInstruction* new_indices =
      computation->AddInstruction(HloInstruction::CreateConcatenate(
          GetConcatenatedShape(indices, 0), indices, 0));
  indices1->SetupDerivedInstruction(new_indices);

  // Create the combined multi slice.
  std::vector<HloInstruction*> multi_slices = {multi_slice1, multi_slice2};
  Shape output = GetConcatenatedShape(multi_slices, 0);

  HloInstruction* new_multi_slice =
      computation->AddInstruction(CreateMultiSlice(output, input, new_indices));
  multi_slice1->SetupDerivedInstruction(new_multi_slice);

  HloInstruction* output1 =
      computation->AddInstruction(HloInstruction::CreateSlice(
          multi_slice1->shape(), new_multi_slice, {0, 0},
          {multi_slice1->shape().dimensions(0), output.dimensions(1)}, {1, 1}));
  multi_slice1->SetupDerivedInstruction(output1);

  HloInstruction* output2 = computation->AddInstruction(
      HloInstruction::CreateSlice(multi_slice2->shape(), new_multi_slice,
                                  {multi_slice1->shape().dimensions(0), 0},
                                  output.dimensions(), {1, 1}));
  multi_slice2->SetupDerivedInstruction(output2);

  computation->ReplaceInstruction(multi_slice1, output1);
  computation->ReplaceInstruction(multi_slice2, output2);

  // Set the sharding device if there is one.
  if (sharding_device) {
    new_multi_slice->set_sharding(HloSharding::AssignDevice(*sharding_device));
    output1->set_sharding(HloSharding::AssignDevice(*sharding_device));
    output2->set_sharding(HloSharding::AssignDevice(*sharding_device));
    new_indices->set_sharding(HloSharding::AssignDevice(*sharding_device));
  }
  return true;
}

}  // namespace poplarplugin
}  // namespace xla
