/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_early.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/conv_with_reverse.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {
namespace {

StatusOr<PatternInstructionOutputs> CreateConvWithReverseFromMatch(
    const HloMatcherMatched& matched) {
  const auto& inputs = matched.GetInputs();
  const auto& outputs = matched.GetOutputs();
  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(outputs.size(), 1);

  const auto* original_conv = outputs[0];

  return PatternInstructionOutputs{
      matched.computation->AddInstruction(CreateConvWithReverse(
          original_conv->shape(), inputs[0], inputs[1],
          original_conv->feature_group_count(),
          original_conv->batch_group_count(), original_conv->window(),
          original_conv->convolution_dimension_numbers(),
          original_conv->precision_config()))};
}
}  // namespace

/*
 * Note about constructing these patterns.  Due to the behaviour of the fuser
 * there must be no backward references.  All nodes should appear after any
 * other nodes that refer to them.
 *
 * NOTE: Highest match priority is nearer the top of the list
 */

// clang-format off
static const std::vector<HloMatcherPattern> patterns = {

  // Conv{2,3}DBackpropInput
  HloMatcherPattern(
    PatternType("conv_with_reverse"),
    PatternReplaceFn(CreateConvWithReverseFromMatch),
    PatternMetaTarget(0),
    PatternInputs({2, 3}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kConvolution, NodeOperands({2, 1}), IsOpWithWindowNoStride},
      {HloOpcode::kReverse, NodeOperands({3}), IsConvFilterTranspose},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})}
    })
  ),
};
// clang-format on

FuseOpsEarly::FuseOpsEarly(struct CompilerAnnotations& annotations)
    : SingleHloMatcher(annotations, patterns, "_pop_op_") {}

}  // namespace poplarplugin
}  // namespace xla
