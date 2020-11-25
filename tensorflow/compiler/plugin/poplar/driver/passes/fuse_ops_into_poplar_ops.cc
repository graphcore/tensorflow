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

#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_into_poplar_ops.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/non_linearity.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {
namespace {

// Even though comment in relu_op.cc says:
// >Return the lhs (incoming gradient) if the rhs (input feature) > 0,
// >otherwise return 0.
// There's compare(direction=GE) instruction generated.
//
// However It's still safe to accept both G and GE comparison directions.
// In case of input feature equals to zero, we either:
//  - return input feature (0) for GE direction,
//  - return zero for G direction.

bool IsCompareGreaterOrGreaterEqual(const HloInstruction* inst) {
  return IsCompareGreater(inst) || IsCompareGreaterOrEqual(inst);
}
}  // namespace

// clang-format off
static const std::vector<HloMatcherPattern> patterns = {
  HloMatcherPattern(
    PatternType("relu"),
    PatternReplaceFn([](const HloMatcherPattern & pattern,
    const HloMatcherMatched& matched)
    -> StatusOr<PatternInstructionOutputs> {
      auto inputs = matched.GetInputs(pattern);
      if (inputs.size() != 1) {
          return InternalError("Relu accepts only one argument");
      }
      return PatternInstructionOutputs {
          matched.computation->AddInstruction(
          CreateRelu(inputs[0]))
      };
    }),
    PatternMetaTarget(0),
    PatternInputs({1}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kMaximum, NodeOperands({1, 2}), IsF16OrF32},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
    })),

  HloMatcherPattern(
    PatternType("relu_grad"),
    PatternReplaceFn([](const HloMatcherPattern & pattern,
    const HloMatcherMatched& matched)
    -> StatusOr<PatternInstructionOutputs> {
      auto inputs = matched.GetInputs(pattern);
      if (inputs.size() != 2) {
        return InternalError("ReluGrad accepts exactly two arguments");
      }
      return PatternInstructionOutputs {
        matched.computation->AddInstruction(
          CreateReluGrad(inputs[0], inputs[1]))
      };
    }),
    PatternMetaTarget(0),
    PatternInputs({5, 4}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kSelect, NodeOperands({1, 4, 2}), IsF16OrF32},
      {HloOpcode::kCompare, NodeOperands({5, 2}),
                                  IsCompareGreaterOrGreaterEqual},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
    })),
};
// clang-format on

FuseOpsIntoPoplarOps::FuseOpsIntoPoplarOps(
    struct CompilerAnnotations& annotations)
    : SingleHloMatcher(annotations, patterns, "_pop_op_") {}

}  // namespace poplarplugin
}  // namespace xla
