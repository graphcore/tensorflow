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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/scaled_inplace.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/xla/literal_util.h"

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

StatusOr<PatternInstructionOutputs> CreateScaledInplaceaXbYFromMatch(
    const HloMatcherMatched& matched) {
  const auto& inputs = matched.GetInputs();
  const auto& outputs = matched.GetOutputs();
  CHECK_EQ(inputs.size(), 4);
  CHECK_EQ(outputs.size(), 1);
  return PatternInstructionOutputs{
      matched.computation->AddInstruction(CreateScaledInplaceaXbY(
          inputs[0], inputs[1], inputs[2], inputs[3], outputs[0]->opcode()))};
}

StatusOr<PatternInstructionOutputs> CreateScaledInplaceaXYFromMatch(
    const HloMatcherMatched& matched) {
  const auto& inputs = matched.GetInputs();
  const auto& outputs = matched.GetOutputs();
  HloComputation* comp = matched.computation;
  CHECK_EQ(inputs.size(), 3);
  CHECK_EQ(outputs.size(), 1);

  // Scale of Y is 1.
  HloInstruction* one = comp->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::One(inputs[0]->shape().element_type())));

  return PatternInstructionOutputs{comp->AddInstruction(CreateScaledInplaceaXbY(
      inputs[0], inputs[1], inputs[2], one, outputs[0]->opcode()))};
}

StatusOr<PatternInstructionOutputs> CreateScaledInplaceXbYFromMatch(
    const HloMatcherMatched& matched) {
  const auto& inputs = matched.GetInputs();
  const auto& outputs = matched.GetOutputs();
  CHECK_EQ(inputs.size(), 3);
  CHECK_EQ(outputs.size(), 1);
  return PatternInstructionOutputs{
      matched.computation->AddInstruction(CreateScaledInplaceXbY(
          inputs[0], inputs[1], inputs[2], outputs[0]->opcode()))};
}
}  // namespace

// clang-format off
static const std::vector<HloMatcherPattern> patterns = {
  HloMatcherPattern(
    PatternType("relu"),
    PatternReplaceFn([](const HloMatcherMatched& matched)
    -> StatusOr<PatternInstructionOutputs> {
      auto inputs = matched.GetInputs();
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
    PatternReplaceFn([](const HloMatcherMatched& matched)
    -> StatusOr<PatternInstructionOutputs> {
      auto inputs = matched.GetInputs();
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
      // NOLINTNEXTLINE
      {HloOpcode::kCompare, NodeOperands({5, 2}), IsCompareGreaterOrGreaterEqual},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
    })),

  // Scaled add/subtract to/from - X = a * X +/-  b * Y (mixed precision)
  HloMatcherPattern(
    PatternType("scaled_inplace_axby"),
    PatternReplaceFn(CreateScaledInplaceaXbYFromMatch),
    PatternMetaTarget(0),
    PatternInputs({7, 8, 9, 10}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({1, 2}), IsAdd},
      {HloOpcode::kMultiply, NodeOperands({7, 3})},
      {HloOpcode::kMultiply, NodeOperands({8, 4})},
      {HloOpcode::kBroadcast, NodeOperands({5})},
      {HloOpcode::kBroadcast, NodeOperands({6})},
      {HloOpcode::kConvert, NodeOperands({9}), IsF32ToF16Convert},
      {HloOpcode::kConvert, NodeOperands({10}), IsF32ToF16Convert},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsScalar},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsScalar}
    })),

  // Scaled add/subtract to/from - X = a * X +/-  b * Y (mixed precision with
  // reshape)
  HloMatcherPattern(
    PatternType("scaled_inplace_axby"),
    PatternReplaceFn(CreateScaledInplaceaXbYFromMatch),
    PatternMetaTarget(0),
    PatternInputs({9, 10, 11, 12}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({1, 2}), IsAdd},
      {HloOpcode::kMultiply, NodeOperands({9, 3})},
      {HloOpcode::kMultiply, NodeOperands({10, 4})},
      {HloOpcode::kBroadcast, NodeOperands({5})},
      {HloOpcode::kBroadcast, NodeOperands({6})},
      {HloOpcode::kReshape, NodeOperands({7}), IsScalar},
      {HloOpcode::kReshape, NodeOperands({8}), IsScalar},
      {HloOpcode::kConvert, NodeOperands({11}), IsF32ToF16Convert},
      {HloOpcode::kConvert, NodeOperands({12}), IsF32ToF16Convert},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})}
    })),

  // Scaled add/subtract to/from - X = a * X +/-  b * Y
  HloMatcherPattern(
    PatternType("scaled_inplace_axby"),
    PatternReplaceFn(CreateScaledInplaceaXbYFromMatch),
    PatternMetaTarget(0),
    PatternInputs({5, 6, 7, 8}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({1, 2}), IsAddOrSubtract},
      {HloOpcode::kMultiply, NodeOperands({5, 3})},
      {HloOpcode::kMultiply, NodeOperands({6, 4})},
      {HloOpcode::kBroadcast, NodeOperands({7})},
      {HloOpcode::kBroadcast, NodeOperands({8})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsScalar},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsScalar}
    })),

  // Scaled add/subtract to/from - X = a * X +/- Y (mixed precision)
  HloMatcherPattern(
    PatternType("scaled_inplace_axy"),
    PatternReplaceFn(CreateScaledInplaceaXYFromMatch),
    PatternMetaTarget(0),
    PatternInputs({4, 5, 6}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({1, 5}), IsAdd},
      {HloOpcode::kMultiply, NodeOperands({4, 2})},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kConvert, NodeOperands({6}), IsF32ToF16Convert},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsScalar}
    })),

  // Scaled add/subtract to/from - X = a * X +/- Y (mixed precision with
  // reshape)
  HloMatcherPattern(
    PatternType("scaled_inplace_axy"),
    PatternReplaceFn(CreateScaledInplaceaXYFromMatch),
    PatternMetaTarget(0),
    PatternInputs({5, 6, 7}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({1, 6}), IsAdd},
      {HloOpcode::kMultiply, NodeOperands({5, 2})},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kReshape, NodeOperands({4}), IsScalar},
      {HloOpcode::kConvert, NodeOperands({7}), IsF32ToF16Convert},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})}
    })),

  // Scaled add/subtract to/from - X = a * X +/- Y
  HloMatcherPattern(
    PatternType("scaled_inplace_axy"),
    PatternReplaceFn(CreateScaledInplaceaXYFromMatch),
    PatternMetaTarget(0),
    PatternInputs({3, 4, 5}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({1, 4}), IsAddOrSubtract},
      {HloOpcode::kMultiply, NodeOperands({3, 2})},
      {HloOpcode::kBroadcast, NodeOperands({5})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsScalar}
    })),

  // Scaled add/subtract to/from - X = X +/- b * Y (mixed precision)
  HloMatcherPattern(
    PatternType("scaled_inplace_xby"),
    PatternReplaceFn(CreateScaledInplaceXbYFromMatch),
    PatternMetaTarget(0),
    PatternInputs({4, 5, 6}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({4, 1}), IsAdd},
      {HloOpcode::kMultiply, NodeOperands({5, 2})},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kConvert, NodeOperands({6}), IsF32ToF16Convert},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsScalar}
    })),

  // Scaled add/subtract to/from - X = X +/- b * Y (mixed precision with
  // reshape)
  HloMatcherPattern(
    PatternType("scaled_inplace_xby"),
    PatternReplaceFn(CreateScaledInplaceXbYFromMatch),
    PatternMetaTarget(0),
    PatternInputs({5, 6, 7}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({5, 1}), IsAdd},
      {HloOpcode::kMultiply, NodeOperands({6, 2})},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kReshape, NodeOperands({4}), IsScalar},
      {HloOpcode::kConvert, NodeOperands({7}), IsF32ToF16Convert},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})}
    })),

  // Scaled add/subtract to/from - X = X +/- b * Y
  HloMatcherPattern(
    PatternType("scaled_inplace_xby"),
    PatternReplaceFn(CreateScaledInplaceXbYFromMatch),
    PatternMetaTarget(0),
    PatternInputs({3, 4, 5}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({3, 1}), IsAddOrSubtract},
      {HloOpcode::kMultiply, NodeOperands({4, 2})},
      {HloOpcode::kBroadcast, NodeOperands({5})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsScalar}
    })),
};
// clang-format on

FuseOpsIntoPoplarOps::FuseOpsIntoPoplarOps(
    struct CompilerAnnotations& annotations)
    : SingleHloMatcher(annotations, patterns, "_pop_op_",
                       /*restart_search_after_match=*/false) {}

}  // namespace poplarplugin
}  // namespace xla
