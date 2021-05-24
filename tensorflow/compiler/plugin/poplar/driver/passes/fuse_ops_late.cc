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

#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_late.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {
namespace {
// Function for describing fusions which are "inplace" on the 0th input with no
// tuple shapes involved.
HloPoplarUseDescriptions GetSimpleInplaceUseDescription(
    const HloMatcherMatched&) {
  return {HloPoplarUseDescription{/*operand_number=*/0,
                                  /*operand_index=*/ShapeIndex{},
                                  /*output_index=*/ShapeIndex{},
                                  BufferUseKind::USE_ALIAS_READ_WRITE}};
}
};  // namespace

/*
 * Note about constructing these patterns.  Due to the behaviour of the fuser
 * there must be no backward references.  All nodes should appear after any
 * other nodes that refer to them.
 *
 * NOTE: Highest match priority is nearer the top of the list
 */

// clang-format off
static const std::vector<HloMatcherPattern> patterns = {
  // BiasAdd on convolution (w/ broadcast)
  HloMatcherPattern(
    PatternType("conv_biasadd"),
    PatternMetaTarget(0),
    PatternInputs({2, 3}),
    PatternOutputs({0}),
    PatternInplaceDescriptionFn(GetSimpleInplaceUseDescription),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({2, 1}), IsBiasAdd},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsPopOpsConvolution},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), Is1DVector}
    })),

  // BiasAdd on convolution (w/ broadcast)
  HloMatcherPattern(
    PatternType("conv_biasadd"),
    PatternMetaTarget(0),
    PatternInputs({2, 3}),
    PatternOutputs({0}),
    PatternInplaceDescriptionFn(GetSimpleInplaceUseDescription),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({2, 1}), IsBiasAdd},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kConvolution, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), Is1DVector}
    })),

  // BiasAdd on convolution (w/ reshape)
  HloMatcherPattern(
    PatternType("conv_biasadd"),
    PatternMetaTarget(0),
    PatternInputs({2, 3}),
    PatternOutputs({0}),
    PatternInplaceDescriptionFn(GetSimpleInplaceUseDescription),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({2, 1}), IsBiasAdd},
      {HloOpcode::kReshape, NodeOperands({3}), IsExpandingReshape},
      {HloOpcode::kConvolution, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), Is1DVector}
    })),

  // BiasAdd on a MatMul (w/ broadcast)
  HloMatcherPattern(
    PatternType("matmul_biasadd"),
    PatternMetaTarget(0),
    PatternInputs({2, 3}),
    PatternOutputs({0}),
    PatternInplaceDescriptionFn(GetSimpleInplaceUseDescription),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({2, 1}), IsBiasAdd},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kDot, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), Is1DVector}
    })),

  // External padding with constant zero
  HloMatcherPattern(
    PatternType("zero_pad"),
    PatternMetaTarget(0),
    PatternInputs({2}),
    PatternOutputs({0}),
    PatternInplaceDescriptionFn(GetSimpleInplaceUseDescription),
    Pattern({
      {HloOpcode::kPad, NodeOperands({2, 1}), IsExternalPadding},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})}
    })),

  // Random normal with post scale and add
  HloMatcherPattern(
    PatternType("norm_scale_add"),
    PatternMetaTarget(4),
    PatternInputs({}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({2, 1})},
      {HloOpcode::kConstant, NodeOperands({}), IsSingleElement},
      {HloOpcode::kMultiply, NodeOperands({4, 3}), HasSingleUser},
      {HloOpcode::kConstant, NodeOperands({}), IsSingleElement},
      {HloOpcode::kRng, NodeOperands({5, 6}), {IsRandomNormal, HasSingleUser}},
      {HloOpcode::kConstant, NodeOperands({}), IsSingleElement},
      {HloOpcode::kConstant, NodeOperands({}), IsSingleElement}
    })),

  // Random normal with broadcasted post scale and add
  HloMatcherPattern(
    PatternType("norm_scale_add"),
    PatternMetaTarget(6),
    PatternInputs({}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({3, 1})},
      {HloOpcode::kBroadcast, NodeOperands({2})},
      {HloOpcode::kConstant, NodeOperands({}), IsSingleElement},
      {HloOpcode::kMultiply, NodeOperands({6, 4}), HasSingleUser},
      {HloOpcode::kBroadcast, NodeOperands({5})},
      {HloOpcode::kConstant, NodeOperands({}), IsSingleElement},
      {HloOpcode::kRng, NodeOperands({7, 8}), {IsRandomNormal, HasSingleUser}},
      {HloOpcode::kConstant, NodeOperands({}), IsSingleElement},
      {HloOpcode::kConstant, NodeOperands({}), IsSingleElement}
    })),

  // Random uniform with post scale and add
  HloMatcherPattern(
    PatternType("uniform_scale_add"),
    PatternMetaTarget(4),
    PatternInputs({}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({2, 1})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kMultiply, NodeOperands({4, 3}), HasSingleUser},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kRng, NodeOperands({5, 6}), {IsRandomUniform, HasSingleUser}},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kConstant, NodeOperands({})}
    })),

  // Random uniform with broadcasted post scale and add
  HloMatcherPattern(
    PatternType("uniform_scale_add"),
    PatternMetaTarget(6),
    PatternInputs({}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({3, 1})},
      {HloOpcode::kBroadcast, NodeOperands({2})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kMultiply, NodeOperands({6, 4}), HasSingleUser},
      {HloOpcode::kBroadcast, NodeOperands({5})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kRng, NodeOperands({7, 8}), {IsRandomUniform, HasSingleUser}},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kConstant, NodeOperands({})}
    })),

  // Bias reduction and application.
  HloMatcherPattern(
    PatternType("bias_apply"),
    PatternMetaTarget(0),
    PatternInputs({5, 6, 7}),
    PatternOutputs({0}),
    PatternInplaceDescriptionFn(GetSimpleInplaceUseDescription),
    Pattern({
      {HloOpcode::kSubtract, NodeOperands({5, 1})},
      {HloOpcode::kMultiply, NodeOperands({3, 2})},
      {HloOpcode::kBroadcast, NodeOperands({7})},
      {HloOpcode::kReduce, NodeOperands({6, 4}), IsBiasReduce},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsScalar},
    })),

  // Convolution followed by scaled add/subtract to - A = A +/- B * c
  HloMatcherPattern(
    PatternType("conv_scaled_inplace"),
    PatternMetaTarget(3),
    PatternInputs({4, 5, 6, 7}),
    PatternOutputs({0}),
    PatternInplaceDescriptionFn(GetSimpleInplaceUseDescription),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({4, 1}), IsAddOrSubtract},
      {HloOpcode::kMultiply, NodeOperands({3, 2})},
      {HloOpcode::kBroadcast, NodeOperands({7})},
      {HloOpcode::kConvolution, NodeOperands({5, 6})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsScalar}
    })),

  // Reduce window with a window size of 1x1, stride 1 and identity reduction
  // function (param 1 is returned)
  HloMatcherPattern(
    PatternType("padding_reduce_window"),
    PatternMetaTarget(0),
    PatternInputs({1, 2}),
    PatternOutputs({0}),
    PatternInplaceDescriptionFn(
      [] (const HloMatcherMatched&) -> HloPoplarUseDescriptions {
      return {HloPoplarUseDescription{/*operand_number=*/0,
                                      /*operand_index=*/ShapeIndex{},
                                      /*output_index=*/ShapeIndex{},
                                      BufferUseKind::USE_ALIAS_READ_ONLY},
              HloPoplarUseDescription{/*operand_number=*/1,
                                      /*operand_index=*/ShapeIndex{},
                                      /*output_index=*/ShapeIndex{},
                                      BufferUseKind::USE_ALIAS_READ_ONLY}};
    }),
    Pattern({
      {HloOpcode::kReduceWindow, NodeOperands({1, 2}), IsPaddingReduceWindow},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})}
    })),

  // Reduce sum a squared input.
  HloMatcherPattern(
    PatternType("reduction_square_add"),
    PatternMetaTarget(0),
    PatternInputs({2, 3}),
    PatternOutputs({0}),
    Pattern({
      // NOLINTNEXTLINE
      {HloOpcode::kReduce, NodeOperands({1, 3}), IsReduceAdd},
      {HloOpcode::kMultiply, NodeOperands({2, 2})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})}
    })),

  // Reduce sum a squared input, where the operand being squared is of
  // type F16 and the accumulation operand is of type F32.
  HloMatcherPattern(
    PatternType("reduction_square_add"),
    PatternMetaTarget(0),
    PatternInputs({3, 4}),
    PatternOutputs({0}),
    Pattern({
      // NOLINTNEXTLINE
      {HloOpcode::kReduce, NodeOperands({1, 4}), IsReduceAdd},
      {HloOpcode::kConvert, NodeOperands({2}), IsF16ToF32Convert},
      {HloOpcode::kMultiply, NodeOperands({3, 3})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsF16},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsF32}
    })),

  // Reduction from FP16 to F32.
  HloMatcherPattern(
    PatternType("reduction_fp16_input"),
    PatternMetaTarget(0),
    PatternInputs({2, 3}),
    PatternOutputs({0}),
    Pattern({
      // NOLINTNEXTLINE
      {HloOpcode::kReduce, NodeOperands({1, 3}), {IsF32, IsReduceAddOrMultiply}},
      {HloOpcode::kConvert, NodeOperands({2}), IsF16ToF32Convert},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsF16},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsF32}
    })),
};
// clang-format on

FuseOpsLate::FuseOpsLate(struct CompilerAnnotations& annotations)
    : SingleHloMatcher(annotations, patterns, "_pop_op_",
                       /*restart_search_after_match*/ false) {}
}  // namespace poplarplugin
}  // namespace xla
