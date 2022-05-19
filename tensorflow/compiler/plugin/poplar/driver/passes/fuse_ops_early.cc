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
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {
namespace {

// Utility function for processing the padding/slice information of a sliced
// convolution. The given apply function is called for each spatial dimension
// where a slice occurs.
template <typename Fn>
void VisitSlicedConv2dPadding(const HloSliceInstruction* slice, Fn&& apply) {
  const HloInstruction* conv = slice->operand(0);
  CHECK_EQ(conv->opcode(), HloOpcode::kConvolution);

  const ConvolutionDimensionNumbers& conv_dim_numbers =
      conv->convolution_dimension_numbers();
  const Window& window = conv->window();
  const Shape& conv_shape = conv->shape();

  CHECK_EQ(conv_dim_numbers.input_spatial_dimensions_size(),
           window.dimensions_size());
  for (int64_t dim = 0; dim < window.dimensions_size(); dim++) {
    // We check the slices across the convolutions spatial dimensions as this is
    // where the padding happens.
    const int64_t slice_dim = conv_dim_numbers.input_spatial_dimensions(dim);
    const int64_t slice_first = slice->slice_starts(slice_dim);
    const int64_t slice_last = slice->slice_limits(slice_dim);
    const int64_t slice_size = slice_last - slice_first;

    const int64_t conv_dim_size = conv_shape.dimensions(slice_dim);
    if (conv_dim_size > slice_size) {
      const WindowDimension& window_dimensions = window.dimensions(dim);

      apply(dim, conv_dim_size, window_dimensions.padding_low(),
            window_dimensions.padding_high(), slice_first, slice_last);
    }
  }
}

bool Is2DSliceOfConv(const HloInstruction* slice) {
  // We can only consider slices that operate within the spatial area of
  // the Convolution. If there is slicing outside of this then it can't be
  // fused.
  const HloInstruction* operand = slice->operand(0);
  if (operand->opcode() == HloOpcode::kConvolution) {
    const Shape& slice_shape = slice->shape();
    const Shape& conv_shape = operand->shape();

    const ConvolutionDimensionNumbers& conv_dim_numbers =
        operand->convolution_dimension_numbers();
    const int64_t batch_dim = conv_dim_numbers.input_batch_dimension();
    const int64_t feature_dim = conv_dim_numbers.input_feature_dimension();

    return slice_shape.rank() == 4 &&
           slice_shape.dimensions(batch_dim) ==
               conv_shape.dimensions(batch_dim) &&
           slice_shape.dimensions(feature_dim) ==
               conv_shape.dimensions(feature_dim);
  }

  return false;
}

bool SliceReducesConv2dPadding(const HloInstruction* slice) {
  if (Is2DSliceOfConv(slice)) {
    bool slice_reduces_padding = true;
    const auto check_slice_reduces_padding =
        [&slice_reduces_padding](int64_t dim, int64_t conv_dim_size,
                                 int64_t padding_low, int64_t padding_high,
                                 int64_t slice_first, int64_t slice_last) {
          if (slice_first > 0) {
            slice_reduces_padding &= padding_low > slice_first;
          }
          if (slice_last < conv_dim_size) {
            slice_reduces_padding &=
                padding_high > (conv_dim_size - slice_last);
          }
        };
    VisitSlicedConv2dPadding(Cast<HloSliceInstruction>(slice),
                             check_slice_reduces_padding);

    return slice_reduces_padding;
  }

  return false;
}

StatusOr<PatternInstructionOutputs> CreateSlicedConvWithReverseFromMatch(
    const HloMatcherMatched& matched) {
  const auto& inputs = matched.GetInputs();
  const auto& outputs = matched.GetOutputs();
  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(outputs.size(), 1);

  const HloSliceInstruction* slice = Cast<HloSliceInstruction>(outputs[0]);
  const HloInstruction* original_conv = slice->operand(0);

  Window reduced_window = original_conv->window();
  const auto reduce_conv_padding = [&reduced_window](int64_t dim,
                                                     int64_t conv_dim_size,
                                                     int64_t padding_low,
                                                     int64_t padding_high,
                                                     int64_t slice_first,
                                                     int64_t slice_last) {
    WindowDimension* dimensions = reduced_window.mutable_dimensions(dim);
    dimensions->set_padding_low(padding_low - slice_first);
    dimensions->set_padding_high(padding_high - (conv_dim_size - slice_last));
  };
  VisitSlicedConv2dPadding(slice, reduce_conv_padding);

  return PatternInstructionOutputs{matched.computation->AddInstruction(
      CreateConvWithReverse(slice->shape(), inputs[0], inputs[1],
                            original_conv->feature_group_count(),
                            original_conv->batch_group_count(), reduced_window,
                            original_conv->convolution_dimension_numbers(),
                            original_conv->precision_config()))};
}

StatusOr<PatternInstructionOutputs> CreateConvWithReverseFromMatch(
    const HloMatcherMatched& matched) {
  const auto& inputs = matched.GetInputs();
  const auto& outputs = matched.GetOutputs();
  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(outputs.size(), 1);

  const HloInstruction* original_conv = outputs[0];

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
    PatternReplaceFn(CreateSlicedConvWithReverseFromMatch),
    PatternMetaTarget(0),
    PatternInputs({3, 4}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kSlice, NodeOperands({1}), SliceReducesConv2dPadding},
      {HloOpcode::kConvolution, NodeOperands({3, 2}), IsOpWithWindowNoStride},
      {HloOpcode::kReverse, NodeOperands({4}), IsConvFilterTranspose},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})}
    })),

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
