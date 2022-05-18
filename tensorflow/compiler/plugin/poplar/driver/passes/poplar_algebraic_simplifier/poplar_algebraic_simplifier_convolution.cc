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

#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier/poplar_algebraic_simplifier_convolution.h"

#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier/poplar_algebraic_simplifier_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"

namespace xla {
namespace poplarplugin::algebraic_simplifier::convolution {
namespace {
namespace m = match;
}
StatusOr<std::unique_ptr<HloInstruction>> FoldConvInputPad(
    HloInstruction* convolution) {
  HloInstruction *lhs, *a, *b;
  if (Match(convolution,
            m::Convolution(m::Pad(&lhs, m::Op(&a), m::ConstantScalar(0)),
                           m::Op(&b)))) {
    const auto& window = convolution->window();
    const ConvolutionDimensionNumbers& dnums =
        convolution->convolution_dimension_numbers();

    const auto& padding = lhs->padding_config();

    // Can't pad batch or feature dims.
    for (int64 dim :
         {dnums.input_batch_dimension(), dnums.input_feature_dimension()}) {
      const auto& p = padding.dimensions(dim);
      if (p.edge_padding_low() != 0 || p.edge_padding_high() != 0 ||
          p.interior_padding() != 0) {
        return {nullptr};
      }
    }

    // Compute the window which is the result of merging the kPad and the
    // convolution's existing window.
    Window new_window = window;
    for (int64 dim = 0; dim < dnums.input_spatial_dimensions_size(); ++dim) {
      auto& w = *new_window.mutable_dimensions(dim);
      const auto& p = padding.dimensions(dnums.input_spatial_dimensions(dim));
      // Edge padding composes with itself in the straightforward way, but
      // composing interior padding is nontrivial, and we cowardly refuse to
      // think about it. If we see interior padding in either the kPad or conv,
      // bail if there's any sort of padding in the other.
      if (p.interior_padding() != 0 &&
          (w.padding_low() != 0 || w.padding_high() != 0 ||
           w.base_dilation() != 1)) {
        return {nullptr};
      }
      if (w.base_dilation() != 1 &&
          (p.edge_padding_low() != 0 || p.edge_padding_high() != 0 ||
           p.interior_padding() != 0)) {
        return {nullptr};
      }

      w.set_padding_low(w.padding_low() + p.edge_padding_low());
      w.set_padding_high(w.padding_high() + p.edge_padding_high());
      if (p.interior_padding() != 0) {
        CHECK_EQ(w.base_dilation(), 1);
        w.set_base_dilation(1 + p.interior_padding());
      }
    }

    auto new_conv =
        convolution->CloneWithNewOperands(convolution->shape(), {a, b});
    new_conv->set_window(new_window);
    return new_conv;
  }
  return {nullptr};
}

StatusOr<std::unique_ptr<HloInstruction>> FoldConvFilterPad(
    const PoplarAlgebraicSimplifier* simplifier, HloInstruction* convolution) {
  auto* lhs = convolution->mutable_operand(0);
  auto* rhs = convolution->mutable_operand(1);
  const ConvolutionDimensionNumbers& dnums =
      convolution->convolution_dimension_numbers();

  if (rhs->opcode() != HloOpcode::kPad) {
    return {nullptr};
  }

  // Convolution's padding is always zero, so bail if the kPad is adding
  // something other than zero.
  if (!util::IsAll(rhs->operand(1), 0)) {
    return {nullptr};
  }

  const auto& padding = rhs->padding_config();

  // Can't pad or dilate feature dims.
  for (int64 dim : {dnums.kernel_input_feature_dimension(),
                    dnums.kernel_output_feature_dimension()}) {
    const auto& p = padding.dimensions(dim);
    if (p.edge_padding_low() != 0 || p.edge_padding_high() != 0 ||
        p.interior_padding() != 0) {
      return {nullptr};
    }
  }

  // Compute the window which is the result of merging the kPad and the
  // convolution's existing window.
  Window new_window = convolution->window();
  for (int64 dim = 0; dim < dnums.kernel_spatial_dimensions_size(); ++dim) {
    auto& w = *new_window.mutable_dimensions(dim);
    const auto& p = padding.dimensions(dnums.kernel_spatial_dimensions(dim));

    // We can only do this transformation if p adds dilation to the filter --
    // edge padding on the filter is not supported in conv.
    if (p.edge_padding_low() != 0 || p.edge_padding_high() != 0) {
      return {nullptr};
    }

    // Nothing to do if the kPad for this dim is entirely a nop.
    if (p.interior_padding() == 0) {
      continue;
    }

    // We cowardly refuse to think about how dilation composes with itself;
    // bail if both the kPad and conv have dilation on this dimension.
    if (w.window_dilation() > 1) {
      return {nullptr};
    }
    CHECK_EQ(w.window_dilation(), 1);
    w.set_window_dilation(1 + p.interior_padding());
    w.set_size(rhs->operand(0)->shape().dimensions(
        dnums.kernel_spatial_dimensions(dim)));
  }

  auto new_conv = convolution->CloneWithNewOperands(
      convolution->shape(), {lhs, rhs->mutable_operand(0)});
  new_conv->set_window(new_window);

  return new_conv;
}
}  // namespace poplarplugin::algebraic_simplifier::convolution
}  // namespace xla
