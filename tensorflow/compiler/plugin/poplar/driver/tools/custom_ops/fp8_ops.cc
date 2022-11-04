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

#include <array>
#include <memory>
#include <string>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/fp8_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/window_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace poplarplugin {

HloF8MatMulInstruction::HloF8MatMulInstruction(const Shape& shape,
                                               HloInstruction* lhs,
                                               HloInstruction* lhs_metadata,
                                               HloInstruction* rhs,
                                               HloInstruction* rhs_metadata)
    : HloPoplarInstruction(shape, {lhs, lhs_metadata, rhs, rhs_metadata},
                           PoplarOp::F8MatMul) {}

std::unique_ptr<HloInstruction>
HloF8MatMulInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* context) const {
  return absl::make_unique<HloF8MatMulInstruction>(
      shape, operands[0], operands[1], operands[2], operands[3]);
}

template <uint16_t D>
constexpr PoplarOp ConvType() {
  return PoplarOp::Unknown;
}

template <>
constexpr PoplarOp ConvType<2>() {
  return PoplarOp::F8Conv2D;
}

template <>
constexpr PoplarOp ConvType<3>() {
  return PoplarOp::F8Conv3D;
}

template <uint16_t D>
StatusOr<ConvolutionDimensionNumbers> ConvDNums() {
  return InternalError("Unsupported conv dim.");
}

template <>
StatusOr<ConvolutionDimensionNumbers> ConvDNums<2>() {
  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.set_input_feature_dimension(3);

  dnums.set_output_batch_dimension(0);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(2);
  dnums.set_output_feature_dimension(3);

  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);

  return dnums;
}

template <>
StatusOr<ConvolutionDimensionNumbers> ConvDNums<3>() {
  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_input_spatial_dimensions(3);
  dnums.set_input_feature_dimension(4);

  dnums.set_output_batch_dimension(0);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(3);
  dnums.set_output_feature_dimension(4);

  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.add_kernel_spatial_dimensions(2);
  dnums.set_kernel_input_feature_dimension(3);
  dnums.set_kernel_output_feature_dimension(4);

  return dnums;
}

template <uint16_t D>
StatusOr<xla::Window> ConvWindow(const HloInstruction* const input,
                                 const HloInstruction* const filter,
                                 const std::vector<int32>& strides,
                                 xla::Padding padding) {
  return InternalError("Unsupported conv dim.");
}

template <>
StatusOr<xla::Window> ConvWindow<2>(const HloInstruction* const input,
                                    const HloInstruction* const filter,
                                    const std::vector<int32>& strides,
                                    xla::Padding padding) {
  const auto in_shape = xla::Shape(
      input->shape().element_type(),
      {input->shape().dimensions(1), input->shape().dimensions(2)}, {}, {});

  const std::array<int64_t, 2> ksizes = {filter->shape().dimensions(0),
                                         filter->shape().dimensions(1)};

  const std::array<int64_t, 2> kstrides = {strides.at(1), strides.at(2)};

  return xla::poplarplugin::MakeWindow(in_shape, ksizes, kstrides, padding, {},
                                       {});
}

template <>
StatusOr<xla::Window> ConvWindow<3>(const HloInstruction* const input,
                                    const HloInstruction* const filter,
                                    const std::vector<int32>& strides,
                                    xla::Padding padding) {
  const auto in_shape =
      xla::Shape(input->shape().element_type(),
                 {input->shape().dimensions(1), input->shape().dimensions(2),
                  input->shape().dimensions(3)},
                 {}, {});

  const std::array<int64_t, 3> ksizes = {filter->shape().dimensions(0),
                                         filter->shape().dimensions(1),
                                         filter->shape().dimensions(2)};

  const std::array<int64_t, 3> kstrides = {strides.at(1), strides.at(2),
                                           strides.at(3)};

  return xla::poplarplugin::MakeWindow(in_shape, ksizes, kstrides, padding, {},
                                       {});
}

template <uint16_t D>
HloF8ConvInstruction<D>::HloF8ConvInstruction(
    const Shape& shape, HloInstruction* input, HloInstruction* filter,
    HloInstruction* input_meta, HloInstruction* filter_meta,
    const std::vector<int32>& strides, const std::string& padding,
    const std::vector<int32>& explicit_paddings, const std::string& data_format,
    const std::vector<int32>& dilations)
    : HloPoplarInstruction(shape, {input, filter, input_meta, filter_meta},
                           ConvType<D>()),
      strides(strides),
      padding(padding),
      explicit_paddings(explicit_paddings),
      data_format(data_format),
      dilations(dilations) {
  // Setup the window.
  const auto padding_t = padding == "SAME" ? Padding::kSame : Padding::kValid;
  window_ = ConvWindow<D>(input, filter, strides, padding_t).ValueOrDie();

  // Setup conv dimension numbers.
  auto dnums = ConvDNums<D>();
  set_convolution_dimension_numbers(dnums.ValueOrDie());
}

template <uint16_t D>
std::unique_ptr<HloInstruction>
HloF8ConvInstruction<D>::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* context) const {
  return absl::make_unique<HloF8ConvInstruction<D>>(
      shape, operands[0], operands[1], operands[2], operands[3], Strides(),
      Padding(), ExplicitPaddings(), DataFormat(), Dilations());
}

namespace {

static HloPoplarInstructionFactory f8_mat_mul_factory(
    PoplarOp::F8MatMul,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      return StatusOr<std::unique_ptr<HloInstruction>>(
          absl::make_unique<HloF8MatMulInstruction>(
              call->shape(), call->mutable_operand(0), call->mutable_operand(1),
              call->mutable_operand(2), call->mutable_operand(3)));
    });

template <uint16_t D>
static StatusOr<std::unique_ptr<HloInstruction>> f8_conv_factory_impl(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  TF_ASSIGN_OR_RETURN(auto strides,
                      attribute_map.GetAttributeInt32Vector("strides"));

  TF_ASSIGN_OR_RETURN(auto padding,
                      attribute_map.GetAttributeAsString("padding"));

  std::vector<int> explicit_paddings;
  if (D == 2) {
    TF_ASSIGN_OR_RETURN(
        explicit_paddings,
        attribute_map.GetAttributeInt32Vector("explicit_paddings"));
  }

  TF_ASSIGN_OR_RETURN(auto data_format,
                      attribute_map.GetAttributeAsString("data_format"));

  std::vector<int> dilations;
  TF_ASSIGN_OR_RETURN(dilations,
                      attribute_map.GetAttributeInt32Vector("dilations"));

  return StatusOr<std::unique_ptr<HloInstruction>>(
      absl::make_unique<HloF8ConvInstruction<D>>(
          call->shape(), call->mutable_operand(0), call->mutable_operand(1),
          call->mutable_operand(2), call->mutable_operand(3), strides, padding,
          explicit_paddings, data_format, dilations));
}

static HloPoplarInstructionFactory f8_conv_2d_factory(PoplarOp::F8Conv2D,
                                                      f8_conv_factory_impl<2>);

static HloPoplarInstructionFactory f8_conv_3d_factory(PoplarOp::F8Conv3D,
                                                      f8_conv_factory_impl<3>);

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
