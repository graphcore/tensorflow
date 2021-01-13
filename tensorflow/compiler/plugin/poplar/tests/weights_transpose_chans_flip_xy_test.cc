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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/weights_transpose_chans_flip_xy.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/entry_visitor.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

class WeightsTransposeChansFlipXYTest
    : public HloTestBase,
      public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(WeightsTransposeChansFlipXYTestCases,
                         WeightsTransposeChansFlipXYTest,
                         ::testing::Values(false, true));

// This tests that the output value has the expected value
TEST_P(WeightsTransposeChansFlipXYTest, TestWeightsTransposeChansFlipXY0) {
  const bool allocate = GetParam();
  HloComputation::Builder builder = HloComputation::Builder("BuilderHloComp0");

  auto weights = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR4<float>({{{{1, 3}, {2, 4}}}})));

  ConvolutionDimensionNumbers conv_dim_num;

  // [B, 0, 1, F]
  conv_dim_num.set_input_batch_dimension(0);
  conv_dim_num.add_input_spatial_dimensions(1);
  conv_dim_num.add_input_spatial_dimensions(2);
  conv_dim_num.set_input_feature_dimension(3);

  // [0, 1, O, I]
  conv_dim_num.add_kernel_spatial_dimensions(0);
  conv_dim_num.add_kernel_spatial_dimensions(1);
  conv_dim_num.set_kernel_output_feature_dimension(2);
  conv_dim_num.set_kernel_input_feature_dimension(3);

  // [B, 0, 1, F]
  conv_dim_num.set_output_batch_dimension(0);
  conv_dim_num.add_output_spatial_dimensions(1);
  conv_dim_num.add_output_spatial_dimensions(2);
  conv_dim_num.set_output_feature_dimension(3);

  std::vector<size_t> conv_input_shape;
  conv_input_shape.push_back(1);
  conv_input_shape.push_back(4);
  conv_input_shape.push_back(4);
  conv_input_shape.push_back(2);

  std::vector<size_t> conv_output_shape;
  conv_output_shape.push_back(1);
  conv_output_shape.push_back(4);
  conv_output_shape.push_back(4);
  conv_output_shape.push_back(2);

  xla::Window window = window_util::MakeWindow({1, 1});
  int64 feature_group_count = 1;

  auto weights_transpose_inst =
      builder.AddInstruction(CreateHloWeightsTransposeChansFlipXY(
          weights, conv_dim_num, conv_input_shape, conv_output_shape, window,
          feature_group_count));

  auto computation = builder.Build();
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(std::move(computation));

  CompilerAnnotations annotations(module.get());
  if (allocate) {
    EXPECT_TRUE(AllocationFinder(annotations).Run(module.get()).ValueOrDie());
  }

  Literal result = ExecuteNoHloPasses(std::move(module), {});

  Literal expected = LiteralUtil::CreateR4<float>({{{{1, 2}, {3, 4}}}});

  EXPECT_TRUE(
      LiteralTestUtil::NearOrEqual(expected, result, ErrorSpec{1e-4, 1e-4}));
}

TEST_P(WeightsTransposeChansFlipXYTest, TestWeightsTransposeChansFlipXYGroup) {
  const bool allocate = GetParam();
  HloComputation::Builder builder = HloComputation::Builder("BuilderHloComp1");

  auto weights = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR4<float>(
          {{{{1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1,
              13.1, 14.1, 15.1, 16.1}},
            {{1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2, 12.2,
              13.2, 14.2, 15.2, 16.2}},
            {{1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 9.3, 10.3, 11.3, 12.3,
              13.3, 14.3, 15.3, 16.3}},
            {{1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.4, 12.4,
              13.4, 14.4, 15.4, 16.4}}},
           {{{1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5,
              13.5, 14.5, 15.5, 16.5}},
            {{1.6, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
            {{1.7, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
            {{1.8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}},
           {{{1.9, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
            {{1.10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
            {{1.11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
            {{1.12, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}},
           {{{1.13, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
            {{1.14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
            {{1.15, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
            {{1.16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}}})));

  ConvolutionDimensionNumbers conv_dim_num;

  // [B, 0, 1, F]
  conv_dim_num.set_input_batch_dimension(0);
  conv_dim_num.add_input_spatial_dimensions(1);
  conv_dim_num.add_input_spatial_dimensions(2);
  conv_dim_num.set_input_feature_dimension(3);

  // [0, 1, I, O]
  conv_dim_num.add_kernel_spatial_dimensions(0);
  conv_dim_num.add_kernel_spatial_dimensions(1);
  conv_dim_num.set_kernel_input_feature_dimension(2);
  conv_dim_num.set_kernel_output_feature_dimension(3);

  // [B, 0, 1, F]
  conv_dim_num.set_output_batch_dimension(0);
  conv_dim_num.add_output_spatial_dimensions(1);
  conv_dim_num.add_output_spatial_dimensions(2);
  conv_dim_num.set_output_feature_dimension(3);

  std::vector<size_t> conv_input_shape;
  conv_input_shape.push_back(16);
  conv_input_shape.push_back(9);
  conv_input_shape.push_back(9);
  conv_input_shape.push_back(16);

  std::vector<size_t> conv_output_shape;
  conv_output_shape.push_back(16);
  conv_output_shape.push_back(9);
  conv_output_shape.push_back(9);
  conv_output_shape.push_back(16);

  Window window = window_util::MakeWindow({4, 4});
  int64 feature_group_count = 16;

  auto weights_transpose_inst =
      builder.AddInstruction(CreateHloWeightsTransposeChansFlipXY(
          weights, conv_dim_num, conv_input_shape, conv_output_shape, window,
          feature_group_count));
  auto reshape = builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(weights->shape().element_type(), {4, 4, 1, 16}),
      weights_transpose_inst));

  auto computation = builder.Build(reshape);
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(std::move(computation));

  CompilerAnnotations annotations(module.get());
  if (allocate) {
    EXPECT_TRUE(AllocationFinder(annotations).Run(module.get()).ValueOrDie());
  }

  Literal result = ExecuteNoHloPasses(std::move(module), {});

  Literal expected = LiteralUtil::CreateR4<float>(
      {{ /*i0=0*/
        {/*i1=0*/
         {1.16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
        {/*i1=1*/
         {1.15, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
        {/*i1=2*/
         {1.14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
        {/*i1=3*/
         {1.13, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}},
       { /*i0=1*/
        {/*i1=0*/
         {1.12, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
        {/*i1=1*/
         {1.11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
        {/*i1=2*/
         {1.1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
        {/*i1=3*/
         {1.9, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}},
       { /*i0=2*/
        {/*i1=0*/
         {1.8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
        {/*i1=1*/
         {1.7, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
        {/*i1=2*/
         {1.6, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
        {/*i1=3*/
         {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5,
          14.5, 15.5, 16.5}}},
       { /*i0=3*/
        {/*i1=0*/
         {1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.4, 12.4, 13.4,
          14.4, 15.4, 16.4}},
        {/*i1=1*/
         {1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 9.3, 10.3, 11.3, 12.3, 13.3,
          14.3, 15.3, 16.3}},
        {/*i1=2*/
         {1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2, 12.2, 13.2,
          14.2, 15.2, 16.2}},
        {/*i1=3*/
         {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1,
          14.1, 15.1, 16.1}}}});

  EXPECT_TRUE(
      LiteralTestUtil::NearOrEqual(expected, result, ErrorSpec{1e-4, 1e-4}));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
