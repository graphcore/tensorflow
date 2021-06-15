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
#include <google/protobuf/util/message_differencer.h>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_early.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/conv_with_reverse.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
namespace {

MATCHER_P(EqualsProto, expected, "") {
  return google::protobuf::util::MessageDifferencer::Equals(arg, expected);
}

struct HloTestCase {
  std::string name;
  std::string hlo;
};

std::ostream& operator<<(std::ostream& stream, const HloTestCase& test_case) {
  stream << test_case.name;
  return stream;
}

struct EarlyFuseTest : HloTestBase, ::testing::WithParamInterface<HloTestCase> {
  void SetUp() override { ASSERT_TRUE(SetUpHloModule(GetParam())); }

  ::testing::AssertionResult SetUpHloModule(const HloTestCase& test_case) {
    const auto& hlo = test_case.hlo;
    auto module = ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest());
    if (module.ok()) {
      hlo_module_owner_ = std::move(module.ValueOrDie());
      hlo_module_ = hlo_module_owner_.get();

      annotations_ = absl::make_unique<CompilerAnnotations>(hlo_module_);

      return ::testing::AssertionSuccess();
    }

    return ::testing::AssertionFailure()
           << "Parsing hlo failed: " << module.status().error_message();
  }

  HloInstruction* FindRootInstruction() {
    auto* entry_comp = hlo_module_->entry_computation();
    return entry_comp->root_instruction();
  }

  VerifiedHloModule* hlo_module_ = nullptr;
  std::unique_ptr<CompilerAnnotations> annotations_;

  std::unique_ptr<VerifiedHloModule> hlo_module_owner_;
};

// Fixture for tests that apply to both convolution with reverse and sliced
// convolution with reverse instructions
using GeneralConvWithReverseEarlyFuseTest = EarlyFuseTest;

TEST_P(GeneralConvWithReverseEarlyFuseTest,
       ConvolutionWithReverseGetsFusedIntoCustomCall) {
  FuseOpsEarly fuse_ops_early(*annotations_);
  ASSERT_TRUE(fuse_ops_early.Run(hlo_module_).ValueOrDie());

  // 3 instructions = conv-with-reverse + its 2 arguments.
  auto* entry_comp = hlo_module_->entry_computation();
  ASSERT_EQ(entry_comp->instruction_count(), 3);

  auto* conv_with_reverse = entry_comp->root_instruction();
  ASSERT_TRUE(DynCast<HloConvWithReverse>(conv_with_reverse));
  ASSERT_EQ(conv_with_reverse->opcode(), HloOpcode::kCustomCall);
  ASSERT_EQ("ConvWithReverse", conv_with_reverse->custom_call_target());
}

TEST_P(GeneralConvWithReverseEarlyFuseTest, HasSameAttributesAsOriginalConv) {
  auto* original_conv = FindInstruction(hlo_module_, "conv");
  ASSERT_TRUE(original_conv);

  auto original_feature_group_count = original_conv->feature_group_count();
  auto original_batch_group_count = original_conv->batch_group_count();
  auto original_dim_labels = original_conv->convolution_dimension_numbers();
  auto original_precision_config = original_conv->precision_config();

  FuseOpsEarly fuse_ops_early(*annotations_);
  ASSERT_TRUE(fuse_ops_early.Run(hlo_module_).ValueOrDie());

  auto* conv_with_reverse = DynCast<HloConvWithReverse>(FindRootInstruction());
  ASSERT_TRUE(conv_with_reverse);

  // Note that the 'window' attribute gets treated differently as its behaviour
  // varies between conv with reverse and sliced conv with reverse.
  ASSERT_EQ(conv_with_reverse->feature_group_count(),
            original_feature_group_count);
  ASSERT_EQ(conv_with_reverse->batch_group_count(), original_batch_group_count);
  ASSERT_THAT(conv_with_reverse->convolution_dimension_numbers(),
              EqualsProto(original_dim_labels));
  ASSERT_THAT(conv_with_reverse->GetPrecisionConfig(),
              EqualsProto(original_precision_config));
}

TEST_P(GeneralConvWithReverseEarlyFuseTest, ValidAfterFusion) {
  const auto* arg0 = FindInstruction(hlo_module_, "arg_0");
  const auto* arg1 = FindInstruction(hlo_module_, "arg_1");
  ASSERT_TRUE(arg0);
  ASSERT_TRUE(arg1);

  auto arg0_value = LiteralUtil::CreateFullWithDescendingLayout(
      arg0->shape().dimensions(), half(1.0));
  auto arg1_value = LiteralUtil::CreateFullWithDescendingLayout(
      arg1->shape().dimensions(), half(2.0));
  const std::vector<Literal*> arg_values = {&arg0_value, &arg1_value};

  ASSERT_TRUE(RunAndCompare(std::move(hlo_module_owner_), arg_values, {}));
}

using ConvWithReverseEarlyFuseTest = EarlyFuseTest;

static const HloTestCase convolution_with_reverse_nhwc_format_test_case = {
    "convolution_with_reverse_nhwc_format", R"(
HloModule top

ENTRY c1 {
   arg_0 = f16[1,1,2,4] parameter(0)
   arg_1 = f16[3,3,4,4] parameter(1)
   reverse = f16[3,3,4,4] reverse(arg_1), dimensions={0,1}
   ROOT conv = f16[1,4,5,4] convolution(arg_0, reverse), window={size=3x3 pad=2_3x2_2 lhs_dilate=2x2}, dim_labels=b01f_01oi->b01f, operand_precision={HIGH, HIGHEST}
}
)"};
static const HloTestCase convolution_with_reverse_nchw_format_test_case = {
    "convolution_with_reverse_nchw_format", R"(
HloModule top

ENTRY c1 {
   arg_0 = f16[1,4,2,3] parameter(0)
   arg_1 = f16[3,3,4,4] parameter(1)
   reverse = f16[3,3,4,4] reverse(arg_1), dimensions={0,1}
   ROOT conv = f16[1,4,4,5] convolution(arg_0, reverse), window={size=3x3 pad=2_2x2_2}, dim_labels=bf01_01oi->bf01, operand_precision={HIGH, HIGHEST}
}
)"};
TEST_P(ConvWithReverseEarlyFuseTest, WindowAttributeMatchesOriginalConv) {
  auto* original_conv = FindInstruction(hlo_module_, "conv");
  ASSERT_TRUE(original_conv);
  auto original_window = original_conv->window();

  FuseOpsEarly fuse_ops_early(*annotations_);
  ASSERT_TRUE(fuse_ops_early.Run(hlo_module_).ValueOrDie());

  auto* conv_with_reverse = DynCast<HloConvWithReverse>(FindRootInstruction());
  ASSERT_TRUE(conv_with_reverse);
  ASSERT_THAT(conv_with_reverse->window(), EqualsProto(original_window));
}

using SlicedConvWithReverseEarlyFuseTest = EarlyFuseTest;

static const HloTestCase sliced_convolution_with_reverse_nhwc_format_test_case =
    {"sliced_convolution_with_reverse_nhwc_format", R"(
HloModule top

ENTRY c1 {
   arg_0 = f16[1,1,2,4] parameter(0)
   arg_1 = f16[3,3,4,4] parameter(1)
   reverse = f16[3,3,4,4] reverse(arg_1), dimensions={0,1}
   conv = f16[1,4,5,4] convolution(arg_0, reverse), window={size=3x3 pad=2_3x2_2 lhs_dilate=2x2}, dim_labels=b01f_01oi->b01f, operand_precision={HIGH, HIGHEST}
   ROOT slice = f16[1,2,3,4] slice(conv), slice={[0:1], [1:3], [1:4], [0:4]}
}
)"};
static const HloTestCase sliced_convolution_with_reverse_nchw_format_test_case =
    {"sliced_convolution_with_reverse_nchw_format", R"(
HloModule top

ENTRY c1 {
   arg_0 = f16[1,4,2,3] parameter(0)
   arg_1 = f16[3,3,4,4] parameter(1)
   reverse = f16[3,3,4,4] reverse(arg_1), dimensions={0,1}
   conv = f16[1,4,4,5] convolution(arg_0, reverse), window={size=3x3 pad=2_2x2_2}, dim_labels=bf01_01oi->bf01, operand_precision={HIGH, HIGHEST}
   ROOT slice = f16[1,4,2,3] slice(conv), slice={[0:1], [0:4], [1:3], [1:4]}
}
)"};
TEST_P(SlicedConvWithReverseEarlyFuseTest, WindowPaddingIsReducedBySlice) {
  // This is the main reason for having the conv-with-reverse instruction. If
  // the convolution is padded and the slice undoes some of that padding then we
  // can drop the slice and reduce the padding instead.
  auto* original_conv = FindInstruction(hlo_module_, "conv");
  ASSERT_TRUE(original_conv);

  auto reduced_window = original_conv->window();
  for (auto dim = 0ul; dim < reduced_window.dimensions_size(); ++dim) {
    auto* reduced_dimensions = reduced_window.mutable_dimensions(dim);
    // -1 as the slice takes off 1 in each direction.
    reduced_dimensions->set_padding_low(reduced_dimensions->padding_low() - 1);
    reduced_dimensions->set_padding_high(reduced_dimensions->padding_high() -
                                         1);
  }

  FuseOpsEarly fuse_ops_early(*annotations_);
  ASSERT_TRUE(fuse_ops_early.Run(hlo_module_).ValueOrDie());

  auto* conv_with_reverse = DynCast<HloConvWithReverse>(FindRootInstruction());
  ASSERT_TRUE(conv_with_reverse);
  ASSERT_THAT(conv_with_reverse->window(), EqualsProto(reduced_window));
}

TEST_P(SlicedConvWithReverseEarlyFuseTest, FusionUsesSliceShape) {
  // We want to make sure that the slice shape is preserved by the fusion.
  auto* slice = FindRootInstruction();
  ASSERT_TRUE(slice);
  ASSERT_EQ(slice->opcode(), HloOpcode::kSlice);
  const auto slice_shape = slice->shape();

  FuseOpsEarly fuse_ops_early(*annotations_);
  ASSERT_TRUE(fuse_ops_early.Run(hlo_module_).ValueOrDie());

  auto* conv_with_reverse = FindRootInstruction();
  ASSERT_EQ(conv_with_reverse->shape(), slice_shape);
}

// Not all slices can be fused, this fixture is used to test HLO that contains
// slices that can't be fused.
using NonFusableSliceConvWithReverseEarlyFuseTest =
    GeneralConvWithReverseEarlyFuseTest;

// If we the convolution doesn't have enough padding then we cant fuse the
// slice.
static const HloTestCase sliced_convolution_exceeding_padding_test_case = {
    "sliced_convolution_exceeding_padding", R"(
HloModule top

ENTRY c1 {
   arg_0 = f16[1,1,2,4] parameter(0)
   arg_1 = f16[3,3,4,4] parameter(1)
   reverse = f16[3,3,4,4] reverse(arg_1), dimensions={0,1}
   conv = f16[1,4,5,4] convolution(arg_0, reverse), window={size=3x3 pad=2_3x2_2 lhs_dilate=2x2}, dim_labels=b01f_01oi->b01f, operand_precision={HIGH, HIGHEST}
   ROOT root = f16[1,2,1,4] slice(conv), slice={[0:1], [1:3], [3:4], [0:4]}
}
)"};
// If dimensions outside of the convolutions height and width are sliced then we
// cant fuse the slice, since the padding only effects height/width.
static const HloTestCase non2d_sliced_convolution_test_case = {
    "non2d_sliced_convolution", R"(
HloModule top

ENTRY c1 {
   arg_0 = f16[1,1,2,4] parameter(0)
   arg_1 = f16[3,3,4,4] parameter(1)
   reverse = f16[3,3,4,4] reverse(arg_1), dimensions={0,1}
   conv = f16[1,4,5,4] convolution(arg_0, reverse), window={size=3x3 pad=2_3x2_2 lhs_dilate=2x2}, dim_labels=b01f_01oi->b01f, operand_precision={HIGH, HIGHEST}
   ROOT root = f16[1,2,3,3] slice(conv), slice={[0:1], [1:3], [1:4], [1:4]}
}
)"};
TEST_P(NonFusableSliceConvWithReverseEarlyFuseTest, SliceDoesntGetFused) {
  FuseOpsEarly fuse_ops_early(*annotations_);
  ASSERT_TRUE(fuse_ops_early.Run(hlo_module_).ValueOrDie());

  auto* entry_comp = hlo_module_->entry_computation();
  auto* slice = entry_comp->root_instruction();
  ASSERT_EQ(slice->opcode(), HloOpcode::kSlice);

  // 4 = conv_with_reverse operand, its 2 operands, and the slice that consumes
  // it.
  ASSERT_EQ(entry_comp->instruction_count(), 4);
  auto* conv_with_reverse = DynCast<HloConvWithReverse>(slice->operand(0));
  ASSERT_TRUE(conv_with_reverse);
}

template <class Fixture>
std::string TestName(
    const ::testing::TestParamInfo<typename Fixture::ParamType>& info) {
  return info.param.name;
}

std::vector<HloTestCase> ConvolutionWithReverseTestCases() {
  return {convolution_with_reverse_nhwc_format_test_case,
          convolution_with_reverse_nchw_format_test_case};
}
INSTANTIATE_TEST_SUITE_P(EarlyFuseHLO, ConvWithReverseEarlyFuseTest,
                         ::testing::ValuesIn(ConvolutionWithReverseTestCases()),
                         TestName<ConvWithReverseEarlyFuseTest>);

std::vector<HloTestCase> SlicedConvolutionWithReverseTestCases() {
  return {sliced_convolution_with_reverse_nhwc_format_test_case,
          sliced_convolution_with_reverse_nchw_format_test_case};
}
INSTANTIATE_TEST_SUITE_P(
    EarlyFuseHLO, SlicedConvWithReverseEarlyFuseTest,
    ::testing::ValuesIn(SlicedConvolutionWithReverseTestCases()),
    TestName<SlicedConvWithReverseEarlyFuseTest>);

std::vector<HloTestCase> GeneralConvolutionWithReverseTestCases() {
  const auto slice_test_cases = SlicedConvolutionWithReverseTestCases();

  auto test_cases = ConvolutionWithReverseTestCases();
  test_cases.insert(test_cases.end(), slice_test_cases.begin(),
                    slice_test_cases.end());
  return test_cases;
}
INSTANTIATE_TEST_SUITE_P(
    EarlyFuseHLO, GeneralConvWithReverseEarlyFuseTest,
    ::testing::ValuesIn(GeneralConvolutionWithReverseTestCases()),
    TestName<GeneralConvWithReverseEarlyFuseTest>);

INSTANTIATE_TEST_SUITE_P(
    EarlyFuseHLO, NonFusableSliceConvWithReverseEarlyFuseTest,
    ::testing::Values(non2d_sliced_convolution_test_case,
                      sliced_convolution_exceeding_padding_test_case),
    TestName<NonFusableSliceConvWithReverseEarlyFuseTest>);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
