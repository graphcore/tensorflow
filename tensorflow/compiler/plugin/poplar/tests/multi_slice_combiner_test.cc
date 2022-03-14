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
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/scatter_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"
#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xp = ::xla::poplarplugin;
namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

se::Platform* GetReferencePlatform() {
  auto result = PlatformUtil::GetPlatform("interpreter");
  return result.ValueOrDie();
}

se::Platform* GetTestPlatform() {
  // auto result = PlatformUtil::GetDefaultPlatform();
  auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
  EXPECT_TRUE(platform.ok());

  auto* p = dynamic_cast<xp::PoplarPlatform*>(platform.ValueOrDie());

  xla::poplarplugin::IpuOptions options;
  options.set_creator_id(IpuOptionsCreator::IPU_UTILS);
  options.set_enable_multi_slice_combiner(true);

  EXPECT_EQ(p->ConfigurePoplarDevices(options), Status::OK());
  return p;
}

class MultiSliceCombinerTest : public HloTestBase {
 public:
  MultiSliceCombinerTest()
      : HloTestBase(GetTestPlatform(), GetReferencePlatform()) {}
};

auto GetNumMultiSlice = GetNumInstructions<HloMultiSliceInstruction>;
auto GetNumSlice = GetNumInstructions<HloSliceInstruction>;
auto GetNumConcatenate = GetNumInstructions<HloConcatenateInstruction>;

TEST_F(MultiSliceCombinerTest, LookupsSharedInput) {
  std::string hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[100,16] parameter(0)
  offsets1 = s32[24,1] parameter(1)
  offsets2 = s32[12,1] parameter(2)
  slice1 = f32[24,16] custom-call(input, offsets1), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  slice2 = f32[12,16] custom-call(input, offsets2), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  ROOT t = (f32[24,16], f32[12,16]) tuple(slice1, slice2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();
  auto* module = module0.get();

  CompilerAnnotations annotations(module);

  ScatterSimplifier sc;
  EXPECT_FALSE(sc.Run(module).ValueOrDie());
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());
  EXPECT_EQ(GetNumMultiSlice(module->entry_computation()), 2);
  EXPECT_EQ(GetNumSlice(module->entry_computation()), 0);
  EXPECT_EQ(GetNumConcatenate(module->entry_computation()), 0);
  HloPassFix<MultiSliceCombiner> ms_combiner(annotations);
  EXPECT_TRUE(ms_combiner.Run(module).ValueOrDie());
  EXPECT_EQ(GetNumMultiSlice(module->entry_computation()), 1);
  EXPECT_EQ(GetNumSlice(module->entry_computation()), 2);
  EXPECT_EQ(GetNumConcatenate(module->entry_computation()), 1);

  auto root = module->entry_computation()->root_instruction();

  EXPECT_TRUE(Match(root->operand(0),
                    m::Slice(m::CustomCall(m::Parameter(), m::Concatenate()))));
  EXPECT_TRUE(Match(root->operand(1),
                    m::Slice(m::CustomCall(m::Parameter(), m::Concatenate()))));
  EXPECT_EQ(root->operand(0)->operand(0), root->operand(1)->operand(0));

  // Check the expected value.
  auto input_shape = ShapeUtil::MakeShape(F32, {100, 16});
  Literal input(input_shape);
  input.Populate<float>([](const xla::DimensionVector& index) {
    return 0.01f * index[1] + index[0];
  });

  auto offset1_shape = ShapeUtil::MakeShape(S32, {24, 1});
  Literal offset1(offset1_shape);
  offset1.Populate<int32>(
      [](const xla::DimensionVector& index) { return index[0] * 4; });

  auto offset2_shape = ShapeUtil::MakeShape(S32, {12, 1});
  Literal offset2(offset2_shape);
  offset2.Populate<int32>(
      [](const xla::DimensionVector& index) { return 99 - index[0] * 5; });

  Literal result =
      Execute(
          std::move(
              ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie()),
          {&input, &offset1, &offset2})
          .ValueOrDie();
  ASSERT_TRUE(result.shape().IsTuple());

  const Shape& slice1 = ShapeUtil::GetSubshape(result.shape(), {0});
  ShapeUtil::ForEachIndex(slice1, [&](absl::Span<const int64> output_index) {
    EXPECT_EQ(output_index.size(), 2);
    float value = result.Get<float>(output_index, {0});
    auto offset_idx = offset1.Get<int32>({output_index[0], 0});
    auto input_value = input.Get<float>({offset_idx, output_index[1]});
    EXPECT_EQ(value, input_value);
    return true;
  });
  const Shape& slice2 = ShapeUtil::GetSubshape(result.shape(), {1});
  ShapeUtil::ForEachIndex(slice2, [&](absl::Span<const int64> output_index) {
    EXPECT_EQ(output_index.size(), 2);
    float value = result.Get<float>(output_index, {1});
    auto offset_idx = offset2.Get<int32>({output_index[0], 0});
    auto input_value = input.Get<float>({offset_idx, output_index[1]});
    EXPECT_EQ(value, input_value);
    return true;
  });
}

TEST_F(MultiSliceCombinerTest, ThreeLookupsSharedInput) {
  std::string hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[100,16] parameter(0)
  offsets1 = s32[24,1] parameter(1)
  offsets2 = s32[12,1] parameter(2)
  offsets3 = s32[8,1] parameter(3)
  slice1 = f32[24,16] custom-call(input, offsets1), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  slice2 = f32[12,16] custom-call(input, offsets2), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  slice3 = f32[8,16] custom-call(input, offsets3), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  ROOT t = (f32[24,16], f32[12,16], f32[8,16]) tuple(slice1, slice2, slice3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();
  auto* module = module0.get();

  CompilerAnnotations annotations(module);

  ScatterSimplifier sc;
  EXPECT_FALSE(sc.Run(module).ValueOrDie());
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());
  EXPECT_EQ(GetNumMultiSlice(module->entry_computation()), 3);
  EXPECT_EQ(GetNumSlice(module->entry_computation()), 0);
  EXPECT_EQ(GetNumConcatenate(module->entry_computation()), 0);
  HloPassFix<MultiSliceCombiner> ms_combiner(annotations);
  EXPECT_TRUE(ms_combiner.Run(module).ValueOrDie());
  PoplarAlgebraicSimplifier simplifier{{}};
  HloPassFix<PoplarAlgebraicSimplifier> algebraic_simplifier{simplifier};
  EXPECT_TRUE(algebraic_simplifier.Run(module).ValueOrDie());
  EXPECT_EQ(GetNumMultiSlice(module->entry_computation()), 1);
  EXPECT_EQ(GetNumSlice(module->entry_computation()), 3);
  EXPECT_EQ(GetNumConcatenate(module->entry_computation()), 1);

  auto root = module->entry_computation()->root_instruction();

  EXPECT_TRUE(Match(root->operand(0),
                    m::Slice(m::CustomCall(m::Parameter(), m::Concatenate()))));
  EXPECT_TRUE(Match(root->operand(1),
                    m::Slice(m::CustomCall(m::Parameter(), m::Concatenate()))));
  EXPECT_TRUE(Match(root->operand(2),
                    m::Slice(m::CustomCall(m::Parameter(), m::Concatenate()))));
  EXPECT_EQ(root->operand(0)->operand(0), root->operand(1)->operand(0));
  EXPECT_EQ(root->operand(0)->operand(0), root->operand(2)->operand(0));

  // Check the expected value.
  auto input_shape = ShapeUtil::MakeShape(F32, {100, 16});
  Literal input(input_shape);
  input.Populate<float>([](const xla::DimensionVector& index) {
    return 0.01f * index[1] + index[0];
  });

  auto offset1_shape = ShapeUtil::MakeShape(S32, {24, 1});
  Literal offset1(offset1_shape);
  offset1.Populate<int32>(
      [](const xla::DimensionVector& index) { return index[0] * 4; });

  auto offset2_shape = ShapeUtil::MakeShape(S32, {12, 1});
  Literal offset2(offset2_shape);
  offset2.Populate<int32>(
      [](const xla::DimensionVector& index) { return 99 - index[0] * 5; });

  auto offset3_shape = ShapeUtil::MakeShape(S32, {8, 1});
  Literal offset3(offset3_shape);
  offset3.PopulateR2<int32>({{4}, {8}, {15}, {16}, {23}, {42}, {8}, {4}});

  Literal result =
      Execute(
          std::move(
              ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie()),
          {&input, &offset1, &offset2, &offset3})
          .ValueOrDie();
  ASSERT_TRUE(result.shape().IsTuple());

  const Shape& slice1 = ShapeUtil::GetSubshape(result.shape(), {0});
  ShapeUtil::ForEachIndex(slice1, [&](absl::Span<const int64> output_index) {
    EXPECT_EQ(output_index.size(), 2);
    float value = result.Get<float>(output_index, {0});
    auto offset_idx = offset1.Get<int32>({output_index[0], 0});
    auto input_value = input.Get<float>({offset_idx, output_index[1]});
    EXPECT_EQ(value, input_value);
    return true;
  });
  const Shape& slice2 = ShapeUtil::GetSubshape(result.shape(), {1});
  ShapeUtil::ForEachIndex(slice2, [&](absl::Span<const int64> output_index) {
    EXPECT_EQ(output_index.size(), 2);
    float value = result.Get<float>(output_index, {1});
    auto offset_idx = offset2.Get<int32>({output_index[0], 0});
    auto input_value = input.Get<float>({offset_idx, output_index[1]});
    EXPECT_EQ(value, input_value);
    return true;
  });
  const Shape& slice3 = ShapeUtil::GetSubshape(result.shape(), {2});
  ShapeUtil::ForEachIndex(slice3, [&](absl::Span<const int64> output_index) {
    EXPECT_EQ(output_index.size(), 2);
    float value = result.Get<float>(output_index, {2});
    auto offset_idx = offset3.Get<int32>({output_index[0], 0});
    auto input_value = input.Get<float>({offset_idx, output_index[1]});
    EXPECT_EQ(value, input_value);
    return true;
  });
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
