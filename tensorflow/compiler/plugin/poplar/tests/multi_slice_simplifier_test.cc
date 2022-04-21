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

#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_slice_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
namespace {

namespace m = match;

class MultiSliceSimplifierTestSpec {
 public:
  explicit MultiSliceSimplifierTestSpec(const std::string& indices)
      : indices_(indices) {}

  std::string GetHlo() const {
    constexpr absl::string_view hlo = R"(
HloModule top

ENTRY main {
  input = f32[6,2] parameter(0)
  indices = %s
  ROOT output = f32[3,2] custom-call(input, indices), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
}

)";
    return absl::StrFormat(hlo, indices_);
  }

 private:
  const std::string indices_;
};

class MultiSliceSimplifierTest
    : public HloTestBase,
      public ::testing::WithParamInterface<MultiSliceSimplifierTestSpec> {};

INSTANTIATE_TEST_SUITE_P(
    Test, MultiSliceSimplifierTest,
    ::testing::Values(
        MultiSliceSimplifierTestSpec{"s32[3,1] constant({{0}, {2}, {4}})"},
        MultiSliceSimplifierTestSpec{"s32[3] constant({0, 2, 4})"}));

TEST_P(MultiSliceSimplifierTest, ReplaceWithStaticMultiSliceTest) {
  const std::string& hlo = GetParam().GetHlo();

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module.get()).ValueOrDie());

  CompilerAnnotations annotations(module.get());
  EXPECT_TRUE(MultiSliceSimplifier(annotations).Run(module.get()).ValueOrDie());

  const Shape output_shape = ShapeUtil::MakeShape(F32, {3, 2});
  const Shape input_shape = ShapeUtil::MakeShape(F32, {6, 2});
  const std::vector<int64> indices = {0, 2, 4};

  HloStaticMultiSliceInstruction* slice = Cast<HloStaticMultiSliceInstruction>(
      module->entry_computation()->root_instruction());

  EXPECT_THAT(slice,
              GmockMatch(m::Op()
                             .WithCustomCallTarget("StaticMultiSlice")
                             .WithShapeEqualTo(&output_shape)
                             .WithOperand(0, m::Parameter().WithShapeEqualTo(
                                                 &input_shape))));
  EXPECT_EQ(GetNumInstructions<HloInstruction>(module->entry_computation()), 2);
  EXPECT_EQ(slice->GetIndices(), indices);
}

TEST_F(MultiSliceSimplifierTest, CanNotSimplify) {
  const char* hlo = R"(
HloModule top

ENTRY main {
  input = f32[6,2] parameter(0)
  indices = s32[3] parameter(1)
  ROOT output = f32[3,2] custom-call(input, indices), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module.get()).ValueOrDie());

  CompilerAnnotations annotations(module.get());
  EXPECT_FALSE(
      MultiSliceSimplifier(annotations).Run(module.get()).ValueOrDie());
}

using MultiUpdateAddSimplifierTest = HloTestBase;

TEST_F(MultiUpdateAddSimplifierTest, ReplaceWithStaticMultiUpdateAddTest) {
  const char* hlo = R"(
HloModule top

ENTRY main {
  input = f32[6,2] parameter(0)
  indices = s32[3,1] constant({{0}, {2}, {4}})
  updates = f32[3,2] parameter(1)
  scale = f32[] parameter(2)
  ROOT output = f32[6,2] custom-call(input, indices, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}"
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module.get()).ValueOrDie());

  CompilerAnnotations annotations(module.get());
  EXPECT_TRUE(MultiSliceSimplifier(annotations).Run(module.get()).ValueOrDie());

  const Shape input_shape = ShapeUtil::MakeShape(F32, {6, 2});
  const Shape updates_shape = ShapeUtil::MakeShape(F32, {3, 2});
  const Shape scale_shape = ShapeUtil::MakeShape(F32, {});
  const std::vector<int64> indices = {0, 2, 4};

  HloStaticMultiUpdateAddInstruction* update =
      Cast<HloStaticMultiUpdateAddInstruction>(
          module->entry_computation()->root_instruction());

  EXPECT_THAT(
      update,
      GmockMatch(
          m::Op()
              .WithCustomCallTarget("StaticMultiUpdateAdd")
              .WithShapeEqualTo(&input_shape)
              .WithOperand(0, m::Parameter().WithShapeEqualTo(&input_shape))
              .WithOperand(1, m::Parameter().WithShapeEqualTo(&updates_shape))
              .WithOperand(2, m::Parameter().WithShapeEqualTo(&scale_shape))));
  EXPECT_EQ(GetNumInstructions<HloInstruction>(module->entry_computation()), 4);
  EXPECT_EQ(update->GetIndices(), indices);
}

TEST_F(MultiUpdateAddSimplifierTest, CanNotSimplify) {
  const char* hlo = R"(
HloModule top

ENTRY main {
  input = f32[6,2] parameter(0)
  indices = s32[3,1] parameter(1)
  updates = f32[3,2] parameter(2)
  scale = f32[] parameter(3)
  ROOT output = f32[6,2] custom-call(input, indices, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}"
}

)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module.get()).ValueOrDie());

  CompilerAnnotations annotations(module.get());
  EXPECT_FALSE(
      MultiSliceSimplifier(annotations).Run(module.get()).ValueOrDie());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
