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
#include <string>

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xp = ::xla::poplarplugin;
namespace xla {
namespace poplarplugin {
namespace {

se::Platform* GetReferencePlatform() {
  auto result = PlatformUtil::GetPlatform("interpreter");
  return result.ValueOrDie();
}

se::Platform* GetTestPlatform() {
  auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
  EXPECT_TRUE(platform.ok());

  auto* p = dynamic_cast<xp::PoplarPlatform*>(platform.ValueOrDie());

  xla::poplarplugin::IpuOptions options;
  options.set_creator_id(IpuOptionsCreator::IPU_UTILS);

  EXPECT_EQ(p->ConfigurePoplarDevices(options), Status::OK());
  return p;
}

class MultiSliceUpdateConstantIndicesTest : public HloTestBase {
 public:
  MultiSliceUpdateConstantIndicesTest()
      : HloTestBase(GetTestPlatform(), GetReferencePlatform()) {}

  static Literal GetTestInputs(bool updates = false) {
    unsigned rows = updates ? 4 : 10;
    auto input_shape = ShapeUtil::MakeShape(F32, {rows, 4});
    Literal input(input_shape);
    input.Populate<float>([](const xla::DimensionVector& index) {
      return 1.0f * index[1] + index[0];
    });
    return input;
  }

  static Literal GetTestIndices() {
    auto indices_shape = ShapeUtil::MakeShape(S32, {4});
    Literal indices(indices_shape);
    indices.PopulateR1<int32>({0, 2, 4, 8});
    return indices;
  }

  static Literal GetScale() {
    auto scale_shape = ShapeUtil::MakeShape(F32, {1});
    Literal scale(scale_shape);
    scale.PopulateR1<float>({2.0});
    return scale;
  }

  static void VerifySlices(const Literal& result) {
    auto inputs = GetTestInputs();
    auto indices = GetTestIndices();

    const auto slice_shape = ShapeUtil::GetSubshape(result.shape(), {0});
    ShapeUtil::ForEachIndex(
        slice_shape, [&](absl::Span<const int64> output_index) {
          EXPECT_EQ(output_index.size(), 2);
          auto value = result.Get<float>(output_index, {0});
          auto idx = indices.Get<int32>({output_index[0], 0});
          auto input_value = inputs.Get<float>({idx, output_index[1]});
          EXPECT_EQ(value, input_value);
          return true;
        });
  }

  static void VerifyUpdates(const Literal& result) {
    auto inputs = GetTestInputs();
    auto updates = GetTestInputs(true);
    auto scale = GetScale().Get<float>({0});
    auto indices = GetTestIndices();
    auto indices_data = indices.data<int>();

    const auto slice_shape = ShapeUtil::GetSubshape(result.shape(), {0});
    ShapeUtil::ForEachIndex(
        slice_shape, [&](absl::Span<const int64> output_index) {
          EXPECT_EQ(output_index.size(), 2);
          auto value = result.Get<float>(output_index, {0});
          for (size_t i = 0; i < indices_data.size(); i++) {
            auto idx = indices_data.at(i);
            if (output_index[0] == idx) {
              auto input_value = inputs.Get<float>({idx, output_index[1]});
              auto update_value = updates.Get<float>({i, output_index[1]});
              EXPECT_EQ(value, scale * update_value + input_value);
              break;
            }
          }
          return true;
        });
  }
};

TEST_F(MultiSliceUpdateConstantIndicesTest, SliceNonConstantIndices) {
  std::string hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[10,4] parameter(0)
  indices = s32[4] parameter(1)
  slices = f32[4,4] custom-call(input, indices), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  ROOT t = (f32[4,4]) tuple(slices)
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  TF_ASSERT_OK_AND_ASSIGN(bool custom_ops_replaced,
                          CustomOpReplacer().Run(module.get()));
  EXPECT_TRUE(custom_ops_replaced);

  // Input to be sliced and indices to slice at.
  auto inputs = GetTestInputs();
  auto indices = GetTestIndices();

  // Execute.
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      Execute(
          std::move(
              ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie()),
          {&inputs, &indices}));
  ASSERT_TRUE(result.shape().IsTuple());

  // Verify output.
  VerifySlices(result);
}

TEST_F(MultiSliceUpdateConstantIndicesTest, SliceConstantIndices) {
  std::string hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[10,4] parameter(0)
  indices = s32[4] constant({0, 2, 4, 8})
  slices = f32[4,4] custom-call(input, indices), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  ROOT t = (f32[4,4]) tuple(slices)
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  TF_ASSERT_OK_AND_ASSIGN(bool custom_ops_replaced,
                          CustomOpReplacer().Run(module.get()));
  EXPECT_TRUE(custom_ops_replaced);

  // Input to be sliced.
  auto inputs = GetTestInputs();

  // Execute.
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      Execute(
          std::move(
              ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie()),
          {&inputs}));
  ASSERT_TRUE(result.shape().IsTuple());

  // Verify output.
  VerifySlices(result);
}

TEST_F(MultiSliceUpdateConstantIndicesTest, UpdateNonConstantIndices) {
  std::string hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[10, 4] parameter(0)
  indices = s32[4, 1] parameter(1)
  updates = f32[4, 4] parameter(2)
  scale = f32[] parameter(3)
  
  zero = f32[] constant(0)
  big_zero = f32[10, 4] broadcast(zero), dimensions={}
  
  update = f32[10, 4] custom-call(big_zero, indices, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  sum = f32[10, 4] add(update, input)
  ROOT t = (f32[10, 4]) tuple(sum)
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  TF_ASSERT_OK_AND_ASSIGN(bool custom_ops_replaced,
                          CustomOpReplacer().Run(module.get()));
  EXPECT_TRUE(custom_ops_replaced);

  // Input to be sliced and indices to slice at.
  auto inputs = GetTestInputs();
  auto updates = GetTestInputs(true);
  auto indices = GetTestIndices();
  auto scale = GetScale();

  // Execute.
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      Execute(
          std::move(
              ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie()),
          {&inputs, &indices, &updates, &scale}));
  ASSERT_TRUE(result.shape().IsTuple());

  // Verify output.
  VerifyUpdates(result);
}

TEST_F(MultiSliceUpdateConstantIndicesTest, UpdateConstantIndices) {
  std::string hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[10, 4] parameter(0)
  updates = f32[4, 4] parameter(1)
  scale = f32[] parameter(2)

  indices = s32[4, 1] constant({{0}, {2}, {4}, {8}})
  
  zero = f32[] constant(0)
  big_zero = f32[10, 4] broadcast(zero), dimensions={}
  
  update = f32[10, 4] custom-call(big_zero, indices, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  sum = f32[10, 4] add(update, input)
  ROOT t = (f32[10, 4]) tuple(sum)
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  TF_ASSERT_OK_AND_ASSIGN(bool custom_ops_replaced,
                          CustomOpReplacer().Run(module.get()));
  EXPECT_TRUE(custom_ops_replaced);

  // Input to be sliced and indices to slice at.
  auto inputs = GetTestInputs();
  auto updates = GetTestInputs(true);
  auto scale = GetScale();

  // Execute.
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      Execute(
          std::move(
              ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie()),
          {&inputs, &updates, &scale}));
  ASSERT_TRUE(result.shape().IsTuple());

  // Verify output.
  VerifyUpdates(result);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
