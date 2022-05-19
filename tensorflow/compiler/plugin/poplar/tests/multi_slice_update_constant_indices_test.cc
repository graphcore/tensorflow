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

StatusOr<Literal> GetTestInputs(PrimitiveType type, bool updates = false) {
  unsigned rows = updates ? 4 : 10;
  auto input_shape = ShapeUtil::MakeShape(F32, {rows, 4});
  Literal input(input_shape);
  input.Populate<float>([](const xla::DimensionVector& index) {
    return 1.0f * index[1] + index[0];
  });

  if (type != F32) {
    TF_ASSIGN_OR_RETURN(input, input.Convert(type));
  }
  return input;
}
Literal GetTestIndices() {
  auto indices_shape = ShapeUtil::MakeShape(S32, {4});
  Literal indices(indices_shape);
  indices.PopulateR1<int32>({0, 2, 4, 8});
  return indices;
}

StatusOr<Literal> GetScale(PrimitiveType type) {
  auto scale_shape = ShapeUtil::MakeShape(F32, {1});
  Literal scale(scale_shape);
  scale.PopulateR1<float>({2.0});

  if (type != F32) {
    TF_ASSIGN_OR_RETURN(scale, scale.Convert(type));
  }
  return scale;
}

Status VerifySlices(const Literal& result) {
  TF_ASSIGN_OR_RETURN(auto inputs, GetTestInputs(F32));
  auto indices = GetTestIndices();

  ShapeUtil::ForEachIndex(
      result.shape(), [&](absl::Span<const int64_t> output_index) {
        auto value = result.Get<float>(output_index);
        auto idx = indices.Get<int32>({output_index[0], 0});
        auto input_value = inputs.Get<float>({idx, output_index[1]});
        EXPECT_EQ(value, input_value);
        return true;
      });
  return Status::OK();
}

Status VerifyUpdates(const Literal& result) {
  TF_ASSIGN_OR_RETURN(auto inputs, GetTestInputs(F32));
  TF_ASSIGN_OR_RETURN(auto updates, GetTestInputs(F32, true));
  TF_ASSIGN_OR_RETURN(auto scale, GetScale(F32));
  auto indices = GetTestIndices();
  auto scale_data = scale.Get<float>({0});
  auto indices_data = indices.data<int>();

  ShapeUtil::ForEachIndex(
      result.shape(), [&](absl::Span<const int64_t> output_index) {
        auto value = result.Get<float>(output_index);
        for (size_t i = 0; i < indices_data.size(); i++) {
          auto idx = indices_data.at(i);
          if (output_index[0] == idx) {
            auto input_value = inputs.Get<float>({idx, output_index[1]});
            auto update_value = updates.Get<float>({i, output_index[1]});
            EXPECT_EQ(value, scale_data * update_value + input_value);
            break;
          }
        }
        return true;
      });
  return Status::OK();
}

using MultiSliceUpdateConstantIndicesTest = HloTestBase;

TEST_F(MultiSliceUpdateConstantIndicesTest, SliceNonConstantIndices) {
  std::string hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[10,4] parameter(0)
  indices = s32[4] parameter(1)
  ROOT slices = f32[4,4] custom-call(input, indices), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
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
  TF_ASSERT_OK_AND_ASSIGN(auto inputs, GetTestInputs(F32));
  auto indices = GetTestIndices();

  // Execute.
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      Execute(
          std::move(
              ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie()),
          {&inputs, &indices}));
  ASSERT_TRUE(result.shape().IsArray());

  // Verify output.
  TF_ASSERT_OK(VerifySlices(result));
}

TEST_F(MultiSliceUpdateConstantIndicesTest, SliceConstantIndices) {
  std::string hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[10,4] parameter(0)
  indices = s32[4] constant({0, 2, 4, 8})
  ROOT slices = f32[4,4] custom-call(input, indices), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
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
  TF_ASSERT_OK_AND_ASSIGN(auto inputs, GetTestInputs(F32));

  // Execute.
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      Execute(
          std::move(
              ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie()),
          {&inputs}));
  ASSERT_TRUE(result.shape().IsArray());

  // Verify output.
  TF_ASSERT_OK(VerifySlices(result));
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
  ROOT sum = f32[10, 4] add(update, input)
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
  TF_ASSERT_OK_AND_ASSIGN(auto inputs, GetTestInputs(F32));
  TF_ASSERT_OK_AND_ASSIGN(auto updates, GetTestInputs(F32, true));
  TF_ASSERT_OK_AND_ASSIGN(auto scale, GetScale(F32));
  auto indices = GetTestIndices();

  // Execute.
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      Execute(
          std::move(
              ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie()),
          {&inputs, &indices, &updates, &scale}));
  ASSERT_TRUE(result.shape().IsArray());

  // Verify output.
  TF_ASSERT_OK(VerifyUpdates(result));
}

struct MultiUpdateAddTestSpec {
  PrimitiveType element_type;

  std::string GetHlo() const {
    const std::string hlo_string = R"(
HloModule main

ENTRY main {
  input = $element_type[10, 4] parameter(0)
  updates = $element_type[4, 4] parameter(1)
  scale = $element_type[] parameter(2)

  indices = s32[4, 1] constant({{0}, {2}, {4}, {8}})
  
  zero = $element_type[] constant(0)
  big_zero = $element_type[10, 4] broadcast(zero), dimensions={}
  
  update = $element_type[10, 4] custom-call(big_zero, indices, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  ROOT sum = $element_type[10, 4] add(update, input)
}
)";
    return tensorflow::str_util::StringReplace(
        hlo_string, "$element_type",
        primitive_util::LowercasePrimitiveTypeName(element_type), true);
  }
};

std::ostream& operator<<(std::ostream& os, const MultiUpdateAddTestSpec& spec) {
  return os << "{element_type: " << spec.element_type << "}";
}

class MultiUpdateAddTest
    : public HloTestBase,
      public ::testing::WithParamInterface<MultiUpdateAddTestSpec> {};

INSTANTIATE_TEST_SUITE_P(
    MultiUpdateAddTestCases, MultiUpdateAddTest,
    ::testing::ValuesIn(std::vector<MultiUpdateAddTestSpec>{{F32}, {F16}}));

TEST_P(MultiUpdateAddTest, DoTest) {
  auto param = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(param.GetHlo()));

  TF_ASSERT_OK_AND_ASSIGN(bool custom_ops_replaced,
                          CustomOpReplacer().Run(module.get()));
  EXPECT_TRUE(custom_ops_replaced);

  // Input to be sliced and indices to slice at.
  TF_ASSERT_OK_AND_ASSIGN(auto inputs, GetTestInputs(param.element_type));
  TF_ASSERT_OK_AND_ASSIGN(auto updates,
                          GetTestInputs(param.element_type, /*update=*/true));
  TF_ASSERT_OK_AND_ASSIGN(auto scale, GetScale(param.element_type));

  // Execute.
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      Execute(
          std::move(ParseAndReturnVerifiedModule(param.GetHlo()).ValueOrDie()),
          {&inputs, &updates, &scale}));
  ASSERT_TRUE(result.shape().IsArray());

  if (param.element_type != F32) {
    TF_ASSERT_OK_AND_ASSIGN(result, result.Convert(F32));
  }

  // Verify output.
  TF_ASSERT_OK(VerifyUpdates(result));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
