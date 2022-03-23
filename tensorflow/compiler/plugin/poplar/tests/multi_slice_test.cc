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

#include <string>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace poplarplugin {
namespace {

using StaticMultiSliceTest = HloTestBase;

TEST_F(StaticMultiSliceTest, NumericalTest) {
  std::string hlo = R"(
HloModule top

ENTRY main (input: f32[6,2]) -> f32[3,2] {
  input = f32[6,2] parameter(0)
  ROOT output = f32[3,2] custom-call(input), custom_call_target="StaticMultiSlice", backend_config="{\"indices\":[0,2,4]}\n"
}

)";

  auto verified_module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(verified_module.ok());
  auto module = std::move(verified_module.ValueOrDie());

  const Shape shape = ShapeUtil::MakeShape(F32, {6, 2});
  Literal input =
      LiteralUtil::CreateRandomLiteral<F32>(shape, 0, 1).ValueOrDie();

  Literal output = Execute(std::move(module), {&input}).ValueOrDie();

  // Numerical test.
  EXPECT_EQ(input.Slice({0, 0}, {1, 2}), output.Slice({0, 0}, {1, 2}));
  EXPECT_EQ(input.Slice({2, 0}, {3, 2}), output.Slice({1, 0}, {2, 2}));
  EXPECT_EQ(input.Slice({4, 0}, {5, 2}), output.Slice({2, 0}, {3, 2}));
}

class StaticMultiSliceInvalidIndicesTestSpec {
 public:
  explicit StaticMultiSliceInvalidIndicesTestSpec(const int64 index_value)
      : index_value_(index_value) {}

  std::string GetHlo() const {
    constexpr absl::string_view hlo = R"(
HloModule top

ENTRY main (input: f32[6,2]) -> f32[3,2] {
  input = f32[6,2] parameter(0)
  ROOT output = f32[3,2] custom-call(input), custom_call_target="StaticMultiSlice", backend_config="{\"indices\":[0,2,%d]}\n"
}

)";
    return absl::StrFormat(hlo, index_value_);
  }

 private:
  const int64 index_value_;
};

class StaticMultiSliceInvalidIndicesTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          StaticMultiSliceInvalidIndicesTestSpec> {};

INSTANTIATE_TEST_SUITE_P(
    Test, StaticMultiSliceInvalidIndicesTest,
    ::testing::Values(
        StaticMultiSliceInvalidIndicesTestSpec{-1},
        StaticMultiSliceInvalidIndicesTestSpec{
            static_cast<int64>(std::numeric_limits<unsigned>::max()) + 1}));

TEST_P(StaticMultiSliceInvalidIndicesTest, StatusNotOkTest) {
  const std::string hlo = GetParam().GetHlo();

  auto verified_module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(verified_module.ok());
  auto module = std::move(verified_module.ValueOrDie());

  const Shape shape = ShapeUtil::MakeShape(F32, {6, 2});
  Literal input =
      LiteralUtil::CreateRandomLiteral<F32>(shape, 0, 1).ValueOrDie();

  auto status = Execute(std::move(module), {&input});
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().error_message(),
            "StaticMultiSliceOp::Creator - cannot cast slice indices.");
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
