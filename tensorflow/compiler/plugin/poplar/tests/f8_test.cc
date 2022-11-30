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

#include <poplar/TypeConversion.hpp>
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/parse_poplar_backend_config.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_base.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
namespace {

class Fp8Test : public HloPoplarTestBase {
 public:
};

TEST_F(Fp8Test, TestConvert) {
  static const char* hlo_string = R"(
  HloModule top

  ENTRY main {
    input = (f16[2,2], u8[]) parameter(0)
    input.1 = f16[2,2] get-tuple-element(input), index=0
    input.2 = u8[] get-tuple-element(input), index=1
    input.fp8 = (u8[2,2], u8[]) custom-call(input.1, input.2), custom_call_target="ConvertToF8"
    input.fp8.1 = u8[2,2] get-tuple-element(input.fp8), index=0
    input.fp8.2 = u8[] get-tuple-element(input.fp8), index=1
    input.fp = f16[2,2] custom-call(input.fp8.1, input.fp8.2), custom_call_target="ConvertFromF8"
    ROOT root = ((f16[2,2], u8[]), (u8[2,2], u8[]), f16[2,2]) tuple(input, input.fp8, input.fp)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  Literal p0 =
      LiteralUtil::CreateR2<half>({{half(4), half(5)}, {half(-1), half(-2)}});
  Literal p1 = LiteralUtil::CreateR0<uint8_t>(
      poplar::QuarterMetadata(poplar::QuarterMetadata::Format::F143, 0)
          .getBinary());

  ASSERT_IS_OK(CustomOpReplacer().Run(module.get()).status());
  ASSERT_IS_OK(ParsePoplarBackendConfig().Run(module.get()).status());
  TF_ASSERT_OK_AND_ASSIGN(
      auto result, ExecuteNoHloPassesOnIpuModel(module.get(), {&p0, &p1}));
  EXPECT_EQ(result.size(), 5);
  poplar::QuarterMetadata metadata(result[3].Get<uint8_t>({}));

  Literal& u8_result = result[2];
  std::vector<float> fp8_output(u8_result.element_count());
  poplar::convertFromDeviceType(poplar::QUARTER, metadata,
                                u8_result.untyped_data(), fp8_output);
  Literal fp8_result(
      ShapeUtil::MakeShapeWithType<float>(p0.shape().dimensions()));
  memcpy(fp8_result.untyped_data(), fp8_output.data(), fp8_result.size_bytes());
  TF_ASSERT_OK_AND_ASSIGN(fp8_result, fp8_result.Convert(PrimitiveType::F16));
  EXPECT_TRUE(
      LiteralTestUtil::NearOrEqual(p0, result[0], ErrorSpec{1e-4, 1e-4}));
  EXPECT_TRUE(
      LiteralTestUtil::NearOrEqual(p0, fp8_result, ErrorSpec{1e-4, 1e-4}));
  EXPECT_TRUE(
      LiteralTestUtil::NearOrEqual(p0, result[4], ErrorSpec{1e-4, 1e-4}));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
