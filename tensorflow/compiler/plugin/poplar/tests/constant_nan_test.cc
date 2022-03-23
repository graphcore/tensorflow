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

#include "tensorflow/compiler/plugin/poplar/driver/passes/constant_nan.h"

//#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using ConstantNaNTest = HloTestBase;

TEST_F(ConstantNaNTest, Test1ConstantNaN) {
  std::string hlo_string = R"(
HloModule top

ENTRY %cluster_1  {
  %c0 = f32[4]{0} constant({1.0, 1.0, 1.0, 1.0})
  %c1 = f32[4]{0} constant({-1.1, 1.2, 1.3, 1.4})
  %c2 = f32[2,4]{1,0} constant({ { 1.0, 2.0, 3.0, 4.0 }, { 5.1, 6.2, 7.3, 8.4 } })
  %c3 = f32[4]{0} constant({10.0, 11.0, 12.0, 13.0})
  %add = f32[4]{0} add(%c0, %c1)
  ROOT %tuple = (f32[]) tuple(%add)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ConstantNaN constant_nan;
  EXPECT_TRUE(constant_nan.Run(module).ValueOrDie());
}

TEST_F(ConstantNaNTest, Test2ConstantNaN) {
  std::string hlo_string = R"(
HloModule top

ENTRY %cluster_2  {
  %c0 = f16[4]{0} constant({1.0, 1.0, 1.0, 1.0})
  %c1 = f16[4]{0} constant({-1.1, 1.2, 1.3, 1.4})
  %c2 = f16[2,4]{1,0} constant({ { 1.0, 2.0, 3.0, 4.0 }, { 5.1, 6.2, 7.3, 8.4 } })
  %c3 = f16[4]{0} constant({10.0, 11.0, 12.0, 13.0})
  %add = f16[4]{0} add(%c0, %c1)
  ROOT %tuple = (f16[]) tuple(%add)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ConstantNaN constant_nan;

  EXPECT_TRUE(constant_nan.Run(module).ValueOrDie());
}

TEST_F(ConstantNaNTest, Test3ConstantNaN) {
  std::string hlo_string = R"(
HloModule top

ENTRY %cluster_3  {
  %c0 = f32[4]{0} constant({1.0, nan, 1.0, 1.0})
  ROOT %tuple = (f32[]) tuple(%c0)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ConstantNaN constant_nan;
  EXPECT_IS_NOT_OK(constant_nan.Run(module));
}

TEST_F(ConstantNaNTest, Test4ConstantNaN) {
  std::string hlo_string = R"(
HloModule top

ENTRY %cluster_4  {
  %c0 = f16[4]{0} constant({1.0, 1.0, 1.0, nan})
  ROOT %tuple = (f16[]) tuple(%c0)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ConstantNaN constant_nan;

  EXPECT_IS_NOT_OK(constant_nan.Run(module));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
