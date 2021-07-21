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

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using LayoutStripTest = HloTestBase;

void VerifyAllShapes(HloModule* module) {
  for (auto* comp : module->computations()) {
    for (auto* inst : comp->instructions()) {
      ASSERT_TRUE(LayoutUtil::HasLayout(inst->shape())) << inst->ToString();
      if (!inst->shape().IsTuple()) {
        ASSERT_TRUE(LayoutUtil::IsDense(inst->shape().layout()))
            << inst->ToString();
        ASSERT_TRUE(
            LayoutUtil::IsMonotonicWithDim0Major(inst->shape().layout()))
            << inst->ToString();
      }
    }
  }
}

TEST_F(LayoutStripTest, ConvertAllToDefaultLayoutTest) {
  std::string hlo = R"(
HloModule top

conv_fn {
  p0_0 = f16[1,16,16,2]{0,1,2,3} parameter(0)
  p0_1 = f16[3,3,2,4]{0,1,2,3} parameter(1)
  ROOT conv = f16[1,16,16,4]{0,1,2,3} convolution(p0_0, p0_1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
}

ENTRY c1 {
  p0 = f16[1,16,16,2]{0,1,2,3} parameter(0)
  p1 = f16[1,16,16,2]{3,2,1,0} parameter(1)
  p2 = f16[3,3,2,4]{0,1,2,3} parameter(2)

  add = f16[1,16,16,2]{3,2,1,0} add(p0, p1)

  call = f16[1,16,16,4]{0,1,2,3} call(p0, p2), to_apply=conv_fn

  ROOT t = (f16[1,16,16,4], f16[1,16,16,2]) tuple(call, add)
}

)";
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  StripAllInstructionLayouts(module0);

  VerifyAllShapes(module0);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
