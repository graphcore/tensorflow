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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_noop.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
namespace {

using StatefulNoopTest = HloTestBase;
TEST_F(StatefulNoopTest, NotRemovedByDCE) {
  std::string hlo = R"(
HloModule top

comp0 {
  ROOT empty_tuple = () tuple()
}

ENTRY e {
  e = () call(), to_apply=comp0
  ROOT t = () tuple()
}
)";

  auto config = GetModuleConfigForTest();
  {
    // First check that the comp0 is removed.
    auto module = ParseAndReturnVerifiedModule(hlo, config);
    EXPECT_TRUE(module.ok());
    auto* module0 = module.ValueOrDie().get();
    HloDCE dce;
    TF_ASSERT_OK_AND_ASSIGN(bool dce_changed, dce.Run(module0));
    EXPECT_TRUE(dce_changed);
  }
  {
    // Now check that when we insert the op into the computation, it is not
    // removed.
    auto module = ParseAndReturnVerifiedModule(hlo, config);
    EXPECT_TRUE(module.ok());
    auto* module0 = module.ValueOrDie().get();
    auto* comp0 = FindComputation(module0, "comp0");
    comp0->AddInstruction(CreateStatefulNoop());
    HloDCE dce;
    TF_ASSERT_OK_AND_ASSIGN(bool dce_changed, dce.Run(module0));
    EXPECT_FALSE(dce_changed);
  }
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
