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
#include "tensorflow/compiler/plugin/poplar/driver/invariant_passes/no_control_deps_checker.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using NoControlDepsCheckerTest = HloTestBase;

TEST_F(NoControlDepsCheckerTest, NoControlDeps) {
  const std::string hlo_string = R"(
HloModule main

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  c = f32[] constant(2)
  d = f32[] add(a, b)
  ROOT e = f32[] add(d, c)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          NoControlDepsChecker().Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(NoControlDepsCheckerTest, ControlDeps) {
  const std::string hlo_string = R"(
HloModule main

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  d = f32[] add(a, b)
  c = f32[] constant(2), control-predecessors={d}
  ROOT e = f32[] add(d, c)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto status = NoControlDepsChecker().Run(module.get());
  EXPECT_FALSE(status.ok());
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
