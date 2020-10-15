/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/passes/fusion_inliner.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using FusionInlinerTest = HloTestBase;

TEST_F(FusionInlinerTest, LowerGradientAccumulationFusion) {
  const string& hlo_string = R"(
HloModule main

_fusion {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  a0 = f32[] add(p0, p1)
  p2 = f32[] parameter(2)
  a1 = f32[] add(a0, p2)
  p3 = f32[] parameter(3)
  ROOT a2 = f32[] add(a1, p3)
}

ENTRY main {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  p2 = f32[] parameter(2)
  p3 = f32[] parameter(3)
  grad = f32[] fusion(p0, p1, p2, p3), kind=kCustom, calls=_fusion
  ROOT t = (f32[]) tuple(grad)
}
)";
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  EXPECT_TRUE(FusionInliner([](const HloInstruction* inst) {
                return IsFusion(inst, "_fusion");
              })
                  .Run(module.get())
                  .ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(
      root->operand(0),
      m::Add(m::Add(m::Add(m::Parameter(0), m::Parameter(1)), m::Parameter(2)),
             m::Parameter(3))));
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
