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
#include "tensorflow/compiler/plugin/poplar/driver/passes/outline_instructions.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using OutlineInstructionsTest = HloTestBase;

TEST_F(OutlineInstructionsTest, MoveInstructionIntoComputation) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  %input = f32[1,1,1,4]{3,2,1,0} parameter(0)
  %weights = f32[1,1,4,16]{3,2,1,0} parameter(1)
  ROOT %convolution = f32[1,1,1,16]{3,2,1,0} convolution(f32[1,1,1,4]{3,2,1,0} %input, f32[1,1,4,16]{3,2,1,0} weights), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="cnv5_1"}
}
)";
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(root->opcode() == HloOpcode::kConvolution);

  const auto* p0 = root->operand(0);
  const auto* p1 = root->operand(1);

  OutlineInstructions oi;
  EXPECT_TRUE(oi.Run(module).ValueOrDie());

  root = module->entry_computation()->root_instruction();
  auto* comp = root->to_apply();
  auto* conv = comp->root_instruction();

  EXPECT_TRUE(Match(root, m::Op()
                              .WithOpcode(HloOpcode::kCall)
                              .WithOperand(0, m::Op().Is(p0))
                              .WithOperand(1, m::Op().Is(p1))));

  EXPECT_TRUE(Match(conv, m::Convolution(m::Parameter(0), m::Parameter(1))));

  EXPECT_TRUE(absl::StrContains(comp->name(), "instruction_cache"));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
