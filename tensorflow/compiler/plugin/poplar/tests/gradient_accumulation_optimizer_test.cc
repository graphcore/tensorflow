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
#include "tensorflow/compiler/plugin/poplar/driver/passes/gradient_accumulation_optimizer.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/scatter_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using GradientAccumulationOptimizerTest = HloTestBase;

TEST_F(GradientAccumulationOptimizerTest, ReplaceGradientAccumulatorAdd) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  p2 = f32[] parameter(2)
  x = f32[] add(p2, p1)
  update = f32[] custom-call(p0, p1), custom_call_target="GradientAccumulatorAdd"
  ROOT t = (f32[], f32[]) tuple(update, x)
}
)";
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  HloInstruction* update = root->mutable_operand(0);
  HloInstruction* x = root->mutable_operand(1);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::GradientAccumulatorAdd)(update));
  // Add a control dependency.
  TF_EXPECT_OK(x->AddControlDependencyTo(update));
  HloInstruction* p0 = module->entry_computation()->parameter_instruction(0);
  EXPECT_EQ(p0->control_predecessors().size(), 0);

  GradientAccumulationOptimizer gao;
  EXPECT_TRUE(gao.Run(module).ValueOrDie());
  update = root->mutable_operand(0);
  EXPECT_EQ(update->opcode(), HloOpcode::kAdd);
  EXPECT_THAT(update->control_predecessors(), ::testing::ElementsAre(x));
  HloInstruction* p1 = module->entry_computation()->parameter_instruction(1);
  EXPECT_THAT(p0->control_predecessors(), ::testing::ElementsAre(p1));
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
