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
#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_update_scale_apply.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/scatter_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using MultiUpdateScaleApplyTest = HloTestBase;

TEST_F(MultiUpdateScaleApplyTest, MultiUpdateWithMultiply) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  offsets = s32[24,1] parameter(0)
  updates = f32[24,16] parameter(1)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
  lr = f32[] constant(-0.1)
  big_lr = f32[100,16] broadcast(lr), dimensions={}
  ROOT m = f32[100,16] multiply(update, big_lr)
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
  MultiUpdateScaleApply musa(annotations);
  EXPECT_TRUE(musa.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(root));
  HloInstruction* big_zero = FindInstruction(module, "big_zero");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  HloInstruction* lr = FindInstruction(module, "lr");
  EXPECT_EQ(root->operand(0), big_zero);
  EXPECT_EQ(root->operand(1), offsets);
  EXPECT_EQ(root->operand(2), updates);
  EXPECT_EQ(root->operand(3), lr);
}

TEST_F(MultiUpdateScaleApplyTest, MultiUpdateWithDivide) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  offsets = s32[24,1] parameter(0)
  updates = f32[24,16] parameter(1)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
  lr = f32[] constant(-0.1)
  big_lr = f32[100,16] broadcast(lr), dimensions={}
  ROOT m = f32[100,16] divide(update, big_lr)
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
  MultiUpdateScaleApply musa(annotations);
  EXPECT_TRUE(musa.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(root));
  HloInstruction* big_zero = FindInstruction(module, "big_zero");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  HloInstruction* lr = FindInstruction(module, "lr");
  EXPECT_EQ(root->operand(0), big_zero);
  EXPECT_EQ(root->operand(1), offsets);
  EXPECT_EQ(root->operand(2), updates);
  EXPECT_EQ(root->operand(3)->opcode(), HloOpcode::kDivide);
  EXPECT_TRUE(IsConstantOne(root->operand(3)->operand(0)));
  EXPECT_EQ(root->operand(3)->operand(1), lr);
}

TEST_F(MultiUpdateScaleApplyTest, MultiUpdateAddWithMultiply) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  offsets = s32[24,1] parameter(0)
  updates = f32[24,16] parameter(1)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  one = f32[] constant(1)
  update = f32[100,16] custom-call(big_zero, offsets, updates, one), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  lr = f32[] constant(-0.1)
  big_lr = f32[100,16] broadcast(lr), dimensions={}
  ROOT m = f32[100,16] multiply(update, big_lr)
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
  MultiUpdateScaleApply musa(annotations);
  EXPECT_TRUE(musa.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(root));
  HloInstruction* big_zero = FindInstruction(module, "big_zero");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  HloInstruction* lr = FindInstruction(module, "lr");
  EXPECT_EQ(root->operand(0), big_zero);
  EXPECT_EQ(root->operand(1), offsets);
  EXPECT_EQ(root->operand(2), updates);
  EXPECT_EQ(root->operand(3), lr);
}

TEST_F(MultiUpdateScaleApplyTest, MultiUpdateAddWithDivide) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  offsets = s32[24,1] parameter(0)
  updates = f32[24,16] parameter(1)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  one = f32[] constant(1)
  update = f32[100,16] custom-call(big_zero, offsets, updates, one), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  lr = f32[] constant(-0.1)
  big_lr = f32[100,16] broadcast(lr), dimensions={}
  ROOT m = f32[100,16] divide(update, big_lr)
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
  MultiUpdateScaleApply musa(annotations);
  EXPECT_TRUE(musa.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(root));
  HloInstruction* big_zero = FindInstruction(module, "big_zero");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  HloInstruction* lr = FindInstruction(module, "lr");
  EXPECT_EQ(root->operand(0), big_zero);
  EXPECT_EQ(root->operand(1), offsets);
  EXPECT_EQ(root->operand(2), updates);
  EXPECT_EQ(root->operand(3)->opcode(), HloOpcode::kDivide);
  EXPECT_TRUE(IsConstantOne(root->operand(3)->operand(0)));
  EXPECT_EQ(root->operand(3)->operand(1), lr);
}

TEST_F(MultiUpdateScaleApplyTest, MultiUpdateAddNonOneScale) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  offsets = s32[24,1] parameter(0)
  updates = f32[24,16] parameter(1)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  two = f32[] constant(2)
  update = f32[100,16] custom-call(big_zero, offsets, updates, two), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  lr = f32[] constant(-0.1)
  big_lr = f32[100,16] broadcast(lr), dimensions={}
  ROOT m = f32[100,16] divide(update, big_lr)
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
  MultiUpdateScaleApply musa(annotations);
  EXPECT_FALSE(musa.Run(module).ValueOrDie());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
