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
#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_update_apply.h"

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

using MultiUpdateApplyTest = HloTestBase;

TEST_F(MultiUpdateApplyTest, MultiUpdateWithAddLhs) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  operand = f32[100,16] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  scale = f32[] parameter(3)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
  ROOT a = f32[100,16] add(update, operand)
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
  MultiUpdateApply mua(annotations);
  EXPECT_TRUE(mua.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(root));
  HloInstruction* operand = FindInstruction(module, "operand");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  EXPECT_EQ(root->operand(0), operand);
  EXPECT_EQ(root->operand(1), offsets);
  EXPECT_EQ(root->operand(2), updates);
  EXPECT_TRUE(IsConstantOne(root->operand(3)));
}

TEST_F(MultiUpdateApplyTest, MultiUpdateWithAddRhs) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  operand = f32[100,16] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  scale = f32[] parameter(3)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
  ROOT a = f32[100,16] add(operand, update)
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
  MultiUpdateApply mua(annotations);
  EXPECT_TRUE(mua.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(root));
  HloInstruction* operand = FindInstruction(module, "operand");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  EXPECT_EQ(root->operand(0), operand);
  EXPECT_EQ(root->operand(1), offsets);
  EXPECT_EQ(root->operand(2), updates);
  EXPECT_TRUE(IsConstantOne(root->operand(3)));
}

TEST_F(MultiUpdateApplyTest, MultiUpdateWithSubtractLhs) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  operand = f32[100,16] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  scale = f32[] parameter(3)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
  ROOT a = f32[100,16] subtract(update, operand)
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
  MultiUpdateApply mua(annotations);
  EXPECT_FALSE(mua.Run(module).ValueOrDie());
}

TEST_F(MultiUpdateApplyTest, MultiUpdateWithSubtractRhs) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  operand = f32[100,16] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  scale = f32[] parameter(3)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
  ROOT a = f32[100,16] subtract(operand, update)
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
  MultiUpdateApply mua(annotations);
  EXPECT_TRUE(mua.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(root));
  HloInstruction* operand = FindInstruction(module, "operand");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  EXPECT_EQ(root->operand(0), operand);
  EXPECT_EQ(root->operand(1), offsets);
  EXPECT_EQ(root->operand(2), updates);
  EXPECT_EQ(root->operand(3)->opcode(), HloOpcode::kNegate);
  EXPECT_TRUE(IsConstantOne(root->operand(3)->operand(0)));
}

TEST_F(MultiUpdateApplyTest, MultiUpdateAddWithAddLhs) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  operand = f32[100,16] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  scale = f32[] parameter(3)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  ROOT a = f32[100,16] add(update, operand)
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
  MultiUpdateApply mua(annotations);
  EXPECT_TRUE(mua.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(root));
  HloInstruction* operand = FindInstruction(module, "operand");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  HloInstruction* scale = FindInstruction(module, "scale");
  EXPECT_EQ(root->operand(0), operand);
  EXPECT_EQ(root->operand(1), offsets);
  EXPECT_EQ(root->operand(2), updates);
  EXPECT_EQ(root->operand(3), scale);
}

TEST_F(MultiUpdateApplyTest, MultiUpdateAddWithAddRhs) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  operand = f32[100,16] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  scale = f32[] parameter(3)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  ROOT a = f32[100,16] add(operand, update)
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
  MultiUpdateApply mua(annotations);
  EXPECT_TRUE(mua.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(root));
  HloInstruction* operand = FindInstruction(module, "operand");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  HloInstruction* scale = FindInstruction(module, "scale");
  EXPECT_EQ(root->operand(0), operand);
  EXPECT_EQ(root->operand(1), offsets);
  EXPECT_EQ(root->operand(2), updates);
  EXPECT_EQ(root->operand(3), scale);
}

TEST_F(MultiUpdateApplyTest, MultiUpdateAddWithSubtractLhs) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  operand = f32[100,16] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  scale = f32[] parameter(3)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  ROOT a = f32[100,16] subtract(update, operand)
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
  MultiUpdateApply mua(annotations);
  EXPECT_FALSE(mua.Run(module).ValueOrDie());
}

TEST_F(MultiUpdateApplyTest, MultiUpdateAddWithSubtractRhs) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  operand = f32[100,16] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  scale = f32[] parameter(3)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  ROOT a = f32[100,16] subtract(operand, update)
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
  MultiUpdateApply mua(annotations);
  EXPECT_TRUE(mua.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(root));
  HloInstruction* operand = FindInstruction(module, "operand");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  HloInstruction* scale = FindInstruction(module, "scale");
  EXPECT_EQ(root->operand(0), operand);
  EXPECT_EQ(root->operand(1), offsets);
  EXPECT_EQ(root->operand(2), updates);
  EXPECT_EQ(root->operand(3)->opcode(), HloOpcode::kNegate);
  EXPECT_EQ(root->operand(3)->operand(0), scale);
}

///////////////
TEST_F(MultiUpdateApplyTest, MultiUpdateWithAddLhsWithReshape) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  operand = f32[16,100] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  scale = f32[] parameter(3)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
  update_reshape = f32[16,100] reshape(update)
  ROOT a = f32[16,100] add(update_reshape, operand)
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
  MultiUpdateApply mua(annotations);
  EXPECT_TRUE(mua.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kReshape);
  const HloInstruction* multi_update = root->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(multi_update));
  HloInstruction* operand = FindInstruction(module, "operand");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  EXPECT_EQ(multi_update->operand(0)->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(multi_update->operand(0)->operand(0), operand);
  EXPECT_EQ(multi_update->operand(1), offsets);
  EXPECT_EQ(multi_update->operand(2), updates);
  EXPECT_TRUE(IsConstantOne(multi_update->operand(3)));
}

TEST_F(MultiUpdateApplyTest, MultiUpdateWithAddRhsWithReshape) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  operand = f32[16,100] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  scale = f32[] parameter(3)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
  update_reshape = f32[16,100] reshape(update)
  ROOT a = f32[16,100] add(operand, update_reshape)
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
  MultiUpdateApply mua(annotations);
  EXPECT_TRUE(mua.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kReshape);
  const HloInstruction* multi_update = root->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(multi_update));
  HloInstruction* operand = FindInstruction(module, "operand");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  EXPECT_EQ(multi_update->operand(0)->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(multi_update->operand(0)->operand(0), operand);
  EXPECT_EQ(multi_update->operand(1), offsets);
  EXPECT_EQ(multi_update->operand(2), updates);
  EXPECT_TRUE(IsConstantOne(multi_update->operand(3)));
}

TEST_F(MultiUpdateApplyTest, MultiUpdateWithSubtractLhsWithReshape) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  operand = f32[16,100] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  scale = f32[] parameter(3)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
  update_reshape = f32[16,100] reshape(update)
  ROOT a = f32[16,100] subtract(update_reshape, operand)
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
  MultiUpdateApply mua(annotations);
  EXPECT_FALSE(mua.Run(module).ValueOrDie());
}

TEST_F(MultiUpdateApplyTest, MultiUpdateWithSubtractRhsWithReshape) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  operand = f32[16,100] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  scale = f32[] parameter(3)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
  update_reshape = f32[16,100] reshape(update)
  ROOT a = f32[16,100] subtract(operand, update_reshape)
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
  MultiUpdateApply mua(annotations);
  EXPECT_TRUE(mua.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kReshape);
  const HloInstruction* multi_update = root->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(multi_update));
  HloInstruction* operand = FindInstruction(module, "operand");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  EXPECT_EQ(multi_update->operand(0)->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(multi_update->operand(0)->operand(0), operand);
  EXPECT_EQ(multi_update->operand(1), offsets);
  EXPECT_EQ(multi_update->operand(2), updates);
  EXPECT_EQ(multi_update->operand(3)->opcode(), HloOpcode::kNegate);
  EXPECT_TRUE(IsConstantOne(multi_update->operand(3)->operand(0)));
}

TEST_F(MultiUpdateApplyTest, MultiUpdateAddWithAddLhsWithReshape) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  operand = f32[16,100] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  scale = f32[] parameter(3)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  update_reshape = f32[16,100] reshape(update)
  ROOT a = f32[16,100] add(update_reshape, operand)
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
  MultiUpdateApply mua(annotations);
  EXPECT_TRUE(mua.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kReshape);
  const HloInstruction* multi_update = root->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(multi_update));
  HloInstruction* operand = FindInstruction(module, "operand");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  HloInstruction* scale = FindInstruction(module, "scale");
  EXPECT_EQ(multi_update->operand(0)->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(multi_update->operand(0)->operand(0), operand);
  EXPECT_EQ(multi_update->operand(1), offsets);
  EXPECT_EQ(multi_update->operand(2), updates);
  EXPECT_EQ(multi_update->operand(3), scale);
}

TEST_F(MultiUpdateApplyTest, MultiUpdateAddWithAddRhsWithReshape) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  operand = f32[16,100] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  scale = f32[] parameter(3)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  update_reshape = f32[16,100] reshape(update)
  ROOT a = f32[16,100] add(operand, update_reshape)
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
  MultiUpdateApply mua(annotations);
  EXPECT_TRUE(mua.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kReshape);
  const HloInstruction* multi_update = root->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(multi_update));
  HloInstruction* operand = FindInstruction(module, "operand");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  HloInstruction* scale = FindInstruction(module, "scale");
  EXPECT_EQ(multi_update->operand(0)->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(multi_update->operand(0)->operand(0), operand);
  EXPECT_EQ(multi_update->operand(1), offsets);
  EXPECT_EQ(multi_update->operand(2), updates);
  EXPECT_EQ(multi_update->operand(3), scale);
}

TEST_F(MultiUpdateApplyTest, MultiUpdateAddWithSubtractLhsWithReshape) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  operand = f32[16,100] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  scale = f32[] parameter(3)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  update_reshape = f32[16,100] reshape(update)
  ROOT a = f32[16,100] subtract(update_reshape, operand)
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
  MultiUpdateApply mua(annotations);
  EXPECT_FALSE(mua.Run(module).ValueOrDie());
}

TEST_F(MultiUpdateApplyTest, MultiUpdateAddWithSubtractRhsWithReshape) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  operand = f32[16,100] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  scale = f32[] parameter(3)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  update_reshape = f32[16,100] reshape(update)
  ROOT a = f32[16,100] subtract(operand, update_reshape)
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
  MultiUpdateApply mua(annotations);
  EXPECT_TRUE(mua.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kReshape);
  const HloInstruction* multi_update = root->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(multi_update));
  HloInstruction* operand = FindInstruction(module, "operand");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  HloInstruction* scale = FindInstruction(module, "scale");
  EXPECT_EQ(multi_update->operand(0)->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(multi_update->operand(0)->operand(0), operand);
  EXPECT_EQ(multi_update->operand(1), offsets);
  EXPECT_EQ(multi_update->operand(2), updates);
  EXPECT_EQ(multi_update->operand(3)->opcode(), HloOpcode::kNegate);
  EXPECT_EQ(multi_update->operand(3)->operand(0), scale);
}

TEST_F(MultiUpdateApplyTest, MultiUpdateWithBinary) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  offsets = s32[24,1] parameter(0)
  updates = f32[24,16] parameter(1)
  scale = f32[] parameter(2)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
  ROOT a = f32[100,16] add(update, update)
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
  MultiUpdateApply mua(annotations);
  EXPECT_TRUE(mua.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(root));
  HloInstruction* big_zero = FindInstruction(module, "big_zero");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  EXPECT_EQ(root->operand(0), big_zero);
  EXPECT_EQ(root->operand(1), offsets);
  EXPECT_EQ(root->operand(2)->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(root->operand(2)->operand(0), updates);
  EXPECT_EQ(root->operand(2)->operand(1), updates);
  EXPECT_TRUE(IsConstantOne(root->operand(3)));
}

TEST_F(MultiUpdateApplyTest, MultiUpdateAddWithBinary) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  offsets = s32[24,1] parameter(0)
  updates = f32[24,16] parameter(1)
  scale = f32[] parameter(2)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  ROOT a = f32[100,16] add(update, update)
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
  MultiUpdateApply mua(annotations);
  EXPECT_TRUE(mua.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(root));
  HloInstruction* big_zero = FindInstruction(module, "big_zero");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  HloInstruction* scale = FindInstruction(module, "scale");
  EXPECT_EQ(root->operand(0), big_zero);
  EXPECT_EQ(root->operand(1), offsets);
  EXPECT_EQ(root->operand(2)->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(root->operand(2)->operand(0), updates);
  EXPECT_EQ(root->operand(2)->operand(1), updates);
  EXPECT_EQ(root->operand(3), scale);
}

TEST_F(MultiUpdateApplyTest, MultiUpdateAddWithBinaryNotSupported) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  offsets = s32[24,1] parameter(0)
  updates = f32[24,16] parameter(1)
  scale = f32[] parameter(2)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  ROOT a = f32[100,16] subtract(update, update)
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
  MultiUpdateApply mua(annotations);
  EXPECT_FALSE(mua.Run(module).ValueOrDie());
}

TEST_F(MultiUpdateApplyTest, MultiUpdateAddWithBinaryUpdateTooBig) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  offsets = s32[200,1] parameter(0)
  updates = f32[200,16] parameter(1)
  scale = f32[] parameter(2)
  zero = f32[] constant(0)
  big_zero = f32[100,16] broadcast(zero), dimensions={}
  update = f32[100,16] custom-call(big_zero, offsets, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  ROOT a = f32[100,16] subtract(update, update)
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
  MultiUpdateApply mua(annotations);
  EXPECT_FALSE(mua.Run(module).ValueOrDie());
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
