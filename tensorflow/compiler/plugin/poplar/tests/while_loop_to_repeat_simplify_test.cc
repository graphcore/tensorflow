/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/while_loop_to_repeat_simplify.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

#include <stdlib.h>

namespace xla {
namespace poplarplugin {
namespace {

using WhileLoopToRepeatSimplifyTest = HloTestBase;

/* Note that you should always run WhileLoopConstantSinking and
   WhileLoopConditionSimplify before the HloPassFix<WhileLoopToRepeatSimplify>
   pass. */

TEST_F(WhileLoopToRepeatSimplifyTest, SingleConditionalS32) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  const = s32[] constant(1)
  add = s32[] add(p_body.0, const)
  p_body.1 = s32[] get-tuple-element(p_body), index=1
  ROOT root = (s32[],s32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[] constant(10)
  repeat_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(repeat_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<WhileLoopToRepeatSimplify> wltrs;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));
  EXPECT_TRUE(changed);

  // Get the trip count
  auto* root = module.get()->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(PoplarBackendConfig cfg,
                          root->backend_config<PoplarBackendConfig>());
  ASSERT_EQ(cfg.call_config().type(),
            PoplarBackendConfig::CallConfig::RepeatLoop);
  ASSERT_EQ(cfg.call_config().repeat_config().repeat_count(), 10);
}

TEST_F(WhileLoopToRepeatSimplifyTest, SingleConditionalS32_Ge) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  const = s32[] constant(1)
  add = s32[] subtract(p_body.0, const)
  p_body.1 = s32[] get-tuple-element(p_body), index=1
  ROOT root = (s32[],s32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(0)
  ROOT result = pred[] compare(p_cond.0, const), direction=GE
}

ENTRY entry {
  const_0 = s32[] constant(999)
  const_1 = s32[] constant(0)
  repeat_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(repeat_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<WhileLoopToRepeatSimplify> wltrs;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));
  EXPECT_TRUE(changed);

  // Get the trip count
  auto* root = module.get()->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(PoplarBackendConfig cfg,
                          root->backend_config<PoplarBackendConfig>());
  ASSERT_EQ(cfg.call_config().type(),
            PoplarBackendConfig::CallConfig::RepeatLoop);
  ASSERT_EQ(cfg.call_config().repeat_config().repeat_count(), 1000);
}

TEST_F(WhileLoopToRepeatSimplifyTest, SingleConditionalS32_Gt) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  const = s32[] constant(1)
  add = s32[] subtract(p_body.0, const)
  p_body.1 = s32[] get-tuple-element(p_body), index=1
  ROOT root = (s32[],s32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(0)
  ROOT result = pred[] compare(p_cond.0, const), direction=GT
}

ENTRY entry {
  const_0 = s32[] constant(1000)
  const_1 = s32[] constant(0)
  repeat_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(repeat_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<WhileLoopToRepeatSimplify> wltrs;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));
  EXPECT_TRUE(changed);

  // Get the trip count
  auto* root = module.get()->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(PoplarBackendConfig cfg,
                          root->backend_config<PoplarBackendConfig>());
  ASSERT_EQ(cfg.call_config().type(),
            PoplarBackendConfig::CallConfig::RepeatLoop);
  ASSERT_EQ(cfg.call_config().repeat_config().repeat_count(), 1000);
}

TEST_F(WhileLoopToRepeatSimplifyTest, SingleConditionalS32_Le) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  const = s32[] constant(1)
  add = s32[] add(p_body.0, const)
  p_body.1 = s32[] get-tuple-element(p_body), index=1
  ROOT root = (s32[],s32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(999)
  ROOT result = pred[] compare(p_cond.0, const), direction=LE
}

ENTRY entry {
  const_0 = s32[] constant(100)
  const_1 = s32[] constant(999)
  repeat_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(repeat_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<WhileLoopToRepeatSimplify> wltrs;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));
  EXPECT_TRUE(changed);

  // Get the trip count
  auto* root = module.get()->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(PoplarBackendConfig cfg,
                          root->backend_config<PoplarBackendConfig>());
  ASSERT_EQ(cfg.call_config().type(),
            PoplarBackendConfig::CallConfig::RepeatLoop);
  ASSERT_EQ(cfg.call_config().repeat_config().repeat_count(), 900);
}

TEST_F(WhileLoopToRepeatSimplifyTest, SingleConditionalS32_NonConstInit) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  const = s32[] constant(1)
  add = s32[] add(p_body.0, const)
  p_body.1 = s32[] get-tuple-element(p_body), index=1
  ROOT root = (s32[],s32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(999)
  ROOT result = pred[] compare(p_cond.0, const), direction=LE
}

ENTRY entry {
  const_0 = s32[] parameter(0)
  const_1 = s32[] constant(999)
  repeat_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(repeat_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<WhileLoopToRepeatSimplify> wltrs;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopToRepeatSimplifyTest, SingleConditionalS32_NonConstDelta) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  p_body.1 = s32[] get-tuple-element(p_body), index=1
  add = s32[] add(p_body.0, p_body.1)
  ROOT root = (s32[],s32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(999)
  ROOT result = pred[] compare(p_cond.0, const), direction=LE
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[] parameter(0)
  repeat_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(repeat_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<WhileLoopToRepeatSimplify> wltrs;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopToRepeatSimplifyTest, SingleConditionalF32IncrementBy2) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[],f32[]) parameter(0)
  p_body.0 = f32[] get-tuple-element(p_body), index=0
  const = f32[] constant(2)
  add = f32[] add(p_body.0, const)
  p_body.1 = f32[] get-tuple-element(p_body), index=1
  ROOT root = (f32[],f32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (f32[],f32[]) parameter(0)
  p_cond.0 = f32[] get-tuple-element(p_cond), index=0
  const = f32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = f32[] constant(0)
  const_1 = f32[] constant(10)
  repeat_init = (f32[],f32[]) tuple(const_0, const_1)
  ROOT while = (f32[],f32[]) while(repeat_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<WhileLoopToRepeatSimplify> wltrs;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));
  EXPECT_TRUE(changed);

  // Get the trip count
  auto* root = module.get()->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(PoplarBackendConfig cfg,
                          root->backend_config<PoplarBackendConfig>());
  ASSERT_EQ(cfg.call_config().type(),
            PoplarBackendConfig::CallConfig::RepeatLoop);
  ASSERT_EQ(cfg.call_config().repeat_config().repeat_count(), 5);
}

TEST_F(WhileLoopToRepeatSimplifyTest, SingleConditionalHoistTheConstant) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  const = s32[] constant(1)
  add = s32[] add(p_body.0, const)
  p_body.1 = s32[] get-tuple-element(p_body), index=1
  ROOT root = (s32[],s32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[] constant(10)
  repeat_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(repeat_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<WhileLoopToRepeatSimplify> wltrs;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));
  EXPECT_TRUE(changed);

  // Get the trip count
  auto* root = module.get()->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(PoplarBackendConfig cfg,
                          root->backend_config<PoplarBackendConfig>());
  ASSERT_EQ(cfg.call_config().type(),
            PoplarBackendConfig::CallConfig::RepeatLoop);
  ASSERT_EQ(cfg.call_config().repeat_config().repeat_count(), 10);

  // Check the constant got hoisted out to input tuple.
  HloInstruction* repeat_inst = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsRepeatLoop(repeat_inst));
  const HloInstruction* repeat_init = repeat_inst->operand(0);
  const HloInstruction* counter = repeat_init->operand(0);
  EXPECT_EQ(counter->opcode(), HloOpcode::kConstant);
  int64 loop_counter =
      LiteralScalarToNativeType<int64>(counter->literal()).ValueOrDie();
  EXPECT_EQ(loop_counter, 10);
}

TEST_F(WhileLoopToRepeatSimplifyTest,
       SingleConditionalHoistTheConstantNoIterations) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  const = s32[] constant(1)
  neg = s32[] negate(const)
  add = s32[] add(p_body.0, neg)
  p_body.1 = s32[] get-tuple-element(p_body), index=1
  ROOT root = (s32[],s32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=GT
}

ENTRY entry {
  const_0 = s32[] constant(10)
  const_1 = s32[] constant(10)
  repeat_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(repeat_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<WhileLoopToRepeatSimplify> wltrs;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));
  EXPECT_TRUE(changed);

  // Get the trip count
  auto* root = module.get()->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(PoplarBackendConfig cfg,
                          root->backend_config<PoplarBackendConfig>());
  ASSERT_EQ(cfg.call_config().type(),
            PoplarBackendConfig::CallConfig::RepeatLoop);
  ASSERT_EQ(cfg.call_config().repeat_config().repeat_count(), 0);

  // Check the constant got hoisted out.
  HloInstruction* repeat_inst = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsRepeatLoop(repeat_inst));
  const HloInstruction* repeat_init = repeat_inst->operand(0);
  const HloInstruction* counter = repeat_init->operand(0);
  EXPECT_EQ(counter->opcode(), HloOpcode::kConstant);
  int64 loop_counter =
      LiteralScalarToNativeType<int64>(counter->literal()).ValueOrDie();
  EXPECT_EQ(loop_counter, 10);
}

TEST_F(WhileLoopToRepeatSimplifyTest,
       SingleConditionalHoistTheConstantNegativeDeltaUnsginedType) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (u32[],u32[]) parameter(0)
  p_body.0 = u32[] get-tuple-element(p_body), index=0
  const = u32[] constant(1)
  neg = u32[] negate(const)
  add = u32[] add(p_body.0, neg)
  p_body.1 = u32[] get-tuple-element(p_body), index=1
  ROOT root = (u32[],u32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (u32[],u32[]) parameter(0)
  p_cond.0 = u32[] get-tuple-element(p_cond), index=0
  const = u32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=GT
}

ENTRY entry {
  const_0 = u32[] constant(20)
  const_1 = u32[] constant(10)
  repeat_init = (u32[],u32[]) tuple(const_0, const_1)
  ROOT while = (u32[],u32[]) while(repeat_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<WhileLoopToRepeatSimplify> wltrs;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));
  EXPECT_TRUE(changed);

  // Get the trip count
  auto* root = module.get()->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(PoplarBackendConfig cfg,
                          root->backend_config<PoplarBackendConfig>());
  ASSERT_EQ(cfg.call_config().type(),
            PoplarBackendConfig::CallConfig::RepeatLoop);
  ASSERT_EQ(cfg.call_config().repeat_config().repeat_count(), 10);

  // Check the constant got hoisted out.
  HloInstruction* repeat_inst = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsRepeatLoop(repeat_inst));
  const HloInstruction* repeat_init = repeat_inst->operand(0);
  const HloInstruction* counter = repeat_init->operand(0);
  EXPECT_EQ(counter->opcode(), HloOpcode::kConstant);
  int64 loop_counter =
      LiteralScalarToNativeType<int64>(counter->literal()).ValueOrDie();
  EXPECT_EQ(loop_counter, 10);
}

TEST_F(WhileLoopToRepeatSimplifyTest,
       SingleConditionalMultipleCountersHoistTheConstant) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  p_body.2 = s32[] get-tuple-element(p_body), index=2
  const = s32[] constant(1)
  add1 = s32[] add(p_body.0, const)
  add2 = s32[] add(p_body.2, const)
  p_body.1 = s32[] get-tuple-element(p_body), index=1
  ROOT root = (s32[],s32[],s32[]) tuple(add1, p_body.1, add2)
}

condition {
  p_cond = (s32[],s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[] constant(10)
  const_2 = s32[] constant(10)
  repeat_init = (s32[],s32[],s32[]) tuple(const_0, const_1, const_2)
  ROOT while = (s32[],s32[],s32[]) while(repeat_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<WhileLoopToRepeatSimplify> wltrs;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));
  EXPECT_TRUE(changed);

  // Get the trip count
  auto* root = module.get()->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(PoplarBackendConfig cfg,
                          root->backend_config<PoplarBackendConfig>());
  ASSERT_EQ(cfg.call_config().type(),
            PoplarBackendConfig::CallConfig::RepeatLoop);
  ASSERT_EQ(cfg.call_config().repeat_config().repeat_count(), 10);

  // Check the constant got hoisted out.
  HloInstruction* repeat_inst = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsRepeatLoop(repeat_inst));
  const HloInstruction* repeat_init = repeat_inst->operand(0);
  const HloInstruction* counter = repeat_init->operand(0);
  const HloInstruction* unused_counter = repeat_init->operand(2);
  EXPECT_EQ(counter->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(unused_counter->opcode(), HloOpcode::kConstant);
  int64 loop_counter =
      LiteralScalarToNativeType<int64>(counter->literal()).ValueOrDie();
  EXPECT_EQ(loop_counter, 10);
  int64 loop_unused_counter =
      LiteralScalarToNativeType<int64>(unused_counter->literal()).ValueOrDie();
  // Note that the other counter started at 10
  EXPECT_EQ(loop_unused_counter, 20);
}

TEST_F(WhileLoopToRepeatSimplifyTest,
       SingleConditionalMultipleCountersHoistTheConstantOverflow) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[],u8[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  p_body.2 = u8[] get-tuple-element(p_body), index=2
  const = s32[] constant(1)
  const1 = u8[] constant(1)
  neg = u8[] negate(const1)
  add1 = s32[] add(p_body.0, const)
  add2 = u8[] add(p_body.2, neg)
  p_body.1 = s32[] get-tuple-element(p_body), index=1
  ROOT root = (s32[],s32[],u8[]) tuple(add1, p_body.1, add2)
}

condition {
  p_cond = (s32[],s32[],u8[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[] constant(10)
  const_2 = u8[] constant(5)
  repeat_init = (s32[],s32[],u8[]) tuple(const_0, const_1, const_2)
  ROOT while = (s32[],s32[],u8[]) while(repeat_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<WhileLoopToRepeatSimplify> wltrs;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));
  EXPECT_TRUE(changed);

  // Get the trip count
  auto* root = module.get()->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(PoplarBackendConfig cfg,
                          root->backend_config<PoplarBackendConfig>());
  ASSERT_EQ(cfg.call_config().type(),
            PoplarBackendConfig::CallConfig::RepeatLoop);
  ASSERT_EQ(cfg.call_config().repeat_config().repeat_count(), 10);

  // Check the constant got hoisted out.
  HloInstruction* repeat_inst = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsRepeatLoop(repeat_inst));
  const HloInstruction* repeat_init = repeat_inst->operand(0);
  const HloInstruction* counter = repeat_init->operand(0);
  const HloInstruction* unused_counter = repeat_init->operand(2);
  EXPECT_EQ(counter->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(unused_counter->opcode(), HloOpcode::kConstant);
  int64 loop_counter =
      LiteralScalarToNativeType<int64>(counter->literal()).ValueOrDie();
  EXPECT_EQ(loop_counter, 10);
  int64 loop_unused_counter =
      LiteralScalarToNativeType<int64>(unused_counter->literal()).ValueOrDie();
  EXPECT_EQ(loop_unused_counter, 251);
}

TEST_F(WhileLoopToRepeatSimplifyTest,
       SingleConditionalMultipleCountersHoistTheConstantResultUser) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body1 {
  p_body = (s32[],s32[],u8[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  p_body.2 = u8[] get-tuple-element(p_body), index=2
  const = s32[] constant(1)
  const1 = u8[] constant(1)
  neg = u8[] negate(const1)
  add1 = s32[] add(p_body.0, const)
  add2 = u8[] add(p_body.2, neg)
  p_body.1 = s32[] get-tuple-element(p_body), index=1
  ROOT root = (s32[],s32[],u8[]) tuple(add1, p_body.1, add2)
}

condition1 {
  p_cond = (s32[],s32[],u8[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

body2 {
  p_body = (u8[]) parameter(0)
  p_body.0 = u8[] get-tuple-element(p_body), index=0
  const = u8[] constant(1)
  add = u8[] add(p_body.0, const)
  ROOT root = (u8[]) tuple(add)
}

condition2 {
  p_cond = (u8[]) parameter(0)
  p_cond.0 = u8[] get-tuple-element(p_cond), index=0
  const = u8[] constant(200)
  ROOT result = pred[] compare(p_cond.0, const), direction=GT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[] constant(10)
  const_2 = u8[] constant(5)
  repeat_init = (s32[],s32[],u8[]) tuple(const_0, const_1, const_2)
  while = (s32[],s32[],u8[]) while(repeat_init), condition=condition1, body=body1
  ROOT gte = u8[] get-tuple-element(while), index=2
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<WhileLoopToRepeatSimplify> wltrs;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));
  EXPECT_TRUE(changed);

  // Expect that the loop got simplified to just a return constant
  HloInstruction* root = module.get()->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConstant);
  int64 result = LiteralScalarToNativeType<int64>(root->literal()).ValueOrDie();
  EXPECT_EQ(result, 251);
}

TEST_F(WhileLoopToRepeatSimplifyTest, MultipleLoops) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body1 {
  p_body = (s32[],s32[],u8[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  p_body.2 = u8[] get-tuple-element(p_body), index=2
  const = s32[] constant(1)
  const1 = u8[] constant(1)
  neg = u8[] negate(const1)
  add1 = s32[] add(p_body.0, const)
  add2 = u8[] add(p_body.2, neg)
  p_body.1 = s32[] get-tuple-element(p_body), index=1
  ROOT root = (s32[],s32[],u8[]) tuple(add1, p_body.1, add2)
}

condition1 {
  p_cond = (s32[],s32[],u8[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

body2 {
  p_body = (u8[]) parameter(0)
  p_body.0 = u8[] get-tuple-element(p_body), index=0
  const = u8[] constant(1)
  add = u8[] add(p_body.0, const)
  ROOT root = (u8[]) tuple(add)
}

condition2 {
  p_cond = (u8[]) parameter(0)
  p_cond.0 = u8[] get-tuple-element(p_cond), index=0
  const = u8[] constant(255)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[] constant(10)
  const_2 = u8[] constant(5)
  repeat_init = (s32[],s32[],u8[]) tuple(const_0, const_1, const_2)
  while = (s32[],s32[],u8[]) while(repeat_init), condition=condition1, body=body1
  gte = u8[] get-tuple-element(while), index=2
  repeat_init1 = (u8[]) tuple(gte)
  while1 = (u8[]) while(repeat_init1), condition=condition2, body=body2
  ROOT gte1 = u8[] get-tuple-element(while1), index=0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<WhileLoopToRepeatSimplify> wltrs;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));
  EXPECT_TRUE(changed);

  // Expect that the loop got simplified to just a return constant
  HloInstruction* root = module.get()->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConstant);
  int64 result = LiteralScalarToNativeType<int64>(root->literal()).ValueOrDie();
  EXPECT_EQ(result, 255);
}

TEST_F(WhileLoopToRepeatSimplifyTest, SingleConditionalDontHoistTheConstant) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  const = s32[] constant(1)
  add = s32[] add(p_body.0, const)
  p_body.1 = s32[] get-tuple-element(p_body), index=1
  add2 = s32[] add(p_body.1, add)
  ROOT root = (s32[],s32[]) tuple(add, add2)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[] constant(10)
  repeat_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(repeat_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<WhileLoopToRepeatSimplify> wltrs;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));
  EXPECT_TRUE(changed);

  // Get the trip count
  auto* root = module.get()->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(PoplarBackendConfig cfg,
                          root->backend_config<PoplarBackendConfig>());
  ASSERT_EQ(cfg.call_config().type(),
            PoplarBackendConfig::CallConfig::RepeatLoop);
  ASSERT_EQ(cfg.call_config().repeat_config().repeat_count(), 10);

  // Check the constant got hoisted out.
  HloInstruction* repeat_inst = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsRepeatLoop(repeat_inst));
  const HloInstruction* repeat_init = repeat_inst->operand(0);
  const HloInstruction* counter = repeat_init->operand(0);
  EXPECT_EQ(counter->opcode(), HloOpcode::kConstant);
  int64 loop_start =
      LiteralScalarToNativeType<int64>(counter->literal()).ValueOrDie();
  EXPECT_EQ(loop_start, 0);
}

TEST_F(WhileLoopToRepeatSimplifyTest, LoopInLoop) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body1 {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  p_body.1 = s32[] get-tuple-element(p_body), index=1
  const = s32[] constant(1)
  add1 = s32[] add(p_body.0, const)
  neg1 = s32[] negate(p_body.1)
  ROOT root = (s32[],s32[]) tuple(add1, neg1)
}

condition1 {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

body2 {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  p_body.1 = s32[] get-tuple-element(p_body), index=1
  const = s32[] constant(1)
  add = s32[] add(p_body.0, const)
  const_0 = s32[] constant(0)
  repeat_init1 = (s32[],s32[]) tuple(const_0, p_body.1)
  while1 = (s32[],s32[]) while(repeat_init1), condition=condition1, body=body1
  gte.1 = s32[] get-tuple-element(while1), index=1
  ROOT root = (s32[],s32[]) tuple(add, gte.1)
}

condition2 {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(255)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[] parameter(0)
  repeat_init = (s32[],s32[]) tuple(const_0, const_1)
  while = (s32[],s32[]) while(repeat_init), condition=condition2, body=body2
  ROOT gte1 = s32[] get-tuple-element(while), index=1
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<WhileLoopToRepeatSimplify> wltrs;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wltrs.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(module->computation_count(), 3);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
