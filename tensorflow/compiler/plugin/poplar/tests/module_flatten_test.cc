/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/module_flatten.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

#include <algorithm>

namespace xla {
namespace poplarplugin {
namespace {

using ModuleFlattenTest = HloTestBase;

TEST_F(ModuleFlattenTest, TestNotFlat) {
  std::string hlo_string = R"(
HloModule top

adder {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  ROOT add = f16[] add(a0, a1)
}

cluster_1  {
  arg0 = f16[] parameter(0)
  arg1 = f16[] parameter(1)
  arg2 = f16[] parameter(2)
  c1 = f16[] call(arg0, arg1), to_apply=adder
  c2 = f16[] call(arg0, arg2), to_apply=adder
  ROOT %tuple = (f16[], f16[]) tuple(c1, c2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  EXPECT_EQ(tensorflow::error::FAILED_PRECONDITION,
            flatten.Run(module).status().code());
}

TEST_F(ModuleFlattenTest, TestNoFlatten) {
  std::string hlo_string = R"(
HloModule top

cluster_1  {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  ROOT %tuple = (f16[4], f16[4], f16[4]) tuple(arg0, arg1, arg2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  EXPECT_TRUE(flatten.Run(module).ValueOrDie());

  ASSERT_TRUE(annotations.flattened_module.get() != nullptr);
  EXPECT_EQ(annotations.flattened_module->computation_count(), 1);
  EXPECT_EQ(annotations.flattened_inst_map_fwd.size(),
            module->instruction_count());
  EXPECT_EQ(annotations.flattened_inst_map_bwd.size(),
            annotations.flattened_module->instruction_count());
}

TEST_F(ModuleFlattenTest, TestNoFusion) {
  std::string hlo_string = R"(
HloModule top

_pop_op_fused {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  ROOT add = f16[] add(a0, a1)
}

cluster_1  {
  arg0 = f16[] parameter(0)
  arg1 = f16[] parameter(1)
  arg2 = f16[] parameter(2)
  c = f16[] fusion(arg0, arg1), kind=kCustom, calls=_pop_op_fused
  ROOT %tuple = (f16[], f16[]) tuple(arg2, c)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  EXPECT_TRUE(flatten.Run(module).ValueOrDie());

  ASSERT_TRUE(annotations.flattened_module.get() != nullptr);
  EXPECT_EQ(annotations.flattened_module->computation_count(), 2);
  EXPECT_EQ(annotations.flattened_inst_map_fwd.size(),
            module->instruction_count());
  EXPECT_EQ(annotations.flattened_inst_map_bwd.size(),
            annotations.flattened_module->instruction_count());
}

TEST_F(ModuleFlattenTest, TestOneCallFlatten) {
  std::string hlo_string = R"(
HloModule top

adder {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  ROOT add = f16[] add(a0, a1)
}

cluster_1  {
  arg0 = f16[] parameter(0)
  arg1 = f16[] parameter(1)
  arg2 = f16[] parameter(2)
  c = f16[] call(arg0, arg1), to_apply=adder
  ROOT %tuple = (f16[], f16[]) tuple(arg2, c)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  EXPECT_TRUE(flatten.Run(module).ValueOrDie());

  ASSERT_TRUE(annotations.flattened_module.get() != nullptr);
  EXPECT_EQ(annotations.flattened_module->computation_count(), 1);
  EXPECT_EQ(annotations.flattened_inst_map_fwd.size(),
            module->instruction_count());
  EXPECT_EQ(annotations.flattened_inst_map_bwd.size(),
            annotations.flattened_module->instruction_count());
}

TEST_F(ModuleFlattenTest, TestTwoCallFlatten) {
  std::string hlo_string = R"(
HloModule top

adder1 {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  ROOT add = f16[] add(a0, a1)
}

adder2 {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  a2 = f16[] sine(a0)
  a3 = f16[] sine(a1)
  ROOT add = f16[] add(a2, a3)
}

cluster_1  {
  arg0 = f16[] parameter(0)
  arg1 = f16[] parameter(1)
  arg2 = f16[] parameter(2)
  c1 = f16[] call(arg0, arg1), to_apply=adder1
  c2 = f16[] call(arg0, arg2), to_apply=adder2
  ROOT %tuple = (f16[], f16[]) tuple(c1, c2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  EXPECT_TRUE(flatten.Run(module).ValueOrDie());

  ASSERT_TRUE(annotations.flattened_module.get() != nullptr);
  EXPECT_EQ(annotations.flattened_module->computation_count(), 1);
  EXPECT_EQ(annotations.flattened_inst_map_fwd.size(),
            module->instruction_count());
  EXPECT_EQ(annotations.flattened_inst_map_bwd.size(),
            annotations.flattened_module->instruction_count());
}

TEST_F(ModuleFlattenTest, TestNestedCall) {
  std::string hlo_string = R"(
HloModule top

adder1 {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  ROOT add = f16[] add(a0, a1)
}

adder2 {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  a2 = f16[] sine(a0)
  a3 = f16[] sine(a1)
  c2 = f16[] call(a2, a3), to_apply=adder1
  ROOT add = f16[] add(c2, c2)
}

cluster_1  {
  arg0 = f16[] parameter(0)
  arg1 = f16[] parameter(1)
  arg2 = f16[] parameter(2)
  c1 = f16[] call(arg0, arg1), to_apply=adder2
  ROOT %tuple = (f16[], f16[]) tuple(c1, c1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  EXPECT_TRUE(flatten.Run(module).ValueOrDie());

  ASSERT_TRUE(annotations.flattened_module.get() != nullptr);
  EXPECT_EQ(annotations.flattened_module->computation_count(), 1);
  EXPECT_EQ(annotations.flattened_inst_map_fwd.size(),
            module->instruction_count());
  EXPECT_EQ(annotations.flattened_inst_map_bwd.size(),
            annotations.flattened_module->instruction_count());
}

TEST_F(ModuleFlattenTest, TestOneWhile) {
  std::string hlo_string = R"(
HloModule top

cond {
  c_a0 = (f16[], f16[]) parameter(0)
  c_t = f16[] get-tuple-element(c_a0), index=0
  c_c = f16[] constant(1.0)
  ROOT c_p = pred[] compare(c_t, c_c), direction=LT
}

body {
  b_a0 = (f16[], f16[]) parameter(0)
  b_t = f16[] get-tuple-element(b_a0), index=0
  b_s = f16[] add(b_t, b_t)
  ROOT out = (f16[], f16[]) tuple(b_s, b_t)
}

cluster_1  {
  arg0 = f16[] parameter(0)
  arg1 = f16[] parameter(1)
  tup = (f16[], f16[]) tuple(arg0, arg1)
  c1 = (f16[], f16[]) while(tup), condition=cond, body=body
  res = f16[] get-tuple-element(c1), index=0
  ROOT %tuple = (f16[]) tuple(res)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  EXPECT_TRUE(flatten.Run(module).ValueOrDie());

  ASSERT_TRUE(annotations.flattened_module.get() != nullptr);
  EXPECT_EQ(annotations.flattened_module->computation_count(), 1);
  EXPECT_EQ(annotations.flattened_inst_map_fwd.size(),
            module->instruction_count());
  EXPECT_EQ(annotations.flattened_inst_map_bwd.size(),
            annotations.flattened_module->instruction_count());
}

TEST_F(ModuleFlattenTest, TestConditionalBinary) {
  std::string hlo_string = R"(
HloModule top

body1 {
  b1_a0 = (f16[], f16[]) parameter(0)
  b1_t = f16[] get-tuple-element(b1_a0), index=0
  b1_s = f16[] sine(b1_t)
  ROOT b1_out = (f16[], f16[]) tuple(b1_s, b1_t)
}

body2 {
  b2_a0 = f16[] parameter(0)
  b2_s = f16[] add(b2_a0, b2_a0)
  ROOT b2_out = (f16[], f16[]) tuple(b2_s, b2_s)
}

cluster_1  {
  arg0 = f16[] parameter(0)
  arg1 = f16[] parameter(1)
  arg2 = s32[] parameter(2)
  constant = f16[] constant(0)
  tup = (f16[], f16[]) tuple(arg0, arg1)
  c1 = (f16[], f16[]) conditional(arg2, tup, arg0),
      branch_computations={body1, body2}, control-predecessors={constant}
  res = f16[] get-tuple-element(c1), index=0
  ROOT %tuple = (f16[]) tuple(res)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  EXPECT_TRUE(flatten.Run(module).ValueOrDie());

  ASSERT_TRUE(annotations.flattened_module.get() != nullptr);
  EXPECT_EQ(annotations.flattened_module->computation_count(), 1);
  EXPECT_EQ(annotations.flattened_inst_map_fwd.size(),
            module->instruction_count());
  EXPECT_EQ(annotations.flattened_inst_map_bwd.size(),
            annotations.flattened_module->instruction_count());
}

TEST_F(ModuleFlattenTest, TestConditionalTrinary) {
  std::string hlo_string = R"(
HloModule top

body1 {
  b1_a0 = (f16[], f16[]) parameter(0)
  b1_t = f16[] get-tuple-element(b1_a0), index=0
  b1_s = f16[] sine(b1_t)
  ROOT b1_out = (f16[], f16[]) tuple(b1_s, b1_t)
}

body2 {
  b2_a0 = f16[] parameter(0)
  b2_s = f16[] add(b2_a0, b2_a0)
  ROOT b2_out = (f16[], f16[]) tuple(b2_s, b2_s)
}

body3 {
  b2_a0 = f16[] parameter(0)
  b2_s = f16[] subtract(b2_a0, b2_a0)
  ROOT b2_out = (f16[], f16[]) tuple(b2_s, b2_s)
}

cluster_1  {
  arg0 = f16[] parameter(0)
  arg1 = f16[] parameter(1)
  arg2 = s32[] parameter(2)
  tup = (f16[], f16[]) tuple(arg0, arg1)
  c1 = (f16[], f16[]) conditional(arg2, tup, arg0, arg1),
      branch_computations={body1, body2, body3}
  res = f16[] get-tuple-element(c1), index=0
  ROOT %tuple = (f16[]) tuple(res)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  EXPECT_TRUE(flatten.Run(module).ValueOrDie());

  ASSERT_TRUE(annotations.flattened_module.get() != nullptr);
  EXPECT_EQ(annotations.flattened_module->computation_count(), 1);
  EXPECT_EQ(annotations.flattened_inst_map_fwd.size(),
            module->instruction_count());
  EXPECT_EQ(annotations.flattened_inst_map_bwd.size(),
            annotations.flattened_module->instruction_count());
}

TEST_F(ModuleFlattenTest, TestTwoWhiles) {
  std::string hlo_string = R"(
HloModule top

cond1 {
  c_a0 = (f16[], f16[]) parameter(0)
  c_t = f16[] get-tuple-element(c_a0), index=0
  c_c = f16[] constant(1.0)
  ROOT c_p = pred[] compare(c_t, c_c), direction=LT
}

body1 {
  b_a0 = (f16[], f16[]) parameter(0)
  b_t = f16[] get-tuple-element(b_a0), index=0
  b_s = f16[] add(b_t, b_t)
  ROOT out = (f16[], f16[]) tuple(b_s, b_t)
}

cond2 {
  c_a0 = (f16[], f16[]) parameter(0)
  c_t = f16[] get-tuple-element(c_a0), index=0
  c_c = f16[] constant(1.0)
  ROOT c_p = pred[] compare(c_t, c_c), direction=LT
}

body2 {
  b_a0 = (f16[], f16[]) parameter(0)
  b_t = f16[] get-tuple-element(b_a0), index=0
  b_s = f16[] add(b_t, b_t)
  ROOT out = (f16[], f16[]) tuple(b_s, b_t)
}

cluster_1  {
  arg0 = f16[] parameter(0)
  arg1 = f16[] parameter(1)
  tup = (f16[], f16[]) tuple(arg0, arg1)
  c1 = (f16[], f16[]) while(tup), condition=cond1, body=body1
  c2 = (f16[], f16[]) while(tup), condition=cond2, body=body2
  res1 = f16[] get-tuple-element(c1), index=0
  res2 = f16[] get-tuple-element(c2), index=0
  ROOT %tuple = (f16[], f16[]) tuple(res1, res2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  EXPECT_TRUE(flatten.Run(module).ValueOrDie());

  ASSERT_TRUE(annotations.flattened_module.get() != nullptr);
  EXPECT_EQ(annotations.flattened_module->computation_count(), 1);
  EXPECT_EQ(annotations.flattened_inst_map_fwd.size(),
            module->instruction_count());
  EXPECT_EQ(annotations.flattened_inst_map_bwd.size(),
            annotations.flattened_module->instruction_count());
}

TEST_F(ModuleFlattenTest, TestNestedWhile) {
  std::string hlo_string = R"(
HloModule top

cond1 {
  c_a0 = (f16[], f16[]) parameter(0)
  c_t = f16[] get-tuple-element(c_a0), index=0
  c_c = f16[] constant(1.0)
  ROOT c_p = pred[] compare(c_t, c_c), direction=LT
}

body1 {
  b_a0 = (f16[], f16[]) parameter(0)
  b_t = f16[] get-tuple-element(b_a0), index=0
  b_s = f16[] add(b_t, b_t)
  ROOT out = (f16[], f16[]) tuple(b_s, b_t)
}

cond2 {
  c_a0 = (f16[], f16[]) parameter(0)
  c_t = f16[] get-tuple-element(c_a0), index=0
  c_c = f16[] constant(1.0)
  ROOT c_p = pred[] compare(c_t, c_c), direction=LT
}

body2 {
  b_a0 = (f16[], f16[]) parameter(0)
  b_t = f16[] get-tuple-element(b_a0), index=0
  b_s = f16[] add(b_t, b_t)
  b_t2 = (f16[], f16[]) tuple(b_s, b_s)
  ROOT b_c1 = (f16[], f16[]) while(b_t2), condition=cond1, body=body1
}

cluster_1  {
  arg0 = f16[] parameter(0)
  arg1 = f16[] parameter(1)
  tup = (f16[], f16[]) tuple(arg0, arg1)
  c1 = (f16[], f16[]) while(tup), condition=cond2, body=body2
  res1 = f16[] get-tuple-element(c1), index=0
  ROOT %tuple = (f16[], f16[]) tuple(res1, res1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  EXPECT_TRUE(flatten.Run(module).ValueOrDie());

  ASSERT_TRUE(annotations.flattened_module.get() != nullptr);
  EXPECT_EQ(annotations.flattened_module->computation_count(), 1);
  EXPECT_EQ(annotations.flattened_inst_map_fwd.size(),
            module->instruction_count());
  EXPECT_EQ(annotations.flattened_inst_map_bwd.size(),
            annotations.flattened_module->instruction_count());
}

TEST_F(ModuleFlattenTest, TestMultipleConditionalBinary) {
  std::string hlo_string = R"(
HloModule top

a_body1 {
  b1_a0 = (f16[], f16[]) parameter(0)
  b1_t = f16[] get-tuple-element(b1_a0), index=0
  b1_s = f16[] sine(b1_t)
  ROOT b1_out = (f16[], f16[]) tuple(b1_s, b1_t)
}

a_body2 {
  b2_a0 = f16[] parameter(0)
  b2_s = f16[] add(b2_a0, b2_a0)
  ROOT b2_out = (f16[], f16[]) tuple(b2_s, b2_s)
}

b_body1 {
  b1_a0 = (f16[], f16[]) parameter(0)
  b1_t = f16[] get-tuple-element(b1_a0), index=0
  b1_s = f16[] sine(b1_t)
  ROOT b1_out = (f16[], f16[]) tuple(b1_s, b1_t)
}

b_body2 {
  b2_a0 = f16[] parameter(0)
  b2_s = f16[] add(b2_a0, b2_a0)
  ROOT b2_out = (f16[], f16[]) tuple(b2_s, b2_s)
}

cluster_1  {
  arg0 = f16[] parameter(0)
  arg1 = f16[] parameter(1)
  arg2 = s32[] parameter(2)
  tup = (f16[], f16[]) tuple(arg0, arg1)
  c1 = (f16[], f16[]) conditional(arg2, tup, arg0),
      branch_computations={a_body1, a_body2}
  c2 = (f16[], f16[]) conditional(arg2, tup, arg1),
      branch_computations={b_body1, b_body2}
  res1 = f16[] get-tuple-element(c1), index=0
  res2 = f16[] get-tuple-element(c2), index=0
  ROOT %tuple = (f16[], f16[]) tuple(res1, res2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  EXPECT_TRUE(flatten.Run(module).ValueOrDie());

  ASSERT_TRUE(annotations.flattened_module.get() != nullptr);
  EXPECT_EQ(annotations.flattened_module->computation_count(), 1);
  EXPECT_EQ(annotations.flattened_inst_map_fwd.size(),
            module->instruction_count());
  EXPECT_EQ(annotations.flattened_inst_map_bwd.size(),
            annotations.flattened_module->instruction_count());
}

TEST_F(ModuleFlattenTest, TestOneCallWithControlDeps) {
  std::string hlo_string = R"(
HloModule top

adder {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  ROOT add = f16[] add(a0, a1)
}

cluster_1  {
  arg0 = f16[] parameter(0)
  arg1 = f16[] parameter(1)
  arg2 = f16[] parameter(2)
  c = f16[] call(arg0, arg1), to_apply=adder
  d = f16[] sine(arg2), control-predecessors={c}
  ROOT %tuple = (f16[], f16[]) tuple(d, c)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  EXPECT_TRUE(flatten.Run(module).ValueOrDie());

  ASSERT_TRUE(annotations.flattened_module.get() != nullptr);
  EXPECT_EQ(annotations.flattened_module->computation_count(), 1);
  EXPECT_EQ(annotations.flattened_inst_map_fwd.size(),
            module->instruction_count());
  EXPECT_EQ(annotations.flattened_inst_map_bwd.size(),
            annotations.flattened_module->instruction_count());
}
const char* while_with_call_in_cond_hlo = R"(
HloModule top

inner_cond {
  i = f16[] parameter(0)
  c = f16[] constant(1.0)
  ROOT c_p = pred[] compare(i, c), direction=LT
}

cond {
  c_a0 = (f16[], f16[]) parameter(0)
  c_t = f16[] get-tuple-element(c_a0), index=0
  ROOT c_call = pred[] call(c_t), to_apply=inner_cond
}

body {
  b_a0 = (f16[], f16[]) parameter(0)
  b_t = f16[] get-tuple-element(b_a0), index=0
  b_s = f16[] add(b_t, b_t)
  ROOT out = (f16[], f16[]) tuple(b_s, b_t)
}

cluster_1  {
  arg0 = f16[] parameter(0)
  arg1 = f16[] parameter(1)
  tup = (f16[], f16[]) tuple(arg0, arg1)
  c1 = (f16[], f16[]) while(tup), condition=cond, body=body
  res = f16[] get-tuple-element(c1), index=0
  ROOT %tuple = (f16[]) tuple(res)
}
  )";

TEST_F(ModuleFlattenTest, TestWhileWithCallInCond) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status =
      ParseAndReturnVerifiedModule(while_with_call_in_cond_hlo, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  EXPECT_TRUE(flatten.Run(module).ValueOrDie());

  ASSERT_TRUE(annotations.flattened_module.get() != nullptr);
  EXPECT_EQ(annotations.flattened_module->computation_count(), 1);
  EXPECT_EQ(annotations.flattened_inst_map_fwd.size(),
            module->instruction_count());
  EXPECT_EQ(annotations.flattened_inst_map_bwd.size(),
            annotations.flattened_module->instruction_count());
}

TEST_F(ModuleFlattenTest, TestWhileConditionFlattened) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status =
      ParseAndReturnVerifiedModule(while_with_call_in_cond_hlo, config);
  ASSERT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  ASSERT_TRUE(flatten.Run(module).ValueOrDie());

  // Check that the condition and it's subcalls have been flattened.
  auto* inner_cond = module->GetComputationWithName("inner_cond");
  ASSERT_TRUE(inner_cond);
  for (auto* inst : inner_cond->instructions()) {
    ASSERT_TRUE(annotations.flattened_inst_map_fwd[inst])
        << inst->name() << " not flattened";
  }

  auto* c_a0 = FindInstruction(module, "c_a0");
  ASSERT_TRUE(c_a0);
  ASSERT_TRUE(annotations.flattened_inst_map_fwd[c_a0]);

  auto* c_t = FindInstruction(module, "c_t");
  ASSERT_TRUE(c_t);
  ASSERT_TRUE(annotations.flattened_inst_map_fwd[c_t]);
}

TEST_F(ModuleFlattenTest, TestCallUnusedOperands) {
  std::string hlo_string = R"(
HloModule top

adder {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  a2 = f16[] parameter(2)
  ROOT add = f16[] add(a0, a1)
}

cluster_1  {
  arg0 = f16[] parameter(0)
  arg1 = f16[] parameter(1)
  arg2 = f16[] parameter(2)
  d = f16[] sine(arg2)
  ROOT c = f16[] call(arg0, arg1, d), to_apply=adder
}
  )";

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  EXPECT_TRUE(flatten.Run(module).ValueOrDie());

  ASSERT_TRUE(annotations.flattened_module.get() != nullptr);
  EXPECT_EQ(annotations.flattened_module->computation_count(), 1);
  // Add has been inlined and also the sine operations has not been elided.
  EXPECT_EQ(
      annotations.flattened_module->entry_computation()->instruction_count(),
      5);
}

TEST_F(ModuleFlattenTest, TestConditionalPredFlattened) {
  std::string hlo_string = R"(
HloModule test

body1 {
  ROOT in = (f16[], f16[]) parameter(0)
}

body2 {
  ROOT in = (f16[], f16[]) parameter(0)
}

cluster_1  {
  arg0 = f16[] parameter(0)
  arg1 = f16[] parameter(1)
  arg2 = s32[5] parameter(2)

  offset = s32[] constant(0)
  slice = s32[1] dynamic-slice(arg2, offset), dynamic_slice_sizes={1}
  pred0 = s32[] reshape(slice)

  body_args = (f16[], f16[]) tuple(arg0, arg1)

  c1 = (f16[], f16[]) conditional(pred0, body_args, body_args),
      branch_computations={body1, body2}
  res = f16[] get-tuple-element(c1), index=0

  ROOT %tuple = (f16[]) tuple(res)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  ASSERT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  ASSERT_TRUE(flatten.Run(module).ValueOrDie());

  auto* pred = FindInstruction(module, "pred0");
  ASSERT_TRUE(pred);

  auto* flat_pred = annotations.flattened_inst_map_fwd[pred];
  ASSERT_TRUE(flat_pred);
  ASSERT_EQ(flat_pred->GetModule(), annotations.flattened_module.get());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
