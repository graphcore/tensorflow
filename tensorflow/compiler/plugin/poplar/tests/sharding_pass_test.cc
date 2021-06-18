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
#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using ShardingPassTest = HloTestBase;

TEST_F(ShardingPassTest, TestNoSharding) {
  std::string hlo_string = R"(
HloModule top

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  sin0 = f16[4] sine(arg0)
  mul0 = f16[4] multiply(sin0, arg1)
  mul1 = f16[4] multiply(mul0, arg2)
  ROOT add = f16[4] add(mul0, mul1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_FALSE(shardingPass.Run(module).ValueOrDie());

  auto insts = module->entry_computation()->instructions();
  for (auto* inst : insts) {
    EXPECT_FALSE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestAddShardingSimple) {
  std::string hlo_string = R"(
HloModule top

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  sin0 = f16[4] sine(arg0), sharding={maximal device=0}
  mul0 = f16[4] multiply(sin0, arg1), sharding={maximal device=0}
  mul1 = f16[4] multiply(mul0, arg2), sharding={maximal device=0}
  ROOT add = f16[4] add(mul0, mul1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = module->entry_computation()->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
    const auto& sharding = inst->sharding();
    EXPECT_TRUE(sharding.HasUniqueDevice());
  }
}

TEST_F(ShardingPassTest, UnsupportedSharding) {
  std::string hlo_string = R"(
HloModule top

main {
  a0 = s32[] parameter(0)
  a1 = f16[4] parameter(1)
  ROOT %tuple = () tuple(), sharding={replicated}
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_FALSE(shardingPass.Run(module).ValueOrDie());
}

TEST_F(ShardingPassTest, UnsupportedAndSupportedShardingMixed) {
  std::string hlo_string = R"(
HloModule top

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  sin0 = f16[4] sine(arg0), sharding={replicated}
  mul0 = f16[4] multiply(sin0, arg1), sharding={maximal device=0}
  mul1 = f16[4] multiply(mul0, arg2), sharding={maximal device=0}
  ROOT add = f16[4] add(mul0, mul1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = module->entry_computation()->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
    const auto& sharding = inst->sharding();
    EXPECT_TRUE(sharding.HasUniqueDevice());
  }
}

TEST_F(ShardingPassTest, TestAddShardingTuplesAfter) {
  std::string hlo_string = R"(
HloModule top

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  sin0 = f16[4] sine(arg0), sharding={maximal device=0}
  mul0 = f16[4] multiply(sin0, arg1), sharding={maximal device=0}
  mul1 = f16[4] multiply(mul0, arg2), sharding={maximal device=1}
  tuple1 = (f16[4], f16[4]) tuple(mul0, mul1), sharding={maximal device=1}
  gte0 = f16[4] get-tuple-element(tuple1), index=0
  gte1 = f16[4] get-tuple-element(tuple1), index=1
  ROOT tuple2 = (f16[4], f16[4]) tuple(gte0, gte1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }

  HloInstruction* inst;

  inst = comp->GetInstructionWithName("tuple1");
  EXPECT_TRUE(inst->sharding().IsTuple());
  EXPECT_EQ(inst->sharding().GetAsShapeTree(inst->shape()).leaf_count(), 2);
  inst = comp->GetInstructionWithName("tuple2");
  EXPECT_TRUE(inst->sharding().IsTuple());
  EXPECT_EQ(inst->sharding().GetAsShapeTree(inst->shape()).leaf_count(), 2);
  inst = comp->GetInstructionWithName("gte0");
  EXPECT_FALSE(inst->sharding().IsTuple());
  EXPECT_EQ(inst->sharding().GetUniqueDevice(), 0);
  inst = comp->GetInstructionWithName("gte1");
  EXPECT_FALSE(inst->sharding().IsTuple());
  EXPECT_EQ(inst->sharding().GetUniqueDevice(), 1);
}

TEST_F(ShardingPassTest, TestIgnoreUnconnectedComputations) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  ROOT r1 = f16[4] add(a0, a1)
}

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  sin0 = f16[4] sine(arg0), sharding={maximal device=0}
  mul0 = f16[4] multiply(sin0, arg1), sharding={maximal device=0}
  mul1 = f16[4] multiply(mul0, arg2), sharding={maximal device=0}
  ROOT add = f16[4] add(mul0, mul1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = module->entry_computation()->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
    const auto& sharding = inst->sharding();
    EXPECT_TRUE(sharding.HasUniqueDevice());
  }

  auto subcomp = module->GetComputationWithName("subcomp");
  insts = subcomp->instructions();
  for (auto* inst : insts) {
    EXPECT_FALSE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestAddShardingTuplesBefore) {
  std::string hlo_string = R"(
HloModule top

main {
  arg0 = (f16[4], f16[4]) parameter(0)
  gte0 = f16[4] get-tuple-element(arg0), index=0
  gte1 = f16[4] get-tuple-element(arg0), index=1
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  mul0 = f16[4] multiply(gte0, arg1), sharding={maximal device=0}
  mul1 = f16[4] multiply(gte1, arg2), sharding={maximal device=1}
  ROOT tuple1 = (f16[4], f16[4]) tuple(mul0, mul1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }

  HloInstruction* inst;

  inst = comp->GetInstructionWithName("arg0");
  EXPECT_TRUE(inst->sharding().IsTuple());
  EXPECT_EQ(inst->sharding().GetAsShapeTree(inst->shape()).leaf_count(), 2);
  EXPECT_EQ(inst->sharding()
                .GetAsShapeTree(inst->shape())
                .element({0})
                .GetUniqueDevice(),
            0);
  EXPECT_EQ(inst->sharding()
                .GetAsShapeTree(inst->shape())
                .element({1})
                .GetUniqueDevice(),
            1);
  inst = comp->GetInstructionWithName("gte0");
  EXPECT_FALSE(inst->sharding().IsTuple());
  EXPECT_EQ(inst->sharding().GetUniqueDevice(), 0);
  inst = comp->GetInstructionWithName("gte1");
  EXPECT_FALSE(inst->sharding().IsTuple());
  EXPECT_EQ(inst->sharding().GetUniqueDevice(), 1);
}

TEST_F(ShardingPassTest, TestAddToCallSiteNonTuple) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  s0 = f16[4] parameter(0)
  s1 = f16[4] parameter(1)
  s2 = f16[4] parameter(2)
  s3 = f16[4] add(s0, s1)
  ROOT s4 = f16[4] add(s3, s2), sharding={maximal device=0}
}

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  cal1 = f16[4] call(arg0, arg1, arg2), to_apply=subcomp
  ROOT add = f16[4] add(cal1, arg0)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
    EXPECT_FALSE(inst->sharding().IsTuple());
    EXPECT_EQ(inst->sharding().GetUniqueDevice(), 0);
  }
}

TEST_F(ShardingPassTest, TestAddToCallSiteTupleOutputCall) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  s0 = f16[4] parameter(0)
  s1 = f16[4] parameter(1)
  s2 = f16[4] parameter(2)
  s3 = f16[4] add(s0, s1), sharding={maximal device=1}
  s4 = f16[4] add(s3, s2), sharding={maximal device=0}
  ROOT t = (f16[4], f16[4]) tuple(s3, s4)
}

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  cal1 = (f16[4], f16[4]) call(arg0, arg1, arg2), to_apply=subcomp
  g0 = f16[4] get-tuple-element(cal1), index=0
  g1 = f16[4] get-tuple-element(cal1), index=1
  ROOT add = f16[4] add(g0, g1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestAddToCallSiteTupleOutputWhileFromBody) {
  std::string hlo_string = R"(
HloModule top

reduce_and {
  lhs = pred[] parameter(0)
  rhs = pred[] parameter(1)
  ROOT and = pred[] and(lhs, rhs)
}

cond {
  c0 = (f16[4], f16[4]) parameter(0)
  c1 = f16[4] get-tuple-element(c0), index=0
  c2 = f16[] constant(1.0)
  c3 = f16[4] broadcast(c2), dimensions={}
  c5 = pred[4] compare(c1, c3), direction=EQ
  c6 = pred[] constant(true)
  ROOT c7 = pred[] reduce(c5, c6), dimensions={0}, to_apply=reduce_and
}

body {
  s0 = (f16[4], f16[4]) parameter(0)
  s1 = f16[4] get-tuple-element(s0), index=0
  s2 = f16[4] get-tuple-element(s0), index=1
  s4 = f16[4] add(s1, s2), sharding={maximal device=1}
  s5 = f16[4] add(s1, s2), sharding={maximal device=0}
  ROOT t = (f16[4], f16[4]) tuple(s4, s5)
}

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  tup1 = (f16[4], f16[4]) tuple(arg0, arg1)
  whl1 = (f16[4], f16[4]) while(tup1), condition=cond, body=body
  g0 = f16[4] get-tuple-element(whl1), index=0
  g1 = f16[4] get-tuple-element(whl1), index=1
  ROOT add = f16[4] add(g0, g1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestAddToCallSiteTupleOutputWhileFromCond) {
  std::string hlo_string = R"(
HloModule top

reduce_and {
  lhs = pred[] parameter(0)
  rhs = pred[] parameter(1)
  ROOT and = pred[] and(lhs, rhs)
}

cond {
  c0 = (f16[4], f16[4]) parameter(0)
  c1 = f16[4] get-tuple-element(c0), index=0
  c2 = f16[] constant(1.0)
  c3 = f16[4] broadcast(c2), dimensions={}, sharding={maximal device=1}
  c4 = pred[4] compare(c1, c3), direction=EQ
  c5 = pred[] constant(true)
  c6 = pred[] reduce(c4, c5), dimensions={0}, to_apply=reduce_and
}

body {
  s0 = (f16[4], f16[4]) parameter(0)
  s1 = f16[4] get-tuple-element(s0), index=0
  s2 = f16[4] get-tuple-element(s0), index=1
  s4 = f16[4] add(s1, s2)
  s5 = f16[4] add(s1, s2)
  ROOT t = (f16[4], f16[4]) tuple(s4, s5)
}

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  tup1 = (f16[4], f16[4]) tuple(arg0, arg1)
  whl1 = (f16[4], f16[4]) while(tup1), condition=cond, body=body
  g0 = f16[4] get-tuple-element(whl1), index=0
  g1 = f16[4] get-tuple-element(whl1), index=1
  ROOT add = f16[4] add(g0, g1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestComputationHasUnusedInput) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  s0 = f16[4] parameter(0)
  s1 = f16[4] parameter(1)
  s2 = f16[4] parameter(2)
  s3 = f16[4] add(s0, s1), sharding={maximal device=1}
  s4 = f16[4] add(s3, s1), sharding={maximal device=0}
  ROOT t = (f16[4], f16[4]) tuple(s3, s4)
}

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  cal1 = (f16[4], f16[4]) call(arg0, arg1, arg2), to_apply=subcomp
  g0 = f16[4] get-tuple-element(cal1), index=0
  g1 = f16[4] get-tuple-element(cal1), index=1
  ROOT add = f16[4] add(g0, g1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestComputationPassesInputToOutput) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  s0 = f16[4] parameter(0)
  s1 = f16[4] parameter(1)
  s2 = f16[4] parameter(2)
  s3 = f16[4] add(s0, s1), sharding={maximal device=1}
  s4 = f16[4] add(s3, s1), sharding={maximal device=0}
  ROOT t = (f16[4], f16[4]) tuple(s3, s2)
}

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  cal1 = (f16[4], f16[4]) call(arg0, arg1, arg2), to_apply=subcomp
  g0 = f16[4] get-tuple-element(cal1), index=0
  g1 = f16[4] get-tuple-element(cal1), index=1
  ROOT add = f16[4] add(g0, g1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestComputationPassesTupleInputToOutput) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  s0 = f16[4] parameter(0)
  s1 = f16[4] parameter(1)
  s2 = (f16[4], f16[4]) parameter(2)
  s3 = f16[4] add(s0, s1), sharding={maximal device=1}
  s4 = f16[4] add(s3, s1), sharding={maximal device=0}
  ROOT t = (f16[4], f16[4]) tuple(s3, s2)
}

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = (f16[4], f16[4]) parameter(2)
  cal1 = (f16[4], (f16[4], f16[4])) call(arg0, arg1, arg2), to_apply=subcomp
  g0 = f16[4] get-tuple-element(cal1), index=0
  ROOT add = f16[4] add(g0, g0)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestComputationPassesTupleInputToOutputInWhileBody) {
  std::string hlo_string = R"(
HloModule top

cond {
  cp = (s32[], f16[4], f16[4]) parameter(0)
  gte = s32[] get-tuple-element(cp), index=0
  cc = s32[] constant(10)
  ROOT lt = pred[] compare(gte, cc), direction=LT
}

body {
  bp = (s32[], f16[4], f16[4]) parameter(0)
  g0 = s32[] get-tuple-element(bp), index=0
  g1 = f16[4] get-tuple-element(bp), index=1
  g2 = f16[4] get-tuple-element(bp), index=2
  s3 = f16[4] add(g1, g1), sharding={maximal device=1}
  s4 = f16[4] add(g2, g2), sharding={maximal device=0}
  c = s32[] constant(1)
  add = s32[] add(g0, c), sharding={maximal device=0}
  ROOT t = (s32[], f16[4], f16[4]) tuple(add, s3, s4)
}

main {
  a0 = s32[] parameter(0)
  a1 = f16[4] parameter(1)
  a2 = f16[4] parameter(2)
  t = (s32[], f16[4], f16[4]) tuple(a0, a1, a2)
  w = (s32[], f16[4], f16[4]) while(t), condition=cond, body=body
  g = f16[4] get-tuple-element(w), index=1
  ROOT add = f16[4] add(g, g)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestSomeInputsPassThrough) {
  std::string hlo_string = R"(
HloModule top

main {
  bp = (s32[], f16[4], f16[4]) parameter(0)
  g0 = s32[] get-tuple-element(bp), index=0
  g1 = f16[4] get-tuple-element(bp), index=1
  g2 = f16[4] get-tuple-element(bp), index=2
  m0 = f16[4] multiply(g1, g2), sharding={maximal device=0}
  m1 = f16[4] multiply(g1, g2), sharding={maximal device=1}
  ROOT tp = (s32[], f16[4], f16[4]) tuple(g0, m0, m1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestSomeGTEsMissingFromTuple) {
  std::string hlo_string = R"(
HloModule top

main {
  bp = (s32[], f16[4], f16[4]) parameter(0)
  g0 = s32[] get-tuple-element(bp), index=0
  g1 = f16[4] get-tuple-element(bp), index=1
  m0 = f16[4] multiply(g1, g1), sharding={maximal device=0}
  ROOT tp = (s32[], f16[4], f16[4]) tuple(g0, m0, m0)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestBatchnormsHaveSingleSharding) {
  std::string hlo_string = R"(
HloModule top

main {
  arg0 = f32[1,4,4,2] parameter(0)
  arg9 = f32[1,1,2,2] parameter(1)
  arg8 = f32[2] parameter(2)
  arg7 = f32[2] parameter(3)
  arg6 = f32[1,1,2,2] parameter(4)
  arg5 = f32[2] parameter(5)
  arg4 = f32[2] parameter(6)
  c1 = f32[1,4,4,2] convolution(arg0, arg9), window={size=1x1},
      dim_labels=b01f_01io->b01f, sharding={maximal device=0}
  bn1 = (f32[1,4,4,2], f32[2], f32[2]) batch-norm-training(c1, arg8, arg7),
      epsilon=0.001, feature_index=3
  gte0 = f32[1,4,4,2] get-tuple-element(bn1), index=0
  c2 = f32[1,4,4,2] convolution(gte0, arg6), window={size=1x1},
      dim_labels=b01f_01io->b01f, sharding={maximal device=1}
  bn2 = (f32[1,4,4,2], f32[2], f32[2]) batch-norm-training(c2, arg5, arg4),
      epsilon=0.001, feature_index=3
  gte1 = f32[1,4,4,2] get-tuple-element(bn2), index=0
  ROOT tuple = (f32[1,4,4,2]) tuple(gte1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }

  auto* bn1 = comp->GetInstructionWithName("bn1");
  EXPECT_FALSE(bn1->sharding().IsTuple());

  auto* bn2 = comp->GetInstructionWithName("bn2");
  EXPECT_FALSE(bn2->sharding().IsTuple());
}

TEST_F(ShardingPassTest, TestInfeedsDontTakeTokenSharding) {
  std::string hlo_string = R"(
HloModule top

main {
  arg0 = f32[1] parameter(0)
  tok1 = token[] after-all(), sharding={maximal device=0}
  inf1 = ((f32[1], f32[1]), token[]) infeed(tok1), sharding={maximal
  device=0} gte1 = (f32[1], f32[1]) get-tuple-element(inf1), index=0,
      sharding={maximal device=0}
  gte2 = f32[1] get-tuple-element(gte1), index=0, sharding={maximal device=0}
  gte3 = f32[1] get-tuple-element(gte1), index=1, sharding={maximal device=0}
  add1 = f32[1] add(arg0, gte2), sharding={maximal device=1}
  add2 = f32[1] add(add1, gte3), sharding={maximal device=1}

  ROOT tuple = (f32[1,4,4,2]) tuple(add2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }

  auto* inf1 = comp->GetInstructionWithName("inf1");
  ASSERT_TRUE(inf1->sharding().IsTuple());
  auto shardings = inf1->sharding().tuple_elements();
  ASSERT_TRUE(shardings[0].HasUniqueDevice());
  EXPECT_EQ(shardings[0].GetUniqueDevice(), 1);
  ASSERT_TRUE(shardings[1].HasUniqueDevice());
  EXPECT_EQ(shardings[1].GetUniqueDevice(), 1);
}

TEST_F(ShardingPassTest, TestGteOpsMatchTheirOperands) {
  std::string hlo_string = R"(
HloModule top

main {
  arg0 = f32[1,4,4,2] parameter(0)
  arg9 = f32[1,1,2,2] parameter(1)
  arg8 = f32[2] parameter(2)
  arg7 = f32[2] parameter(3)
  arg6 = f32[1,1,2,2] parameter(4)
  arg5 = f32[2] parameter(5)
  arg4 = f32[2] parameter(6)
  c1 = f32[1,4,4,2] convolution(arg0, arg9), window={size=1x1},
      dim_labels=b01f_01io->b01f, sharding={maximal device=0}
  bn1 = (f32[1,4,4,2], f32[2], f32[2]) batch-norm-training(c1, arg8, arg7),
      epsilon=0.001, feature_index=3, sharding={maximal device=0}
  gte0 = f32[1,4,4,2] get-tuple-element(bn1), index=0
  c2 = f32[1,4,4,2] convolution(gte0, arg6), window={size=1x1},
      dim_labels=b01f_01io->b01f, sharding={maximal device=1}
  bn2 = (f32[1,4,4,2], f32[2], f32[2]) batch-norm-training(c2, arg5, arg4),
      epsilon=0.001, feature_index=3, sharding={maximal device=1}
  gte1 = f32[1,4,4,2] get-tuple-element(bn2), index=0
  ROOT tuple = (f32[1,4,4,2]) tuple(gte1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }

  // gte0 should match its operand bn1, not its user c2
  auto* gte0 = comp->GetInstructionWithName("gte0");
  ASSERT_TRUE(gte0->sharding().HasUniqueDevice());
  EXPECT_EQ(gte0->sharding().GetUniqueDevice(), 0);
}

TEST_F(ShardingPassTest, TestCalledSubcompIsEntirelyEmptyFillFromUsers) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  s0 = f16[4] parameter(0)
  s1 = f16[4] parameter(1)
  ROOT s3 = f16[4] add(s0, s1)
}

main {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  cal1 = f16[4] call(a0, a1), to_apply=subcomp
  sin1 = f16[4] sine(cal1), sharding={maximal device=1}
  ROOT t = (f16[4]) tuple(sin1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  for (auto* comp : module->computations()) {
    auto insts = comp->instructions();
    for (auto* inst : insts) {
      ASSERT_TRUE(inst->has_sharding());
      EXPECT_EQ(inst->sharding().GetUniqueDevice(), 1);
    }
  }
}

TEST_F(ShardingPassTest, TestCalledSubcompIsEntirelyEmptyFillFromParams) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  s0 = f16[4] parameter(0)
  s1 = f16[4] parameter(1)
  ROOT s3 = f16[4] add(s0, s1)
}

main {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  c0 = f16[4] cosine(a0), sharding={maximal device=1}
  c1 = f16[4] cosine(a1), sharding={maximal device=1}
  cal1 = f16[4] call(a0, a1), to_apply=subcomp
  sin1 = f16[4] sine(cal1)
  ROOT t = (f16[4]) tuple(sin1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  for (auto* comp : module->computations()) {
    auto insts = comp->instructions();
    for (auto* inst : insts) {
      ASSERT_TRUE(inst->has_sharding());
      EXPECT_EQ(inst->sharding().GetUniqueDevice(), 1);
    }
  }
}

TEST_F(ShardingPassTest, TestCalledSubcompIsEntirelyEmptyFillFromUsersTuple) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  s0 = f16[4] parameter(0)
  s1 = f16[4] parameter(1)
  s2 = f16[4] add(s0, s1)
  s3 = f16[4] add(s0, s1)
  s4 = (f16[4], f16[4]) tuple(s2, s3)
}

main {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  cal1 = (f16[4], f16[4]) call(a0, a1), to_apply=subcomp
  gte0 = f16[4] get-tuple-element(cal1), index=0
  gte1 = f16[4] get-tuple-element(cal1), index=1
  sin1 = f16[4] sine(gte0), sharding={maximal device=1}
  sin2 = f16[4] sine(gte1), sharding={maximal device=1}
  ROOT t = (f16[4], f16[4]) tuple(sin1, sin2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  for (auto* comp : module->computations()) {
    auto insts = comp->instructions();
    for (auto* inst : insts) {
      ASSERT_TRUE(inst->has_sharding());
      EXPECT_EQ(inst->sharding().GetUniqueDevice(), 1);
    }
  }
}

TEST_F(ShardingPassTest, TestCalledSubcompIsEntirelyEmptyFillFromParamsTuple) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  s0 = f16[4] parameter(0)
  s1 = f16[4] parameter(1)
  s2 = f16[4] add(s0, s1)
  s3 = f16[4] add(s0, s1)
  s4 = (f16[4], f16[4]) tuple(s2, s3)
}

main {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  c0 = f16[4] cosine(a0), sharding={maximal device=1}
  c1 = f16[4] cosine(a1), sharding={maximal device=1}
  cal1 = (f16[4], f16[4]) call(c0, c1), to_apply=subcomp
  gte0 = f16[4] get-tuple-element(cal1), index=0
  gte1 = f16[4] get-tuple-element(cal1), index=1
  sin1 = f16[4] sine(gte0)
  sin2 = f16[4] sine(gte1)
  ROOT t = (f16[4], f16[4]) tuple(sin1, sin2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  for (auto* comp : module->computations()) {
    auto insts = comp->instructions();
    for (auto* inst : insts) {
      ASSERT_TRUE(inst->has_sharding());
      if (!inst->sharding().IsTuple()) {
        EXPECT_EQ(inst->sharding().GetUniqueDevice(), 1);
      }
    }
  }
}

TEST_F(ShardingPassTest, TestEmptyCompInEmptyComp) {
  std::string hlo_string = R"(
HloModule top

subsubcomp {
  s0 = f16[4] parameter(0)
  s1 = f16[4] parameter(1)
  s2 = f16[4] add(s0, s1)
  s3 = f16[4] add(s0, s1)
  s4 = (f16[4], f16[4]) tuple(s2, s3)
}

subcomp {
  s0 = f16[4] parameter(0)
  s1 = f16[4] parameter(1)
  s2 = f16[4] add(s0, s1)
  s3 = f16[4] add(s0, s1)
  cl = (f16[4], f16[4]) call(s2, s3), to_apply=subsubcomp
  g0 = f16[4] get-tuple-element(cl), index=0
  g1 = f16[4] get-tuple-element(cl), index=1
  s4 = (f16[4], f16[4]) tuple(g0, g1)
}

main {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  c0 = f16[4] cosine(a0), sharding={maximal device=1}
  c1 = f16[4] cosine(a1), sharding={maximal device=1}
  cal1 = (f16[4], f16[4]) call(c0, c1), to_apply=subcomp
  gte0 = f16[4] get-tuple-element(cal1), index=0
  gte1 = f16[4] get-tuple-element(cal1), index=1
  sin1 = f16[4] sine(gte0)
  sin2 = f16[4] sine(gte1)
  ROOT t = (f16[4], f16[4]) tuple(sin1, sin2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  for (auto* comp : module->computations()) {
    auto insts = comp->instructions();
    for (auto* inst : insts) {
      ASSERT_TRUE(inst->has_sharding());
      if (!inst->sharding().IsTuple()) {
        EXPECT_EQ(inst->sharding().GetUniqueDevice(), 1);
      }
    }
  }
}

TEST_F(ShardingPassTest, TestConditionalAsSelect) {
  std::string hlo_string = R"(
HloModule top

reduce_and {
  lhs = pred[] parameter(0)
  rhs = pred[] parameter(1)
  ROOT and = pred[] and(lhs, rhs)
}

cond1 {
  s0 = (f16[4], f16[4]) parameter(0)
  s1 = f16[4] get-tuple-element(s0), index=0
  s2 = f16[4] get-tuple-element(s0), index=1
  s3 = f16[4] add(s1, s2)
  s4 = f16[4] add(s1, s2), sharding={maximal device=1}
  s5 = (f16[4], f16[4]) tuple(s3, s4)
}

cond2 {
  s0 = (f16[4], f16[4]) parameter(0)
  s1 = f16[4] get-tuple-element(s0), index=0
  s2 = f16[4] get-tuple-element(s0), index=1
  s3 = f16[4] add(s1, s2)
  s4 = f16[4] add(s1, s2)
  s5 = (f16[4], f16[4]) tuple(s3, s4)
}

main {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  tu = (f16[4], f16[4]) tuple(a0, a1)
  lt = pred[4] compare(a0, a1), direction=LT
  c0 = pred[] constant(true)
  lt_r = pred[] reduce(lt, c0), dimensions={0}, to_apply=reduce_and
  con1 = (f16[4], f16[4]) conditional(lt_r, tu, tu),
      true_computation=cond1, false_computation=cond2
  gte0 = f16[4] get-tuple-element(con1), index=0
  gte1 = f16[4] get-tuple-element(con1), index=1
  sin1 = f16[4] sine(gte0)
  sin2 = f16[4] sine(gte1)
  ROOT t = (f16[4], f16[4]) tuple(sin1, sin2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  for (auto* comp : module->computations()) {
    if (comp->name() == "reduce_and") {
      continue;
    }
    auto insts = comp->instructions();
    for (auto* inst : insts) {
      ASSERT_TRUE(inst->has_sharding());
      if (!inst->sharding().IsTuple()) {
        EXPECT_EQ(inst->sharding().GetUniqueDevice(), 1);
      }
    }
  }
}

TEST_F(ShardingPassTest, TestConditionalAsSwitch) {
  std::string hlo_string = R"(
HloModule top

cond1 {
  p0 = (f16[4], f16[4]) parameter(0)
  s0 = f16[4] get-tuple-element(p0), index=0
  s1 = f16[4] get-tuple-element(p0), index=1
  s2 = f16[4] subtract(s0, s1), sharding={maximal device=0}
  s3 = f16[4] add(s0, s1)
  s4 = (f16[4], f16[4]) tuple(s2, s3)
}

cond2 {
  p0 = (f16[4], f16[4]) parameter(0)
  s0 = f16[4] get-tuple-element(p0), index=0
  s1 = f16[4] get-tuple-element(p0), index=1
  s2 = f16[4] add(s0, s1)
  s3 = f16[4] subtract(s0, s1)
  s4 = (f16[4], f16[4]) tuple(s2, s3)
}

cond3 {
  p0 = (f16[4], f16[4]) parameter(0)
  s0 = f16[4] get-tuple-element(p0), index=0
  s1 = f16[4] get-tuple-element(p0), index=1
  s2 = f16[4] multiply(s0, s1)
  s3 = f16[4] divide(s0, s1)
  s4 = (f16[4], f16[4]) tuple(s2, s3)
}

main {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  se = s32[] parameter(2)
  t0 = (f16[4], f16[4]) tuple(a0, a1)
  con1 = (f16[4], f16[4]) conditional(se, t0, t0, t0),
      branch_computations={cond1, cond2, cond3}
  gte0 = f16[4] get-tuple-element(con1), index=0
  gte1 = f16[4] get-tuple-element(con1), index=1
  sin1 = f16[4] sine(gte0)
  sin2 = f16[4] sine(gte1)
  ROOT t = (f16[4], f16[4]) tuple(sin1, sin2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  for (auto* comp : module->computations()) {
    auto insts = comp->instructions();
    for (auto* inst : insts) {
      ASSERT_TRUE(inst->has_sharding());
      if (!inst->sharding().IsTuple()) {
        EXPECT_EQ(inst->sharding().GetUniqueDevice(), 0);
      }
    }
  }
}

TEST_F(ShardingPassTest, TestConditionalAsSwitchMismatchingSubcomps) {
  std::string hlo_string = R"(
HloModule top

cond1 {
  p0 = (f16[4], f16[4]) parameter(0)
  s0 = f16[4] get-tuple-element(p0), index=0
  s1 = f16[4] get-tuple-element(p0), index=1
  s2 = f16[4] subtract(s0, s1), sharding={maximal device=0}
  s3 = f16[4] add(s0, s1)
  s4 = (f16[4], f16[4]) tuple(s2, s3)
}

cond2 {
  p0 = (f16[4], f16[4]) parameter(0)
  s0 = f16[4] get-tuple-element(p0), index=0
  s1 = f16[4] get-tuple-element(p0), index=1
  s2 = f16[4] add(s0, s1), sharding={maximal device=1}
  s3 = f16[4] subtract(s0, s1)
  s4 = (f16[4], f16[4]) tuple(s2, s3)
}

cond3 {
  p0 = (f16[4], f16[4]) parameter(0)
  s0 = f16[4] get-tuple-element(p0), index=0
  s1 = f16[4] get-tuple-element(p0), index=1
  s2 = f16[4] multiply(s0, s1), sharding={maximal device=2}
  s3 = f16[4] divide(s0, s1)
  s4 = (f16[4], f16[4]) tuple(s2, s3)
}

main {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  se = s32[] parameter(2)
  t0 = (f16[4], f16[4]) tuple(a0, a1)
  con1 = (f16[4], f16[4]) conditional(se, t0, t0, t0),
      branch_computations={cond1, cond2, cond3}
  gte0 = f16[4] get-tuple-element(con1), index=0
  gte1 = f16[4] get-tuple-element(con1), index=1
  sin1 = f16[4] sine(gte0)
  sin2 = f16[4] sine(gte1)
  ROOT t = (f16[4], f16[4]) tuple(sin1, sin2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  for (auto* comp : module->computations()) {
    auto insts = comp->instructions();
    for (auto* inst : insts) {
      ASSERT_TRUE(inst->has_sharding());
    }
  }

  auto* comp1 = module->GetComputationWithName("cond1");
  auto* comp2 = module->GetComputationWithName("cond2");
  auto* comp3 = module->GetComputationWithName("cond3");
  EXPECT_TRUE(comp1->parameter_instruction(0)->has_sharding());
  EXPECT_TRUE(comp2->parameter_instruction(0)->has_sharding());
  EXPECT_TRUE(comp3->parameter_instruction(0)->has_sharding());
  EXPECT_NE(comp1->parameter_instruction(0)->sharding(),
            comp2->parameter_instruction(0)->sharding());
  EXPECT_NE(comp1->parameter_instruction(0)->sharding(),
            comp3->parameter_instruction(0)->sharding());
  EXPECT_TRUE(comp1->root_instruction()->has_sharding());
  EXPECT_TRUE(comp2->root_instruction()->has_sharding());
  EXPECT_TRUE(comp3->root_instruction()->has_sharding());
  EXPECT_EQ(comp1->root_instruction()->sharding(),
            comp2->root_instruction()->sharding());
  EXPECT_EQ(comp1->root_instruction()->sharding(),
            comp3->root_instruction()->sharding());
}

TEST_F(ShardingPassTest, TestConditionalAsSwitchCopyToSubcomp) {
  std::string hlo_string = R"(
HloModule top

cond1 {
  p0 = (f16[4], f16[4]) parameter(0)
  s0 = f16[4] get-tuple-element(p0), index=0
  s1 = f16[4] get-tuple-element(p0), index=1
  s2 = f16[4] subtract(s0, s1)
  s3 = f16[4] add(s0, s1)
  s4 = (f16[4], f16[4]) tuple(s2, s3)
}

cond2 {
  p0 = (f16[4], f16[4]) parameter(0)
  s0 = f16[4] get-tuple-element(p0), index=0
  s1 = f16[4] get-tuple-element(p0), index=1
  s2 = f16[4] add(s0, s1)
  s3 = f16[4] subtract(s0, s1)
  s4 = (f16[4], f16[4]) tuple(s2, s3)
}

cond3 {
  p0 = (f16[4], f16[4]) parameter(0)
  s0 = f16[4] get-tuple-element(p0), index=0
  s1 = f16[4] get-tuple-element(p0), index=1
  s2 = f16[4] multiply(s0, s1)
  s3 = f16[4] divide(s0, s1)
  s4 = (f16[4], f16[4]) tuple(s2, s3)
}

main {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  se = s32[] parameter(2)
  t0 = (f16[4], f16[4]) tuple(a0, a1)
  con1 = (f16[4], f16[4]) conditional(se, t0, t0, t0),
      branch_computations={cond1, cond2, cond3}
  gte0 = f16[4] get-tuple-element(con1), index=0
  gte1 = f16[4] get-tuple-element(con1), index=1
  sin1 = f16[4] sine(gte0), sharding={maximal device=0}
  sin2 = f16[4] sine(gte1), sharding={maximal device=0}
  ROOT t = (f16[4], f16[4]) tuple(sin1, sin2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  for (auto* comp : module->computations()) {
    auto insts = comp->instructions();
    for (auto* inst : insts) {
      ASSERT_TRUE(inst->has_sharding());
      if (!inst->sharding().IsTuple()) {
        EXPECT_EQ(inst->sharding().GetUniqueDevice(), 0);
      }
    }
  }
}

TEST_F(ShardingPassTest, TestSettingGteFromOperandIsConsideredProgress) {
  std::string hlo_string = R"(
HloModule top

main {
  a0 = f16[] parameter(0), sharding={maximal device=0}
  a1 = f16[] add(a0, a0), sharding={maximal device=0}
  a2 = (f16[], f16[]) tuple(a0, a0), sharding={{maximal device=0}, {maximal
  device=0}} a3 = ((f16[], f16[])) tuple(a2), sharding={{maximal device=0},
  {maximal device=0}} a4 = (f16[], f16[]) get-tuple-element(a3), index=0 a5 =
  f16[] get-tuple-element(a4), index=0 ROOT tuple = (f16[], f16[]) tuple(a1,
  a5)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto* comp = module->entry_computation();
  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestWhileRepeatHasMatchingInputAndOutputSharding) {
  std::string hlo_string = R"(
HloModule top

_pop_op_wide_const.1 {
  c = f16[] constant(0)
  ROOT b = f16[5,1,512] broadcast(c), dimensions={}
}

_pop_op_wide_const.2 {
  c = f16[] constant(0)
  ROOT b = f16[5,512] broadcast(c), dimensions={}
}

max_half {
  x = f16[] parameter(0)
  y = f16[] parameter(1)
  ROOT m = f16[] maximum(x, y)
}

add_float {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT a = f32[] add(x, y)
}

body {
  arg_tuple.1 = (f16[1,5,1000], s32[1,5], f16[512,1000]) parameter(0)
  gte.68 = f16[1,5,1000] get-tuple-element(arg_tuple.1), index=0
  gte.69 = s32[1,5] get-tuple-element(arg_tuple.1), index=1
  gte.75 = f16[512,1000] get-tuple-element(arg_tuple.1), index=2
  transpose.1 = f16[1000,512] transpose(gte.75), dimensions={1,0},
  sharding={maximal device=0} reshape.29 = s32[5] reshape(gte.69) fusion.11 =
  f16[5,512] fusion(), kind=kCustom, calls=_pop_op_wide_const.2 reshape.30 =
  f16[1,5,512] reshape(fusion.11), sharding={maximal device=0} transpose.2 =
  f16[5,1,512]{2,0,1} transpose(reshape.30), dimensions={1,0,2},
  sharding={maximal device=1} fusion.12 = f16[5,1,512] fusion(),
  kind=kCustom, calls=_pop_op_wide_const.1 transpose.3 = f16[1,5,512]{2,0,1}
  transpose(fusion.12), dimensions={1,0,2}, sharding={maximal device=1}
  reshape.31 = f16[1,1000,512] reshape(transpose.1)
  transpose.4 = f16[1,512,1000] transpose(reshape.31), dimensions={0,2,1}
  dot.1 = f16[1,5,1000] dot(transpose.3, transpose.4), lhs_batch_dims={0},
  lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
  constant.48 = f16[] constant(-inf)
  reduce = f16[1,5] reduce(dot.1, constant.48), dimensions={2},
  to_apply=max_half broadcast.15 = f16[1,5,1000] broadcast(reduce),
  dimensions={0,1} subtract = f16[1,5,1000] subtract(dot.1, broadcast.15)
  exponential = f16[1,5,1000] exponential(subtract)
  constant.43 = f16[] constant(0)
  reduce.1 = f16[1,5] reduce(exponential, constant.43), dimensions={2},
  to_apply=add_float broadcast.16 = f16[1,5,1000] broadcast(reduce.1),
  dimensions={0,1} divide = f16[1,5,1000] divide(exponential, broadcast.16)
  ROOT tuple.10 = (f16[1,5,1000], s32[1,5], f16[512,1000]) tuple(divide,
  gte.69, gte.75)
}

ENTRY main {
  arg0.1 = f16[1,5,1000] parameter(0)
  arg1.2 = s32[1,5] parameter(1)
  arg2.3 = f16[512,1000] parameter(2)
  tuple.13 = (f16[1,5,1000], s32[1,5], f16[512,1000]) tuple(arg0.1, arg1.2,
  arg2.3) call.4 = (f16[1,5,1000], s32[1,5], f16[512,1000]) call(tuple.13),
  to_apply=body,
  backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
  gte.270 = f16[1,5,1000] get-tuple-element(call.4), index=0
  ROOT tuple.284 = (f16[1,5,1000]) tuple(gte.270)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto* comp = module->entry_computation();
  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }

  auto* body = module->GetComputationWithName("body");
  EXPECT_TRUE(body->parameter_instruction(0)->has_sharding());
  EXPECT_TRUE(body->root_instruction()->has_sharding());
  EXPECT_EQ(body->parameter_instruction(0)->sharding(),
            body->root_instruction()->sharding());
}

TEST_F(ShardingPassTest, TestPipeliningSharding) {
  std::string hlo_string = R"(
HloModule top

add_float {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT a = f32[] add(f32[] x, f32[] y)
}

stage_0_fwd {
  stage_0_fwd_input0 = f32[1,4,4,2] parameter(0)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2]) tuple(stage_0_fwd_input0)
}

comp0 {
  ROOT x = f32[1,4,4,2] parameter(0)
}

comp1 {
  x = f32[1,4,4,2] parameter(0)
  ROOT r = f32[1,4,4,2] call(x), to_apply=comp0
}

comp2 {
  ROOT x = f32[1,4,4,2] parameter(0)
}

stage_1_fwd {
  stage_1_fwd_input1 = f32[1,4,4,2] parameter(0)
  c1_result = f32[1,4,4,2] call(stage_1_fwd_input1), to_apply=comp1
  c2_result = f32[1,4,4,2] call(c1_result), to_apply=comp2
  stage_1_fwd_zero = f32[] constant(0)
  stage_1_fwd_reduce = f32[] reduce(c2_result, stage_1_fwd_zero), dimensions={0,1,2,3}, to_apply=add_float
  ROOT stage_1_fwd_tuple = (f32[]) tuple(stage_1_fwd_reduce)
  after-all = token[] after-all()
  outfeed = token[] outfeed(stage_1_fwd_tuple, after-all), outfeed_config="\010\001\022\005feed3\"\001\001(\001"
}

pipeline {
  pipeline_input0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2]) call(pipeline_input0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_i0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_input1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[]) call(pipeline_input1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_i1 = f32[] get-tuple-element(pipeline_stage_1), index=0
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[]) tuple(pipeline_stage_0_i0, pipeline_stage_1_i1)
}

ENTRY e {
  e.input0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.input1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[]) call(e.input0, e.input1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  HloInstruction* pipeline_input0 = FindInstruction(module, "pipeline_input0");
  EXPECT_THAT(pipeline_input0->sharding().GetUniqueDevice(), 0);

  HloInstruction* pipeline_stage_0 =
      FindInstruction(module, "pipeline_stage_0");
  EXPECT_THAT(pipeline_stage_0->sharding().GetUniqueDevice(), 0);

  HloInstruction* pipeline_stage_0_i0 =
      FindInstruction(module, "pipeline_stage_0_i0");
  EXPECT_THAT(pipeline_stage_0_i0->sharding().GetUniqueDevice(), 0);

  HloInstruction* pipeline_input1 = FindInstruction(module, "pipeline_input1");
  EXPECT_THAT(pipeline_input1->sharding().GetUniqueDevice(), 1);

  HloInstruction* pipeline_stage_1 =
      FindInstruction(module, "pipeline_stage_1");
  EXPECT_THAT(pipeline_stage_1->sharding().GetUniqueDevice(), 1);

  HloInstruction* pipeline_stage_1_i1 =
      FindInstruction(module, "pipeline_stage_1_i1");
  EXPECT_THAT(pipeline_stage_1_i1->sharding().GetUniqueDevice(), 1);

  HloInstruction* pipeline_tuple = FindInstruction(module, "pipeline_tuple");
  EXPECT_THAT(pipeline_tuple->sharding().tuple_elements(),
              ::testing::ElementsAre(pipeline_input0->sharding(),
                                     pipeline_input1->sharding()));

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);

  TF_ASSERT_OK_AND_ASSIGN(
      auto stage_0_comps,
      GetAllComputationsCalledBy(pipeline_stage_0, call_graph.get()));
  for (HloComputation* comp : stage_0_comps) {
    for (auto* inst : comp->instructions()) {
      EXPECT_THAT(inst->sharding().GetUniqueDevice(), 0);
    }
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto stage_1_comps,
      GetAllComputationsCalledBy(pipeline_stage_1, call_graph.get()));
  for (HloComputation* comp : stage_1_comps) {
    for (auto* inst : comp->instructions()) {
      EXPECT_THAT(inst->sharding().GetUniqueDevice(), 1);
    }
  }
}

TEST_F(ShardingPassTest, TestPipeliningShardingWithSubStage) {
  std::string hlo_string = R"(
HloModule top

add_float {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT a = f32[] add(f32[] x, f32[] y)
}

stage_0_0_fwd {
  stage_0_0_fwd_input0 = f32[1,4,4,2] parameter(0)
  ROOT stage_0_0_fwd_tuple = (f32[1,4,4,2]) tuple(stage_0_0_fwd_input0)
}

stage_0_1_fwd {
  stage_0_1_fwd_input0 = f32[1,4,4,2] parameter(0)
  ROOT stage_0_1_fwd_tuple = (f32[1,4,4,2]) tuple(stage_0_1_fwd_input0)
}

stage_0_fwd {
  stage_0_fwd_input0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0_0 = (f32[1,4,4,2]) call(stage_0_fwd_input0), to_apply=stage_0_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_1 = (f32[1,4,4,2]) call(stage_0_fwd_input0), to_apply=stage_0_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_0_0_0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0_0), index=0
  pipeline_stage_0_1_0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0_1), index=0
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_0_0_0, pipeline_stage_0_1_0)
}

comp0 {
  ROOT x = f32[1,4,4,2] parameter(0)
}

comp1 {
  x = f32[1,4,4,2] parameter(0)
  ROOT r = f32[1,4,4,2] call(x), to_apply=comp0
}

comp2 {
  ROOT x = f32[1,4,4,2] parameter(0)
}

stage_1_fwd {
  stage_1_fwd_input1 = f32[1,4,4,2] parameter(0)
  c1_result = f32[1,4,4,2] call(stage_1_fwd_input1), to_apply=comp1
  c2_result = f32[1,4,4,2] call(c1_result), to_apply=comp2
  stage_1_fwd_zero = f32[] constant(0)
  stage_1_fwd_reduce = f32[] reduce(c2_result, stage_1_fwd_zero), dimensions={0,1,2,3}, to_apply=add_float
  ROOT stage_1_fwd_tuple = (f32[]) tuple(stage_1_fwd_reduce)
  after-all = token[] after-all()
  outfeed = token[] outfeed(stage_1_fwd_tuple, after-all), outfeed_config="\010\001\022\005feed3\"\001\001(\001"
}

pipeline {
  pipeline_input0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_input0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=-1}
  pipeline_stage_0_i0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_input1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[]) call(pipeline_input1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_i1 = f32[] get-tuple-element(pipeline_stage_1), index=0
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[]) tuple(pipeline_stage_0_i0, pipeline_stage_1_i1)
}

ENTRY e {
  e.input0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.input1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[]) call(e.input0, e.input1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto stage_0 = module->GetComputationWithName("stage_0_fwd");

  // Parameters should be sharded on -1.
  auto stage_0_fwd_input0 = stage_0->parameter_instruction(0);
  EXPECT_THAT(stage_0_fwd_input0->sharding().GetUniqueDevice(), -1);

  // Root should be sharded on -1.
  auto stage_0_root = stage_0->root_instruction();
  EXPECT_THAT(stage_0_root->sharding().GetUniqueDevice(), -1);

  // Each substage should be sharded appropriate, and not recieve the parent
  // shard -1.
  auto stage_0_0 = module->GetComputationWithName("stage_0_0_fwd");
  for (auto inst : stage_0_0->instructions()) {
    EXPECT_THAT(inst->sharding().GetUniqueDevice(), 0);
  }

  auto stage_0_1 = module->GetComputationWithName("stage_0_1_fwd");
  for (auto inst : stage_0_1->instructions()) {
    EXPECT_THAT(inst->sharding().GetUniqueDevice(), 1);
  }
}

TEST_F(ShardingPassTest, TestCondWithZeroArgs) {
  std::string hlo_string = R"(
HloModule root

cond_true {
  aa = () parameter(0)
  ca = f32[] constant(0.001)
  ROOT ta = (f32[]) tuple(ca)
}

cond_false  {
  ab = () parameter(0)
  cb = f32[] constant(0.01)
  ROOT tb = (f32[]) tuple(cb)
}

ENTRY main {
  a = (pred[]) parameter(0)
  g = pred[] get-tuple-element(a), index=0
  t = () tuple()
  c = (f32[]) conditional(g, t, t), true_computation=cond_true, false_computation=cond_false
  o = f32[] get-tuple-element(c), index=0
  ROOT s = f32[] sine(o), sharding={maximal device=1}
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  EXPECT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto* comp = module->entry_computation();
  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestCallSubcompWithNestedTuple) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  s0 = f16[4] parameter(0)
  s1 = f16[4] parameter(1)
  s2 = f16[4] add(s0, s1), sharding={maximal device=1}
  s3 = f16[4] add(s0, s1), sharding={maximal device=1}
  s4 = (f16[4], f16[4]) tuple(s2, s3)
  ROOT s5 = ((f16[4], f16[4])) tuple(s4)
}

main {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  c0 = f16[4] cosine(a0)
  c1 = f16[4] cosine(a1)
  call1 = ((f16[4], f16[4])) call(c0, c1), to_apply=subcomp
  ROOT gte0 = (f16[4], f16[4]) get-tuple-element(call1), index=0
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  for (auto* comp : module->computations()) {
    auto insts = comp->instructions();
    for (auto* inst : insts) {
      EXPECT_TRUE(inst->has_sharding());
      EXPECT_EQ(inst->sharding().GetUniqueDevice(), 1);
    }
  }
}

TEST_F(ShardingPassTest, TestCallSingleShardingUniqueSharding) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  p0 = f16[4,4] parameter(0)
  p1 = f16[4,4] parameter(1)
  p2 = f16[4,4] parameter(2)
  a1 = f16[4,4] add(p0, p1), sharding={maximal device=1}
  a2 = f16[4,4] add(a1, p2), sharding={maximal device=1}
  ROOT t = (f16[4,4], f16[4,4]) tuple(a1, a2)
}

main {
  p0 = f16[4,4] parameter(0), sharding={maximal device=1}
  p1 = f16[4,4] parameter(1), sharding={maximal device=1}
  p2 = f16[4,4] parameter(2), sharding={maximal device=0}
  c2 = f16[4,4] cosine(p2), sharding={maximal device=0}
  l2 = f16[4,4] log(c2), sharding={maximal device=0}
  ROOT call1 = (f16[4,4], f16[4,4]) call(p0, p1, l2), to_apply=subcomp, backend_config="{\"callConfig\":{\"type\":\"Function\", \"functionConfig\":{\"uniqueSharding\":\"1\"}}}"
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());
  HloComputation* subcomp = FindComputation(module, "subcomp");
  for (auto* inst : subcomp->instructions()) {
    EXPECT_TRUE(inst->has_sharding());
    EXPECT_EQ(inst->sharding().GetUniqueDevice(), 1);
  }
}

TEST_F(ShardingPassTest, FixResourceUpdateSharding) {
  const std::string hlo_string = R"(
HloModule top

resource_update {
  ru_arg0 = f32[] parameter(0)
  ru_arg1 = f32[] parameter(1)
  add0 = f32[] add(ru_arg0, ru_arg1), sharding={maximal device=0}
  add1 = f32[] add(ru_arg0, ru_arg1), sharding={maximal device=1}
  ROOT root = (f32[],f32[]) tuple(add1, add0)
}

loop {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  add2 = f32[] add(param0, param0), sharding={maximal device=1}
  add3 = f32[] add(param1, param1), sharding={maximal device=1}
  call_ru = (f32[],f32[]) call(add2, add3), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  gte1 = f32[] get-tuple-element(call_ru), index=1
  ROOT root = (f32[], f32[]) tuple(gte1, gte0), sharding={{maximal device=0}, {maximal device=0}}
}

ENTRY e {
  e0 = f32[] parameter(0)
  e1 = f32[] parameter(1)
  ROOT e.call = (f32[], f32[]) call(e0, e1), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, ShardingPass().Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* resource_update =
      FindComputation(module.get(), "resource_update");
  HloInstruction* ru_root = resource_update->root_instruction();
  EXPECT_EQ(ru_root->sharding().GetUniqueDevice(), 1);

  HloComputation* comp = FindComputation(module.get(), "loop");
  HloInstruction* gte0 = FindInstruction(module.get(), "gte0");
  EXPECT_EQ(gte0->sharding().GetUniqueDevice(), 1);
  HloInstruction* gte1 = FindInstruction(module.get(), "gte1");
  EXPECT_EQ(gte1->sharding().GetUniqueDevice(), 1);
  HloInstruction* comp_root = comp->root_instruction();
  EXPECT_EQ(comp_root->sharding().GetUniqueDevice(), 1);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
