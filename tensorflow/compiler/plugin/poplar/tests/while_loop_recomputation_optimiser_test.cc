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

#include "tensorflow/compiler/plugin/poplar/driver/passes/while_loop_recomputation_optimiser.h"

#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
namespace {

namespace m = match;

using WhileLoopRecomputationOptimiserTest = HloTestBase;

TEST_F(WhileLoopRecomputationOptimiserTest, TestNothingToDo) {
  const char* const hlo_string = R"(
HloModule NothingToDo

loop_body {
  params = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  X = f32[10, 10] get-tuple-element(params), index=0
  W = f32[10, 10] get-tuple-element(params), index=1
  V = f32[1, 10] get-tuple-element(params), index=2
  n = s32[] get-tuple-element(params), index=3
  
  zero = s32[] constant(0)
  one = s32[] constant(1)
  Xp = f32[10, 10] dynamic-update-slice(X, V, n, zero)
  Wp = f32[10, 10] dynamic-update-slice(W, V, n, one)

  np = s32[] add(n, one)

  ROOT res_body = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(Xp, Wp, V, np)
}

loop_condition {
  params = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  limit = s32[] constant(10)
  n = s32[] get-tuple-element(params), index=3

  ROOT res_cond = pred[] compare(n, limit), direction=LT
}

ENTRY entry {
  zero = f32[] constant(0)
  one = f32[] constant(1)
  A = f32[10, 10] broadcast(zero), dimensions={}
  B = f32[10, 10] broadcast(zero), dimensions={}
  C = f32[1, 10] broadcast(one), dimensions={}

  zero_i = s32[] constant(0)
  loop_init = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(A, B, C, zero_i)
  while = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) while(loop_init), condition=loop_condition, body=loop_body

  D = f32[10, 10] get-tuple-element(while), index=0
  E = f32[10, 10] get-tuple-element(while), index=1 

  a = f32[1, 10] dynamic-slice(D, zero_i, zero_i), dynamic_slice_sizes={1, 10}
  b = f32[1, 10] dynamic-slice(E, zero_i, zero_i), dynamic_slice_sizes={1, 10}

  ROOT res_entry = (f32[1, 10], f32[1, 10]) tuple(a, b)
}
)";

  // Run the pass.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PoplarWhileLoopRecomputationOptimiser().Run(module.get()));
  ASSERT_FALSE(changed);

  // Check the root is sane.
  const auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 2);

  // Verify that 'a' and it's operands haven't been touched.
  const auto* a = root->operand(0);
  EXPECT_EQ(a->name(), "a");
  EXPECT_EQ(a->opcode(), HloOpcode::kDynamicSlice);

  const auto* D = a->operand(0);
  EXPECT_EQ(D->name(), "D");
  EXPECT_EQ(D->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(D->tuple_index(), 0);

  const auto* loop = D->operand(0);
  EXPECT_EQ(loop->name(), "while");
  EXPECT_EQ(loop->opcode(), HloOpcode::kWhile);

  // Verify that 'b' and it's operands haven't been touched.
  const auto* b = root->operand(1);
  EXPECT_EQ(b->name(), "b");
  EXPECT_EQ(b->opcode(), HloOpcode::kDynamicSlice);

  const auto* E = b->operand(0);
  EXPECT_EQ(E->name(), "E");
  EXPECT_EQ(E->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(E->tuple_index(), 1);

  EXPECT_EQ(E->operand(0), loop);
}

TEST_F(WhileLoopRecomputationOptimiserTest, TestUnary) {
  const char* const hlo_string = R"(
HloModule Unary

loop_body {
  params = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  X = f32[10, 10] get-tuple-element(params), index=0
  W = f32[10, 10] get-tuple-element(params), index=1
  V = f32[1, 10] get-tuple-element(params), index=2
  n = s32[] get-tuple-element(params), index=3
  
  zero = s32[] constant(0)
  Xp = f32[10, 10] dynamic-update-slice(X, V, n, zero)

  Z = f32[1, 10] tanh(V)
  Wp = f32[10, 10] dynamic-update-slice(W, Z, n, zero)

  one = s32[] constant(1)
  np = s32[] add(n, one)

  ROOT res_body = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(Xp, Wp, V, np)
}

loop_condition {
  params = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  limit = s32[] constant(10)
  n = s32[] get-tuple-element(params), index=3

  ROOT res_cond = pred[] compare(n, limit), direction=LT
}

ENTRY entry {
  zero = f32[] constant(0)
  one = f32[] constant(1)
  A = f32[10, 10] broadcast(zero), dimensions={}
  B = f32[10, 10] broadcast(zero), dimensions={}
  C = f32[1, 10] broadcast(one), dimensions={}

  zero_i = s32[] constant(0)
  loop_init = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(A, B, C, zero_i)
  while = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) while(loop_init), condition=loop_condition, body=loop_body

  D = f32[10, 10] get-tuple-element(while), index=0
  E = f32[10, 10] get-tuple-element(while), index=1 

  a = f32[1, 10] dynamic-slice(D, zero_i, zero_i), dynamic_slice_sizes={1, 10}
  b = f32[1, 10] dynamic-slice(E, zero_i, zero_i), dynamic_slice_sizes={1, 10}

  ROOT res_entry = (f32[1, 10], f32[1, 10]) tuple(a, b)
}
)";

  // Run the pass.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PoplarWhileLoopRecomputationOptimiser().Run(module.get()));
  ASSERT_TRUE(changed);

  // Check the root is sane.
  const auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 2);

  // Verify that 'a' and it's operands haven't been touched.
  const auto* a = root->operand(0);
  EXPECT_EQ(a->name(), "a");
  EXPECT_EQ(a->opcode(), HloOpcode::kDynamicSlice);

  const auto* D = a->operand(0);
  EXPECT_EQ(D->name(), "D");
  EXPECT_EQ(D->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(D->tuple_index(), 0);

  const auto* loop = D->operand(0);
  EXPECT_EQ(loop->name(), "while");
  EXPECT_EQ(loop->opcode(), HloOpcode::kWhile);

  // Verify that 'b' has been replaced with tanh(a).
  const auto* tanh = root->operand(1);
  EXPECT_EQ(tanh->opcode(), HloOpcode::kTanh);
  EXPECT_EQ(tanh->operand(0), a);
}

TEST_F(WhileLoopRecomputationOptimiserTest, TestUnaryWithReshape) {
  const char* const hlo_string = R"(
HloModule UnaryWithReshape

loop_body {
  params = (f32[10, 10], f32[10, 10], f32[2, 5], s32[]) parameter(0)

  X = f32[10, 10] get-tuple-element(params), index=0
  W = f32[10, 10] get-tuple-element(params), index=1
  V = f32[2, 5] get-tuple-element(params), index=2
  n = s32[] get-tuple-element(params), index=3

  Vp = f32[1, 10] reshape(V)
  
  zero = s32[] constant(0)
  Xp = f32[10, 10] dynamic-update-slice(X, Vp, n, zero)

  Z = f32[1, 10] tanh(Vp)
  Wp = f32[10, 10] dynamic-update-slice(W, Z, n, zero)

  one = s32[] constant(1)
  np = s32[] add(n, one)

  ROOT res_body = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(Xp, Wp, V, np)
}

loop_condition {
  params = (f32[10, 10], f32[10, 10], f32[2, 5], s32[]) parameter(0)

  limit = s32[] constant(10)
  n = s32[] get-tuple-element(params), index=3

  ROOT res_cond = pred[] compare(n, limit), direction=LT
}

ENTRY entry {
  zero = f32[] constant(0)
  one = f32[] constant(1)
  A = f32[10, 10] broadcast(zero), dimensions={}
  B = f32[10, 10] broadcast(zero), dimensions={}
  C = f32[2, 5] broadcast(one), dimensions={}

  zero_i = s32[] constant(0)
  loop_init = (f32[10, 10], f32[10, 10], f32[2, 5], s32[]) tuple(A, B, C, zero_i)
  while = (f32[10, 10], f32[10, 10], f32[2, 5], s32[]) while(loop_init), condition=loop_condition, body=loop_body

  D = f32[10, 10] get-tuple-element(while), index=0
  E = f32[10, 10] get-tuple-element(while), index=1 

  a = f32[1, 10] dynamic-slice(D, zero_i, zero_i), dynamic_slice_sizes={1, 10}
  b = f32[1, 10] dynamic-slice(E, zero_i, zero_i), dynamic_slice_sizes={1, 10}

  ROOT res_entry = (f32[1, 10], f32[1, 10]) tuple(a, b)
}
)";

  // Run the pass.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PoplarWhileLoopRecomputationOptimiser().Run(module.get()));
  ASSERT_TRUE(changed);

  // Check the root is sane.
  const auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 2);

  // Verify that 'a' and it's operands haven't been touched.
  const auto* a = root->operand(0);
  EXPECT_EQ(a->name(), "a");
  EXPECT_EQ(a->opcode(), HloOpcode::kDynamicSlice);

  const auto* D = a->operand(0);
  EXPECT_EQ(D->name(), "D");
  EXPECT_EQ(D->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(D->tuple_index(), 0);

  const auto* loop = D->operand(0);
  EXPECT_EQ(loop->name(), "while");
  EXPECT_EQ(loop->opcode(), HloOpcode::kWhile);

  // Verify that 'b' has been replaced with tanh(reshape(a)).
  const auto* tanh = root->operand(1);
  EXPECT_EQ(tanh->opcode(), HloOpcode::kTanh);
  EXPECT_EQ(tanh->name(), "tanh");

  const auto* reshape = tanh->operand(0);
  EXPECT_EQ(reshape->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(reshape->name(), "reshape");

  EXPECT_EQ(reshape->operand(0), a);
}

TEST_F(WhileLoopRecomputationOptimiserTest, TestUnaryWithTwoReshapes) {
  const char* const hlo_string = R"(
HloModule UnaryWithTwoReshapes

loop_body {
  params = (f32[10, 10], f32[10, 10], f32[2, 5], s32[]) parameter(0)

  X = f32[10, 10] get-tuple-element(params), index=0
  W = f32[10, 10] get-tuple-element(params), index=1
  V = f32[2, 5] get-tuple-element(params), index=2
  n = s32[] get-tuple-element(params), index=3

  Vp = f32[1, 10] reshape(V)

  zero = s32[] constant(0)
  Xp = f32[10, 10] dynamic-update-slice(X, Vp, n, zero)

  Z = f32[2, 5] reshape(Vp)
  Zp = f32[2, 5] tanh(Z)
  Zpp = f32[1, 10] reshape(Zp)
  Wp = f32[10, 10] dynamic-update-slice(W, Zpp, n, zero)

  one = s32[] constant(1)
  np = s32[] add(n, one)

  ROOT res_body = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(Xp, Wp, V, np)
}

loop_condition {
  params = (f32[10, 10], f32[10, 10], f32[2, 5], s32[]) parameter(0)

  limit = s32[] constant(10)
  n = s32[] get-tuple-element(params), index=3

  ROOT res_cond = pred[] compare(n, limit), direction=LT
}

ENTRY entry {
  zero = f32[] constant(0)
  one = f32[] constant(1)
  A = f32[10, 10] broadcast(zero), dimensions={}
  B = f32[10, 10] broadcast(zero), dimensions={}
  C = f32[2, 5] broadcast(one), dimensions={}

  zero_i = s32[] constant(0)
  loop_init = (f32[10, 10], f32[10, 10], f32[2, 5], s32[]) tuple(A, B, C, zero_i)
  while = (f32[10, 10], f32[10, 10], f32[2, 5], s32[]) while(loop_init), condition=loop_condition, body=loop_body

  D = f32[10, 10] get-tuple-element(while), index=0
  E = f32[10, 10] get-tuple-element(while), index=1 

  a = f32[1, 10] dynamic-slice(D, zero_i, zero_i), dynamic_slice_sizes={1, 10}
  b = f32[1, 10] dynamic-slice(E, zero_i, zero_i), dynamic_slice_sizes={1, 10}

  ROOT res_entry = (f32[1, 10], f32[1, 10]) tuple(a, b)
}
)";

  // Run the pass.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PoplarWhileLoopRecomputationOptimiser().Run(module.get()));
  ASSERT_TRUE(changed);

  // Check the root is sane.
  const auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 2);

  // Verify that 'a' and it's operands haven't been touched.
  const auto* a = root->operand(0);
  EXPECT_EQ(a->name(), "a");
  EXPECT_EQ(a->opcode(), HloOpcode::kDynamicSlice);

  const auto* D = a->operand(0);
  EXPECT_EQ(D->name(), "D");
  EXPECT_EQ(D->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(D->tuple_index(), 0);

  const auto* loop = D->operand(0);
  EXPECT_EQ(loop->name(), "while");
  EXPECT_EQ(loop->opcode(), HloOpcode::kWhile);

  // Verify that 'b' has been replaced with reshape(tanh(reshape(a))).
  const auto* reshape_2 = root->operand(1);
  EXPECT_EQ(reshape_2->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(reshape_2->name(), "reshape.2");

  const auto* tanh = reshape_2->operand(0);
  EXPECT_EQ(tanh->opcode(), HloOpcode::kTanh);
  EXPECT_EQ(tanh->name(), "tanh");

  const auto* reshape_1 = tanh->operand(0);
  EXPECT_EQ(reshape_1->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(reshape_1->name(), "reshape.1");

  const auto* reshape = reshape_1->operand(0);
  EXPECT_EQ(reshape->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(reshape->name(), "reshape");

  VLOG(0) << reshape->operand(0)->name();
  EXPECT_EQ(reshape->operand(0), a);
}

TEST_F(WhileLoopRecomputationOptimiserTest, TestUnaryInMultiplePaths) {
  const char* const hlo_string = R"(
HloModule UnaryInMultiplePaths

loop_body {
  params = (f32[10, 10], f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  Y = f32[10, 10] get-tuple-element(params), index=0
  X = f32[10, 10] get-tuple-element(params), index=1
  W = f32[10, 10] get-tuple-element(params), index=2
  V = f32[1, 10] get-tuple-element(params), index=3
  n = s32[] get-tuple-element(params), index=4
  
  zero = s32[] constant(0)
  Yp = f32[10, 10] dynamic-update-slice(Y, V, n, zero)

  Z = f32[1, 10] tanh(V)
  Xp = f32[10, 10] dynamic-update-slice(X, Z, n, zero)

  Zp = f32[1, 10] tanh(Z)
  Wp = f32[10, 10] dynamic-update-slice(W, Zp, n, zero)

  one = s32[] constant(1)
  np = s32[] add(n, one)

  ROOT res_body = (f32[10, 10], f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(Yp, Xp, Wp, V, np)
}

loop_condition {
  params = (f32[10, 10], f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  limit = s32[] constant(10)
  n = s32[] get-tuple-element(params), index=4

  ROOT res_cond = pred[] compare(n, limit), direction=LT
}

ENTRY entry {
  zero = f32[] constant(0)
  one = f32[] constant(1)
  A = f32[10, 10] broadcast(zero), dimensions={}
  B = f32[10, 10] broadcast(zero), dimensions={}
  C = f32[10, 10] broadcast(zero), dimensions={}
  D = f32[1, 10] broadcast(one), dimensions={}

  zero_i = s32[] constant(0)
  loop_init = (f32[10, 10], f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(A, B, C, D, zero_i)
  while = (f32[10, 10], f32[10, 10], f32[10, 10], f32[1, 10], s32[]) while(loop_init), condition=loop_condition, body=loop_body

  E = f32[10, 10] get-tuple-element(while), index=0
  F = f32[10, 10] get-tuple-element(while), index=1
  G = f32[10, 10] get-tuple-element(while), index=2

  a = f32[1, 10] dynamic-slice(E, zero_i, zero_i), dynamic_slice_sizes={1, 10}
  b = f32[1, 10] dynamic-slice(F, zero_i, zero_i), dynamic_slice_sizes={1, 10}
  c = f32[1, 10] dynamic-slice(G, zero_i, zero_i), dynamic_slice_sizes={1, 10}

  ROOT res_entry = (f32[1, 10], f32[1, 10], f32[1, 10]) tuple(a, b, c)
}
)";

  // Run the pass.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PoplarWhileLoopRecomputationOptimiser().Run(module.get()));
  ASSERT_TRUE(changed);

  // Check the root is sane.
  const auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 3);

  // Verify that 'a' and it's operands haven't been touched.
  const auto* a = root->operand(0);
  EXPECT_EQ(a->name(), "a");
  EXPECT_EQ(a->opcode(), HloOpcode::kDynamicSlice);

  const auto* E = a->operand(0);
  EXPECT_EQ(E->name(), "E");
  EXPECT_EQ(E->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(E->tuple_index(), 0);

  const auto* loop = E->operand(0);
  EXPECT_EQ(loop->name(), "while");
  EXPECT_EQ(loop->opcode(), HloOpcode::kWhile);

  // Verify that 'b' has been replaced with tanh(a).
  const auto* tanh = root->operand(1);
  EXPECT_EQ(tanh->name(), "tanh");
  EXPECT_EQ(tanh->opcode(), HloOpcode::kTanh);
  EXPECT_EQ(tanh->operand(0), a);

  // Verify that 'c' has been replaced with 'tanh(tanh(a))'.
  const auto* tanhtanh = root->operand(2);
  EXPECT_EQ(tanhtanh->name(), "tanh.1");
  EXPECT_EQ(tanhtanh->opcode(), HloOpcode::kTanh);
  EXPECT_EQ(tanhtanh->operand(0), tanh);
}

TEST_F(WhileLoopRecomputationOptimiserTest, TestUnarySliceInDifferentLoop) {
  const char* const hlo_string = R"(
HloModule UnarySliceInDifferentLoop

loop_body {
  params = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  X = f32[10, 10] get-tuple-element(params), index=0
  W = f32[10, 10] get-tuple-element(params), index=1
  V = f32[1, 10] get-tuple-element(params), index=2
  n = s32[] get-tuple-element(params), index=3
  
  zero = s32[] constant(0)
  Xp = f32[10, 10] dynamic-update-slice(X, V, n, zero)

  Z = f32[1, 10] tanh(V)
  Wp = f32[10, 10] dynamic-update-slice(W, Z, n, zero)

  one = s32[] constant(1)
  np = s32[] add(n, one)

  ROOT res_body = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(Xp, Wp, V, np)
}

loop_condition {
  params = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  limit = s32[] constant(10)
  n = s32[] get-tuple-element(params), index=3

  ROOT res_cond = pred[] compare(n, limit), direction=LT
}

ds_loop_body {
  params = (f32[10, 10], f32[10, 10], f32[10, 10], f32[10, 10], s32[], s32[]) parameter(0)

  A = f32[10, 10] get-tuple-element(params), index=0
  B = f32[10, 10] get-tuple-element(params), index=1
  C = f32[10, 10] get-tuple-element(params), index=2
  D = f32[10, 10] get-tuple-element(params), index=3
  n = s32[] get-tuple-element(params), index=4

  zero_i = s32[] constant(0)
  old_trip_count = s32[] get-tuple-element(params), index=5
  negative_one = s32[] constant(-1)
  negative_n = s32[] multiply(n, negative_one)
  slice_index = s32[] add(old_trip_count, negative_n)
  slice_index_p = s32[] add(slice_index, negative_one)
  a = f32[1, 10] dynamic-slice(A, slice_index_p, zero_i), dynamic_slice_sizes={1, 10}
  b = f32[1, 10] dynamic-slice(B, slice_index_p, zero_i), dynamic_slice_sizes={1, 10}

  Cp = f32[10, 10] dynamic-update-slice(C, a, n, zero_i)
  Dp = f32[10, 10] dynamic-update-slice(D, b, n, zero_i)

  one = s32[] constant(1)
  np = s32[] add(n, one)

  ROOT res_ds_body = (f32[10, 10], f32[10, 10], f32[10, 10], f32[10, 10], s32[], s32[]) tuple(A, B, Cp, Dp, np, old_trip_count)
}

ds_loop_condition {
  params = (f32[10, 10], f32[10, 10], f32[10, 10], f32[10, 10], s32[], s32[]) parameter(0)

  limit = s32[] constant(10)
  n = s32[] get-tuple-element(params), index=4

  ROOT res_cond = pred[] compare(n, limit), direction=LT
}

ENTRY entry {
  zero = f32[] constant(0)
  one = f32[] constant(1)
  A = f32[10, 10] broadcast(zero), dimensions={}
  B = f32[10, 10] broadcast(zero), dimensions={}
  C = f32[1, 10] broadcast(one), dimensions={}

  zero_i = s32[] constant(0)
  loop_init = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(A, B, C, zero_i)
  while = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) while(loop_init), condition=loop_condition, body=loop_body

  D = f32[10, 10] get-tuple-element(while), index=0
  E = f32[10, 10] get-tuple-element(while), index=1

  F = f32[10, 10] broadcast(zero), dimensions={}
  G = f32[10, 10] broadcast(zero), dimensions={}

  old_trip_count = s32[] get-tuple-element(while), index=3

  ds_loop_init = (f32[10, 10], f32[10, 10], f32[10, 10], f32[10, 10], s32[], s32[]) tuple(D, E, F, G, zero_i, old_trip_count)
  ds_while = (f32[10, 10], f32[10, 10], f32[10, 10], f32[10, 10], s32[], s32[]) while(ds_loop_init), condition=ds_loop_condition, body=ds_loop_body

  H = get-tuple-element(ds_while), index=2
  I = get-tuple-element(ds_while), index=3

  ROOT res_entry = (f32[10, 10], f32[10, 10]) tuple(H, I)
}
)";

  // Run the pass.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PoplarWhileLoopRecomputationOptimiser().Run(module.get()));
  ASSERT_TRUE(changed);

  // Check the root is sane.
  const auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 2);

  // Verify that 'H' and it's operand chain haven't been touched.
  const auto* H = root->operand(0);
  EXPECT_EQ(H->name(), "H");
  EXPECT_EQ(H->opcode(), HloOpcode::kGetTupleElement);

  const auto* ds_while = H->operand(0);
  EXPECT_EQ(ds_while->name(), "ds_while");
  EXPECT_EQ(ds_while->opcode(), HloOpcode::kWhile);

  const auto* ds_while_root = ds_while->while_body()->root_instruction();
  EXPECT_EQ(ds_while_root->name(), "res_ds_body");
  EXPECT_EQ(ds_while_root->operand_count(), 6);

  const auto* Cp = ds_while_root->operand(2);
  EXPECT_EQ(Cp->name(), "Cp");
  EXPECT_EQ(Cp->opcode(), HloOpcode::kDynamicUpdateSlice);

  const auto* a = Cp->operand(1);
  EXPECT_EQ(a->name(), "a");
  EXPECT_EQ(a->opcode(), HloOpcode::kDynamicSlice);

  // Verify that the dynamic-slice 'b' in 'I's operand chain has been replaced
  // with tanh(a), where the tanh is a clone of Z in the first loop, on a.
  const auto* Dp = ds_while_root->operand(3);
  EXPECT_EQ(Dp->name(), "Dp");
  EXPECT_EQ(Dp->opcode(), HloOpcode::kDynamicUpdateSlice);

  const auto* tanh = Dp->operand(1);
  EXPECT_EQ(tanh->name(), "tanh");
  EXPECT_EQ(tanh->opcode(), HloOpcode::kTanh);
  EXPECT_EQ(tanh->operand(0), a);
}

TEST_F(WhileLoopRecomputationOptimiserTest, TestUnaryDifferingTLUses) {
  const char* const hlo_string = R"(
HloModule UnaryDifferingTLUses

loop_body {
  params = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  X = f32[10, 10] get-tuple-element(params), index=0
  W = f32[10, 10] get-tuple-element(params), index=1
  V = f32[1, 10] get-tuple-element(params), index=2
  n = s32[] get-tuple-element(params), index=3
  
  zero = s32[] constant(0)
  Xp = f32[10, 10] dynamic-update-slice(X, V, n, zero)

  Z = f32[1, 10] tanh(V)
  Wp = f32[10, 10] dynamic-update-slice(W, Z, n, zero)

  one = s32[] constant(1)
  np = s32[] add(n, one)

  ROOT res_body = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(Xp, Wp, V, np)
}

loop_condition {
  params = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  limit = s32[] constant(10)
  n = s32[] get-tuple-element(params), index=3

  ROOT res_cond = pred[] compare(n, limit), direction=LT
}

ds_loop_body {
  params = (f32[10, 10], f32[10, 10], f32[10, 10], f32[10, 10], s32[], s32[]) parameter(0)

  A = f32[10, 10] get-tuple-element(params), index=0
  B = f32[10, 10] get-tuple-element(params), index=1
  C = f32[10, 10] get-tuple-element(params), index=2
  D = f32[10, 10] get-tuple-element(params), index=3
  n = s32[] get-tuple-element(params), index=4

  zero_i = s32[] constant(0)
  old_trip_count = s32[] get-tuple-element(params), index=5
  negative_one = s32[] constant(-1)
  negative_n = s32[] multiply(n, negative_one)
  slice_index = s32[] add(old_trip_count, negative_n)
  slice_index_p = s32[] add(slice_index, negative_one)
  a = f32[1, 10] dynamic-slice(A, slice_index_p, zero_i), dynamic_slice_sizes={1, 10}

  Cp = f32[10, 10] dynamic-update-slice(C, a, n, zero_i)

  one = s32[] constant(1)
  np = s32[] add(n, one)

  ROOT res_ds_body = (f32[10, 10], f32[10, 10], f32[10, 10], f32[10, 10], s32[], s32[]) tuple(A, B, Cp, D, np, old_trip_count)
}

ds_loop_condition {
  params = (f32[10, 10], f32[10, 10], f32[10, 10], f32[10, 10], s32[], s32[]) parameter(0)

  limit = s32[] constant(10)
  n = s32[] get-tuple-element(params), index=4

  ROOT res_cond = pred[] compare(n, limit), direction=LT
}

ENTRY entry {
  zero = f32[] constant(0)
  one = f32[] constant(1)
  A = f32[10, 10] broadcast(zero), dimensions={}
  B = f32[10, 10] broadcast(zero), dimensions={}
  C = f32[1, 10] broadcast(one), dimensions={}

  zero_i = s32[] constant(0)
  loop_init = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(A, B, C, zero_i)
  while = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) while(loop_init), condition=loop_condition, body=loop_body

  D = f32[10, 10] get-tuple-element(while), index=0
  E = f32[10, 10] get-tuple-element(while), index=1

  Ep = f32[1, 10] dynamic-slice(E, zero_i, zero_i), dynamic_slice_sizes={1, 10}

  F = f32[10, 10] broadcast(zero), dimensions={}
  G = f32[10, 10] broadcast(zero), dimensions={}

  old_trip_count = s32[] get-tuple-element(while), index=3

  ds_loop_init = (f32[10, 10], f32[10, 10], f32[10, 10], f32[10, 10], s32[], s32[]) tuple(D, E, F, G, zero_i, old_trip_count)
  ds_while = (f32[10, 10], f32[10, 10], f32[10, 10], f32[10, 10], s32[], s32[]) while(ds_loop_init), condition=ds_loop_condition, body=ds_loop_body

  H = f32[10, 10] get-tuple-element(ds_while), index=2

  ROOT res_entry = (f32[10, 10], f32[1, 10]) tuple(H, Ep)
}
)";

  // Run the pass.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PoplarWhileLoopRecomputationOptimiser().Run(module.get()));
  ASSERT_FALSE(changed);

  // Check the root is sane.
  const auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 2);

  // Verify that 'H' and it's operand chain haven't been touched.
  const auto* H = root->operand(0);
  EXPECT_EQ(H->name(), "H");
  EXPECT_EQ(H->opcode(), HloOpcode::kGetTupleElement);

  const auto* ds_while = H->operand(0);
  EXPECT_EQ(ds_while->name(), "ds_while");
  EXPECT_EQ(ds_while->opcode(), HloOpcode::kWhile);

  const auto* ds_while_root = ds_while->while_body()->root_instruction();
  EXPECT_EQ(ds_while_root->name(), "res_ds_body");
  EXPECT_EQ(ds_while_root->operand_count(), 6);

  const auto* Cp = ds_while_root->operand(2);
  EXPECT_EQ(Cp->name(), "Cp");
  EXPECT_EQ(Cp->opcode(), HloOpcode::kDynamicUpdateSlice);

  const auto* a = Cp->operand(1);
  EXPECT_EQ(a->name(), "a");
  EXPECT_EQ(a->opcode(), HloOpcode::kDynamicSlice);

  // Verify that 'Ep' and it's operand chain haven't been touched.
  const auto* Ep = root->operand(1);
  EXPECT_EQ(Ep->name(), "Ep");
  EXPECT_EQ(Ep->opcode(), HloOpcode::kDynamicSlice);

  const auto* E = Ep->operand(0);
  EXPECT_EQ(E->name(), "E");
  EXPECT_EQ(E->opcode(), HloOpcode::kGetTupleElement);

  const auto* while_loop = E->operand(0);
  EXPECT_EQ(while_loop->name(), "while");
  EXPECT_EQ(while_loop->opcode(), HloOpcode::kWhile);
}

TEST_F(WhileLoopRecomputationOptimiserTest, TestUnaryDifferingIndices) {
  const char* const hlo_string = R"(
HloModule UnaryDifferingIndices

loop_body {
  params = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  X = f32[10, 10] get-tuple-element(params), index=0
  W = f32[10, 10] get-tuple-element(params), index=1
  V = f32[1, 10] get-tuple-element(params), index=2
  n = s32[] get-tuple-element(params), index=3

  one = s32[] constant(1)
  m = add(n, one)
  
  zero = s32[] constant(0)
  Xp = f32[10, 10] dynamic-update-slice(X, V, n, zero)

  Z = f32[1, 10] tanh(V)
  Wp = f32[10, 10] dynamic-update-slice(W, Z, m, zero)

  np = s32[] add(n, one)

  ROOT res_body = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(Xp, Wp, V, np)
}

loop_condition {
  params = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  limit = s32[] constant(10)
  n = s32[] get-tuple-element(params), index=3

  ROOT res_cond = pred[] compare(n, limit), direction=LT
}

ENTRY entry {
  zero = f32[] constant(0)
  one = f32[] constant(1)
  A = f32[10, 10] broadcast(zero), dimensions={}
  B = f32[10, 10] broadcast(zero), dimensions={}
  C = f32[1, 10] broadcast(one), dimensions={}

  zero_i = s32[] constant(0)
  loop_init = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(A, B, C, zero_i)
  while = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) while(loop_init), condition=loop_condition, body=loop_body

  D = f32[10, 10] get-tuple-element(while), index=0
  E = f32[10, 10] get-tuple-element(while), index=1 

  a = f32[1, 10] dynamic-slice(D, zero_i, zero_i), dynamic_slice_sizes={1, 10}
  b = f32[1, 10] dynamic-slice(E, zero_i, zero_i), dynamic_slice_sizes={1, 10}

  ROOT res_entry = (f32[1, 10], f32[1, 10]) tuple(a, b)
}
)";

  // Run the pass.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PoplarWhileLoopRecomputationOptimiser().Run(module.get()));
  ASSERT_FALSE(changed);

  // Check the root is sane.
  const auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 2);

  // Verify that 'a' and it's operands haven't been touched.
  const auto* a = root->operand(0);
  EXPECT_EQ(a->name(), "a");
  EXPECT_EQ(a->opcode(), HloOpcode::kDynamicSlice);

  const auto* D = a->operand(0);
  EXPECT_EQ(D->name(), "D");
  EXPECT_EQ(D->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(D->tuple_index(), 0);

  const auto* loop = D->operand(0);
  EXPECT_EQ(loop->name(), "while");
  EXPECT_EQ(loop->opcode(), HloOpcode::kWhile);

  // Verify that 'b' and it's operands haven't been touched.
  const auto* b = root->operand(1);
  EXPECT_EQ(b->name(), "b");
  EXPECT_EQ(b->opcode(), HloOpcode::kDynamicSlice);

  const auto* E = b->operand(0);
  EXPECT_EQ(E->name(), "E");
  EXPECT_EQ(E->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(E->tuple_index(), 1);

  EXPECT_EQ(E->operand(0), loop);
}

TEST_F(WhileLoopRecomputationOptimiserTest, TestUnaryNonWhitelistedOp) {
  const char* const hlo_string = R"(
HloModule UnaryNonWhitelistedOp

loop_body {
  params = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  X = f32[10, 10] get-tuple-element(params), index=0
  W = f32[10, 10] get-tuple-element(params), index=1
  V = f32[1, 10] get-tuple-element(params), index=2
  n = s32[] get-tuple-element(params), index=3

  C = f32[10, 10] constant(1)
  L = f32[1, 10] dot(V, C), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  
  zero = s32[] constant(0)
  Xp = f32[10, 10] dynamic-update-slice(X, L, n, zero)

  Z = f32[1, 10] tanh(L)
  Wp = f32[10, 10] dynamic-update-slice(W, Z, n, zero)

  one = s32[] constant(1)
  np = s32[] add(n, one)

  ROOT res_body = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(Xp, Wp, V, np)
}

loop_condition {
  params = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  limit = s32[] constant(10)
  n = s32[] get-tuple-element(params), index=3

  ROOT res_cond = pred[] compare(n, limit), direction=LT
}

ENTRY entry {
  zero = f32[] constant(0)
  one = f32[] constant(1)
  A = f32[10, 10] broadcast(zero), dimensions={}
  B = f32[10, 10] broadcast(zero), dimensions={}
  C = f32[1, 10] broadcast(one), dimensions={}

  zero_i = s32[] constant(0)
  loop_init = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(A, B, C, zero_i)
  while = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) while(loop_init), condition=loop_condition, body=loop_body

  D = f32[10, 10] get-tuple-element(while), index=0
  E = f32[10, 10] get-tuple-element(while), index=1 

  a = f32[1, 10] dynamic-slice(D, zero_i, zero_i), dynamic_slice_sizes={1, 10}
  b = f32[1, 10] dynamic-slice(E, zero_i, zero_i), dynamic_slice_sizes={1, 10}

  ROOT res_entry = (f32[1, 10], f32[1, 10]) tuple(a, b)
}
)";

  // Run the pass.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PoplarWhileLoopRecomputationOptimiser().Run(module.get()));
  ASSERT_FALSE(changed);

  // Check the root is sane.
  const auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 2);

  // Verify that 'a' and it's operands haven't been touched.
  const auto* a = root->operand(0);
  EXPECT_EQ(a->name(), "a");
  EXPECT_EQ(a->opcode(), HloOpcode::kDynamicSlice);

  const auto* D = a->operand(0);
  EXPECT_EQ(D->name(), "D");
  EXPECT_EQ(D->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(D->tuple_index(), 0);

  const auto* loop = D->operand(0);
  EXPECT_EQ(loop->name(), "while");
  EXPECT_EQ(loop->opcode(), HloOpcode::kWhile);

  // Verify that 'b' and it's operands haven't been touched.
  const auto* b = root->operand(1);
  EXPECT_EQ(b->name(), "b");
  EXPECT_EQ(b->opcode(), HloOpcode::kDynamicSlice);

  const auto* E = b->operand(0);
  EXPECT_EQ(E->name(), "E");
  EXPECT_EQ(E->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(E->tuple_index(), 1);

  EXPECT_EQ(E->operand(0), loop);
}

TEST_F(WhileLoopRecomputationOptimiserTest, TestNonUnary) {
  const char* const hlo_string = R"(
HloModule NonUnary

loop_body {
  params = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  X = f32[10, 10] get-tuple-element(params), index=0
  W = f32[10, 10] get-tuple-element(params), index=1
  V = f32[1, 10] get-tuple-element(params), index=2
  n = s32[] get-tuple-element(params), index=3
  
  zero = s32[] constant(0)
  Xp = f32[10, 10] dynamic-update-slice(X, V, n, zero)

  Z = f32[1, 10] tanh(V)
  C = f32[1, 10] constant(1)
  L = f32[1, 10] add(Z, C)
  Wp = f32[10, 10] dynamic-update-slice(W, L, n, zero)

  one = s32[] constant(1)
  np = s32[] add(n, one)

  ROOT res_body = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(Xp, Wp, V, np)
}

loop_condition {
  params = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  limit = s32[] constant(10)
  n = s32[] get-tuple-element(params), index=3

  ROOT res_cond = pred[] compare(n, limit), direction=LT
}

ENTRY entry {
  zero = f32[] constant(0)
  one = f32[] constant(1)
  A = f32[10, 10] broadcast(zero), dimensions={}
  B = f32[10, 10] broadcast(zero), dimensions={}
  C = f32[1, 10] broadcast(one), dimensions={}

  zero_i = s32[] constant(0)
  loop_init = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(A, B, C, zero_i)
  while = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) while(loop_init), condition=loop_condition, body=loop_body

  D = f32[10, 10] get-tuple-element(while), index=0
  E = f32[10, 10] get-tuple-element(while), index=1 

  a = f32[1, 10] dynamic-slice(D, zero_i, zero_i), dynamic_slice_sizes={1, 10}
  b = f32[1, 10] dynamic-slice(E, zero_i, zero_i), dynamic_slice_sizes={1, 10}

  ROOT res_entry = (f32[1, 10], f32[1, 10]) tuple(a, b)
}
)";

  // Run the pass.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PoplarWhileLoopRecomputationOptimiser().Run(module.get()));
  ASSERT_FALSE(changed);

  // Check the root is sane.
  const auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 2);

  // Verify that 'a' and it's operands haven't been touched.
  const auto* a = root->operand(0);
  EXPECT_EQ(a->name(), "a");
  EXPECT_EQ(a->opcode(), HloOpcode::kDynamicSlice);

  const auto* D = a->operand(0);
  EXPECT_EQ(D->name(), "D");
  EXPECT_EQ(D->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(D->tuple_index(), 0);

  const auto* loop = D->operand(0);
  EXPECT_EQ(loop->name(), "while");
  EXPECT_EQ(loop->opcode(), HloOpcode::kWhile);

  // Verify that 'b' and it's operands haven't been touched.
  const auto* b = root->operand(1);
  EXPECT_EQ(b->name(), "b");
  EXPECT_EQ(b->opcode(), HloOpcode::kDynamicSlice);

  const auto* E = b->operand(0);
  EXPECT_EQ(E->name(), "E");
  EXPECT_EQ(E->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(E->tuple_index(), 1);

  EXPECT_EQ(E->operand(0), loop);
}

TEST_F(WhileLoopRecomputationOptimiserTest, TestCompositeUnary) {
  const char* const hlo_string = R"(
HloModule CompositeUnary

loop_body {
  params = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  X = f32[10, 10] get-tuple-element(params), index=0
  W = f32[10, 10] get-tuple-element(params), index=1
  V = f32[1, 10] get-tuple-element(params), index=2
  n = s32[] get-tuple-element(params), index=3
  
  zero = s32[] constant(0)
  Xp = f32[10, 10] dynamic-update-slice(X, V, n, zero)

  Z = f32[1, 10] tanh(V)
  L = f32[1, 10] sine(Z)
  Wp = f32[10, 10] dynamic-update-slice(W, L, n, zero)

  one = s32[] constant(1)
  np = s32[] add(n, one)

  ROOT res_body = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(Xp, Wp, V, np)
}

loop_condition {
  params = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  limit = s32[] constant(10)
  n = s32[] get-tuple-element(params), index=3

  ROOT res_cond = pred[] compare(n, limit), direction=LT
}

ENTRY entry {
  zero = f32[] constant(0)
  one = f32[] constant(1)
  A = f32[10, 10] broadcast(zero), dimensions={}
  B = f32[10, 10] broadcast(zero), dimensions={}
  C = f32[1, 10] broadcast(one), dimensions={}

  zero_i = s32[] constant(0)
  loop_init = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(A, B, C, zero_i)
  while = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) while(loop_init), condition=loop_condition, body=loop_body

  D = f32[10, 10] get-tuple-element(while), index=0
  E = f32[10, 10] get-tuple-element(while), index=1 

  a = f32[1, 10] dynamic-slice(D, zero_i, zero_i), dynamic_slice_sizes={1, 10}
  b = f32[1, 10] dynamic-slice(E, zero_i, zero_i), dynamic_slice_sizes={1, 10}

  ROOT res_entry = (f32[1, 10], f32[1, 10]) tuple(a, b)
}
)";

  // Run the pass.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PoplarWhileLoopRecomputationOptimiser().Run(module.get()));
  ASSERT_TRUE(changed);

  // Check the root is sane.
  const auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 2);

  // Verify that 'a' and it's operands haven't been touched.
  const auto* a = root->operand(0);
  EXPECT_EQ(a->name(), "a");
  EXPECT_EQ(a->opcode(), HloOpcode::kDynamicSlice);

  const auto* D = a->operand(0);
  EXPECT_EQ(D->name(), "D");
  EXPECT_EQ(D->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(D->tuple_index(), 0);

  const auto* loop = D->operand(0);
  EXPECT_EQ(loop->name(), "while");
  EXPECT_EQ(loop->opcode(), HloOpcode::kWhile);

  // Verify that 'b' has been replaced with sine(tanh(a)).
  const auto* sine = root->operand(1);
  EXPECT_EQ(sine->opcode(), HloOpcode::kSin);

  const auto* tanh = sine->operand(0);
  EXPECT_EQ(tanh->opcode(), HloOpcode::kTanh);
  EXPECT_EQ(tanh->operand(0), a);
}

TEST_F(WhileLoopRecomputationOptimiserTest, TestManyCompositeUnary) {
  const char* const hlo_string = R"(
HloModule CompositeUnary

loop_body {
  params = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  X = f32[10, 10] get-tuple-element(params), index=0
  W = f32[10, 10] get-tuple-element(params), index=1
  V = f32[1, 10] get-tuple-element(params), index=2
  n = s32[] get-tuple-element(params), index=3
  
  zero = s32[] constant(0)
  Xp = f32[10, 10] dynamic-update-slice(X, V, n, zero)

  Za = f32[1, 10] tanh(V)
  Zb = f32[1, 10] tanh(Za)
  Zc = f32[1, 10] tanh(Zb)
  Zd = f32[1, 10] tanh(Zc)
  Ze = f32[1, 10] tanh(Zd)
  Zf = f32[1, 10] tanh(Ze)
  Zg = f32[1, 10] tanh(Zf)
  Zh = f32[1, 10] tanh(Zg)
  Zi = f32[1, 10] tanh(Zh)
  Zj = f32[1, 10] tanh(Zi)
  Wp = f32[10, 10] dynamic-update-slice(W, Zj, n, zero)

  one = s32[] constant(1)
  np = s32[] add(n, one)

  ROOT res_body = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(Xp, Wp, V, np)
}

loop_condition {
  params = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  limit = s32[] constant(10)
  n = s32[] get-tuple-element(params), index=3

  ROOT res_cond = pred[] compare(n, limit), direction=LT
}

ENTRY entry {
  zero = f32[] constant(0)
  one = f32[] constant(1)
  A = f32[10, 10] broadcast(zero), dimensions={}
  B = f32[10, 10] broadcast(zero), dimensions={}
  C = f32[1, 10] broadcast(one), dimensions={}

  zero_i = s32[] constant(0)
  loop_init = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(A, B, C, zero_i)
  while = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) while(loop_init), condition=loop_condition, body=loop_body

  D = f32[10, 10] get-tuple-element(while), index=0
  E = f32[10, 10] get-tuple-element(while), index=1 

  a = f32[1, 10] dynamic-slice(D, zero_i, zero_i), dynamic_slice_sizes={1, 10}
  b = f32[1, 10] dynamic-slice(E, zero_i, zero_i), dynamic_slice_sizes={1, 10}

  ROOT res_entry = (f32[1, 10], f32[1, 10]) tuple(a, b)
}
)";

  // Run the pass.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PoplarWhileLoopRecomputationOptimiser().Run(module.get()));
  ASSERT_FALSE(changed);

  // Check the root is sane.
  const auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 2);

  // Verify that 'a' and it's operands haven't been touched.
  const auto* a = root->operand(0);
  EXPECT_EQ(a->name(), "a");
  EXPECT_EQ(a->opcode(), HloOpcode::kDynamicSlice);

  const auto* D = a->operand(0);
  EXPECT_EQ(D->name(), "D");
  EXPECT_EQ(D->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(D->tuple_index(), 0);

  const auto* loop = D->operand(0);
  EXPECT_EQ(loop->name(), "while");
  EXPECT_EQ(loop->opcode(), HloOpcode::kWhile);

  // Verify that 'b' and it's operands haven't been touched.
  const auto* b = root->operand(1);
  EXPECT_EQ(b->name(), "b");
  EXPECT_EQ(b->opcode(), HloOpcode::kDynamicSlice);

  const auto* E = b->operand(0);
  EXPECT_EQ(E->name(), "E");
  EXPECT_EQ(E->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(E->tuple_index(), 1);

  EXPECT_EQ(E->operand(0), loop);
}

TEST_F(WhileLoopRecomputationOptimiserTest, TestDoNothingLoopInvariantInput) {
  const char* const hlo_string = R"(
HloModule LoopInvariantInput

loop_body {
  params = (f32[10, 10], f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  A = f32[10, 10] get-tuple-element(params), index=0
  X = f32[10, 10] get-tuple-element(params), index=1
  W = f32[10, 10] get-tuple-element(params), index=2
  V = f32[1, 10] get-tuple-element(params), index=3
  n = s32[] get-tuple-element(params), index=4
  
  zero = s32[] constant(0)
  Xp = f32[10, 10] dynamic-update-slice(A, V, n, zero)

  Z = f32[1, 10] tanh(V)
  Wp = f32[10, 10] dynamic-update-slice(A, Z, n, zero)

  one = s32[] constant(1)
  np = s32[] add(n, one)

  ROOT res_body = (f32[10, 10], f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(A, Xp, Wp, V, np)
}

loop_condition {
  params = (f32[10, 10], f32[10, 10], f32[10, 10], f32[1, 10], s32[]) parameter(0)

  limit = s32[] constant(10)
  n = s32[] get-tuple-element(params), index=4

  ROOT res_cond = pred[] compare(n, limit), direction=LT
}

ENTRY entry {
  zero = f32[] constant(0)
  one = f32[] constant(1)
  A = f32[10, 10] broadcast(zero), dimensions={}
  B = f32[10, 10] broadcast(zero), dimensions={}
  C = f32[10, 10] broadcast(zero), dimensions={}
  D = f32[1, 10] broadcast(one), dimensions={}

  zero_i = s32[] constant(0)
  loop_init = (f32[10, 10], f32[10, 10], f32[1, 10], s32[]) tuple(A, B, C, D, zero_i)
  while = (f32[10, 10], f32[10, 10], f32[10, 10], f32[1, 10], s32[]) while(loop_init), condition=loop_condition, body=loop_body

  E = f32[10, 10] get-tuple-element(while), index=0
  F = f32[10, 10] get-tuple-element(while), index=1 

  a = f32[1, 10] dynamic-slice(E, zero_i, zero_i), dynamic_slice_sizes={1, 10}
  b = f32[1, 10] dynamic-slice(F, zero_i, zero_i), dynamic_slice_sizes={1, 10}

  ROOT res_entry = (f32[1, 10], f32[1, 10]) tuple(a, b)
}
)";

  // Run the pass.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PoplarWhileLoopRecomputationOptimiser().Run(module.get()));
  ASSERT_FALSE(changed);

  // Check the root is sane.
  const auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 2);

  // Verify that 'a' and it's operands haven't been touched.
  const auto* a = root->operand(0);
  EXPECT_EQ(a->name(), "a");
  EXPECT_EQ(a->opcode(), HloOpcode::kDynamicSlice);

  const auto* E = a->operand(0);
  EXPECT_EQ(E->name(), "E");
  EXPECT_EQ(E->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(E->tuple_index(), 0);

  const auto* loop = E->operand(0);
  EXPECT_EQ(loop->name(), "while");
  EXPECT_EQ(loop->opcode(), HloOpcode::kWhile);

  // Verify that 'b' and it's operands haven't been touched.
  const auto* b = root->operand(1);
  EXPECT_EQ(b->name(), "b");
  EXPECT_EQ(b->opcode(), HloOpcode::kDynamicSlice);

  const auto* F = b->operand(0);
  EXPECT_EQ(F->name(), "F");
  EXPECT_EQ(F->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(F->tuple_index(), 1);

  EXPECT_EQ(F->operand(0), loop);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
