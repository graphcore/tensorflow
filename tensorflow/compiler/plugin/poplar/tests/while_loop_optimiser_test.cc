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

#include "tensorflow/compiler/plugin/poplar/driver/passes/while_loop_optimiser.h"

#include <stdlib.h>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_analysis.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/while_loop_to_repeat_simplify.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using WhileLoopOptimiserTest = HloTestBase;

TEST_F(WhileLoopOptimiserTest, DetectSingleBroadcast) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[10, 1]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  one = s32[] constant(1)
  zero = s32[] constant(0)
  add = s32[] add(p_body.0, one)
  two = s32[] constant(2)
  slice-input = s32[1, 1] reshape(two)
  p_body.1 = s32[10, 1] get-tuple-element(p_body), index=1
  dyn_update = s32[10, 1] dynamic-update-slice(p_body.1, slice-input, p_body.0, zero)
  ROOT root = (s32[],s32[]) tuple(add, dyn_update)
}

condition {
  p_cond = (s32[],s32[10, 1]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[10, 1] broadcast(const_0), dimensions={}
  const_2 = s32[] constant(1)
  repeat_init = (s32[],s32[10, 1]) tuple(const_0, const_1)
  while = (s32[],s32[10, 1]) while(repeat_init), condition=condition, body=body
  broadcast = s32[10, 1] get-tuple-element(while), index=1
  ROOT slice = s32[1, 1] dynamic-slice(broadcast, const_0, const_0), dynamic_slice_sizes={1, 1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          PoplarWhileLoopOptimiser().Run(module.get()));
  ASSERT_TRUE(changed);
}

TEST_F(WhileLoopOptimiserTest, DetectTwoIdentical) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[10, 1], s32[10, 1]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  one = s32[] constant(1)
  zero = s32[] constant(0)
  add = s32[] add(p_body.0, one)
  two = s32[] constant(2)
  slice-input = s32[1, 1] reshape(two)
  p_body.1 = s32[10, 1] get-tuple-element(p_body), index=1
  p_body.2 = s32[10, 1] get-tuple-element(p_body), index=2
  dyn_update = s32[10, 1] dynamic-update-slice(p_body.1, slice-input, p_body.0, zero)
  dyn_update.2 = s32[10, 1] dynamic-update-slice(p_body.2, slice-input, p_body.0, zero)
  ROOT root = (s32[],s32[10, 1], s32[10,1]) tuple(add, dyn_update, dyn_update.2)
}

condition {
  p_cond = (s32[],s32[10, 1], s32[10, 1]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[10, 1] broadcast(const_0), dimensions={}
  const_2 = s32[] constant(1)
  repeat_init = (s32[],s32[10, 1]) tuple(const_0, const_1, const_1)
  while = (s32[],s32[10, 1], s32[10, 1]) while(repeat_init), condition=condition, body=body
  broadcast = s32[10, 1] get-tuple-element(while), index=1
  broadcast.2 = s32[10, 1] get-tuple-element(while), index=2
  slice.2 = s32[1, 1] dynamic-slice(broadcast.2, const_0, const_0), dynamic_slice_sizes={1, 1}
  ROOT slice = s32[1, 1] dynamic-slice(broadcast, const_0, const_0), dynamic_slice_sizes={1, 1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          PoplarWhileLoopOptimiser().Run(module.get()));
  ASSERT_TRUE(changed);
}

TEST_F(WhileLoopOptimiserTest, UsedByRoot) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[10, 1]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  one = s32[] constant(1)
  zero = s32[] constant(0)
  add = s32[] add(p_body.0, one)
  two = s32[] constant(2)
  slice-input = s32[1, 1] reshape(two)
  p_body.1 = s32[10, 1] get-tuple-element(p_body), index=1
  dyn_update = s32[10, 1] dynamic-update-slice(p_body.1, slice-input, p_body.0, zero)
  ROOT root = (s32[],s32[]) tuple(add, dyn_update)
}

condition {
  p_cond = (s32[],s32[10, 1]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[10, 1] broadcast(const_0), dimensions={}
  const_2 = s32[] constant(1)
  repeat_init = (s32[],s32[10, 1]) tuple(const_0, const_1)
  while = (s32[],s32[10, 1]) while(repeat_init), condition=condition, body=body
  ROOT broadcast = s32[10, 1] get-tuple-element(while), index=1
  slice = s32[1, 1] dynamic-slice(broadcast, const_0, const_0), dynamic_slice_sizes={1, 1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto count = PoplarWhileLoopOptimiser().Run(module.get()).ValueOrDie();
  ASSERT_FALSE(count);
}

TEST_F(WhileLoopOptimiserTest, TestReshaping) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[10, 1]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  one = s32[] constant(1)
  zero = s32[] constant(0)
  add = s32[] add(p_body.0, one)
  two = s32[] constant(2)
  slice-input = s32[1, 1] reshape(two)
  p_body.1 = s32[10, 1] get-tuple-element(p_body), index=1
  dyn_update = s32[10, 1] dynamic-update-slice(p_body.1, slice-input, p_body.0, zero)
  ROOT root = (s32[],s32[]) tuple(add, dyn_update)
}

condition {
  p_cond = (s32[],s32[10, 1]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

func {
  p_func = s32[10, 1] parameter(0)
  unused = s32[10, 1] parameter(1)
  ROOT copym = s32[10, 1] copy(p_func)
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[10, 1] broadcast(const_0), dimensions={}
  new_shape = s32[11, 1] broadcast(const_0), dimensions={}
  const_2 = s32[] constant(1)
  repeat_init = (s32[],s32[10, 1]) tuple(const_0, const_1)
  while = (s32[],s32[10, 1]) while(repeat_init), condition=condition, body=body
  call = s32[10, 1] call(const_1, const_1), to_apply=func
  broadcast = s32[10, 1] get-tuple-element(while), index=1
  ROOT slice = s32[1, 1] dynamic-slice(broadcast, const_0, const_0), dynamic_slice_sizes={1, 1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto* comp = module->GetComputationWithName("entry");
  auto* replacement = comp->GetInstructionWithName("const_1");
  auto* producer = comp->GetInstructionWithName("new_shape");
  TF_ASSERT_OK(replacement->ReplaceAllUsesWithDifferentShape(producer));
  std::vector<HloInstruction*> new_shapes = {producer};
  TF_ASSERT_OK(PoplarWhileLoopOptimiser().PropagateNewShapes(new_shapes));
  module->VerifyOrAddFailure("End of test check");
}

TEST_F(WhileLoopOptimiserTest, UsedInsideWhile) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[10, 1]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  one = s32[] constant(1)
  zero = s32[] constant(0)
  add = s32[] add(p_body.0, one)
  two = s32[] constant(2)
  slice-input = s32[1, 1] reshape(two)
  p_body.1 = s32[10, 1] get-tuple-element(p_body), index=1
  dyn_update = s32[10, 1] dynamic-update-slice(p_body.1, slice-input, p_body.0, zero)
  ROOT root = (s32[],s32[]) tuple(add, dyn_update)
}

condition {
  p_cond = (s32[],s32[10, 1]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

bwd_body {
  zilch = s32[] constant(0)
  uno = s32[] constant(1)
  b_body = (s32[], s32[1, 1], s32[10, 1]) parameter(0)
  index = s32[] get-tuple-element(b_body), index=0
  to_add = s32[1, 1] get-tuple-element(b_body), index=1
  to_slice = s32[10, 1] get-tuple-element(b_body), index=2
  update = s32[1, 1] dynamic-slice(to_slice, zilch, zilch), dynamic_slice_sizes={1, 1}
  counter = s32[1, 1] add(update, to_add)
  next_index = s32[] add(index, uno)
  ROOT broot = (s32[], s32[1, 1], s32[10, 1]) tuple(next_index, counter, to_slice)
}

bwd_condition {
  b_cond = (s32[], s32[1, 1], s32[10, 1]) parameter(0)
  b_cond.0 = s32[] get-tuple-element(b_cond), index=0
  b_const = s32[] constant(10)
  ROOT result = pred[] compare(b_cond.0, b_const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[10, 1] broadcast(const_0), dimensions={}
  const_2 = s32[] constant(1)
  repeat_init = (s32[],s32[10, 1]) tuple(const_0, const_1)
  while = (s32[],s32[10, 1]) while(repeat_init), condition=condition, body=body
  broadcast = s32[10, 1] get-tuple-element(while), index=1
  slice = s32[1, 1] dynamic-slice(broadcast, const_0, const_0), dynamic_slice_sizes={1, 1}

  bwd_init = (s32[], s32[1, 1], s32[10, 1]) tuple(const_0, slice, broadcast)
  bwd_while = (s32[], s32[1, 1], s32[10, 1]) while(bwd_init), condition=bwd_condition, body=bwd_body
  ROOT final = s32[1, 1] get-tuple-element(bwd_while), index=1
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto count = PoplarWhileLoopOptimiser().Run(module.get()).ValueOrDie();
  ASSERT_TRUE(count);
}

TEST_F(WhileLoopOptimiserTest, IndexTooLarge) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[10, 1]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  one = s32[] constant(1)
  zero = s32[] constant(0)
  add = s32[] add(p_body.0, one)
  two = s32[] constant(2)
  slice-input = s32[1, 1] reshape(two)
  p_body.1 = s32[10, 1] get-tuple-element(p_body), index=1
  dyn_update = s32[10, 1] dynamic-update-slice(p_body.1, slice-input, p_body.0, zero)
  ROOT root = (s32[],s32[]) tuple(add, dyn_update)
}

condition {
  p_cond = (s32[],s32[10, 1]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(3)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[10, 1] broadcast(const_0), dimensions={}
  const_2 = s32[] constant(1)
  const_3 = s32[] constant(9)
  repeat_init = (s32[],s32[10, 1]) tuple(const_0, const_1)
  while = (s32[],s32[10, 1]) while(repeat_init), condition=condition, body=body
  broadcast = s32[10, 1] get-tuple-element(while), index=1
  ROOT slice = s32[1, 1] dynamic-slice(broadcast, const_3, const_0), dynamic_slice_sizes={1, 1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto count = PoplarWhileLoopOptimiser().Run(module.get()).ValueOrDie();
  ASSERT_FALSE(count);
}

TEST_F(WhileLoopOptimiserTest, IndexUnknown) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[10, 1]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  one = s32[] constant(1)
  zero = s32[] constant(0)
  add = s32[] add(p_body.0, one)
  two = s32[] constant(2)
  slice-input = s32[1, 1] reshape(two)
  p_body.1 = s32[10, 1] get-tuple-element(p_body), index=1
  dyn_update = s32[10, 1] dynamic-update-slice(p_body.1, slice-input, p_body.0, zero)
  ROOT root = (s32[],s32[]) tuple(add, dyn_update)
}

condition {
  p_cond = (s32[],s32[10, 1]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(3)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[10, 1] broadcast(const_0), dimensions={}
  const_2 = s32[] constant(1)
  const_3 = s32[] parameter(0)
  repeat_init = (s32[],s32[10, 1]) tuple(const_0, const_1)
  while = (s32[],s32[10, 1]) while(repeat_init), condition=condition, body=body
  broadcast = s32[10, 1] get-tuple-element(while), index=1
  ROOT slice = s32[1, 1] dynamic-slice(broadcast, const_3, const_0), dynamic_slice_sizes={1, 1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto count = PoplarWhileLoopOptimiser().Run(module.get()).ValueOrDie();
  ASSERT_FALSE(count);
}

TEST_F(WhileLoopOptimiserTest, IndexDecrementedInBwds) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[], s32[10, 1], s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  one = s32[] constant(1)
  zero = s32[] constant(0)
  add = s32[] add(p_body.0, one)
  two = s32[] constant(2)
  slice-input = s32[1, 1] reshape(two)
  p_body.1 = s32[10, 1] get-tuple-element(p_body), index=1
  dyn_update = s32[10, 1] dynamic-update-slice(p_body.1, slice-input, p_body.0, zero)
  count = s32[] get-tuple-element(p_body), index=2
  ROOT root = (s32[],s32[]) tuple(add, dyn_update, count)
}

condition {
  p_cond = (s32[],s32[10, 1], s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] get-tuple-element(p_cond), index=2
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

bwd_body {
  zilch = s32[] constant(0)
  uno = s32[] constant(1)
  b_body = (s32[], s32[1, 1], s32[10, 1], s32[]) parameter(0)
  index = s32[] get-tuple-element(b_body), index=0
  to_add = s32[1, 1] get-tuple-element(b_body), index=1
  to_slice = s32[10, 1] get-tuple-element(b_body), index=2

  old-trip-count = s32[] get-tuple-element(b_body), index=3
  minus-one = s32[] constant(-1)
  negative-index = s32[] multiply(index, minus-one)
  slice-index-p1 = s32[] add(old-trip-count, negative-index)
  slice-index = s32[] add(slice-index-p1, minus-one)

  update = s32[1, 1] dynamic-slice(to_slice, slice-index, zilch), dynamic_slice_sizes={1, 1}
  counter = s32[1, 1] add(update, to_add)
  next_index = s32[] add(index, uno)
  ROOT broot = (s32[], s32[1, 1], s32[10, 1], s32[]) tuple(next_index, counter, to_slice, old-trip-count)
}

bwd_condition {
  b_cond = (s32[], s32[1, 1], s32[10, 1], s32[]) parameter(0)
  b_cond.0 = s32[] get-tuple-element(b_cond), index=0
  b_const = s32[] constant(10)
  ROOT result = pred[] compare(b_cond.0, b_const), direction=LT
}

ENTRY entry {
  loop_counter = s32[] parameter(0)
  const_0 = s32[] constant(0)
  const_1 = s32[10, 1] broadcast(const_0), dimensions={}
  const_2 = s32[] constant(1)
  const_m1 = s32[] constant(-1)
  repeat_init = (s32[],s32[10, 1], s32[]) tuple(const_0, const_1, loop_counter)
  while = (s32[],s32[10, 1], s32[]) while(repeat_init), condition=condition, body=body
  old_trip_count = s32[] get-tuple-element(while), index=0
  broadcast = s32[10, 1] get-tuple-element(while), index=1
  otc-m1 = s32[] add(old_trip_count, const_m1)
  slice = s32[1, 1] dynamic-slice(broadcast, otc-m1, const_0), dynamic_slice_sizes={1, 1}
  bwd_init = (s32[], s32[1, 1], s32[10, 1], s32[]) tuple(const_0, slice, broadcast, old_trip_count)
  bwd_while = (s32[], s32[1, 1], s32[10, 1], s32[]) while(bwd_init), condition=bwd_condition, body=bwd_body
  ROOT final = s32[1, 1] get-tuple-element(bwd_while), index=1
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto count = PoplarWhileLoopOptimiser().Run(module.get()).ValueOrDie();
  ASSERT_TRUE(count);
}

TEST_F(WhileLoopOptimiserTest, OneBadRead) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[], s32[10, 1], s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  one = s32[] constant(1)
  zero = s32[] constant(0)
  add = s32[] add(p_body.0, one)
  two = s32[] constant(2)
  slice-input = s32[1, 1] reshape(two)
  p_body.1 = s32[10, 1] get-tuple-element(p_body), index=1
  dyn_update = s32[10, 1] dynamic-update-slice(p_body.1, slice-input, p_body.0, zero)
  count = s32[] get-tuple-element(p_body), index=2
  ROOT root = (s32[],s32[]) tuple(add, dyn_update, count)
}

condition {
  p_cond = (s32[],s32[10, 1], s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] get-tuple-element(p_cond), index=2
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

bwd_body {
  zilch = s32[] constant(0)
  uno = s32[] constant(1)
  b_body = (s32[], s32[1, 1], s32[10, 1], s32[]) parameter(0)
  index = s32[] get-tuple-element(b_body), index=0
  to_add = s32[1, 1] get-tuple-element(b_body), index=1
  to_slice = s32[10, 1] get-tuple-element(b_body), index=2

  old-trip-count = s32[] get-tuple-element(b_body), index=3
  minus-one = s32[] constant(-1)
  negative-index = s32[] multiply(index, minus-one)
  slice-index-p1 = s32[] add(old-trip-count, negative-index)
  slice-index = s32[] add(slice-index-p1, minus-one)

  update = s32[1, 1] dynamic-slice(to_slice, slice-index-p1, zilch), dynamic_slice_sizes={1, 1}
  counter = s32[1, 1] add(update, to_add)
  next_index = s32[] add(index, uno)
  ROOT broot = (s32[], s32[1, 1], s32[10, 1], s32[]) tuple(next_index, counter, to_slice, old-trip-count)
}

bwd_condition {
  b_cond = (s32[], s32[1, 1], s32[10, 1], s32[]) parameter(0)
  b_cond.0 = s32[] get-tuple-element(b_cond), index=0
  b_const = s32[] constant(10)
  ROOT result = pred[] compare(b_cond.0, b_const), direction=LT
}

ENTRY entry {
  loop_counter = s32[] parameter(0)
  const_0 = s32[] constant(0)
  const_1 = s32[10, 1] broadcast(const_0), dimensions={}
  const_2 = s32[] constant(1)
  const_m1 = s32[] constant(-1)
  repeat_init = (s32[],s32[10, 1], s32[]) tuple(const_0, const_1, loop_counter)
  while = (s32[],s32[10, 1], s32[]) while(repeat_init), condition=condition, body=body
  old_trip_count = s32[] get-tuple-element(while), index=0
  broadcast = s32[10, 1] get-tuple-element(while), index=1
  otc-m1 = s32[] add(old_trip_count, const_m1)
  slice = s32[1, 1] dynamic-slice(broadcast, otc-m1, const_0), dynamic_slice_sizes={1, 1}
  bwd_init = (s32[], s32[1, 1], s32[10, 1], s32[]) tuple(const_0, slice, broadcast, old_trip_count)
  bwd_while = (s32[], s32[1, 1], s32[10, 1], s32[]) while(bwd_init), condition=bwd_condition, body=bwd_body
  ROOT final = s32[1, 1] get-tuple-element(bwd_while), index=1
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto count = PoplarWhileLoopOptimiser().Run(module.get()).ValueOrDie();
  ASSERT_FALSE(count);
}

TEST_F(WhileLoopOptimiserTest, EliminateSingleBroadcast) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[10, 1]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  one = s32[] constant(1)
  zero = s32[] constant(0)
  add = s32[] add(p_body.0, one)
  two = s32[] constant(2)
  slice-input = s32[1, 1] reshape(two)
  p_body.1 = s32[10, 1] get-tuple-element(p_body), index=1
  dyn_update = s32[10, 1] dynamic-update-slice(p_body.1, slice-input, p_body.0, zero)
  ROOT root = (s32[],s32[]) tuple(add, dyn_update)
}

condition {
  p_cond = (s32[],s32[10, 1]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[10, 1] broadcast(const_0), dimensions={}
  const_2 = s32[] constant(1)
  repeat_init = (s32[],s32[10, 1]) tuple(const_0, const_1)
  while = (s32[],s32[10, 1]) while(repeat_init), condition=condition, body=body
  broadcast = s32[10, 1] get-tuple-element(while), index=1
  ROOT slice = s32[1, 1] dynamic-slice(broadcast, const_0, const_0), dynamic_slice_sizes={1, 1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          PoplarWhileLoopOptimiser().Run(module.get()));
  ASSERT_EQ(changed, true);
  TF_ASSERT_OK_AND_ASSIGN(changed, TupleSimplifier().Run(module.get()));

  auto* comp = module->GetComputationWithName("entry");
  auto* slice = comp->root_instruction();
  // Check this slice will be removed by later passes
  ASSERT_EQ(slice->shape(), slice->operand(0)->shape());
  ASSERT_TRUE(slice->operand(0)->opcode() != HloOpcode::kGetTupleElement);
}

TEST_F(WhileLoopOptimiserTest, EliminateMultipleBroadcast) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[], s32[10, 1], s32[1, 1], s32[10, 1]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  p_body.2 = s32[1, 1] get-tuple-element(p_body), index=2
  p_body.3 = s32[10, 1] get-tuple-element(p_body), index=3
  one = s32[] constant(1)
  zero = s32[] constant(0)
  add = s32[] add(p_body.0, one)
  two = s32[] constant(2)
  slice-input = s32[1, 1] reshape(two)
  p_body.1 = s32[10, 1] get-tuple-element(p_body), index=1
  dyn_update = s32[10, 1] dynamic-update-slice(p_body.1, slice-input, p_body.0, zero)
  dyn_update.2 = s32[10, 1] dynamic-update-slice(p_body.3, p_body.2, p_body.0, zero)
  ROOT root = (s32[], s32[10, 1], s32[1, 1], s32[10, 1]) tuple(add, dyn_update, p_body.2, dyn_update.2)
}

condition {
  p_cond = (s32[], s32[10, 1], s32[1, 1], s32[10, 1]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[10, 1] broadcast(const_0), dimensions={}
  const_2 = s32[] constant(1)
  reshape = s32[1, 1] reshape(const_0)
  repeat_init = (s32[], s32[10, 1], s32[1, 1], s32[10, 1]) tuple(const_0, const_1, reshape, const_1)
  while = (s32[], s32[10, 1], s32[1, 1], s32[10, 1]) while(repeat_init), condition=condition, body=body
  broadcast.1 = s32[10, 1] get-tuple-element(while), index=1
  ROOT slice = s32[1, 1] dynamic-slice(broadcast.1, const_0, const_0), dynamic_slice_sizes={1, 1}
  broadcast.2 = s32[10, 1] get-tuple-element(while), index=3
  slice.2 = s32[1, 1] dynamic-slice(broadcast.2, const_0, const_0), dynamic_slice_sizes={1, 1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          PoplarWhileLoopOptimiser().Run(module.get()));
  ASSERT_EQ(changed, true);
  TF_ASSERT_OK_AND_ASSIGN(changed, TupleSimplifier().Run(module.get()));

  auto* comp = module->GetComputationWithName("entry");
  auto* slice1 = comp->root_instruction();
  auto* slice2 = comp->GetInstructionWithName("slice.2");
  for (auto* slice : {slice1, slice2}) {
    // Check this slice will be removed by later passes
    ASSERT_EQ(slice->shape(), slice->operand(0)->shape());
    ASSERT_TRUE(slice->operand(0)->opcode() != HloOpcode::kGetTupleElement);
  }
}

TEST_F(WhileLoopOptimiserTest, WhileGroups) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[], s32[10, 1], s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  one = s32[] constant(1)
  zero = s32[] constant(0)
  add = s32[] add(p_body.0, one)
  two = s32[] constant(2)
  slice-input = s32[1, 1] reshape(two)
  p_body.1 = s32[10, 1] get-tuple-element(p_body), index=1
  dyn_update = s32[10, 1] dynamic-update-slice(p_body.1, slice-input, p_body.0, zero)
  count = s32[] get-tuple-element(p_body), index=2
  ROOT root = (s32[],s32[]) tuple(add, dyn_update, count)
}

condition {
  p_cond = (s32[],s32[10, 1], s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] get-tuple-element(p_cond), index=2
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

bwd_body {
  zilch = s32[] constant(0)
  uno = s32[] constant(1)
  b_body = (s32[], s32[1, 1], s32[10, 1], s32[]) parameter(0)
  index = s32[] get-tuple-element(b_body), index=0
  to_add = s32[1, 1] get-tuple-element(b_body), index=1
  to_slice = s32[10, 1] get-tuple-element(b_body), index=2

  old-trip-count = s32[] get-tuple-element(b_body), index=3
  minus-one = s32[] constant(-1)
  negative-index = s32[] multiply(index, minus-one)
  slice-index-p1 = s32[] add(old-trip-count, negative-index)
  slice-index = s32[] add(slice-index-p1, minus-one)

  update = s32[1, 1] dynamic-slice(to_slice, slice-index, zilch), dynamic_slice_sizes={1, 1}
  counter = s32[1, 1] add(update, to_add)
  next_index = s32[] add(index, uno)
  ROOT broot = (s32[], s32[1, 1], s32[10, 1], s32[]) tuple(next_index, counter, to_slice, old-trip-count)
}

bwd_condition {
  b_cond = (s32[], s32[1, 1], s32[10, 1], s32[]) parameter(0)
  b_cond.0 = s32[] get-tuple-element(b_cond), index=0
  b_const = s32[] constant(10)
  ROOT result = pred[] compare(b_cond.0, b_const), direction=LT
}

ENTRY entry {
  loop_counter = s32[] parameter(0)
  const_00 = s32[] constant(0)
  const_1 = s32[10, 1] broadcast(const_00), dimensions={}
  const_2 = s32[] constant(1)
  const_m1 = s32[] constant(-1)
  repeat_init = (s32[],s32[10, 1], s32[]) tuple(const_00, const_1, loop_counter)
  while = (s32[],s32[10, 1], s32[]) while(repeat_init), condition=condition, body=body
  old_trip_count = s32[] get-tuple-element(while), index=0
  broadcast = s32[10, 1] get-tuple-element(while), index=1
  otc-m1 = s32[] add(old_trip_count, const_m1)
  slice = s32[1, 1] dynamic-slice(broadcast, otc-m1, const_00), dynamic_slice_sizes={1, 1}
  bwd_init = (s32[], s32[1, 1], s32[10, 1], s32[]) tuple(const_00, slice, broadcast, old_trip_count)
  bwd_while = (s32[], s32[1, 1], s32[10, 1], s32[]) while(bwd_init), condition=bwd_condition, body=bwd_body
  ROOT final = s32[1, 1] get-tuple-element(bwd_while), index=1
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto allocations =
      AllocationGroups::CreateAllocationGroups(module.get()).ValueOrDie();
  allocations.Verify(module.get());

  auto* const_00 =
      module->entry_computation()->GetInstructionWithName("const_00");
  const auto loc_to_group = allocations.CreateLocationToGroupMap();

  auto* otc_m1 = module->entry_computation()->GetInstructionWithName("otc-m1");
  auto* bwd_while =
      module->entry_computation()->GetInstructionWithName("bwd_while");

  ASSERT_EQ(loc_to_group.at({const_00, {}}), loc_to_group.at({otc_m1, {}}));
  ASSERT_EQ(loc_to_group.at({const_00, {}}), loc_to_group.at({bwd_while, {3}}));
}

TEST_F(WhileLoopOptimiserTest, RepeatGroups) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[], s32[10, 1], s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  one = s32[] constant(1)
  zero = s32[] constant(0)
  add = s32[] add(p_body.0, one)
  two = s32[] constant(2)
  slice-input = s32[1, 1] reshape(two)
  p_body.1 = s32[10, 1] get-tuple-element(p_body), index=1
  dyn_update = s32[10, 1] dynamic-update-slice(p_body.1, slice-input, p_body.0, zero)
  count = s32[] get-tuple-element(p_body), index=2
  ROOT root = (s32[],s32[]) tuple(add, dyn_update, count)
}

condition {
  p_cond = (s32[],s32[10, 1], s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] get-tuple-element(p_cond), index=2
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

bwd_body {
  zilch = s32[] constant(0)
  uno = s32[] constant(1)
  b_body = (s32[], s32[1, 1], s32[10, 1], s32[]) parameter(0)
  index = s32[] get-tuple-element(b_body), index=0
  to_add = s32[1, 1] get-tuple-element(b_body), index=1
  to_slice = s32[10, 1] get-tuple-element(b_body), index=2

  old-trip-count = s32[] get-tuple-element(b_body), index=3
  minus-one = s32[] constant(-1)
  negative-index = s32[] multiply(index, minus-one)
  slice-index-p1 = s32[] add(old-trip-count, negative-index)
  slice-index = s32[] add(slice-index-p1, minus-one)

  update = s32[1, 1] dynamic-slice(to_slice, slice-index, zilch), dynamic_slice_sizes={1, 1}
  counter = s32[1, 1] add(update, to_add)
  next_index = s32[] add(index, uno)
  ROOT broot = (s32[], s32[1, 1], s32[10, 1], s32[]) tuple(next_index, counter, to_slice, old-trip-count)
}

bwd_condition {
  b_cond = (s32[], s32[1, 1], s32[10, 1], s32[]) parameter(0)
  b_cond.0 = s32[] get-tuple-element(b_cond), index=0
  b_const = s32[] constant(10)
  ROOT result = pred[] compare(b_cond.0, b_const), direction=LT
}

ENTRY entry {
  loop_counter = s32[] constant(20)
  const_00 = s32[] constant(0)
  const_1 = s32[10, 1] broadcast(const_00), dimensions={}
  const_2 = s32[] constant(1)
  const_m1 = s32[] constant(-1)
  repeat_init = (s32[],s32[10, 1], s32[]) tuple(const_00, const_1, loop_counter)
  while = (s32[],s32[10, 1], s32[]) while(repeat_init), condition=condition, body=body
  old_trip_count = s32[] get-tuple-element(while), index=0
  broadcast = s32[10, 1] get-tuple-element(while), index=1
  otc-m1 = s32[] add(old_trip_count, const_m1)
  slice = s32[1, 1] dynamic-slice(broadcast, otc-m1, const_00), dynamic_slice_sizes={1, 1}
  bwd_init = (s32[], s32[1, 1], s32[10, 1], s32[]) tuple(const_00, slice, broadcast, old_trip_count)
  bwd_while = (s32[], s32[1, 1], s32[10, 1], s32[]) while(bwd_init), condition=bwd_condition, body=bwd_body
  ROOT final = s32[1, 1] get-tuple-element(bwd_while), index=1
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  CHECK(WhileLoopToRepeatSimplify().Run(module.get()).ValueOrDie());
  WhileLoopToRepeatSimplify().Run(module.get()).ValueOrDie();
  auto allocations =
      AllocationGroups::CreateAllocationGroups(module.get()).ValueOrDie();

  allocations.Verify(module.get());
  auto* const_00 =
      module->entry_computation()->GetInstructionWithName("const_00");
  const auto loc_to_group = allocations.CreateLocationToGroupMap();

  auto* otc_m1 = module->entry_computation()->GetInstructionWithName("otc-m1");
  auto* bwd_while = module->entry_computation()->GetInstructionWithName("call");

  ASSERT_EQ(loc_to_group.at({const_00, {}}), loc_to_group.at({otc_m1, {}}));
  ASSERT_EQ(loc_to_group.at({const_00, {}}), loc_to_group.at({bwd_while, {3}}));
}

TEST_F(WhileLoopOptimiserTest, CallGroups) {
  const char* const hlo_string = R"(
HloModule top

add_float {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT a = f32[] add(f32[] x, f32[] y)
}

stage_0_fwd {
  stage_0_fwd_t = token[] after-all()
  stage_0_fwd_feed = (f32[1,4,4,2], token[]) infeed(stage_0_fwd_t)
  stage_0_fwd_input = f32[1,4,4,2] get-tuple-element(stage_0_fwd_feed),
  index=0 stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_acts_0 = f32[1,4,4,2] add(stage_0_fwd_input, stage_0_fwd_weights0)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_0_fwd_acts_0, stage_0_fwd_input)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_1_fwd_acts_1 = f32[1,4,4,2] add(stage_1_fwd_acts_0,
  stage_1_fwd_weights1) stage_1_fwd_zero = f32[] constant(0)
  stage_1_fwd_reduce = f32[] reduce(stage_1_fwd_acts_1, stage_1_fwd_zero),
  dimensions={0,1,2,3}, to_apply=add_float ROOT stage_1_fwd_tuple = (f32[],
  f32[1,4,4,2]) tuple(stage_1_fwd_reduce, stage_1_fwd_acts_0)
}

stage_1_bwd {
  stage_1_bwd_reduce = f32[] parameter(0)
  stage_1_bwd_bcast1 = f32[1,4,4,2] broadcast(stage_1_bwd_reduce),
  dimensions={} stage_1_bwd_acts_0 = f32[1,4,4,2] parameter(1)
  stage_1_bwd_acts_0_bwd = f32[1,4,4,2] add(stage_1_bwd_acts_0,
  stage_1_bwd_bcast1) stage_1_bwd_lr = f32[] constant(0.01)
  stage_1_bwd_lr_bcast = f32[1,4,4,2] broadcast(stage_1_bwd_lr),
  dimensions={} stage_1_bwd_update = f32[1,4,4,2]
  multiply(stage_1_bwd_acts_0_bwd, stage_1_bwd_lr_bcast) stage_1_bwd_weights1
  = f32[1,4,4,2] parameter(2) stage_1_bwd_weights1_new = f32[1,4,4,2]
  subtract(stage_1_bwd_weights1, stage_1_bwd_update) ROOT stage_1_bwd_tuple =
  (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_bwd_acts_0_bwd,
  stage_1_bwd_weights1_new, stage_1_bwd_update)
}

stage_0_bwd {
  stage_0_bwd_acts_0_bwd = f32[1,4,4,2] parameter(0)
  stage_0_bwd_input = f32[1,4,4,2] parameter(1)
  stage_0_bwd_input_bwd = f32[1,4,4,2] add(stage_0_bwd_input,
  stage_0_bwd_acts_0_bwd) stage_0_bwd_lr = f32[] constant(0.01)
  stage_0_bwd_lr_bcast = f32[1,4,4,2] broadcast(stage_0_bwd_lr),
  dimensions={} stage_0_bwd_update = f32[1,4,4,2]
  multiply(stage_0_bwd_input_bwd, stage_0_bwd_lr_bcast) stage_0_bwd_weights0
  = f32[1,4,4,2] parameter(2) stage_0_bwd_weights0_new = f32[1,4,4,2]
  subtract(stage_0_bwd_weights0, stage_0_bwd_update) ROOT stage_0_bwd_tuple =
  (f32[1,4,4,2]) tuple(stage_0_bwd_weights0_new)
}

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0, arg1)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0),
    to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  pipeline_acts_0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_input = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[], f32[1,4,4,2]) call(pipeline_acts_0, pipeline_weights1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  pipeline_reduce = f32[] get-tuple-element(pipeline_stage_1), index=0
  pipeline_acts_0_local = f32[1,4,4,2] get-tuple-element(pipeline_stage_1),index=1
  pipeline_stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_reduce, pipeline_acts_0_local, pipeline_weights1), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  pipeline_acts_0_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd), index=0
  pipeline_weights1_new = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd), index=1
  pipeline_stage_0_bwd = (f32[1,4,4,2]) call(pipeline_acts_0_bwd, pipeline_input, pipeline_weights0), to_apply=stage_0_bwd,
  backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  pipeline_weights0_new = f32[1,4,4,2] get-tuple-element(pipeline_stage_0_bwd), index=0
  call_ru = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0_new, pipeline_weights1_new), to_apply=resource_update
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1),
  to_apply=pipeline,
  backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  CustomOpReplacer().Run(module.get()).ValueOrDie();
  auto allocations =
      AllocationGroups::CreateAllocationGroups(module.get()).ValueOrDie();
  allocations.Verify(module.get());
  const auto loc_to_group = allocations.CreateLocationToGroupMap();

  auto* e_weights1 =
      module->entry_computation()->GetInstructionWithName("e.weights1");
  auto* call_ru = module->GetComputationWithName("pipeline")
                      ->GetInstructionWithName("call_ru");
  ASSERT_EQ(loc_to_group.at({e_weights1, {}}), loc_to_group.at({call_ru, {1}}));
}

TEST_F(WhileLoopOptimiserTest, GroupsWithTokens) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[], s32[10, 1], s32[], token[]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  one = s32[] constant(1)
  zero = s32[] constant(0)
  add = s32[] add(p_body.0, one)
  two = s32[] constant(2)
  slice-input = s32[1, 1] reshape(two)
  p_body.1 = s32[10, 1] get-tuple-element(p_body), index=1
  dyn_update = s32[10, 1] dynamic-update-slice(p_body.1, slice-input, p_body.0, zero)
  count = s32[] get-tuple-element(p_body), index=2
  tok = token[] get-tuple-element(p_body), index=3
  ROOT root = (s32[],s32[]) tuple(add, dyn_update, count, tok)
}

condition {
  p_cond = (s32[],s32[10, 1], s32[], token[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] get-tuple-element(p_cond), index=2
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

bwd_body {
  zilch = s32[] constant(0)
  uno = s32[] constant(1)
  b_body = (s32[], s32[1, 1], s32[10, 1], s32[]) parameter(0)
  index = s32[] get-tuple-element(b_body), index=0
  to_add = s32[1, 1] get-tuple-element(b_body), index=1
  to_slice = s32[10, 1] get-tuple-element(b_body), index=2

  old-trip-count = s32[] get-tuple-element(b_body), index=3
  minus-one = s32[] constant(-1)
  negative-index = s32[] multiply(index, minus-one)
  slice-index-p1 = s32[] add(old-trip-count, negative-index)
  slice-index = s32[] add(slice-index-p1, minus-one)

  update = s32[1, 1] dynamic-slice(to_slice, slice-index, zilch), dynamic_slice_sizes={1, 1}
  counter = s32[1, 1] add(update, to_add)
  next_index = s32[] add(index, uno)
  ROOT broot = (s32[], s32[1, 1], s32[10, 1], s32[]) tuple(next_index, counter, to_slice, old-trip-count)
}

bwd_condition {
  b_cond = (s32[], s32[1, 1], s32[10, 1], s32[]) parameter(0)
  b_cond.0 = s32[] get-tuple-element(b_cond), index=0
  b_const = s32[] constant(10)
  ROOT result = pred[] compare(b_cond.0, b_const), direction=LT
}

ENTRY entry {
  loop_counter = s32[] parameter(0)
  p_token = token[] after-all()
  c_token = token[] copy(p_token)
  const_00 = s32[] constant(0)
  const_1 = s32[10, 1] broadcast(const_00), dimensions={}
  const_2 = s32[] constant(1)
  const_m1 = s32[] constant(-1)
  repeat_init = (s32[],s32[10, 1], s32[], token[]) tuple(const_00, const_1, loop_counter, p_token)
  while = (s32[],s32[10, 1], s32[], token[]) while(repeat_init), condition=condition, body=body
  old_trip_count = s32[] get-tuple-element(while), index=0
  broadcast = s32[10, 1] get-tuple-element(while), index=1
  otc-m1 = s32[] add(old_trip_count, const_m1)
  slice = s32[1, 1] dynamic-slice(broadcast, otc-m1, const_00), dynamic_slice_sizes={1, 1}
  bwd_init = (s32[], s32[1, 1], s32[10, 1], s32[]) tuple(const_00, slice, broadcast, old_trip_count)
  bwd_while = (s32[], s32[1, 1], s32[10, 1], s32[]) while(bwd_init), condition=bwd_condition, body=bwd_body
  ROOT final = s32[1, 1] get-tuple-element(bwd_while), index=1
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto allocations =
      AllocationGroups::CreateAllocationGroups(module.get()).ValueOrDie();
  allocations.Verify(module.get());

  auto* const_00 =
      module->entry_computation()->GetInstructionWithName("const_00");
  const auto loc_to_group = allocations.CreateLocationToGroupMap();

  auto* otc_m1 = module->entry_computation()->GetInstructionWithName("otc-m1");
  auto* bwd_while =
      module->entry_computation()->GetInstructionWithName("bwd_while");

  auto* p_token =
      module->entry_computation()->GetInstructionWithName("p_token");
  auto* c_token =
      module->entry_computation()->GetInstructionWithName("c_token");
  auto* fwd_while =
      module->entry_computation()->GetInstructionWithName("while");

  ASSERT_EQ(loc_to_group.at({const_00, {}}), loc_to_group.at({otc_m1, {}}));
  ASSERT_EQ(loc_to_group.at({const_00, {}}), loc_to_group.at({bwd_while, {3}}));
  ASSERT_EQ(loc_to_group.find({p_token, {}}), loc_to_group.end());
  ASSERT_EQ(loc_to_group.find({c_token, {}}), loc_to_group.end());
  ASSERT_EQ(loc_to_group.find({fwd_while, {3}}), loc_to_group.end());
  ASSERT_NE(loc_to_group.find({fwd_while, {2}}), loc_to_group.end());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
