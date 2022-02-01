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
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
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

  auto count = PoplarWhileLoopOptimiser().CountOptimisations(module.get());
  ASSERT_EQ(count, 1);
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

  auto count = PoplarWhileLoopOptimiser().CountOptimisations(module.get());
  ASSERT_EQ(count, 0);
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

  auto count = PoplarWhileLoopOptimiser().CountOptimisations(module.get());
  ASSERT_EQ(count, 1);
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

  auto count = PoplarWhileLoopOptimiser().CountOptimisations(module.get());
  ASSERT_EQ(count, 0);
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

  auto count = PoplarWhileLoopOptimiser().CountOptimisations(module.get());
  ASSERT_EQ(count, 0);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
