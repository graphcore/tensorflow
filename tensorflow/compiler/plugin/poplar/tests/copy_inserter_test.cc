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

#include "tensorflow/compiler/plugin/poplar/driver/passes/copy_inserter.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_information.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using CopyInserterTest = HloTestBase;

TEST_F(CopyInserterTest, DontInsertCopyParams) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT t = (f32[], f32[], f32[], f32[]) tuple(p0, p1, p1, p0)
}

)";
  auto module = ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  EXPECT_FALSE(CopyInserter().Run(module0).ValueOrDie());
}

TEST_F(CopyInserterTest, DontInsertCopyParamsAndConst) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  c = f32[] constant(0)
  ROOT t = (f32[], f32[], f32[], f32[]) tuple(p0, p1, c, p0)
}

)";
  auto module = ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  EXPECT_FALSE(CopyInserter().Run(module0).ValueOrDie());
}

TEST_F(CopyInserterTest, InsertCopy) {
  std::string hlo = R"(
HloModule top

body {
  p = (s32[4],s32[4]) parameter(0)
  p.0 = s32[4] get-tuple-element(p), index=0
  p.1 = s32[4] get-tuple-element(p), index=1
  c.0 = s32[8] concatenate(p.0, p.1), dimensions={0}
  s.0 = s32[4] slice(c.0), slice={[2:6]}
  a.0 = s32[4] add(p.0, p.0)
  ROOT root = (s32[4],s32[4]) tuple(a.0, s.0)
}

condition {
  p_cond = (s32[4],s32[4]) parameter(0)
  p_cond.0 = s32[4] get-tuple-element(p_cond), index=0
  p_s0 = s32[1] slice(p_cond.0), slice={[0:1]}
  p_s1 = s32[] reshape(p_s0)
  p_const = s32[] constant(10)
  ROOT result = pred[] compare(p_s1, p_const), direction=LT
}

ENTRY entry {
  const_0 = s32[4] constant({0, 0, 0, 0})
  repeat_init = (s32[4],s32[4]) tuple(const_0, const_0)
  ROOT while = (s32[4],s32[4]) while(repeat_init), condition=condition, body=body
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  EXPECT_TRUE(CopyInserter().Run(module0).ValueOrDie());
  HloInstruction *op0, *op1;
  EXPECT_TRUE(Match(
      module0->entry_computation()->root_instruction()->mutable_operand(0),
      m::Tuple(m::Op(&op0), m::Copy(m::Op(&op1)))));
  EXPECT_EQ(op0, op1);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
