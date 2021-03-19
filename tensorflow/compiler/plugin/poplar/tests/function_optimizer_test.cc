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

#include "tensorflow/compiler/plugin/poplar/driver/passes/function_optimizer.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using FunctionOptimizerTest = HloTestBase;

TEST_F(FunctionOptimizerTest, RemoveParameter) {
  std::string hlo = R"(
HloModule top

func {
  p0 = s32[2] parameter(0)
  p1 = s32[2] parameter(1)
  a = s32[2] add(p0, p1)
  ROOT t = (s32[2], s32[2], s32[2], s32[2]) tuple(a, p0, p0, p1)
}

ENTRY e {
  p0 = s32[2] parameter(0)
  p1 = s32[2] parameter(1)
  ROOT c = (s32[2], s32[2], s32[2], s32[2]) call(p0, p1), to_apply=func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest()));
  EXPECT_TRUE(FunctionOptimizer().Run(module.get()).ValueOrDie());

  auto root = module->entry_computation()->root_instruction();
  HloInstruction* func;
  ASSERT_TRUE(
      Match(root, m::Tuple(m::GetTupleElement(m::Op(&func), 0), m::Parameter(0),
                           m::Parameter(0), m::Parameter(1))));
  ASSERT_TRUE(IsFunction(func));
  auto func_root = func->to_apply()->root_instruction();
  EXPECT_TRUE(
      Match(func_root, m::Tuple(m::Add(m::Parameter(0), m::Parameter(1)))));
}

TEST_F(FunctionOptimizerTest, RemoveParameter2) {
  std::string hlo = R"(
HloModule top

func {
  p0 = s32[2] parameter(0)
  p1 = s32[2] parameter(1)
  a = s32[2] add(p0, p1)
  ROOT t = (s32[2], s32[2], s32[2], s32[2]) tuple(a, p0, p1, p0)
}

ENTRY e {
  p0 = s32[2] parameter(0)
  p1 = s32[2] parameter(1)
  c = (s32[2], s32[2], s32[2], s32[2]) call(p0, p1), to_apply=func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  gte0 = s32[2] get-tuple-element(c), index=0
  gte1 = s32[2] get-tuple-element(c), index=1
  gte2 = s32[2] get-tuple-element(c), index=2
  gte3 = s32[2] get-tuple-element(c), index=3
  a0 = s32[2] add(gte0, gte1)
  a1 = s32[2] add(a0, gte2)
  ROOT a2 = s32[2] add(a1, gte3)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest()));
  EXPECT_TRUE(FunctionOptimizer().Run(module.get()).ValueOrDie());

  auto root = module->entry_computation()->root_instruction();
  HloInstruction* func;
  ASSERT_TRUE(
      Match(root, m::Add(m::Add(m::Add(m::GetTupleElement(m::Op(&func), 0),
                                       m::Parameter(0)),
                                m::Parameter(1)),
                         m::Parameter(0))));
  ASSERT_TRUE(IsFunction(func));
  auto func_root = func->to_apply()->root_instruction();
  EXPECT_TRUE(
      Match(func_root, m::Tuple(m::Add(m::Parameter(0), m::Parameter(1)))));
}

TEST_F(FunctionOptimizerTest, NoParameterOutputs) {
  std::string hlo = R"(
HloModule top

func {
  p0 = s32[2] parameter(0)
  p1 = s32[2] parameter(1)
  a = s32[2] add(p0, p1)
  ROOT t = (s32[2]) tuple(a)
}

ENTRY e {
  p0 = s32[2] parameter(0)
  p1 = s32[2] parameter(1)
  ROOT c = (s32[2]) call(p0, p1), to_apply=func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest()));
  EXPECT_FALSE(FunctionOptimizer().Run(module.get()).ValueOrDie());
}

TEST_F(FunctionOptimizerTest, RemoveAllUsers) {
  std::string hlo = R"(
HloModule top

func {
  p0 = s32[2] parameter(0)
  p1 = s32[2] parameter(1)
  a = s32[2] add(p0, p1)
  ROOT t = (s32[2], s32[2], s32[2], s32[2]) tuple(a, p0, p0, p1)
}

ENTRY e {
  p0 = s32[2] parameter(0)
  p1 = s32[2] parameter(1)
  c = (s32[2], s32[2], s32[2], s32[2]) call(p0, p1), to_apply=func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  gte1 = s32[2] get-tuple-element(c), index=1
  gte2 = s32[2] get-tuple-element(c), index=2
  gte3 = s32[2] get-tuple-element(c), index=3
  ROOT tuple = (s32[2], s32[2], s32[2]) tuple(gte1, gte2, gte3)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest()));
  EXPECT_TRUE(FunctionOptimizer().Run(module.get()).ValueOrDie());

  // All the GTEs and the function have been removed due to no users.
  ASSERT_EQ(module->entry_computation()->instruction_count(), 3);

  auto root = module->entry_computation()->root_instruction();
  ASSERT_TRUE(
      Match(root, m::Tuple(m::Parameter(0), m::Parameter(0), m::Parameter(1))));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
