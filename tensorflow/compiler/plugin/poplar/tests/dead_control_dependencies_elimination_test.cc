/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/dead_control_dependencies_elimination.h"

#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {
using DeadControlDependenciesEliminationTest = HloTestBase;

TEST_F(DeadControlDependenciesEliminationTest, DropControlDeps) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  arg0 = f32[] parameter(0)
  arg1 = f32[] parameter(1)
  log = f32[] log(arg0)
  add = f32[] add(arg0, arg1)
  ROOT tuple = (f32[], f32[]) tuple(log, add)
  sub = f32[] subtract(arg0, arg1), control-predecessors={log, add, tuple}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest()));
  // Can't remove any instructions - sub has a control dependency.
  EXPECT_FALSE(HloDCE().Run(module.get()).ValueOrDie());

  EXPECT_TRUE(
      DeadControlDependenciesElimination().Run(module.get()).ValueOrDie());

  HloInstruction* sub = FindInstruction(module.get(), "sub");
  EXPECT_TRUE(sub->control_predecessors().empty());
  EXPECT_TRUE(sub->control_successors().empty());

  // Check that sub has now been removed.
  EXPECT_TRUE(HloDCE().Run(module.get()).ValueOrDie());
  EXPECT_EQ(module->instruction_count(), 5);
}

TEST_F(DeadControlDependenciesEliminationTest, TransferDeps) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  arg0 = f32[] parameter(0)
  arg1 = f32[] parameter(1)
  log = f32[] log(arg0)
  sub = f32[] subtract(arg0, arg1), control-predecessors={log}
  add = f32[] add(arg0, arg1), control-predecessors={sub}
  ROOT tuple = (f32[], f32[]) tuple(log, add)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest()));
  EXPECT_TRUE(
      DeadControlDependenciesElimination().Run(module.get()).ValueOrDie());

  HloInstruction* log = FindInstruction(module.get(), "log");
  HloInstruction* sub = FindInstruction(module.get(), "sub");
  HloInstruction* add = FindInstruction(module.get(), "add");

  EXPECT_TRUE(sub->control_predecessors().empty());
  EXPECT_TRUE(sub->control_successors().empty());
  EXPECT_THAT(log->control_successors(), ::testing::ElementsAre(add));
}

TEST_F(DeadControlDependenciesEliminationTest, TestParameters) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  arg0 = f32[] parameter(0)
  arg1 = f32[] parameter(1)
  ROOT log = f32[] log(arg0)
  arg2 = f32[] parameter(2), control-predecessors={log, arg1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest()));
  EXPECT_TRUE(
      DeadControlDependenciesElimination().Run(module.get()).ValueOrDie());

  HloInstruction* arg2 = FindInstruction(module.get(), "arg2");

  EXPECT_TRUE(arg2->control_predecessors().empty());
  EXPECT_TRUE(arg2->control_successors().empty());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
