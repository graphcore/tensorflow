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

#include "tensorflow/compiler/plugin/poplar/driver/passes/recomputation_input_remover.h"

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using RecomputationInputRemoverTest = HloTestBase;

TEST_F(RecomputationInputRemoverTest, TestReplaceOne) {
  const std::string hlo = R"(
HloModule top

%cluster_1  {
  p0 = f16[] parameter(0)
  p1 = f16[] parameter(1)
  p2 = f16[] parameter(2)
  s0 = f16[] sine(p0)
  a0 = f16[] add(s0, p1)
  ri0 = f16[] custom-call(p2, a0), custom_call_target="RecomputationInput"
  s1 = f16[] sine(ri0)
  m0 = f16[] multiply(s1, a0)
  m1 = f16[] multiply(m0, s0)
  ROOT %tuple = (f16[]) tuple(m1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());

  auto* comp = module->entry_computation();

  EXPECT_TRUE(RecomputationInputRemover().Run(module.get()).ValueOrDie());
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* p1 = FindInstruction(module.get(), "p1");
  HloInstruction* p2 = FindInstruction(module.get(), "p2");
  HloInstruction* s0 = FindInstruction(module.get(), "s0");
  HloInstruction* s1 = FindInstruction(module.get(), "s1");
  HloInstruction* m0 = FindInstruction(module.get(), "m0");
  HloInstruction* m1 = FindInstruction(module.get(), "m1");

  // Check that the use of ri0 has been replaced.
  EXPECT_THAT(s1->operands(), ::testing::ElementsAre(p2));

  // Check that the use of a0 in m0 has been replaced.
  EXPECT_THAT(m0->operands(), ::testing::ElementsAre(s1, p2));

  // Check the control dependencies.
  EXPECT_THAT(p0->control_predecessors(),
              ::testing::UnorderedElementsAre(p2, m0, s1));
  EXPECT_THAT(s0->control_predecessors(),
              ::testing::UnorderedElementsAre(p2, m0, s1));
  EXPECT_THAT(m1->control_predecessors().size(), 0);
}

TEST_F(RecomputationInputRemoverTest, TestReplaceMultiple) {
  const std::string hlo = R"(
HloModule top

%cluster_1  {
  p0 = f16[] parameter(0)
  p1 = f16[] parameter(1)
  l0 = f16[] log(p0)
  p2 = f16[] parameter(2)
  ri0 = f16[] custom-call(p2, l0), custom_call_target="RecomputationInput"
  a0 = f16[] add(ri0, p1)
  l1 = f16[] log(a0)
  p3 = f16[] parameter(3)
  ri1 = f16[] custom-call(p3, l1), custom_call_target="RecomputationInput"
  l2 = f16[] log(ri1)
  p4 = f16[] parameter(4)
  ri2 = f16[] custom-call(p4, l2), custom_call_target="RecomputationInput"
  m0 = f16[] multiply(ri2, l2)
  m1 = f16[] multiply(m0, l1)
  m2 = f16[] multiply(m1, l0)
  ROOT %tuple = (f16[]) tuple(m2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());

  auto* comp = module->entry_computation();

  EXPECT_TRUE(RecomputationInputRemover().Run(module.get()).ValueOrDie());
  HloInstruction* p1 = FindInstruction(module.get(), "p1");
  HloInstruction* p2 = FindInstruction(module.get(), "p2");
  HloInstruction* a0 = FindInstruction(module.get(), "a0");
  HloInstruction* l1 = FindInstruction(module.get(), "l1");
  HloInstruction* p4 = FindInstruction(module.get(), "p4");
  HloInstruction* m0 = FindInstruction(module.get(), "m0");
  HloInstruction* m1 = FindInstruction(module.get(), "m1");
  HloInstruction* m2 = FindInstruction(module.get(), "m2");

  // Check that the use of recompute instructions has been replaced.
  EXPECT_THAT(a0->operands(), ::testing::ElementsAre(p2, p1));
  EXPECT_THAT(m0->operands(), ::testing::ElementsAre(p4, p4));
  EXPECT_THAT(m2->operands(), ::testing::ElementsAre(m1, p2));

  // Check that the control dependencies have been inserted.
  EXPECT_THAT(a0->control_predecessors(),
              ::testing::UnorderedElementsAre(p4, m0));
  EXPECT_THAT(l1->control_predecessors(),
              ::testing::UnorderedElementsAre(p4, m0));
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
