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

#include "tensorflow/compiler/plugin/poplar/driver/passes/elementwise_preapply.h"

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_base.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

namespace m = match;

using ElementwisePreapplyTest = HloPoplarTestBase;

TEST_F(ElementwisePreapplyTest, TestOneHotUnary) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = s32[8, 8] parameter(0)
  on = f16[] constant(1)
  off = f16[] constant(0)
  one-hot = f16[8, 8, 256] custom-call(param, on, off), custom_call_target="OneHot", backend_config="{\"depth\": 256, \"axis\": 2}"
  ROOT negate = f16[8, 8, 256] negate(one-hot)
}
)";
  // Simple case where the optimisation is done, then the negate operation is
  // folded into the on and off constants.
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(ElementwisePreapply().Run(module.get()).ValueOrDie());
  ASSERT_TRUE(HloConstantFolding().Run(module.get()).ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::OneHot)(root));
  EXPECT_TRUE(Match(root, m::CustomCall(m::Parameter(0), m::ConstantScalar(-1),
                                        m::ConstantScalar(0))));
}

// Check that there are no issues when the elementwise op is a convert,
// since the one-hot shape needs to change.
TEST_F(ElementwisePreapplyTest, TestOneHotConvert) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = s32[2] constant({1, 2})
  on = f32[] constant(1)
  off = f32[] constant(0)
  one-hot = f32[2, 3] custom-call(param, on, off), custom_call_target="OneHot", backend_config="{\"depth\": 3, \"axis\": 1}"
  ROOT convert = s32[2, 3] convert(f32[2, 3] one-hot)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());

  // Compute initial numeric result.
  TF_ASSERT_OK_AND_ASSIGN(auto expected,
                          ExecuteNoHloPassesOnIpuModel(module.get(), {}));
  // Need to clear schedule between passes when using increased logging level.
  EXPECT_TRUE(HloDescheduler().Run(module.get()).ValueOrDie());

  // Run the pass and check that we get the correct structure.
  EXPECT_TRUE(ElementwisePreapply().Run(module.get()).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::OneHot)(root));
  EXPECT_TRUE(
      Match(root, m::CustomCall(m::Constant(), m::Convert(m::ConstantScalar(1)),
                                m::Convert(m::ConstantScalar(0)))));

  // Recompute numeric result and check for equality.
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          ExecuteNoHloPassesOnIpuModel(module.get(), {}));
  EXPECT_EQ(expected.size(), 1);
  EXPECT_EQ(result.size(), 1);
  EXPECT_TRUE(LiteralTestUtil::Equal(result[0], expected[0]));
}

TEST_F(ElementwisePreapplyTest, TestOneHotBinary) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = s32[8, 8] parameter(0)
  on = f16[] constant(1)
  off = f16[] constant(0)
  one-hot = f16[8, 8, 256] custom-call(param, on, off), custom_call_target="OneHot", backend_config="{\"depth\": 256, \"axis\": 2}"
  a = f16[] constant(0.5)
  broadcast_a = f16[8, 8, 256] broadcast(a), dimensions={}
  ROOT add = f16[8, 8, 256] add(one-hot, broadcast_a)
}
)";
  // Simple case where the optimisation is done, then the add operation is
  // folded into the on and off constants.
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(ElementwisePreapply().Run(module.get()).ValueOrDie());
  ASSERT_TRUE(HloConstantFolding().Run(module.get()).ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::OneHot)(root));
  EXPECT_TRUE(Match(root, m::CustomCall(m::Parameter(0), m::ConstantScalar(1.5),
                                        m::ConstantScalar(0.5))));
}

TEST_F(ElementwisePreapplyTest, TestOneHotSubsequent) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = s32[8, 8] parameter(0)
  on = f16[] constant(1)
  off = f16[] constant(0)
  one-hot = f16[8, 8, 256] custom-call(param, on, off), custom_call_target="OneHot", backend_config="{\"depth\": 256, \"axis\": 2}"
  a = f16[] constant(0.5)
  broadcast_a = f16[8, 8, 256] broadcast(a), dimensions={}
  multiply = f16[8, 8, 256] multiply(one-hot, broadcast_a)
  b = f16[] constant(2)
  broadcast_b = f16[8, 8, 256] broadcast(b), dimensions={}
  ROOT add = f16[8, 8, 256] add(multiply, broadcast_b)
}
)";
  // The pass is called twice and both elementwise ops are preapplied, then
  // they are folded into the on and off constants.
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(ElementwisePreapply().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(ElementwisePreapply().Run(module.get()).ValueOrDie());
  ASSERT_TRUE(HloConstantFolding().Run(module.get()).ValueOrDie());
  VLOG(2) << "After rewrite \n" << module->ToString();

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::OneHot)(root));
  EXPECT_TRUE(Match(root, m::CustomCall(m::Parameter(0), m::ConstantScalar(2.5),
                                        m::ConstantScalar(2))));
}

TEST_F(ElementwisePreapplyTest, TestOneHotMultipleUsers) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = s32[8, 8] parameter(0)
  on = f16[] constant(1)
  off = f16[] constant(0)
  one-hot = f16[8, 8, 256] custom-call(param, on, off), custom_call_target="OneHot", backend_config="{\"depth\": 256, \"axis\": 2}"
  a = f16[] constant(0.5)
  broadcast_a = f16[8, 8, 256] broadcast(a), dimensions={}
  add = f16[8, 8, 256] add(one-hot, broadcast_a)
  ROOT concat = f16[8, 16, 256] concatenate(add, one-hot), dimensions={1}
}
)";
  // Because the one-hot is used in multiple places, the elementwise op cannot
  // be preapplied.
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_FALSE(ElementwisePreapply().Run(module.get()).ValueOrDie());
}

TEST_F(ElementwisePreapplyTest, TestOneHotNonUniform) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = s32[8, 8] parameter(0)
  on = f16[] constant(1)
  off = f16[] constant(0)
  one-hot = f16[8, 8, 256] custom-call(param, on, off), custom_call_target="OneHot", backend_config="{\"depth\": 256, \"axis\": 2}"
  a = f16[8] constant({0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875})
  broadcast_a = f16[8, 8, 256] broadcast(a), dimensions={1}
  ROOT add = f16[8, 8, 256] add(one-hot, broadcast_a)
}
)";
  // The elementwise op is not uniform since the constant being added is
  // multiple elements broadcasted.
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_FALSE(ElementwisePreapply().Run(module.get()).ValueOrDie());
}

TEST_F(ElementwisePreapplyTest, TestOneHotCopy) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = s32[8, 8] parameter(0)
  on = f16[] constant(1)
  off = f16[] constant(0)
  one-hot = f16[8, 8, 256] custom-call(param, on, off), custom_call_target="OneHot", backend_config="{\"depth\": 256, \"axis\": 2}"
  ROOT copy = f16[8, 8, 256] copy(one-hot)
}
)";
  // The elementwise op is a copy so the optimisation does not apply.
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_FALSE(ElementwisePreapply().Run(module.get()).ValueOrDie());
}

TEST_F(ElementwisePreapplyTest, TestBroadcastUnary) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = s32[2] constant({1,2})
  broadcast = s32[2, 3] broadcast(param), dimensions={0}
  ROOT negate = s32[2, 3] negate(broadcast)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(auto expected,
                          ExecuteNoHloPassesOnIpuModel(module.get(), {}));
  // Need to clear schedule between passes when using increased logging level.
  EXPECT_TRUE(HloDescheduler().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(ElementwisePreapply().Run(module.get()).ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(root, m::Broadcast(m::Negate(m::Constant()))));
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          ExecuteNoHloPassesOnIpuModel(module.get(), {}));
  EXPECT_EQ(expected.size(), 1);
  EXPECT_EQ(result.size(), 1);
  EXPECT_TRUE(LiteralTestUtil::Equal(result[0], expected[0]));
}

// Check that there are no issues when the elementwise op is a convert,
// since the broadcast shape needs to change.
TEST_F(ElementwisePreapplyTest, TestBroadcastConvert) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = s32[2] constant({1,2})
  broadcast = s32[2, 3] broadcast(param), dimensions={0}
  ROOT convert = f32[2, 3] convert(s32[2, 3] broadcast)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Compute initial numeric result.
  TF_ASSERT_OK_AND_ASSIGN(auto expected,
                          ExecuteNoHloPassesOnIpuModel(module.get(), {}));
  // Need to clear schedule between passes when using increased logging level.
  EXPECT_TRUE(HloDescheduler().Run(module.get()).ValueOrDie());

  // Run the pass and check that we get the correct structure.
  EXPECT_TRUE(ElementwisePreapply().Run(module.get()).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(root, m::Broadcast(m::Convert(m::Constant()))));

  // Recompute numeric result and check for equality.
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          ExecuteNoHloPassesOnIpuModel(module.get(), {}));
  EXPECT_EQ(expected.size(), 1);
  EXPECT_EQ(result.size(), 1);
  EXPECT_TRUE(LiteralTestUtil::Equal(result[0], expected[0]));
}

TEST_F(ElementwisePreapplyTest, TestBroadcastBinaryNotAllScalars) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  c1 = s32[2] constant({1, 2})
  broadcast_c1 = s32[2, 3] broadcast(c1), dimensions={0}
  c2 = s32[] constant(3)
  broadcast_c2 = s32[2, 3] broadcast(c2), dimensions={}
  ROOT add = s32[2, 3] add(broadcast_c1, broadcast_c2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Compute initial numeric result.
  TF_ASSERT_OK_AND_ASSIGN(auto expected,
                          ExecuteNoHloPassesOnIpuModel(module.get(), {}));
  // Need to clear schedule between passes when using increased logging level.
  EXPECT_TRUE(HloDescheduler().Run(module.get()).ValueOrDie());

  // Run the pass and check that we get the correct structure.
  EXPECT_TRUE(ElementwisePreapply().Run(module.get()).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(root, m::Broadcast(m::AddAnyOrder(
                              m::Broadcast(m::Constant()), m::Constant()))));

  // Recompute numeric result and check for equality.
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          ExecuteNoHloPassesOnIpuModel(module.get(), {}));
  EXPECT_EQ(expected.size(), 1);
  EXPECT_EQ(result.size(), 1);
  EXPECT_TRUE(LiteralTestUtil::Equal(result[0], expected[0]));
}

// Check that none of binary op parameters are broadcasted when they're all
// scalars
TEST_F(ElementwisePreapplyTest, TestBroadcastBinaryAllScalars) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  c1 = s32[] constant(1)
  broadcast_c1 = s32[2, 3] broadcast(c1), dimensions={}
  c2 = s32[] constant(3)
  broadcast_c2 = s32[2, 3] broadcast(c2), dimensions={}
  ROOT add = s32[2, 3] add(broadcast_c1, broadcast_c2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Compute initial numeric result.
  TF_ASSERT_OK_AND_ASSIGN(auto expected,
                          ExecuteNoHloPassesOnIpuModel(module.get(), {}));
  // Need to clear schedule between passes when using increased logging level.
  EXPECT_TRUE(HloDescheduler().Run(module.get()).ValueOrDie());

  // Run the pass and check that we get the correct structure.
  EXPECT_TRUE(ElementwisePreapply().Run(module.get()).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(
      Match(root, m::Broadcast(m::AddAnyOrder(m::Constant(), m::Constant()))));

  // Recompute numeric result and check for equality.
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          ExecuteNoHloPassesOnIpuModel(module.get(), {}));
  EXPECT_EQ(expected.size(), 1);
  EXPECT_EQ(result.size(), 1);
  EXPECT_TRUE(LiteralTestUtil::Equal(result[0], expected[0]));
}

TEST_F(ElementwisePreapplyTest, TestBroadcastSubsequent) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  a = s32[2] constant({1, 2})
  broadcast_a = s32[2, 3] broadcast(a), dimensions={0}
  b = s32[] constant(5)
  broadcast_b = s32[2, 3] broadcast(b), dimensions={}
  c = s32[] constant(4)
  broadcast_c = s32[2, 3] broadcast(c), dimensions={}
  add1 = s32[2, 3] add(broadcast_a, broadcast_b)
  ROOT add2 = s32[2, 3] add(add1, broadcast_c)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Compute initial numeric result.
  TF_ASSERT_OK_AND_ASSIGN(auto expected,
                          ExecuteNoHloPassesOnIpuModel(module.get(), {}));
  // Need to clear schedule between passes when using increased logging level.
  EXPECT_TRUE(HloDescheduler().Run(module.get()).ValueOrDie());

  // Run passes and check that we get expected results.
  EXPECT_TRUE(ElementwisePreapply().Run(module.get()).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(
      root, m::AddAnyOrder(m::Broadcast(m::Constant()),
                           m::Broadcast(m::AddAnyOrder(
                               m::Constant(), m::Broadcast(m::Constant()))))));

  EXPECT_TRUE(ElementwisePreapply().Run(module.get()).ValueOrDie());
  root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(
      root, m::Broadcast(m::AddAnyOrder(
                m::Broadcast(m::Constant()),
                m::AddAnyOrder(m::Constant(), m::Broadcast(m::Constant()))))));

  // Recompute numeric result and check for equality.
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          ExecuteNoHloPassesOnIpuModel(module.get(), {}));
  EXPECT_EQ(expected.size(), 1);
  EXPECT_EQ(result.size(), 1);
  EXPECT_TRUE(LiteralTestUtil::Equal(result[0], expected[0]));
}

TEST_F(ElementwisePreapplyTest, TestBroadcastMultipleUsers) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = s32[2] parameter(0)
  broadcast_param = s32[2, 3] broadcast(param), dimensions={0}
  a = s32[] parameter(1)
  broadcast_a = s32[2, 3] broadcast(a), dimensions={}
  b = s32[] parameter(2)
  broadcast_b = s32[2, 3] broadcast(b), dimensions={}
  add1 = s32[2, 3] add(broadcast_param, broadcast_a)
  ROOT add2 = s32[2, 3] add(broadcast_param, broadcast_b)
}
)";
  // Because the broadcast is used in multiple places, the elementwise op cannot
  // be preapplied.
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(ElementwisePreapply().Run(module.get()).ValueOrDie());
}

TEST_F(ElementwisePreapplyTest, TestBroadcastNonUniform) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = s32[2] parameter(0)
  broadcast_param = s32[2, 3] broadcast(param), dimensions={0}
  a = s32[3] parameter(1)
  broadcast_a = s32[2, 3] broadcast(a), dimensions={1}
  ROOT add = s32[2, 3] add(broadcast_param, broadcast_a)
}
)";
  // The elementwise op is not uniform since the parameter being added is
  // multiple elements broadcasted.
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(ElementwisePreapply().Run(module.get()).ValueOrDie());
}

TEST_F(ElementwisePreapplyTest, TestBroadcastCopy) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = s32[2] parameter(0)
  broadcast = s32[2, 3] broadcast(param), dimensions={0}
  ROOT copy = s32[2, 3] copy(broadcast)
}
)";
  // The elementwise op is a copy so the optimisation does not apply.
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(ElementwisePreapply().Run(module.get()).ValueOrDie());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
