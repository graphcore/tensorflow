/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/commutative_instruction_reorder_operands.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_into_poplar_ops.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using CommutativeInstructionReorderOperandsTest = HloTestBase;

TEST_F(CommutativeInstructionReorderOperandsTest, ReorderUnary) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  i1 = f16[] parameter(0)
  i2 = f16[2, 2] parameter(1)
  b1 = f16[2, 2] broadcast(i1), dimensions={}
  ROOT a1 = f16[2, 2] add(b1, i2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  CommutativeInstructionReorderOperands ciro;
  EXPECT_TRUE(ciro.Run(module).ValueOrDie());

  {
    const auto* root_inst = comp->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kParameter);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kBroadcast);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, ReorderWithAddDependency1) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  i1 = f16[] parameter(0)
  aa = token[] after-all()
  i2 = f16[2, 2] parameter(1)
  ad = f16[2, 2] add-dependency(i2, aa)
  b1 = f16[2, 2] broadcast(i1), dimensions={}
  ROOT a1 = f16[2, 2] add(b1, ad)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  CommutativeInstructionReorderOperands ciro;
  EXPECT_TRUE(ciro.Run(module).ValueOrDie());

  {
    const auto* root_inst = comp->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kAddDependency);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kBroadcast);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, ReorderWithAddDependency2) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  i1 = f16[] parameter(0)
  aa = token[] after-all()
  i2 = f16[2, 2] parameter(1)
  b1 = f16[2, 2] broadcast(i1), dimensions={}
  ad = f16[2, 2] add-dependency(b1, aa)
  ROOT a1 = f16[2, 2] add(ad, i2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  CommutativeInstructionReorderOperands ciro;
  EXPECT_TRUE(ciro.Run(module).ValueOrDie());

  {
    const auto* root_inst = comp->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kParameter);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kAddDependency);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, DontReorderUnaryElementwise) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  i1 = f16[2, 2] parameter(0)
  i2 = f16[2, 2] parameter(1)
  e1 = f16[2, 2] exponential(i1)
  ROOT a1 = f16[2, 2] add(e1, i2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  CommutativeInstructionReorderOperands ciro;
  EXPECT_FALSE(ciro.Run(module).ValueOrDie());

  {
    const auto* root_inst = comp->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kExp);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kParameter);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, ReorderBinary) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  zero = f16[] constant(0)
  i1 = f16[2, 1] parameter(0)
  i2 = f16[2, 2] parameter(1)
  p1 = f16[2, 2] pad(i1, zero), padding=0_0x0_1
  ROOT a1 = f16[2, 2] multiply(p1, i2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  CommutativeInstructionReorderOperands ciro;
  EXPECT_TRUE(ciro.Run(module).ValueOrDie());

  {
    const auto* root_inst = comp->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kParameter);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kPad);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, DontReorderBinary) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  i1 = f16[2, 2] parameter(0)
  i2 = f16[2, 2] parameter(1)
  i3 = f16[2, 2] parameter(2)
  a1 = f16[2, 2] add(i1, i2)
  ROOT m1 = f16[2, 2] multiply(a1, i3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  CommutativeInstructionReorderOperands ciro;
  EXPECT_FALSE(ciro.Run(module).ValueOrDie());

  {
    const auto* root_inst = comp->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kAdd);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kParameter);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, DontReorderBothBroadcast) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  i1 = f16[] parameter(0)
  i2 = f16[] parameter(1)
  b1 = f16[2, 2] broadcast(i1), dimensions={}
  b2 = f16[2, 2] broadcast(i2), dimensions={}
  ROOT a1 = f16[2, 2] add(b1, b2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  auto* b1 = comp->GetInstructionWithName("b1");
  auto* b2 = comp->GetInstructionWithName("b2");

  CommutativeInstructionReorderOperands ciro;
  EXPECT_FALSE(ciro.Run(module).ValueOrDie());

  {
    const auto* root_inst = comp->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kBroadcast);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kBroadcast);
    EXPECT_THAT(root_inst->operand(0), b1);
    EXPECT_THAT(root_inst->operand(1), b2);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, MoveBroadcast) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  i1 = f16[] parameter(0)
  b1 = f16[2, 2] broadcast(i1), dimensions={}
  zero = f16[] constant(0)
  i2 = f16[2, 1] parameter(1)
  p1 = f16[2, 2] pad(i2, zero), padding=0_0x0_1
  ROOT a1 = f16[2, 2] add(b1, p1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  auto* b1 = comp->GetInstructionWithName("b1");
  auto* p1 = comp->GetInstructionWithName("p1");

  CommutativeInstructionReorderOperands ciro;
  EXPECT_TRUE(ciro.Run(module).ValueOrDie());

  {
    const auto* root_inst = comp->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kPad);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kBroadcast);
    EXPECT_THAT(root_inst->operand(0), p1);
    EXPECT_THAT(root_inst->operand(1), b1);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, DontMoveBroadcast) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  i1 = f16[] parameter(0)
  b1 = f16[2, 2] broadcast(i1), dimensions={}
  zero = f16[] constant(0)
  i2 = f16[2, 1] parameter(1)
  p1 = f16[2, 2] pad(i2, zero), padding=0_0x0_1
  ROOT a1 = f16[2, 2] add(p1, b1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  auto* b1 = comp->GetInstructionWithName("b1");
  auto* p1 = comp->GetInstructionWithName("p1");

  CommutativeInstructionReorderOperands ciro;
  EXPECT_FALSE(ciro.Run(module).ValueOrDie());

  {
    const auto* root_inst = comp->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kPad);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kBroadcast);
    EXPECT_THAT(root_inst->operand(0), p1);
    EXPECT_THAT(root_inst->operand(1), b1);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, ReorderLiveness) {
  std::string hlo = R"(
HloModule top

entry  {
  i1 = f16[2, 2] parameter(0)
  i2 = f16[2, 2] parameter(1)
  a1 = f16[2, 2] add(i1, i2)
  ROOT a2 = f16[2, 2] add(i1, a1)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));

  EXPECT_TRUE(
      CommutativeInstructionReorderOperands().Run(m.get()).ValueOrDie());
  EXPECT_TRUE(
      Match(m->entry_computation()->root_instruction(),
            m::Add(m::Parameter(0), m::Add(m::Parameter(1), m::Parameter(0)))));
}

TEST_F(CommutativeInstructionReorderOperandsTest, DontReorderBroadcast) {
  std::string hlo = R"(
HloModule top

entry  {
  i1 = f16[2, 2] parameter(0)
  c1 = f16[] constant(0)
  b1 = f16[2, 2] broadcast(c1), dimensions={}
  a1 = f16[2, 2] add(i1, b1)
  ROOT a2 = f16[2, 2] add(i1, a1)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));

  EXPECT_FALSE(
      CommutativeInstructionReorderOperands().Run(m.get()).ValueOrDie());
}

TEST_F(CommutativeInstructionReorderOperandsTest, ReorderScaledAddaXbY) {
  std::string hlo = R"(
HloModule top

entry  {
  i1 = f16[2, 2] parameter(0)
  i2 = f16[2, 2] parameter(1)
  c1 = f16[] constant(1)
  c2 = f16[] constant(2)
  b1 = f16[2, 2] broadcast(c1), dimensions={}
  b2 = f16[2, 2] broadcast(c2), dimensions={}
  m1 = f16[2, 2] multiply(i1, b1)
  m2 = f16[2, 2] multiply(i2, b2)
  a1 = f16[2, 2] add(m1, m2)
  ROOT a2 = f16[2, 2] add(i1, a1)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));

  CompilerAnnotations annotations(m.get());
  EXPECT_TRUE(FuseOpsIntoPoplarOps(annotations).Run(m.get()).ValueOrDie());
  EXPECT_TRUE(
      CommutativeInstructionReorderOperands().Run(m.get()).ValueOrDie());
  EXPECT_TRUE(
      Match(m->entry_computation()->root_instruction(),
            m::Add(m::Parameter(0),
                   m::CustomCall(m::Parameter(1), m::Parameter(0),
                                 m::ConstantScalar(2), m::ConstantScalar(1)))));
}

TEST_F(CommutativeInstructionReorderOperandsTest, ReorderScaledAddXbY) {
  std::string hlo = R"(
HloModule top

entry  {
  i1 = f16[2, 2] parameter(0)
  i2 = f16[2, 2] parameter(1)
  c1 = f16[] constant(10)
  b1 = f16[2, 2] broadcast(c1), dimensions={}
  m1 = f16[2, 2] multiply(i2, b1)
  a1 = f16[2, 2] add(i1, m1)
  ROOT a2 = f16[2, 2] add(i1, a1)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));

  CompilerAnnotations annotations(m.get());
  EXPECT_TRUE(FuseOpsIntoPoplarOps(annotations).Run(m.get()).ValueOrDie());
  EXPECT_TRUE(
      CommutativeInstructionReorderOperands().Run(m.get()).ValueOrDie());
  EXPECT_TRUE(Match(
      m->entry_computation()->root_instruction(),
      m::Add(m::Parameter(0),
             m::CustomCall(m::Parameter(1), m::Parameter(0),
                           m::ConstantScalar(10), m::ConstantScalar(1)))));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
