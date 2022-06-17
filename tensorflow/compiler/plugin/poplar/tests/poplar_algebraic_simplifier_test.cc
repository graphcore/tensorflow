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

#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier.h"

#include <memory>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using ::testing::ElementsAre;
namespace m = match;

using PoplarAlgebraicSimplifierTest = HloTestBase;

// Test that A + 0 is simplified to A
TEST_F(PoplarAlgebraicSimplifierTest, AddZero) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, param0, zero));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

TEST_F(PoplarAlgebraicSimplifierTest, FactorIntegerAddition) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = s32[8] parameter(0)
      p1 = s32[8] parameter(1)
      p2 = s32[8] parameter(2)
      x = s32[8] multiply(p0, p2)
      y = s32[8] multiply(p1, p2)
      ROOT sum = s32[8] add(x, y)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::MultiplyAnyOrder(
          m::AddAnyOrder(m::Parameter(0), m::Parameter(1)), m::Parameter(2))));
}

// A*C + B*C => (A+B)*C if C is a floating-point power of 2.
TEST_F(PoplarAlgebraicSimplifierTest, FactorFpAddition) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      c = f32[] constant(0.125)
      x = f32[] multiply(p0, c)
      y = f32[] multiply(p1, c)
      ROOT sum = f32[] add(x, y)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::MultiplyAnyOrder(
                  m::AddAnyOrder(m::Parameter(0), m::Parameter(1)),
                  m::ConstantScalar(0.125))));
}

// A*C + B*C => (A+B)*C if C is a broadcast of a floating-point power of 2.
TEST_F(PoplarAlgebraicSimplifierTest, FactorFpAdditionWithBroadcast) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      c = f32[] constant(0.125)
      b = f32[4] broadcast(c), dimensions={}
      x = f32[4] multiply(p0, b)
      y = f32[4] multiply(p1, b)
      ROOT sum = f32[4] add(x, y)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::MultiplyAnyOrder(
                  m::AddAnyOrder(m::Parameter(0), m::Parameter(1)),
                  m::Broadcast(m::ConstantScalar(0.125)))));
}

// A*C + B*C => (A+B)*C simplification should not happen if C is not a
// floating-point power of 2.
TEST_F(PoplarAlgebraicSimplifierTest, FactorFpAdditionNotPowerOf2) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      c = f32[] constant(0.3)
      x = f32[] multiply(p0, c)
      y = f32[] multiply(p1, c)
      ROOT sum = f32[] add(x, y)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  EXPECT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
}

// A*C + B*C => (A+B)*C simplification should not happen if A, B, and C are
// complex numbers.
TEST_F(PoplarAlgebraicSimplifierTest, FactorFpAdditionComplex) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = c64[8] parameter(0)
      p1 = c64[8] parameter(1)
      p2 = c64[8] parameter(2)
      x = c64[8] multiply(p0, p2)
      y = c64[8] multiply(p1, p2)
      ROOT sum = c64[8] add(x, y)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  EXPECT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
}

// A*C + B*C => (A+B)*C simplification is OK if A, B, and C are complex.
TEST_F(PoplarAlgebraicSimplifierTest, FactorFpAdditionBfloat16) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = bf16[4] parameter(0)
      p1 = bf16[4] parameter(1)
      c = bf16[] constant(0.125)
      b = bf16[4] broadcast(c), dimensions={}
      x = bf16[4] multiply(p0, b)
      y = bf16[4] multiply(p1, b)
      ROOT sum = bf16[4] add(x, y)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::MultiplyAnyOrder(
                  m::AddAnyOrder(m::Parameter(0), m::Parameter(1)),
                  m::Broadcast(m::ConstantScalar(0.125)))));
}

TEST_F(PoplarAlgebraicSimplifierTest, UnsignedDivideByPowerOf2) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = u32[4] parameter(0)
      c = u32[] constant(8)
      b = u32[4] broadcast(c), dimensions={}
      ROOT d = u32[4] divide(p, b)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::ShiftRightLogical(
                  m::Parameter(0), m::Broadcast(m::ConstantScalar(3)))));
}

TEST_F(PoplarAlgebraicSimplifierTest, SignedDivideByPowerOf2) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = s32[4] parameter(0)
      c = s32[] constant(8)
      b = s32[4] broadcast(c), dimensions={}
      ROOT d = s32[4] divide(p, b)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  auto match_dividend_is_negative =
      m::Lt(m::Parameter(0), m::Broadcast(m::ConstantScalar(0)));
  auto match_abs = m::Select(match_dividend_is_negative,
                             m::Negate(m::Parameter(0)), m::Parameter(0));
  auto match_shift =
      m::ShiftRightLogical(match_abs, m::Broadcast(m::ConstantScalar(3)));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Select(match_dividend_is_negative,
                                   m::Negate(match_shift), match_shift)));
}

TEST_F(PoplarAlgebraicSimplifierTest, UnsignedRemainderByPowerOf2) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = u32[4] parameter(0)
      c = u32[] constant(8)
      b = u32[4] broadcast(c), dimensions={}
      ROOT r = u32[4] remainder(p, b)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::AndAnyOrder(m::Parameter(0),
                                        m::Broadcast(m::ConstantScalar(7)))));
}

TEST_F(PoplarAlgebraicSimplifierTest, SignedRemainderByPowerOf2) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = s32[4] parameter(0)
      c = s32[] constant(8)
      b = s32[4] broadcast(c), dimensions={}
      ROOT r = s32[4] remainder(p, b)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  auto match_dividend_is_negative =
      m::Lt(m::Parameter(0), m::Broadcast(m::ConstantScalar(0)));
  auto match_abs = m::Select(match_dividend_is_negative,
                             m::Negate(m::Parameter(0)), m::Parameter(0));
  auto match_and =
      m::AndAnyOrder(match_abs, m::Broadcast(m::ConstantScalar(7)));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Select(match_dividend_is_negative,
                                   m::Negate(match_and), match_and)));
}

// Test that A * 0 is simplified to 0
TEST_F(PoplarAlgebraicSimplifierTest, MulZero) {
  auto m = CreateNewVerifiedModule();
  Shape r0s32 = ShapeUtil::MakeShape(S32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0s32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0s32, HloOpcode::kMultiply, param0, zero));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kMultiply);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_EQ(computation->root_instruction(), zero);
}

// Test that A * 0 is simplified to 0 (where A is a float)
TEST_F(PoplarAlgebraicSimplifierTest, MulZeroFloatRHS) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kMultiply, param0, zero));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kMultiply);
  {
    PoplarAlgebraicSimplifier simplifier;
    ASSERT_FALSE(simplifier.Run(m.get()).ValueOrDie());
  }
  {
    IpuOptions_IpuAlgebraicSimplifierConfig config;
    config.set_enable_fast_math(true);
    PoplarAlgebraicSimplifier simplifier(config);
    ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
    EXPECT_EQ(computation->root_instruction(), zero);
  }
}

// Test that 0 * A is simplified to 0
TEST_F(PoplarAlgebraicSimplifierTest, MulZeroFloatLHS) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kMultiply, zero, param0));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kMultiply);
  {
    PoplarAlgebraicSimplifier simplifier;
    ASSERT_FALSE(simplifier.Run(m.get()).ValueOrDie());
  }
  {
    IpuOptions_IpuAlgebraicSimplifierConfig config;
    config.set_enable_fast_math(true);
    PoplarAlgebraicSimplifier simplifier(config);
    ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
    EXPECT_EQ(computation->root_instruction(), zero);
  }
}

// Test that A * Broadcast(0) is simplified to 0
TEST_F(PoplarAlgebraicSimplifierTest, MulZeroFloatArrayRHS) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {3, 2});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  HloInstruction* bcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(r2f32, zero, {}));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kMultiply, param0, bcast));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kMultiply);
  {
    PoplarAlgebraicSimplifier simplifier;
    ASSERT_FALSE(simplifier.Run(m.get()).ValueOrDie());
  }
  {
    IpuOptions_IpuAlgebraicSimplifierConfig config;
    config.set_enable_fast_math(true);
    PoplarAlgebraicSimplifier simplifier(config);
    ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
    EXPECT_EQ(computation->root_instruction(), bcast);
  }
}

// Test that Broadcast(0) * A is simplified to 0
TEST_F(PoplarAlgebraicSimplifierTest, MulZeroFloatArrayLHS) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {3, 2});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  HloInstruction* bcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(r2f32, zero, {}));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kMultiply, bcast, param0));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kMultiply);
  {
    PoplarAlgebraicSimplifier simplifier;
    ASSERT_FALSE(simplifier.Run(m.get()).ValueOrDie());
  }
  {
    IpuOptions_IpuAlgebraicSimplifierConfig config;
    config.set_enable_fast_math(true);
    PoplarAlgebraicSimplifier simplifier(config);
    ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
    EXPECT_EQ(computation->root_instruction(), bcast);
  }
}

TEST_F(PoplarAlgebraicSimplifierTest, MulMinus1LHS) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = s32[] parameter(0)
      c = s32[] constant(-1)
      ROOT r = s32[] multiply(c, p)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Negate(m::Parameter(0))));
}

TEST_F(PoplarAlgebraicSimplifierTest, MulMinus1RHS) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = s32[] parameter(0)
      c = s32[] constant(-1)
      ROOT r = s32[] multiply(p, c)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Negate(m::Parameter(0))));
}

TEST_F(PoplarAlgebraicSimplifierTest, MultiplyReassociateMergeConstants) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      c0 = f32[] constant(2.0)
      c1 = f32[] constant(3.0)
      multiply0 = f32[] multiply(p0, c0)
      ROOT multiply1 = f32[] multiply(multiply0, c1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0),
                                     m::Multiply(m::ConstantScalar(2.0),
                                                 m::ConstantScalar(3.0)))));
}

TEST_F(PoplarAlgebraicSimplifierTest,
       MultiplyReassociateMergeBroadcastedConstants) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4] parameter(0)
      c0 = f32[] constant(2.0)
      c1 = f32[] constant(3.0)
      b0 = f32[4] broadcast(c0), dimensions={}
      b1 = f32[4] broadcast(c1), dimensions={}
      multiply0 = f32[4] multiply(p0, b0)
      ROOT multiply1 = f32[4] multiply(multiply0, b1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Multiply(
          m::Parameter(0), m::Broadcast(m::Multiply(m::ConstantScalar(2.0),
                                                    m::ConstantScalar(3.0))))));
}

TEST_F(PoplarAlgebraicSimplifierTest,
       MultiplyReassociateMultiplyOfConstantAndBroadcast) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      c0 = f32[4] constant({2.0, 3.0, 4.0, 5.0})
      c1 = f32[] constant(3.0)
      c2 = f32[] constant(4.0)
      b0 = f32[4] broadcast(c1), dimensions={}
      b1 = f32[4] broadcast(c2), dimensions={}
      multiply0 = f32[4] multiply(c0, b0)
      ROOT multiply1 = f32[4] multiply(multiply0, b1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Multiply(
          m::Constant(), m::Broadcast(m::Multiply(m::ConstantScalar(3.0),
                                                  m::ConstantScalar(4.0))))));
}

// Test that select(true, a, b) is simplified to a
TEST_F(PoplarAlgebraicSimplifierTest, SelectTrue) {
  Shape r0s32 = ShapeUtil::MakeShape(S32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0s32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0s32, "param1"));
  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  builder.AddInstruction(HloInstruction::CreateTernary(
      r0s32, HloOpcode::kSelect, one, param0, param1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSelect);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_EQ(computation->root_instruction(), param0);
}

// Test that select(false, a, b) is simplified to b
TEST_F(PoplarAlgebraicSimplifierTest, SelectFalse) {
  Shape r0s32 = ShapeUtil::MakeShape(S32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0s32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0s32, "param1"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  builder.AddInstruction(HloInstruction::CreateTernary(
      r0s32, HloOpcode::kSelect, zero, param0, param1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSelect);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_EQ(computation->root_instruction(), param1);
}

// Test that select(a, b, b) is simplified to b
TEST_F(PoplarAlgebraicSimplifierTest, SelectIdentical) {
  Shape r0s32 = ShapeUtil::MakeShape(S32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0s32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0s32, "param1"));
  builder.AddInstruction(HloInstruction::CreateTernary(
      r0s32, HloOpcode::kSelect, param0, param1, param1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSelect);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_EQ(computation->root_instruction(), param1);
}

// Test that Reduce(Reduce(A)) -> Reduce(A)
TEST_F(PoplarAlgebraicSimplifierTest, TwoReducesToOne) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  // Create add computation.
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  HloComputation* add_computation = nullptr;
  {
    HloComputation::Builder builder(TestName() + ".add");
    const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
    HloInstruction* p0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    HloInstruction* p1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "p1"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd, p0, p1));
    add_computation = m->AddEmbeddedComputation(builder.Build());
  }
  Shape r4f32 = ShapeUtil::MakeShape(F32, {4, 5, 6, 7});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r4f32, "param"));
  std::vector<int64_t> dims0({0});
  Shape r3f32 = ShapeUtil::MakeShape(F32, {5, 6, 7});
  HloInstruction* reduce0 = builder.AddInstruction(
      HloInstruction::CreateReduce(r3f32, param, zero, dims0, add_computation));
  std::vector<int64_t> dims1({1, 2});
  Shape r1f32 = ShapeUtil::MakeShape(F32, {5});
  builder.AddInstruction(HloInstruction::CreateReduce(r1f32, reduce0, zero,
                                                      dims1, add_computation));
  m->AddEntryComputation(builder.Build());
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Reduce(m::Parameter(0), m::Op().Is(zero))));
  EXPECT_EQ(root->dimensions(), std::vector<int64_t>({0, 2, 3}));
}

// Test that Const + A is canonicalized to A + Const.
TEST_F(PoplarAlgebraicSimplifierTest, AddConstOnLHS) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(42.0f)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, constant, param0));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Add(m::Parameter(0), m::Constant())));
}

// Test that [(A + C1) + C2] => [A + (C1 + C2)] for constants C1 and C2.
TEST_F(PoplarAlgebraicSimplifierTest, AddReassociateMergeConstants) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(42.0f)));
  HloInstruction* constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(3.14159f)));

  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, param0, constant1));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, add1, constant2));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Add(
                        m::Op().Is(param0),
                        m::Add(m::Op().Is(constant1), m::Op().Is(constant2)))));
}

TEST_F(PoplarAlgebraicSimplifierTest, AddReassociateMergeBroadcastedConstants) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4] parameter(0)
      c0 = f32[] constant(1.0)
      c1 = f32[] constant(2.0)
      b0 = f32[4] broadcast(c0), dimensions={}
      b1 = f32[4] broadcast(c1), dimensions={}
      add0 = f32[4] add(p0, b0)
      ROOT add1 = f32[4] add(add0, b1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Add(m::Parameter(0),
                                m::Broadcast(m::Add(m::ConstantScalar(1.0),
                                                    m::ConstantScalar(2.0))))));
}

TEST_F(PoplarAlgebraicSimplifierTest, AddBroadcastZeroR0Operand) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {3, 2});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  HloInstruction* bcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(r2f32, zero, {0, 1}));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kAdd, bcast, param0));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

TEST_F(PoplarAlgebraicSimplifierTest, InlineTrivialMap) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  // Create add computation.
  HloComputation* add_computation = nullptr;
  {
    HloComputation::Builder builder(TestName() + ".add");
    const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
    HloInstruction* p0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    HloInstruction* p1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "p1"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd, p0, p1));
    add_computation = m->AddEmbeddedComputation(builder.Build());
  }
  Shape r2f32 = ShapeUtil::MakeShape(F32, {32, 1});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  builder.AddInstruction(HloInstruction::CreateMap(
      r2f32,
      {param0, builder.AddInstruction(
                   HloInstruction::CreateBroadcast(r2f32, zero, {}))},
      add_computation));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kMap);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Add(m::Parameter(0),
                                      m::Broadcast(m::Op().Is(zero)))));
}

TEST_F(PoplarAlgebraicSimplifierTest, KeepNontrivialMap) {
  const char* kModuleStr = R"(
    HloModule m
    fusion {
      x = f32[] parameter(0)
      c = f32[] constant(42)
      m = f32[] multiply(x, x)
      ROOT a = f32[] add(m, c)
    }

    map {
      x = f32[] parameter(0)
      ROOT f = f32[] fusion(x), kind=kLoop, calls=fusion
    }

    ENTRY test {
      p = f32[2,2] parameter(0)
      ROOT map = f32[2,2] map(p), dimensions={0,1}, to_apply=map
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(PoplarAlgebraicSimplifierTest, AddBroadcastZeroR1Operand) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {3, 2});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({0, 0, 0})));
  HloInstruction* bcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(r2f32, zero, {1}));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kAdd, bcast, param0));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

TEST_F(PoplarAlgebraicSimplifierTest, ConstantToBroadcast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({3.14f, 3.14f, 3.14f})));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::Constant())));
  EXPECT_EQ(3.14f, root->operand(0)->literal().GetFirstElement<float>());
}

TEST_F(PoplarAlgebraicSimplifierTest, ConstantToBroadcastPreservesSharding) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  arg0 = s32[] parameter(0)
  ROOT constant = s32[2] constant({2, 2}), sharding={maximal device=1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  auto* original_constant = FindInstruction(module.get(), "constant");
  ASSERT_TRUE(original_constant);
  auto expected_sharding = original_constant->sharding();

  PoplarAlgebraicSimplifier simplifier;
  TF_ASSERT_OK_AND_ASSIGN(auto changed, simplifier.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* entry = module->entry_computation();
  auto* broadcast = entry->root_instruction();
  ASSERT_TRUE(broadcast->has_sharding());
  ASSERT_EQ(broadcast->sharding(), expected_sharding);

  auto* scalar_constant = broadcast->operand(0);
  ASSERT_TRUE(scalar_constant->has_sharding());
  ASSERT_EQ(scalar_constant->sharding(), expected_sharding);
}

TEST_F(PoplarAlgebraicSimplifierTest, ConstantNotToBroadcast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({3.14, 3.14, 4})));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_FALSE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
}

TEST_F(PoplarAlgebraicSimplifierTest, IotaToBroadcast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({0.0f, 1.0f, 2.0f})));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Iota()));
}

// Test that A - 0 is simplified to A
TEST_F(PoplarAlgebraicSimplifierTest, SubZero) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kSubtract, param0, zero));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSubtract);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that A - Const is canonicalized to A + (-Const).
TEST_F(PoplarAlgebraicSimplifierTest, SubConstCanonicalization) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kSubtract, param0, constant));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSubtract);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Add(m::Parameter(0),
                                      m::Negate(m::Op().Is(constant)))));
}

// Test that A - Broadcast(Const) is canonicalized to A + Broadcast(-Const).
TEST_F(PoplarAlgebraicSimplifierTest, SubBroadcastConstCanonicalization) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4] parameter(0)
      c = f32[] constant(0.125)
      b = f32[4] broadcast(c), dimensions={}
      ROOT sub = f32[4] subtract(p0, b)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Add(m::Parameter(0),
                        m::Broadcast(m::Negate(m::ConstantScalar(0.125))))));
}

// Test that (A/B)/C is simplified to A/(B*C).
TEST_F(PoplarAlgebraicSimplifierTest, LhsDivOfDiv) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r0f32, "param2"));
  HloInstruction* div = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, param0, param1));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, div, param2));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Divide(m::Divide(m::Parameter(0), m::Parameter(1)),
                                   m::Parameter(2))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Parameter(0),
                           m::Multiply(m::Parameter(1), m::Parameter(2)))));
}

// Test that A/(B/C) is simplified to (A*C)/B.
TEST_F(PoplarAlgebraicSimplifierTest, RhsDivOfDiv) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r0f32, "param2"));
  HloInstruction* div = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, param1, param2));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, param0, div));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Parameter(0),
                           m::Divide(m::Parameter(1), m::Parameter(2)))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Multiply(m::Parameter(0), m::Parameter(2)),
                           m::Parameter(1))));
}

// Test that (A/B)/(C/D) is simplified to (A*D)/(B*C).
TEST_F(PoplarAlgebraicSimplifierTest, DivOfDivAndDiv) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {42, 123});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r2f32, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r2f32, "param2"));
  HloInstruction* param3 = builder.AddInstruction(
      HloInstruction::CreateParameter(3, r2f32, "param3"));
  HloInstruction* div0 = builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kDivide, param0, param1));
  HloInstruction* div1 = builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kDivide, param2, param3));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kDivide, div0, div1));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Divide(m::Parameter(0), m::Parameter(1)),
                           m::Divide(m::Parameter(2), m::Parameter(3)))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Multiply(m::Parameter(0), m::Parameter(3)),
                           m::Multiply(m::Parameter(1), m::Parameter(2)))));
}

// Test that A/exp(B) is simplified to A*exp(-B).
TEST_F(PoplarAlgebraicSimplifierTest, DivOfExp) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* exp = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, param1));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, param0, exp));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Divide(m::Parameter(0), m::Exp(m::Parameter(1)))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0),
                                     m::Exp(m::Negate(m::Parameter(1))))));
}

// Test that A/pow(B,C) is simplified to A*pow(B,-C).
TEST_F(PoplarAlgebraicSimplifierTest, DivOfPower) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r0f32, "param2"));
  HloInstruction* power = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, param1, param2));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, param0, power));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Parameter(0),
                           m::Power(m::Parameter(1), m::Parameter(2)))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(
                  m::Parameter(0),
                  m::Power(m::Parameter(1), m::Negate(m::Parameter(2))))));
}

// Test that broadcasting is done on the right step when simplifying A/pow(B,C)
// to A*pow(B,-C).
TEST_F(PoplarAlgebraicSimplifierTest, DivOfBroadcastingPower) {
  auto m = CreateNewVerifiedModule();
  Shape r1f32 = ShapeUtil::MakeShape(F32, {7});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r1f32, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r1f32, "param2"));
  HloInstruction* power = builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kPower, param1, param2));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kDivide, param0, power));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Parameter(0),
                           m::Power(m::Parameter(1), m::Parameter(2)))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  ASSERT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(
                  m::Parameter(0),
                  m::Power(m::Parameter(1), m::Negate(m::Parameter(2))))));
}

// A / Const => A * InvertedConst
TEST_F(PoplarAlgebraicSimplifierTest, DivideByConstant) {
  auto m = CreateNewVerifiedModule();
  Shape r1f32 = ShapeUtil::MakeShape(F32, {3});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* constant =
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR1<float>({1.f, 2.f, 3.f})));
  builder.AddInstruction(HloInstruction::CreateBinary(r1f32, HloOpcode::kDivide,
                                                      param0, constant));

  auto computation = m->AddEntryComputation(builder.Build());

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0), m::Constant())));
}

// A / Broadcast(Const) => A * Broadcast(InvertedConst)
TEST_F(PoplarAlgebraicSimplifierTest, DivideByBroadcastedConstant) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = f32[4] parameter(0)
      c = f32[] constant(256.0)
      b = f32[4] broadcast(c), dimensions={}
      ROOT d = f32[4] divide(p, b)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());

  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Multiply(
                  m::Parameter(0),
                  m::Broadcast(m::Op().IsConstantScalar(1.0f / 256.0f)))));
}

// A / Broadcast(B) => A * Broadcast(InvertedB)
TEST_F(PoplarAlgebraicSimplifierTest, FastMathDivideByBroadcastedScalarF16) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f16[4] parameter(0)
      p1 = f16[] parameter(1)
      b = f16[4] broadcast(p1), dimensions={}
      ROOT d = f16[4] divide(p0, b)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));

  IpuOptions_IpuAlgebraicSimplifierConfig config;
  config.set_enable_fast_math(false);
  ASSERT_FALSE(PoplarAlgebraicSimplifier(config).Run(m.get()).ValueOrDie());
  config.set_enable_fast_math(true);
  ASSERT_TRUE(PoplarAlgebraicSimplifier(config).Run(m.get()).ValueOrDie());

  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Multiply(
          m::Parameter(0), m::Broadcast(m::Divide(m::Op().IsConstantScalar(1.f),
                                                  m::Parameter(1))))));
}

// pow(pow(A, X), Y) => pow(A, X*Y)
TEST_F(PoplarAlgebraicSimplifierTest, PowerOfPower) {
  auto m = CreateNewVerifiedModule();
  Shape r1f32 = ShapeUtil::MakeShape(F32, {7});
  HloComputation::Builder builder(TestName());
  HloInstruction* base = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r1f32, "param1"));
  HloInstruction* exp2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r1f32, "param2"));
  HloInstruction* inner_power = builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kPower, base, exp1));
  builder.AddInstruction(HloInstruction::CreateBinary(r1f32, HloOpcode::kPower,
                                                      inner_power, exp2));

  auto computation = m->AddEntryComputation(builder.Build());
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Power(m::Op().Is(base),
                          m::Multiply(m::Op().Is(exp1), m::Op().Is(exp2)))));
}

// Don't simplify pow(pow(A, X), Y) => pow(A, X*Y) if X and Y are complex
// numbers.
TEST_F(PoplarAlgebraicSimplifierTest, PowerOfPowerComplex) {
  auto m = CreateNewVerifiedModule();
  Shape r1c64 = ShapeUtil::MakeShape(C64, {7});
  HloComputation::Builder builder(TestName());
  HloInstruction* base = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1c64, "param0"));
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r1c64, "param1"));
  HloInstruction* exp2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r1c64, "param2"));
  HloInstruction* inner_power = builder.AddInstruction(
      HloInstruction::CreateBinary(r1c64, HloOpcode::kPower, base, exp1));
  builder.AddInstruction(HloInstruction::CreateBinary(r1c64, HloOpcode::kPower,
                                                      inner_power, exp2));

  m->AddEntryComputation(builder.Build());
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_FALSE(simplifier.Run(m.get()).ValueOrDie());
}

// Test that A/1 is simplified to A for a scalar.
TEST_F(PoplarAlgebraicSimplifierTest, DivOneScalar) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  HloInstruction* div = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, param0, one));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, div);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that A/1 is simplified to A for an array.
TEST_F(PoplarAlgebraicSimplifierTest, DivOneArray) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {2, 2});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* one = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 1.0}, {1.0, 1.0}})));
  HloInstruction* div = builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kDivide, param0, one));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, div);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that complex(real(c), imag(c)) is simplified to c.
TEST_F(PoplarAlgebraicSimplifierTest, ComplexOfRealImagC) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {2, 2});
  Shape r2c64 = ShapeUtil::MakeShape(C64, {2, 2});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2c64, "param0"));
  HloInstruction* real = builder.AddInstruction(
      HloInstruction::CreateUnary(r2f32, HloOpcode::kReal, param0));
  HloInstruction* imag = builder.AddInstruction(
      HloInstruction::CreateUnary(r2f32, HloOpcode::kImag, param0));
  HloInstruction* cplx = builder.AddInstruction(
      HloInstruction::CreateBinary(r2c64, HloOpcode::kComplex, real, imag));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, cplx);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that real(complex(r,i)) is simplified to r.
TEST_F(PoplarAlgebraicSimplifierTest, RealOfComplex) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {2, 2});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r2f32, "param1"));
  HloInstruction* cplx = builder.AddInstruction(
      HloInstruction::CreateBinary(ShapeUtil::ChangeElementType(r2f32, C64),
                                   HloOpcode::kComplex, param0, param1));
  HloInstruction* real = builder.AddInstruction(
      HloInstruction::CreateUnary(r2f32, HloOpcode::kReal, cplx));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, real);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that imag(complex(r,i)) is simplified to i.
TEST_F(PoplarAlgebraicSimplifierTest, ImagOfComplex) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {2, 2});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r2f32, "param1"));
  HloInstruction* cplx = builder.AddInstruction(
      HloInstruction::CreateBinary(ShapeUtil::ChangeElementType(r2f32, C64),
                                   HloOpcode::kComplex, param0, param1));
  HloInstruction* imag = builder.AddInstruction(
      HloInstruction::CreateUnary(r2f32, HloOpcode::kImag, cplx));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, imag);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param1);
}

// Test that get_element(make_tuple({A,B}),1) is simplified to B
TEST_F(PoplarAlgebraicSimplifierTest, SelectMakeTuple) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r0f32, "param2"));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({param0, param1}));
  HloInstruction* get = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(r0f32, tuple, 1));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, get, param2));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, add);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Add(m::Parameter(1), m::Parameter(2))));
}

// Test that exp(A)/exp(B) is simplified to exp(A-B)
TEST_F(PoplarAlgebraicSimplifierTest, ExpDiv) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* exp0 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, param0));
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, param1));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, exp0, exp1));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Exp(m::Parameter(0)), m::Exp(m::Parameter(1)))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Exp(m::Subtract(m::Parameter(0), m::Parameter(1)))));
}

// Test that exp(A)*exp(B) is simplified to exp(A+B)
TEST_F(PoplarAlgebraicSimplifierTest, ExpMul) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* exp0 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, param0));
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, param1));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kMultiply, exp0, exp1));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Exp(m::Parameter(0)),
                                     m::Exp(m::Parameter(1)))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Exp(m::Add(m::Parameter(0), m::Parameter(1)))));
}

// Test that pow(exp(A), B) is simplified to exp(A*B)
TEST_F(PoplarAlgebraicSimplifierTest, PowExp) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* exp0 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, param0));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, exp0, param1));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Exp(m::Parameter(0)), m::Parameter(1))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Exp(m::Multiply(m::Parameter(0), m::Parameter(1)))));
}

// Test that ln(pow(A, B)) is simplified to ln(abs(A))*B
TEST_F(PoplarAlgebraicSimplifierTest, LnPow) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* pow = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, param0, param1));
  builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kLog, pow));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Log(m::Power(m::Parameter(0), m::Parameter(1)))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Log(m::Abs(m::Parameter(0))),
                                     m::Parameter(1))));
}

// Test that ln(exp(A)) is simplified to A
TEST_F(PoplarAlgebraicSimplifierTest, LnExp) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* exp0 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, param0));
  builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kLog, exp0));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Log(m::Exp(m::Parameter(0)))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_EQ(computation->root_instruction(), param0);
}

// Test that ln(exp(A)/exp(B)) is simplified to A-B
TEST_F(PoplarAlgebraicSimplifierTest, LnExpDiv) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* exp0 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, param0));
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, param1));
  HloInstruction* div = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, exp0, exp1));
  builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kLog, div));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Log(m::Divide(m::Exp(m::Parameter(0)),
                                          m::Exp(m::Parameter(1))))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Subtract(m::Parameter(0), m::Parameter(1))));
}

// Test that pow(A, 0) where A is a scalar is simplified to the scalar
// constant 1.
TEST_F(PoplarAlgebraicSimplifierTest, Pow0Scalar) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, param0, zero));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(zero))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
  EXPECT_EQ(root->literal().GetFirstElement<float>(), 1);
}

// Test that pow(A, 0) where A is not a scalar is simplified to broadcast(1).
TEST_F(PoplarAlgebraicSimplifierTest, Pow0Vector) {
  auto m = CreateNewVerifiedModule();
  Shape r1f32 = ShapeUtil::MakeShape(F32, {42});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kPower, param0, zero));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(zero))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast()));
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), r1f32))
      << ShapeUtil::HumanString(root->shape());
  EXPECT_EQ(root->dimensions().size(), 0);
  EXPECT_TRUE(ShapeUtil::IsScalar(root->operand(0)->shape()));
  EXPECT_EQ(root->operand(0)->literal().GetFirstElement<float>(), 1);
}

// Test that pow(A, 1) is simplified to A.
TEST_F(PoplarAlgebraicSimplifierTest, Pow1) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, param0, one));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(one))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_EQ(computation->root_instruction(), param0);
}

// Test that pow(A, 2) is simplified to A*A.
TEST_F(PoplarAlgebraicSimplifierTest, Pow2) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* two = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, param0, two));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(two))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0), m::Parameter(0))));
}

// Test that pow(A, -2) is simplified to 1/(A*A).
TEST_F(PoplarAlgebraicSimplifierTest, PowNegative2) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* negative_two = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(-2)));
  builder.AddInstruction(HloInstruction::CreateBinary(r0f32, HloOpcode::kPower,
                                                      param0, negative_two));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(negative_two))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Divide(m::Broadcast(), m::Multiply())));
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(root->operand(0)->operand(0)->literal().GetFirstElement<float>(),
            1);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kMultiply);
  EXPECT_EQ(root->operand(1)->operand(0), param0);
  EXPECT_EQ(root->operand(1)->operand(1), param0);
}

// Test that pow(A, -1) is simplified to 1/A.
TEST_F(PoplarAlgebraicSimplifierTest, PowNegative1) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* negative_one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(-1)));
  builder.AddInstruction(HloInstruction::CreateBinary(r0f32, HloOpcode::kPower,
                                                      param0, negative_one));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(negative_one))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Divide(m::Broadcast(), m::Parameter(0))));
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(root->operand(0)->operand(0)->literal().GetFirstElement<float>(),
            1);
}

TEST_F(PoplarAlgebraicSimplifierTest, PowNoughtPointFive) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* exponent = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.5)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, param0, exponent));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(exponent))));

  IpuOptions_IpuAlgebraicSimplifierConfig config;
  config.set_enable_fast_math(true);
  PoplarAlgebraicSimplifier simplifier(config);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Sqrt(m::Parameter(0))));
}

// Test that pow(A, -0.5) is simplified to 1/sqrt(A).
TEST_F(PoplarAlgebraicSimplifierTest, PowNegativeNoughtPointFive) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* negative_half = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(-0.5)));
  builder.AddInstruction(HloInstruction::CreateBinary(r0f32, HloOpcode::kPower,
                                                      param0, negative_half));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(negative_half))));

  IpuOptions_IpuAlgebraicSimplifierConfig config;
  config.set_enable_fast_math(true);
  PoplarAlgebraicSimplifier simplifier(config);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Rsqrt()));
  EXPECT_THAT(root->operand(0), param0);
}

TEST_F(PoplarAlgebraicSimplifierTest, ZeroSizedConvolution) {
  auto m = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {3, 3, 0}), "lhs"));

  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {3, 0, 3}), "rhs"));

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.set_input_feature_dimension(2);

  dnums.set_output_batch_dimension(0);
  dnums.add_output_spatial_dimensions(1);
  dnums.set_output_feature_dimension(2);

  dnums.add_kernel_spatial_dimensions(0);
  dnums.set_kernel_input_feature_dimension(1);
  dnums.set_kernel_output_feature_dimension(2);
  Window window;
  WindowDimension* dim = window.add_dimensions();
  dim->set_size(3);
  dim->set_padding_low(0);
  dim->set_padding_high(0);
  dim->set_stride(1);
  dim->set_window_dilation(1);
  dim->set_base_dilation(1);
  dim->set_window_reversal(false);
  // Create add computation.
  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {3, 3, 3}), lhs, rhs, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, DefaultPrecisionConfig(2)));
  m->AddEntryComputation(builder.Build());
  HloPassFix<PoplarAlgebraicSimplifier> simplifier;
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Convolution(m::Op().Is(lhs), m::Op().Is(rhs))));
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::Constant())));
}

TEST_F(PoplarAlgebraicSimplifierTest, ZeroSizedReduceWindow) {
  auto m = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {3, 0}), "op"));
  Window window;
  for (int64_t i = 0; i < 2; ++i) {
    WindowDimension* dim = window.add_dimensions();
    dim->set_size(1);
    dim->set_padding_low(1);
    dim->set_padding_high(1);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }
  // Create add computation.
  HloComputation* add_computation = nullptr;
  {
    HloComputation::Builder builder(TestName() + ".add");
    const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
    HloInstruction* p0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    HloInstruction* p1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "p1"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd, p0, p1));
    add_computation = m->AddEmbeddedComputation(builder.Build());
  }
  builder.AddInstruction(HloInstruction::CreateReduceWindow(
      ShapeUtil::MakeShape(F32, {5, 2}), param,
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f))),
      window, add_computation));
  m->AddEntryComputation(builder.Build());
  HloPassFix<PoplarAlgebraicSimplifier> simplifier;
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceWindow(m::Parameter(0), m::Constant())));
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::Constant())));
}

TEST_F(PoplarAlgebraicSimplifierTest, ZeroSizedVariadicReduceWindow) {
  const char* const hlo_string = R"(
HloModule ZeroSizedVariadicReduceWindow

ZeroSizedVariadicReduceWindow.add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  p2 = f32[] parameter(2)
  p3 = f32[] parameter(3)
  add.0 = f32[] add(p0, p1)
  add.1 = f32[] add(p2, p3)
  ROOT r = tuple(add.0, add.1)
}

ENTRY ZeroSizedReduceWindow {
  op = f32[3,0] parameter(0)
  constant = f32[] constant(0)
  ROOT reduce-window = (f32[5,2], f32[5,2]) reduce-window(op, op, constant, constant), window={size=1x1 pad=1_1x1_1}, to_apply=ZeroSizedVariadicReduceWindow.add
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<PoplarAlgebraicSimplifier> simplifier;
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceWindow(m::Parameter(0), m::Parameter(0),
                                         m::Constant(), m::Constant())));
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Broadcast(m::Constant()),
                                  m::Broadcast(m::Constant()))));
}

TEST_F(PoplarAlgebraicSimplifierTest, ZeroSizedPad) {
  auto m = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {3, 0}), "op"));
  PaddingConfig padding;
  for (int i = 0; i < 2; ++i) {
    PaddingConfig::PaddingConfigDimension* dimension = padding.add_dimensions();
    dimension->set_edge_padding_low(1);
    dimension->set_edge_padding_high(1);
    dimension->set_interior_padding(0);
  }
  builder.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(F32, {5, 2}), param,
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0(0.0f))),
      padding));
  m->AddEntryComputation(builder.Build());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Pad(m::Parameter(0), m::Constant())));
  HloPassFix<PoplarAlgebraicSimplifier> simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::Constant())));
}

TEST_F(PoplarAlgebraicSimplifierTest, ReshapeBroadcast) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});

  auto builder = HloComputation::Builder(TestName());
  auto op = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {3, 2}), "op"));
  auto reshape1 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {6}), op));
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {1, 6}), reshape1, {1}));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {3, 2}), broadcast));

  auto computation = builder.Build();
  m->AddEntryComputation(std::move(computation));

  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Reshape(m::Op().Is(op))))));

  HloPassFix<PoplarAlgebraicSimplifier> simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(m->entry_computation()->root_instruction(), op);
}

// Test that convert(A, $TYPE) is simplified to A if A is of type $TYPE.
TEST_F(PoplarAlgebraicSimplifierTest, ConvertBetweenSameType) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(
      HloInstruction::CreateConvert(ShapeUtil::MakeShape(F32, {}), input));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Convert(m::Op().Is(input))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), input);
}

// Test that copies are removed.
TEST_F(PoplarAlgebraicSimplifierTest, RemoveCopy) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  builder.AddInstruction(
      HloInstruction::CreateUnary(param0->shape(), HloOpcode::kCopy, param0));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Parameter(0))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), param0);
}

// Test that unary concatenates are removed.
TEST_F(PoplarAlgebraicSimplifierTest, RemoveUnaryConcatenate) {
  auto m = CreateNewVerifiedModule();
  Shape r1f32 = ShapeUtil::MakeShape(F32, {100});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  builder.AddInstruction(
      HloInstruction::CreateConcatenate(param0->shape(), {param0}, 0));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Concatenate(m::Parameter(0))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), param0);
}

// Test that empty operands of concatenates are removed.
TEST_F(PoplarAlgebraicSimplifierTest, RemoveEmptyConcatenateOperands) {
  auto m = CreateNewVerifiedModule();
  const int kParamLength = 100;
  Shape r1f32 = ShapeUtil::MakeShape(F32, {kParamLength});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r1f32, "param1"));
  HloInstruction* empty_literal = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({})));
  HloInstruction* empty_slice =
      builder.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {0}), param1, {42}, {42}, {1}));
  Shape result_shape = ShapeUtil::MakeShape(F32, {3 * kParamLength});
  builder.AddInstruction(HloInstruction::CreateConcatenate(
      result_shape, {empty_literal, param0, param0, empty_slice, param1}, 0));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Concatenate(
                  m::Op().Is(empty_literal), m::Parameter(0), m::Parameter(0),
                  m::Op().Is(empty_slice), m::Parameter(1))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Concatenate(m::Parameter(0), m::Parameter(0),
                                        m::Parameter(1))));
}

// Test that reduce of concat is simplified.
TEST_F(PoplarAlgebraicSimplifierTest, SimplifyReduceOfConcat) {
  auto m = CreateNewVerifiedModule();
  const int kParamLength = 100;
  Shape r3f32 =
      ShapeUtil::MakeShape(F32, {kParamLength, kParamLength, kParamLength});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r3f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r3f32, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r3f32, "param2"));
  Shape concat_shape =
      ShapeUtil::MakeShape(F32, {kParamLength, 3 * kParamLength, kParamLength});
  HloInstruction* Concatenate =
      builder.AddInstruction(HloInstruction::CreateConcatenate(
          concat_shape, {param0, param1, param2}, 1));
  HloComputation* add_computation = nullptr;
  {
    HloComputation::Builder builder(TestName() + ".add");
    const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
    HloInstruction* p0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    HloInstruction* p1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "p1"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd, p0, p1));
    add_computation = m->AddEmbeddedComputation(builder.Build());
  }
  Shape r4f32 = ShapeUtil::MakeShape(F32, {4, 5, 6, 7});
  Shape reduce_shape = ShapeUtil::MakeShape(F32, {kParamLength});

  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));
  builder.AddInstruction(HloInstruction::CreateReduce(
      reduce_shape, Concatenate, zero, {1, 2}, add_computation));

  auto computation = m->AddEntryComputation(builder.Build());

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Map(m::Map(m::Reduce(m::Parameter(0), m::Op().Is(zero)),
                               m::Reduce(m::Parameter(1), m::Op().Is(zero))),
                        m::Reduce(m::Parameter(2), m::Op().Is(zero)))));
}

// Test a concatenate with only empty operands is removed.
TEST_F(PoplarAlgebraicSimplifierTest, OnlyEmptyConcatenateOperands) {
  auto m = CreateNewVerifiedModule();
  const int kParamLength = 100;
  Shape r1f32 = ShapeUtil::MakeShape(F32, {kParamLength});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* empty_literal = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({})));
  HloInstruction* empty_slice =
      builder.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {0}), param0, {42}, {42}, {1}));
  Shape result_shape = ShapeUtil::MakeShape(F32, {0});
  builder.AddInstruction(HloInstruction::CreateConcatenate(
      result_shape, {empty_literal, empty_slice}, 0));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Concatenate(m::Op().Is(empty_literal),
                                        m::Op().Is(empty_slice))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_EQ(computation->root_instruction(), empty_literal);
}

// Test that concat with a scalar broadcast becomes a pad.
TEST_F(PoplarAlgebraicSimplifierTest, ConcatenateOfBroadcastBecomesPad) {
  auto m = CreateNewVerifiedModule();
  Shape r1f32 = ShapeUtil::MakeShape(F32, {100});
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(r1f32, param1, {}));
  builder.AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeShape(F32, {200}), {broadcast, param0}, 0));

  auto computation = m->AddEntryComputation(builder.Build());

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Pad(m::Parameter(0), m::Parameter(1))));
}

TEST_F(PoplarAlgebraicSimplifierTest, SimplifyConcatenateOfSlices) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {100, 99});
  Shape concat_shape = ShapeUtil::MakeShape(F32, {50, 80});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r2f32, "param1"));

  HloInstruction* slice0 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {50, 10}), param0, /*start_indices=*/{0, 0},
      /*limit_indices=*/{50, 10}, /*strides=*/{1, 1}));

  // Cannot merge 'slice0' and 'slice1' because of different start indices in
  // dimension 0.
  HloInstruction* slice1 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {50, 10}), param0, /*start_indices=*/{50, 10},
      /*limit_indices=*/{100, 20}, /*strides=*/{1, 1}));

  // Cannot merge 'slice1' and 'slice2' because of stride in dimension 2.
  HloInstruction* slice2 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {50, 10}), param0, /*start_indices=*/{50, 20},
      /*limit_indices=*/{100, 40}, /*strides=*/{1, 2}));

  // Cannot merge 'slice2' and 'slice3' because of stride in dimension 2.
  HloInstruction* slice3 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {50, 10}), param0, /*start_indices=*/{50, 40},
      /*limit_indices=*/{100, 50}, /*strides=*/{1, 1}));

  // Can merge 'slice3' and 'slice4'.
  HloInstruction* slice4 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {50, 10}), param0, /*start_indices=*/{50, 50},
      /*limit_indices=*/{100, 60}, /*strides=*/{1, 1}));

  // Can merge 'slice4' and 'slice5'.
  HloInstruction* slice5 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {50, 10}), param0, /*start_indices=*/{50, 60},
      /*limit_indices=*/{100, 70}, /*strides=*/{1, 1}));

  // Cannot merge 'slice5' and 'slice6' because of overlap.
  HloInstruction* slice6 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {50, 10}), param0, /*start_indices=*/{50, 69},
      /*limit_indices=*/{100, 79}, /*strides=*/{1, 1}));

  // Cannot merge 'slice6' and 'slice7' because of slicing from a different
  // parameter.
  HloInstruction* slice7 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {50, 10}), param1, /*start_indices=*/{50, 79},
      /*limit_indices=*/{100, 89}, /*strides=*/{1, 1}));

  builder.AddInstruction(HloInstruction::CreateConcatenate(
      concat_shape,
      {slice0, slice1, slice2, slice3, slice4, slice5, slice6, slice7}, 1));
  auto computation = m->AddEntryComputation(builder.Build());

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  auto s = m::Slice(m::Parameter(0));
  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Concatenate(s, s, s, s, s, m::Slice(m::Parameter(1)))));
  // The operand 3 should be a merge of 'slice3', 'slice4' and 'slice5', so its
  // shape should have dimensions {50, 30}.
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->operand(3)->shape(),
                       ShapeUtil::MakeShape(F32, {50, 30})));
  EXPECT_EQ(computation->root_instruction()->operand(3)->slice_starts(1), 40);
}

TEST_F(PoplarAlgebraicSimplifierTest, SimplifyConcatenateOfSameSlices_1_2_2_1) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {3, 4});
  Shape concat_shape = ShapeUtil::MakeShape(F32, {3, 6});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));

  HloInstruction* slice0 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {3, 1}), param0, /*start_indices=*/{0, 0},
      /*limit_indices=*/{3, 1}, /*strides=*/{1, 1}));

  HloInstruction* slice1 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {3, 1}), param0, /*start_indices=*/{0, 1},
      /*limit_indices=*/{3, 2}, /*strides=*/{1, 1}));

  HloInstruction* slice2 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {3, 1}), param0, /*start_indices=*/{0, 2},
      /*limit_indices=*/{3, 3}, /*strides=*/{1, 1}));

  HloInstruction* slice3 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {3, 1}), param0, /*start_indices=*/{0, 3},
      /*limit_indices=*/{3, 4}, /*strides=*/{1, 1}));

  builder.AddInstruction(HloInstruction::CreateConcatenate(
      concat_shape, {slice0, slice1, slice1, slice2, slice2, slice3}, 1));
  auto computation = m->AddEntryComputation(builder.Build());

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  auto root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Concatenate(
                        m::Slice(m::Parameter(0)),
                        m::Reshape(m::Broadcast(m::Slice(m::Parameter(0)))),
                        m::Reshape(m::Broadcast(m::Slice(m::Parameter(0)))),
                        m::Slice(m::Parameter(0)))));
  // Check shape of broadcast
  EXPECT_TRUE(ShapeUtil::Equal(root->operand(1)->operand(0)->shape(),
                               ShapeUtil::MakeShape(F32, {3, 1, 2})));
  // check slices start/limit indices
  EXPECT_THAT(root->operand(0)->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(root->operand(0)->slice_limits(), ElementsAre(3, 1));
  EXPECT_THAT(root->operand(1)->operand(0)->operand(0)->slice_starts(),
              ElementsAre(0, 1));
  EXPECT_THAT(root->operand(1)->operand(0)->operand(0)->slice_limits(),
              ElementsAre(3, 2));
  EXPECT_THAT(root->operand(2)->operand(0)->operand(0)->slice_starts(),
              ElementsAre(0, 2));
  EXPECT_THAT(root->operand(2)->operand(0)->operand(0)->slice_limits(),
              ElementsAre(3, 3));
  EXPECT_THAT(root->operand(3)->slice_starts(), ElementsAre(0, 3));
  EXPECT_THAT(root->operand(3)->slice_limits(), ElementsAre(3, 4));
}

TEST_F(PoplarAlgebraicSimplifierTest, SimplifyConcatenateOfSameSlices_5_5_5_5) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {1, 4});
  Shape concat_shape = ShapeUtil::MakeShape(F32, {1, 20});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));

  HloInstruction* slice0 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {1, 1}), param0, /*start_indices=*/{0, 0},
      /*limit_indices=*/{1, 1}, /*strides=*/{1, 1}));

  HloInstruction* slice1 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {1, 1}), param0, /*start_indices=*/{0, 1},
      /*limit_indices=*/{1, 2}, /*strides=*/{1, 1}));

  HloInstruction* slice2 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {1, 1}), param0, /*start_indices=*/{0, 2},
      /*limit_indices=*/{1, 3}, /*strides=*/{1, 1}));

  HloInstruction* slice3 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {1, 1}), param0, /*start_indices=*/{0, 3},
      /*limit_indices=*/{1, 4}, /*strides=*/{1, 1}));

  builder.AddInstruction(HloInstruction::CreateConcatenate(
      concat_shape, {slice0, slice0, slice0, slice0, slice0, slice1, slice1,
                     slice1, slice1, slice1, slice2, slice2, slice2, slice2,
                     slice2, slice3, slice3, slice3, slice3, slice3},
      1));
  auto computation = m->AddEntryComputation(builder.Build());

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  auto root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Concatenate(
                        m::Reshape(m::Broadcast(m::Slice(m::Parameter(0)))),
                        m::Reshape(m::Broadcast(m::Slice(m::Parameter(0)))),
                        m::Reshape(m::Broadcast(m::Slice(m::Parameter(0)))),
                        m::Reshape(m::Broadcast(m::Slice(m::Parameter(0)))))));
  EXPECT_TRUE(ShapeUtil::Equal(root->operand(0)->operand(0)->shape(),
                               ShapeUtil::MakeShape(F32, {1, 1, 5})));
  EXPECT_THAT(root->operand(0)->operand(0)->operand(0)->slice_starts(),
              ElementsAre(0, 0));
  EXPECT_THAT(root->operand(0)->operand(0)->operand(0)->slice_limits(),
              ElementsAre(1, 1));
  EXPECT_THAT(root->operand(1)->operand(0)->operand(0)->slice_starts(),
              ElementsAre(0, 1));
  EXPECT_THAT(root->operand(1)->operand(0)->operand(0)->slice_limits(),
              ElementsAre(1, 2));
  EXPECT_THAT(root->operand(2)->operand(0)->operand(0)->slice_starts(),
              ElementsAre(0, 2));
  EXPECT_THAT(root->operand(2)->operand(0)->operand(0)->slice_limits(),
              ElementsAre(1, 3));
  EXPECT_THAT(root->operand(3)->operand(0)->operand(0)->slice_starts(),
              ElementsAre(0, 3));
  EXPECT_THAT(root->operand(3)->operand(0)->operand(0)->slice_limits(),
              ElementsAre(1, 4));
}

TEST_F(PoplarAlgebraicSimplifierTest, SimplifyConcatenateOfSameSlices_1x5) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {1, 4});
  Shape concat_shape = ShapeUtil::MakeShape(F32, {1, 5});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));

  HloInstruction* slice0 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {1, 1}), param0, /*start_indices=*/{0, 0},
      /*limit_indices=*/{1, 1}, /*strides=*/{1, 1}));

  builder.AddInstruction(HloInstruction::CreateConcatenate(
      concat_shape, {slice0, slice0, slice0, slice0, slice0}, 1));
  auto computation = m->AddEntryComputation(builder.Build());

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  auto root = computation->root_instruction();

  // There shouldn't be the concatenate if the inputs to the concatenate are all
  // from the same slice
  EXPECT_THAT(root,
              GmockMatch(m::Reshape(m::Broadcast(m::Slice(m::Parameter(0))))));

  EXPECT_TRUE(ShapeUtil::Equal(root->operand(0)->shape(),
                               ShapeUtil::MakeShape(F32, {1, 1, 5})));
  EXPECT_THAT(root->operand(0)->operand(0)->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(root->operand(0)->operand(0)->slice_limits(), ElementsAre(1, 1));
}

// Test transforming reshapes and transposes of rng.
TEST_F(PoplarAlgebraicSimplifierTest, ReshapeOfTransposeOfRngToRng) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  HloInstruction* rng0 = builder.AddInstruction(
      HloInstruction::CreateRng(ShapeUtil::MakeShape(F32, {2, 2}),
                                RandomDistribution::RNG_UNIFORM, {zero, one}));

  HloInstruction* transpose = builder.AddInstruction(
      HloInstruction::CreateTranspose(rng0->shape(), rng0, {1, 0}));
  Shape reshape_shape = builder
                            .AddInstruction(HloInstruction::CreateReshape(
                                ShapeUtil::MakeShape(F32, {4}), transpose))
                            ->shape();

  auto computation = m->AddEntryComputation(builder.Build());

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  // Verify that reshape(transpose(rng)) is replace by a single rng of the
  // same shape as the reshape.
  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Rng()));
  EXPECT_TRUE(ShapeUtil::Equal(computation->root_instruction()->shape(),
                               reshape_shape));
}

// Regression test for a bug where if we failed to sink a reshape, we'd set the
// 'changed' bit in PoplarAlgebraicSimplifier to false.
TEST_F(PoplarAlgebraicSimplifierTest,
       FailureToSinkReshapeDoesntAffectChangedBit) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  // This add (param0 + 0) can be simplified.
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd,
      builder.AddInstruction(
          HloInstruction::CreateParameter(0, shape, "param0")),
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR2<float>({{0, 0}, {0, 0}})))));

  builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {4}), add));

  PoplarAlgebraicSimplifier simplifier;
  m->AddEntryComputation(builder.Build());
  EXPECT_TRUE(simplifier.Run(m.get()).ValueOrDie());
}

// Regression test for a bug where if we failed to sink a reshape, we'd set the
// 'changed' bit in PoplarAlgebraicSimplifier to false.
TEST_F(PoplarAlgebraicSimplifierTest,
       FailureToSinkBroadcastDoesntAffectChangedBit) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  // This add (param0 + 0) can be simplified.
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd,
      builder.AddInstruction(
          HloInstruction::CreateParameter(0, shape, "param0")),
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR2<float>({{0, 0}, {0, 0}})))));

  builder.AddInstruction(
      HloInstruction::CreateBroadcast(ShapeUtil::MakeShape(F32, {2, 2, 2}), add,
                                      /*broadcast_dimensions=*/{0, 1}));

  PoplarAlgebraicSimplifier simplifier;
  m->AddEntryComputation(builder.Build());
  EXPECT_TRUE(simplifier.Run(m.get()).ValueOrDie());
}

TEST_F(PoplarAlgebraicSimplifierTest, ReshapesMerged) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 2}), "param0"));

  HloInstruction* reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {2, 1, 2}), param0));

  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {1, 2, 1, 1, 2, 1}), reshape1));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Reshape(m::Parameter(0)))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Parameter(0))));
}

TEST_F(PoplarAlgebraicSimplifierTest, TransposesMerged) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 3, 4}), "param0"));

  HloInstruction* transpose1 =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {3, 4, 2}), param0, {1, 2, 0}));

  builder.AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {4, 3, 2}), transpose1, {1, 0, 2}));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Transpose(m::Op().Is(transpose1))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Transpose(m::Parameter(0))));
  EXPECT_EQ(std::vector<int64_t>({2, 1, 0}),
            computation->root_instruction()->dimensions());
}

TEST_F(PoplarAlgebraicSimplifierTest, TransposeIsReshape) {
  const char* hlo_string = R"(
  HloModule module

  ENTRY test {
    param = f32[10] parameter(0)
    reshaped = f32[1,1,10] reshape(f32[10] param)
    transposed = f32[10,1,1] transpose(f32[1,1,10] reshaped), dimensions={2,1,0}
    ROOT reshaped_again = f32[10] reshape(f32[10,1,1] transposed)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloPassFix<PoplarAlgebraicSimplifier> simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Parameter()));
}

// Test merging reshape and broadcast.
TEST_F(PoplarAlgebraicSimplifierTest, ReshapeAndBroadcastMerged) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {5}), "param0"));
  auto reshape1 = builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {1, 5, 1}), param0));
  builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {1, 2, 3, 5, 1}), reshape1, {0, 3, 2}));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Reshape(m::Parameter(0)))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Parameter(0))));
}

// Test merging broadcast and reshape.
TEST_F(PoplarAlgebraicSimplifierTest, BroadcastAndReshapeMerged) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {2, 3}), "param0"));
  auto broadcast1 = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {1, 2, 3, 7, 12, 1}), param0, {1, 2}));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {2, 3, 7, 2, 1, 3, 2}), broadcast1));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Parameter(0)))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Parameter(0))));
}

TEST_F(PoplarAlgebraicSimplifierTest, BroadcastAndReshape_1_3x1_3) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1}), "param"));
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 1}), param, {1}));
  builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {3}), broadcast));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Parameter(0)))));

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_FALSE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Parameter(0)))));
}

TEST_F(PoplarAlgebraicSimplifierTest, BroadcastAndReshape_4_3x2x4_6x1x1x4) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {4}), "param"));
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 2, 4}), param, {2}));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {6, 1, 1, 4}), broadcast));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Parameter(0)))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Parameter(0))));
  EXPECT_THAT(computation->root_instruction()->dimensions(),
              ::testing::ElementsAre(3));
}

TEST_F(PoplarAlgebraicSimplifierTest, BroadcastAndReshape_1_3x2x1_6x1x1x1) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1}), "param"));
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 2, 1}), param, {2}));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {6, 1, 1, 1}), broadcast));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Parameter(0)))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Parameter(0))));
  const std::vector<int64_t> broadcast_dims =
      computation->root_instruction()->dimensions();
  EXPECT_EQ(1, broadcast_dims.size());
  EXPECT_THAT(broadcast_dims[0], ::testing::AnyOf(1, 2, 3));
}

TEST_F(PoplarAlgebraicSimplifierTest, BroadcastAndReshape_4_3x2x4x2_6x8) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {4}), "param"));
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 2, 4, 2}), param, {2}));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {6, 8}), broadcast));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Parameter(0)))));

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_FALSE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Parameter(0)))));
}

TEST_F(PoplarAlgebraicSimplifierTest, IotaAndReshapeMerged) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto iota = builder.AddInstruction(HloInstruction::CreateIota(
      ShapeUtil::MakeShape(F32, {1, 2, 3, 7, 12, 1}), 2));
  Shape result_shape = ShapeUtil::MakeShape(F32, {2, 3, 7, 2, 1, 3, 2});
  builder.AddInstruction(HloInstruction::CreateReshape(result_shape, iota));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Iota()));
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), result_shape));
}

TEST_F(PoplarAlgebraicSimplifierTest, IotaAndReshapeToMixedRadix) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto iota = builder.AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(F32, {21}), 0));
  Shape result_shape = ShapeUtil::MakeShape(F32, {7, 3});
  builder.AddInstruction(HloInstruction::CreateReshape(result_shape, iota));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Add(
                  m::Iota(),
                  m::Multiply(m::Iota(), m::Broadcast(m::ConstantScalar())))));
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), result_shape));
}

TEST_F(PoplarAlgebraicSimplifierTest, IotaAndReshapeToMixedRadixExtraDims) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto iota = builder.AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(F32, {42, 24, 15}), 1));
  Shape result_shape = ShapeUtil::MakeShape(F32, {3, 14, 4, 3, 2, 5, 3});
  builder.AddInstruction(HloInstruction::CreateReshape(result_shape, iota));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Add(
          m::Add(m::Iota(),
                 m::Multiply(m::Iota(), m::Broadcast(m::ConstantScalar()))),
          m::Multiply(m::Iota(), m::Broadcast(m::ConstantScalar())))));
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), result_shape));
}

TEST_F(PoplarAlgebraicSimplifierTest, IotaEffectiveScalar) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto iota = builder.AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(F32, {1, 1}), 0));
  auto result_shape = iota->shape();

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Iota()));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  auto root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::Constant())));
  EXPECT_EQ(0.0f, root->operand(0)->literal().GetFirstElement<float>());
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), result_shape));
}

TEST_F(PoplarAlgebraicSimplifierTest, IotaAndReshape_1_3x2_6) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto iota = builder.AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(F32, {3, 2}), 1));
  builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {6}), iota));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_FALSE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));
}

TEST_F(PoplarAlgebraicSimplifierTest, IotaAndReshape_4_3x2x4_6x1x1x4) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto iota = builder.AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(F32, {3, 2, 4}), 2));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {6, 1, 1, 4}), iota));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Iota()));
  EXPECT_EQ(Cast<HloIotaInstruction>(computation->root_instruction())
                ->iota_dimension(),
            3);
}

TEST_F(PoplarAlgebraicSimplifierTest, IotaAndReshape_1_3x2x2_6x1x1x2) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto iota = builder.AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(F32, {3, 2, 2}), 2));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {6, 1, 1, 2}), iota));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Iota()));
  const int64_t iota_dim =
      Cast<HloIotaInstruction>(computation->root_instruction())
          ->iota_dimension();
  EXPECT_THAT(iota_dim, ::testing::AnyOf(1, 2, 3));
}

TEST_F(PoplarAlgebraicSimplifierTest, IotaAndReshape_4_3x2x4x2_6x8) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto iota = builder.AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(F32, {3, 2, 4, 2}), 2));
  builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {6, 8}), iota));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_FALSE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));
}

TEST_F(PoplarAlgebraicSimplifierTest, RemoveNoopPad) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 2}), "param"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  PaddingConfig no_padding;
  for (int i = 0; i < 2; ++i) {
    auto dimension = no_padding.add_dimensions();
    dimension->set_edge_padding_low(0);
    dimension->set_edge_padding_high(0);
    dimension->set_interior_padding(0);
  }
  builder.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(F32, {2, 2}), param, zero, no_padding));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Pad(m::Parameter(0), m::Op().Is(zero))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), param);
}

TEST_F(PoplarAlgebraicSimplifierTest, NegativePadding) {
  // Verify that a pad instruction with negative padding is replaced with a
  // pad with non-negative padding followed by a slice.
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {10, 10}), "param"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  PaddingConfig padding;
  int64_t low_padding[2] = {-1, -2};
  int64_t high_padding[2] = {2, -3};
  for (int i = 0; i < 2; ++i) {
    auto dimension = padding.add_dimensions();
    dimension->set_edge_padding_low(low_padding[i]);
    dimension->set_edge_padding_high(high_padding[i]);
    dimension->set_interior_padding(0);
  }
  HloInstruction* pad = builder.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(F32, {11, 5}), param, zero, padding));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  PoplarAlgebraicSimplifier simplifier;

  auto has_negative_padding = [](const HloInstruction* pad) {
    for (auto& padding_dimension : pad->padding_config().dimensions()) {
      if (padding_dimension.edge_padding_low() < 0 ||
          padding_dimension.edge_padding_high() < 0) {
        return true;
      }
    }
    return false;
  };

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Pad(m::Parameter(0), m::Op().Is(zero))));
  EXPECT_TRUE(has_negative_padding(pad));

  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Pad(m::Parameter(0), m::Op().Is(zero)))));
  EXPECT_FALSE(
      has_negative_padding(computation->root_instruction()->operand(0)));
}

TEST_F(PoplarAlgebraicSimplifierTest, TrivialInteriorPadding) {
  // Verify that a pad instruction with interior padding on one-sized
  // dimensions, removes the interior padding.
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 1}), "param"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  PaddingConfig padding;
  for (int i = 0; i < 2; ++i) {
    auto dimension = padding.add_dimensions();
    dimension->set_edge_padding_low(3);
    dimension->set_edge_padding_high(3);
    dimension->set_interior_padding(i * 3);
  }
  HloInstruction* pad = builder.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(F32, {8, 7}), param, zero, padding));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  PoplarAlgebraicSimplifier simplifier;

  ASSERT_THAT(computation->root_instruction(),
              GmockMatch(m::Pad(m::Parameter(0), m::Op().Is(zero))));
  ASSERT_TRUE(HasInteriorPadding(pad->padding_config()));

  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Pad(m::Parameter(0), m::Op().Is(zero))));
  EXPECT_FALSE(
      HasInteriorPadding(computation->root_instruction()->padding_config()));
}

TEST_F(PoplarAlgebraicSimplifierTest, RemoveNoopReshape) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 3}), "param"));
  builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {2, 3}), param));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Parameter(0))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), param);
}

TEST_F(PoplarAlgebraicSimplifierTest, RemoveNoopSlice) {
  HloComputation::Builder builder(TestName());
  const int64_t dim0 = 2;
  const int64_t dim1 = 3;
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {dim0, dim1}), "param"));
  builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {dim0, dim1}), param,
      /*start_indices=*/{0, 0},
      /*limit_indices=*/{dim0, dim1}, /*strides=*/{1, 1}));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Parameter(0))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), param);
}

TEST_F(PoplarAlgebraicSimplifierTest, SliceOfSliceToSlice) {
  HloComputation::Builder builder(TestName());
  const int64_t dim0 = 11;
  const int64_t dim1 = 12;
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {dim0, dim1}), "param"));
  HloInstruction* original_slice =
      builder.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {dim0 - 2, dim1 - 4}), param,
          /*start_indices=*/{1, 2},
          /*limit_indices=*/{dim0 - 1, dim1 - 2}, /*strides=*/{1, 1}));

  builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {dim0 - 5, dim1 - 9}), original_slice,
      /*start_indices=*/{2, 3},
      /*limit_indices=*/{dim0 - 3, dim1 - 6}, /*strides=*/{1, 1}));
  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Slice(m::Parameter(0)))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Parameter(0))));
  EXPECT_EQ(computation->root_instruction()->slice_starts(0), 3);
  EXPECT_EQ(computation->root_instruction()->slice_starts(1), 5);
  EXPECT_EQ(computation->root_instruction()->slice_limits(0), dim0 - 2);
  EXPECT_EQ(computation->root_instruction()->slice_limits(1), dim1 - 4);
}

TEST_F(PoplarAlgebraicSimplifierTest, SliceOfBroadcastToBroadcast) {
  HloComputation::Builder builder(TestName());
  const int64_t dim0 = 11;
  const int64_t dim1 = 12;
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {dim0}), "param"));
  HloInstruction* broadcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(
          ShapeUtil::MakeShape(F32, {dim0, dim1}), param, {0}));
  builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {dim0, dim1 - 9}), broadcast,
      /*start_indices=*/{0, 3},
      /*limit_indices=*/{dim0, dim1 - 6}, /*strides=*/{1, 1}));
  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Broadcast(m::Parameter(0)))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Parameter(0))));
}

TEST_F(PoplarAlgebraicSimplifierTest, SliceOfReshapeToReshapeOfSlice) {
  HloComputation::Builder builder(TestName());
  const int64_t dim0 = 11;
  const int64_t dim1 = 12;
  const int64_t dim2 = 13;
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {dim0 * dim1, dim2}), "param"));
  HloInstruction* original_reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {dim0, dim1, dim2}), param));

  builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {dim0 - 2, dim1, dim2}), original_reshape,
      /*start_indices=*/{0, 0, 0},
      /*limit_indices=*/{dim0 - 2, dim1, dim2}, /*strides=*/{1, 1, 1}));
  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Reshape(m::Parameter(0)))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Slice(m::Parameter(0)))));
}

TEST_F(PoplarAlgebraicSimplifierTest, SliceOfReshapeUnchanged) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1, 144, 25, 1, 512}), "param"));
  HloInstruction* original_reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {3600, 512}), param));

  builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {960, 512}), original_reshape,
      /*start_indices=*/{0, 0},
      /*limit_indices=*/{960, 512}, /*strides=*/{1, 1}));
  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Reshape(m::Parameter(0)))));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_FALSE(simplifier.Run(module.get()).ValueOrDie());
}

TEST_F(PoplarAlgebraicSimplifierTest, RemoveNoopSort) {
  auto builder = HloComputation::Builder(TestName());
  auto module = CreateNewVerifiedModule();

  Shape keys_shape = ShapeUtil::MakeShape(F32, {1});
  auto keys = builder.AddInstruction(
      HloInstruction::CreateParameter(0, keys_shape, "keys"));
  TF_ASSERT_OK(MakeSortHlo(keys_shape, {keys}, 0, /*is_stable=*/false, &builder,
                           module.get())
                   .status());
  HloComputation* computation = module->AddEntryComputation(builder.Build());
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), keys);
}

TEST_F(PoplarAlgebraicSimplifierTest,
       ReplaceEffectiveScalarKeyValueSortWithTuple) {
  auto builder = HloComputation::Builder(TestName());
  auto module = CreateNewVerifiedModule();

  Shape keys_shape = ShapeUtil::MakeShape(F32, {5, 0});
  Shape values_shape = ShapeUtil::MakeShape(S32, {5, 0});
  auto keys = builder.AddInstruction(
      HloInstruction::CreateParameter(0, keys_shape, "keys"));
  auto values0 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, values_shape, "values0"));
  auto values1 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, values_shape, "values1"));
  TF_ASSERT_OK(MakeSortHlo(ShapeUtil::MakeTupleShape(
                               {keys_shape, values_shape, values_shape}),
                           {keys, values0, values1}, 0, /*is_stable=*/false,
                           &builder, module.get())
                   .status());
  HloComputation* computation = module->AddEntryComputation(builder.Build());
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Tuple(m::Op().Is(keys), m::Op().Is(values0),
                                  m::Op().Is(values1))));
}

// Test that A && True is simplified to A
TEST_F(PoplarAlgebraicSimplifierTest, AndTrue) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_true = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  builder.AddInstruction(HloInstruction::CreateBinary(r0pred, HloOpcode::kAnd,
                                                      param0, const_true));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAnd);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that True && A is simplified to A
TEST_F(PoplarAlgebraicSimplifierTest, AndTrue2) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_true = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  builder.AddInstruction(HloInstruction::CreateBinary(r0pred, HloOpcode::kAnd,
                                                      const_true, param0));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAnd);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that A && False is simplified to False
TEST_F(PoplarAlgebraicSimplifierTest, AndFalse) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_false = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  builder.AddInstruction(HloInstruction::CreateBinary(r0pred, HloOpcode::kAnd,
                                                      param0, const_false));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAnd);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, const_false);
}

// Test that False && A is simplified to False
TEST_F(PoplarAlgebraicSimplifierTest, AndFalse2) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_false = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  builder.AddInstruction(HloInstruction::CreateBinary(r0pred, HloOpcode::kAnd,
                                                      const_false, param0));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAnd);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, const_false);
}

// Test that A || True is simplified to True
TEST_F(PoplarAlgebraicSimplifierTest, OrTrue) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_true = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0pred, HloOpcode::kOr, param0, const_true));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kOr);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, const_true);
}

// Test that True || A is simplified to True
TEST_F(PoplarAlgebraicSimplifierTest, OrTrue2) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_true = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0pred, HloOpcode::kOr, const_true, param0));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kOr);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, const_true);
}

// Test that A || False is simplified to A
TEST_F(PoplarAlgebraicSimplifierTest, OrFalse) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_false = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  builder.AddInstruction(HloInstruction::CreateBinary(r0pred, HloOpcode::kOr,
                                                      param0, const_false));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kOr);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that False || A is simplified to A
TEST_F(PoplarAlgebraicSimplifierTest, OrFalse2) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_false = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  builder.AddInstruction(HloInstruction::CreateBinary(r0pred, HloOpcode::kOr,
                                                      const_false, param0));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kOr);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Used for TEST_Ps that test merging (or not) of a kPad instruction into a
// convolution's Window.
struct ConvPaddingTestcase {
  ConvPaddingTestcase(absl::string_view padding,
                      absl::string_view orig_conv_window,
                      absl::string_view expected_conv_window)
      : ConvPaddingTestcase(padding, orig_conv_window, expected_conv_window,
                            /*pad_value=*/0) {}

  ConvPaddingTestcase(absl::string_view padding,
                      absl::string_view orig_conv_window,
                      absl::string_view expected_conv_window, float pad_value)
      : padding(padding),
        orig_conv_window(orig_conv_window),
        expected_conv_window(expected_conv_window),
        pad_value(pad_value) {}

  string ToString() const {
    return absl::StrFormat(
        "padding=%s, orig_conv_window=%s, expected_conv_window=%s, "
        "pad_value=%f",
        padding, orig_conv_window, expected_conv_window, pad_value);
  }

  string padding;
  string orig_conv_window;
  string expected_conv_window;
  float pad_value;
};

// ConvInputPaddingTest (and its one associated TEST_P testcase) checks that a
// computation that does
//
//   conv(pad(param0, padding=padding), param1), window=orig_conv_window
//
// gets transformed by PoplarAlgebraicSimplifier to
//
//   conv(param0, param1), window=expected_conv_window
//
// or, if expected_conv_window is the empty string, checks that
// PoplarAlgebraicSimplifier does *not* transform the original convolution.
class ConvInputPaddingTest
    : public PoplarAlgebraicSimplifierTest,
      public ::testing::WithParamInterface<ConvPaddingTestcase> {};

INSTANTIATE_TEST_SUITE_P(
    ConvInputPaddingTestCases, ConvInputPaddingTest,
    ::testing::ValuesIn(std::vector<ConvPaddingTestcase>{
        // Merge this edge padding into the conv.
        {"0_0x0_0x1_1x2_2", "", "pad=1_1x2_2"},
        // Merge this edge padding with the conv's edge padding.
        {"0_0x0_0x1_2x3_4", "pad=10_10x20_20", "pad=11_12x23_24"},
        // Merge this interior-padded kPad with the unpadded conv.  The 3x6
        // interior padding gets transformed to 4x7 conv lhs dilation.
        {"0_0x0_0x1_2_3x4_5_6", "", "pad=1_2x4_5 lhs_dilate=4x7"},
        // kPad has dilation on one dim, conv has it on the other; merge them.
        {"0_0x0_0x0_0_1x0_0_0", "lhs_dilate=1x10", "lhs_dilate=2x10"},
        // kPad has dilation and edge padding on one dim, conv has them on the
        // other; merge them.
        {"0_0x0_0x0_1_1x0_0_0", "pad=0_0x3_0 lhs_dilate=1x10",
         "pad=0_1x3_0 lhs_dilate=2x10"},

        // Don't transform if the pad value is nonzero.
        {"0_0x0_0x1_1x2_2", "", "", /*pad_value=*/1},

        // We refuse to transform the following because on some dimension, one
        // of the kPad and conv has dilation and the other has some sort of
        // padding.
        {"0_0x0_0x0_0_1x0_0", "pad=1_0x0_0", ""},
        {"0_0x0_0x0_0_1x0_0", "pad=0_1x0_0", ""},
        {"0_0x0_0x0_0_1x0_0", "lhs_dilate=2x1", ""},
        {"0_0x0_0x1_0_0x0_0", "lhs_dilate=2x1", ""},
        {"0_0x0_0x0_1_0x0_0", "lhs_dilate=2x1", ""},
        {"0_0x0_0x0_0_1x0_0", "lhs_dilate=2x1", ""},

        // We can't merge feature or batch padding into the conv.
        {"1_0x0_0x0_0x0_0", "", ""},
        {"0_0x1_0x0_0x0_0", "", ""},
    }));

TEST_P(ConvInputPaddingTest, DoTest) {
  ConvPaddingTestcase testcase = GetParam();

  // It would be better to put the testcase's ToString into the test name, but
  // gUnit has constraints on what can go into test names, and any reasonable
  // implementation of ToString() seems to violate them.
  SCOPED_TRACE(testcase.ToString());

  auto builder = HloComputation::Builder(TestName());
  auto* input = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1024, 128, 100, 100}),  // bf01
      "input"));
  auto* pad_value = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR0(testcase.pad_value)));

  PaddingConfig padding_config =
      ParsePaddingConfig(testcase.padding).ValueOrDie();
  auto* lhs_pad = builder.AddInstruction(HloInstruction::CreatePad(
      ShapeInference::InferPadShape(input->shape(), pad_value->shape(),
                                    padding_config)
          .ValueOrDie(),
      input, pad_value, padding_config));

  auto* filter = builder.AddInstruction(HloInstruction::CreateParameter(
      1,
      ShapeUtil::MakeShape(
          F32, {lhs_pad->shape().dimensions(1), 256, 3, 3}),  // io01
      "input"));

  ConvolutionDimensionNumbers dnums =
      ParseConvolutionDimensionNumbers("bf01_io01->bf01").ValueOrDie();
  Window window =
      ParseWindow(absl::StrCat("size=3x3 ", testcase.orig_conv_window))
          .ValueOrDie();
  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeInference::InferConvolveShape(lhs_pad->shape(), filter->shape(),
                                         /*feature_group_count=*/1,
                                         /*batch_group_count=*/1, window, dnums,
                                         absl::nullopt)
          .ValueOrDie(),
      lhs_pad, filter, /*feature_group_count=*/1, /*batch_group_count=*/1,
      window, dnums, DefaultPrecisionConfig(2)));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  PoplarAlgebraicSimplifier simplifier;
  if (testcase.expected_conv_window.empty()) {
    ASSERT_FALSE(simplifier.Run(module.get()).ValueOrDie());
  } else {
    ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
    auto* conv = module->entry_computation()->root_instruction();
    SCOPED_TRACE(module->ToString());
    ASSERT_THAT(conv,
                GmockMatch(m::Convolution(m::Parameter(), m::Parameter())));
    EXPECT_EQ(window_util::ToString(conv->window()),
              absl::StrCat("size=3x3 ", testcase.expected_conv_window));
  }
}

// ConvFilterPaddingTest (and its one associated TEST_P) checks that a
// computation that does
//
//   conv(param0, pad(param1, padding=padding)), window=orig_conv_window
//
// gets transformed by PoplarAlgebraicSimplifier to
//
//   conv(param0, param1), window=expected_conv_window
//
// or, if expected_conv_window is the empty string, checks that
// PoplarAlgebraicSimplifier does *not* transform the original convolution.
class ConvFilterPaddingTest
    : public PoplarAlgebraicSimplifierTest,
      public ::testing::WithParamInterface<ConvPaddingTestcase> {};

INSTANTIATE_TEST_SUITE_P(
    ConvFilterPaddingTestCases, ConvFilterPaddingTest,
    ::testing::ValuesIn(std::vector<ConvPaddingTestcase>{
        // Can only merge interior padding on the filter's spatial dimensions;
        // all
        // other paddings (edge padding and interior padding on the channel
        // dims)
        // should be rejected out of hand.
        {"1_0_0x0_0_0x0_0x0_0", "", ""},
        {"0_1_0x0_0_0x0_0x0_0", "", ""},
        {"0_0_1x0_0_0x0_0x0_0", "", ""},
        {"0_0_0x1_0_0x0_0x0_0", "", ""},
        {"0_0_0x0_1_0x0_0x0_0", "", ""},
        {"0_0_0x0_0_1x0_0x0_0", "", ""},
        {"0_0_0x0_0_0x1_0x0_0", "", ""},
        {"0_0_0x0_0_0x0_1x0_0", "", ""},
        {"0_0_0x0_0_0x0_0x1_0", "", ""},
        {"0_0_0x0_0_0x0_0x0_1", "", ""},

        // Interior padding on channel dims can be merged into the conv, so
        // long
        // as the conv and pad don't have interior padding on the same dim.
        {"0_0x0_0x0_0_5x0_0", "", "rhs_dilate=6x1"},
        {"0_0x0_0x0_0x0_0_10", "", "rhs_dilate=1x11"},
        {"0_0x0_0x0_0_10x0_0_100", "", "rhs_dilate=11x101"},
        {"0_0x0_0x0_0_1x0_0", "rhs_dilate=1x10", "rhs_dilate=2x10"},
        {"0_0x0_0x0_0x0_0_5", "rhs_dilate=10x1", "rhs_dilate=10x6"},

        // Can't merge if for a given dim there's interior padding on both the
        // pad and conv.
        {"0_0x0_0x0_0_1x0_0", "rhs_dilate=2x10", ""},
        {"0_0x0_0x0_0x0_0_5", "rhs_dilate=10x2", ""},

        // Don't transform if the pad value is nonzero.
        {"0_0x0_0x0_0_5x0_0", "", "", /*pad_value=*/1},
    }));

TEST_P(ConvFilterPaddingTest, DoIt) {
  ConvPaddingTestcase testcase = GetParam();

  // It would be better to put the testcase's ToString into the test name, but
  // gUnit has constraints on what can go into test names, and any reasonable
  // implementation of ToString() seems to violate them.
  SCOPED_TRACE(testcase.ToString());

  auto builder = HloComputation::Builder(TestName());
  auto* pad_value = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR0(testcase.pad_value)));
  auto* filter = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {128, 256, 3, 3}),  // io01
      "input"));
  PaddingConfig padding_config =
      ParsePaddingConfig(testcase.padding).ValueOrDie();
  auto* rhs_pad = builder.AddInstruction(HloInstruction::CreatePad(
      ShapeInference::InferPadShape(filter->shape(), pad_value->shape(),
                                    padding_config)
          .ValueOrDie(),
      filter, pad_value, padding_config));

  auto* input = builder.AddInstruction(HloInstruction::CreateParameter(
      0,
      ShapeUtil::MakeShape(
          F32, {1024, rhs_pad->shape().dimensions(0), 100, 100}),  // bf01
      "input"));

  ConvolutionDimensionNumbers dnums =
      ParseConvolutionDimensionNumbers("bf01_io01->bf01").ValueOrDie();
  Window window = ParseWindow(absl::StrFormat("size=%dx%d %s",
                                              rhs_pad->shape().dimensions(2),
                                              rhs_pad->shape().dimensions(3),
                                              testcase.orig_conv_window))
                      .ValueOrDie();

  // Add a PrecisionConfig and check that PoplarAlgebraicSimplifier keeps it
  // in place after the transformation.
  PrecisionConfig precision_config;
  precision_config.add_operand_precision(PrecisionConfig::HIGH);
  precision_config.add_operand_precision(PrecisionConfig::HIGHEST);

  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeInference::InferConvolveShape(input->shape(), rhs_pad->shape(),
                                         /*feature_group_count=*/1,
                                         /*batch_group_count=*/1, window, dnums,
                                         absl::nullopt)
          .ValueOrDie(),
      input, rhs_pad, /*feature_group_count=*/1, /*batch_group_count=*/1,
      window, dnums, precision_config));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  PoplarAlgebraicSimplifier simplifier;
  if (testcase.expected_conv_window.empty()) {
    ASSERT_FALSE(simplifier.Run(module.get()).ValueOrDie());
  } else {
    ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
    auto* conv = module->entry_computation()->root_instruction();
    SCOPED_TRACE(module->ToString());
    ASSERT_THAT(conv,
                GmockMatch(m::Convolution(m::Parameter(), m::Parameter())));
    EXPECT_EQ(window_util::ToString(conv->window()),
              absl::StrFormat("size=%dx%d %s",
                              conv->operand(1)->shape().dimensions(2),
                              conv->operand(1)->shape().dimensions(3),
                              testcase.expected_conv_window));
    EXPECT_THAT(Cast<HloConvolutionInstruction>(conv)
                    ->precision_config()
                    .operand_precision(),
                ElementsAre(PrecisionConfig::HIGH, PrecisionConfig::HIGHEST));
  }
}

// Test that slice(broadcast(/*scalar value*/)) simplifies to a single
// broadcast.
TEST_F(PoplarAlgebraicSimplifierTest, ScalarBroadcastToSlice) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* scalar_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "scalar_param"));

  Shape broadcast_shape = ShapeUtil::MakeShape(F32, {4, 5, 6, 7});
  HloInstruction* broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(broadcast_shape, scalar_param, {}));

  Shape slice_shape = ShapeUtil::MakeShape(F32, {2, 2, 3, 3});
  HloInstruction* slice = builder.AddInstruction(HloInstruction::CreateSlice(
      slice_shape, broadcast, {0, 1, 2, 3}, {2, 3, 5, 6}, {1, 1, 1, 1}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, slice);
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), slice_shape));

  PoplarAlgebraicSimplifier simplifier;

  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  // Running simplification again should not result in any further changes.
  ASSERT_FALSE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Op().Is(scalar_param))
                             .WithShapeEqualTo(&slice_shape)));
}

// Test that reshape(transpose(broadcast(/*scalar value*/))) simplifies to a
// single broadcast.
TEST_F(PoplarAlgebraicSimplifierTest, ScalarBroadcastToTransposeReshape) {
  HloComputation::Builder builder(TestName());
  HloInstruction* forty_two = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));

  Shape broadcast_shape = ShapeUtil::MakeShape(F32, {4, 5, 6});
  HloInstruction* broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(broadcast_shape, forty_two, {}));

  HloInstruction* transpose =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {6, 5, 4}), broadcast, {2, 1, 0}));

  Shape reshape_shape = ShapeUtil::MakeShape(F32, {30, 1, 4});
  HloInstruction* reshape = builder.AddInstruction(
      HloInstruction::CreateReshape(reshape_shape, transpose));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, reshape);
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), reshape_shape));

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Op().Is(forty_two))
                             .WithShapeEqualTo(&reshape_shape)));
}

// Test that ReduceWindow(Pad(op, x), y) can simplify to ReduceWindow(op, x).
TEST_F(PoplarAlgebraicSimplifierTest, FoldPadIntoReduceWindow) {
  // TODO(b/80488902): verify this module.
  auto module = CreateNewUnverifiedModule();
  HloComputation::Builder builder(TestName());

  // Create operand to the pad.
  HloInstruction* operand =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1, 2, 3, 4}), "p0"));

  // Create the pad.
  PaddingConfig padding = MakeNoPaddingConfig(4);
  padding.mutable_dimensions(1)->set_edge_padding_low(1);
  padding.mutable_dimensions(3)->set_edge_padding_high(2);

  HloInstruction* pad_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(5.0f)));
  HloInstruction* pad = builder.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(F32, {1, 3, 3, 5}), operand, pad_value, padding));

  // Create add computation.
  HloComputation* add_computation = nullptr;
  {
    HloComputation::Builder builder(TestName() + ".add");
    const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
    HloInstruction* p0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    HloInstruction* p1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "p1"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd, p0, p1));
    add_computation = module->AddEmbeddedComputation(builder.Build());
  }

  // Create the reduce-window.
  Window window;
  for (int64_t i = 0; i < pad->shape().rank(); ++i) {
    auto* dim = window.add_dimensions();
    dim->set_size(1);
    dim->set_padding_low(10);
    dim->set_padding_high(100);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }
  const Shape reduce_window_shape =
      ShapeUtil::MakeShape(F32, {111, 113, 113, 115});
  HloInstruction* reduce_init_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(5.0f)));
  HloInstruction* reduce_window =
      builder.AddInstruction(HloInstruction::CreateReduceWindow(
          reduce_window_shape, pad, reduce_init_value, window,
          add_computation));

  // Build the computation and run the simplifier.
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, reduce_window);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  // Running simplification again should not result in any further changes.
  ASSERT_FALSE(simplifier.Run(module.get()).ValueOrDie());

  // Verify the result
  root = computation->root_instruction();
  EXPECT_THAT(root,
              GmockMatch(m::ReduceWindow(m::Op().Is(operand), m::Constant())));
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), reduce_window_shape))
      << ShapeUtil::HumanString(root->shape()) << " vs "
      << ShapeUtil::HumanString(reduce_window_shape);
  EXPECT_EQ(root->window().dimensions(0).padding_low(), 10);
  EXPECT_EQ(root->window().dimensions(1).padding_low(), 11);
  EXPECT_EQ(root->window().dimensions(2).padding_low(), 10);
  EXPECT_EQ(root->window().dimensions(3).padding_low(), 10);
  EXPECT_EQ(root->window().dimensions(0).padding_high(), 100);
  EXPECT_EQ(root->window().dimensions(1).padding_high(), 100);
  EXPECT_EQ(root->window().dimensions(2).padding_high(), 100);
  EXPECT_EQ(root->window().dimensions(3).padding_high(), 102);
}

// Test that ReduceWindow(Convert(Pad(op, x)), y) can simplify to
// ReduceWindow(Convert(op), x).
TEST_F(PoplarAlgebraicSimplifierTest, FoldConvertedPadIntoReduceWindow) {
  // TODO(b/80488902): verify this module.
  auto module = CreateNewUnverifiedModule();
  HloComputation::Builder builder(TestName());

  // Create operand to the pad.
  HloInstruction* parameter =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(BF16, {1, 2, 3, 4}), "p0"));

  // Create the pad.
  PaddingConfig padding = MakeNoPaddingConfig(4);
  padding.mutable_dimensions(1)->set_edge_padding_low(1);
  padding.mutable_dimensions(3)->set_edge_padding_high(2);

  HloInstruction* pad_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(5.0f)));
  HloInstruction* pad = builder.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(BF16, {1, 3, 3, 5}), parameter, pad_value, padding));

  HloInstruction* convert =
      builder.AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(pad->shape(), F32), pad));

  // Create add computation.
  HloComputation* add_computation = nullptr;
  {
    HloComputation::Builder builder(TestName() + ".add");
    const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
    HloInstruction* p0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    HloInstruction* p1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "p1"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd, p0, p1));
    add_computation = module->AddEmbeddedComputation(builder.Build());
  }

  // Create the reduce-window.
  Window window;
  for (int64_t i = 0; i < pad->shape().rank(); ++i) {
    auto* dim = window.add_dimensions();
    dim->set_size(1);
    dim->set_padding_low(10);
    dim->set_padding_high(100);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }
  const Shape reduce_window_shape =
      ShapeUtil::MakeShape(F32, {111, 113, 113, 115});
  HloInstruction* reduce_init_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(5.0f)));
  HloInstruction* reduce_window =
      builder.AddInstruction(HloInstruction::CreateReduceWindow(
          reduce_window_shape, convert, reduce_init_value, window,
          add_computation));

  // Build the computation and run the simplifier.
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, reduce_window);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  // Running simplification again should not result in any further changes.
  ASSERT_FALSE(simplifier.Run(module.get()).ValueOrDie());

  // Verify the result
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::ReduceWindow(m::Convert(m::Parameter(0)),
                                               m::Constant())));
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), reduce_window_shape))
      << ShapeUtil::HumanString(root->shape()) << " vs "
      << ShapeUtil::HumanString(reduce_window_shape);
  EXPECT_EQ(root->window().dimensions(0).padding_low(), 10);
  EXPECT_EQ(root->window().dimensions(1).padding_low(), 11);
  EXPECT_EQ(root->window().dimensions(2).padding_low(), 10);
  EXPECT_EQ(root->window().dimensions(3).padding_low(), 10);
  EXPECT_EQ(root->window().dimensions(0).padding_high(), 100);
  EXPECT_EQ(root->window().dimensions(1).padding_high(), 100);
  EXPECT_EQ(root->window().dimensions(2).padding_high(), 100);
  EXPECT_EQ(root->window().dimensions(3).padding_high(), 102);
}

TEST_F(PoplarAlgebraicSimplifierTest, ReversalOfTrivialDimensionsToBitcast) {
  HloComputation::Builder builder(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {448, 2048, 1, 1});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  builder.AddInstruction(
      HloInstruction::CreateReverse(shape, a, /*dimensions=*/{2, 3}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(a, root);
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), shape));
}

TEST_F(PoplarAlgebraicSimplifierTest, IteratorInvalidation) {
  // Dots add computations to the parent module. Test that, when the
  // HloModule's computations are updated, then iterator invalidation doesn't
  // occur when running on subsequent computations.
  auto m = CreateNewVerifiedModule();
  Shape r1f32 = ShapeUtil::MakeShape(F32, {1});
  HloComputation::Builder builder(TestName() + ".Dot");
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r1f32, "x"));
  HloInstruction* y =
      builder.AddInstruction(HloInstruction::CreateParameter(1, r1f32, "y"));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(0);
  builder.AddInstruction(HloInstruction::CreateDot(r1f32, x, y, dot_dnums,
                                                   DefaultPrecisionConfig(2)));
  std::unique_ptr<HloComputation> dot_computation(builder.Build());

  HloComputation::Builder call_builder(TestName() + ".Call");
  HloInstruction* zero = call_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({0.0f})));
  HloInstruction* one = call_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({1.0f})));
  call_builder.AddInstruction(
      HloInstruction::CreateCall(r1f32, {zero, one}, dot_computation.get()));

  m->AddEmbeddedComputation(std::move(dot_computation));
  m->AddEntryComputation(call_builder.Build());
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
}

// Verify that when the LHS of a dot operation is zero the operation is
// skipped and the output is just 0 broadcast to the output shape.
TEST_F(PoplarAlgebraicSimplifierTest, TestZeroLHSInDot) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName() + ".DotZeroLHS");

  auto r2f32 = ShapeUtil::MakeShape(F32, {2, 2});

  auto zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto lhs =
      builder.AddInstruction(HloInstruction::CreateBroadcast(r2f32, zero, {}));

  auto rhs =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r2f32, "rhs"));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot_shape = ShapeUtil::MakeShape(F32, {2, 2});
  builder.AddInstruction(HloInstruction::CreateDot(
      dot_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  const auto computation = m->AddEntryComputation(builder.Build());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Dot(m::Broadcast(), m::Parameter(0))));

  IpuOptions_IpuAlgebraicSimplifierConfig config;
  config.set_enable_fast_math(true);
  PoplarAlgebraicSimplifier simplifier(config);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  const auto root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast()));
}

TEST_F(PoplarAlgebraicSimplifierTest, TestIdentityInDot) {
  const char* hlo_string = R"(
HloModule IdentityDot

ENTRY IdentityDot {
  p.0 = f32[100,100] parameter(0)
  iota.0 = s32[100,100] iota(), iota_dimension=0
  iota.1 = s32[100,100] iota(), iota_dimension=1
  c = pred[100,100] compare(iota.0, iota.1), direction=EQ
  i = f32[100,100] convert(c)
  ROOT dot = f32[100,100] dot(p.0, i), lhs_contracting_dims={1},rhs_contracting_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  IpuOptions_IpuAlgebraicSimplifierConfig config;
  PoplarAlgebraicSimplifier simplifier(config);
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  const auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Parameter()));
}

// Verify that when the RHS of a dot operation is zero the operation is
// skipped and the output is just 0 broadcast to the output shape.
TEST_F(PoplarAlgebraicSimplifierTest, TestZeroRHSInDot) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName() + ".DotZeroRHS");

  auto r2f32 = ShapeUtil::MakeShape(F32, {2, 2});

  auto lhs =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r2f32, "lhs"));

  auto zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto rhs =
      builder.AddInstruction(HloInstruction::CreateBroadcast(r2f32, zero, {}));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot_shape = ShapeUtil::MakeShape(F32, {2, 2});
  builder.AddInstruction(HloInstruction::CreateDot(
      dot_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  const auto computation = m->AddEntryComputation(builder.Build());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Dot(m::Parameter(0), m::Broadcast())));

  IpuOptions_IpuAlgebraicSimplifierConfig config;
  config.set_enable_fast_math(true);
  PoplarAlgebraicSimplifier simplifier(config);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  const auto root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast()));
}

// Test that a constant with tuple shape becomes a tuple of constants.
TEST_F(PoplarAlgebraicSimplifierTest, ConstantTupleBecomesTupleOfConstants) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  const float constant_scalar = 7.3f;
  std::initializer_list<float> constant_vector = {1.1f, 2.0f, 3.3f};
  Literal elements[] = {LiteralUtil::CreateR0<float>(constant_scalar),
                        LiteralUtil::CreateR1<float>(constant_vector)};
  Literal value = LiteralUtil::MakeTuple({&elements[0], &elements[1]});
  builder.AddInstruction(HloInstruction::CreateConstant(std::move(value)));

  auto computation = m->AddEntryComputation(builder.Build());

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Tuple(m::Constant(), m::Constant())));
}

// A dynamic-slice is trivial if its start indices are all zeroes and the size
// of its input equals the size of its output.  In this case, the dynamic
// slice is equal to its input.
TEST_F(PoplarAlgebraicSimplifierTest, TrivialDynamicSlice) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape shape = ShapeUtil::MakeShape(F32, {10, 100, 1000});
  std::vector<HloInstruction*> params;
  for (int i = 0; i < 3; ++i) {
    params.push_back(builder.AddInstruction(HloInstruction::CreateParameter(
        i + 1, ShapeUtil::MakeShape(U32, {}), "slice_indices")));
  }
  builder.AddInstruction(HloInstruction::CreateDynamicSlice(
      shape,
      builder.AddInstruction(
          HloInstruction::CreateParameter(0, shape, "slice_from")),
      params,
      /*slice_sizes=*/{10, 100, 1000}));

  auto computation = m->AddEntryComputation(builder.Build());
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Parameter()));
}

// A dynamic-update-slice is trivial if its start indices are all zeroes and
// the size of its "update" equals the size of its output.  In this case, the
// dynamic-update-slice is equal to its update.
TEST_F(PoplarAlgebraicSimplifierTest, TrivialDynamicUpdateSlice) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape full_shape = ShapeUtil::MakeShape(F32, {10, 100, 1000});
  Shape slice_shape = ShapeUtil::MakeShape(F32, {10, 1, 1000});

  std::vector<HloInstruction*> slice_indices, update_indices;
  for (int i = 0; i < 3; ++i) {
    slice_indices.push_back(
        builder.AddInstruction(HloInstruction::CreateParameter(
            i + 1, ShapeUtil::MakeShape(U32, {}), "slice_indices")));
    update_indices.push_back(
        builder.AddInstruction(HloInstruction::CreateParameter(
            i + 5, ShapeUtil::MakeShape(U32, {}), "update_indices")));
  }
  HloInstruction* slice =
      builder.AddInstruction(HloInstruction::CreateDynamicSlice(
          slice_shape,
          builder.AddInstruction(
              HloInstruction::CreateParameter(0, full_shape, "slice_from")),
          slice_indices,
          /*slice_sizes=*/{10, 1, 1000}));

  builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      slice_shape,
      builder.AddInstruction(
          HloInstruction::CreateParameter(4, slice_shape, "to_update")),
      slice, update_indices));

  auto computation = m->AddEntryComputation(builder.Build());
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::DynamicSlice(m::Parameter(), m::Parameter(),
                                         m::Parameter(), m::Parameter())));
}

// Test that dynamic-update-slice with a scalar broadcast becomes a pad.
TEST_F(PoplarAlgebraicSimplifierTest, DynamicUpdateSliceOfBroadcastToPad) {
  const char* hlo_string = R"(
HloModule AddBroadcastZeroWithDynamicSlice

ENTRY AddBroadcastZeroWithDynamicSlice {
  param0 = f32[1800,12,512]{2,1,0} parameter(0)
  constant = f32[] constant(0)
  broadcast = f32[1800,12,512]{2,1,0} broadcast(constant), dimensions={}
  param1 = f32[1,12,512]{2,1,0} parameter(1)
  constant.1 = s32[] constant(0)
  dynamic-update-slice = f32[1800,12,512]{2,1,0} dynamic-update-slice(broadcast, param1, constant.1, constant.1, constant.1)
  ROOT add = f32[1800,12,512]{2,1,0} add(param0, dynamic-update-slice)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  VLOG(2) << "Before rewrite dus->pad\n" << module->ToString();
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  VLOG(2) << "After rewrite dus->pad\n" << module->ToString();
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root->opcode(), HloOpcode::kAdd);
  EXPECT_THAT(root->operand(1)->opcode(), HloOpcode::kPad);
}

// Test that
//    add(buffer, dynamic-update-slice(zero, update))
//  is converted to:
//    dynamic-update-slice(add(dynamic-slice(buffer), update))
TEST_F(PoplarAlgebraicSimplifierTest, AddBroadcastZeroWithDynamicSlice) {
  auto m = CreateNewVerifiedModule();
  Shape full_shape = ShapeUtil::MakeShape(F32, {1800, 12, 512});
  Shape partial_shape = ShapeUtil::MakeShape(F32, {1, 12, 512});
  Shape index_shape = ShapeUtil::MakeShape(S32, {});

  HloComputation::Builder builder(TestName());
  HloInstruction* full_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, full_shape, "param0"));
  HloInstruction* partial_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, partial_shape, "param1"));
  HloInstruction* index = builder.AddInstruction(
      HloInstruction::CreateParameter(2, index_shape, "param2"));

  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));

  HloInstruction* bcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(full_shape, zero, {}));

  HloInstruction* dynamic_update_slice =
      builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
          full_shape, bcast, partial_param, {index, index, index}));

  builder.AddInstruction(HloInstruction::CreateBinary(
      full_shape, HloOpcode::kAdd, full_param, dynamic_update_slice));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root->opcode(), HloOpcode::kDynamicUpdateSlice);
  EXPECT_THAT(root->operand(0), full_param);
  EXPECT_THAT(root->operand(1), GmockMatch(m::Add(m::DynamicSlice(), m::Op())));
}

// Test that two consecutive broadcasts can be merged to one.
TEST_F(PoplarAlgebraicSimplifierTest, MergeBroadcasts) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  Shape r2f32 = ShapeUtil::MakeShape(F32, {2, 2});
  HloInstruction* input_array = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({3, 4})));
  HloInstruction* inner_bcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(r2f32, input_array, {1}));
  Shape r3f32 = ShapeUtil::MakeShape(F32, {2, 2, 2});
  builder.AddInstruction(
      HloInstruction::CreateBroadcast(r3f32, inner_bcast, {0, 2}));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBroadcast);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::Constant())));
  EXPECT_THAT(root->dimensions(), ElementsAre(2));
}

// Test that two consecutive broadcasts can be merged to one.
TEST_F(PoplarAlgebraicSimplifierTest, MergeBroadcasts2) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  Shape r2f32 = ShapeUtil::MakeShape(F32, {2, 3});
  Shape r3f32 = ShapeUtil::MakeShape(F32, {2, 5, 3});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  // The initial dimensions go to places 0 and 2 in the 3-dim array,
  // and to places 1 and 3 in the 4-dim array,
  HloInstruction* inner_bcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(r3f32, param0, {0, 2}));
  Shape r4f32 = ShapeUtil::MakeShape(F32, {4, 2, 5, 3});
  builder.AddInstruction(
      HloInstruction::CreateBroadcast(r4f32, inner_bcast, {1, 2, 3}));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBroadcast);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::Parameter(0))));
  EXPECT_THAT(root->dimensions(), ElementsAre(1, 3));
}

// Test that a broadcast of an iota can be merged to one iota.
TEST_F(PoplarAlgebraicSimplifierTest, MergeBroadcastAndIota) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  Shape r2f32 = ShapeUtil::MakeShape(F32, {2, 2});
  HloInstruction* iota =
      builder.AddInstruction(HloInstruction::CreateIota(r2f32, 1));
  Shape r3f32 = ShapeUtil::MakeShape(F32, {2, 2, 2});
  builder.AddInstruction(HloInstruction::CreateBroadcast(r3f32, iota, {0, 2}));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBroadcast);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Iota()));
  EXPECT_EQ(Cast<HloIotaInstruction>(root)->iota_dimension(), 2);
}

// Test that a broadcast of an iota can be merged to one iota.
TEST_F(PoplarAlgebraicSimplifierTest, MergeBroadcastAndIota2) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  Shape r3f32 = ShapeUtil::MakeShape(F32, {2, 5, 3});
  HloInstruction* iota =
      builder.AddInstruction(HloInstruction::CreateIota(r3f32, 1));
  Shape r4f32 = ShapeUtil::MakeShape(F32, {4, 2, 5, 3});
  builder.AddInstruction(
      HloInstruction::CreateBroadcast(r4f32, iota, {1, 2, 3}));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBroadcast);
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Iota()));
  EXPECT_EQ(Cast<HloIotaInstruction>(root)->iota_dimension(), 2);
}

TEST_F(PoplarAlgebraicSimplifierTest, TransposeOfDot) {
  const char* hlo_string = R"(
  HloModule module

  ENTRY test {
    lhs = f32[3,4,5] parameter(0)
    rhs = f32[6,3,4] parameter(1)
    dot = f32[5,6] dot(lhs,rhs), lhs_contracting_dims={0,1}, rhs_contracting_dims={1,2}
    ROOT transpose = f32[6,5] transpose(dot), dimensions={1,0}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Dot(m::Parameter(1), m::Parameter(0))));
}

TEST_F(PoplarAlgebraicSimplifierTest, SliceOfPadLow) {
  const char* hlo_string = R"(
  HloModule module

  ENTRY test {
    param = f32[3,4] parameter(0)
    constant = f32[] constant(0.0)
    pad = f32[8,10] pad(f32[3,4] param, f32[] constant), padding=3_2x1_5
    ROOT slice = f32[1,1] slice(f32[8,10] pad), slice={[2:3],[0:1]}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Reshape(m::Constant())));
}

TEST_F(PoplarAlgebraicSimplifierTest, SliceOfPadHigh) {
  const char* hlo_string = R"(
  HloModule module

  ENTRY test {
    param = f32[3,4] parameter(0)
    constant = f32[] constant(0.0)
    pad = f32[8,10] pad(f32[3,4] param, f32[] constant), padding=3_2x1_5
    ROOT slice = f32[1,1] slice(f32[8,10] pad), slice={[6:7],[9:10]}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Reshape(m::Constant())));
}

TEST_F(PoplarAlgebraicSimplifierTest, SliceOfPadMidNonScalar) {
  const char* hlo_string = R"(
  HloModule module

  ENTRY test {
    param = f32[3,4] parameter(0)
    constant = f32[] constant(0.0)
    pad = f32[8,10] pad(f32[3,4] param, f32[] constant), padding=3_2x1_5
    ROOT slice = f32[1,1] slice(f32[8,10] pad), slice={[5:6],[4:5]}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_FALSE(simplifier.Run(module.get()).ValueOrDie());
}

TEST_F(PoplarAlgebraicSimplifierTest, SliceOfPadMidScalarConstant) {
  const char* hlo_string = R"(
  HloModule module

  ENTRY test {
    param = f32[3,4] parameter(0)
    constant = f32[] constant(0.0)
    pad = f32[8,10] pad(f32[3,4] param, f32[] constant), padding=3_2x1_5
    ROOT slice = f32[1,1] slice(f32[8,10] pad), slice={[5:6],[9:10]}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Reshape(m::Constant())));
}

TEST_F(PoplarAlgebraicSimplifierTest, SliceOfPadMidScalar) {
  const char* hlo_string = R"(
  HloModule module

  ENTRY test {
    param = f32[1,1] parameter(0)
    constant = f32[] constant(0.0)
    pad = f32[8,10] pad(f32[1,1] param, f32[] constant), padding=3_4x4_5
    ROOT slice = f32[1,1] slice(f32[8,10] pad), slice={[3:4],[4:5]}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Parameter()));
}

TEST_F(PoplarAlgebraicSimplifierTest, SliceOfPadSomeDimsInPadding) {
  const char* hlo_string = R"(
  HloModule module

  ENTRY entry () -> f32[1]{0} {
    constant.val = f32[] constant(4)
    constant.pad = f32[] constant(-7)
    reshape.1 = f32[1,1,1]{2,1,0} reshape(f32[] constant.val)
    pad = f32[3,3,3]{2,1,0} pad(f32[1,1,1]{2,1,0} reshape.1, f32[] constant.pad), padding=0_2x0_2x2_0
    slice = f32[1,1,1]{2,1,0} slice(f32[3,3,3]{2,1,0} pad), slice={[0:1], [0:1], [0:1]}
    ROOT reshape.2 = f32[1]{0} reshape(f32[1,1,1]{2,1,0} slice)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Reshape(m::ConstantScalar(-7.0))));
}

TEST_F(PoplarAlgebraicSimplifierTest, SliceOfConcatScalarInput) {
  const char* hlo_string = R"(
  HloModule module

  ENTRY test {
    param.0 = f32[2] parameter(0)
    param.1 = f32[1] parameter(1)
    param.2 = f32[3] parameter(2)
    concat = f32[6] concatenate(param.0, param.1, param.2), dimensions={0}
    ROOT slice = f32[1] slice(concat), slice={[2:3]}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Parameter(1)));
}

TEST_F(PoplarAlgebraicSimplifierTest, SliceOfConcatNonScalarInput) {
  const char* hlo_string = R"(
  HloModule module

  ENTRY test {
    param.0 = f32[2] parameter(0)
    param.1 = f32[1] parameter(1)
    param.2 = f32[3] parameter(2)
    concat = f32[6] concatenate(param.0, param.1, param.2), dimensions={0}
    ROOT slice = f32[2] slice(concat), slice={[4:6]}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Slice(m::Parameter(2))));
  EXPECT_EQ(root->slice_starts(0), 1);
  EXPECT_EQ(root->slice_limits(0), 3);
}

TEST_F(PoplarAlgebraicSimplifierTest, NegateNegate) {
  const char* hlo_string = R"(
  HloModule module

  ENTRY test {
    param.0 = f32[2] parameter(0)
    neg.0 = f32[2] negate(param.0)
    ROOT neg.1 = f32[2] negate(neg.0)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Parameter(0)));
}

TEST_F(PoplarAlgebraicSimplifierTest, NotNot) {
  const char* hlo_string = R"(
  HloModule module

  ENTRY test {
    param.0 = pred[2] parameter(0)
    not.0 = pred[2] not(param.0)
    ROOT not.1 = pred[2] not(not.0)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Parameter(0)));
}

struct PadReduceWindowEffectiveBroadcastCase {
  std::vector<int64_t> input_spatials;
  std::vector<int64_t> symmetric_pad_spatials;
  std::vector<int64_t> reduce_window_spatials;
  // Whether to use `B F S0 S1` form vs `B S0 S1 F` form.
  //
  // This doesn't test any different functionality but is useful for making
  // sure kBroadcast nodes are well formed.
  bool prepend_a;
  bool should_become_broadcast;

  string ToTestCaseName() const {
    return absl::StrCat(absl::StrJoin(input_spatials, ","), ";",
                        absl::StrJoin(symmetric_pad_spatials, ","), ";",
                        absl::StrJoin(reduce_window_spatials, ","), ";",
                        prepend_a, ";", should_become_broadcast);
  }
};

void PrintTo(const PadReduceWindowEffectiveBroadcastCase& c, std::ostream* os) {
  *os << c.ToTestCaseName();
}

class PadReduceWindowEffectiveBroadcastTest
    : public PoplarAlgebraicSimplifierTest,
      public ::testing::WithParamInterface<
          PadReduceWindowEffectiveBroadcastCase> {};

TEST_P(PadReduceWindowEffectiveBroadcastTest, DoIt) {
  auto m = CreateNewVerifiedModule();
  const auto& param = GetParam();

  // a and b are parallel bounds we can either turn into a B F S0 S1 or
  // `B S0 S1 F` kind of pattern.
  auto decorate_spatials = [&param](absl::Span<const int64_t> spatials,
                                    int64_t a, int64_t b) {
    std::vector<int64_t> result;
    if (param.prepend_a) {
      result.push_back(a);
    }
    for (int64_t s : spatials) {
      result.push_back(s);
    }
    if (!param.prepend_a) {
      result.push_back(a);
    }
    result.push_back(b);
    return result;
  };

  HloComputation::Builder builder(TestName());
  const Shape input_shape = ShapeUtil::MakeShape(
      F32, decorate_spatials(param.input_spatials, 128, 2048));
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "input"));

  PaddingConfig padding = window_util::MakeSymmetricPadding(
      decorate_spatials(param.symmetric_pad_spatials, 0, 0));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape pad_shape,
      ShapeInference::InferPadShape(input->shape(),
                                    ShapeUtil::MakeShape(F32, {}), padding));
  HloInstruction* pad = builder.AddInstruction(HloInstruction::CreatePad(
      pad_shape, input,
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0(0.0f))),
      padding));

  HloComputation* add_computation = nullptr;
  {
    HloComputation::Builder builder(TestName() + ".add");
    const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
    HloInstruction* p0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    HloInstruction* p1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "p1"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd, p0, p1));
    add_computation = m->AddEmbeddedComputation(builder.Build());
  }

  Window window = window_util::MakeWindow(
      decorate_spatials(param.reduce_window_spatials, 1, 1));
  auto zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_shape,
                          ShapeInference::InferReduceWindowShape(
                              pad->shape(), zero->shape(), window,
                              add_computation->ComputeProgramShape()));
  builder.AddInstruction(HloInstruction::CreateReduceWindow(
      output_shape, pad, zero, window, add_computation));

  auto computation = m->AddEntryComputation(builder.Build());
  PoplarAlgebraicSimplifier simplifier;
  TF_ASSERT_OK_AND_ASSIGN(bool run_successful, simplifier.Run(m.get()));
  ASSERT_TRUE(run_successful);

  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), output_shape));

  if (param.should_become_broadcast) {
    EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Broadcast()));
  } else {
    EXPECT_THAT(computation->root_instruction(),
                GmockMatch(m::ReduceWindow(m::Op(), m::Op().Is(zero))));
  }
}

const std::vector<PadReduceWindowEffectiveBroadcastCase>&
PadReduceWindowEffectiveBroadcastCases() {
  static auto* cases = new std::vector<PadReduceWindowEffectiveBroadcastCase>{
      {/*input_spatials=*/{1, 1}, /*symmetric_pad_amount=*/{6, 6},
       /*reduce_window_spatials=*/{7, 7}, /*prepend_a=*/true,
       /*should_become_broadcast=*/true},  //
      {/*input_spatials=*/{1, 1}, /*symmetric_pad_amount=*/{6, 6},
       /*reduce_window_spatials=*/{7, 7}, /*prepend_a=*/false,
       /*should_become_broadcast=*/true},  //
      {/*input_spatials=*/{2, 2}, /*symmetric_pad_amount=*/{6, 6},
       /*reduce_window_spatials=*/{7, 7}, /*prepend_a=*/true,
       /*should_become_broadcast=*/false},  //
      {/*input_spatials=*/{1, 1}, /*symmetric_pad_amount=*/{2, 2},
       /*reduce_window_spatials=*/{1, 1}, /*prepend_a=*/true,
       /*should_become_broadcast=*/false},  //
      {/*input_spatials=*/{5, 1}, /*symmetric_pad_amount=*/{0, 2},
       /*reduce_window_spatials=*/{2, 5}, /*prepend_a=*/true,
       /*should_become_broadcast=*/false},  //
  };
  return *cases;
}

INSTANTIATE_TEST_SUITE_P(
    PadReduceWindowEffectiveBroadcastInstantiation,
    PadReduceWindowEffectiveBroadcastTest,
    ::testing::ValuesIn(PadReduceWindowEffectiveBroadcastCases()));

class BatchDotStrengthReductionTest
    : public PoplarAlgebraicSimplifierTest,
      public ::testing::WithParamInterface<
          ::testing::tuple<int, int, int, PrimitiveType>> {};

INSTANTIATE_TEST_SUITE_P(BatchDotStrengthReductionTestInstantiation,
                         BatchDotStrengthReductionTest,
                         ::testing::Combine(::testing::Values(-1, 1, 2),
                                            ::testing::Values(-1, 1, 2),
                                            ::testing::Values(-1, 1, 2),
                                            ::testing::Values(F32, BF16)));

class DotStrengthReductionTest
    : public PoplarAlgebraicSimplifierTest,
      public ::testing::WithParamInterface<
          ::testing::tuple<int, int, int, bool, bool, PrimitiveType>> {};

INSTANTIATE_TEST_SUITE_P(
    DotStrengthReductionTestInstantiation, DotStrengthReductionTest,
    ::testing::Combine(::testing::Values(1, 2), ::testing::Values(1, 2),
                       ::testing::Values(1, 2), ::testing::Bool(),
                       ::testing::Bool(), ::testing::Values(F32, BF16)));

struct DotOfConcatTestSpec {
  int64_t m;
  int64_t k;
  int64_t n;
};

class DotOfConcatSimplificationTest
    : public PoplarAlgebraicSimplifierTest,
      public ::testing::WithParamInterface<DotOfConcatTestSpec> {};

// Test that we transform
//  dot(const, concat(A, B, C))
// to
//  add(dot(const_0, A), dot(const_1, B),  dot(const_2, C))
TEST_P(DotOfConcatSimplificationTest, ConstantLHS) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  DotOfConcatTestSpec spec = GetParam();

  ASSERT_GE(spec.k, 3);

  int64_t k0 = spec.k / 3;
  int64_t k1 = spec.k / 3;
  int64_t k2 = spec.k - k0 - k1;

  Shape lhs_shape = ShapeUtil::MakeShape(F32, {spec.m, spec.k});
  auto* lhs = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/10.0, /*to=*/10000.0, /*rows=*/spec.m, /*cols=*/spec.k)));

  Shape rhs0_shape = ShapeUtil::MakeShape(F32, {k0, spec.n});
  Shape rhs1_shape = ShapeUtil::MakeShape(F32, {k1, spec.n});
  Shape rhs2_shape = ShapeUtil::MakeShape(F32, {k2, spec.n});

  HloInstruction* rhs0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, rhs0_shape, "rhs0"));
  HloInstruction* rhs1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, rhs1_shape, "rhs1"));
  HloInstruction* rhs2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, rhs2_shape, "rhs2"));

  Shape rhs_shape = ShapeUtil::MakeShape(F32, {spec.k, spec.n});
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateConcatenate(rhs_shape, {rhs0, rhs1, rhs2}, 0));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);

  Shape dot_shape = ShapeUtil::MakeShape(F32, {spec.m, spec.n});
  builder.AddInstruction(HloInstruction::CreateDot(
      dot_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto computation = m->AddEntryComputation(builder.Build());
  PoplarAlgebraicSimplifier simplifier;
  TF_ASSERT_OK_AND_ASSIGN(bool run_successful, simplifier.Run(m.get()));
  ASSERT_TRUE(run_successful);

  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), dot_shape));

  auto match_dot_0 = m::Dot(m::Slice(m::Constant()), m::Parameter(0));
  auto match_dot_1 = m::Dot(m::Slice(m::Constant()), m::Parameter(1));
  auto match_dot_2 = m::Dot(m::Slice(m::Constant()), m::Parameter(2));
  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Add(m::Add(match_dot_0, match_dot_1), match_dot_2)));
}

// Test that we transform
//  dot(concat(A, B, C), const)
// to
//  add(dot(A, const_0), dot(B, const_1),  dot(C, const_2))
TEST_P(DotOfConcatSimplificationTest, ConstantRHS) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  DotOfConcatTestSpec spec = GetParam();

  ASSERT_GE(spec.k, 4);

  int64_t k0 = spec.k / 4;
  int64_t k1 = spec.k / 4;
  int64_t k2 = spec.k / 4;
  int64_t k3 = spec.k - k0 - k1 - k2;

  Shape lhs0_shape = ShapeUtil::MakeShape(F32, {spec.m, k0});
  Shape lhs1_shape = ShapeUtil::MakeShape(F32, {spec.m, k1});
  Shape lhs2_shape = ShapeUtil::MakeShape(F32, {spec.m, k2});
  Shape lhs3_shape = ShapeUtil::MakeShape(F32, {spec.m, k3});

  HloInstruction* lhs0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs0_shape, "lhs0"));
  HloInstruction* lhs1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, lhs1_shape, "lhs1"));
  HloInstruction* lhs2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, lhs2_shape, "lhs2"));
  HloInstruction* lhs3 = builder.AddInstruction(
      HloInstruction::CreateParameter(3, lhs3_shape, "lhs3"));

  Shape lhs_shape = ShapeUtil::MakeShape(F32, {spec.m, spec.k});
  HloInstruction* lhs =
      builder.AddInstruction(HloInstruction::CreateConcatenate(
          lhs_shape, {lhs0, lhs1, lhs2, lhs3}, 1));

  Shape rhs_shape = ShapeUtil::MakeShape(F32, {spec.k, spec.n});
  auto* rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/10.0, /*to=*/10000.0, /*rows=*/spec.k, /*cols=*/spec.n)));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);

  Shape dot_shape = ShapeUtil::MakeShape(F32, {spec.m, spec.n});
  builder.AddInstruction(HloInstruction::CreateDot(
      dot_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto computation = m->AddEntryComputation(builder.Build());
  PoplarAlgebraicSimplifier simplifier;
  TF_ASSERT_OK_AND_ASSIGN(bool run_successful, simplifier.Run(m.get()));
  ASSERT_TRUE(run_successful);
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), dot_shape));

  auto match_dot_0 = m::Dot(m::Parameter(0), m::Slice(m::Constant()));
  auto match_dot_1 = m::Dot(m::Parameter(1), m::Slice(m::Constant()));
  auto match_dot_2 = m::Dot(m::Parameter(2), m::Slice(m::Constant()));
  auto match_dot_3 = m::Dot(m::Parameter(3), m::Slice(m::Constant()));
  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Add(m::Add(m::Add(match_dot_0, match_dot_1), match_dot_2),
                        match_dot_3)));
}

DotOfConcatTestSpec kDotOfConcatTestSpecs[] = {
    {/*m=*/3, /*k=*/9, /*n=*/3},    //
    {/*m=*/3, /*k=*/20, /*n=*/3},   //
    {/*m=*/1, /*k=*/18, /*n=*/5},   //
    {/*m=*/20, /*k=*/20, /*n=*/1},  //
    {/*m=*/1, /*k=*/16, /*n=*/1},   //
};

TEST_F(DotOfConcatSimplificationTest, ConcatIntoScalarDot) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    param0 = f32[4] parameter(0)
    param1 = f32[1] parameter(1)
    constant = f32[5] constant({-0.38, 0.07, -0.62, 0.66, 0.20})
    concat = f32[5] concatenate(param0, param1), dimensions={0}
    ROOT dot = f32[] dot(concat, constant), lhs_contracting_dims={0},
                                            rhs_contracting_dims={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(DotOfConcatSimplificationTest, RankOneRHS) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      param0 = f32[2, 2, 2] parameter(0)
      param1 = f32[2, 2, 2] parameter(1)
      constant = f32[4] constant({-0.38, 0.07, -0.62, 0.66})
      concat = f32[2, 2, 4] concatenate(param0, param1), dimensions={2}
      ROOT dot = f32[2, 2] dot(concat, constant), lhs_contracting_dims={2},
                                                  rhs_contracting_dims={0}
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  IpuOptions_IpuAlgebraicSimplifierConfig config;
  config.set_enable_dot_strength(false);
  ASSERT_TRUE(PoplarAlgebraicSimplifier(config).Run(m.get()).ValueOrDie());

  const HloInstruction* slice0;
  const HloInstruction* slice1;
  auto rhs = m::Constant();
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Add(m::Dot(m::Parameter(0), m::Slice(&slice0, rhs)),
                        m::Dot(m::Parameter(1), m::Slice(&slice1, rhs)))));

  // Check slices target the same constant.
  EXPECT_EQ(slice0->operand(0), slice1->operand(0));

  // Check slice indices.
  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(2));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(2));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(4));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1));
}

// Test that DynamicUpdateSlice update param with any dimension equal to zero
// gets removed.
TEST_F(PoplarAlgebraicSimplifierTest, DynamicUpdateSliceZeroUpdate) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  const Shape dslice_shape = ShapeUtil::MakeShape(F32, {10});
  HloInstruction* const operand = builder.AddInstruction(
      HloInstruction::CreateParameter(0, dslice_shape, "operand"));
  const Shape update_shape = ShapeUtil::MakeShape(F32, {0});
  HloInstruction* const update = builder.AddInstruction(
      HloInstruction::CreateParameter(1, update_shape, "update"));
  HloInstruction* const start_indices = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>({})));
  builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      dslice_shape, operand, update,
      std::initializer_list<HloInstruction*>({start_indices})));
  const HloComputation* const computation =
      m->AddEntryComputation(builder.Build());

  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), operand);
}

INSTANTIATE_TEST_SUITE_P(DotOfConcatSimplificationTestInstantiation,
                         DotOfConcatSimplificationTest,
                         ::testing::ValuesIn(kDotOfConcatTestSpecs));

struct DotOfGatherTestSpec {
  int64_t m;
  int64_t k;
  int64_t n;
  int s;  // start index for dynamic slice on the non-contracting dimension
  int64_t lcd;  // left contracting dimension
  int64_t rcd;  // right contracting dimension
  bool neg;     // is negative testcase
};

class DotOfGatherSimplificationTest
    : public PoplarAlgebraicSimplifierTest,
      public ::testing::WithParamInterface<DotOfGatherTestSpec> {};

// input: dot(DS(ctA), ctB))
// where DS(ctA) = DS({M x K}, {s, 0}, {1, K}) and ctB = {K x N}.
// => input dimensions: dot({1 x K}, {K x N}) => {1 x N}.
// output: DS(dot(ctA, ctB))
// => output dimensions: DS ({M x N}, {s, 0}, {1, N}) => {1 x N}.
TEST_P(DotOfGatherSimplificationTest, ConstantRHS) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  DotOfGatherTestSpec spec = GetParam();

  ASSERT_LE(spec.s, spec.m);

  // For negative tests, increase k of the dynamic slice argument to prevent
  // the optimization (constants ctA, ctB must have equal contracting
  // dimensions).
  int64_t k_increase = spec.neg ? 5 : 0;
  int64_t lhs_rows = (spec.lcd == 0) ? (spec.k + k_increase) : spec.m;
  int64_t lhs_cols = (spec.lcd == 0) ? spec.m : (spec.k + k_increase);
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {lhs_rows, lhs_cols});
  auto* lhs = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/10.0, /*to=*/10000.0, /*rows=*/lhs_rows,
          /*cols=*/lhs_cols)));

  int32 start_row = (spec.lcd == 0) ? 0 : spec.s;
  int32 start_col = (spec.lcd == 0) ? spec.s : 0;
  std::vector<HloInstruction*> start_indices = {
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32>(start_row))),
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32>(start_col)))};
  int64_t slice_row_size = (spec.lcd == 0) ? spec.k : 1;
  int64_t slice_col_size = (spec.lcd == 0) ? 1 : spec.k;
  std::vector<int64_t> slice_sizes = {slice_row_size, slice_col_size};
  Shape ds_shape = ShapeUtil::MakeShape(F32, slice_sizes);
  auto* ds = builder.AddInstruction(HloInstruction::CreateDynamicSlice(
      ds_shape, lhs, start_indices, slice_sizes));

  int64_t rhs_rows = (spec.rcd == 0) ? spec.k : spec.n;
  int64_t rhs_cols = (spec.rcd == 0) ? spec.n : spec.k;
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {rhs_rows, rhs_cols});
  auto* rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/10.0, /*to=*/10000.0, /*rows=*/rhs_rows,
          /*cols=*/rhs_cols)));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(spec.lcd);
  dot_dnums.add_rhs_contracting_dimensions(spec.rcd);

  int64_t dot_row_size = 1;
  int64_t dot_col_size = spec.n;
  Shape dot_shape = ShapeUtil::MakeShape(F32, {dot_row_size, dot_col_size});
  builder.AddInstruction(HloInstruction::CreateDot(
      dot_shape, ds, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto computation = m->AddEntryComputation(builder.Build());
  PoplarAlgebraicSimplifier simplifier;
  TF_ASSERT_OK_AND_ASSIGN(bool run_successful, simplifier.Run(m.get()));
  ASSERT_TRUE(run_successful);
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), dot_shape));

  if (spec.neg) {
    EXPECT_NE(computation->root_instruction()->opcode(),
              HloOpcode::kDynamicSlice);
  } else {
    EXPECT_THAT(computation->root_instruction(),
                GmockMatch(m::DynamicSlice(m::Dot(m::Constant(), m::Constant()),
                                           m::Constant(), m::Constant())));
  }
}

// input: dot(ctA, DS(ctB))
// where ctA = {M x K} and DS(ctB) = DS({K x N}, {0, s}, {K, 1}).
// => input dimensions: dot({M x K}, {K x 1}) => {M x 1}.
// output: DS(dot(ctA, ctB))
// => output dimensions: DS ({M x N}, {0, s}, {M, 1}) => {M x 1}.
TEST_P(DotOfGatherSimplificationTest, ConstantLHS) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  DotOfGatherTestSpec spec = GetParam();

  ASSERT_LE(spec.s, spec.n);

  int64_t lhs_rows = (spec.lcd == 0) ? spec.k : spec.m;
  int64_t lhs_cols = (spec.lcd == 0) ? spec.m : spec.k;
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {lhs_rows, lhs_cols});
  auto* lhs = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/10.0, /*to=*/10000.0, /*rows=*/lhs_rows,
          /*cols=*/lhs_cols)));

  // For negative tests increase k of the dynamic slice argument to prevent
  // the optimization
  int64_t k_increase = spec.neg ? 5 : 0;
  int64_t rhs_rows = (spec.rcd == 0) ? (spec.k + k_increase) : spec.n;
  int64_t rhs_cols = (spec.rcd == 0) ? spec.n : (spec.k + k_increase);
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {rhs_rows, rhs_cols});
  auto* rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/10.0, /*to=*/10000.0, /*rows=*/rhs_rows,
          /*cols=*/rhs_cols)));

  int32 start_row = (spec.rcd == 0) ? 0 : spec.s;
  int32 start_col = (spec.rcd == 0) ? spec.s : 0;
  std::vector<HloInstruction*> start_indices = {
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32>(start_row))),
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32>(start_col)))};
  int64_t slice_row_size = (spec.rcd == 0) ? spec.k : 1;
  int64_t slice_col_size = (spec.rcd == 0) ? 1 : spec.k;
  std::vector<int64_t> slice_sizes = {slice_row_size, slice_col_size};
  Shape ds_shape = ShapeUtil::MakeShape(F32, slice_sizes);
  auto* ds = builder.AddInstruction(HloInstruction::CreateDynamicSlice(
      ds_shape, rhs, start_indices, slice_sizes));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(spec.lcd);
  dot_dnums.add_rhs_contracting_dimensions(spec.rcd);

  int64_t dot_row_size = spec.m;
  int64_t dot_col_size = 1;
  Shape dot_shape = ShapeUtil::MakeShape(F32, {dot_row_size, dot_col_size});
  builder.AddInstruction(HloInstruction::CreateDot(
      dot_shape, lhs, ds, dot_dnums, DefaultPrecisionConfig(2)));

  auto computation = m->AddEntryComputation(builder.Build());
  PoplarAlgebraicSimplifier simplifier;
  TF_ASSERT_OK_AND_ASSIGN(bool run_successful, simplifier.Run(m.get()));
  ASSERT_TRUE(run_successful);
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), dot_shape));

  if (spec.neg) {
    EXPECT_NE(computation->root_instruction()->opcode(),
              HloOpcode::kDynamicSlice);
  } else {
    EXPECT_THAT(computation->root_instruction(),
                GmockMatch(m::DynamicSlice(m::Dot(m::Constant(), m::Constant()),
                                           m::Constant(), m::Constant())));
  }
}

std::vector<DotOfGatherTestSpec> DotOfGatherPositiveNegativeTests() {
  std::vector<DotOfGatherTestSpec> positives = {
      // "Classical dot", i.e. matrix multiply:
      {/*m=*/10, /*k=*/10, /*n=*/5, /*s=*/0, /*lcd=*/1, /*rcd=*/0,
       /*neg=*/false},
      {/*m=*/20, /*k=*/20, /*n=*/3, /*s=*/2, /*lcd=*/1, /*rcd=*/0,
       /*neg=*/false},
      {/*m=*/10, /*k=*/3, /*n=*/10, /*s=*/9, /*lcd=*/1, /*rcd=*/0,
       /*neg=*/false},
      // Note: testing for m=1 and n=1 is unnecessary, as this optimizes to
      // dot(ct, ct) before DotOfGather optimization kicks in.
      // Contract on rows:
      {/*m=*/10, /*k=*/10, /*n=*/5, /*s=*/0, /*lcd=*/0, /*rcd=*/0,
       /*neg=*/false},
      {/*m=*/20, /*k=*/20, /*n=*/3, /*s=*/2, /*lcd=*/0, /*rcd=*/0,
       /*neg=*/false},
      {/*m=*/10, /*k=*/3, /*n=*/10, /*s=*/9, /*lcd=*/0, /*rcd=*/0,
       /*neg=*/false},
      // Reverse matrix multiply:
      {/*m=*/10, /*k=*/10, /*n=*/5, /*s=*/0, /*lcd=*/0, /*rcd=*/1,
       /*neg=*/false},
      {/*m=*/20, /*k=*/20, /*n=*/3, /*s=*/2, /*lcd=*/0, /*rcd=*/1,
       /*neg=*/false},
      {/*m=*/10, /*k=*/3, /*n=*/10, /*s=*/9, /*lcd=*/0, /*rcd=*/1,
       /*neg=*/false},
      // Contract on columns:
      {/*m=*/10, /*k=*/10, /*n=*/5, /*s=*/0, /*lcd=*/1, /*rcd=*/1,
       /*neg=*/false},
      {/*m=*/20, /*k=*/20, /*n=*/3, /*s=*/2, /*lcd=*/1, /*rcd=*/1,
       /*neg=*/false},
      {/*m=*/10, /*k=*/3, /*n=*/10, /*s=*/9, /*lcd=*/1, /*rcd=*/1,
       /*neg=*/false},
  };
  std::vector<DotOfGatherTestSpec> all;
  for (int i = 0; i < positives.size(); i++) {
    DotOfGatherTestSpec positive_test = positives[i];
    all.push_back(positive_test);
    DotOfGatherTestSpec negative_test = positive_test;
    negative_test.neg = true;
    all.push_back(negative_test);
  }
  return all;
}

INSTANTIATE_TEST_SUITE_P(
    DotOfGatherSimplificationTestInstantiation, DotOfGatherSimplificationTest,
    ::testing::ValuesIn(DotOfGatherPositiveNegativeTests()));

TEST_F(PoplarAlgebraicSimplifierTest, TupleReduceReshape) {
  const char* hlo_string = R"(
HloModule module

reducer {
parameter.1 = f32[] parameter(0)
parameter.3 = f32[] parameter(2)
add.2 = f32[] add(parameter.1, parameter.3)
parameter.0 = f32[] parameter(1)
parameter.2 = f32[] parameter(3)
add.3 = f32[] add(parameter.0, parameter.2)
ROOT tuple.4 = (f32[], f32[]) tuple(add.2, add.3)
}

ENTRY entry {
parameter.6 = (f32[], f32[]) parameter(0)
get-tuple-element.10 = f32[] get-tuple-element(parameter.6), index=0
get-tuple-element.11 = f32[] get-tuple-element(parameter.6), index=1
constant = f32[] constant(0)
ROOT reduce = (f32[], f32[]) reduce(get-tuple-element.10, get-tuple-element.11, constant, constant), dimensions={}, to_apply=reducer
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Tuple(
                        m::Reshape(m::GetTupleElement(m::Parameter(), 0)),
                        m::Reshape(m::GetTupleElement(m::Parameter(), 1)))));
}

TEST_F(PoplarAlgebraicSimplifierTest, TupleReduceBroadcast) {
  const char* hlo_string = R"(
HloModule module

reducer {
parameter.1 = f32[] parameter(0)
parameter.3 = f32[] parameter(2)
mul.2 = f32[] add(parameter.1, parameter.3)
parameter.0 = f32[] parameter(1)
parameter.2 = f32[] parameter(3)
add.3 = f32[] add(parameter.0, parameter.2)
ROOT tuple.4 = (f32[], f32[]) tuple(mul.2, add.3)
}

ENTRY entry {
parameter.6 = (f32[0, 10, 10], f32[0, 10, 10]) parameter(0)
get-tuple-element.10 = f32[0, 10, 10] get-tuple-element(parameter.6), index=0
get-tuple-element.11 = f32[0, 10, 10] get-tuple-element(parameter.6), index=1
constant.0 = f32[] constant(0)
constant.1 = f32[] constant(1)
ROOT reduce = (f32[10, 10], f32[10, 10]) reduce(get-tuple-element.10, get-tuple-element.11, constant.0, constant.1), dimensions={0}, to_apply=reducer
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Tuple(m::Broadcast(m::ConstantScalar(0)),
                                        m::Broadcast(m::ConstantScalar(1)))));
}

TEST_F(PoplarAlgebraicSimplifierTest, ZeroSizedReshapeWithoutLayout) {
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1}), "param"));
  HloInstruction* broadcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(
          ShapeUtil::MakeShape(F32, {0, 1}), param, {1}));

  // Create a reshape with zero sized result and without layout.
  Shape reshaped_shape = ShapeUtil::MakeShape(F32, {0});
  reshaped_shape.clear_layout();
  builder.AddInstruction(
      HloInstruction::CreateReshape(reshaped_shape, broadcast));

  std::unique_ptr<VerifiedHloModule> module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
}

TEST_F(PoplarAlgebraicSimplifierTest,
       DividedByConstantInstructionWithoutLayout) {
  Shape shape = ShapeUtil::MakeShape(F32, {});
  shape.clear_layout();
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));

  HloInstruction* const_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(20.0f)));
  builder.AddInstruction(HloInstruction::CreateBinary(shape, HloOpcode::kDivide,
                                                      param, const_value));

  std::unique_ptr<VerifiedHloModule> module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  PoplarAlgebraicSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Multiply()));
}

// Test that 1/sqrt(X) is simplified to rsqrt(X).
TEST_F(PoplarAlgebraicSimplifierTest, RecipSqrt) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    sqrt = f32[] sqrt(p0)
    ROOT div = f32[] divide(p1, sqrt)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::MultiplyAnyOrder(m::Parameter(1),
                                             m::Rsqrt(m::Parameter(0)))));
}

// Test that 1/rsqrt(X) is simplified to sqrt(X).
TEST_F(PoplarAlgebraicSimplifierTest, RecipRsqrt) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    rsqrt = f32[] rsqrt(p0)
    ROOT div = f32[] divide(p1, rsqrt)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::MultiplyAnyOrder(m::Parameter(1),
                                             m::Sqrt(m::Parameter(0)))));
}

TEST_F(PoplarAlgebraicSimplifierTest, DotContractingReorder_RL) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    rhs = f32[6, 2] constant({{1, 2},{3, 4},{5, 6},{1, 1},{1, 1},{1, 1}})
    t0 = f32[2, 2, 3] parameter(0)
    t1 = f32[2, 3, 2] transpose(t0), dimensions={0, 2, 1}
    lhs = f32[2, 6] reshape(t1)
    ROOT dot.5 = f32[2, 2] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  auto shape1 = ShapeUtil::MakeShape(F32, {2, 6});
  auto shape2 = ShapeUtil::MakeShape(F32, {3, 2, 2});
  auto shape3 = ShapeUtil::MakeShape(F32, {2, 3, 2});
  // The transformation of moving transpose and reshape to the constant side
  // is layout insensitive. We ignore layout when checking shapes.
  const HloInstruction* transpose;
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(
                  m::Reshape(m::Parameter(0)).WithShapeCompatibleTo(&shape1),
                  m::Reshape(m::Transpose(&transpose,
                                          m::Reshape(m::Constant())
                                              .WithShapeCompatibleTo(&shape2))
                                 .WithShapeCompatibleTo(&shape3)))));
  EXPECT_THAT(transpose->dimensions(), ElementsAre(1, 0, 2));
}

TEST_F(PoplarAlgebraicSimplifierTest, DotContractingReorder_RR) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    rhs = f32[2, 6] constant({{1, 2, 3, 4, 5, 6},
                              {1, 1, 1, 1, 1, 1}})
    t0 = f32[2, 2, 3] parameter(0)
    t1 = f32[2, 3, 2] transpose(t0), dimensions={0, 2, 1}
    lhs = f32[2, 6] reshape(t1)
    ROOT dot.5 = f32[2, 2] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  auto shape1 = ShapeUtil::MakeShape(F32, {2, 6});
  auto shape2 = ShapeUtil::MakeShape(F32, {2, 3, 2});
  auto shape3 = ShapeUtil::MakeShape(F32, {2, 2, 3});
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(
                  m::Reshape(m::Parameter(0)).WithShapeCompatibleTo(&shape1),
                  m::Reshape(m::Transpose(m::Reshape(m::Constant())
                                              .WithShapeCompatibleTo(&shape2))
                                 .WithShapeCompatibleTo(&shape3)))));
}

TEST_F(PoplarAlgebraicSimplifierTest, DotContractingReorder_LR) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    rhs = f32[2, 6] constant({{1, 2, 3, 4, 5, 6},
                              {1, 1, 1, 1, 1, 1}})
    t0 = f32[2, 3, 2] parameter(0)
    t1 = f32[3, 2, 2] transpose(t0), dimensions={1, 0, 2}
    lhs = f32[6, 2] reshape(t1)
    ROOT dot.5 = f32[2, 2] dot(lhs, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={1}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  auto shape1 = ShapeUtil::MakeShape(F32, {6, 2});
  auto shape2 = ShapeUtil::MakeShape(F32, {2, 3, 2});
  auto shape3 = ShapeUtil::MakeShape(F32, {2, 2, 3});
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(
                  m::Reshape(m::Parameter(0)).WithShapeCompatibleTo(&shape1),
                  m::Reshape(m::Transpose(m::Reshape(m::Constant())
                                              .WithShapeCompatibleTo(&shape2))
                                 .WithShapeCompatibleTo(&shape3)))));
}

TEST_F(PoplarAlgebraicSimplifierTest, DotContractingReorder_LR2) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    rhs = f32[8, 2] constant({{1, 1},{2, 2},{3, 3},{4, 4},{5, 5},{6, 6},{7, 7},{8, 8}})
    t0 = f32[2, 2, 2, 2] parameter(0)
    t1 = f32[2, 2, 2, 2] transpose(t0), dimensions={0, 2, 3, 1}
    lhs = f32[2, 8] reshape(t1)
    ROOT dot.5 = f32[2, 2] dot(lhs, rhs), lhs_contracting_dims={1},
                                          rhs_contracting_dims={0}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  auto shape1 = ShapeUtil::MakeShape(F32, {2, 8});
  auto shape2 = ShapeUtil::MakeShape(F32, {2, 2, 2, 2});
  const HloInstruction* transpose;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Dot(
          m::Reshape(m::Parameter(0)).WithShapeCompatibleTo(&shape1),
          m::Reshape(m::Transpose(
              &transpose,
              m::Reshape(m::Constant()).WithShapeCompatibleTo(&shape2))))));
  EXPECT_THAT(transpose->dimensions(), ElementsAre(2, 0, 1, 3));
}

TEST_F(PoplarAlgebraicSimplifierTest, DotContractingReorder_MM) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    rhs = f32[2, 6, 2] constant({{{1, 1},{2, 2},{3, 3},{4, 4},{5, 5},{6, 6}},
                                 {{1, 1},{2, 2},{3, 3},{4, 4},{5, 5},{6, 6}}})
    t0 = f32[2, 2, 3, 2] parameter(0)
    t1 = f32[2, 3, 2, 2] transpose(t0), dimensions={0, 2, 1, 3}
    lhs = f32[2, 6, 2] reshape(t1)
    ROOT dot.5 = f32[2, 2, 2] dot(lhs, rhs), lhs_batch_dims={0}, lhs_contracting_dims={1},
                                             rhs_batch_dims={0}, rhs_contracting_dims={1}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  auto shape1 = ShapeUtil::MakeShape(F32, {2, 6, 2});
  auto shape2 = ShapeUtil::MakeShape(F32, {2, 3, 2, 2});
  auto shape3 = ShapeUtil::MakeShape(F32, {2, 2, 3, 2});
  const HloInstruction* transpose;
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(
                  m::Reshape(m::Parameter(0)).WithShapeCompatibleTo(&shape1),
                  m::Reshape(m::Transpose(&transpose,
                                          m::Reshape(m::Constant())
                                              .WithShapeCompatibleTo(&shape2))
                                 .WithShapeCompatibleTo(&shape3)))));
  EXPECT_THAT(transpose->dimensions(), ElementsAre(0, 2, 1, 3));
}

TEST_F(PoplarAlgebraicSimplifierTest, DotContractingReorder_NegTranspose) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    rhs = f32[12, 2] constant({{1, 1},{2, 2},{3, 3},{4, 4},{5, 5},{6, 6},{1, 1},{2, 2},{3, 3},{4, 4},{5, 5},{6, 6}})
    t0 = f32[3, 4, 2] parameter(0)
    t1 = f32[2, 3, 4] transpose(t0), dimensions={2, 0, 1}
    lhs = f32[2, 12] reshape(t1)
    ROOT dot.5 = f32[2, 2] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  // Transpose affects non-contracting dimension. The transpose and reshape
  // should not be moved to the constant side.
  ASSERT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(PoplarAlgebraicSimplifierTest, DotContractingReorder_NegReshape) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    rhs = f32[8, 2] constant({{1, 1},{2, 2},{3, 3},{4, 4},{1, 1},{2, 2},{3, 3},{4, 4}})
    t0 = f32[2, 4, 3] parameter(0)
    t1 = f32[2, 3, 4] transpose(t0), dimensions={0, 2, 1}
    lhs = f32[3, 8] reshape(t1)
    ROOT dot.5 = f32[3, 2] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  // Reshape affects non-contracting dimensions. The transpose and reshape
  // should not be moved to the constant side.
  ASSERT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(PoplarAlgebraicSimplifierTest, DotContractingReorder_NegConstant) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    t0 = f32[2, 3, 4] parameter(0)
    t1 = f32[2, 4, 3] transpose(t0), dimensions={0, 2, 1}
    lhs = f32[2, 12] reshape(t1)
    rhs = f32[12, 2] parameter(1)
    ROOT dot.5 = f32[2, 2] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  // Both operands are non-constant, so the optimization should not happen.
  ASSERT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(PoplarAlgebraicSimplifierTest,
       DotContractingReorder_SizeOneDimsNoChange) {
  // This isn't transformed (notice that the relative order of the `2` and `3`
  // dims doesn't change, so there's no opportunity here), but it's
  // nonetheless an interesting testcase because of the presence of the size-1
  // dimensions.
  const char* kModuleStr = R"(
  HloModule m
  test {
   param = f32[1,2,5,3] parameter(0)
   transpose = f32[1,5,2,3] transpose(param), dimensions={0,2,1,3}
   reshape = f32[5,6] reshape(transpose)
   constant = f32[6,4] constant({{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4}})
   ROOT dot = f32[5,4] dot(reshape, constant),
     lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(PoplarAlgebraicSimplifierTest, DotContractingReorder_SizeOneDims) {
  const char* kModuleStr = R"(
  HloModule m
  test {
   param = f32[1,2,3,5] parameter(0)
   transpose = f32[1,3,2,5] transpose(param), dimensions={0,2,1,3}
   reshape = f32[6,5] reshape(transpose)
   constant = f32[6,4] constant({{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4}})
   ROOT dot = f32[5,4] dot(reshape, constant),
     lhs_contracting_dims={0}, rhs_contracting_dims={0}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  auto shape1 = ShapeUtil::MakeShape(F32, {6, 5});
  auto shape2 = ShapeUtil::MakeShape(F32, {1, 3, 2, 4});
  auto shape3 = ShapeUtil::MakeShape(F32, {1, 2, 3, 4});
  const HloInstruction* transpose;
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(
                  m::Reshape(m::Parameter(0)).WithShapeCompatibleTo(&shape1),
                  m::Reshape(m::Transpose(&transpose,
                                          m::Reshape(m::Constant())
                                              .WithShapeCompatibleTo(&shape2))
                                 .WithShapeCompatibleTo(&shape3)))));
  EXPECT_THAT(transpose->dimensions(), ElementsAre(0, 2, 1, 3));
}

TEST_F(PoplarAlgebraicSimplifierTest,
       DotContractingReorder_NoChangeInContractingDimsOrder) {
  // No optimization opportunity here because the transpose does not reorder
  // the contracting dims.
  const char* kModuleStr = R"(
  HloModule m
  test {
    param = f32[2,5,1,3] parameter(0)
    transpose = f32[1,5,2,3] transpose(param), dimensions={2,1,0,3}
    reshape = f32[5,6] reshape(transpose)
    constant = f32[6,4] constant({{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4}})
    ROOT dot = f32[5,4] dot(reshape, constant),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(PoplarAlgebraicSimplifierTest, CompareIota) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    zero = s32[] constant(0)
    iota = s32[128] iota(), iota_dimension=0
    broad = s32[128] broadcast(zero), dimensions={}
    ROOT compare = pred[128] compare(iota, broad), direction=LT
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::ConstantScalar(false))));
}

TEST_F(PoplarAlgebraicSimplifierTest, CompareLtZero) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      zero = u32[] constant(0)
      param = u32[] parameter(0)
      ROOT compare = pred[] compare(param, zero), direction=LT
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::ConstantScalar(false)));
}

TEST_F(PoplarAlgebraicSimplifierTest, CompareLeZero) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      zero = u32[] constant(0)
      param = u32[] parameter(0)
      ROOT compare = pred[] compare(param, zero), direction=LE
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Le(m::Parameter(0), m::ConstantEffectiveScalar(0))));
}

TEST_F(PoplarAlgebraicSimplifierTest, CompareGeZero) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      zero = u32[] constant(0)
      param = u32[] parameter(0)
      ROOT compare = pred[] compare(param, zero), direction=GE
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::ConstantScalar(true)));
}

TEST_F(PoplarAlgebraicSimplifierTest, CompareGtZero) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      zero = u32[] constant(0)
      param = u32[] parameter(0)
      ROOT compare = pred[] compare(param, zero), direction=GT
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Gt(m::Parameter(0), m::ConstantEffectiveScalar(0))));
}

TEST_F(PoplarAlgebraicSimplifierTest, CompareZeroGt) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      zero = u32[] constant(0)
      param = u32[] parameter(0)
      ROOT compare = pred[] compare(zero, param), direction=GT
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::ConstantScalar(false)));
}

TEST_F(PoplarAlgebraicSimplifierTest, CompareZeroGe) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      zero = u32[] constant(0)
      param = u32[] parameter(0)
      ROOT compare = pred[] compare(zero, param), direction=GE
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Ge(m::ConstantEffectiveScalar(0), m::Parameter(0))));
}

TEST_F(PoplarAlgebraicSimplifierTest, CompareZeroLe) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      zero = u32[] constant(0)
      param = u32[] parameter(0)
      ROOT compare = pred[] compare(zero, param), direction=LE
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::ConstantScalar(true)));
}

TEST_F(PoplarAlgebraicSimplifierTest, CompareZeroLt) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      zero = u32[] constant(0)
      param = u32[] parameter(0)
      ROOT compare = pred[] compare(zero, param), direction=LT
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Lt(m::ConstantEffectiveScalar(0), m::Parameter(0))));
}

TEST_F(PoplarAlgebraicSimplifierTest, CompareSame) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    param = s32[123] parameter(0)
    ROOT compare = pred[123] compare(param, param), direction=GE
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::ConstantScalar(true))));
}

TEST_F(PoplarAlgebraicSimplifierTest, CompareAdd) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    param = s32[] parameter(0)
    const_3 = s32[] constant(3)
    const_4 = s32[] constant(4)
    add = s32[] add(param, const_3)
    ROOT compare = pred[] compare(add, const_4), direction=GE
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  ASSERT_TRUE(HloConstantFolding().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Compare(m::Parameter(0), m::ConstantScalar(1))));
}

TEST_F(PoplarAlgebraicSimplifierTest, CompareNegate) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    p = s32[4] parameter(0)
    c1 = s32[4] constant({3, 2, 1, 0})
    c2 = s32[4] constant({3, 2, 1, 0})
    c3 = s32[4] constant({3, 2, 1, 0})
    c4 = s32[4] constant({3, 2, 1, 0})
    c5 = s32[4] constant({3, 2, 1, 0})
    c6 = s32[4] constant({3, 2, 1, 0})
    n = s32[4] negate(p)
    cmp1 = pred[4] compare(n, c1), direction=LT
    cmp2 = pred[4] compare(n, c2), direction=LE
    cmp3 = pred[4] compare(n, c3), direction=GT
    cmp4 = pred[4] compare(n, c4), direction=GE
    cmp5 = pred[4] compare(n, c5), direction=EQ
    cmp6 = pred[4] compare(n, c6), direction=NE

    ROOT tuple = (pred[4], pred[4], pred[4], pred[4], pred[4], pred[4]) tuple(cmp1, cmp2, cmp3, cmp4, cmp5, cmp6)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(
      HloPassFix<PoplarAlgebraicSimplifier>().Run(m.get()).ValueOrDie());
  ASSERT_TRUE(HloPassFix<HloConstantFolding>().Run(m.get()).ValueOrDie());

  auto* cmp1 = FindInstruction(m.get(), "compare");
  CHECK_NOTNULL(cmp1);
  EXPECT_EQ(cmp1->comparison_direction(), ComparisonDirection::kGt);
  auto* cmp2 = FindInstruction(m.get(), "compare.1");
  CHECK_NOTNULL(cmp2);
  EXPECT_EQ(cmp2->comparison_direction(), ComparisonDirection::kGe);
  auto* cmp3 = FindInstruction(m.get(), "compare.2");
  CHECK_NOTNULL(cmp3);
  EXPECT_EQ(cmp3->comparison_direction(), ComparisonDirection::kLt);
  auto* cmp4 = FindInstruction(m.get(), "compare.3");
  CHECK_NOTNULL(cmp4);
  EXPECT_EQ(cmp4->comparison_direction(), ComparisonDirection::kLe);
  auto* cmp5 = FindInstruction(m.get(), "compare.4");
  CHECK_NOTNULL(cmp5);
  EXPECT_EQ(cmp5->comparison_direction(), ComparisonDirection::kEq);
  auto* cmp6 = FindInstruction(m.get(), "compare.5");
  CHECK_NOTNULL(cmp6);
  EXPECT_EQ(cmp6->comparison_direction(), ComparisonDirection::kNe);

  VLOG(1) << "MODULE " << m->ToString();
  const HloInstruction* c[6];
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Compare(m::Parameter(0), m::Constant(&c[0])),
                          m::Compare(m::Parameter(0), m::Constant(&c[1])),
                          m::Compare(m::Parameter(0), m::Constant(&c[2])),
                          m::Compare(m::Parameter(0), m::Constant(&c[3])),
                          m::Compare(m::Parameter(0), m::Constant(&c[4])),
                          m::Compare(m::Parameter(0), m::Constant(&c[5])))));
  for (size_t i = 0; i < 6; ++i) {
    const Literal& l = c[i]->literal();
    EXPECT_EQ(l.GetIntegralAsS64({0}), -3);
    EXPECT_EQ(l.GetIntegralAsS64({1}), -2);
    EXPECT_EQ(l.GetIntegralAsS64({2}), -1);
    EXPECT_EQ(l.GetIntegralAsS64({3}), 0);
  }
}

TEST_F(PoplarAlgebraicSimplifierTest, CompareSub) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    param_1 = s32[] parameter(0)
    const_3 = s32[] constant(3)
    const_4 = s32[] constant(4)
    sub = s32[] subtract(param_1, const_3)
    ROOT compare = pred[] compare(sub, const_4), direction=GE
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  ASSERT_TRUE(HloConstantFolding().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Compare(m::Parameter(0), m::ConstantScalar(7))));
}

TEST_F(PoplarAlgebraicSimplifierTest, RemainderOfIota) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    iota = s32[5,1000] iota(), iota_dimension=0
    five = s32[] constant(5)
    five_bcast = s32[5,1000] broadcast(s32[] five), dimensions={}
    ROOT remainder = s32[5,1000] remainder(iota, s32[5,1000] five_bcast)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Iota()));
}

TEST_F(PoplarAlgebraicSimplifierTest, RemainderOfNPlusIota) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    iota = s32[5,1000] iota(), iota_dimension=0
    five = s32[] constant(5)
    five_bcast = s32[5,1000] broadcast(five), dimensions={}
    sum = s32[5,1000] add(iota, five_bcast)
    ROOT remainder = s32[5,1000] remainder(sum, s32[5,1000] five_bcast)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Remainder(m::Iota(), m::Broadcast())));
}

// No simplification because 125 + 5 overflows S8.
TEST_F(PoplarAlgebraicSimplifierTest, RemainderOfNPlusIotaOverflow) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    iota = s8[126] iota(), iota_dimension=0
    five = s8[] constant(5)
    five_bcast = s8[126] broadcast(five), dimensions={}
    sum = s8[126] add(iota, five_bcast)
    ROOT remainder = s8[126] remainder(sum, s8[126] five_bcast)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(PoplarAlgebraicSimplifierTest, RepeatedRemainder) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    p = s32[1000] parameter(0)
    q = s32[1000] parameter(1)
    r = s32[1000] remainder(p, q)
    ROOT rr = s32[1000] remainder(r, q)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Remainder(m::Parameter(), m::Parameter())));
}

TEST_F(PoplarAlgebraicSimplifierTest, MinOfMaxToClamp) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    p0 = f32[4] parameter(0)
    c0 = f32[] constant(3.0)
    c1 = f32[] constant(4.0)
    b0 = f32[4] broadcast(c0), dimensions={}
    b1 = f32[4] broadcast(c1), dimensions={}
    m0 = f32[4] maximum(b0, p0)
    ROOT m1 = f32[4] minimum(m0, b1)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Clamp(m::Broadcast(m::ConstantScalar(3.0)), m::Parameter(0),
                          m::Broadcast(m::ConstantScalar(4.0)))));
}

TEST_F(PoplarAlgebraicSimplifierTest, MaxOfMinToClamp) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    p0 = f32[4] parameter(0)
    c0 = f32[] constant(3.0)
    c1 = f32[] constant(4.0)
    b0 = f32[4] broadcast(c0), dimensions={}
    b1 = f32[4] broadcast(c1), dimensions={}
    m0 = f32[4] minimum(p0, b1)
    ROOT m1 = f32[4] maximum(b0, m0)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Clamp(m::Broadcast(m::ConstantScalar(3.0)), m::Parameter(0),
                          m::Broadcast(m::ConstantScalar(4.0)))));
}

TEST_F(PoplarAlgebraicSimplifierTest, ClampOfClamp) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    p2 = f32[] parameter(2)
    c0 = f32[] clamp(p0, p1, p2)
    ROOT c1 = f32[] clamp(p0, c0, p2)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Clamp(m::Parameter(0), m::Parameter(1), m::Parameter(2))));
}

TEST_F(PoplarAlgebraicSimplifierTest, MaxOfClamp) {
  const char* kModuleStr = R"(
  HloModule m
  test {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    p2 = f32[] parameter(2)
    c0 = f32[] clamp(p0, p1, p2)
    ROOT m0 = f32[] maximum(p0, c0)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Clamp(m::Parameter(0), m::Parameter(1), m::Parameter(2))));
}

TEST_F(PoplarAlgebraicSimplifierTest, FuseConcat) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4,2] parameter(0)
      p1 = f32[6,2] parameter(1)
      p2 = f32[3,2] parameter(2)
      c0 = f32[10,2] concatenate(p0, p1), dimensions={0}
      ROOT c1 = f32[13,2] concatenate(c0, p2), dimensions={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Concatenate(m::Parameter(0), m::Parameter(1),
                                        m::Parameter(2))));
}

TEST_F(PoplarAlgebraicSimplifierTest, FuseConcatDifferentDim) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4,2] parameter(0)
      p1 = f32[6,2] parameter(1)
      p2 = f32[10,3] parameter(2)
      c0 = f32[10,2] concatenate(p0, p1), dimensions={0}
      ROOT c1 = f32[10,5] concatenate(c0, p2), dimensions={1}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Concatenate(
          m::Concatenate(m::Parameter(0), m::Parameter(1)), m::Parameter(2))));
}

TEST_F(PoplarAlgebraicSimplifierTest, FuseConcatSeveralUsers) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4,2] parameter(0)
      p1 = f32[6,2] parameter(1)
      p2 = f32[3,2] parameter(2)
      c0 = f32[10,2] concatenate(p0, p1), dimensions={0}
      c1 = f32[13,2] concatenate(c0, p2), dimensions={0}
      ROOT t = (f32[10,2], f32[13,2]) tuple(c0,c1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Concatenate(m::Parameter(0), m::Parameter(1)),
          m::Concatenate(m::Concatenate(m::Parameter(0), m::Parameter(1)),
                         m::Parameter(2)))));
}

TEST_F(PoplarAlgebraicSimplifierTest, FuseConcatMultiple) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4,2] parameter(0)
      p1 = f32[6,2] parameter(1)
      p2 = f32[3,2] parameter(2)
      c0 = f32[10,2] concatenate(p0, p1), dimensions={0}
      c1 = f32[9,2] concatenate(p2, p1), dimensions={0}
      c2 = f32[7,2] concatenate(p2, p0), dimensions={0}
      ROOT c3 = f32[26,2] concatenate(c0, p2, c1, p0), dimensions={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(
      HloPassFix<PoplarAlgebraicSimplifier>().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Concatenate(m::Parameter(0), m::Parameter(1),
                                        m::Parameter(2), m::Parameter(2),
                                        m::Parameter(1), m::Parameter(0))));
}

TEST_F(PoplarAlgebraicSimplifierTest, ElideStatefulGradientAccumulate) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4,2] parameter(0)
      p1 = f32[4,2] parameter(1)
      ga = f32[4,2] custom-call(p1), custom_call_target="StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":1}\n"
      ROOT a = f32[4,2] add(p0, ga)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(CustomOpReplacer().Run(m.get()).ValueOrDie());
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Add(m::Parameter(0), m::Parameter(1))));
}

TEST_F(PoplarAlgebraicSimplifierTest, ElideDotReduceBatchDim) {
  const char* kModuleStr = R"(
    HloModule m

    sum (x: f32[], y: f32[]) -> f32[] {
      y = f32[] parameter(1), backend_config="{}"
      x = f32[] parameter(0), backend_config="{}"
      ROOT add = f32[] add(x, y), backend_config="{\"isInplace\":true}"
    }

    test {
      p0 = f32[2, 3, 4] parameter(0)
      p1 = f32[2, 3, 5] parameter(1)
      d = f32[2,4,5] dot(p0, p1), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
      z = f32[] constant(0)
      ROOT r = f32[4,5] reduce(d, z), to_apply=sum, dimensions={0}
    }
  )";
  {
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
    ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
    EXPECT_THAT(m->entry_computation()->root_instruction(),
                GmockMatch(m::Dot()));
  }

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  Literal param0 = LiteralUtil::CreateR3<float>({
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
      {{-1, -2, -3, -4}, {-5, -6, -7, -8}, {-9, -10, -11, -12}},
  });
  Literal param1 = LiteralUtil::CreateR3<float>({
      {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
      {{-1, -2, -3, -4, -5}, {-6, -7, -8, -9, -10}, {-11, -12, -13, -14, -15}},
  });

  std::vector<Literal*> inputs = {&param0, &param1};
  EXPECT_TRUE(RunAndCompare(std::move(m), inputs, ErrorSpec{1e-3, 1e-3}));
}

TEST_F(PoplarAlgebraicSimplifierTest, ElideDotReduceMultipleBatchDims) {
  const char* kModuleStr = R"(
    HloModule m

    sum (x: f32[], y: f32[]) -> f32[] {
      y = f32[] parameter(1), backend_config="{}"
      x = f32[] parameter(0), backend_config="{}"
      ROOT add = f32[] add(x, y), backend_config="{\"isInplace\":true}"
    }

    test {
      p0 = f32[2, 2, 3, 4] parameter(0)
      p1 = f32[2, 2, 3, 5] parameter(1)
      d = f32[2, 2, 4, 5] dot(p0, p1), lhs_batch_dims={0, 1}, lhs_contracting_dims={2}, rhs_batch_dims={0, 1}, rhs_contracting_dims={2}
      z = f32[] constant(0)
      ROOT r = f32[4, 5] reduce(d, z), to_apply=sum, dimensions={0, 1}
    }
  )";

  {
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
    ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
    EXPECT_THAT(m->entry_computation()->root_instruction(),
                GmockMatch(m::Dot()));
  }

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  Literal param0 = LiteralUtil::CreateR4<float>(
      {{
           {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
           {{-1, -2, -3, -4}, {-5, -6, -7, -8}, {-9, -10, -11, -12}},
       },
       {
           {{-1, -2, -3, -4}, {-5, -6, -7, -8}, {-9, -10, -11, -12}},
           {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
       }});

  Literal param1 = LiteralUtil::CreateR4<float>(
      {{
           {{-1, -2, -3, -4, 1}, {-5, -6, -7, -8, 1}, {-9, -10, -11, -12, 1}},
           {{1, 2, 3, 4, 1}, {5, 6, 7, 8, 1}, {9, 10, 11, 12, 1}},
       },
       {
           {{1, 2, 3, 4, 1}, {5, 6, 7, 8, 1}, {9, 10, 11, 12, 1}},
           {{-1, -2, -3, -4, 1}, {-5, -6, -7, -8, 1}, {-9, -10, -11, -12, 1}},
       }});

  std::vector<Literal*> inputs = {&param0, &param1};
  EXPECT_TRUE(RunAndCompare(std::move(m), inputs, ErrorSpec{1e-3, 1e-3}));
}

TEST_F(PoplarAlgebraicSimplifierTest, ElideDotReduceDontOptimiseNonBatchDims) {
  const char* kModuleStr = R"(
    HloModule m

    sum (x: f32[], y: f32[]) -> f32[] {
      y = f32[] parameter(1), backend_config="{}"
      x = f32[] parameter(0), backend_config="{}"
      ROOT add = f32[] add(x, y), backend_config="{\"isInplace\":true}"
    }

    test {
      p0 = f32[2, 3, 4] parameter(0)
      p1 = f32[2, 3, 5] parameter(1)
      d = f32[2,4,5] dot(p0, p1), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
      z = f32[] constant(0)
      ROOT r = f32[2, 4] reduce(d, z), to_apply=sum, dimensions={2}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(PoplarAlgebraicSimplifierTest, ElideBitcastConvert) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4] parameter(0)
      a = s32[4] bitcast-convert(p0)
      ROOT b = u32[4] bitcast-convert(a)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::BitcastConvert(m::Parameter(0))));
}

TEST_F(PoplarAlgebraicSimplifierTest, MaxAndArgMax_1D) {
  absl::string_view hlo_string = R"(
HloModule top

argmax {
  running_max = f32[] parameter(0)
  running_max_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  cmp_val = pred[] compare(running_max, current_value), direction=GE
  new_max = f32[] select(cmp_val, running_max, current_value)

  compare_eq = pred[] compare(running_max, current_value), direction=EQ
  min_idx = u32[] minimum(running_max_idx, current_value_idx)
  min_val_idx = u32[] select(cmp_val, running_max_idx, current_value_idx)
  new_idx = u32[] select(compare_eq, min_idx, min_val_idx)

  ROOT out = (f32[], u32[]) tuple(new_max, new_idx)
}

ENTRY main {
  input = f32[10] parameter(0)
  idxs = u32[10] iota(), iota_dimension=0
  init = f32[] constant(-inf)
  zero_idx = u32[] constant(0)

  ROOT out = (f32[], u32[]) reduce(input, idxs, init, zero_idx), dimensions={0}, to_apply=%argmax
}
)";
  {
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
    ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
    HloInstruction* root = m->entry_computation()->root_instruction();
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MaxAndArgMax)(root));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
    Literal param0 =
        LiteralUtil::CreateR1<float>({1, 5, 2, 3, 4, 9, 6, 7, 8, 0});
    EXPECT_TRUE(RunAndCompare(std::move(m), {&param0}, ErrorSpec{1e-3, 1e-3}));
  }
}

TEST_F(PoplarAlgebraicSimplifierTest, MinAndArgMin_1D) {
  absl::string_view hlo_string = R"(
HloModule top

argmin {
  running_min = f32[] parameter(0)
  running_min_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  cmp_val = pred[] compare(running_min, current_value), direction=LE
  new_min = f32[] select(cmp_val, running_min, current_value)

  compare_eq = pred[] compare(running_min, current_value), direction=EQ
  min_idx = u32[] minimum(running_min_idx, current_value_idx)
  min_val_idx = u32[] select(cmp_val, running_min_idx, current_value_idx)
  new_idx = u32[] select(compare_eq, min_idx, min_val_idx)

  ROOT out = (f32[], u32[]) tuple(new_min, new_idx)
}

ENTRY main {
  input = f32[10] parameter(0)
  idxs = u32[10] iota(), iota_dimension=0
  init = f32[] constant(inf)
  zero_idx = u32[] constant(0)

  ROOT out = (f32[], u32[]) reduce(input, idxs, init, zero_idx), dimensions={0}, to_apply=%argmin
}
)";
  {
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
    ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
    HloInstruction* root = m->entry_computation()->root_instruction();
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MinAndArgMin)(root));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
    Literal param0 =
        LiteralUtil::CreateR1<float>({1, 5, 2, 3, 4, 9, 6, 7, 8, 0});
    EXPECT_TRUE(RunAndCompare(std::move(m), {&param0}, ErrorSpec{1e-3, 1e-3}));
  }
}

TEST_F(PoplarAlgebraicSimplifierTest, MaxAndArgMax1_2D) {
  absl::string_view hlo_string = R"(
HloModule top

argmax {
  running_max = f32[] parameter(0)
  running_max_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  cmp_val = pred[] compare(running_max, current_value), direction=GE
  new_max = f32[] select(cmp_val, running_max, current_value)

  compare_eq = pred[] compare(running_max, current_value), direction=EQ
  min_idx = u32[] minimum(running_max_idx, current_value_idx)
  min_val_idx = u32[] select(cmp_val, running_max_idx, current_value_idx)
  new_idx = u32[] select(compare_eq, min_idx, min_val_idx)

  ROOT out = (f32[], u32[]) tuple(new_max, new_idx)
}

ENTRY main {
  input = f32[2, 10] parameter(0)
  idxs = u32[2, 10] iota(), iota_dimension=0
  init = f32[] constant(-inf)
  zero_idx = u32[] constant(0)

  ROOT out = (f32[10], u32[10]) reduce(input, idxs, init, zero_idx), dimensions={0}, to_apply=%argmax
}
)";
  {
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
    ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
    HloInstruction* root = m->entry_computation()->root_instruction();
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MaxAndArgMax)(root));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
    Literal param0 = LiteralUtil::CreateR2<float>(
        {{8, 5, 2, 4, 4, 9, 6, 7, 8, 2}, {1, 9, 5, 3, 2, 8, 1, 2, -1, 0}});
    EXPECT_TRUE(RunAndCompare(std::move(m), {&param0}, ErrorSpec{1e-3, 1e-3}));
  }
}

TEST_F(PoplarAlgebraicSimplifierTest, MaxAndArgMax2_2D) {
  absl::string_view hlo_string = R"(
HloModule top

argmax {
  running_max = f32[] parameter(0)
  running_max_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  cmp_val = pred[] compare(running_max, current_value), direction=GE
  new_max = f32[] select(cmp_val, running_max, current_value)

  compare_eq = pred[] compare(running_max, current_value), direction=EQ
  min_idx = u32[] minimum(running_max_idx, current_value_idx)
  min_val_idx = u32[] select(cmp_val, running_max_idx, current_value_idx)
  new_idx = u32[] select(compare_eq, min_idx, min_val_idx)

  ROOT out = (f32[], u32[]) tuple(new_max, new_idx)
}

ENTRY main {
  input = f32[2, 10] parameter(0)
  idxs = u32[2, 10] iota(), iota_dimension=0
  init = f32[] constant(-inf)
  zero_idx = u32[] constant(0)

  ROOT out = (f32[10], u32[10]) reduce(input, idxs, init, zero_idx), dimensions={0}, to_apply=%argmax
}
)";
  {
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
    ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
    HloInstruction* root = m->entry_computation()->root_instruction();
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MaxAndArgMax)(root));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
    Literal param0 = LiteralUtil::CreateR2<float>(
        {{8, 5, 2, 4, 4, 9, 6, 7, 8, 2}, {1, 9, 5, 3, 2, 8, 1, 2, -1, 0}});
    EXPECT_TRUE(RunAndCompare(std::move(m), {&param0}, ErrorSpec{1e-3, 1e-3}));
  }
}

TEST_F(PoplarAlgebraicSimplifierTest, MinAndArgMin1_2D) {
  absl::string_view hlo_string = R"(
HloModule top

argmin {
  running_min = f32[] parameter(0)
  running_min_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  cmp_val = pred[] compare(running_min, current_value), direction=LE
  new_min = f32[] select(cmp_val, running_min, current_value)

  compare_eq = pred[] compare(running_min, current_value), direction=EQ
  min_idx = u32[] minimum(running_min_idx, current_value_idx)
  min_val_idx = u32[] select(cmp_val, running_min_idx, current_value_idx)
  new_idx = u32[] select(compare_eq, min_idx, min_val_idx)

  ROOT out = (f32[], u32[]) tuple(new_min, new_idx)
}

ENTRY main {
  input = f32[2, 10] parameter(0)
  idxs = u32[2, 10] iota(), iota_dimension=1
  init = f32[] constant(inf)
  zero_idx = u32[] constant(0)

  ROOT out = (f32[2], u32[2]) reduce(input, idxs, init, zero_idx), dimensions={1}, to_apply=%argmin
}
)";
  {
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
    ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
    HloInstruction* root = m->entry_computation()->root_instruction();
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MinAndArgMin)(root));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
    Literal param0 = LiteralUtil::CreateR2<float>(
        {{8, 5, 2, 4, 4, 9, 6, 7, 8, 2}, {1, 9, 5, 3, 2, 8, 1, 2, -1, 0}});
    EXPECT_TRUE(RunAndCompare(std::move(m), {&param0}, ErrorSpec{1e-3, 1e-3}));
  }
}

TEST_F(PoplarAlgebraicSimplifierTest, MinAndArgMinInvalidDims) {
  absl::string_view hlo_string = R"(
HloModule top

argmin {
  running_min = f32[] parameter(0)
  running_min_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  cmp_val = pred[] compare(running_min, current_value), direction=LE
  new_min = f32[] select(cmp_val, running_min, current_value)

  compare_eq = pred[] compare(running_min, current_value), direction=EQ
  min_idx = u32[] minimum(running_min_idx, current_value_idx)
  min_val_idx = u32[] select(cmp_val, running_min_idx, current_value_idx)
  new_idx = u32[] select(compare_eq, min_idx, min_val_idx)

  ROOT out = (f32[], u32[]) tuple(new_min, new_idx)
}

ENTRY main {
  input = f32[2, 10] parameter(0)
  idxs = u32[2, 10] iota(), iota_dimension=1
  init = f32[] constant(inf)
  zero_idx = u32[] constant(0)
  outs = (f32[10], u32[10]) reduce(input, idxs, init, zero_idx), dimensions={0}, to_apply=%argmin
  ROOT gte = u32[10] get-tuple-element(outs), index=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(PoplarAlgebraicSimplifierTest, MinAndArgMinInvalidStartingValues) {
  absl::string_view hlo_string = R"(
HloModule top

argmin {
  running_min = f32[] parameter(0)
  running_min_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  cmp_val = pred[] compare(running_min, current_value), direction=LE
  new_min = f32[] select(cmp_val, running_min, current_value)

  compare_eq = pred[] compare(running_min, current_value), direction=EQ
  min_idx = u32[] minimum(running_min_idx, current_value_idx)
  min_val_idx = u32[] select(cmp_val, running_min_idx, current_value_idx)
  new_idx = u32[] select(compare_eq, min_idx, min_val_idx)

  ROOT out = (f32[], u32[]) tuple(new_min, new_idx)
}

ENTRY main {
  input = f32[2, 10] parameter(0)
  idxs = u32[2, 10] iota(), iota_dimension=0
  init = f32[] constant(-inf)
  zero_idx = u32[] constant(0)
  outs = (f32[10], u32[10]) reduce(input, idxs, init, zero_idx), dimensions={0}, to_apply=%argmin
  ROOT gte = u32[10] get-tuple-element(outs), index=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(PoplarAlgebraicSimplifierTest, MinAndArgMinInvaliComparison) {
  absl::string_view hlo_string = R"(
HloModule top

argmin {
  running_min = f32[] parameter(0)
  running_min_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  cmp_val = pred[] compare(running_min, current_value), direction=LT
  new_min = f32[] select(cmp_val, running_min, current_value)

  compare_eq = pred[] compare(running_min, current_value), direction=EQ
  min_idx = u32[] minimum(running_min_idx, current_value_idx)
  min_val_idx = u32[] select(cmp_val, running_min_idx, current_value_idx)
  new_idx = u32[] select(compare_eq, min_idx, min_val_idx)

  ROOT out = (f32[], u32[]) tuple(new_min, new_idx)
}

ENTRY main {
  input = f32[2, 10] parameter(0)
  idxs = u32[2, 10] iota(), iota_dimension=1
  init = f32[] constant(inf)
  zero_idx = u32[] constant(0)
  outs = (f32[10], u32[10]) reduce(input, idxs, init, zero_idx), dimensions={0}, to_apply=%argmin
  ROOT gte = u32[10] get-tuple-element(outs), index=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(PoplarAlgebraicSimplifierTest, SimplifyAllReduceNormaliseAllReduce) {
  const char* kModuleStr = R"(
    HloModule m
    sum {
      y = f32[] parameter(1)
      x = f32[] parameter(0), control-predecessors={y}
      ROOT add = f32[] add(x, y), backend_config="{\"isInplace\":true}"
    }

    ENTRY main {
      arg0 = f32[1000] parameter(0)
      all-reduce0 = f32[1000] all-reduce(arg0), to_apply=sum
      normalise = f32[1000] custom-call(all-reduce0), custom_call_target="ReplicationNormalise"
      ROOT all-reduce1 = f32[1000] all-reduce(normalise), to_apply=sum
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(CustomOpReplacer().Run(m.get()).ValueOrDie());
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::AllReduce(m::Parameter(0))));
}

TEST_F(PoplarAlgebraicSimplifierTest,
       OptimiseNestedBroadcastsForElementwiseOps) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      p1 = f16[] parameter(1)
      b0 = f32[4] broadcast(p0), dimensions={}
      b1 = f16[4] broadcast(p1), dimensions={}
      c1 = f32[4] convert(b1)
      c2 = f32[4] convert(c1)
      ROOT multiply0 = f32[4] multiply(b0, c2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::Multiply(
                  m::Parameter(0), m::Convert(m::Convert(m::Parameter(1)))))));
}

TEST_F(PoplarAlgebraicSimplifierTest,
       OptimiseDoubleNestedBroadcastsForElementwiseOps) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      p1 = f16[] parameter(1)
      b0 = f32[4] broadcast(p0), dimensions={}
      b1 = f16[4] broadcast(p1), dimensions={}
      c1 = f32[4] convert(b1)
      c2 = f32[4] convert(c1)
      multiply0 = f32[4] multiply(b0, c2)
      ROOT convert0 = f32[4] convert(multiply0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::Convert(m::Multiply(
                  m::Parameter(0), m::Convert(m::Convert(m::Parameter(1))))))));
}

TEST_F(PoplarAlgebraicSimplifierTest, OptimiseBroadcastsForElementwiseOps) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      p1 = f16[] parameter(1)
      b0 = f32[4] broadcast(p0), dimensions={}
      b1 = f16[4] broadcast(p1), dimensions={}
      c1 = f32[4] convert(b1)
      ROOT multiply0 = f32[4] multiply(b0, c1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(
                  m::Multiply(m::Parameter(0), m::Convert(m::Parameter(1))))));
}

TEST_F(PoplarAlgebraicSimplifierTest, OptimiseBroadcastsForNonElementwiseOps) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      p1 = f16[] parameter(1)
      b0 = f32[4] broadcast(p0), dimensions={}
      b1 = f16[4] broadcast(p1), dimensions={}
      c1 = f32[4] convert(b1)
      ROOT d0 = f32[4,4] dot(b0, c1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Op().WithBinaryOperandsAnyOrder(
                  m::Broadcast(m::Parameter(0)),
                  m::Broadcast(m::Convert(m::Parameter(1))))));
}

TEST_F(PoplarAlgebraicSimplifierTest, MultipleDotStrengthReductions) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY test {
      a = c64[2,2] parameter(0)
      b = c64[2] parameter(1)
      cd = c64[2] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      c = f64[2,2] parameter(2)
      d = f64[2] parameter(3)
      dd = f64[2] dot(c, d), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT tuple = (c64[2], f64[2]) tuple(cd, dd)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  IpuOptions_IpuAlgebraicSimplifierConfig config;
  config.set_enable_dot_strength(true);
  ASSERT_TRUE(PoplarAlgebraicSimplifier(config).Run(m.get()).ValueOrDie());
  EXPECT_EQ(3, m->computation_count());
}

// Test that dynamic-update-slice with a scalar broadcast becomes a pad when the
// start_indices are too big.
TEST_F(PoplarAlgebraicSimplifierTest, DynamicUpdateSliceOfBroadcastToPadOob) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  constant.546 = f32[] constant(0)
  broadcast.467 = f32[2]{0} broadcast(constant.546), dimensions={}
  parameter.1 = f32[1]{0} parameter(0)
  constant.551 = s32[] constant(2)
  ROOT dynamic-update-slice.44 = f32[2]{0} dynamic-update-slice(broadcast.467, parameter.1, constant.551)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  VLOG(2) << "Before rewrite dus->pad\n" << module->ToString();
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  VLOG(2) << "After rewrite dus->pad\n" << module->ToString();
  auto* pad = module->entry_computation()->root_instruction();
  EXPECT_THAT(pad,
              GmockMatch(m::Pad(m::Parameter(0), m::ConstantScalar(0.0f))));
  EXPECT_FALSE(HasInteriorPadding(pad->padding_config()));
  ASSERT_EQ(pad->padding_config().dimensions_size(), 1);
  EXPECT_EQ(pad->padding_config().dimensions(0).edge_padding_low(), 1);
  EXPECT_EQ(pad->padding_config().dimensions(0).edge_padding_high(), 0);
}

TEST_F(PoplarAlgebraicSimplifierTest, UnaryVariadicReduceWindow) {
  const char* kModuleStr = R"(
    HloModule m
    fn {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      a = f32[] add(p0, p1)
      ROOT t = (f32[]) tuple(a)
    }
    test {
      p0 = f32[32,8,6,7] parameter(0)
      c = f32[] constant(0)
      ROOT r = (f32[32,8,6,7]) reduce-window(p0, c), to_apply=fn, window={size=1x1x1x1}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(PoplarAlgebraicSimplifier().Run(m.get()).ValueOrDie());
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(
                  m::ReduceWindow(m::Parameter(0), m::ConstantScalar(0)))));
  ASSERT_EQ(m->entry_computation()
                ->root_instruction()
                ->operand(0)
                ->called_computations()
                .size(),
            1);
  EXPECT_THAT(m->entry_computation()
                  ->root_instruction()
                  ->operand(0)
                  ->called_computations()[0]
                  ->root_instruction(),
              GmockMatch(m::Add(m::Parameter(0), m::Parameter(1))));
}

// Test folding of dynamic_slice(iota, index) -> clamp(index, 0, size-1)
TEST_F(PoplarAlgebraicSimplifierTest, DynamicSliceOfIota) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  %cst = s32[2]{0} constant({0, 1})
  %index = u32[] parameter(0)
  ROOT %dynamic-slice = s32[1]{0} dynamic-slice(s32[2]{0} %cst, u32[] %index),
                                  dynamic_slice_sizes={1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  VLOG(2) << "After rewrite \n" << module->ToString();

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Convert(m::Reshape(
                  m::Clamp(m::Constant(), m::Parameter(0), m::Constant())))));
}

TEST_F(PoplarAlgebraicSimplifierTest, ConstantToIota) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  %cst = s32[4] constant({0, 25, 50, 75})
  ROOT %s = s32[4] copy(s32[4] %cst)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  VLOG(2) << "After rewrite \n" << module->ToString();

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Multiply(m::Iota(), m::Broadcast())));
}

// Test folding of clamp(pid, 0, limit) -> pid
TEST_F(PoplarAlgebraicSimplifierTest, ClampOfPartitionId) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  %pid = u32[] partition-id()
  %low = u32[] constant(0)
  %high = u32[] constant(5)
  ROOT %c = u32[] clamp(%low, %pid, %high)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo_string, /*replica_count=*/1,
                                                /*num_partitions=*/6));
  PoplarAlgebraicSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  VLOG(2) << "After rewrite \n" << module->ToString();

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::PartitionId()));
}

// Test folding pad into maxpool with no existing padding.
TEST_F(PoplarAlgebraicSimplifierTest, FoldPadIntoMaxPool) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = f32[8, 64, 64, 16] parameter(0)
  zero = f32[] constant(0)
  pad = f32[8, 66, 66, 16] pad(param, zero), padding=0_0x1_1x1_1x0_0
  maxpool = f32[8, 32, 32, 16] custom-call(pad), custom_call_target="MaxPool", backend_config="{\n\t\"window\" : \"{\\\"dimensions\\\":[{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"4\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"4\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false}]}\"\n}"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(PoplarAlgebraicSimplifier().Run(module.get()).ValueOrDie());
  VLOG(2) << "After rewrite \n" << module->ToString();

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MaxPool)(root));
  EXPECT_THAT(root, GmockMatch(m::CustomCall(m::Parameter(0))));
  auto window = root->window();
  EXPECT_THAT(window.dimensions_size(), 4);

  EXPECT_THAT(window.dimensions(0).size(), 1);
  EXPECT_THAT(window.dimensions(1).size(), 4);
  EXPECT_THAT(window.dimensions(2).size(), 4);
  EXPECT_THAT(window.dimensions(3).size(), 1);

  EXPECT_THAT(window.dimensions(0).stride(), 1);
  EXPECT_THAT(window.dimensions(1).stride(), 2);
  EXPECT_THAT(window.dimensions(2).stride(), 2);
  EXPECT_THAT(window.dimensions(3).stride(), 1);

  EXPECT_THAT(window.dimensions(0).padding_low(), 0);
  EXPECT_THAT(window.dimensions(1).padding_low(), 1);
  EXPECT_THAT(window.dimensions(2).padding_low(), 1);
  EXPECT_THAT(window.dimensions(3).padding_low(), 0);

  EXPECT_THAT(window.dimensions(0).padding_high(), 0);
  EXPECT_THAT(window.dimensions(1).padding_high(), 1);
  EXPECT_THAT(window.dimensions(2).padding_high(), 1);
  EXPECT_THAT(window.dimensions(3).padding_high(), 0);
}

// Test folding pad into maxpool which already has padding.
TEST_F(PoplarAlgebraicSimplifierTest, FoldPadIntoMaxPoolWithPadding) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = f32[8, 64, 64, 16] parameter(0)
  zero = f32[] constant(0)
  pad = f32[8, 68, 68, 16] pad(param, zero), padding=0_0x2_2x2_2x0_0
  maxpool = f32[8, 28, 28, 16] custom-call(pad), custom_call_target="MaxPool", backend_config="{\n\t\"window\" : \"{\\\"dimensions\\\":[{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"4\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"2\\\",\\\"padding_high\\\":\\\"2\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"4\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"2\\\",\\\"padding_high\\\":\\\"2\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false}]}\"\n}"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(PoplarAlgebraicSimplifier().Run(module.get()).ValueOrDie());
  VLOG(2) << "After rewrite \n" << module->ToString();

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MaxPool)(root));
  EXPECT_THAT(root, GmockMatch(m::CustomCall(m::Parameter(0))));
  auto window = root->window();
  EXPECT_THAT(window.dimensions_size(), 4);

  EXPECT_THAT(window.dimensions(0).size(), 1);
  EXPECT_THAT(window.dimensions(1).size(), 4);
  EXPECT_THAT(window.dimensions(2).size(), 4);
  EXPECT_THAT(window.dimensions(3).size(), 1);

  EXPECT_THAT(window.dimensions(0).stride(), 1);
  EXPECT_THAT(window.dimensions(1).stride(), 2);
  EXPECT_THAT(window.dimensions(2).stride(), 2);
  EXPECT_THAT(window.dimensions(3).stride(), 1);

  EXPECT_THAT(window.dimensions(0).padding_low(), 0);
  EXPECT_THAT(window.dimensions(1).padding_low(), 4);
  EXPECT_THAT(window.dimensions(2).padding_low(), 4);
  EXPECT_THAT(window.dimensions(3).padding_low(), 0);

  EXPECT_THAT(window.dimensions(0).padding_high(), 0);
  EXPECT_THAT(window.dimensions(1).padding_high(), 4);
  EXPECT_THAT(window.dimensions(2).padding_high(), 4);
  EXPECT_THAT(window.dimensions(3).padding_high(), 0);
}

// Test folding incompatible pad (with interior padding) into maxpool.
TEST_F(PoplarAlgebraicSimplifierTest, FoldInteriorPadIntoMaxPool) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = f32[8, 32, 32, 16] parameter(0)
  zero = f32[] constant(0)
  pad = f32[8, 65, 65, 16] pad(param, zero), padding=0_0_0x1_1_1x1_1_1x0_0_0
  maxpool = f32[8, 32, 32, 16] custom-call(pad), custom_call_target="MaxPool", backend_config="{\n\t\"window\" : \"{\\\"dimensions\\\":[{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"2\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"2\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false}]}\"\n}"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_FALSE(PoplarAlgebraicSimplifier().Run(module.get()).ValueOrDie());
}

// Test folding pad into maxpoolgrad with no existing padding.
TEST_F(PoplarAlgebraicSimplifierTest, FoldPadIntoMaxPoolGrad) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = f32[8, 64, 64, 16] parameter(0)
  zero = f32[] constant(0)
  pad = f32[8, 66, 66, 16] pad(param, zero), padding=0_0x1_1x1_1x0_0
  maxpool = f32[8, 32, 32, 16] custom-call(pad), custom_call_target="MaxPool", backend_config="{\n\t\"window\" : \"{\\\"dimensions\\\":[{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"4\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"4\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false}]}\"\n}"
  maxpoolgrad = f32[8, 66, 66, 16] custom-call(pad, maxpool, param), custom_call_target="MaxPoolGrad", backend_config="{\n\t\"window\" : \"{\\\"dimensions\\\":[{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"4\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"4\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false}]}\"\n}"
  ROOT slice = f32[8, 64, 64, 16] slice(maxpoolgrad), slice={[0:8],[1:65], [1:65], [0:16]}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(PoplarAlgebraicSimplifier().Run(module.get()).ValueOrDie());
  VLOG(2) << "After rewrite \n" << module->ToString();

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MaxPoolGrad)(root));
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MaxPool)(root->operand(1)));
  EXPECT_THAT(root, GmockMatch(m::CustomCall(m::Parameter(0),
                                             m::CustomCall(m::Parameter(0)),
                                             m::Parameter(0))));

  auto window = root->window();
  EXPECT_THAT(window.dimensions_size(), 4);

  EXPECT_THAT(window.dimensions(0).size(), 1);
  EXPECT_THAT(window.dimensions(1).size(), 4);
  EXPECT_THAT(window.dimensions(2).size(), 4);
  EXPECT_THAT(window.dimensions(3).size(), 1);

  EXPECT_THAT(window.dimensions(0).stride(), 1);
  EXPECT_THAT(window.dimensions(1).stride(), 2);
  EXPECT_THAT(window.dimensions(2).stride(), 2);
  EXPECT_THAT(window.dimensions(3).stride(), 1);

  EXPECT_THAT(window.dimensions(0).padding_low(), 0);
  EXPECT_THAT(window.dimensions(1).padding_low(), 1);
  EXPECT_THAT(window.dimensions(2).padding_low(), 1);
  EXPECT_THAT(window.dimensions(3).padding_low(), 0);

  EXPECT_THAT(window.dimensions(0).padding_high(), 0);
  EXPECT_THAT(window.dimensions(1).padding_high(), 1);
  EXPECT_THAT(window.dimensions(2).padding_high(), 1);
  EXPECT_THAT(window.dimensions(3).padding_high(), 0);
}

// Test folding pad into maxpoolgrad which already has padding.
TEST_F(PoplarAlgebraicSimplifierTest, FoldPadIntoMaxPoolGradWithPadding) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = f32[8, 64, 64, 16] parameter(0)
  zero = f32[] constant(0)
  pad = f32[8, 68, 68, 16] pad(param, zero), padding=0_0x2_2x2_2x0_0
  maxpool = f32[8, 28, 28, 16] custom-call(pad), custom_call_target="MaxPool", backend_config="{\n\t\"window\" : \"{\\\"dimensions\\\":[{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"4\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"2\\\",\\\"padding_high\\\":\\\"2\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"4\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"2\\\",\\\"padding_high\\\":\\\"2\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false}]}\"\n}"
  maxpoolgrad = f32[8, 68, 68, 16] custom-call(pad, maxpool, param), custom_call_target="MaxPoolGrad", backend_config="{\n\t\"window\" : \"{\\\"dimensions\\\":[{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"4\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"2\\\",\\\"padding_high\\\":\\\"2\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"4\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"2\\\",\\\"padding_high\\\":\\\"2\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false}]}\"\n}"
  ROOT slice = f32[8, 64, 64, 16] slice(maxpoolgrad), slice={[0:8],[2:66], [2:66], [0:16]}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(PoplarAlgebraicSimplifier().Run(module.get()).ValueOrDie());
  VLOG(2) << "After rewrite \n" << module->ToString();

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MaxPoolGrad)(root));
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MaxPool)(root->operand(1)));
  EXPECT_THAT(root, GmockMatch(m::CustomCall(m::Parameter(0),
                                             m::CustomCall(m::Parameter(0)),
                                             m::Parameter(0))));

  auto window = root->window();
  EXPECT_THAT(window.dimensions_size(), 4);

  EXPECT_THAT(window.dimensions(0).size(), 1);
  EXPECT_THAT(window.dimensions(1).size(), 4);
  EXPECT_THAT(window.dimensions(2).size(), 4);
  EXPECT_THAT(window.dimensions(3).size(), 1);

  EXPECT_THAT(window.dimensions(0).stride(), 1);
  EXPECT_THAT(window.dimensions(1).stride(), 2);
  EXPECT_THAT(window.dimensions(2).stride(), 2);
  EXPECT_THAT(window.dimensions(3).stride(), 1);

  EXPECT_THAT(window.dimensions(0).padding_low(), 0);
  EXPECT_THAT(window.dimensions(1).padding_low(), 4);
  EXPECT_THAT(window.dimensions(2).padding_low(), 4);
  EXPECT_THAT(window.dimensions(3).padding_low(), 0);

  EXPECT_THAT(window.dimensions(0).padding_high(), 0);
  EXPECT_THAT(window.dimensions(1).padding_high(), 4);
  EXPECT_THAT(window.dimensions(2).padding_high(), 4);
  EXPECT_THAT(window.dimensions(3).padding_high(), 0);
}

// Test folding incompatible pad (with interior padding) into maxpoolgrad.
TEST_F(PoplarAlgebraicSimplifierTest, FoldInteriorPadIntoMaxPoolGrad) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = f32[8, 32, 32, 16] parameter(0)
  zero = f32[] constant(0)
  pad = f32[8, 65, 65, 16] pad(param, zero), padding=0_0_0x1_1_1x1_1_1x0_0_0
  maxpool = f32[8, 32, 32, 16] custom-call(pad), custom_call_target="MaxPool", backend_config="{\n\t\"window\" : \"{\\\"dimensions\\\":[{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"2\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"2\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false}]}\"\n}"
  maxpoolgrad = f32[8, 65, 65, 16] custom-call(pad, maxpool, param), custom_call_target="MaxPoolGrad", backend_config="{\n\t\"window\" : \"{\\\"dimensions\\\":[{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"2\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"2\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false}]}\"\n}"
  ROOT slice = f32[8, 32, 32, 16] slice(maxpoolgrad), slice={[0:8:1],[1:64:2], [1:64:2], [0:16:1]}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_FALSE(PoplarAlgebraicSimplifier().Run(module.get()).ValueOrDie());
}

// Test folding missmatched pad into maxpoolgrad.
TEST_F(PoplarAlgebraicSimplifierTest, FoldMissmatchedPadIntoMaxPoolGrad) {
  // The slice is not the inverse of the pad since it slices two elements off of
  // the end rather than one element off of each side.
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = f32[8, 64, 64, 16] parameter(0)
  zero = f32[] constant(0)
  pad = f32[8, 66, 66, 16] pad(param, zero), padding=0_0x1_1x1_1x0_0
  maxpool = f32[8, 32, 32, 16] custom-call(pad), custom_call_target="MaxPool", backend_config="{\n\t\"window\" : \"{\\\"dimensions\\\":[{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"4\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"4\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false}]}\"\n}"
  maxpoolgrad = f32[8, 66, 66, 16] custom-call(pad, maxpool, param), custom_call_target="MaxPoolGrad", backend_config="{\n\t\"window\" : \"{\\\"dimensions\\\":[{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"4\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"4\\\",\\\"stride\\\":\\\"2\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false},{\\\"size\\\":\\\"1\\\",\\\"stride\\\":\\\"1\\\",\\\"padding_low\\\":\\\"0\\\",\\\"padding_high\\\":\\\"0\\\",\\\"window_dilation\\\":\\\"1\\\",\\\"base_dilation\\\":\\\"1\\\",\\\"window_reversal\\\":false}]}\"\n}"
  ROOT slice = f32[8, 64, 64, 16] slice(maxpoolgrad), slice={[0:8],[0:64], [0:64], [0:16]}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(PoplarAlgebraicSimplifier().Run(module.get()).ValueOrDie());
  VLOG(2) << "After rewrite \n" << module->ToString();

  HloInstruction* root = module->entry_computation()->root_instruction();

  // The pad/slice were not folded into the maxpoolgrad.
  EXPECT_THAT(root, GmockMatch(m::Slice(m::CustomCall(
                        m::Pad(m::Parameter(0), m::ConstantScalar(0)),
                        m::CustomCall(m::Parameter(0)), m::Parameter(0)))));
}

// Test eliding two casts into one.
TEST_F(PoplarAlgebraicSimplifierTest, ElidingDoubleCast) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  param = u32[8, 8] parameter(0)
  cast1 = f32[8, 8] convert(param)
  cast2 = f16[8, 8] convert(cast1)
  dot = f16[8, 8] dot(cast2, cast2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(PoplarAlgebraicSimplifier().Run(module.get()).ValueOrDie());
  VLOG(2) << "After rewrite \n" << module->ToString();

  HloInstruction* root = module->entry_computation()->root_instruction();

  // The two casts are elided into one.
  EXPECT_THAT(root, GmockMatch(m::Dot(m::Convert(m::Parameter(0)),
                                      m::Convert(m::Parameter(0)))));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
