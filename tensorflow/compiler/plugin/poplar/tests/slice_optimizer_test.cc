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

#include "tensorflow/compiler/plugin/poplar/driver/passes/slice_optimizer.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/slice_apply.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using SliceOptimizerTest = HloTestBase;

TEST_F(SliceOptimizerTest, SliceApply) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f32[1,16,16,4] parameter(0)
  p1 = f32[1,16,16,4] parameter(1)
  p2 = f32[2,16,16,4] parameter(2)
  p3 = f32[1,16,16,4] parameter(3)
  p4 = f32[5,16,16,4] parameter(4)
  a = f32[5,16,16,4] concatenate(p0, p1, p2, p3), dimensions={0}
  ROOT %add = f32[5,16,16,4] add(p4, a)
}
)";

  auto config = GetModuleConfigForTest();
  config.set_resource_update_to_input_index({0});
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  SliceOptimizer so(annotations);
  EXPECT_TRUE(so.Run(module0).ValueOrDie());

  HloInstruction* input = module0->entry_computation()->root_instruction();
  int64 end_index = 5;
  for (int64 i = 0; i != 4; ++i) {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApply)(input));
    HloInstruction* lhs = input->mutable_operand(0);
    const HloInstruction* rhs = input->operand(1);

    // Check the instruction parameters.
    auto* casted = Cast<HloSliceApply>(input);
    EXPECT_EQ(casted->GetOperation(), HloOpcode::kAdd);
    EXPECT_EQ(casted->GetApplyDimension(), 0);
    int64 slice_size = rhs->shape().dimensions(casted->GetApplyDimension());
    EXPECT_EQ(casted->GetStartIndex(), end_index - slice_size);

    // Check the instruction operands.
    EXPECT_EQ(rhs->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(rhs->parameter_number(), 3 - i);

    // Go up the chain.
    input = lhs;
    end_index -= slice_size;
  }
  EXPECT_EQ(end_index, 0);
  EXPECT_EQ(input->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(input->parameter_number(), 4);

  {
    // Excute and compare to CPU.
    auto module_to_execut = ParseAndReturnVerifiedModule(hlo, config);
    auto p0 = LiteralUtil::CreateFullWithDescendingLayout({1, 16, 16, 4}, 1.0f);
    auto p1 = LiteralUtil::CreateFullWithDescendingLayout({1, 16, 16, 4}, 2.0f);
    auto p2 = LiteralUtil::CreateFullWithDescendingLayout({2, 16, 16, 4}, 3.0f);
    auto p3 = LiteralUtil::CreateFullWithDescendingLayout({1, 16, 16, 4}, 4.0f);
    auto p4 = LiteralUtil::CreateFullWithDescendingLayout({5, 16, 16, 4}, 5.0f);
    std::vector<Literal*> inputs = {&p0, &p1, &p2, &p3, &p4};
    EXPECT_TRUE(RunAndCompare(std::move(module_to_execut.ValueOrDie()), inputs,
                              ErrorSpec{1e-4, 1e-4}));
  }
}

TEST_F(SliceOptimizerTest, SliceApplyabY) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f32[1,16,1,4] parameter(0)
  p1 = f32[1,16,5,4] parameter(1)
  p2 = f32[1,16,5,4] parameter(2)
  p3 = f32[1,16,5,4] parameter(3)
  p4 = f32[1,16,16,4] parameter(4)
  a = f32[1,16,16,4] concatenate(p0, p1, p2, p3), dimensions={2}
  c = f32[] constant(0.1)
  bcast = f32[1,16,16,4] broadcast(c), dimensions={}
  ac = f32[1,16,16,4] multiply(a, bcast)
  ROOT %sub = f32[1,16,16,4] subtract(p4, ac)
}
)";

  auto config = GetModuleConfigForTest();
  config.set_resource_update_to_input_index({0});
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  SliceOptimizer so(annotations);
  EXPECT_TRUE(so.Run(module0).ValueOrDie());

  const HloInstruction* c = FindInstruction(module0, "c");
  HloInstruction* input = module0->entry_computation()->root_instruction();
  int64 end_index = 16;
  for (int64 i = 0; i != 4; ++i) {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyabY)(input));
    HloInstruction* lhs = input->mutable_operand(0);
    const HloInstruction* rhs = input->operand(1);

    // Check the instruction parameters.
    auto* casted = Cast<HloSliceApplyabY>(input);
    EXPECT_EQ(casted->GetOperation(), HloOpcode::kSubtract);
    EXPECT_EQ(casted->GetApplyDimension(), 2);
    int64 slice_size = rhs->shape().dimensions(casted->GetApplyDimension());
    EXPECT_EQ(casted->GetStartIndex(), end_index - slice_size);

    // Check the instruction operands.
    EXPECT_EQ(rhs->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(rhs->parameter_number(), 3 - i);
    EXPECT_EQ(input->operand(2), c);

    // Go up the chain.
    input = lhs;
    end_index -= slice_size;
  }
  EXPECT_EQ(end_index, 0);
  EXPECT_EQ(input->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(input->parameter_number(), 4);

  {
    // Excute and compare to CPU.
    auto module_to_execut = ParseAndReturnVerifiedModule(hlo, config);
    auto p0 = LiteralUtil::CreateFullWithDescendingLayout({1, 16, 1, 4}, 1.0f);
    auto p1 = LiteralUtil::CreateFullWithDescendingLayout({1, 16, 5, 4}, 2.0f);
    auto p2 = LiteralUtil::CreateFullWithDescendingLayout({1, 16, 5, 4}, 3.0f);
    auto p3 = LiteralUtil::CreateFullWithDescendingLayout({1, 16, 5, 4}, 4.0f);
    auto p4 = LiteralUtil::CreateFullWithDescendingLayout({1, 16, 16, 4}, 5.0f);
    std::vector<Literal*> inputs = {&p0, &p1, &p2, &p3, &p4};
    EXPECT_TRUE(RunAndCompare(std::move(module_to_execut.ValueOrDie()), inputs,
                              ErrorSpec{1e-4, 1e-4}));
  }
}

TEST_F(SliceOptimizerTest, SliceApplyaXb) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f32[1,16,16,1] parameter(0)
  p1 = f32[1,16,16,1] parameter(1)
  p2 = f32[1,16,16,1] parameter(2)
  p3 = f32[1,16,16,1] parameter(3)
  p4 = f32[1,16,16,4] parameter(4)
  a = f32[1,16,16,4] concatenate(p0, p1, p2, p3), dimensions={3}
  c = f32[] constant(0.1)
  bcast = f32[1,16,16,4] broadcast(c), dimensions={}
  p4c = f32[1,16,16,4] multiply(p4, bcast)
  ROOT %sub = f32[1,16,16,4] subtract(p4c, a)
}
)";

  auto config = GetModuleConfigForTest();
  config.set_resource_update_to_input_index({0});
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  SliceOptimizer so(annotations);
  EXPECT_TRUE(so.Run(module0).ValueOrDie());

  const HloInstruction* c = FindInstruction(module0, "c");
  HloInstruction* input = module0->entry_computation()->root_instruction();
  int64 end_index = 4;
  for (int64 i = 0; i != 4; ++i) {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyaXb)(input));
    HloInstruction* lhs = input->mutable_operand(0);
    const HloInstruction* rhs = input->operand(1);

    // Check the instruction parameters.
    auto* casted = Cast<HloSliceApplyaXb>(input);
    EXPECT_EQ(casted->GetOperation(), HloOpcode::kSubtract);
    EXPECT_EQ(casted->GetApplyDimension(), 3);
    int64 slice_size = rhs->shape().dimensions(casted->GetApplyDimension());
    EXPECT_EQ(casted->GetStartIndex(), end_index - slice_size);

    // Check the instruction operands.
    EXPECT_EQ(rhs->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(rhs->parameter_number(), 3 - i);
    EXPECT_EQ(input->operand(2), c);

    // Go up the chain.
    input = lhs;
    end_index -= slice_size;
  }
  EXPECT_EQ(end_index, 0);
  EXPECT_EQ(input->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(input->parameter_number(), 4);

  {
    // Excute and compare to CPU.
    auto module_to_execut = ParseAndReturnVerifiedModule(hlo, config);
    auto p0 = LiteralUtil::CreateFullWithDescendingLayout({1, 16, 16, 1}, 1.0f);
    auto p1 = LiteralUtil::CreateFullWithDescendingLayout({1, 16, 16, 1}, 2.0f);
    auto p2 = LiteralUtil::CreateFullWithDescendingLayout({1, 16, 16, 1}, 3.0f);
    auto p3 = LiteralUtil::CreateFullWithDescendingLayout({1, 16, 16, 1}, 4.0f);
    auto p4 = LiteralUtil::CreateFullWithDescendingLayout({1, 16, 16, 4}, 5.0f);
    std::vector<Literal*> inputs = {&p0, &p1, &p2, &p3, &p4};
    EXPECT_TRUE(RunAndCompare(std::move(module_to_execut.ValueOrDie()), inputs,
                              ErrorSpec{1e-4, 1e-4}));
  }
}

TEST_F(SliceOptimizerTest, SliceApplyaXbY) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f32[1,16,16,1] parameter(0)
  p1 = f32[1,16,16,1] parameter(1)
  p2 = f32[1,16,16,1] parameter(2)
  p3 = f32[1,16,16,1] parameter(3)
  p4 = f32[1,16,16,4] parameter(4)
  a = f32[1,16,16,4] concatenate(p0, p1, p2, p3), dimensions={3}
  c1 = f32[] constant(0.1)
  c2 = f32[] constant(0.2)
  bcast1 = f32[1,16,16,4] broadcast(c1), dimensions={}
  bcast2 = f32[1,16,16,4] broadcast(c2), dimensions={}
  p4c1 = f32[1,16,16,4] multiply(p4, bcast1)
  ac2 = f32[1,16,16,4] multiply(a, bcast2)
  ROOT %sub = f32[1,16,16,4] subtract(p4c1, ac2)
}
)";

  auto config = GetModuleConfigForTest();
  config.set_resource_update_to_input_index({0});
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  SliceOptimizer so(annotations);
  EXPECT_TRUE(so.Run(module0).ValueOrDie());

  const HloInstruction* c1 = FindInstruction(module0, "c1");
  const HloInstruction* c2 = FindInstruction(module0, "c2");
  HloInstruction* input = module0->entry_computation()->root_instruction();
  int64 end_index = 4;
  for (int64 i = 0; i != 4; ++i) {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyaXbY)(input));
    HloInstruction* lhs = input->mutable_operand(0);
    const HloInstruction* rhs = input->operand(1);

    // Check the instruction parameters.
    auto* casted = Cast<HloSliceApplyaXbY>(input);
    EXPECT_EQ(casted->GetOperation(), HloOpcode::kSubtract);
    EXPECT_EQ(casted->GetApplyDimension(), 3);
    int64 slice_size = rhs->shape().dimensions(casted->GetApplyDimension());
    EXPECT_EQ(casted->GetStartIndex(), end_index - slice_size);

    // Check the instruction operands.
    EXPECT_EQ(rhs->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(rhs->parameter_number(), 3 - i);
    EXPECT_EQ(input->operand(2), c1);
    EXPECT_EQ(input->operand(3), c2);

    // Go up the chain.
    input = lhs;
    end_index -= slice_size;
  }
  EXPECT_EQ(end_index, 0);
  EXPECT_EQ(input->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(input->parameter_number(), 4);

  {
    // Excute and compare to CPU.
    auto module_to_execut = ParseAndReturnVerifiedModule(hlo, config);
    auto p0 = LiteralUtil::CreateFullWithDescendingLayout({1, 16, 16, 1}, 1.0f);
    auto p1 = LiteralUtil::CreateFullWithDescendingLayout({1, 16, 16, 1}, 2.0f);
    auto p2 = LiteralUtil::CreateFullWithDescendingLayout({1, 16, 16, 1}, 3.0f);
    auto p3 = LiteralUtil::CreateFullWithDescendingLayout({1, 16, 16, 1}, 4.0f);
    auto p4 = LiteralUtil::CreateFullWithDescendingLayout({1, 16, 16, 4}, 5.0f);
    std::vector<Literal*> inputs = {&p0, &p1, &p2, &p3, &p4};
    EXPECT_TRUE(RunAndCompare(std::move(module_to_execut.ValueOrDie()), inputs,
                              ErrorSpec{1e-4, 1e-4}));
  }
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
