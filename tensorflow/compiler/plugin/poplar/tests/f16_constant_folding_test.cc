/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/f16_constant_folding.h"

#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using F16ConstantFoldingTest = HloTestBase;

TEST_F(F16ConstantFoldingTest, Test1F16Folding) {
  std::string hlo_string = R"(
HloModule top

ENTRY %cluster_1  {
  %c0 = f16[] constant(5.0)
  %c1 = f16[] constant(6.0)
  %p0 = f16[] parameter(0)
  %add = f16[] add(%c0, %c1)
  %div = f16[] divide(%add, %p0)
  ROOT %tuple = (f16[]) tuple(%div)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  F16ConstantFolding f16constant_folding;
  EXPECT_TRUE(f16constant_folding.Run(module).ValueOrDie());

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module));
  EXPECT_TRUE(result);

  const auto* root = module->entry_computation()->root_instruction();
  const auto* i_div = root->operand(0);
  const auto* i_const = i_div->operand(0);
  const auto* i_parameter = i_div->operand(1);

  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(i_div->opcode(), HloOpcode::kDivide);
  EXPECT_EQ(i_const->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(i_parameter->opcode(), HloOpcode::kParameter);
  EXPECT_TRUE(i_const->literal().IsAll(11));
}

TEST_F(F16ConstantFoldingTest, Test2F16Folding) {
  std::string hlo_string = R"(
HloModule top

ENTRY %cluster_2  {
  %ca = f16[] constant(256)
  %cb = f16[] constant(32)
  %x = f16[] multiply(%ca, %ca)
  %y = f16[] divide(%x, %cb)
  ROOT %tuple = (f16[]) tuple(%y)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  F16ConstantFolding f16constant_folding;
  EXPECT_TRUE(f16constant_folding.Run(module).ValueOrDie());

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module));
  EXPECT_TRUE(result);

  const auto* root = module->entry_computation()->root_instruction();
  const auto* i_const = root->operand(0);

  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(i_const->opcode(), HloOpcode::kConstant);
  EXPECT_TRUE(i_const->literal().IsAllFloat(2048));
}

TEST_F(F16ConstantFoldingTest, Test3F16Folding) {
  std::string hlo_string = R"(
HloModule top

ENTRY %cluster_3  {
  %ca = f16[] constant(900)
  %cb = f16[] constant(100)
  %cc = f16[] constant(60000)
  %numerator = f16[] multiply(%ca, %cb)
  %denominator = f16[] subtract(%numerator, %cc)
  %result = f16[] divide(%numerator, %denominator)
  ROOT %tuple = (f16[]) tuple(%result)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  F16ConstantFolding f16constant_folding;
  EXPECT_TRUE(f16constant_folding.Run(module).ValueOrDie());

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module));
  EXPECT_TRUE(result);

  const auto* root = module->entry_computation()->root_instruction();
  const auto* i_const = root->operand(0);

  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(i_const->opcode(), HloOpcode::kConstant);
  EXPECT_TRUE(i_const->literal().IsAll(3));
}

TEST_F(F16ConstantFoldingTest, Test4F16Folding) {
  std::string hlo_string = R"(
HloModule top

ENTRY %cluster_4  {
  %c0 = f16[] constant(5.0)
  %c1 = f16[] constant(6.0)
  %p0 = f16[] parameter(0)
  %add = f16[] add(%c0, %c1)
  ROOT %div = f16[] divide(%add, %p0)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  F16ConstantFolding f16constant_folding;
  EXPECT_TRUE(f16constant_folding.Run(module).ValueOrDie());

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module));
  EXPECT_TRUE(result);

  const auto* root = module->entry_computation()->root_instruction();
  const auto* i_div = root;
  const auto* i_const = i_div->operand(0);
  const auto* i_parameter = i_div->operand(1);

  EXPECT_EQ(i_div->opcode(), HloOpcode::kDivide);
  EXPECT_EQ(i_const->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(i_parameter->opcode(), HloOpcode::kParameter);
  EXPECT_TRUE(i_const->literal().IsAll(11));
}

// test root in DAG
TEST_F(F16ConstantFoldingTest, Test5F16Folding) {
  std::string hlo_string = R"(
HloModule top

ENTRY %cluster_5  {
  %ca = f16[] constant(256)
  %cb = f16[] constant(32)
  %x = f16[] multiply(%ca, %ca)
  ROOT %y = f16[] divide(%x, %cb)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  F16ConstantFolding f16constant_folding;
  EXPECT_TRUE(f16constant_folding.Run(module).ValueOrDie());

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module));
  EXPECT_TRUE(result);

  const auto* root = module->entry_computation()->root_instruction();

  EXPECT_TRUE(root->literal().IsAllFloat(2048));
  EXPECT_EQ(root->opcode(), HloOpcode::kConstant);
}

TEST_F(F16ConstantFoldingTest, Test6F16Folding) {
  std::string hlo_string = R"(
HloModule top

ENTRY %cluster_6  {
  %ca = f16[] constant(900)
  %cb = f16[] constant(100)
  %cc = f16[] constant(60000)
  %numerator = f16[] multiply(%ca, %cb)
  %denominator = f16[] subtract(%numerator, %cc)
  ROOT %result = f16[] divide(%numerator, %denominator)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  F16ConstantFolding f16constant_folding;
  EXPECT_TRUE(f16constant_folding.Run(module).ValueOrDie());

  HloConstantFolding const_folder;
  TF_ASSERT_OK_AND_ASSIGN(bool result, const_folder.Run(module));
  EXPECT_TRUE(result);

  const auto* root = module->entry_computation()->root_instruction();

  EXPECT_TRUE(root->literal().IsAllFloat(3));
  EXPECT_EQ(root->opcode(), HloOpcode::kConstant);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
