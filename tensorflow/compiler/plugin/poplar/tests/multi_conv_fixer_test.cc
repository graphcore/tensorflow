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

#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_conv_fixer.h"

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_conv.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {
using MultiConvFixerTest = HloTestBase;

TEST_F(MultiConvFixerTest, SingleConv) {
  std::string hlo = R"(
HloModule top

multi_conv {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[3,3,4,2] parameter(1)
  p1_t = f16[3,3,2,4] transpose(p1), dimensions={0, 1, 3, 2}
  ROOT conv = f16[1,16,16,4] convolution(p0, p1_t), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
}

ENTRY e {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[3,3,4,2] parameter(1)
  ROOT c = f16[1,16,16,4] call(p0, p1), to_apply=multi_conv, backend_config="{\"callConfig\":{\"type\":\"MultiConv\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  auto root = module->entry_computation()->root_instruction();
  auto attributes = root->frontend_attributes();
  auto* map = attributes.mutable_map();
  (*map)[FrontendAttributeId_Name(OPTION_FLAGS)] = "{}";
  root->set_frontend_attributes(attributes);

  TF_ASSERT_OK_AND_ASSIGN(bool changed, MultiConvFixer().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 4);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kConvolution);
}

TEST_F(MultiConvFixerTest, MultiConv) {
  std::string hlo = R"(
HloModule top

multi_conv {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[3,3,4,2] parameter(1)
  p1_t = f16[3,3,2,4] transpose(p1), dimensions={0, 1, 3, 2}
  conv1 = f16[1,16,16,4] convolution(p0, p1_t), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  p2 = f16[1,16,16,2] parameter(2)
  p3 = f16[3,3,2,4] parameter(3)
  conv2 = f16[1,16,16,4] convolution(p2, p3), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  ROOT add = f16[1,16,16,4] add(conv1, conv2)
}

ENTRY e {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[3,3,4,2] parameter(1)
  p2 = f16[1,16,16,2] parameter(2)
  p3 = f16[3,3,2,4] parameter(3)
  ROOT c = f16[1,16,16,4] call(p0, p1, p2, p3), to_apply=multi_conv, backend_config="{\"callConfig\":{\"type\":\"MultiConv\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  auto root = module->entry_computation()->root_instruction();
  auto attributes = root->frontend_attributes();
  auto* map = attributes.mutable_map();
  (*map)[FrontendAttributeId_Name(OPTION_FLAGS)] = "{}";
  root->set_frontend_attributes(attributes);

  TF_ASSERT_OK_AND_ASSIGN(bool changed, MultiConvFixer().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 9);
  root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(root->operand(0)->operand(0), root->operand(1)->operand(0));
  auto multi_conv = Cast<HloMultiConvInstruction>(root->operand(0)->operand(0));
  EXPECT_EQ(multi_conv->GetConvolutionSpecs().size(), 2);
  EXPECT_FALSE(multi_conv->IsWeightUpdate());
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
