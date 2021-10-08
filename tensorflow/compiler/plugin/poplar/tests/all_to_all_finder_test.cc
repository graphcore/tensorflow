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
#include "tensorflow/compiler/plugin/poplar/driver/passes/all_to_all_finder.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/scatter_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using AllToAllFinderTest = HloTestBase;

TEST_F(AllToAllFinderTest, ReplaceMultiUpdate) {
  const string& hlo_string = R"(
HloModule main

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

ENTRY main {
  offsets = s32[24,1] parameter(0)
  updates = f32[24,16] parameter(1)
  zero = f32[] constant(0)
  big_zero = f32[1000,16] broadcast(zero), dimensions={}
  operand = f32[1000,16] custom-call(big_zero, offsets, updates), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
  operand_all = f32[1000,16] all-reduce(operand), to_apply=add
  ROOT operand_norm = f32[1000,16] custom-call(operand_all), custom_call_target="ReplicationNormalise", backend_config="{}\n"
}
)";
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());
  AllToAllFinder ataf(annotations, 4);
  EXPECT_TRUE(ataf.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdate)(root));
  auto* multi_update = Cast<HloMultiUpdateInstruction>(root);
  EXPECT_EQ(multi_update->GetSerializationFactor(), 1);
  HloInstruction* big_zero = FindInstruction(module, "big_zero");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  EXPECT_EQ(root->operand(0), big_zero);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kReshape);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::AllGather)(root->operand(1)->operand(0)));
  EXPECT_EQ(root->operand(1)->operand(0)->operand(0), offsets);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::ReplicationNormalise)(root->operand(2)));
  EXPECT_EQ(root->operand(2)->operand(0)->opcode(), HloOpcode::kReshape);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::AllGather)(
      root->operand(2)->operand(0)->operand(0)));
  EXPECT_EQ(root->operand(2)->operand(0)->operand(0)->operand(0), updates);
}

TEST_F(AllToAllFinderTest, ReplaceMultiUpdateSerialized) {
  const string& hlo_string = R"(
HloModule main

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

ENTRY main {
  offsets = s32[24,1] parameter(0)
  updates = f32[24,16] parameter(1)
  zero = f32[] constant(0)
  big_zero = f32[1000,16] broadcast(zero), dimensions={}
  operand = f32[1000,16] custom-call(big_zero, offsets, updates), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
  operand_all = f32[1000,16] all-reduce(operand), to_apply=add
  ROOT operand_norm = f32[1000,16] custom-call(operand_all), custom_call_target="ReplicationNormalise", backend_config="{}\n"
}
)";
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());
  AllToAllFinder ataf(annotations, 16);
  EXPECT_TRUE(ataf.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdate)(root));
  auto* multi_update = Cast<HloMultiUpdateInstruction>(root);
  EXPECT_EQ(multi_update->GetSerializationFactor(), 4);
  HloInstruction* big_zero = FindInstruction(module, "big_zero");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  EXPECT_EQ(root->operand(0), big_zero);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kReshape);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::AllGather)(root->operand(1)->operand(0)));
  EXPECT_EQ(root->operand(1)->operand(0)->operand(0), offsets);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::ReplicationNormalise)(root->operand(2)));
  EXPECT_EQ(root->operand(2)->operand(0)->opcode(), HloOpcode::kReshape);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::AllGather)(
      root->operand(2)->operand(0)->operand(0)));
  EXPECT_EQ(root->operand(2)->operand(0)->operand(0)->operand(0), updates);
}

TEST_F(AllToAllFinderTest, ReplaceMultiUpdateAdd) {
  const string& hlo_string = R"(
HloModule main

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

ENTRY main {
  offsets = s32[24,1] parameter(0)
  updates = f32[24,16] parameter(1)
  zero = f32[] constant(0)
  scale = f32[] constant(1)
  big_zero = f32[1000,16] broadcast(zero), dimensions={}
  operand = f32[1000,16] custom-call(big_zero, offsets, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  operand_all = f32[1000,16] all-reduce(operand), to_apply=add
  ROOT operand_norm = f32[1000,16] custom-call(operand_all), custom_call_target="ReplicationNormalise", backend_config="{}\n"
}
)";
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());
  AllToAllFinder ataf(annotations, 4);
  EXPECT_TRUE(ataf.Run(module).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(root));
  auto* multi_update = Cast<HloMultiUpdateInstruction>(root);
  EXPECT_EQ(multi_update->GetSerializationFactor(), 1);
  HloInstruction* big_zero = FindInstruction(module, "big_zero");
  HloInstruction* offsets = FindInstruction(module, "offsets");
  HloInstruction* updates = FindInstruction(module, "updates");
  HloInstruction* scale = FindInstruction(module, "scale");
  EXPECT_EQ(root->operand(0), big_zero);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kReshape);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::AllGather)(root->operand(1)->operand(0)));
  EXPECT_EQ(root->operand(1)->operand(0)->operand(0), offsets);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::ReplicationNormalise)(root->operand(2)));
  EXPECT_EQ(root->operand(2)->operand(0)->opcode(), HloOpcode::kReshape);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::AllGather)(
      root->operand(2)->operand(0)->operand(0)));
  EXPECT_EQ(root->operand(2)->operand(0)->operand(0)->operand(0), updates);
  EXPECT_EQ(root->operand(3), scale);
}

TEST_F(AllToAllFinderTest, ReplaceMultiUpdateAddTooBig) {
  const string& hlo_string = R"(
HloModule main

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

ENTRY main {
  offsets = s32[500,1] parameter(0)
  updates = f32[500,16] parameter(1)
  zero = f32[] constant(0)
  scale = f32[] constant(1)
  big_zero = f32[1000,16] broadcast(zero), dimensions={}
  operand = f32[1000,16] custom-call(big_zero, offsets, updates, scale), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
  operand_all = f32[1000,16] all-reduce(operand), to_apply=add
  ROOT operand_norm = f32[1000,16] custom-call(operand_all), custom_call_target="ReplicationNormalise", backend_config="{}\n"
}
)";
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());
  AllToAllFinder ataf(annotations, 4);
  EXPECT_FALSE(ataf.Run(module).ValueOrDie());
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
