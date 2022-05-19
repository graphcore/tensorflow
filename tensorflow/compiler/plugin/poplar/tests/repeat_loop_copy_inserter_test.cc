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

#include "tensorflow/compiler/plugin/poplar/driver/passes/repeat_loop_copy_inserter.h"

#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inter_tileset_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/io_tiles_placer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/inter_tileset_copy.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using RepeatLoopCopyInserterTest = HloTestBase;

int64_t resource = 0;

TEST_F(RepeatLoopCopyInserterTest, CopyIOOutput) {
  const auto hlo_string = R"(
HloModule top

loop_body (arg_tuple.0: s32[]) -> s32[] {
  tok = token[] after-all()
  a = s32[] parameter(0)
  b = s32[] constant(1)
  o1 = token[] outfeed(a, tok), outfeed_config="\010\001\022\005feed0\"\001\001(\001"
  ROOT c = s32[] add(a, b)
}

ENTRY entry () -> s32[] {
  arg.0 = s32[] parameter(0)
  ROOT call = s32[] call(arg.0), to_apply=loop_body, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  EXPECT_TRUE(
      IoTilesPlacer(true, 32, 0x4000, 0.5, resource).Run(module).ValueOrDie());
  EXPECT_TRUE(InterTilesetCopyInserter().Run(module).ValueOrDie());
  EXPECT_TRUE(RepeatLoopCopyInserter().Run(module).ValueOrDie());

  auto loop_body = module->GetComputationWithName("loop_body");
  auto a = loop_body->parameter_instruction(0);

  // Expect to only see add and copy as users of parameter `a`.
  absl::flat_hash_set<HloOpcode> valid_users = {HloOpcode::kAdd,
                                                HloOpcode::kCopy};
  for (auto user : a->users()) {
    EXPECT_TRUE(valid_users.contains(user->opcode()));
  }

  // Find the inter-tileset-copy.
  auto instructions = loop_body->instructions();
  auto itr =
      absl::c_find_if(instructions, IsPoplarInstruction(InterTilesetCopy));
  EXPECT_NE(itr, instructions.end());

  // Expect the inter-tileset-copy to be in the io tilset and have a single
  // user.
  auto inter_tileset_copy = *itr;
  EXPECT_EQ(inter_tileset_copy->user_count(), 1);
  EXPECT_EQ(GetTileset(inter_tileset_copy).ValueOrDie(), TILESET_IO_TILES);

  // Expect the inter-tileset-copy operand to be a regular copy on the compute
  // tiles.
  auto copy = inter_tileset_copy->operand(0);
  EXPECT_EQ(copy->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(GetTileset(copy).ValueOrDie(), TILESET_COMPUTE_TILES);

  // Expect the inter-tileset-copy user to be the outfeed on the compute tiles.
  auto outfeed = inter_tileset_copy->users().front();
  EXPECT_EQ(outfeed->opcode(), HloOpcode::kOutfeed);
  EXPECT_EQ(GetTileset(outfeed).ValueOrDie(), TILESET_IO_TILES);
}

TEST_F(RepeatLoopCopyInserterTest, NoCopyIOOutput) {
  const auto hlo_string = R"(
HloModule top

loop_body (arg_tuple.0: s32[]) -> s32[] {
  tok = token[] after-all()
  a = s32[] parameter(0)
  b = s32[] constant(1)
  ROOT c = s32[] add(a, b)
  o1 = token[] outfeed(c, tok), outfeed_config="\010\001\022\005feed0\"\001\001(\001"
}

ENTRY entry () -> s32[] {
  arg.0 = s32[] parameter(0)
  ROOT call = s32[] call(arg.0), to_apply=loop_body, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  EXPECT_TRUE(
      IoTilesPlacer(true, 32, 0x4000, 0.5, resource).Run(module).ValueOrDie());
  EXPECT_TRUE(InterTilesetCopyInserter().Run(module).ValueOrDie());

  // Don't expect this pass to change the module.
  EXPECT_FALSE(RepeatLoopCopyInserter().Run(module).ValueOrDie());

  auto loop_body = module->GetComputationWithName("loop_body");

  // Find the inter-tileset-copy.
  auto instructions = loop_body->instructions();
  auto itr =
      absl::c_find_if(instructions, IsPoplarInstruction(InterTilesetCopy));
  EXPECT_NE(itr, instructions.end());

  // Expect the inter-tileset-copy to be in the io tilset and have a single
  // user.
  auto inter_tileset_copy = *itr;
  EXPECT_EQ(inter_tileset_copy->user_count(), 1);
  EXPECT_EQ(GetTileset(inter_tileset_copy).ValueOrDie(), TILESET_IO_TILES);

  // Expect the inter-tileset-copy operand to be a regular copy on the compute
  // tiles.
  auto add = inter_tileset_copy->operand(0);
  EXPECT_EQ(add->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(GetTileset(add).ValueOrDie(), TILESET_COMPUTE_TILES);

  // Expect the inter-tileset-copy user to be the outfeed on the compute tiles.
  auto outfeed = inter_tileset_copy->users().front();
  EXPECT_EQ(outfeed->opcode(), HloOpcode::kOutfeed);
  EXPECT_EQ(GetTileset(outfeed).ValueOrDie(), TILESET_IO_TILES);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
