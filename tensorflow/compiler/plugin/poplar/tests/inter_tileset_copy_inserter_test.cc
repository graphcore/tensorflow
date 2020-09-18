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

#include "tensorflow/compiler/plugin/poplar/driver/passes/inter_tileset_copy_inserter.h"

#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
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

using InterTilesetCopyInserterTest = HloTestBase;

TEST_F(InterTilesetCopyInserterTest, RemoteParameterLoadStore) {
  const auto hlo_string = R"(
HloModule top

ENTRY top {
  arg1 = f32[] parameter(0)
  arg2 = f32[] parameter(1)
  load = f32[] custom-call(arg1), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n"
  mul = f32[] multiply(load, arg2)
  store = f32[] custom-call(arg1, mul), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n"
  ROOT tuple = (f32[]) tuple(store)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  EXPECT_TRUE(IoTilesPlacer(true).Run(module).ValueOrDie());
  EXPECT_TRUE(InterTilesetCopyInserter().Run(module).ValueOrDie());

  const auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);

  const auto* store = root->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(RemoteParameterStore)(store));

  const auto* arg1 = store->operand(0);
  EXPECT_EQ(arg1->parameter_number(), 0);

  const auto* copy_to_store = store->operand(1);
  EXPECT_TRUE(IsPoplarInstruction(InterTilesetCopy)(copy_to_store));
  EXPECT_EQ(GetTileset(copy_to_store).ValueOrDie(), TILESET_IO_TILES);

  const auto* mul = copy_to_store->operand(0);
  const HloInstruction* copy_from_load;
  EXPECT_TRUE(Match(mul, m::Multiply(m::Op(&copy_from_load), m::Parameter(1))));
  EXPECT_TRUE(IsPoplarInstruction(InterTilesetCopy)(copy_from_load));
  EXPECT_EQ(GetTileset(copy_from_load).ValueOrDie(), TILESET_COMPUTE_TILES);

  const auto* load = copy_from_load->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(RemoteParameterLoad)(load));
  EXPECT_EQ(GetTileset(load).ValueOrDie(), TILESET_IO_TILES);
  EXPECT_EQ(load->operand(0)->parameter_number(), 0);
}

TEST_F(InterTilesetCopyInserterTest, BufferLoadStoreSlice) {
  const auto hlo_string = R"(
HloModule top

ENTRY top {
  buffer = f32[] parameter(0)
  offset = s32[] parameter(1)
  constant = f32[] constant(2)
  load = f32[] custom-call(buffer, offset), custom_call_target="BufferLoadSlice"
  mul = f32[] multiply(load, constant)
  store = f32[] custom-call(buffer, mul, offset), custom_call_target="BufferStoreSlice"
  ROOT tuple = (f32[]) tuple(store)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  EXPECT_TRUE(IoTilesPlacer(true).Run(module).ValueOrDie());
  EXPECT_TRUE(InterTilesetCopyInserter().Run(module).ValueOrDie());

  const auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);

  const auto* store = root->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(BufferStoreSlice)(store));

  const auto* buffer = store->operand(0);
  EXPECT_EQ(buffer->parameter_number(), 0);

  const auto* mul_copy = store->operand(1);
  EXPECT_TRUE(IsPoplarInstruction(InterTilesetCopy)(mul_copy));
  EXPECT_EQ(GetTileset(mul_copy).ValueOrDie(), TILESET_IO_TILES);

  const auto* mul = mul_copy->operand(0);
  EXPECT_EQ(mul->opcode(), HloOpcode::kMultiply);

  const auto* load_copy = mul->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(InterTilesetCopy)(load_copy));
  EXPECT_EQ(GetTileset(load_copy).ValueOrDie(), TILESET_COMPUTE_TILES);

  EXPECT_EQ(mul->operand(1)->opcode(), HloOpcode::kConstant);

  const auto* load = load_copy->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(BufferLoadSlice)(load));
  EXPECT_EQ(GetTileset(load).ValueOrDie(), TILESET_IO_TILES);
  EXPECT_EQ(load->operand(0)->parameter_number(), 0);

  const auto* offset_copy = store->operand(2);
  EXPECT_TRUE(IsPoplarInstruction(InterTilesetCopy)(offset_copy));
  EXPECT_EQ(GetTileset(offset_copy).ValueOrDie(), TILESET_IO_TILES);

  const auto* offset = offset_copy->operand(0);
  EXPECT_EQ(offset->parameter_number(), 1);
}

TEST_F(InterTilesetCopyInserterTest, InsertCopyForMultipleUsers) {
  const auto hlo_string = R"(
HloModule top

ENTRY top {
  after-all = token[] after-all()
  infeed = ((f16[]), token[]) infeed(after-all)
  gte = (f16[]) get-tuple-element(infeed), index=0
  gte0 = f16[] get-tuple-element(gte), index=0
  mul = f16[] multiply(gte0, gte0)
  ROOT mul2 = f16[] multiply(mul, gte0)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(IoTilesPlacer(true).Run(module).ValueOrDie());
  EXPECT_TRUE(InterTilesetCopyInserter().Run(module).ValueOrDie());

  const auto* mul2 = module->entry_computation()->root_instruction();
  CHECK_EQ(mul2->opcode(), HloOpcode::kMultiply);
  const auto* mul = mul2->operand(0);
  CHECK_EQ(mul->opcode(), HloOpcode::kMultiply);

  // All the operands need an inter-tileset-copy.
  const auto* copy = mul2->operand(1);
  EXPECT_TRUE(IsPoplarInstruction(InterTilesetCopy)(copy));

  const auto* copy1 = mul->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(InterTilesetCopy)(copy1));

  const auto* copy2 = mul->operand(1);
  EXPECT_TRUE(IsPoplarInstruction(InterTilesetCopy)(copy2));

  // In fact, they should all be the same copy.
  EXPECT_EQ(copy, copy1);
  EXPECT_EQ(copy, copy2);

  EXPECT_EQ(GetTileset(copy).ValueOrDie(), TILESET_COMPUTE_TILES);

  const auto* gte0 = copy->operand(0);
  EXPECT_EQ(GetTileset(gte0).ValueOrDie(), TILESET_IO_TILES);
  EXPECT_TRUE(
      Match(gte0, m::GetTupleElement(m::GetTupleElement(m::Infeed(), 0), 0)));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
