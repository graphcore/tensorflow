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

#include "tensorflow/compiler/plugin/poplar/driver/passes/io_tiles_placer.h"

#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/parse_poplar_backend_config.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using IoTilesPlacerTest = HloTestBase;

TEST_F(IoTilesPlacerTest, TestRemoteParameterLoad) {
  const auto hlo_string = R"(
HloModule top

ENTRY top {
  arg1 = f32[] parameter(0)
  arg2 = f32[] parameter(1)
  load = f32[] custom-call(arg1), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n"
  ROOT mul = f32[] multiply(load, arg2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());

  EXPECT_FALSE(IoTilesPlacer(/*enabled=*/false).Run(module).ValueOrDie());

  EXPECT_TRUE(IoTilesPlacer(/*enabled=*/true).Run(module).ValueOrDie());

  const auto* mul = module->entry_computation()->root_instruction();
  const auto* load = mul->operand(0);

  EXPECT_EQ(GetTileset(mul).ValueOrDie(), TILESET_COMPUTE_TILES);
  EXPECT_EQ(GetTileset(load).ValueOrDie(), TILESET_IO_TILES);
}

TEST_F(IoTilesPlacerTest, TestInfeedGtes) {
  const auto hlo_string = R"(
HloModule top

ENTRY top {
  after-all = token[] after-all()
  infeed = ((f16[]), token[]) infeed(after-all)
  gte = (f16[]) get-tuple-element(infeed), index=0
  gte0 = f16[] get-tuple-element(gte), index=0
  ROOT mul = f16[] multiply(gte0, gte0)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(IoTilesPlacer(/*enabled=*/true).Run(module).ValueOrDie());

  const auto* mul = module->entry_computation()->root_instruction();
  const auto* gte0 = mul->operand(0);
  const auto* gte = gte0->operand(0);

  EXPECT_EQ(GetTileset(mul).ValueOrDie(), TILESET_COMPUTE_TILES);
  EXPECT_EQ(GetTileset(gte0).ValueOrDie(), TILESET_IO_TILES);
  EXPECT_EQ(GetTileset(gte).ValueOrDie(), TILESET_IO_TILES);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
