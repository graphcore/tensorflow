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

#include "tensorflow/compiler/plugin/poplar/driver/poplar_passes/map_hlo_instruction_to_debug_id.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using MapHloInstructionToDebugId = HloTestBase;

TEST_F(MapHloInstructionToDebugId, TestMapping) {
  std::string hlo_string = R"(
HloModule top

adder {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  ROOT add = f16[] add(a0, a1)
}

cluster_1  {
  arg0 = f16[] parameter(0)
  arg1 = f16[] parameter(1)
  arg2 = f16[] parameter(2)
  c1 = f16[] call(arg0, arg1), to_apply=adder
  c2 = f16[] call(arg0, arg2), to_apply=adder
  ROOT %tuple = (f16[], f16[]) tuple(c1, c2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  absl::flat_hash_map<const HloInstruction*, std::uint64_t> map;
  MapHloInstructionToDebugIdPass mapping(map);

  bool result;
  TF_ASSERT_OK_AND_ASSIGN(result, mapping.Run(module));
  EXPECT_TRUE(result);

  EXPECT_EQ(map.size(), 9);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
