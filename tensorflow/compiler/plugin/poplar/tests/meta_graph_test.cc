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

#include "tensorflow/compiler/plugin/poplar/driver/tools/meta_graph.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using MetaGraphTest = HloTestBase;

TEST_F(MetaGraphTest, ShortestPath) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
    src = f32[12,24] parameter(0)
    pathA = f32[24,12] transpose(f32[12,24] %src), dimensions={1,0}
    pathA1 = f32[12,24] transpose(f32[24,12] %pathA), dimensions={1,0}
    pathA2 = f32[24,12] transpose(f32[12,24] %pathA1), dimensions={1,0}

    pathB = f32[24,12] transpose(f32[12,24] %src), dimensions={1,0}

    dst = f32[24,12] multiply(f32[24,12] %pathA2, f32[24,12] %pathB)
}
)";
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  // Get the MetaGraph for the entry computation
  const auto get_operands = [](HloInstruction* inst) {
    return inst->operands();
  };
  HloInstruction* root = module->entry_computation()->root_instruction();
  MetaGraph<HloInstruction*, HloPtrComparator> meta_graph(root, get_operands);

  // Get shortest paths from src
  HloInstruction* src = FindInstruction(module, "src");
  auto shortest_paths_from_src = meta_graph.ShortestPathsFrom(src);

  // Get shortest path from src to dst
  HloInstruction* dst = FindInstruction(module, "dst");
  auto optional_path_to_dst = shortest_paths_from_src.To(dst);
  EXPECT_TRUE(optional_path_to_dst);
  auto path_to_dst = *optional_path_to_dst;

  // Verify it goes through the shorter path B
  bool via_B = false;
  for (HloInstruction* instruction : path_to_dst) {
    if (instruction->name() == "pathB") {
      via_B = true;
    }
  }
  EXPECT_TRUE(via_B);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
