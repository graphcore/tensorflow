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

#include <poplar/Device.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/replication_factor.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_base.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/entry_visitor.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using ExecutionCounterUtilTest = HloPoplarTestBase;

TEST_F(ExecutionCounterUtilTest, TestNoShards) {
  const string hlo_text = R"(
HloModule dummy
ENTRY main {
  ROOT operand = s32[] parameter(0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  auto resources = GetMockResources(module.get(), 1);
  ExecutionCounters counters(*resources.get(), {});
  EXPECT_FALSE(counters.Initialized());
  // Put the counters on the stack.
  resources->execution_counter_scopes.push(&counters);
  // Create new counters.
  ExecutionCounters sub_counters(*resources.get(), {});
  // Mark one counter as used.
  TF_EXPECT_OK(sub_counters.GetCounter(0).status());
  EXPECT_THAT(sub_counters.GetLiveCounters(), ::testing::ElementsAre(true));
  EXPECT_THAT(counters.GetLiveCounters(), ::testing::ElementsAre(false));
  // Populate the counters from the outer scope.
  DriverProgramSequence seq(*resources->main_graph);
  TF_EXPECT_OK(CopyExecutionCountersFromScope(
      *resources->main_graph, *resources.get(), sub_counters, seq));
  // Check that the counter has now been marked as live.
  EXPECT_EQ(counters.GetLiveCounters(), sub_counters.GetLiveCounters());
}

TEST_F(ExecutionCounterUtilTest, TestSharded) {
  const string hlo_text = R"(
HloModule dummy
ENTRY main {
  ROOT operand = s32[] parameter(0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  const int32 num_shards = 4;
  auto device = CreateIpuModel(num_shards);
  auto resources = GetMockResources(device, module.get(), false, num_shards);

  ExecutionCounters counters(*resources.get(), {});
  EXPECT_FALSE(counters.Initialized());
  // Put the counters on the stack.
  resources->execution_counter_scopes.push(&counters);
  // Create new counters.
  ExecutionCounters sub_counters(*resources.get(), {});
  // Mark counter as used.
  TF_EXPECT_OK(sub_counters.GetCounter(0).status());
  TF_EXPECT_OK(sub_counters.GetCounter(2).status());
  EXPECT_THAT(sub_counters.GetLiveCounters(),
              ::testing::ElementsAre(true, false, true, false));
  EXPECT_THAT(counters.GetLiveCounters(),
              ::testing::ElementsAre(false, false, false, false));
  // Populate the counters from the outer scope.
  DriverProgramSequence seq(*resources->main_graph);
  TF_EXPECT_OK(CopyExecutionCountersFromScope(
      *resources->main_graph, *resources.get(), sub_counters, seq));
  // Check that the counter has now been marked as live.
  EXPECT_THAT(counters.GetLiveCounters(), sub_counters.GetLiveCounters());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
