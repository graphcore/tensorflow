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

#include "tensorflow/compiler/plugin/poplar/driver/passes/host_compute_schedule_optimizer.h"

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/recv_from_host.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using HostComputeScheduleOptimizerTest = HloTestBase;

TEST_F(HostComputeScheduleOptimizerTest,
       TestScheduleOneSendOneRecvInEachShard) {
  std::string hlo_string = R"(
HloModule top

ENTRY %top (arg: f32[]) -> (f32[], f32[]) {
  %arg = f32[] parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %send1 = () custom-call(f32[] %arg), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"send1_key\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}, sharding={maximal device=0}
  %send2 = () custom-call(f32[] %arg), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"send2_key\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}, sharding={maximal device=1}
  %recv1 = f32[] custom-call(), custom_call_target="RecvFromHost", backend_config="{\"rendezvous_key\":\"recv1_key\"}", metadata={op_type="XlaHostCompute" op_name="host_compute"}, sharding={maximal device=0}
  %recv2 = f32[] custom-call(), custom_call_target="RecvFromHost", backend_config="{\"rendezvous_key\":\"recv2_key\"}", metadata={op_type="XlaHostCompute" op_name="host_compute"}, sharding={maximal device=1}
  ROOT %tuple = (f32[], f32[]) tuple(%recv1, %recv2)
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ASSERT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());

  ASSERT_TRUE(HostComputeScheduleOptimizer().Run(module).ValueOrDie());

  int num_insts = 0;

  for (HloInstruction* inst : comp->instructions()) {
    if (IsPoplarInstruction(SendToHost)(inst) ||
        IsPoplarInstruction(RecvFromHost)(inst)) {
      ++num_insts;
      EXPECT_TRUE(inst->has_sharding());
      if (inst->sharding_unique_device().value() == 0) {
        // First shard has no predecessor.
        EXPECT_EQ(inst->control_predecessors().size(), 0);
        EXPECT_EQ(inst->control_successors().size(), 1);
        EXPECT_EQ(inst->custom_call_target(),
                  inst->control_successors()[0]->custom_call_target());
      } else {
        // Second shard has first shard as predecessor.
        EXPECT_EQ(inst->sharding_unique_device().value(), 1);
        EXPECT_EQ(inst->control_predecessors().size(), 1);
        EXPECT_EQ(inst->control_successors().size(), 0);
        EXPECT_EQ(inst->custom_call_target(),
                  inst->control_predecessors()[0]->custom_call_target());
      }
    }
  }

  EXPECT_EQ(num_insts, 4);
}

TEST_F(HostComputeScheduleOptimizerTest, TestScheduleTwoSendsInEachShard) {
  std::string hlo_string = R"(
HloModule top

ENTRY %top (arg1: f32[], arg2: f32[], arg3: f32[], arg4: f32[]) -> () {
  %arg1 = f32[] parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg2 = f32[] parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg3 = f32[] parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg4 = f32[] parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %send1 = () custom-call(f32[] %arg1), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"send1_key\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}, sharding={maximal device=0}
  %send2 = () custom-call(f32[] %arg2), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"send2_key\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}, sharding={maximal device=0}
  %send3 = () custom-call(f32[] %arg3), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"send3_key\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}, sharding={maximal device=1}
  %send4 = () custom-call(f32[] %arg4), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"send4_key\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}, sharding={maximal device=1}
  ROOT %tuple = () tuple()
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ASSERT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());

  ASSERT_TRUE(HostComputeScheduleOptimizer().Run(module).ValueOrDie());

  int num_insts = 0;

  for (HloInstruction* inst : comp->instructions()) {
    if (IsPoplarInstruction(SendToHost)(inst)) {
      ++num_insts;
      EXPECT_TRUE(inst->has_sharding());
      if (inst->sharding_unique_device().value() == 0) {
        // First shard has no predecessor.
        EXPECT_EQ(inst->control_predecessors().size(), 0);
        EXPECT_EQ(inst->control_successors().size(), 2);
      } else {
        // Second shard has first shard as predecessor.
        EXPECT_EQ(inst->sharding_unique_device().value(), 1);
        EXPECT_EQ(inst->control_predecessors().size(), 2);
        EXPECT_EQ(inst->control_successors().size(), 0);
      }
    }
  }

  EXPECT_EQ(num_insts, 4);
}

TEST_F(HostComputeScheduleOptimizerTest, TestNoSharding) {
  std::string hlo_string = R"(
HloModule top

ENTRY %top (arg: f32[]) -> (f32[], f32[]) {
  %arg = f32[] parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %send1 = () custom-call(f32[] %arg), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"send1_key\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}
  %send2 = () custom-call(f32[] %arg), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"send2_key\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}
  %recv1 = f32[] custom-call(), custom_call_target="RecvFromHost", backend_config="{\"rendezvous_key\":\"recv1_key\"}", metadata={op_type="XlaHostCompute" op_name="host_compute"}
  %recv2 = f32[] custom-call(), custom_call_target="RecvFromHost", backend_config="{\"rendezvous_key\":\"recv2_key\"}", metadata={op_type="XlaHostCompute" op_name="host_compute"}
  ROOT %tuple = (f32[], f32[]) tuple(%recv1, %recv2)
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ASSERT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());

  ASSERT_FALSE(HostComputeScheduleOptimizer().Run(module).ValueOrDie());
}

TEST_F(HostComputeScheduleOptimizerTest, TestScheduleOneInstruction) {
  std::string hlo_string = R"(
HloModule top

ENTRY %top () -> f32[] {
  ROOT %recv = f32[] custom-call(), custom_call_target="RecvFromHost", backend_config="{\"rendezvous_key\":\"recv_key\"}", metadata={op_type="XlaHostCompute" op_name="host_compute"}, sharding={maximal device=0}
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ASSERT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());

  ASSERT_FALSE(HostComputeScheduleOptimizer().Run(module).ValueOrDie());
}

TEST_F(HostComputeScheduleOptimizerTest, TestCycleDetection) {
  std::string hlo_string = R"(
HloModule top

ENTRY %top (arg1: f32[], arg2: f32[], arg3: f32[], arg4: f32[]) -> () {
  %arg1 = f32[] parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg2 = f32[] parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %send1 = () custom-call(f32[] %arg1), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"send1_key\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}, sharding={maximal device=0}
  %send2 = () custom-call(f32[] %arg2), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"send2_key\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}, sharding={maximal device=1}
  ROOT %tuple = () tuple()
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ASSERT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());

  HloInstruction* send1 = comp->parameter_instruction(0)->users().at(0);
  HloInstruction* send2 = comp->parameter_instruction(1)->users().at(0);
  ASSERT_TRUE(IsPoplarInstruction(SendToHost)(send1));
  ASSERT_TRUE(IsPoplarInstruction(SendToHost)(send2));

  // Add dependency send2 -> send1 which contradicts the sharding order.
  ASSERT_TRUE(send2->AddControlDependencyTo(send1).ok());

  const auto ret = HostComputeScheduleOptimizer().Run(module);
  ASSERT_EQ(ret.status().code(), tensorflow::errors::Code::FAILED_PRECONDITION);
  ASSERT_THAT(ret.status().error_message(),
              ::testing::StartsWith("Unexpected dependency would cause cycle"));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
