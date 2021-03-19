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

#include "tensorflow/compiler/plugin/poplar/driver/passes/host_compute_barrier_inserter.h"

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

using HostComputeBarrierInserterTest = HloTestBase;

TEST_F(HostComputeBarrierInserterTest, TestInsertOneBarrier) {
  std::string hlo_string = R"(
HloModule top

ENTRY %top (arg: f32[]) -> f32[] {
  %arg = f32[] parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %send = () custom-call(f32[] %arg), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"send_key\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}
  ROOT %recv = f32[] custom-call(), custom_call_target="RecvFromHost", backend_config="{\"rendezvous_key\":\"recv_key\"}", metadata={op_type="XlaHostCompute" op_name="host_compute"}
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ASSERT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());

  HostComputeBarrierInserter inserter;
  ASSERT_TRUE(inserter.Run(module).ValueOrDie());

  auto* barrier = comp->GetInstructionWithName("host_compute.barrier");
  ASSERT_NE(barrier, nullptr);

  int num_sends = 0;
  int num_recvs = 0;

  for (HloInstruction* inst : comp->instructions()) {
    if (IsPoplarInstruction(SendToHost)(inst)) {
      ++num_sends;
      ASSERT_EQ(inst->control_successors().size(), 1);
      ASSERT_EQ(inst->control_successors()[0], barrier);
    } else if (IsPoplarInstruction(RecvFromHost)(inst)) {
      ++num_recvs;
      ASSERT_EQ(inst->control_predecessors().size(), 1);
      ASSERT_EQ(inst->control_predecessors()[0], barrier);
    }
  }

  ASSERT_EQ(num_sends, 1);
  ASSERT_EQ(num_recvs, 1);
}

TEST_F(HostComputeBarrierInserterTest, TestNoBarrierBetweenDifferentOps) {
  std::string hlo_string = R"(
HloModule top

ENTRY %top (arg: f32[]) -> f32[] {
  %arg = f32[] parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %send = () custom-call(f32[] %arg), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"send_key\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}
  ROOT %recv = f32[] custom-call(), custom_call_target="RecvFromHost", backend_config="{\"rendezvous_key\":\"recv_key\"}", metadata={op_type="XlaHostCompute" op_name="host_compute_2"}
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ASSERT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());

  HostComputeBarrierInserter inserter;
  ASSERT_FALSE(inserter.Run(module).ValueOrDie());

  ASSERT_EQ(nullptr, comp->GetInstructionWithName("host_compute.barrier"));
  ASSERT_EQ(nullptr, comp->GetInstructionWithName("host_compute_2.barrier"));

  int num_sends = 0;
  int num_recvs = 0;

  for (HloInstruction* inst : comp->instructions()) {
    if (IsPoplarInstruction(SendToHost)(inst)) {
      ++num_sends;
      ASSERT_EQ(inst->control_successors().size(), 0);
    } else if (IsPoplarInstruction(RecvFromHost)(inst)) {
      ++num_recvs;
      ASSERT_EQ(inst->control_predecessors().size(), 0);
    }
  }

  ASSERT_EQ(num_sends, 1);
  ASSERT_EQ(num_recvs, 1);
}

TEST_F(HostComputeBarrierInserterTest,
       TestInsertDependenciesFromAllSendsToAllRecvs) {
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

  HostComputeBarrierInserter inserter;
  ASSERT_TRUE(inserter.Run(module).ValueOrDie());

  auto* barrier = comp->GetInstructionWithName("host_compute.barrier");
  ASSERT_NE(barrier, nullptr);
  ASSERT_EQ(barrier->control_predecessors().size(), 2);
  ASSERT_EQ(barrier->control_successors().size(), 2);

  int num_sends = 0;
  int num_recvs = 0;

  for (HloInstruction* inst : comp->instructions()) {
    if (IsPoplarInstruction(SendToHost)(inst)) {
      ++num_sends;
      ASSERT_EQ(inst->control_successors().size(), 1);
      ASSERT_EQ(inst->control_successors()[0], barrier);
    } else if (IsPoplarInstruction(RecvFromHost)(inst)) {
      ++num_recvs;
      ASSERT_EQ(inst->control_predecessors().size(), 1);
      ASSERT_EQ(inst->control_predecessors()[0], barrier);
    }
  }

  ASSERT_EQ(num_sends, 2);
  ASSERT_EQ(num_recvs, 2);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
