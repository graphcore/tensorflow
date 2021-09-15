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

#include "tensorflow/compiler/plugin/poplar/driver/passes/remote_buffer_merger.h"

#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/passes/combine_instructions.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/forward_allocation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/parse_poplar_backend_config.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using RemoteBufferMergerTest = HloTestBase;

TEST_F(RemoteBufferMergerTest, TestNotInsideFunction) {
  const auto hlo_string = R"(
HloModule top

ENTRY top {
  buffer1 = f32[5,2] parameter(0), sharding={maximal device=0}
  buffer2 = f32[5,2] parameter(1), sharding={maximal device=0}
  loaded1 = f32[5,2] custom-call(buffer1), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
  loaded2 = f32[5,2] custom-call(buffer2), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
  ROOT ret = (f32[5,2], f32[5,2]) tuple(loaded1, loaded2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  EXPECT_TRUE(FlattenCallGraph().Run(module).ValueOrDie());

  CompilerAnnotations annotations(module);
  annotations.remote_parameter_infos.emplace(0);
  annotations.remote_parameter_infos.emplace(1);

  // Loads are not inside functions; should do nothing by default.
  EXPECT_FALSE(RemoteBufferMerger(annotations, THREESTATE_UNDEFINED)
                   .Run(module)
                   .ValueOrDie());

  // When turned on explicitly, it should still merge them.
  EXPECT_TRUE(
      RemoteBufferMerger(annotations, THREESTATE_ON).Run(module).ValueOrDie());

  // Check that the merging was performed correctly.
  const auto* root = module->entry_computation()->root_instruction();
  const auto* loaded1 = root->operand(0);
  const auto* loaded2 = root->operand(1);
  EXPECT_TRUE(IsPoplarInstruction(BufferLoadSlice, loaded1));
  EXPECT_TRUE(IsPoplarInstruction(BufferLoadSlice, loaded2));

  const auto* offset1 = loaded1->operand(1);
  const auto* offset2 = loaded2->operand(1);
  EXPECT_EQ(offset1->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(offset2->opcode(), HloOpcode::kConstant);

  const auto offset_values = {
      offset1->literal().data<int32>()[0],
      offset2->literal().data<int32>()[0],
  };
  EXPECT_THAT(offset_values, ::testing::UnorderedElementsAre(0, 1));

  std::vector<RemoteParameterInfo> merged(
      annotations.remote_parameter_infos.begin(),
      annotations.remote_parameter_infos.end());

  EXPECT_EQ(merged.size(), 2);
  EXPECT_EQ(merged[0].buffer_name, merged[1].buffer_name);
  EXPECT_EQ(merged[0].num_merged, 2);
  EXPECT_EQ(merged[1].num_merged, 2);
  EXPECT_EQ(merged[0].buffer_offset, 0);
  EXPECT_EQ(merged[1].buffer_offset, 1);
  EXPECT_FALSE(merged[0].is_replica_partitioned);
  EXPECT_FALSE(merged[1].is_replica_partitioned);
}

TEST_F(RemoteBufferMergerTest, TestRemoteParametersPassedToFunction) {
  const auto hlo_string = R"(
HloModule top

load_func {
  buffer = f32[5,2] parameter(0), sharding={maximal device=0}
  ROOT load = f32[5,2] custom-call(buffer), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
}

ENTRY top {
  buffer1 = f32[5,2] parameter(0), sharding={maximal device=0}
  buffer2 = f32[5,2] parameter(1), sharding={maximal device=0}
  loaded1 = f32[5,2] call(buffer1), to_apply=load_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  loaded2 = f32[5,2] call(buffer2), to_apply=load_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  ROOT ret = (f32[5,2], f32[5,2]) tuple(loaded1, loaded2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  EXPECT_TRUE(FlattenCallGraph().Run(module).ValueOrDie());

  // Mark both parameters as remote.
  CompilerAnnotations annotations(module);
  annotations.remote_parameter_infos.emplace(0);
  annotations.remote_parameter_infos.emplace(1);

  // If turned off, nothing should be done.
  EXPECT_FALSE(
      RemoteBufferMerger(annotations, THREESTATE_OFF).Run(module).ValueOrDie());

  // However, by default, they should be merged.
  EXPECT_TRUE(RemoteBufferMerger(annotations).Run(module).ValueOrDie());

  // Check that the merging was performed correctly.
  const auto* root = module->entry_computation()->root_instruction();

  const auto* loaded1 = root->operand(0);
  EXPECT_EQ(loaded1->opcode(), HloOpcode::kCall);
  EXPECT_EQ(loaded1->operand_count(), 2);
  const auto* offset1 = loaded1->operand(1);
  EXPECT_EQ(offset1->opcode(), HloOpcode::kConstant);

  const auto* loaded2 = root->operand(1);
  EXPECT_EQ(loaded2->opcode(), HloOpcode::kCall);
  EXPECT_EQ(loaded2->operand_count(), 2);
  const auto* offset2 = loaded2->operand(1);
  EXPECT_EQ(offset2->opcode(), HloOpcode::kConstant);

  const auto offset_values = {
      offset1->literal().data<int32>()[0],
      offset2->literal().data<int32>()[0],
  };
  EXPECT_THAT(offset_values, ::testing::UnorderedElementsAre(0, 1));

  std::vector<RemoteParameterInfo> merged(
      annotations.remote_parameter_infos.begin(),
      annotations.remote_parameter_infos.end());

  EXPECT_EQ(merged.size(), 2);
  EXPECT_EQ(merged[0].buffer_name, merged[1].buffer_name);
  EXPECT_EQ(merged[0].num_merged, 2);
  EXPECT_EQ(merged[1].num_merged, 2);
  EXPECT_EQ(merged[0].buffer_offset, 0);
  EXPECT_EQ(merged[1].buffer_offset, 1);
  EXPECT_FALSE(merged[0].is_replica_partitioned);
  EXPECT_FALSE(merged[1].is_replica_partitioned);
}

TEST_F(RemoteBufferMergerTest, TestRemoteParametersDifferentShapes) {
  const auto hlo_string = R"(
HloModule top

load_func1 {
  buffer = f32[5,2] parameter(0), sharding={maximal device=0}
  ROOT load = f32[5,2] custom-call(buffer), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
}

load_func2 {
  buffer = f32[2,5] parameter(0), sharding={maximal device=0}
  ROOT load = f32[2,5] custom-call(buffer), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
}

ENTRY top {
  buffer1 = f32[5,2] parameter(0), sharding={maximal device=0}
  buffer2 = f32[2,5] parameter(1), sharding={maximal device=0}
  loaded1 = f32[5,2] call(buffer1), to_apply=load_func1, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  loaded2 = f32[2,5] call(buffer2), to_apply=load_func2, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  ROOT ret = (f32[5,2], f32[2,5]) tuple(loaded1, loaded2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());

  // Mark both parameters as remote.
  CompilerAnnotations annotations(module);
  annotations.remote_parameter_infos.emplace(0);
  annotations.remote_parameter_infos.emplace(1);

  EXPECT_FALSE(RemoteBufferMerger(annotations).Run(module).ValueOrDie());
}

TEST_F(RemoteBufferMergerTest, TestRetainControlDependencies) {
  const auto hlo_string = R"(
HloModule top

load_func {
  buffer = f32[5,2] parameter(0), sharding={maximal device=0}
  ROOT load = f32[5,2] custom-call(buffer), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
}

ENTRY top {
  buffer1 = f32[5,2] parameter(0), sharding={maximal device=0}
  buffer2 = f32[5,2] parameter(1), sharding={maximal device=0}
  loaded1 = f32[5,2] call(buffer1), to_apply=load_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  loaded2 = f32[5,2] call(buffer2), to_apply=load_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  ROOT ret = (f32[5,2], f32[5,2]) tuple(loaded1, loaded2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());

  // Add control dependencies loaded1 -> loaded2 and buffer -> load.
  auto* entry_comp = module->entry_computation();
  auto* loaded1 = entry_comp->root_instruction()->mutable_operand(0);
  auto* loaded2 = entry_comp->root_instruction()->mutable_operand(1);
  TF_EXPECT_OK(loaded1->AddControlDependencyTo(loaded2));
  auto* load = loaded1->to_apply()->root_instruction();
  auto* buffer = load->mutable_operand(0);
  TF_EXPECT_OK(buffer->AddControlDependencyTo(load));

  EXPECT_TRUE(FlattenCallGraph().Run(module).ValueOrDie());

  // Mark both parameters as remote.
  CompilerAnnotations annotations(module);
  annotations.remote_parameter_infos.emplace(0);
  annotations.remote_parameter_infos.emplace(1);

  EXPECT_TRUE(RemoteBufferMerger(annotations).Run(module).ValueOrDie());

  // Check that the control dependencies were retained.
  auto* new_loaded1 = entry_comp->root_instruction()->mutable_operand(0);
  auto* new_loaded2 = entry_comp->root_instruction()->mutable_operand(1);
  EXPECT_EQ(new_loaded2->control_predecessors().size(), 1);
  EXPECT_EQ(new_loaded2->control_predecessors()[0], new_loaded1);

  auto* new_load = new_loaded1->to_apply()->root_instruction();
  auto* new_buffer = new_load->mutable_operand(0);
  EXPECT_EQ(new_load->control_predecessors().size(), 1);
  EXPECT_EQ(new_load->control_predecessors()[0], new_buffer);
}

TEST_F(RemoteBufferMergerTest, TestRemoteParameterLoadWithReplicationFactor) {
  const auto hlo_string = R"(
HloModule top

ENTRY top {
  buffer1 = f32[5,4] parameter(0), sharding={maximal device=0}
  buffer2 = f32[5,4] parameter(1), sharding={maximal device=0}
  loaded1 = f32[10] custom-call(buffer1), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":2}\n", sharding={maximal device=0}
  loaded2 = f32[10] custom-call(buffer2), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":2}\n", sharding={maximal device=0}
  ROOT ret = (f32[10], f32[10]) tuple(loaded1, loaded2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  EXPECT_TRUE(FlattenCallGraph().Run(module).ValueOrDie());

  // Mark both parameters as remote and replica partitioned.
  CompilerAnnotations annotations(module);

  annotations.remote_parameter_infos.emplace(
      /*parameter_number=*/0, /*is_replica_partitioned=*/true,
      /*name=*/"p0", /*buffer_offset=*/0, /*num_merged=*/0);

  annotations.remote_parameter_infos.emplace(
      /*parameter_number=*/1, /*is_replica_partitioned=*/true,
      /*name=*/"p1", /*buffer_offset=*/0, /*num_merged=*/0);

  EXPECT_TRUE(
      RemoteBufferMerger(annotations, THREESTATE_ON).Run(module).ValueOrDie());

  // Check that the merging was performed correctly.
  const auto* root = module->entry_computation()->root_instruction();
  const auto* loaded1 = Cast<HloBufferLoadSlice>(root->operand(0));
  const auto* loaded2 = Cast<HloBufferLoadSlice>(root->operand(1));
  EXPECT_EQ(loaded1->GetReplicationFactorCount(), 1);
  EXPECT_EQ(loaded2->GetReplicationFactorCount(), 1);
  EXPECT_EQ(loaded1->GetReplicationFactor(0), 2);
  EXPECT_EQ(loaded2->GetReplicationFactor(0), 2);

  const auto merged = std::vector<RemoteParameterInfo>(
      annotations.remote_parameter_infos.begin(),
      annotations.remote_parameter_infos.end());

  EXPECT_EQ(merged.size(), 2);
  EXPECT_EQ(merged[0].buffer_name, merged[1].buffer_name);
  EXPECT_EQ(merged[0].num_merged, 2);
  EXPECT_EQ(merged[1].num_merged, 2);
  EXPECT_EQ(merged[0].buffer_offset, 0);
  EXPECT_EQ(merged[1].buffer_offset, 1);
  EXPECT_TRUE(merged[0].is_replica_partitioned);
  EXPECT_TRUE(merged[1].is_replica_partitioned);
}

TEST_F(RemoteBufferMergerTest, TestRemoteParameterStoreWithReplicationFactor) {
  const auto hlo_string = R"(
HloModule top

ENTRY top {
  buffer1 = f32[5,4] parameter(0), sharding={maximal device=0}
  buffer2 = f32[5,4] parameter(1), sharding={maximal device=0}
  value1 = f32[10] parameter(2), sharding={maximal device=0}
  value2 = f32[10] parameter(3), sharding={maximal device=0}
  stored1 = f32[5,4] custom-call(buffer1, value1), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":2}\n", sharding={maximal device=0}
  stored2 = f32[5,4] custom-call(buffer2, value2), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":2}\n", sharding={maximal device=0}
  ROOT ret = (f32[5,4], f32[5,4]) tuple(stored1, stored2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  EXPECT_TRUE(FlattenCallGraph().Run(module).ValueOrDie());

  // Mark both parameters as remote and replica partitioned.
  CompilerAnnotations annotations(module);

  annotations.remote_parameter_infos.emplace(
      /*parameter_number=*/0, /*is_replica_partitioned=*/true,
      /*name=*/"p0", /*buffer_offset=*/0, /*num_merged=*/0);

  annotations.remote_parameter_infos.emplace(
      /*parameter_number=*/1, /*is_replica_partitioned=*/true,
      /*name=*/"p1", /*buffer_offset=*/0, /*num_merged=*/0);

  EXPECT_TRUE(
      RemoteBufferMerger(annotations, THREESTATE_ON).Run(module).ValueOrDie());

  // Check that the merging was performed correctly.
  const auto* root = module->entry_computation()->root_instruction();
  const auto* stored1 = Cast<HloBufferStoreSlice>(root->operand(0));
  const auto* stored2 = Cast<HloBufferStoreSlice>(root->operand(1));
  EXPECT_EQ(stored1->GetReplicationFactorCount(), 1);
  EXPECT_EQ(stored2->GetReplicationFactorCount(), 1);
  EXPECT_EQ(stored1->GetReplicationFactor(0), 2);
  EXPECT_EQ(stored2->GetReplicationFactor(0), 2);

  const auto merged = std::vector<RemoteParameterInfo>(
      annotations.remote_parameter_infos.begin(),
      annotations.remote_parameter_infos.end());

  EXPECT_EQ(merged.size(), 2);
  EXPECT_EQ(merged[0].buffer_name, merged[1].buffer_name);
  EXPECT_EQ(merged[0].num_merged, 2);
  EXPECT_EQ(merged[1].num_merged, 2);
  EXPECT_EQ(merged[0].buffer_offset, 0);
  EXPECT_EQ(merged[1].buffer_offset, 1);
  EXPECT_TRUE(merged[0].is_replica_partitioned);
  EXPECT_TRUE(merged[1].is_replica_partitioned);
}

TEST_F(RemoteBufferMergerTest,
       TestRemoteParametersWithIncompatibleReplicaPartitioning) {
  const auto hlo_string = R"(
HloModule top

ENTRY top {
  buffer1 = f32[5,4] parameter(0), sharding={maximal device=0}
  buffer2 = f32[5,4] parameter(1), sharding={maximal device=0}
  loaded1 = f32[10] custom-call(buffer1), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":2}\n", sharding={maximal device=0}
  loaded2 = f32[5,4] custom-call(buffer2), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
  ROOT ret = (f32[10], f32[5,4]) tuple(loaded1, loaded2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  EXPECT_TRUE(FlattenCallGraph().Run(module).ValueOrDie());

  // Mark both parameters as remote and replica partitioned.
  CompilerAnnotations annotations(module);

  annotations.remote_parameter_infos.emplace(
      /*parameter_number=*/0, /*is_replica_partitioned=*/true,
      /*name=*/"p0", /*buffer_offset=*/0, /*num_merged=*/1);

  annotations.remote_parameter_infos.emplace(
      /*parameter_number=*/1, /*is_replica_partitioned=*/false,
      /*name=*/"p1", /*buffer_offset=*/0, /*num_merged=*/1);

  EXPECT_FALSE(
      RemoteBufferMerger(annotations, THREESTATE_ON).Run(module).ValueOrDie());

  // Check that the remote parameters were unchanged.
  const auto infos = std::vector<RemoteParameterInfo>(
      annotations.remote_parameter_infos.begin(),
      annotations.remote_parameter_infos.end());
  EXPECT_EQ(infos.size(), 2);
  EXPECT_EQ(infos[0].num_merged, 1);
  EXPECT_EQ(infos[1].num_merged, 1);
  EXPECT_EQ(infos[0].buffer_offset, 0);
  EXPECT_EQ(infos[1].buffer_offset, 0);
  EXPECT_EQ(infos[0].buffer_name, "p0");
  EXPECT_EQ(infos[1].buffer_name, "p1");
  EXPECT_TRUE(infos[0].is_replica_partitioned);
  EXPECT_FALSE(infos[1].is_replica_partitioned);
}

TEST_F(RemoteBufferMergerTest, TestOnlyOneRemoteParameterPassedToFunction) {
  const auto hlo_string = R"(
HloModule top

load_func {
  buffer = f32[5,2] parameter(0), sharding={maximal device=0}
  ROOT load = f32[5,2] custom-call(buffer), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
}

ENTRY top {
  buffer1 = f32[5,2] parameter(0), sharding={maximal device=0}
  buffer2 = f32[5,2] parameter(1), sharding={maximal device=0}
  loaded1 = f32[5,2] call(buffer1), to_apply=load_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  ROOT ret = (f32[5,2], f32[5,2]) tuple(loaded1, buffer2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  EXPECT_TRUE(FlattenCallGraph().Run(module).ValueOrDie());

  // Mark both parameters as remote.
  CompilerAnnotations annotations(module);
  annotations.remote_parameter_infos.emplace(0);
  annotations.remote_parameter_infos.emplace(1);

  EXPECT_FALSE(RemoteBufferMerger(annotations).Run(module).ValueOrDie());
}

TEST_F(RemoteBufferMergerTest, TestRemoteParametersNotShapeCompatible) {
  const auto hlo_string = R"(
HloModule top

load1_func {
  buffer = f32[5,2] parameter(0), sharding={maximal device=0}
  ROOT load = f32[5,2] custom-call(buffer), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
}

load2_func {
  buffer = f32[5,1] parameter(0), sharding={maximal device=0}
  ROOT load = f32[5,1] custom-call(buffer), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
}

ENTRY top {
  buffer1 = f32[5,2] parameter(0), sharding={maximal device=0}
  buffer2 = f32[5,1] parameter(1), sharding={maximal device=0}
  loaded1 = f32[5,2] call(buffer1), to_apply=load1_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  loaded2 = f32[5,1] call(buffer2), to_apply=load2_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  ROOT ret = (f32[5,2], f32[5,1]) tuple(loaded1, loaded2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  EXPECT_TRUE(FlattenCallGraph().Run(module).ValueOrDie());

  // Mark both parameters as remote.
  CompilerAnnotations annotations(module);
  annotations.remote_parameter_infos.emplace(0);
  annotations.remote_parameter_infos.emplace(1);

  EXPECT_FALSE(RemoteBufferMerger(annotations).Run(module).ValueOrDie());
}

TEST_F(RemoteBufferMergerTest, TestRemoteParametersNotShardingCompatible) {
  const auto hlo_string = R"(
HloModule top

load1_func {
  buffer = f32[5,2] parameter(0), sharding={maximal device=0}
  ROOT load = f32[5,2] custom-call(buffer), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
}

load2_func {
  buffer = f32[5,2] parameter(0), sharding={maximal device=1}
  ROOT load = f32[5,2] custom-call(buffer), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=1}
}

ENTRY top {
  buffer1 = f32[5,2] parameter(0), sharding={maximal device=0}
  buffer2 = f32[5,2] parameter(1), sharding={maximal device=1}
  loaded1 = f32[5,2] call(buffer1), to_apply=load1_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  loaded2 = f32[5,2] call(buffer2), to_apply=load2_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=1}
  ROOT ret = (f32[5,2], f32[5,2]) tuple(loaded1, loaded2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  EXPECT_TRUE(FlattenCallGraph().Run(module).ValueOrDie());

  // Mark both parameters as remote.
  CompilerAnnotations annotations(module);
  annotations.remote_parameter_infos.emplace(0);
  annotations.remote_parameter_infos.emplace(1);

  EXPECT_FALSE(RemoteBufferMerger(annotations).Run(module).ValueOrDie());
}

TEST_F(RemoteBufferMergerTest, TestRemoteParametersPassedToNestedFunction) {
  const auto hlo_string = R"(
HloModule top

load_func {
  buffer = f32[5,2] parameter(0), sharding={maximal device=0}
  ROOT load = f32[5,2] custom-call(buffer), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
}

wrapper_func {
  buffer1 = f32[5,2] parameter(0), sharding={maximal device=0}
  buffer2 = f32[5,2] parameter(1), sharding={maximal device=0}
  loaded1 = f32[5,2] call(buffer1), to_apply=load_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  loaded2 = f32[5,2] call(buffer2), to_apply=load_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  ROOT ret = (f32[5,2], f32[5,2]) tuple(loaded1, loaded2)
}

ENTRY top {
  buffer1 = f32[5,2] parameter(0), sharding={maximal device=0}
  buffer2 = f32[5,2] parameter(1), sharding={maximal device=0}
  ROOT ret = (f32[5,2], f32[5,2]) call(buffer1, buffer2), to_apply=wrapper_func, sharding={maximal device=0}
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  EXPECT_TRUE(FlattenCallGraph().Run(module).ValueOrDie());

  // Mark both parameters as remote.
  CompilerAnnotations annotations(module);
  annotations.remote_parameter_infos.emplace(0);
  annotations.remote_parameter_infos.emplace(1);

  EXPECT_TRUE(RemoteBufferMerger(annotations).Run(module).ValueOrDie());

  const auto* entry_root = module->entry_computation()->root_instruction();
  const auto* root = entry_root->to_apply()->root_instruction();

  const auto* loaded1 = root->operand(0);
  EXPECT_EQ(loaded1->opcode(), HloOpcode::kCall);
  EXPECT_EQ(loaded1->operand_count(), 2);
  const auto* offset1 = loaded1->operand(1);
  EXPECT_EQ(offset1->opcode(), HloOpcode::kConstant);

  const auto* loaded2 = root->operand(1);
  EXPECT_EQ(loaded2->opcode(), HloOpcode::kCall);
  EXPECT_EQ(loaded2->operand_count(), 2);
  const auto* offset2 = loaded2->operand(1);
  EXPECT_EQ(offset2->opcode(), HloOpcode::kConstant);

  const auto offset_values = {
      offset1->literal().data<int32>()[0],
      offset2->literal().data<int32>()[0],
  };
  EXPECT_THAT(offset_values, ::testing::UnorderedElementsAre(0, 1));
}

TEST_F(RemoteBufferMergerTest, TestGradientAccumulatorPassedToFunction) {
  const auto hlo_string = R"(
HloModule top

load_func {
  buffer = f32[5,2] parameter(0), sharding={maximal device=0}
  ROOT load = f32[5,2] custom-call(buffer), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
}

ENTRY top {
  buffer1 = f32[5,2] parameter(0), sharding={maximal device=0}
  buffer2 = f32[5,1] parameter(1), sharding={maximal device=0}
  accumulator1 = f32[5,2] custom-call(buffer1), custom_call_target="GradientAccumulatorCreate", backend_config="{\"is_remote\":1}\n", sharding={maximal device=0}
  accumulator2 = f32[5,2] custom-call(buffer2), custom_call_target="GradientAccumulatorCreate", backend_config="{\"is_remote\":1}\n", sharding={maximal device=0}
  loaded1 = f32[5,2] call(accumulator1), to_apply=load_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  loaded2 = f32[5,2] call(accumulator2), to_apply=load_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  ROOT ret = (f32[5,2], f32[5,2]) tuple(loaded1, loaded2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  EXPECT_TRUE(FlattenCallGraph().Run(module).ValueOrDie());

  CompilerAnnotations annotations(module);
  EXPECT_TRUE(RemoteBufferMerger(annotations).Run(module).ValueOrDie());

  const auto* root = module->entry_computation()->root_instruction();

  const auto* loaded1 = root->operand(0);
  EXPECT_EQ(loaded1->opcode(), HloOpcode::kCall);
  EXPECT_EQ(loaded1->operand_count(), 2);
  const auto* offset1 = loaded1->operand(1);
  EXPECT_EQ(offset1->opcode(), HloOpcode::kConstant);

  const auto* loaded2 = root->operand(1);
  EXPECT_EQ(loaded2->opcode(), HloOpcode::kCall);
  EXPECT_EQ(loaded2->operand_count(), 2);
  const auto* offset2 = loaded2->operand(1);
  EXPECT_EQ(offset2->opcode(), HloOpcode::kConstant);

  const auto offset_values = {
      offset1->literal().data<int32>()[0],
      offset2->literal().data<int32>()[0],
  };
  EXPECT_THAT(offset_values, ::testing::UnorderedElementsAre(0, 1));

  const auto* accumulator1 = loaded1->operand(0);
  EXPECT_EQ(accumulator1->opcode(), HloOpcode::kCustomCall);
  EXPECT_TRUE(IsPoplarInstruction(GradientAccumulatorCreate, accumulator1));
  const auto buffer1_info =
      Cast<HloGradientAccumulatorCreate>(accumulator1)->RemoteBufferInfo();

  const auto* accumulator2 = loaded2->operand(0);
  EXPECT_EQ(accumulator2->opcode(), HloOpcode::kCustomCall);
  EXPECT_TRUE(IsPoplarInstruction(GradientAccumulatorCreate, accumulator2));
  const auto buffer2_info =
      Cast<HloGradientAccumulatorCreate>(accumulator2)->RemoteBufferInfo();

  ASSERT_TRUE(buffer1_info.has_value());
  ASSERT_TRUE(buffer2_info.has_value());
  EXPECT_EQ(buffer1_info->name, buffer2_info->name);
  EXPECT_EQ(buffer1_info->num_merged, 2);
  EXPECT_EQ(buffer2_info->num_merged, 2);
  EXPECT_EQ(buffer1_info->merge_offset, 0);
  EXPECT_EQ(buffer2_info->merge_offset, 1);
}

TEST_F(RemoteBufferMergerTest, TestCreateBufferPassedToFunction) {
  const auto hlo_string = R"(
HloModule top

load_store_func {
  buffer = f32[5,2] parameter(0), sharding={maximal device=0}
  offset = s32[] parameter(1), sharding={maximal device=0}
  load = f32[2] custom-call(buffer, offset), custom_call_target="BufferLoadSlice", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
  ROOT store = f32[5,2] custom-call(buffer, load, offset), custom_call_target="BufferStoreSlice", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
}

ENTRY top {
  buffer1 = f32[5,2] custom-call(), custom_call_target="CreateBuffer", backend_config="{\"is_remote\":1}\n", sharding={maximal device=0}
  buffer2 = f32[5,2] custom-call(), custom_call_target="CreateBuffer", backend_config="{\"is_remote\":1}\n", sharding={maximal device=0}
  offset1 = s32[] parameter(0), sharding={maximal device=0}
  offset2 = s32[] parameter(1), sharding={maximal device=0}
  stored1 = f32[5,2] call(buffer1, offset1), to_apply=load_store_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  stored2 = f32[5,2] call(buffer2, offset2), to_apply=load_store_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  ROOT ret = (f32[5,2], f32[5,2]) tuple(stored1, stored2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  EXPECT_TRUE(FlattenCallGraph().Run(module).ValueOrDie());

  CompilerAnnotations annotations(module);
  EXPECT_TRUE(RemoteBufferMerger(annotations).Run(module).ValueOrDie());

  const auto* root = module->entry_computation()->root_instruction();

  const auto* stored1 = root->operand(0);
  EXPECT_EQ(stored1->opcode(), HloOpcode::kCall);
  EXPECT_EQ(stored1->operand_count(), 4);
  const auto* offset1_load = stored1->operand(2);
  EXPECT_EQ(offset1_load->opcode(), HloOpcode::kConstant);
  const auto* offset1_store = stored1->operand(3);
  EXPECT_EQ(offset1_store->opcode(), HloOpcode::kConstant);

  EXPECT_TRUE(Match(
      stored1->to_apply()->root_instruction(),
      m::CustomCall(m::Parameter(0),
                    m::CustomCall(m::Parameter(0),
                                  m::Add(m::Parameter(1), m::Parameter(2))),
                    m::Add(m::Parameter(1), m::Parameter(3)))));

  const auto* stored2 = root->operand(1);
  EXPECT_EQ(stored2->opcode(), HloOpcode::kCall);
  EXPECT_EQ(stored2->operand_count(), 4);
  const auto* offset2_load = stored2->operand(2);
  EXPECT_EQ(offset2_load->opcode(), HloOpcode::kConstant);
  const auto* offset2_store = stored2->operand(3);
  EXPECT_EQ(offset2_store->opcode(), HloOpcode::kConstant);

  EXPECT_TRUE(Match(
      stored2->to_apply()->root_instruction(),
      m::CustomCall(m::Parameter(0),
                    m::CustomCall(m::Parameter(0),
                                  m::Add(m::Parameter(1), m::Parameter(2))),
                    m::Add(m::Parameter(1), m::Parameter(3)))));

  // The new total "num_repeats" is going to be 2 * 5 because we merged 2
  // buffers with outer dimension 5. Hence, offsets into the second buffer must
  // start from 5.
  const auto offset_values = {
      offset1_load->literal().data<int32>()[0],
      offset2_load->literal().data<int32>()[0],
  };
  EXPECT_THAT(offset_values, ::testing::UnorderedElementsAre(0, 5));
  EXPECT_EQ(offset1_load->literal().data<int32>()[0],
            offset1_store->literal().data<int32>()[0]);
  EXPECT_EQ(offset2_load->literal().data<int32>()[0],
            offset2_store->literal().data<int32>()[0]);

  const auto* buffer1 = stored1->operand(0);
  EXPECT_EQ(buffer1->opcode(), HloOpcode::kCustomCall);
  EXPECT_TRUE(IsPoplarInstruction(CreateBuffer, buffer1));
  const auto buffer1_info = Cast<HloCreateBuffer>(buffer1)->RemoteBufferInfo();

  const auto* buffer2 = stored2->operand(0);
  EXPECT_EQ(buffer2->opcode(), HloOpcode::kCustomCall);
  EXPECT_TRUE(IsPoplarInstruction(CreateBuffer, buffer2));
  const auto buffer2_info = Cast<HloCreateBuffer>(buffer2)->RemoteBufferInfo();

  ASSERT_TRUE(buffer1_info.has_value());
  ASSERT_TRUE(buffer2_info.has_value());
  EXPECT_EQ(buffer1_info->name, buffer2_info->name);
  EXPECT_EQ(buffer1_info->num_merged, 2);
  EXPECT_EQ(buffer2_info->num_merged, 2);
  EXPECT_EQ(buffer1_info->merge_offset, 0);
  EXPECT_EQ(buffer2_info->merge_offset, 1);
}

TEST_F(RemoteBufferMergerTest,
       TestRemoteParameterUsedBothInsideAndOutsideFunction) {
  const auto hlo_string = R"(
HloModule top

load_func {
  buffer = f32[5,2] parameter(0), sharding={maximal device=1}
  ROOT load = f32[5,2] custom-call(buffer), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=1}
}

ENTRY top {
  buffer1 = f32[5,2] parameter(0), sharding={maximal device=1}
  buffer2 = f32[5,2] parameter(1), sharding={maximal device=1}
  loaded1 = f32[5,2] call(buffer1), to_apply=load_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=1}
  loaded2 = f32[5,2] call(buffer2), to_apply=load_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=1}
  stored2 = f32[5,2] custom-call(buffer2, loaded2), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=1}
  ROOT ret = (f32[5,2], f32[5,2], f32[5,2]) tuple(loaded1, loaded2, stored2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  EXPECT_TRUE(FlattenCallGraph().Run(module).ValueOrDie());

  // Mark both parameters as remote.
  CompilerAnnotations annotations(module);
  annotations.remote_parameter_infos.emplace(0);
  annotations.remote_parameter_infos.emplace(1);

  EXPECT_TRUE(RemoteBufferMerger(annotations).Run(module).ValueOrDie());

  const auto* root = module->entry_computation()->root_instruction();

  const auto* loaded1 = root->operand(0);
  EXPECT_EQ(loaded1->opcode(), HloOpcode::kCall);
  EXPECT_EQ(loaded1->sharding_unique_device(), 1);
  EXPECT_EQ(loaded1->operand_count(), 2);
  const auto* loaded1_offset_argument = loaded1->operand(1);
  EXPECT_EQ(loaded1_offset_argument->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(loaded1_offset_argument->sharding_unique_device().value(), 1);
  const auto* loaded1_offset_parameter =
      loaded1->to_apply()->parameter_instruction(1);
  EXPECT_EQ(loaded1_offset_parameter->sharding_unique_device().value(), 1);

  const auto* loaded2 = root->operand(1);
  EXPECT_EQ(loaded2->opcode(), HloOpcode::kCall);
  EXPECT_EQ(loaded2->sharding_unique_device(), 1);
  EXPECT_EQ(loaded2->operand_count(), 2);

  const auto* loaded2_offset_argument = loaded2->operand(1);
  EXPECT_EQ(loaded2_offset_argument->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(loaded2_offset_argument->sharding_unique_device().value(), 1);
  const auto* loaded2_offset_parameter =
      loaded2->to_apply()->parameter_instruction(1);
  EXPECT_EQ(loaded2_offset_parameter->sharding_unique_device().value(), 1);

  const auto* stored2 = root->operand(2);
  EXPECT_EQ(stored2->opcode(), HloOpcode::kCustomCall);
  EXPECT_TRUE(IsPoplarInstruction(BufferStoreSlice, stored2));
  EXPECT_EQ(stored2->sharding_unique_device(), 1);
  const auto* stored2_offset = stored2->operand(2);
  EXPECT_EQ(stored2_offset->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(loaded2_offset_argument->literal().data<int32>()[0],
            stored2_offset->literal().data<int32>()[0]);
}

TEST_F(RemoteBufferMergerTest, TestRemoteParametersReturnedFromFunction) {
  const auto hlo_string = R"(
HloModule top

identity_func {
  ROOT ret = f32[5,2] parameter(0), sharding={maximal device=0}
}

load_func {
  buffer = f32[5,2] parameter(0), sharding={maximal device=0}
  ROOT load = f32[5,2] custom-call(buffer), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
}

ENTRY top {
  buffer1 = f32[5,2] parameter(0), sharding={maximal device=0}
  buffer2 = f32[5,2] parameter(1), sharding={maximal device=0}
  buffer1_ret = f32[5,2] call(buffer1), to_apply=identity_func, sharding={maximal device=0}
  buffer2_ret = f32[5,2] call(buffer2), to_apply=identity_func, sharding={maximal device=0}
  loaded1 = f32[5,2] call(buffer1_ret), to_apply=load_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  loaded2 = f32[5,2] call(buffer2_ret), to_apply=load_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  ROOT ret = (f32[5,2], f32[5,2]) tuple(loaded1, loaded2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  EXPECT_TRUE(FlattenCallGraph().Run(module).ValueOrDie());

  // Mark both parameters as remote.
  CompilerAnnotations annotations(module);
  annotations.remote_parameter_infos.emplace(0);
  annotations.remote_parameter_infos.emplace(1);

  EXPECT_TRUE(RemoteBufferMerger(annotations).Run(module).ValueOrDie());

  const auto* root = module->entry_computation()->root_instruction();

  const auto* loaded1 = root->operand(0);
  EXPECT_EQ(loaded1->opcode(), HloOpcode::kCall);
  EXPECT_EQ(loaded1->operand_count(), 2);
  const auto* offset1 = loaded1->operand(1);
  EXPECT_EQ(offset1->opcode(), HloOpcode::kConstant);

  const auto* loaded2 = root->operand(1);
  EXPECT_EQ(loaded2->opcode(), HloOpcode::kCall);
  EXPECT_EQ(loaded2->operand_count(), 2);
  const auto* offset2 = loaded2->operand(1);
  EXPECT_EQ(offset2->opcode(), HloOpcode::kConstant);

  const auto offset_values = {
      offset1->literal().data<int32>()[0],
      offset2->literal().data<int32>()[0],
  };
  EXPECT_THAT(offset_values, ::testing::UnorderedElementsAre(0, 1));

  std::vector<RemoteParameterInfo> merged(
      annotations.remote_parameter_infos.begin(),
      annotations.remote_parameter_infos.end());

  EXPECT_EQ(merged.size(), 2);
  EXPECT_EQ(merged[0].buffer_name, merged[1].buffer_name);
  EXPECT_EQ(merged[0].num_merged, 2);
  EXPECT_EQ(merged[1].num_merged, 2);
  EXPECT_EQ(merged[0].buffer_offset, 0);
  EXPECT_EQ(merged[1].buffer_offset, 1);
}

TEST_F(RemoteBufferMergerTest, TestRemoteParametersReturnedFromFunctionTuple) {
  const auto hlo_string = R"(
HloModule top

identity_func {
  buffer1 = f32[5,2] parameter(0), sharding={maximal device=0}
  buffer2 = f32[5,2] parameter(1), sharding={maximal device=0}
  const1 = f32[5,2] constant(0), sharding={maximal device=0}
  stored1 = f32[5,2] custom-call(buffer1, const1), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
  ROOT ret = (f32[5,2], f32[5,2]) tuple(stored1, buffer2), sharding={maximal device=0}
}

load_func {
  buffer = f32[5,2] parameter(0), sharding={maximal device=0}
  ROOT load = f32[5,2] custom-call(buffer), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
}

ENTRY top {
  buffer1 = f32[5,2] parameter(0), sharding={maximal device=0}
  buffer2 = f32[5,2] parameter(1), sharding={maximal device=0}
  buffers_ret = (f32[5,2], f32[5,2]) call(buffer1, buffer2), to_apply=identity_func
  buffer1_ret = f32[5,2] get-tuple-element(buffers_ret), index=0, sharding={maximal device=0}
  buffer2_ret = f32[5,2] get-tuple-element(buffers_ret), index=1, sharding={maximal device=0}
  loaded1 = f32[5,2] call(buffer1_ret), to_apply=load_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  loaded2 = f32[5,2] call(buffer2_ret), to_apply=load_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  ROOT ret = (f32[5,2], f32[5,2]) tuple(loaded1, loaded2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  EXPECT_TRUE(FlattenCallGraph().Run(module).ValueOrDie());

  // Mark both parameters as remote.
  CompilerAnnotations annotations(module);
  annotations.remote_parameter_infos.emplace(0);
  annotations.remote_parameter_infos.emplace(1);

  EXPECT_TRUE(RemoteBufferMerger(annotations).Run(module).ValueOrDie());

  const auto* root = module->entry_computation()->root_instruction();

  const auto* loaded1 = root->operand(0);
  EXPECT_EQ(loaded1->opcode(), HloOpcode::kCall);
  EXPECT_EQ(loaded1->operand_count(), 2);
  const auto* offset1 = loaded1->operand(1);
  EXPECT_EQ(offset1->opcode(), HloOpcode::kConstant);

  const auto* loaded2 = root->operand(1);
  EXPECT_EQ(loaded2->opcode(), HloOpcode::kCall);
  EXPECT_EQ(loaded2->operand_count(), 2);
  const auto* offset2 = loaded2->operand(1);
  EXPECT_EQ(offset2->opcode(), HloOpcode::kConstant);

  const auto offset_values = {
      offset1->literal().data<int32>()[0],
      offset2->literal().data<int32>()[0],
  };
  EXPECT_THAT(offset_values, ::testing::UnorderedElementsAre(0, 1));

  std::vector<RemoteParameterInfo> merged(
      annotations.remote_parameter_infos.begin(),
      annotations.remote_parameter_infos.end());

  EXPECT_EQ(merged.size(), 2);
  EXPECT_EQ(merged[0].buffer_name, merged[1].buffer_name);
  EXPECT_EQ(merged[0].num_merged, 2);
  EXPECT_EQ(merged[1].num_merged, 2);
  EXPECT_EQ(merged[0].buffer_offset, 0);
  EXPECT_EQ(merged[1].buffer_offset, 1);
}

TEST_F(RemoteBufferMergerTest,
       TestMergeRemoteParameterWithGradientAccumulator) {
  const auto hlo_string = R"(
HloModule top

load_func {
  buffer = f32[5,2] parameter(0), sharding={maximal device=0}
  ROOT load = f32[5,2] custom-call(buffer), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
}


ENTRY top {
  buffer = f32[5,2] parameter(0), sharding={maximal device=0}
  accumulator = f32[5,2] custom-call(), custom_call_target="GradientAccumulatorCreate", backend_config="{\"is_remote\":1}\n", sharding={maximal device=0}
  loaded1 = f32[5,2] call(buffer), to_apply=load_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  loaded2 = f32[5,2] call(accumulator), to_apply=load_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}", sharding={maximal device=0}
  ROOT ret = (f32[5,2], f32[5,2]) tuple(loaded1, loaded2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  EXPECT_TRUE(FlattenCallGraph().Run(module).ValueOrDie());

  CompilerAnnotations annotations(module);
  annotations.remote_parameter_infos.emplace(0);

  EXPECT_TRUE(RemoteBufferMerger(annotations, THREESTATE_UNDEFINED)
                  .Run(module)
                  .ValueOrDie());

  std::vector<RemoteParameterInfo> merged(
      annotations.remote_parameter_infos.begin(),
      annotations.remote_parameter_infos.end());
  EXPECT_EQ(merged.size(), 1);
  const auto parameter_info = merged[0];

  const auto* root = module->entry_computation()->root_instruction();
  const auto* loaded2 = root->operand(1);
  const auto* accumulator =
      Cast<HloGradientAccumulatorCreate>(loaded2->operand(0));
  const auto accumulator_info = accumulator->RemoteBufferInfo();
  EXPECT_TRUE(accumulator_info.has_value());

  EXPECT_EQ(parameter_info.buffer_name, accumulator_info->name);
  EXPECT_EQ(parameter_info.num_merged, 2);
  EXPECT_EQ(parameter_info.buffer_offset, 0);
  EXPECT_EQ(accumulator_info->num_merged, 2);
  EXPECT_EQ(accumulator_info->merge_offset, 1);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
