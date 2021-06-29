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

#include "tensorflow/compiler/plugin/poplar/driver/passes/variables_offload_and_partition.h"

#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using VariablesOffloadAndPartitionTest = HloTestBase;

std::string GetHlo(
    ThreeState offload_resource_variables,
    ThreeState partition_resource_variables = THREESTATE_OFF,
    ThreeState offload_variables = THREESTATE_OFF,
    ThreeState partition_variables = THREESTATE_OFF,
    PoplarBackendConfig::CallConfig::PipelineConfig::Schedule schedule =
        PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved,
    int64 batch_serialization_iterations = 1) {
  constexpr absl::string_view hlo_format = R"(
HloModule top

stage_0_fwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,2] parameter(1)
  add = f32[1,4,4,2] add(in0,in1)
  ROOT tuple = (f32[1,4,4,2]) tuple(add)
}

stage_1_fwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,2] parameter(1)
  add = f32[1,4,4,2] add(in0,in1)
  ROOT tuple = (f32[1,4,4,2]) tuple(add)
}

stage_1_bwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,2] parameter(1)
  sub = f32[1,4,4,2] subtract(in0,in1)
  ROOT tuple = (f32[1,4,4,2]) tuple(sub)
}

stage_0_bwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,2] parameter(1)
  sub = f32[1,4,4,2] subtract(in0,in1)
  ROOT tuple = (f32[1,4,4,2]) tuple(sub)
}

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  arg2 = f32[1,4,4,2] parameter(2)
  arg3 = f32[1,4,4,2] parameter(3)
  arg4 = f32[1,4,4,2] parameter(4)
  arg5 = f32[1,4,4,2] parameter(5)
  arg4_new = f32[1,4,4,2] add(arg4, arg2)
  arg5_new = f32[1,4,4,2] add(arg5, arg3)
  arg0_new = f32[1,4,4,2] add(arg0, arg4_new)
  arg1_new = f32[1,4,4,2] add(arg1, arg5_new)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0_new, arg1_new, arg4_new, arg5_new)
}

pipeline {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  stage_0 = (f32[1,4,4,2]) call(in0, in1), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[1,4,4,2] get-tuple-element(stage_0), index=0
  stage_1 = (f32[1,4,4,2]) call(stage_0_0, in2), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_1 = f32[1,4,4,2] get-tuple-element(stage_1), index=0
  stage_1_bwd = (f32[1,4,4,2]) call(stage_1_1, in2), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_bwd_1 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=0
  stage_0_bwd = (f32[1,4,4,2]) call(stage_1_bwd_1, in1), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[1,4,4,2] get-tuple-element(stage_0_bwd), index=0
  in3 = f32[1,4,4,2] parameter(2)
  in4 = f32[1,4,4,2] parameter(3)
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(in1, in2, stage_0_bwd_0, stage_1_bwd_1, in3, in4), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"%s\",\"partitionOffloadedVariables\":\"%s\"}}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(call_ru), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(call_ru), index=3
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.in2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  e.in3 = f32[1,4,4,2] parameter(3), parameter_replication={false}
  e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(e.in0, e.in1, e.in2, e.in3), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"offloadVariables\":\"%s\",\"partitionVariables\":\"%s\",\"schedule\":\"%s\",\"batchSerializationIterations\":\"%d\"}}}"
  gte0 = f32[1,4,4,2] get-tuple-element(e.call), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(e.call), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(e.call), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(e.call), index=3
  ROOT t =  (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}
)";
  return absl::StrFormat(
      hlo_format, ThreeState_Name(offload_resource_variables),
      ThreeState_Name(partition_resource_variables),
      ThreeState_Name(offload_variables), ThreeState_Name(partition_variables),
      PoplarBackendConfig::CallConfig::PipelineConfig::Schedule_Name(schedule),
      batch_serialization_iterations);
}

bool MatchesReplicatedParameterLoadFusion(const HloInstruction* inst,
                                          bool sliced) {
  if (!IsReplicatedParameterLoadFusion(inst)) {
    return false;
  }
  if (sliced) {
    const HloInstruction* reshape = inst->fused_expression_root();
    EXPECT_TRUE(reshape->opcode() == HloOpcode::kReshape);
    const HloInstruction* slice = reshape->operand(0);
    EXPECT_TRUE(slice->opcode() == HloOpcode::kSlice);
    const HloInstruction* flatten = slice->operand(0);
    EXPECT_TRUE(flatten->opcode() == HloOpcode::kReshape);
    const HloInstruction* all_gather = flatten->operand(0);
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::AllGather)(all_gather));
    const HloInstruction* param = all_gather->operand(0);
    EXPECT_EQ(param->parameter_number(), 0);
    return true;

  } else {
    const HloInstruction* reshape = inst->fused_expression_root();
    EXPECT_TRUE(reshape->opcode() == HloOpcode::kReshape);
    const HloInstruction* all_gather = reshape->operand(0);
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::AllGather)(all_gather));
    const HloInstruction* param = all_gather->operand(0);
    EXPECT_EQ(param->parameter_number(), 0);
    return true;
  }
}

bool MatchesReplicatedParameterStoreFusion(const HloInstruction* inst,
                                           bool padded) {
  if (!IsReplicatedParameterStoreFusion(inst)) {
    return false;
  }
  if (padded) {
    const HloInstruction* reduce_scatter = inst->fused_expression_root();
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ReduceScatter, reduce_scatter));
    const HloInstruction* pad = reduce_scatter->operand(0);
    EXPECT_TRUE(pad->opcode() == HloOpcode::kPad);
    const HloInstruction* flatten = pad->operand(0);
    EXPECT_TRUE(flatten->opcode() == HloOpcode::kReshape);
    const HloInstruction* param = flatten->operand(0);
    EXPECT_EQ(param->parameter_number(), 0);
    return true;
  } else {
    const HloInstruction* reduce_scatter = inst->fused_expression_root();
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ReduceScatter, reduce_scatter));
    const HloInstruction* reshape = reduce_scatter->operand(0);
    EXPECT_TRUE(reshape->opcode() == HloOpcode::kReshape);
    const HloInstruction* param = reshape->operand(0);
    EXPECT_EQ(param->parameter_number(), 0);
    return true;
  }
}

TEST_F(VariablesOffloadAndPartitionTest, ReplaceRoot) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0, in1)
}

stage_1_fwd {
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1, in2)
}

stage_1_bwd {
  in1_grad = f32[1,4,4,2] parameter(0)
  in2_grad = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1_grad, in2_grad)
}

stage_0_bwd {
  in0_grad = f32[1,4,4,2] parameter(0)
  in1_grad = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0_grad, in1_grad)
}

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  arg2 = f32[1,4,4,2] parameter(2)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0, arg1, arg2)
}

pipeline {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(in1, in0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[1,4,4,2] get-tuple-element(stage_0), index=0
  stage_0_1 = f32[1,4,4,2] get-tuple-element(stage_0), index=1
  stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_0, stage_0_1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_1 = f32[1,4,4,2] get-tuple-element(stage_1), index=0
  stage_1_2 = f32[1,4,4,2] get-tuple-element(stage_1), index=1
  stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_1, stage_1_2), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_bwd_1 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=0
  stage_1_bwd_2 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=1
  stage_0_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_bwd_1, stage_0_0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[1,4,4,2] get-tuple-element(stage_0_bwd), index=0
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_bwd_0, stage_1_bwd_1, stage_1_bwd_2), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\"}}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(call_ru), index=2
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(e.in0, e.in1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kCall);
  auto pipeline = root;

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, true, 0, 1);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_TRUE(changed);
  root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  for (auto operand : root->operands()) {
    EXPECT_EQ(operand->opcode(), HloOpcode::kGetTupleElement);
    EXPECT_EQ(operand->operand(0), pipeline);
  }
}

TEST_F(VariablesOffloadAndPartitionTest, OffloadVariable) {
  std::string hlo = GetHlo(THREESTATE_ON);
  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, true, 0, 1);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_TRUE(changed);
  auto root = module->entry_computation()->root_instruction();
  HloComputation* pipeline_computation =
      root->operand(0)->operand(0)->to_apply();
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  HloComputation* resource_update = (*stages.resource_update)->to_apply();
  EXPECT_EQ(resource_update->num_parameters(), 6);
  EXPECT_EQ(ShapeUtil::TupleElementCount(
                resource_update->root_instruction()->shape()),
            4);

  // Check load and stores.
  for (int64 i : {4, 5}) {
    HloInstruction* param = resource_update->parameter_instruction(i);
    EXPECT_EQ(param->user_count(), 2);
    HloInstruction *load, *store;
    TF_ASSERT_OK(GetRemoteLoadStoreUsers(param, &load, &store));
  }
}

TEST_F(VariablesOffloadAndPartitionTest,
       DontOffloadPipelineVariablesNoPartition) {
  std::string hlo = GetHlo(THREESTATE_ON, THREESTATE_OFF, THREESTATE_UNDEFINED,
                           THREESTATE_OFF);
  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, true, 0, 1);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_TRUE(changed);
  auto root = module->entry_computation()->root_instruction();
  HloComputation* pipeline_computation =
      root->operand(0)->operand(0)->to_apply();
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  HloComputation* resource_update = (*stages.resource_update)->to_apply();
  EXPECT_EQ(resource_update->num_parameters(), 6);
  EXPECT_EQ(ShapeUtil::TupleElementCount(
                resource_update->root_instruction()->shape()),
            4);

  // Check load and stores.
  for (int64 i : {4, 5}) {
    HloInstruction* param = resource_update->parameter_instruction(i);
    EXPECT_EQ(param->user_count(), 2);
    HloInstruction *load, *store;
    TF_ASSERT_OK(GetRemoteLoadStoreUsers(param, &load, &store));
  }

  auto is_not_offloaded = [](const HloInstruction* inst) {
    return !absl::c_any_of(inst->users(),
                           IsPoplarInstruction(PoplarOp::RemoteParameterLoad));
  };
  // Check parameters used in other pipeline stages have not been offloaded.
  for (int64 i : {0, 1}) {
    HloInstruction* param = resource_update->parameter_instruction(i);
    EXPECT_TRUE(is_not_offloaded(param));
  }
  // Check loads inside of pipeline stages.
  for (auto& stages : {stages.forward, stages.backward}) {
    for (auto& stage : stages) {
      HloInstruction* param = stage->to_apply()->parameter_instruction(1);
      EXPECT_TRUE(is_not_offloaded(param));
    }
  }
}

TEST_F(VariablesOffloadAndPartitionTest, OffloadPipelineVariablesNoPartition) {
  std::string hlo =
      GetHlo(THREESTATE_ON, THREESTATE_OFF, THREESTATE_ON, THREESTATE_OFF);
  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, true, 0, 1);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_TRUE(changed);
  auto root = module->entry_computation()->root_instruction();
  HloComputation* pipeline_computation =
      root->operand(0)->operand(0)->to_apply();
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  HloComputation* resource_update = (*stages.resource_update)->to_apply();
  EXPECT_EQ(resource_update->num_parameters(), 6);
  EXPECT_EQ(ShapeUtil::TupleElementCount(
                resource_update->root_instruction()->shape()),
            4);

  // Check load and stores.
  for (int64 i : {0, 1, 4, 5}) {
    HloInstruction* param = resource_update->parameter_instruction(i);
    EXPECT_EQ(param->user_count(), 2);
    HloInstruction *load, *store;
    TF_ASSERT_OK(GetRemoteLoadStoreUsers(param, &load, &store));
  }
  // Check loads inside of pipeline stages.
  for (auto& stages : {stages.forward, stages.backward}) {
    for (auto& stage : stages) {
      HloInstruction* param = stage->to_apply()->parameter_instruction(1);
      EXPECT_EQ(param->user_count(), 1);
      EXPECT_TRUE(IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(
          param->users()[0]));
    }
  }
}

TEST_F(VariablesOffloadAndPartitionTest,
       OffloadPipelineVariablesByDefaultNoPartition) {
  std::string hlo = GetHlo(
      THREESTATE_ON, THREESTATE_OFF, THREESTATE_UNDEFINED, THREESTATE_OFF,
      PoplarBackendConfig::CallConfig::PipelineConfig::Sequential, 4);
  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, true, 0, 1);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_TRUE(changed);
  auto root = module->entry_computation()->root_instruction();
  HloComputation* pipeline_computation =
      root->operand(0)->operand(0)->to_apply();
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  HloComputation* resource_update = (*stages.resource_update)->to_apply();
  EXPECT_EQ(resource_update->num_parameters(), 6);
  EXPECT_EQ(ShapeUtil::TupleElementCount(
                resource_update->root_instruction()->shape()),
            4);

  // Check load and stores.
  for (int64 i : {0, 1, 4, 5}) {
    HloInstruction* param = resource_update->parameter_instruction(i);
    EXPECT_EQ(param->user_count(), 2);
    HloInstruction *load, *store;
    TF_ASSERT_OK(GetRemoteLoadStoreUsers(param, &load, &store));
  }
  // Check loads inside of pipeline stages.
  for (auto& stages : {stages.forward, stages.backward}) {
    for (auto& stage : stages) {
      HloInstruction* param = stage->to_apply()->parameter_instruction(1);
      EXPECT_EQ(param->user_count(), 1);
      EXPECT_TRUE(IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(
          param->users()[0]));
    }
  }
}

TEST_F(VariablesOffloadAndPartitionTest, OffloadVariableRepeat) {
  std::string hlo = R"(
HloModule top

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  arg2 = f32[1,4,4,2] parameter(2)
  arg3 = f32[1,4,4,2] parameter(3)
  arg2_new = f32[1,4,4,2] add(arg2, arg0)
  arg3_new = f32[1,4,4,2] add(arg3, arg1)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0, arg1, arg2_new, arg3_new)
}

loop {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  add1 = f32[1,4,4,2] add(in0, in1)
  in2 = f32[1,4,4,2] parameter(1)
  add2 = f32[1,4,4,2] add(in0, in2)
  in3 = f32[1,4,4,2] parameter(2)
  in4 = f32[1,4,4,2] parameter(3)
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(add1, add2, in3, in4), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\"}}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(call_ru), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(call_ru), index=3
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.in2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  e.in3 = f32[1,4,4,2] parameter(3), parameter_replication={false}
  e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(e.in0, e.in1, e.in2, e.in3), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
  gte0 = f32[1,4,4,2] get-tuple-element(e.call), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(e.call), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(e.call), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(e.call), index=3
  ROOT t =  (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}
)";
  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, true, 0, 1);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_TRUE(changed);
  auto root = module->entry_computation()->root_instruction();
  HloComputation* repeat_computation = root->operand(0)->operand(0)->to_apply();
  HloInstruction* repeat_root = repeat_computation->root_instruction();
  HloComputation* resource_update =
      repeat_root->mutable_operand(0)->mutable_operand(0)->to_apply();
  EXPECT_EQ(resource_update->num_parameters(), 4);
  EXPECT_EQ(ShapeUtil::TupleElementCount(
                resource_update->root_instruction()->shape()),
            4);

  // Check load and stores.
  for (int64 i : {0, 1}) {
    HloInstruction* param = resource_update->parameter_instruction(i);
    EXPECT_EQ(param->user_count(), 2);
    HloInstruction* add = nullptr;
    for (auto* user : param->users()) {
      if (user->opcode() == HloOpcode::kAdd) {
        add = user;
        break;
      }
    }
    EXPECT_TRUE(add);
    const HloInstruction* load = add->operand(0);
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(load));
    EXPECT_EQ(add->user_count(), 1);
    const HloInstruction* store = add->users()[0];
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::RemoteParameterStore)(store));
  }
}

TEST_F(VariablesOffloadAndPartitionTest, OffloadVariableRepeatReadOnly) {
  std::string hlo = R"(
HloModule top

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  arg2 = f32[1,4,4,2] parameter(2)
  arg3 = f32[1,4,4,2] parameter(3)
  arg4 = f32[1,4,4,2] parameter(4)
  arg2_new = f32[1,4,4,2] add(arg2, arg0)
  a = f32[1,4,4,2] add(arg4, arg1)
  arg3_new = f32[1,4,4,2] add(arg3, a)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0, arg1, arg2_new, arg3_new)
}

loop {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  add1 = f32[1,4,4,2] add(in0, in1)
  in2 = f32[1,4,4,2] parameter(1)
  add2 = f32[1,4,4,2] add(in0, in2)
  in3 = f32[1,4,4,2] parameter(2)
  in4 = f32[1,4,4,2] parameter(3)
  in5 = f32[1,4,4,2] parameter(4)
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(add1, add2, in3, in4, in5), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\"}}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(call_ru), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(call_ru), index=3
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3, in5)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.in2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  e.in3 = f32[1,4,4,2] parameter(3), parameter_replication={false}
  e.in4 = f32[1,4,4,2] parameter(4), parameter_replication={false}
  e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(e.in0, e.in1, e.in2, e.in3, e.in4), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
  gte0 = f32[1,4,4,2] get-tuple-element(e.call), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(e.call), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(e.call), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(e.call), index=3
  ROOT t =  (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}
)";
  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3, 4});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, true, 0, 1);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_TRUE(changed);
  auto root = module->entry_computation()->root_instruction();
  HloComputation* repeat_computation = root->operand(0)->operand(0)->to_apply();
  HloInstruction* repeat_root = repeat_computation->root_instruction();
  HloComputation* resource_update =
      repeat_root->mutable_operand(0)->mutable_operand(0)->to_apply();
  EXPECT_EQ(resource_update->num_parameters(), 5);
  EXPECT_EQ(ShapeUtil::TupleElementCount(
                resource_update->root_instruction()->shape()),
            4);

  // Check load and stores.
  for (int64 i : {2, 3}) {
    HloInstruction* param = resource_update->parameter_instruction(i);
    EXPECT_EQ(param->user_count(), 2);
    HloInstruction *load, *store;
    TF_ASSERT_OK(GetRemoteLoadStoreUsers(param, &load, &store));
  }
  // Check load only.
  {
    HloInstruction* param = resource_update->parameter_instruction(4);
    EXPECT_EQ(param->user_count(), 1);
    EXPECT_TRUE(
        IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(param->users()[0]));
  }
}

TEST_F(VariablesOffloadAndPartitionTest, Inference) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0, in1)
}

stage_1_fwd {
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1, in2)
}

pipeline {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(in1, in0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[1,4,4,2] get-tuple-element(stage_0), index=0
  stage_0_1 = f32[1,4,4,2] get-tuple-element(stage_0), index=1
  stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_0, stage_0_1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_1 = f32[1,4,4,2] get-tuple-element(stage_1), index=0
  stage_1_2 = f32[1,4,4,2] get-tuple-element(stage_1), index=1
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1, in2)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.in0, e.in1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(e.call), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(e.call), index=1
  ROOT t =  (f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1)
}
)";
  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, true, 0, 1);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(VariablesOffloadAndPartitionTest, DisabledByDevice) {
  std::string hlo = GetHlo(THREESTATE_ON);
  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, false, 0, 1);
  auto status_or = prvo.Run(module.get());
  EXPECT_FALSE(status_or.ok());
  EXPECT_THAT(status_or.status().error_message(),
              ::testing::StartsWith("Current configuration of the IPU"));
}

TEST_F(VariablesOffloadAndPartitionTest, DisabledByDeviceDefaultConfig) {
  std::string hlo = GetHlo(THREESTATE_UNDEFINED);
  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, false, 0, 1);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(VariablesOffloadAndPartitionTest, DisabledByUser) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0, in1)
}

stage_1_fwd {
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1, in2)
}

stage_1_bwd {
  in1_grad = f32[1,4,4,2] parameter(0)
  in2_grad = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1_grad, in2_grad)
}

stage_0_bwd {
  in0_grad = f32[1,4,4,2] parameter(0)
  in1_grad = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0_grad, in1_grad)
}

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  arg2 = f32[1,4,4,2] parameter(2)
  arg3 = f32[1,4,4,2] parameter(3)
  arg2_new = f32[1,4,4,2] add(arg2, arg0)
  arg3_new = f32[1,4,4,2] add(arg3, arg1)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0, arg1, arg2_new, arg3_new)
}

pipeline {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(in1, in0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[1,4,4,2] get-tuple-element(stage_0), index=0
  stage_0_1 = f32[1,4,4,2] get-tuple-element(stage_0), index=1
  stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_0, stage_0_1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_1 = f32[1,4,4,2] get-tuple-element(stage_1), index=0
  stage_1_2 = f32[1,4,4,2] get-tuple-element(stage_1), index=1
  stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_1, stage_1_2), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_bwd_1 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=0
  stage_0_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_bwd_1, stage_0_0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[1,4,4,2] get-tuple-element(stage_0_bwd), index=0
  in3 = f32[1,4,4,2] parameter(2)
  in4 = f32[1,4,4,2] parameter(3)
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_bwd_0, stage_1_bwd_1, in3, in4), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_OFF\"}}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(call_ru), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(call_ru), index=3
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.in2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  e.in3 = f32[1,4,4,2] parameter(3), parameter_replication={false}
  e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(e.in0, e.in1, e.in2, e.in3), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(e.call), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(e.call), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(e.call), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(e.call), index=3
  ROOT t =  (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}
)";
  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, true, 0, 1);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(VariablesOffloadAndPartitionTest, NonResourceVariables) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0, in1)
}

stage_1_fwd {
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1, in2)
}

stage_1_bwd {
  in1_grad = f32[1,4,4,2] parameter(0)
  in2_grad = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1_grad, in2_grad)
}

stage_0_bwd {
  in0_grad = f32[1,4,4,2] parameter(0)
  in1_grad = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0_grad, in1_grad)
}

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  arg2 = f32[1,4,4,2] parameter(2)
  arg3 = f32[1,4,4,2] parameter(3)
  arg2_new = f32[1,4,4,2] add(arg2, arg0)
  arg3_new = f32[1,4,4,2] add(arg3, arg1)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0, arg1, arg2_new, arg3_new)
}

pipeline {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(in1, in0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[1,4,4,2] get-tuple-element(stage_0), index=0
  stage_0_1 = f32[1,4,4,2] get-tuple-element(stage_0), index=1
  stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_0, stage_0_1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_1 = f32[1,4,4,2] get-tuple-element(stage_1), index=0
  stage_1_2 = f32[1,4,4,2] get-tuple-element(stage_1), index=1
  stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_1, stage_1_2), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_bwd_1 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=0
  stage_0_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_bwd_1, stage_0_0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[1,4,4,2] get-tuple-element(stage_0_bwd), index=0
  in3 = f32[1,4,4,2] parameter(2)
  in4 = f32[1,4,4,2] parameter(3)
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_bwd_0, stage_1_bwd_1, in3, in4), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_OFF\"}}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(call_ru), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(call_ru), index=3
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.in2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  e.in3 = f32[1,4,4,2] parameter(3), parameter_replication={false}
  e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(e.in0, e.in1, e.in2, e.in3), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(e.call), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(e.call), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(e.call), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(e.call), index=3
  ROOT t =  (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}
)";
  auto config = GetModuleConfigForTest();
  // Note that unlike other tests we do not mark inputs/outputs as resources
  // here.
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, true, 0, 1);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(VariablesOffloadAndPartitionTest, PipelineInputOutputDoesntAlign) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0, in1)
}

stage_1_fwd {
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1, in2)
}

stage_1_bwd {
  in1_grad = f32[1,4,4,2] parameter(0)
  in2_grad = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1_grad, in2_grad)
}

stage_0_bwd {
  in0_grad = f32[1,4,4,2] parameter(0)
  in1_grad = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0_grad, in1_grad)
}

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  arg2 = f32[1,4,4,2] parameter(2)
  arg3 = f32[1,4,4,2] parameter(3)
  arg2_new = f32[1,4,4,2] add(arg2, arg0)
  arg3_new = f32[1,4,4,2] add(arg3, arg1)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0, arg1, arg2_new, arg3_new)
}

pipeline {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(in1, in0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[1,4,4,2] get-tuple-element(stage_0), index=0
  stage_0_1 = f32[1,4,4,2] get-tuple-element(stage_0), index=1
  stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_0, stage_0_1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_1 = f32[1,4,4,2] get-tuple-element(stage_1), index=0
  stage_1_2 = f32[1,4,4,2] get-tuple-element(stage_1), index=1
  stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_1, stage_1_2), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_bwd_1 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=0
  stage_0_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_bwd_1, stage_0_0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[1,4,4,2] get-tuple-element(stage_0_bwd), index=0
  in3 = f32[1,4,4,2] parameter(2)
  in4 = f32[1,4,4,2] parameter(3)
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_bwd_0, stage_1_bwd_1, in3, in4), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_OFF\"}}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(call_ru), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(call_ru), index=3
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte3, gte2)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.in2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  e.in3 = f32[1,4,4,2] parameter(3), parameter_replication={false}
  e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(e.in0, e.in1, e.in2, e.in3), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(e.call), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(e.call), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(e.call), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(e.call), index=3
  ROOT t =  (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}
)";
  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, true, 0, 1);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_FALSE(changed);
}

// Add a test case which checks that a read-only offloaded variable is loaded,
// but no store instruction is created.
TEST_F(VariablesOffloadAndPartitionTest, ReadOnly) {
  std::string hlo = R"(
HloModule top

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  arg2 = f32[1,4,4,2] parameter(2)
  arg3 = f32[1,4,4,2] parameter(3)
  arg2_new = f32[1,4,4,2] add(arg2, arg0)
  arg3_new = f32[1,4,4,2] add(arg3, arg1)
  arg3_new_new = f32[1,4,4,2] add(arg3_new, arg2_new)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0, arg1, arg2, arg3_new_new)
}

loop {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  add1 = f32[1,4,4,2] add(in0, in1)
  in2 = f32[1,4,4,2] parameter(1)
  add2 = f32[1,4,4,2] add(in0, in2)
  in3 = f32[1,4,4,2] parameter(2)
  in4 = f32[1,4,4,2] parameter(3)
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(add1, add2, in3, in4), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\"}}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(call_ru), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(call_ru), index=3
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.in2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  e.in3 = f32[1,4,4,2] parameter(3), parameter_replication={false}
  e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(e.in0, e.in1, e.in2, e.in3), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
  gte0 = f32[1,4,4,2] get-tuple-element(e.call), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(e.call), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(e.call), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(e.call), index=3
  ROOT t =  (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}
)";
  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, true, 0, 1);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_TRUE(changed);

  auto root = module->entry_computation()->root_instruction();
  HloComputation* repeat_computation = root->operand(0)->operand(0)->to_apply();
  HloInstruction* repeat_root = repeat_computation->root_instruction();
  HloComputation* resource_update =
      repeat_root->mutable_operand(0)->mutable_operand(0)->to_apply();
  EXPECT_EQ(resource_update->num_parameters(), 4);
  EXPECT_EQ(ShapeUtil::TupleElementCount(
                resource_update->root_instruction()->shape()),
            4);

  // Check there is 1 store and 2 loads.
  auto insts = resource_update->instructions();
  EXPECT_EQ(absl::c_count_if(
                insts, IsPoplarInstruction(PoplarOp::RemoteParameterStore)),
            1);
  EXPECT_EQ(absl::c_count_if(
                insts, IsPoplarInstruction(PoplarOp::RemoteParameterLoad)),
            2);

  HloInstruction* resource_update_root = resource_update->root_instruction();

  // Expect the first three inputs to be pass-through.
  for (int i : {0, 1, 2}) {
    const HloInstruction* operand = resource_update_root->operand(i);
    EXPECT_EQ(operand->opcode(), HloOpcode::kParameter);
  }

  // The final input should be updated with a store.
  const HloInstruction* final_operand = resource_update_root->operand(3);
  EXPECT_EQ(final_operand->opcode(), HloOpcode::kCustomCall);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::RemoteParameterStore)(final_operand));
}

TEST_F(VariablesOffloadAndPartitionTest, OffloadVariableMinimumSize) {
  std::string hlo = R"(
HloModule top

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  arg2 = f32[1,4,4,2] parameter(2)
  arg3 = f32[2,4,4,2] parameter(3)
  arg2_new = f32[1,4,4,2] add(arg2, arg0)
  concat = f32[2,4,4,2] concatenate(arg1, arg1), dimensions={0}
  arg3_new = f32[2,4,4,2] add(arg3, concat)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[2,4,4,2]) tuple(arg0, arg1, arg2_new, arg3_new)
}

loop {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  add1 = f32[1,4,4,2] add(in0, in1)
  in2 = f32[1,4,4,2] parameter(1)
  add2 = f32[1,4,4,2] add(in0, in2)
  in3 = f32[1,4,4,2] parameter(2)
  in4 = f32[2,4,4,2] parameter(3)
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[2,4,4,2]) call(add1, add2, in3, in4), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\"}}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(call_ru), index=2
  gte3 = f32[2,4,4,2] get-tuple-element(call_ru), index=3
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[2,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.in2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  e.in3 = f32[2,4,4,2] parameter(3), parameter_replication={false}
  e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[2,4,4,2]) call(e.in0, e.in1, e.in2, e.in3), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
  gte0 = f32[1,4,4,2] get-tuple-element(e.call), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(e.call), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(e.call), index=2
  gte3 = f32[2,4,4,2] get-tuple-element(e.call), index=3
  ROOT t =  (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[2,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}
)";
  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, true, 129, 1);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_TRUE(changed);
  auto root = module->entry_computation()->root_instruction();
  HloComputation* repeat_computation = root->operand(0)->operand(0)->to_apply();
  HloInstruction* repeat_root = repeat_computation->root_instruction();
  HloComputation* resource_update =
      repeat_root->mutable_operand(0)->mutable_operand(0)->to_apply();
  EXPECT_EQ(resource_update->num_parameters(), 4);
  EXPECT_EQ(ShapeUtil::TupleElementCount(
                resource_update->root_instruction()->shape()),
            4);

  // Check load and stores.
  HloInstruction* param = resource_update->parameter_instruction(3);
  ASSERT_EQ(param->user_count(), 2);
  HloInstruction *load, *store;
  if (IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(param->users()[0])) {
    load = param->users()[0];
    store = param->users()[1];
  } else {
    load = param->users()[1];
    store = param->users()[0];
  }

  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(load));
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::RemoteParameterStore)(store));

  ASSERT_EQ(load->user_count(), 1);
  HloInstruction* add = load->users()[0];
  EXPECT_EQ(add->opcode(), HloOpcode::kAdd);
  ASSERT_EQ(add->user_count(), 1);
  EXPECT_EQ(add->users()[0], store);

  // Check other parameter was not offloaded as it is too small.
  param = resource_update->parameter_instruction(2);
  ASSERT_EQ(param->user_count(), 1);
  add = param->users()[0];
  EXPECT_EQ(add->opcode(), HloOpcode::kAdd);
}

TEST_F(VariablesOffloadAndPartitionTest, OffloadVariableReplicated) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0, in1)
}

stage_1_fwd {
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1, in2)
}

stage_1_bwd {
  in1_grad = f32[1,4,4,2] parameter(0)
  in2_grad = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1_grad, in2_grad)
}

stage_0_bwd {
  in0_grad = f32[1,4,4,2] parameter(0)
  in1_grad = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0_grad, in1_grad)
}

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  arg2 = f32[1,4,4,2] parameter(2)
  arg3 = f32[1,4,4,2] parameter(3)
  arg2_new = f32[1,4,4,2] add(arg2, arg0)
  arg3_new = f32[1,4,4,2] add(arg3, arg1)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0, arg1, arg2_new, arg3_new)
}

pipeline {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(in1, in0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[1,4,4,2] get-tuple-element(stage_0), index=0
  stage_0_1 = f32[1,4,4,2] get-tuple-element(stage_0), index=1
  stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_0, stage_0_1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_1 = f32[1,4,4,2] get-tuple-element(stage_1), index=0
  stage_1_2 = f32[1,4,4,2] get-tuple-element(stage_1), index=1
  stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_1, stage_1_2), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_bwd_1 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=0
  stage_0_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_bwd_1, stage_0_0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[1,4,4,2] get-tuple-element(stage_0_bwd), index=0
  in3 = f32[1,4,4,2] parameter(2)
  in4 = f32[1,4,4,2] parameter(3)
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_bwd_0, stage_1_bwd_1, in3, in4), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\", \"partitionOffloadedVariables\":\"THREESTATE_ON\"}}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(call_ru), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(call_ru), index=3
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.in2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  e.in3 = f32[1,4,4,2] parameter(3), parameter_replication={false}
  e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(e.in0, e.in1, e.in2, e.in3), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(e.call), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(e.call), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(e.call), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(e.call), index=3
  ROOT t =  (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}
)";
  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, true, 0, 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_TRUE(changed);
  auto root = module->entry_computation()->root_instruction();
  HloComputation* pipeline_computation =
      root->operand(0)->operand(0)->to_apply();
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  HloComputation* resource_update = (*stages.resource_update)->to_apply();
  EXPECT_EQ(resource_update->num_parameters(), 4);
  EXPECT_EQ(ShapeUtil::TupleElementCount(
                resource_update->root_instruction()->shape()),
            4);

  // Check load and stores.
  for (int64 i : {0, 1}) {
    HloInstruction* param = resource_update->parameter_instruction(i);
    EXPECT_EQ(param->user_count(), 2);
    HloInstruction* add = nullptr;
    for (auto* user : param->users()) {
      if (user->opcode() == HloOpcode::kAdd) {
        add = user;
        break;
      }
    }
    EXPECT_TRUE(add);
    auto input = add->operand(0);
    EXPECT_TRUE(MatchesReplicatedParameterLoadFusion(input, false));
    EXPECT_TRUE(
        IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(input->operand(0)));

    EXPECT_EQ(add->user_count(), 1);
    auto user = add->users()[0];
    EXPECT_TRUE(MatchesReplicatedParameterStoreFusion(user, false));
    const HloInstruction* store = user->users()[0];
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::RemoteParameterStore)(store));
  }
}

TEST_F(VariablesOffloadAndPartitionTest, OffloadVariableReplicatedNoPartition) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0, in1)
}

stage_1_fwd {
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1, in2)
}

stage_1_bwd {
  in1_grad = f32[1,4,4,2] parameter(0)
  in2_grad = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1_grad, in2_grad)
}

stage_0_bwd {
  in0_grad = f32[1,4,4,2] parameter(0)
  in1_grad = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0_grad, in1_grad)
}

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  arg2 = f32[1,4,4,2] parameter(2)
  arg3 = f32[1,4,4,2] parameter(3)
  arg2_new = f32[1,4,4,2] add(arg2, arg0)
  arg3_new = f32[1,4,4,2] add(arg3, arg1)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0, arg1, arg2_new, arg3_new)
}

pipeline {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(in1, in0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[1,4,4,2] get-tuple-element(stage_0), index=0
  stage_0_1 = f32[1,4,4,2] get-tuple-element(stage_0), index=1
  stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_0, stage_0_1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_1 = f32[1,4,4,2] get-tuple-element(stage_1), index=0
  stage_1_2 = f32[1,4,4,2] get-tuple-element(stage_1), index=1
  stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_1, stage_1_2), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_bwd_1 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=0
  stage_0_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_bwd_1, stage_0_0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[1,4,4,2] get-tuple-element(stage_0_bwd), index=0
  in3 = f32[1,4,4,2] parameter(2)
  in4 = f32[1,4,4,2] parameter(3)
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_bwd_0, stage_1_bwd_1, in3, in4), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\", \"partitionOffloadedVariables\":\"THREESTATE_OFF\"}}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(call_ru), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(call_ru), index=3
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.in2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  e.in3 = f32[1,4,4,2] parameter(3), parameter_replication={false}
  e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(e.in0, e.in1, e.in2, e.in3), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(e.call), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(e.call), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(e.call), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(e.call), index=3
  ROOT t =  (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}
)";
  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, true, 0, 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_TRUE(changed);
  auto root = module->entry_computation()->root_instruction();
  HloComputation* pipeline_computation =
      root->operand(0)->operand(0)->to_apply();
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  HloComputation* resource_update = (*stages.resource_update)->to_apply();
  EXPECT_EQ(resource_update->num_parameters(), 4);
  EXPECT_EQ(ShapeUtil::TupleElementCount(
                resource_update->root_instruction()->shape()),
            4);

  // Check load and stores.
  for (int64 i : {0, 1}) {
    HloInstruction* param = resource_update->parameter_instruction(i);
    EXPECT_EQ(param->user_count(), 2);
    HloInstruction* add = nullptr;
    for (auto* user : param->users()) {
      if (user->opcode() == HloOpcode::kAdd) {
        add = user;
        break;
      }
    }
    EXPECT_TRUE(add);
    const HloInstruction* load = add->operand(0);
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(load));
    EXPECT_EQ(add->user_count(), 1);
    const HloInstruction* store = add->users()[0];
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::RemoteParameterStore)(store));
  }
}

TEST_F(VariablesOffloadAndPartitionTest, OffloadVariableReplicatedPadding) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  in0 = f32[1,5,5,3] parameter(0)
  in1 = f32[1,5,5,3] parameter(1)
  ROOT tuple = (f32[1,5,5,3], f32[1,5,5,3]) tuple(in0, in1)
}

stage_1_fwd {
  in1 = f32[1,5,5,3] parameter(0)
  in2 = f32[1,5,5,3] parameter(1)
  ROOT tuple = (f32[1,5,5,3], f32[1,5,5,3]) tuple(in1, in2)
}

stage_1_bwd {
  in1_grad = f32[1,5,5,3] parameter(0)
  in2_grad = f32[1,5,5,3] parameter(1)
  ROOT tuple = (f32[1,5,5,3], f32[1,5,5,3]) tuple(in1_grad, in2_grad)
}

stage_0_bwd {
  in0_grad = f32[1,5,5,3] parameter(0)
  in1_grad = f32[1,5,5,3] parameter(1)
  ROOT tuple = (f32[1,5,5,3], f32[1,5,5,3]) tuple(in0_grad, in1_grad)
}

resource_update {
  arg0 = f32[1,5,5,3] parameter(0)
  arg1 = f32[1,5,5,3] parameter(1)
  arg2 = f32[1,5,5,3] parameter(2)
  arg3 = f32[1,5,5,3] parameter(3)
  arg2_new = f32[1,5,5,3] add(arg2, arg0)
  arg3_new = f32[1,5,5,3] add(arg3, arg1)
  ROOT t = (f32[1,5,5,3], f32[1,5,5,3], f32[1,5,5,3]) tuple(arg0, arg1, arg2_new, arg3_new)
}

pipeline {
  after-all = token[] after-all()
  infeed = (f32[1,5,5,3], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,5,5,3] get-tuple-element(infeed), index=0
  in1 = f32[1,5,5,3] parameter(0)
  in2 = f32[1,5,5,3] parameter(1)
  stage_0 = (f32[1,5,5,3], f32[1,5,5,3]) call(in1, in0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[1,5,5,3] get-tuple-element(stage_0), index=0
  stage_0_1 = f32[1,5,5,3] get-tuple-element(stage_0), index=1
  stage_1 = (f32[1,5,5,3], f32[1,5,5,3]) call(stage_0_0, stage_0_1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_1 = f32[1,5,5,3] get-tuple-element(stage_1), index=0
  stage_1_2 = f32[1,5,5,3] get-tuple-element(stage_1), index=1
  stage_1_bwd = (f32[1,5,5,3], f32[1,5,5,3]) call(stage_1_1, stage_1_2), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_bwd_1 = f32[1,5,5,3] get-tuple-element(stage_1_bwd), index=0
  stage_0_bwd = (f32[1,5,5,3], f32[1,5,5,3]) call(stage_1_bwd_1, stage_0_0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[1,5,5,3] get-tuple-element(stage_0_bwd), index=0
  in3 = f32[1,5,5,3] parameter(2)
  in4 = f32[1,5,5,3] parameter(3)
  call_ru = (f32[1,5,5,3], f32[1,5,5,3], f32[1,5,5,3], f32[1,5,5,3]) call(stage_0_bwd_0, stage_1_bwd_1, in3, in4), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\", \"partitionOffloadedVariables\":\"THREESTATE_ON\"}}}"
  gte0 = f32[1,5,5,3] get-tuple-element(call_ru), index=0
  gte1 = f32[1,5,5,3] get-tuple-element(call_ru), index=1
  gte2 = f32[1,5,5,3] get-tuple-element(call_ru), index=2
  gte3 = f32[1,5,5,3] get-tuple-element(call_ru), index=3
  ROOT tuple = (f32[1,5,5,3], f32[1,5,5,3], f32[1,5,5,3], f32[1,5,5,3]) tuple(gte0, gte1, gte2, gte3)
}

ENTRY e {
  e.in0 = f32[1,5,5,3] parameter(0), parameter_replication={false}
  e.in1 = f32[1,5,5,3] parameter(1), parameter_replication={false}
  e.in2 = f32[1,5,5,3] parameter(2), parameter_replication={false}
  e.in3 = f32[1,5,5,3] parameter(3), parameter_replication={false}
  e.call = (f32[1,5,5,3], f32[1,5,5,3], f32[1,5,5,3], f32[1,5,5,3]) call(e.in0, e.in1, e.in2, e.in3), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  gte0 = f32[1,5,5,3] get-tuple-element(e.call), index=0
  gte1 = f32[1,5,5,3] get-tuple-element(e.call), index=1
  gte2 = f32[1,5,5,3] get-tuple-element(e.call), index=2
  gte3 = f32[1,5,5,3] get-tuple-element(e.call), index=3
  ROOT t =  (f32[1,5,5,3], f32[1,5,5,3], f32[1,5,5,3], f32[1,5,5,3]) tuple(gte0, gte1, gte2, gte3)
}
)";
  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, true, 0, 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_TRUE(changed);
  auto root = module->entry_computation()->root_instruction();
  HloComputation* pipeline_computation =
      root->operand(0)->operand(0)->to_apply();
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  HloComputation* resource_update = (*stages.resource_update)->to_apply();
  EXPECT_EQ(resource_update->num_parameters(), 4);
  EXPECT_EQ(ShapeUtil::TupleElementCount(
                resource_update->root_instruction()->shape()),
            4);

  // Check load and stores.
  for (int64 i : {0, 1}) {
    HloInstruction* param = resource_update->parameter_instruction(i);
    EXPECT_EQ(param->user_count(), 2);
    HloInstruction* add = nullptr;
    for (auto* user : param->users()) {
      if (user->opcode() == HloOpcode::kAdd) {
        add = user;
        break;
      }
    }
    EXPECT_TRUE(add);

    auto input = add->operand(0);
    EXPECT_TRUE(MatchesReplicatedParameterLoadFusion(input, true));
    EXPECT_TRUE(
        IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(input->operand(0)));

    EXPECT_EQ(add->user_count(), 1);
    auto user = add->users()[0];
    EXPECT_TRUE(MatchesReplicatedParameterStoreFusion(user, true));
    const HloInstruction* store = user->users()[0];
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::RemoteParameterStore)(store));
  }
}

TEST_F(VariablesOffloadAndPartitionTest, ReadOnlyReplicated) {
  std::string hlo = R"(
HloModule top

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  arg2 = f32[1,4,4,2] parameter(2)
  arg3 = f32[1,4,4,2] parameter(3)
  arg2_new = f32[1,4,4,2] add(arg2, arg0)
  arg3_new = f32[1,4,4,2] add(arg3, arg1)
  arg3_new_new = f32[1,4,4,2] add(arg3_new, arg2_new)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0, arg1, arg2, arg3_new_new)
}

loop {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  add1 = f32[1,4,4,2] add(in0, in1)
  in2 = f32[1,4,4,2] parameter(1)
  add2 = f32[1,4,4,2] add(in0, in2)
  in3 = f32[1,4,4,2] parameter(2)
  in4 = f32[1,4,4,2] parameter(3)
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(add1, add2, in3, in4), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\", \"partitionOffloadedVariables\":\"THREESTATE_ON\"}}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(call_ru), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(call_ru), index=3
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.in2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  e.in3 = f32[1,4,4,2] parameter(3), parameter_replication={false}
  e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(e.in0, e.in1, e.in2, e.in3), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
  gte0 = f32[1,4,4,2] get-tuple-element(e.call), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(e.call), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(e.call), index=2
  gte3 = f32[1,4,4,2] get-tuple-element(e.call), index=3
  ROOT t =  (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1, gte2, gte3)
}
)";
  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  VariablesOffloadAndPartition prvo(annotations, true, 0, 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_TRUE(changed);

  auto root = module->entry_computation()->root_instruction();
  HloComputation* repeat_computation = root->operand(0)->operand(0)->to_apply();
  HloInstruction* repeat_root = repeat_computation->root_instruction();
  HloComputation* resource_update =
      repeat_root->mutable_operand(0)->mutable_operand(0)->to_apply();
  EXPECT_EQ(resource_update->num_parameters(), 4);
  EXPECT_EQ(ShapeUtil::TupleElementCount(
                resource_update->root_instruction()->shape()),
            4);

  // Check there is 1 store 2 loads, and 2 replicated load fusions.
  auto insts = resource_update->instructions();
  EXPECT_EQ(absl::c_count_if(
                insts, IsPoplarInstruction(PoplarOp::RemoteParameterStore)),
            1);
  EXPECT_EQ(absl::c_count_if(
                insts, IsPoplarInstruction(PoplarOp::RemoteParameterLoad)),
            2);
  EXPECT_EQ(absl::c_count_if(insts, IsReplicatedParameterLoadFusion), 2);

  HloInstruction* resource_update_root = resource_update->root_instruction();

  // Expect the first three inputs to be pass-through.
  for (int i : {0, 1, 2}) {
    const HloInstruction* operand = resource_update_root->operand(i);
    EXPECT_EQ(operand->opcode(), HloOpcode::kParameter);
  }

  // The final input should be updated with a store.
  const HloInstruction* final_operand = resource_update_root->operand(3);
  EXPECT_EQ(final_operand->opcode(), HloOpcode::kCustomCall);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::RemoteParameterStore)(final_operand));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
