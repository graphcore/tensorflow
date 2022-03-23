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
#include "tensorflow/compiler/plugin/poplar/driver/passes/inter_ipu_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_control_dependency_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

std::vector<HloInstruction*> SelectPipelineStages(
    const std::vector<HloInstruction*>& insts) {
  std::vector<HloInstruction*> result;
  absl::c_copy_if(insts, std::back_inserter(result),
                  IsPipelineStageOrBackwardOp);

  return result;
}

std::vector<HloInstruction*> SelectInterIPUCopies(
    const std::vector<HloInstruction*>& insts) {
  std::vector<HloInstruction*> result;
  absl::c_copy_if(insts, std::back_inserter(result),
                  IsPoplarInstruction(PoplarOp::InterIpuCopy));

  return result;
}

using PipelineControlDepTest = HloTestBase;

TEST_F(PipelineControlDepTest, TestPipeliningShardingWitSubStage) {
  std::string hlo_string = R"(
HloModule top

add_float {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT a = f32[] add(f32[] x, f32[] y)
}

stage_0_0_fwd {
  stage_0_0_fwd_input0 = f32[1,4,4,2] parameter(0)
  ROOT stage_0_0_fwd_tuple = (f32[1,4,4,2]) tuple(stage_0_0_fwd_input0)
}

stage_0_1_fwd {
  stage_0_1_fwd_input0 = f32[1,4,4,2] parameter(0)
  ROOT stage_0_1_fwd_tuple = (f32[1,4,4,2]) tuple(stage_0_1_fwd_input0)
}

stage_0_fwd {
  stage_0_fwd_input0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0_0 = (f32[1,4,4,2]) call(stage_0_fwd_input0), to_apply=stage_0_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_1 = (f32[1,4,4,2]) call(stage_0_fwd_input0), to_apply=stage_0_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_0_0_0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0_0), index=0
  pipeline_stage_0_1_0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0_1), index=0
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_0_0_0, pipeline_stage_0_1_0)
}

comp0 {
  ROOT x = f32[1,4,4,2] parameter(0)
}

comp1 {
  x = f32[1,4,4,2] parameter(0)
  ROOT r = f32[1,4,4,2] call(x), to_apply=comp0
}

comp2 {
  ROOT x = f32[1,4,4,2] parameter(0)
}

stage_1_fwd {
  stage_1_fwd_input1 = f32[1,4,4,2] parameter(0)
  c1_result = f32[1,4,4,2] call(stage_1_fwd_input1), to_apply=comp1
  c2_result = f32[1,4,4,2] call(c1_result), to_apply=comp2
  stage_1_fwd_zero = f32[] constant(0)
  stage_1_fwd_reduce = f32[] reduce(c2_result, stage_1_fwd_zero), dimensions={0,1,2,3}, to_apply=add_float
  ROOT stage_1_fwd_tuple = (f32[]) tuple(stage_1_fwd_reduce)
  after-all = token[] after-all()
  outfeed = token[] outfeed(stage_1_fwd_tuple, after-all), outfeed_config="\010\001\022\005feed3\"\001\001(\001"
}

pipeline {
  pipeline_input0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_input0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=-1}
  pipeline_stage_0_i0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_input1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[]) call(pipeline_input1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_i1 = f32[] get-tuple-element(pipeline_stage_1), index=0
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[]) tuple(pipeline_stage_0_i0, pipeline_stage_1_i1)
}

ENTRY e {
  e.input0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.input1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[]) call(e.input0, e.input1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  InterIpuCopyInserter interIpuCopyInserterPass;
  ASSERT_TRUE(interIpuCopyInserterPass.Run(module).ValueOrDie());

  PipelineControlDependencyInserter pipelineControlDependencyInserterPass;
  ASSERT_TRUE(pipelineControlDependencyInserterPass.Run(module).ValueOrDie());

  auto stage_0 = module->GetComputationWithName("stage_0_fwd");

  auto insts = stage_0->MakeInstructionPostOrder();
  auto substages = SelectPipelineStages(insts);
  auto inter_ipu_copies = SelectInterIPUCopies(insts);
  absl::c_sort(substages);

  // Each inter-ipu-copy must either be before *all* substages or after *all*
  // substages.
  for (auto inter_ipu_copy : inter_ipu_copies) {
    if (inter_ipu_copy->sharding().GetUniqueDevice() == -1) {
      auto predecessors = inter_ipu_copy->control_predecessors();
      absl::c_sort(predecessors);
      ASSERT_EQ(predecessors, substages);
    } else {
      auto successors = inter_ipu_copy->control_successors();
      absl::c_sort(successors);
      ASSERT_EQ(successors, substages);
    }
  }
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
