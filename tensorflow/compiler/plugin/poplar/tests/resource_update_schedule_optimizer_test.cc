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

#include "tensorflow/compiler/plugin/poplar/driver/passes/resource_update_schedule_optimizer.h"

#include "tensorflow/compiler/plugin/poplar/driver/passes/inter_ipu_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using ResourceUpdateScheduleOptimizerTest = HloTestBase;

TEST_F(ResourceUpdateScheduleOptimizerTest, TestResourceUpdate) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,2] parameter(1)
  in2 = f32[] parameter(2)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[]) tuple(in0, in1, in2)
}

stage_1_fwd {
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  in3 = f32[] parameter(2)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[]) tuple(in1, in2, in3)
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
  arg3 = f32[] parameter(3)
  bcast = f32[1,4,4,2] broadcast(arg3), dimensions={}
  new_arg0 = f32[1,4,4,2] add(arg0, bcast)
  new_arg1 = f32[1,4,4,2] add(arg1, bcast)
  new_arg2 = f32[1,4,4,2] add(arg2, bcast)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(new_arg0, new_arg1, new_arg2)
}

pipeline {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  in3 = f32[] parameter(2)
  stage_0 = (f32[1,4,4,2], f32[1,4,4,2], f32[]) call(in1, in0, in3), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[1,4,4,2] get-tuple-element(stage_0), index=0
  stage_0_1 = f32[1,4,4,2] get-tuple-element(stage_0), index=1
  stage_0_2 = f32[] get-tuple-element(stage_0), index=2
  stage_1 = (f32[1,4,4,2], f32[1,4,4,2], f32[]) call(stage_0_0, in2, stage_0_2), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_1 = f32[1,4,4,2] get-tuple-element(stage_1), index=0
  stage_1_2 = f32[1,4,4,2] get-tuple-element(stage_1), index=1
  stage_1_3 = f32[] get-tuple-element(stage_1), index=2
  stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_1, stage_1_2), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_bwd_1 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=0
  stage_1_bwd_2 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=1
  stage_0_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_bwd_1, stage_0_0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[1,4,4,2] get-tuple-element(stage_0_bwd), index=0
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_bwd_0, stage_1_bwd_1, stage_1_bwd_2, stage_1_3), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[]) tuple(gte0, gte1, in3)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0)
  e.in1 = f32[1,4,4,2] parameter(1)
  e.in2 = f32[] parameter(2)
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[]) call(e.in0, e.in1, e.in2), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ShardingPass sp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, sp.Run(module.get()));
  EXPECT_TRUE(changed);
  InterIpuCopyInserter iici;
  TF_ASSERT_OK_AND_ASSIGN(changed, iici.Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* pipeline_comp = FindComputation(module.get(), "pipeline");

  ResourceUpdateScheduleOptimizer ruso;
  TF_ASSERT_OK_AND_ASSIGN(changed, ruso.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_comp));

  EXPECT_TRUE(stages.resource_update);
  auto resource_update_comp = (*stages.resource_update)->to_apply();
  auto arg0 = resource_update_comp->parameter_instruction(0);
  auto arg1 = resource_update_comp->parameter_instruction(1);
  auto arg2 = resource_update_comp->parameter_instruction(2);
  auto arg3 = resource_update_comp->parameter_instruction(3);

  std::vector<HloInstruction*> copies;
  absl::c_copy_if(resource_update_comp->instructions(),
                  std::back_inserter(copies), [](HloInstruction* inst) -> bool {
                    return IsPoplarInstruction(PoplarOp::InterIpuCopy)(inst);
                  });
  EXPECT_EQ(copies.size(), 1);
  EXPECT_THAT(arg0->control_predecessors(), ::testing::ElementsAre(copies[0]));
  EXPECT_THAT(arg1->control_predecessors(), ::testing::ElementsAre(copies[0]));
  EXPECT_THAT(arg2->control_predecessors(), ::testing::ElementsAre(copies[0]));
  EXPECT_EQ(arg3->control_predecessors().size(), 0);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
