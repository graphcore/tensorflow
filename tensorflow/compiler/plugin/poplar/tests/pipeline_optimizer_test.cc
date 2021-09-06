/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_optimizer.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using PipelineOptimizerTest = HloTestBase;

TEST_F(PipelineOptimizerTest, TestMoveParameters) {
  std::string hlo = R"(
HloModule top

add_float {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT a = f32[] add(f32[] x, f32[] y)
}

stage_0_fwd {
  stage_0_fwd_t = token[] after-all()
  stage_0_fwd_feed = (f32[1,4,4,2], token[]) infeed(stage_0_fwd_t)
  stage_0_fwd_input = f32[1,4,4,2] get-tuple-element(stage_0_fwd_feed), index=0
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_acts_0 = f32[1,4,4,2] add(stage_0_fwd_input, stage_0_fwd_weights0)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_0_fwd_acts_0, stage_0_fwd_input, stage_0_fwd_weights0)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_1_fwd_acts_1 = f32[1,4,4,2] add(stage_1_fwd_acts_0, stage_1_fwd_weights1)
  stage_1_fwd_zero = f32[] constant(0)
  stage_1_fwd_reduce = f32[] reduce(stage_1_fwd_acts_1, stage_1_fwd_zero), dimensions={0,1,2,3}, to_apply=add_float
  ROOT stage_1_fwd_tuple = (f32[], f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_fwd_reduce, stage_1_fwd_acts_0, stage_1_fwd_weights1)
}

stage_1_bwd {
  stage_1_bwd_reduce = f32[] parameter(0)
  stage_1_bwd_bcast1 = f32[1,4,4,2] broadcast(stage_1_bwd_reduce), dimensions={}
  stage_1_bwd_acts_0 = f32[1,4,4,2] parameter(1)
  stage_1_bwd_acts_0_bwd = f32[1,4,4,2] add(stage_1_bwd_acts_0, stage_1_bwd_bcast1)
  stage_1_bwd_lr = f32[] constant(0.01)
  stage_1_bwd_lr_bcast = f32[1,4,4,2] broadcast(stage_1_bwd_lr), dimensions={}
  stage_1_bwd_update = f32[1,4,4,2] multiply(stage_1_bwd_acts_0_bwd, stage_1_bwd_lr_bcast)
  stage_1_bwd_weights1 = f32[1,4,4,2] parameter(2)
  stage_1_bwd_weights1_new = f32[1,4,4,2] subtract(stage_1_bwd_weights1, stage_1_bwd_update)
  ROOT stage_1_bwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_bwd_acts_0_bwd, stage_1_bwd_weights1_new)
}

stage_0_bwd {
  stage_0_bwd_acts_0_bwd = f32[1,4,4,2] parameter(0)
  stage_0_bwd_input = f32[1,4,4,2] parameter(1)
  stage_0_bwd_input_bwd = f32[1,4,4,2] add(stage_0_bwd_input, stage_0_bwd_acts_0_bwd)
  stage_0_bwd_lr = f32[] constant(0.01)
  stage_0_bwd_lr_bcast = f32[1,4,4,2] broadcast(stage_0_bwd_lr), dimensions={}
  stage_0_bwd_update = f32[1,4,4,2] multiply(stage_0_bwd_input_bwd, stage_0_bwd_lr_bcast)
  stage_0_bwd_weights0 = f32[1,4,4,2] parameter(2)
  stage_0_bwd_weights0_new = f32[1,4,4,2] subtract(stage_0_bwd_weights0, stage_0_bwd_update)
  ROOT stage_0_bwd_tuple = (f32[1,4,4,2]) tuple(stage_0_bwd_weights0_new)
}

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0, arg1)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  pipeline_acts_0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_input = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_weights0_through_stage_0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=2
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[], f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_acts_0, pipeline_weights1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  pipeline_reduce = f32[] get-tuple-element(pipeline_stage_1), index=0
  pipeline_acts_0_local = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  pipeline_weights1_through_stage_1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=2
  pipeline_stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_reduce, pipeline_acts_0_local, pipeline_weights1_through_stage_1), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  pipeline_acts_0_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd), index=0
  pipeline_weights1_new = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd), index=1
  pipeline_stage_0_bwd = (f32[1,4,4,2]) call(pipeline_acts_0_bwd, pipeline_input, pipeline_weights0_through_stage_0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  pipeline_weights0_new = f32[1,4,4,2] get-tuple-element(pipeline_stage_0_bwd), index=0
  call_ru = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0_new, pipeline_weights1_new), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  EXPECT_THAT(stages.backward[0]->operand_count(), 3);
  EXPECT_THAT(stages.backward[0]->operand(2),
              FindInstruction(module0, "pipeline_weights0_through_stage_0"));
  EXPECT_THAT(stages.backward[1]->operand_count(), 3);
  EXPECT_THAT(stages.backward[1]->operand(2),
              FindInstruction(module0, "pipeline_weights1_through_stage_1"));

  PipelineOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, optimizer.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(stages.backward[0]->operand_count(), 3);
  EXPECT_THAT(stages.backward[0]->operand(2),
              FindInstruction(module0, "pipeline_weights0"));
  EXPECT_THAT(stages.backward[1]->operand_count(), 3);
  EXPECT_THAT(stages.backward[1]->operand(2),
              FindInstruction(module0, "pipeline_weights1"));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
