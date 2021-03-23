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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_stage_merger.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using PipelineStageMergerTest = HloTestBase;

TEST_F(PipelineStageMergerTest, Sequential) {
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
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2]) tuple(stage_0_fwd_acts_0)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights1 = f32[1,4,4,2] parameter(1)
  x = f32[1,4,4,2] add(stage_1_fwd_acts_0, stage_1_fwd_weights1) stage_1_fwd_zero = f32[] constant(0)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2]) tuple(x)
}

stage_1_bwd {
  stage_1_fwd_acts_2_bwd = f32[1,4,4,2] parameter(0)
  x = f32[1,4,4,2] parameter(1)
  x_bwd = f32[1,4,4,2] subtract(stage_1_fwd_acts_2_bwd, x)
  ROOT stage_1_bwd_tuple = (f32[1,4,4,2]) tuple(x_bwd)
}

stage_0_bwd {
  stage_0_fwd_acts_1_bwd = f32[1,4,4,2] parameter(0)
  stage_0_fwd_acts_0 = f32[1,4,4,2] parameter(1)
  stage_0_fwd_acts_0_bwd = f32[1,4,4,2] subtract(stage_0_fwd_acts_1_bwd, stage_0_fwd_acts_0)
  ROOT stage_0_bwd_tuple = (f32[1,4,4,2]) tuple(stage_0_fwd_acts_0_bwd)
}

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0, arg1)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0), sharding={maximal device=0}
  pipeline_stage_0 = (f32[1,4,4,2]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={{maximal device=0}}

  stage_0_fwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0, sharding={maximal device=0}
  pipeline_weights1 = f32[1,4,4,2] parameter(1), sharding={maximal device=0}

  pipeline_stage_1 = (f32[1,4,4,2]) call(stage_0_fwd, pipeline_weights1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={{maximal device=0}, {maximal device=0}}
  stage_1_fwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0, sharding={maximal device=0}

  pipeline_stage_1_bwd = (f32[1,4,4,2]) call(pipeline_weights1, stage_1_fwd), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={{maximal device=0}}
  stage_1_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd), index=0, sharding={maximal device=0}

  pipeline_stage_0_bwd = (f32[1,4,4,2]) call(stage_1_bwd, stage_0_fwd), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={{maximal device=0}}
  stage_0_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_0_bwd), index=0, sharding={maximal device=0}

  call_ru = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":2}}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          PipelineStageMerger().Run(module.get()));
  // Expect changes.
  EXPECT_TRUE(changed);
  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));

  // Expect the forward stage to just pass a value through.
  EXPECT_TRUE(Match(stages.forward[1]->to_apply()->root_instruction(),
                    m::Tuple(m::Parameter(0))));

  // Expect the backward stage to do all the computation.
  EXPECT_TRUE(
      Match(stages.backward[1]->to_apply()->root_instruction(),
            m::Tuple(m::Add(m::Parameter(0), m::Parameter(1)),
                     m::Subtract(m::Parameter(1),
                                 m::Add(m::Parameter(0), m::Parameter(1))))));
  EXPECT_THAT(stages.backward[1]->operand_count(), 2);
  EXPECT_THAT(stages.backward[1]->operand(0)->opcode(),
              HloOpcode::kGetTupleElement);
  EXPECT_THAT(stages.backward[1]->operand(0)->operand(0), stages.forward[1]);
  EXPECT_THAT(stages.backward[1]->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_THAT(stages.backward[1]->operand(1)->parameter_number(), 1);
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
