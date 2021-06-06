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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_resource_update_input_optimizer.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/fifo.h"
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

using PipelineResourceUpdateInputOptimizerTest = HloTestBase;

TEST_F(PipelineResourceUpdateInputOptimizerTest, TestOutputFromNonLastStage) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_t = token[] after-all()
  stage_0_fwd_feed = (f32[], token[]) infeed(stage_0_fwd_t)
  stage_0_fwd_input = f32[] get-tuple-element(stage_0_fwd_feed), index=0
  stage_0_fwd_weights0 = f32[] parameter(0)
  stage_0_fwd_acts_0 = f32[] add(stage_0_fwd_input, stage_0_fwd_weights0)
  ROOT stage_0_fwd_tuple = (f32[], f32[]) tuple(stage_0_fwd_acts_0, stage_0_fwd_weights0)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[] parameter(0)
  stage_1_fwd_weights1 = f32[] parameter(1)
  stage_1_fwd_acts_1 = f32[] add(stage_1_fwd_acts_0, stage_1_fwd_weights1) stage_1_fwd_zero = f32[] constant(0)
  ROOT stage_1_fwd_tuple = (f32[], f32[]) tuple(stage_1_fwd_acts_1, stage_1_fwd_weights1)
}

stage_2_fwd {
  stage_2_fwd_acts_0 = f32[] parameter(0)
  stage_2_fwd_acts_1 = f32[] parameter(1)
  stage_2_fwd_acts_2 = f32[] add(stage_2_fwd_acts_0, stage_2_fwd_acts_1)
  ROOT stage_2_fwd_tuple = (f32[], f32[]) tuple(stage_2_fwd_acts_2, stage_2_fwd_acts_1)
}

stage_2_bwd {
  stage_2_fwd_acts_2 = f32[] parameter(0)
  stage_2_fwd_acts_1 = f32[] parameter(1)
  stage_2_fwd_acts_2_bwd = f32[] subtract(stage_2_fwd_acts_2, stage_2_fwd_acts_1)
  ROOT stage_2_bwd_tuple = (f32[]) tuple(stage_2_fwd_acts_2_bwd)
}

stage_1_bwd {
  stage_1_fwd_acts_2_bwd = f32[] parameter(0)
  stage_1_fwd_acts_1 = f32[] parameter(1)
  stage_1_fwd_acts_1_bwd = f32[] subtract(stage_1_fwd_acts_2_bwd, stage_1_fwd_acts_1)
  ROOT stage_1_bwd_tuple = (f32[]) tuple(stage_1_fwd_acts_1_bwd)
}

stage_0_bwd {
  stage_0_fwd_acts_1_bwd = f32[] parameter(0)
  stage_0_fwd_acts_0 = f32[] parameter(1)
  stage_0_fwd_acts_0_bwd = f32[] subtract(stage_0_fwd_acts_1_bwd, stage_0_fwd_acts_0)
  ROOT stage_0_bwd_tuple = (f32[]) tuple(stage_0_fwd_acts_0_bwd)
}

resource_update {
  arg0 = f32[] parameter(0)
  arg1 = f32[] parameter(1)
  add = f32[] add(arg0, arg1)
  ROOT t = (f32[]) tuple(add)
}

pipeline {
  pipeline_weights0 = f32[] parameter(0)
  pipeline_stage_0 = (f32[], f32[]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}

  stage_0_fwd.0 = f32[] get-tuple-element(pipeline_stage_0), index=0
  stage_0_fwd.1 = f32[] get-tuple-element(pipeline_stage_0), index=1

  pipeline_stage_1 = (f32[], f32[]) call(stage_0_fwd.0, stage_0_fwd.1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_fwd.0 = f32[] get-tuple-element(pipeline_stage_1), index=0
  stage_1_fwd.1 = f32[] get-tuple-element(pipeline_stage_1), index=1

  pipeline_stage_2 = (f32[], f32[]) call(stage_0_fwd.0, stage_1_fwd.0), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_fwd = f32[] get-tuple-element(pipeline_stage_2), index=0
  stage_1_fwd_x = f32[] get-tuple-element(pipeline_stage_2), index=1

  pipeline_stage_2_bwd = (f32[]) call(stage_2_fwd, stage_1_fwd_x), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_bwd = f32[] get-tuple-element(pipeline_stage_2_bwd), index=0

  pipeline_stage_1_bwd = (f32[]) call(stage_2_bwd, stage_1_fwd.0), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_bwd = f32[] get-tuple-element(pipeline_stage_1_bwd), index=0

  pipeline_stage_0_bwd = (f32[]) call(stage_1_bwd, stage_0_fwd.0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd = f32[] get-tuple-element(pipeline_stage_0_bwd), index=0

  call_ru = (f32[]) call(pipeline_weights0, stage_0_fwd.1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  ROOT pipeline_tuple = (f32[]) tuple(gte0)
}

ENTRY e {
  e.weights0 = f32[] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();

  PipelineResourceUpdateInputOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PipelineResourceUpdateInputOptimizer().Run(module.get()));
  EXPECT_TRUE(changed);
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  HloInstruction* resource_update = *stages.resource_update;
  EXPECT_EQ(resource_update->operand(1),
            pipeline_computation->parameter_instruction(0));
}

TEST_F(PipelineResourceUpdateInputOptimizerTest, TestOutputFromLastStage) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_t = token[] after-all()
  stage_0_fwd_feed = (f32[], token[]) infeed(stage_0_fwd_t)
  stage_0_fwd_input = f32[] get-tuple-element(stage_0_fwd_feed), index=0
  stage_0_fwd_weights0 = f32[] parameter(0)
  stage_0_fwd_acts_0 = f32[] add(stage_0_fwd_input, stage_0_fwd_weights0)
  ROOT stage_0_fwd_tuple = (f32[], f32[]) tuple(stage_0_fwd_acts_0, stage_0_fwd_weights0)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[] parameter(0)
  stage_1_fwd_weights1 = f32[] parameter(1)
  stage_1_fwd_acts_1 = f32[] add(stage_1_fwd_acts_0, stage_1_fwd_weights1) stage_1_fwd_zero = f32[] constant(0)
  ROOT stage_1_fwd_tuple = (f32[], f32[]) tuple(stage_1_fwd_acts_1, stage_1_fwd_weights1)
}

stage_2_fwd {
  stage_2_fwd_acts_0 = f32[] parameter(0)
  stage_2_fwd_acts_1 = f32[] parameter(1)
  stage_2_fwd_weights2 = f32[] parameter(2)
  stage_2_fwd_acts_2 = f32[] add(stage_2_fwd_acts_0, stage_2_fwd_acts_1)
  ROOT stage_2_fwd_tuple = (f32[], f32[], f32[]) tuple(stage_2_fwd_acts_2, stage_2_fwd_acts_1, stage_2_fwd_weights2)
}

stage_2_bwd {
  stage_2_fwd_acts_2 = f32[] parameter(0)
  stage_2_fwd_acts_1 = f32[] parameter(1)
  stage_2_fwd_acts_2_bwd = f32[] subtract(stage_2_fwd_acts_2, stage_2_fwd_acts_1)
  ROOT stage_2_bwd_tuple = (f32[]) tuple(stage_2_fwd_acts_2_bwd)
}

stage_1_bwd {
  stage_1_fwd_acts_2_bwd = f32[] parameter(0)
  stage_1_fwd_acts_1 = f32[] parameter(1)
  stage_1_fwd_acts_1_bwd = f32[] subtract(stage_1_fwd_acts_2_bwd, stage_1_fwd_acts_1)
  ROOT stage_1_bwd_tuple = (f32[]) tuple(stage_1_fwd_acts_1_bwd)
}

stage_0_bwd {
  stage_0_fwd_acts_1_bwd = f32[] parameter(0)
  stage_0_fwd_acts_0 = f32[] parameter(1)
  stage_0_fwd_acts_0_bwd = f32[] subtract(stage_0_fwd_acts_1_bwd, stage_0_fwd_acts_0)
  ROOT stage_0_bwd_tuple = (f32[]) tuple(stage_0_fwd_acts_0_bwd)
}

resource_update {
  arg0 = f32[] parameter(0)
  arg1 = f32[] parameter(1)
  add = f32[] add(arg0, arg1)
  ROOT t = (f32[]) tuple(add)
}

pipeline {
  pipeline_weights0 = f32[] parameter(0)
  pipeline_stage_0 = (f32[], f32[]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}

  stage_0_fwd.0 = f32[] get-tuple-element(pipeline_stage_0), index=0
  stage_0_fwd.1 = f32[] get-tuple-element(pipeline_stage_0), index=1

  pipeline_stage_1 = (f32[], f32[]) call(stage_0_fwd.0, stage_0_fwd.1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_fwd.0 = f32[] get-tuple-element(pipeline_stage_1), index=0
  stage_1_fwd.1 = f32[] get-tuple-element(pipeline_stage_1), index=1

  pipeline_stage_2 = (f32[], f32[], f32[]) call(stage_0_fwd.0, stage_1_fwd.0, stage_1_fwd.1), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_fwd = f32[] get-tuple-element(pipeline_stage_2), index=0
  stage_1_fwd_x = f32[] get-tuple-element(pipeline_stage_2), index=1
  stage_weights = f32[] get-tuple-element(pipeline_stage_2), index=2

  pipeline_stage_2_bwd = (f32[]) call(stage_2_fwd, stage_1_fwd_x), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_bwd = f32[] get-tuple-element(pipeline_stage_2_bwd), index=0

  pipeline_stage_1_bwd = (f32[]) call(stage_2_bwd, stage_1_fwd.0), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_bwd = f32[] get-tuple-element(pipeline_stage_1_bwd), index=0

  pipeline_stage_0_bwd = (f32[]) call(stage_1_bwd, stage_0_fwd.0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd = f32[] get-tuple-element(pipeline_stage_0_bwd), index=0

  call_ru = (f32[]) call(pipeline_weights0, stage_weights), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  ROOT pipeline_tuple = (f32[]) tuple(gte0)
}

ENTRY e {
  e.weights0 = f32[] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();

  PipelineResourceUpdateInputOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PipelineResourceUpdateInputOptimizer().Run(module.get()));
  EXPECT_TRUE(changed);
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  HloInstruction* resource_update = *stages.resource_update;
  EXPECT_EQ(resource_update->operand(1),
            pipeline_computation->parameter_instruction(0));
}

TEST_F(PipelineResourceUpdateInputOptimizerTest, TestModifier) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_t = token[] after-all()
  stage_0_fwd_feed = (f32[], token[]) infeed(stage_0_fwd_t)
  stage_0_fwd_input = f32[] get-tuple-element(stage_0_fwd_feed), index=0
  stage_0_fwd_weights0 = f32[] parameter(0)
  stage_0_fwd_acts_0 = f32[] add(stage_0_fwd_input, stage_0_fwd_weights0)
  ROOT stage_0_fwd_tuple = (f32[], f32[]) tuple(stage_0_fwd_acts_0, stage_0_fwd_weights0)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[] parameter(0)
  stage_1_fwd_weights1 = f32[] parameter(1)
  stage_1_fwd_acts_1 = f32[] add(stage_1_fwd_acts_0, stage_1_fwd_weights1) stage_1_fwd_zero = f32[] constant(0)
  ROOT stage_1_fwd_tuple = (f32[], f32[]) tuple(stage_1_fwd_acts_1, stage_1_fwd_weights1)
}

stage_2_fwd {
  stage_2_fwd_acts_0 = f32[] parameter(0)
  stage_2_fwd_acts_1 = f32[] parameter(1)
  stage_2_fwd_weights2 = f32[] parameter(2)
  const = f32[] constant(0.125)
  mul = f32[] multiply(const, stage_2_fwd_weights2)
  stage_2_fwd_acts_2 = f32[] add(stage_2_fwd_acts_0, stage_2_fwd_acts_1)
  ROOT stage_2_fwd_tuple = (f32[], f32[], f32[]) tuple(stage_2_fwd_acts_2, stage_2_fwd_acts_1, mul)
}

stage_2_bwd {
  stage_2_fwd_acts_2 = f32[] parameter(0)
  stage_2_fwd_acts_1 = f32[] parameter(1)
  stage_2_fwd_acts_2_bwd = f32[] subtract(stage_2_fwd_acts_2, stage_2_fwd_acts_1)
  ROOT stage_2_bwd_tuple = (f32[]) tuple(stage_2_fwd_acts_2_bwd)
}

stage_1_bwd {
  stage_1_fwd_acts_2_bwd = f32[] parameter(0)
  stage_1_fwd_acts_1 = f32[] parameter(1)
  stage_1_fwd_acts_1_bwd = f32[] subtract(stage_1_fwd_acts_2_bwd, stage_1_fwd_acts_1)
  ROOT stage_1_bwd_tuple = (f32[]) tuple(stage_1_fwd_acts_1_bwd)
}

stage_0_bwd {
  stage_0_fwd_acts_1_bwd = f32[] parameter(0)
  stage_0_fwd_acts_0 = f32[] parameter(1)
  stage_0_fwd_acts_0_bwd = f32[] subtract(stage_0_fwd_acts_1_bwd, stage_0_fwd_acts_0)
  ROOT stage_0_bwd_tuple = (f32[]) tuple(stage_0_fwd_acts_0_bwd)
}

resource_update {
  arg0 = f32[] parameter(0)
  arg1 = f32[] parameter(1)
  add = f32[] add(arg0, arg1)
  ROOT t = (f32[]) tuple(add)
}

pipeline {
  pipeline_weights0 = f32[] parameter(0)
  pipeline_stage_0 = (f32[], f32[]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}

  stage_0_fwd.0 = f32[] get-tuple-element(pipeline_stage_0), index=0
  stage_0_fwd.1 = f32[] get-tuple-element(pipeline_stage_0), index=1

  pipeline_stage_1 = (f32[], f32[]) call(stage_0_fwd.0, stage_0_fwd.1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_fwd.0 = f32[] get-tuple-element(pipeline_stage_1), index=0
  stage_1_fwd.1 = f32[] get-tuple-element(pipeline_stage_1), index=1

  pipeline_stage_2 = (f32[], f32[], f32[]) call(stage_0_fwd.0, stage_1_fwd.0, stage_1_fwd.1), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_fwd = f32[] get-tuple-element(pipeline_stage_2), index=0
  stage_1_fwd_x = f32[] get-tuple-element(pipeline_stage_2), index=1
  stage_weights = f32[] get-tuple-element(pipeline_stage_2), index=2

  pipeline_stage_2_bwd = (f32[]) call(stage_2_fwd, stage_1_fwd_x), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_bwd = f32[] get-tuple-element(pipeline_stage_2_bwd), index=0

  pipeline_stage_1_bwd = (f32[]) call(stage_2_bwd, stage_1_fwd.0), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_bwd = f32[] get-tuple-element(pipeline_stage_1_bwd), index=0

  pipeline_stage_0_bwd = (f32[]) call(stage_1_bwd, stage_0_fwd.0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd = f32[] get-tuple-element(pipeline_stage_0_bwd), index=0

  call_ru = (f32[]) call(pipeline_weights0, stage_weights), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  ROOT pipeline_tuple = (f32[]) tuple(gte0)
}

ENTRY e {
  e.weights0 = f32[] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();

  PipelineResourceUpdateInputOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PipelineResourceUpdateInputOptimizer().Run(module.get()));
  EXPECT_TRUE(changed);
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  HloInstruction* resource_update = *stages.resource_update;
  EXPECT_EQ(resource_update->operand(1),
            pipeline_computation->parameter_instruction(0));
  HloComputation* resource_update_comp = resource_update->to_apply();
  EXPECT_TRUE(Match(
      resource_update_comp->root_instruction(),
      m::Tuple(m::Add(m::Parameter(0), m::Multiply(m::ConstantScalar(0.125),
                                                   m::Parameter(1))))));
}

TEST_F(PipelineResourceUpdateInputOptimizerTest, TestModifierMultipleUsers) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_t = token[] after-all()
  stage_0_fwd_feed = (f32[], token[]) infeed(stage_0_fwd_t)
  stage_0_fwd_input = f32[] get-tuple-element(stage_0_fwd_feed), index=0
  stage_0_fwd_weights0 = f32[] parameter(0)
  stage_0_fwd_acts_0 = f32[] add(stage_0_fwd_input, stage_0_fwd_weights0)
  ROOT stage_0_fwd_tuple = (f32[], f32[]) tuple(stage_0_fwd_acts_0, stage_0_fwd_weights0)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[] parameter(0)
  stage_1_fwd_weights1 = f32[] parameter(1)
  stage_1_fwd_acts_1 = f32[] add(stage_1_fwd_acts_0, stage_1_fwd_weights1) stage_1_fwd_zero = f32[] constant(0)
  ROOT stage_1_fwd_tuple = (f32[], f32[]) tuple(stage_1_fwd_acts_1, stage_1_fwd_weights1)
}

stage_2_fwd {
  stage_2_fwd_acts_0 = f32[] parameter(0)
  stage_2_fwd_acts_1 = f32[] parameter(1)
  stage_2_fwd_weights2 = f32[] parameter(2)
  const = f32[] constant(0.125)
  mul = f32[] multiply(const, stage_2_fwd_weights2)
  stage_2_fwd_acts_2 = f32[] add(stage_2_fwd_acts_0, stage_2_fwd_acts_1)
  t = (f32[]) tuple(stage_2_fwd_weights2)
  ROOT stage_2_fwd_tuple = (f32[], f32[], f32[]) tuple(stage_2_fwd_acts_2, stage_2_fwd_acts_1, mul)
}

stage_2_bwd {
  stage_2_fwd_acts_2 = f32[] parameter(0)
  stage_2_fwd_acts_1 = f32[] parameter(1)
  stage_2_fwd_acts_2_bwd = f32[] subtract(stage_2_fwd_acts_2, stage_2_fwd_acts_1)
  ROOT stage_2_bwd_tuple = (f32[]) tuple(stage_2_fwd_acts_2_bwd)
}

stage_1_bwd {
  stage_1_fwd_acts_2_bwd = f32[] parameter(0)
  stage_1_fwd_acts_1 = f32[] parameter(1)
  stage_1_fwd_acts_1_bwd = f32[] subtract(stage_1_fwd_acts_2_bwd, stage_1_fwd_acts_1)
  ROOT stage_1_bwd_tuple = (f32[]) tuple(stage_1_fwd_acts_1_bwd)
}

stage_0_bwd {
  stage_0_fwd_acts_1_bwd = f32[] parameter(0)
  stage_0_fwd_acts_0 = f32[] parameter(1)
  stage_0_fwd_acts_0_bwd = f32[] subtract(stage_0_fwd_acts_1_bwd, stage_0_fwd_acts_0)
  ROOT stage_0_bwd_tuple = (f32[]) tuple(stage_0_fwd_acts_0_bwd)
}

resource_update {
  arg0 = f32[] parameter(0)
  arg1 = f32[] parameter(1)
  add = f32[] add(arg0, arg1)
  ROOT t = (f32[]) tuple(add)
}

pipeline {
  pipeline_weights0 = f32[] parameter(0)
  pipeline_stage_0 = (f32[], f32[]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}

  stage_0_fwd.0 = f32[] get-tuple-element(pipeline_stage_0), index=0
  stage_0_fwd.1 = f32[] get-tuple-element(pipeline_stage_0), index=1

  pipeline_stage_1 = (f32[], f32[]) call(stage_0_fwd.0, stage_0_fwd.1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_fwd.0 = f32[] get-tuple-element(pipeline_stage_1), index=0
  stage_1_fwd.1 = f32[] get-tuple-element(pipeline_stage_1), index=1

  pipeline_stage_2 = (f32[], f32[], f32[]) call(stage_0_fwd.0, stage_1_fwd.0, stage_1_fwd.1), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_fwd = f32[] get-tuple-element(pipeline_stage_2), index=0
  stage_1_fwd_x = f32[] get-tuple-element(pipeline_stage_2), index=1
  stage_weights = f32[] get-tuple-element(pipeline_stage_2), index=2

  pipeline_stage_2_bwd = (f32[]) call(stage_2_fwd, stage_1_fwd_x), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_bwd = f32[] get-tuple-element(pipeline_stage_2_bwd), index=0

  pipeline_stage_1_bwd = (f32[]) call(stage_2_bwd, stage_1_fwd.0), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_bwd = f32[] get-tuple-element(pipeline_stage_1_bwd), index=0

  pipeline_stage_0_bwd = (f32[]) call(stage_1_bwd, stage_0_fwd.0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd = f32[] get-tuple-element(pipeline_stage_0_bwd), index=0

  call_ru = (f32[]) call(pipeline_weights0, stage_weights), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  ROOT pipeline_tuple = (f32[]) tuple(gte0)
}

ENTRY e {
  e.weights0 = f32[] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();

  PipelineResourceUpdateInputOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PipelineResourceUpdateInputOptimizer().Run(module.get()));
  EXPECT_TRUE(changed);
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  HloInstruction* resource_update = *stages.resource_update;
  EXPECT_EQ(resource_update->operand(1),
            pipeline_computation->parameter_instruction(0));
  HloComputation* resource_update_comp = resource_update->to_apply();
  EXPECT_TRUE(Match(
      resource_update_comp->root_instruction(),
      m::Tuple(m::Add(m::Parameter(0), m::Multiply(m::ConstantScalar(0.125),
                                                   m::Parameter(1))))));
}

TEST_F(PipelineResourceUpdateInputOptimizerTest, TestUnaryModifier) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_t = token[] after-all()
  stage_0_fwd_feed = (f32[], token[]) infeed(stage_0_fwd_t)
  stage_0_fwd_input = f32[] get-tuple-element(stage_0_fwd_feed), index=0
  stage_0_fwd_weights0 = f32[] parameter(0)
  stage_0_fwd_acts_0 = f32[] add(stage_0_fwd_input, stage_0_fwd_weights0)
  ROOT stage_0_fwd_tuple = (f32[], f32[]) tuple(stage_0_fwd_acts_0, stage_0_fwd_weights0)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[] parameter(0)
  stage_1_fwd_weights1 = f32[] parameter(1)
  stage_1_fwd_acts_1 = f32[] add(stage_1_fwd_acts_0, stage_1_fwd_weights1) stage_1_fwd_zero = f32[] constant(0)
  ROOT stage_1_fwd_tuple = (f32[], f32[]) tuple(stage_1_fwd_acts_1, stage_1_fwd_weights1)
}

stage_2_fwd {
  stage_2_fwd_acts_0 = f32[] parameter(0)
  stage_2_fwd_acts_1 = f32[] parameter(1)
  stage_2_fwd_weights2 = f32[] parameter(2)
  log = f32[] log(stage_2_fwd_weights2)
  stage_2_fwd_acts_2 = f32[] add(stage_2_fwd_acts_0, stage_2_fwd_acts_1)
  ROOT stage_2_fwd_tuple = (f32[], f32[], f32[]) tuple(stage_2_fwd_acts_2, stage_2_fwd_acts_1, log)
}

stage_2_bwd {
  stage_2_fwd_acts_2 = f32[] parameter(0)
  stage_2_fwd_acts_1 = f32[] parameter(1)
  stage_2_fwd_acts_2_bwd = f32[] subtract(stage_2_fwd_acts_2, stage_2_fwd_acts_1)
  ROOT stage_2_bwd_tuple = (f32[]) tuple(stage_2_fwd_acts_2_bwd)
}

stage_1_bwd {
  stage_1_fwd_acts_2_bwd = f32[] parameter(0)
  stage_1_fwd_acts_1 = f32[] parameter(1)
  stage_1_fwd_acts_1_bwd = f32[] subtract(stage_1_fwd_acts_2_bwd, stage_1_fwd_acts_1)
  ROOT stage_1_bwd_tuple = (f32[]) tuple(stage_1_fwd_acts_1_bwd)
}

stage_0_bwd {
  stage_0_fwd_acts_1_bwd = f32[] parameter(0)
  stage_0_fwd_acts_0 = f32[] parameter(1)
  stage_0_fwd_acts_0_bwd = f32[] subtract(stage_0_fwd_acts_1_bwd, stage_0_fwd_acts_0)
  ROOT stage_0_bwd_tuple = (f32[]) tuple(stage_0_fwd_acts_0_bwd)
}

resource_update {
  arg0 = f32[] parameter(0)
  arg1 = f32[] parameter(1)
  add = f32[] add(arg0, arg1)
  ROOT t = (f32[]) tuple(add)
}

pipeline {
  pipeline_weights0 = f32[] parameter(0)
  pipeline_stage_0 = (f32[], f32[]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}

  stage_0_fwd.0 = f32[] get-tuple-element(pipeline_stage_0), index=0
  stage_0_fwd.1 = f32[] get-tuple-element(pipeline_stage_0), index=1

  pipeline_stage_1 = (f32[], f32[]) call(stage_0_fwd.0, stage_0_fwd.1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_fwd.0 = f32[] get-tuple-element(pipeline_stage_1), index=0
  stage_1_fwd.1 = f32[] get-tuple-element(pipeline_stage_1), index=1

  pipeline_stage_2 = (f32[], f32[], f32[]) call(stage_0_fwd.0, stage_1_fwd.0, stage_1_fwd.1), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_fwd = f32[] get-tuple-element(pipeline_stage_2), index=0
  stage_1_fwd_x = f32[] get-tuple-element(pipeline_stage_2), index=1
  stage_weights = f32[] get-tuple-element(pipeline_stage_2), index=2

  pipeline_stage_2_bwd = (f32[]) call(stage_2_fwd, stage_1_fwd_x), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_bwd = f32[] get-tuple-element(pipeline_stage_2_bwd), index=0

  pipeline_stage_1_bwd = (f32[]) call(stage_2_bwd, stage_1_fwd.0), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_bwd = f32[] get-tuple-element(pipeline_stage_1_bwd), index=0

  pipeline_stage_0_bwd = (f32[]) call(stage_1_bwd, stage_0_fwd.0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd = f32[] get-tuple-element(pipeline_stage_0_bwd), index=0

  call_ru = (f32[]) call(pipeline_weights0, stage_weights), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  ROOT pipeline_tuple = (f32[]) tuple(gte0)
}

ENTRY e {
  e.weights0 = f32[] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();

  PipelineResourceUpdateInputOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PipelineResourceUpdateInputOptimizer().Run(module.get()));
  EXPECT_TRUE(changed);
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  HloInstruction* resource_update = *stages.resource_update;
  EXPECT_EQ(resource_update->operand(1),
            pipeline_computation->parameter_instruction(0));
  HloComputation* resource_update_comp = resource_update->to_apply();
  EXPECT_TRUE(
      Match(resource_update_comp->root_instruction(),
            m::Tuple(m::Add(m::Parameter(0), m::Log(m::Parameter(1))))));
}

TEST_F(PipelineResourceUpdateInputOptimizerTest, TestInvalidModifier) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_t = token[] after-all()
  stage_0_fwd_feed = (f32[], token[]) infeed(stage_0_fwd_t)
  stage_0_fwd_input = f32[] get-tuple-element(stage_0_fwd_feed), index=0
  stage_0_fwd_weights0 = f32[] parameter(0)
  stage_0_fwd_acts_0 = f32[] add(stage_0_fwd_input, stage_0_fwd_weights0)
  ROOT stage_0_fwd_tuple = (f32[], f32[]) tuple(stage_0_fwd_acts_0, stage_0_fwd_weights0)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[] parameter(0)
  stage_1_fwd_weights1 = f32[] parameter(1)
  stage_1_fwd_acts_1 = f32[] add(stage_1_fwd_acts_0, stage_1_fwd_weights1) stage_1_fwd_zero = f32[] constant(0)
  ROOT stage_1_fwd_tuple = (f32[], f32[]) tuple(stage_1_fwd_acts_1, stage_1_fwd_weights1)
}

stage_2_fwd {
  stage_2_fwd_acts_0 = f32[] parameter(0)
  stage_2_fwd_acts_1 = f32[] parameter(1)
  stage_2_fwd_weights2 = f32[] parameter(2)
  add = f32[] add(stage_2_fwd_weights2, stage_2_fwd_acts_0)
  stage_2_fwd_acts_2 = f32[] add(stage_2_fwd_acts_0, stage_2_fwd_acts_1)
  ROOT stage_2_fwd_tuple = (f32[], f32[], f32[]) tuple(stage_2_fwd_acts_2, stage_2_fwd_acts_1, add)
}

stage_2_bwd {
  stage_2_fwd_acts_2 = f32[] parameter(0)
  stage_2_fwd_acts_1 = f32[] parameter(1)
  stage_2_fwd_acts_2_bwd = f32[] subtract(stage_2_fwd_acts_2, stage_2_fwd_acts_1)
  ROOT stage_2_bwd_tuple = (f32[]) tuple(stage_2_fwd_acts_2_bwd)
}

stage_1_bwd {
  stage_1_fwd_acts_2_bwd = f32[] parameter(0)
  stage_1_fwd_acts_1 = f32[] parameter(1)
  stage_1_fwd_acts_1_bwd = f32[] subtract(stage_1_fwd_acts_2_bwd, stage_1_fwd_acts_1)
  ROOT stage_1_bwd_tuple = (f32[]) tuple(stage_1_fwd_acts_1_bwd)
}

stage_0_bwd {
  stage_0_fwd_acts_1_bwd = f32[] parameter(0)
  stage_0_fwd_acts_0 = f32[] parameter(1)
  stage_0_fwd_acts_0_bwd = f32[] subtract(stage_0_fwd_acts_1_bwd, stage_0_fwd_acts_0)
  ROOT stage_0_bwd_tuple = (f32[]) tuple(stage_0_fwd_acts_0_bwd)
}

resource_update {
  arg0 = f32[] parameter(0)
  arg1 = f32[] parameter(1)
  add = f32[] add(arg0, arg1)
  ROOT t = (f32[]) tuple(add)
}

pipeline {
  pipeline_weights0 = f32[] parameter(0)
  pipeline_stage_0 = (f32[], f32[]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}

  stage_0_fwd.0 = f32[] get-tuple-element(pipeline_stage_0), index=0
  stage_0_fwd.1 = f32[] get-tuple-element(pipeline_stage_0), index=1

  pipeline_stage_1 = (f32[], f32[]) call(stage_0_fwd.0, stage_0_fwd.1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_fwd.0 = f32[] get-tuple-element(pipeline_stage_1), index=0
  stage_1_fwd.1 = f32[] get-tuple-element(pipeline_stage_1), index=1

  pipeline_stage_2 = (f32[], f32[], f32[]) call(stage_0_fwd.0, stage_1_fwd.0, stage_1_fwd.1), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_fwd = f32[] get-tuple-element(pipeline_stage_2), index=0
  stage_1_fwd_x = f32[] get-tuple-element(pipeline_stage_2), index=1
  stage_weights = f32[] get-tuple-element(pipeline_stage_2), index=2

  pipeline_stage_2_bwd = (f32[]) call(stage_2_fwd, stage_1_fwd_x), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_bwd = f32[] get-tuple-element(pipeline_stage_2_bwd), index=0

  pipeline_stage_1_bwd = (f32[]) call(stage_2_bwd, stage_1_fwd.0), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_bwd = f32[] get-tuple-element(pipeline_stage_1_bwd), index=0

  pipeline_stage_0_bwd = (f32[]) call(stage_1_bwd, stage_0_fwd.0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd = f32[] get-tuple-element(pipeline_stage_0_bwd), index=0

  call_ru = (f32[]) call(pipeline_weights0, stage_weights), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  ROOT pipeline_tuple = (f32[]) tuple(gte0)
}

ENTRY e {
  e.weights0 = f32[] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();

  PipelineResourceUpdateInputOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PipelineResourceUpdateInputOptimizer().Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(PipelineResourceUpdateInputOptimizerTest, TestMultipleModifiers) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_t = token[] after-all()
  stage_0_fwd_feed = (f32[], token[]) infeed(stage_0_fwd_t)
  stage_0_fwd_input = f32[] get-tuple-element(stage_0_fwd_feed), index=0
  stage_0_fwd_weights0 = f32[] parameter(0)
  one = f32[] constant(1)
  stage_0_fwd_weights0_new = f32[] add(stage_0_fwd_weights0, one)
  ROOT stage_0_fwd_tuple = (f32[], f32[]) tuple(stage_0_fwd_input, stage_0_fwd_weights0_new)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[] parameter(0)
  stage_1_fwd_weights1 = f32[] parameter(1)
  one = f32[] constant(1)
  stage_1_fwd_weights1_new = f32[] add(stage_1_fwd_weights1, one)
  ROOT stage_1_fwd_tuple = (f32[], f32[]) tuple(stage_1_fwd_acts_0, stage_1_fwd_weights1_new)
}

stage_2_fwd {
  stage_2_fwd_acts_0 = f32[] parameter(0)
  stage_2_fwd_acts_1 = f32[] parameter(1)
  stage_2_fwd_acts_2 = f32[] add(stage_2_fwd_acts_0, stage_2_fwd_acts_1)
  ROOT stage_2_fwd_tuple = (f32[], f32[]) tuple(stage_2_fwd_acts_2, stage_2_fwd_acts_1)
}

stage_2_bwd {
  stage_2_fwd_acts_2 = f32[] parameter(0)
  stage_2_fwd_acts_1 = f32[] parameter(1)
  stage_2_fwd_acts_2_bwd = f32[] subtract(stage_2_fwd_acts_2, stage_2_fwd_acts_1)
  ROOT stage_2_bwd_tuple = (f32[]) tuple(stage_2_fwd_acts_2_bwd)
}

stage_1_bwd {
  stage_1_fwd_acts_2_bwd = f32[] parameter(0)
  stage_1_fwd_acts_1 = f32[] parameter(1)
  stage_1_fwd_acts_1_bwd = f32[] subtract(stage_1_fwd_acts_2_bwd, stage_1_fwd_acts_1)
  ROOT stage_1_bwd_tuple = (f32[]) tuple(stage_1_fwd_acts_1_bwd)
}

stage_0_bwd {
  stage_0_fwd_acts_1_bwd = f32[] parameter(0)
  stage_0_fwd_acts_0 = f32[] parameter(1)
  stage_0_fwd_acts_0_bwd = f32[] subtract(stage_0_fwd_acts_1_bwd, stage_0_fwd_acts_0)
  ROOT stage_0_bwd_tuple = (f32[]) tuple(stage_0_fwd_acts_0_bwd)
}

resource_update {
  arg0 = f32[] parameter(0)
  arg1 = f32[] parameter(1)
  add = f32[] add(arg0, arg1)
  ROOT t = (f32[]) tuple(add)
}

pipeline {
  pipeline_weights0 = f32[] parameter(0)
  pipeline_stage_0 = (f32[], f32[]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}

  stage_0_fwd.0 = f32[] get-tuple-element(pipeline_stage_0), index=0
  stage_0_fwd.1 = f32[] get-tuple-element(pipeline_stage_0), index=1

  pipeline_stage_1 = (f32[], f32[]) call(stage_0_fwd.0, stage_0_fwd.1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_fwd.0 = f32[] get-tuple-element(pipeline_stage_1), index=0
  stage_1_fwd.1 = f32[] get-tuple-element(pipeline_stage_1), index=1

  pipeline_stage_2 = (f32[], f32[]) call(stage_0_fwd.0, stage_1_fwd.0), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_fwd = f32[] get-tuple-element(pipeline_stage_2), index=0
  stage_1_fwd_x = f32[] get-tuple-element(pipeline_stage_2), index=1

  pipeline_stage_2_bwd = (f32[]) call(stage_2_fwd, stage_1_fwd_x), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_bwd = f32[] get-tuple-element(pipeline_stage_2_bwd), index=0

  pipeline_stage_1_bwd = (f32[]) call(stage_2_bwd, stage_1_fwd.0), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_bwd = f32[] get-tuple-element(pipeline_stage_1_bwd), index=0

  pipeline_stage_0_bwd = (f32[]) call(stage_1_bwd, stage_0_fwd.0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd = f32[] get-tuple-element(pipeline_stage_0_bwd), index=0

  call_ru = (f32[]) call(pipeline_weights0, stage_1_fwd.1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  ROOT pipeline_tuple = (f32[]) tuple(gte0)
}

ENTRY e {
  e.weights0 = f32[] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();

  PipelineResourceUpdateInputOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PipelineResourceUpdateInputOptimizer().Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(PipelineResourceUpdateInputOptimizerTest, TestIndirectModifier) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_t = token[] after-all()
  stage_0_fwd_feed = (f32[], token[]) infeed(stage_0_fwd_t)
  stage_0_fwd_input = f32[] get-tuple-element(stage_0_fwd_feed), index=0
  stage_0_fwd_weights0 = f32[] parameter(0)
  one = f32[] constant(1)
  stage_0_fwd_weights0_new = f32[] add(stage_0_fwd_weights0, one)
  ROOT stage_0_fwd_tuple = (f32[], f32[]) tuple(stage_0_fwd_input, stage_0_fwd_weights0_new)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[] parameter(0)
  stage_1_fwd_weights1 = f32[] parameter(1)
  ROOT stage_1_fwd_tuple = (f32[], f32[]) tuple(stage_1_fwd_acts_0, stage_1_fwd_weights1)
}

stage_2_fwd {
  stage_2_fwd_acts_0 = f32[] parameter(0)
  stage_2_fwd_acts_1 = f32[] parameter(1)
  stage_2_fwd_acts_2 = f32[] add(stage_2_fwd_acts_0, stage_2_fwd_acts_1)
  ROOT stage_2_fwd_tuple = (f32[], f32[]) tuple(stage_2_fwd_acts_2, stage_2_fwd_acts_1)
}

stage_2_bwd {
  stage_2_fwd_acts_2 = f32[] parameter(0)
  stage_2_fwd_acts_1 = f32[] parameter(1)
  stage_2_fwd_acts_2_bwd = f32[] subtract(stage_2_fwd_acts_2, stage_2_fwd_acts_1)
  ROOT stage_2_bwd_tuple = (f32[]) tuple(stage_2_fwd_acts_2_bwd)
}

stage_1_bwd {
  stage_1_fwd_acts_2_bwd = f32[] parameter(0)
  stage_1_fwd_acts_1 = f32[] parameter(1)
  stage_1_fwd_acts_1_bwd = f32[] subtract(stage_1_fwd_acts_2_bwd, stage_1_fwd_acts_1)
  ROOT stage_1_bwd_tuple = (f32[]) tuple(stage_1_fwd_acts_1_bwd)
}

stage_0_bwd {
  stage_0_fwd_acts_1_bwd = f32[] parameter(0)
  stage_0_fwd_acts_0 = f32[] parameter(1)
  stage_0_fwd_acts_0_bwd = f32[] subtract(stage_0_fwd_acts_1_bwd, stage_0_fwd_acts_0)
  ROOT stage_0_bwd_tuple = (f32[]) tuple(stage_0_fwd_acts_0_bwd)
}

resource_update {
  arg0 = f32[] parameter(0)
  arg1 = f32[] parameter(1)
  add = f32[] add(arg0, arg1)
  ROOT t = (f32[]) tuple(add)
}

pipeline {
  pipeline_weights0 = f32[] parameter(0)
  pipeline_stage_0 = (f32[], f32[]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}

  stage_0_fwd.0 = f32[] get-tuple-element(pipeline_stage_0), index=0
  stage_0_fwd.1 = f32[] get-tuple-element(pipeline_stage_0), index=1

  pipeline_stage_1 = (f32[], f32[]) call(stage_0_fwd.0, stage_0_fwd.1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_fwd.0 = f32[] get-tuple-element(pipeline_stage_1), index=0
  stage_1_fwd.1 = f32[] get-tuple-element(pipeline_stage_1), index=1

  pipeline_stage_2 = (f32[], f32[]) call(stage_0_fwd.0, stage_1_fwd.0), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_fwd = f32[] get-tuple-element(pipeline_stage_2), index=0
  stage_1_fwd_x = f32[] get-tuple-element(pipeline_stage_2), index=1

  pipeline_stage_2_bwd = (f32[]) call(stage_2_fwd, stage_1_fwd_x), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_bwd = f32[] get-tuple-element(pipeline_stage_2_bwd), index=0

  pipeline_stage_1_bwd = (f32[]) call(stage_2_bwd, stage_1_fwd.0), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_bwd = f32[] get-tuple-element(pipeline_stage_1_bwd), index=0

  pipeline_stage_0_bwd = (f32[]) call(stage_1_bwd, stage_0_fwd.0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd = f32[] get-tuple-element(pipeline_stage_0_bwd), index=0

  call_ru = (f32[]) call(pipeline_weights0, stage_1_fwd.1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  ROOT pipeline_tuple = (f32[]) tuple(gte0)
}

ENTRY e {
  e.weights0 = f32[] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();

  PipelineResourceUpdateInputOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PipelineResourceUpdateInputOptimizer().Run(module.get()));
  EXPECT_TRUE(changed);
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  HloInstruction* resource_update = *stages.resource_update;
  EXPECT_EQ(resource_update->operand(1),
            pipeline_computation->parameter_instruction(0));
  HloComputation* resource_update_comp = resource_update->to_apply();
  EXPECT_TRUE(
      Match(resource_update_comp->root_instruction(),
            m::Tuple(m::Add(m::Parameter(0),
                            m::Add(m::Parameter(1), m::ConstantScalar(1))))));
}

TEST_F(PipelineResourceUpdateInputOptimizerTest, TestCastModifier) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_t = token[] after-all()
  stage_0_fwd_feed = (f32[], token[]) infeed(stage_0_fwd_t)
  stage_0_fwd_input = f32[] get-tuple-element(stage_0_fwd_feed), index=0
  stage_0_fwd_weights0 = f32[] parameter(0)
  stage_0_fwd_acts_0 = f32[] add(stage_0_fwd_input, stage_0_fwd_weights0)
  ROOT stage_0_fwd_tuple = (f32[], f32[]) tuple(stage_0_fwd_acts_0, stage_0_fwd_weights0)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[] parameter(0)
  stage_1_fwd_weights1 = f32[] parameter(1)
  stage_1_fwd_acts_1 = f32[] add(stage_1_fwd_acts_0, stage_1_fwd_weights1) stage_1_fwd_zero = f32[] constant(0)
  ROOT stage_1_fwd_tuple = (f32[], f32[]) tuple(stage_1_fwd_acts_1, stage_1_fwd_weights1)
}

stage_2_fwd {
  stage_2_fwd_acts_0 = f32[] parameter(0)
  stage_2_fwd_acts_1 = f32[] parameter(1)
  stage_2_fwd_weights2 = f32[] parameter(2)
  log = f16[] convert(stage_2_fwd_weights2)
  stage_2_fwd_acts_2 = f32[] add(stage_2_fwd_acts_0, stage_2_fwd_acts_1)
  ROOT stage_2_fwd_tuple = (f32[], f32[], f16[]) tuple(stage_2_fwd_acts_2, stage_2_fwd_acts_1, log)
}

stage_2_bwd {
  stage_2_fwd_acts_2 = f32[] parameter(0)
  stage_2_fwd_acts_1 = f32[] parameter(1)
  stage_2_fwd_acts_2_bwd = f32[] subtract(stage_2_fwd_acts_2, stage_2_fwd_acts_1)
  ROOT stage_2_bwd_tuple = (f32[]) tuple(stage_2_fwd_acts_2_bwd)
}

stage_1_bwd {
  stage_1_fwd_acts_2_bwd = f32[] parameter(0)
  stage_1_fwd_acts_1 = f32[] parameter(1)
  stage_1_fwd_acts_1_bwd = f32[] subtract(stage_1_fwd_acts_2_bwd, stage_1_fwd_acts_1)
  ROOT stage_1_bwd_tuple = (f32[]) tuple(stage_1_fwd_acts_1_bwd)
}

stage_0_bwd {
  stage_0_fwd_acts_1_bwd = f32[] parameter(0)
  stage_0_fwd_acts_0 = f32[] parameter(1)
  stage_0_fwd_acts_0_bwd = f32[] subtract(stage_0_fwd_acts_1_bwd, stage_0_fwd_acts_0)
  ROOT stage_0_bwd_tuple = (f32[]) tuple(stage_0_fwd_acts_0_bwd)
}

resource_update {
  arg0 = f32[] parameter(0)
  arg1 = f16[] parameter(1)
  c_arg0 = f16[] convert(arg0)
  add = f16[] add(c_arg0, arg1)
  c_add = f32[] convert(add)
  ROOT t = (f32[]) tuple(c_add)
}

pipeline {
  pipeline_weights0 = f32[] parameter(0)
  pipeline_stage_0 = (f32[], f32[]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}

  stage_0_fwd.0 = f32[] get-tuple-element(pipeline_stage_0), index=0
  stage_0_fwd.1 = f32[] get-tuple-element(pipeline_stage_0), index=1

  pipeline_stage_1 = (f32[], f32[]) call(stage_0_fwd.0, stage_0_fwd.1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_fwd.0 = f32[] get-tuple-element(pipeline_stage_1), index=0
  stage_1_fwd.1 = f32[] get-tuple-element(pipeline_stage_1), index=1

  pipeline_stage_2 = (f32[], f32[], f16[]) call(stage_0_fwd.0, stage_1_fwd.0, stage_1_fwd.1), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_fwd = f32[] get-tuple-element(pipeline_stage_2), index=0
  stage_1_fwd_x = f32[] get-tuple-element(pipeline_stage_2), index=1
  stage_weights = f16[] get-tuple-element(pipeline_stage_2), index=2

  pipeline_stage_2_bwd = (f32[]) call(stage_2_fwd, stage_1_fwd_x), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_bwd = f32[] get-tuple-element(pipeline_stage_2_bwd), index=0

  pipeline_stage_1_bwd = (f32[]) call(stage_2_bwd, stage_1_fwd.0), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_bwd = f32[] get-tuple-element(pipeline_stage_1_bwd), index=0

  pipeline_stage_0_bwd = (f32[]) call(stage_1_bwd, stage_0_fwd.0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd = f32[] get-tuple-element(pipeline_stage_0_bwd), index=0

  call_ru = (f32[]) call(pipeline_weights0, stage_weights), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  ROOT pipeline_tuple = (f32[]) tuple(gte0)
}

ENTRY e {
  e.weights0 = f32[] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();

  PipelineResourceUpdateInputOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, PipelineResourceUpdateInputOptimizer().Run(module.get()));
  EXPECT_TRUE(changed);
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  HloInstruction* resource_update = *stages.resource_update;
  EXPECT_EQ(resource_update->operand(1),
            pipeline_computation->parameter_instruction(0));
  HloComputation* resource_update_comp = resource_update->to_apply();
  EXPECT_EQ(
      resource_update_comp->parameter_instruction(1)->shape().element_type(),
      F32);
  EXPECT_TRUE(Match(resource_update_comp->root_instruction(),
                    m::Tuple(m::Convert(m::Add(m::Convert(m::Parameter(0)),
                                               m::Convert(m::Parameter(1)))))));
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
