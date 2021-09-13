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

#include "tensorflow/compiler/plugin/poplar/driver/passes/call_optimizer.h"
#include <chrono>
#include <thread>

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

using CallOptimizerTest = HloTestBase;

TEST_F(CallOptimizerTest, TestRemoveUnusedOutputs) {
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
  stage_0_fwd_input = f32[1,4,4,2] get-tuple-element(stage_0_fwd_feed),
  index=0 stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0) stage_0_fwd_acts_0
  = f32[1,4,4,2] add(stage_0_fwd_input, stage_0_fwd_weights0) ROOT
  stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_0_fwd_acts_0,
  stage_0_fwd_input)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_1_fwd_acts_1 = f32[1,4,4,2] add(stage_1_fwd_acts_0,
  stage_1_fwd_weights1) stage_1_fwd_zero = f32[] constant(0)
  stage_1_fwd_reduce = f32[] reduce(stage_1_fwd_acts_1, stage_1_fwd_zero),
  dimensions={0,1,2,3}, to_apply=add_float ROOT stage_1_fwd_tuple = (f32[],
  f32[1,4,4,2]) tuple(stage_1_fwd_reduce, stage_1_fwd_acts_0)
}

stage_1_bwd {
  stage_1_bwd_reduce = f32[] parameter(0)
  stage_1_bwd_bcast1 = f32[1,4,4,2] broadcast(stage_1_bwd_reduce),
  dimensions={} stage_1_bwd_acts_0 = f32[1,4,4,2] parameter(1)
  stage_1_bwd_acts_0_bwd = f32[1,4,4,2] add(stage_1_bwd_acts_0,
  stage_1_bwd_bcast1) stage_1_bwd_lr = f32[] constant(0.01)
  stage_1_bwd_lr_bcast = f32[1,4,4,2] broadcast(stage_1_bwd_lr),
  dimensions={} stage_1_bwd_update = f32[1,4,4,2]
  multiply(stage_1_bwd_acts_0_bwd, stage_1_bwd_lr_bcast) stage_1_bwd_weights1
  = f32[1,4,4,2] parameter(2) stage_1_bwd_weights1_new = f32[1,4,4,2]
  subtract(stage_1_bwd_weights1, stage_1_bwd_update) ROOT stage_1_bwd_tuple =
  (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_bwd_acts_0_bwd,
  stage_1_bwd_weights1_new, stage_1_bwd_update)
}

stage_0_bwd {
  stage_0_bwd_acts_0_bwd = f32[1,4,4,2] parameter(0)
  stage_0_bwd_input = f32[1,4,4,2] parameter(1)
  stage_0_bwd_input_bwd = f32[1,4,4,2] add(stage_0_bwd_input,
  stage_0_bwd_acts_0_bwd) stage_0_bwd_lr = f32[] constant(0.01)
  stage_0_bwd_lr_bcast = f32[1,4,4,2] broadcast(stage_0_bwd_lr),
  dimensions={} stage_0_bwd_update = f32[1,4,4,2]
  multiply(stage_0_bwd_input_bwd, stage_0_bwd_lr_bcast) stage_0_bwd_weights0
  = f32[1,4,4,2] parameter(2) stage_0_bwd_weights0_new = f32[1,4,4,2]
  subtract(stage_0_bwd_weights0, stage_0_bwd_update) ROOT stage_0_bwd_tuple =
  (f32[1,4,4,2]) tuple(stage_0_bwd_weights0_new)
}

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0, arg1)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0),
    to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  pipeline_acts_0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_input = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[], f32[1,4,4,2]) call(pipeline_acts_0,
  pipeline_weights1), to_apply=stage_1_fwd,
  backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  pipeline_reduce = f32[] get-tuple-element(pipeline_stage_1), index=0
  pipeline_acts_0_local = f32[1,4,4,2] get-tuple-element(pipeline_stage_1),
  index=1 pipeline_stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2])
  call(pipeline_reduce, pipeline_acts_0_local, pipeline_weights1),
  to_apply=stage_1_bwd,
  backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  pipeline_acts_0_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd),
  index=0 pipeline_weights1_new = f32[1,4,4,2]
  get-tuple-element(pipeline_stage_1_bwd), index=1 pipeline_stage_0_bwd =
  (f32[1,4,4,2]) call(pipeline_acts_0_bwd, pipeline_input,
  pipeline_weights0), to_apply=stage_0_bwd,
  backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  pipeline_weights0_new = f32[1,4,4,2]
  get-tuple-element(pipeline_stage_0_bwd), index=0
  call_ru = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0_new, pipeline_weights1_new), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1),
  to_apply=pipeline,
  backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();

  HloInstruction* pipeline_stage_1_bwd =
      FindInstruction(module0, "pipeline_stage_1_bwd");
  EXPECT_THAT(ShapeUtil::TupleElementCount(pipeline_stage_1_bwd->shape()), 3);
  HloInstruction* pipeline_stage_1_bwd_root =
      pipeline_stage_1_bwd->to_apply()->root_instruction();
  std::vector<HloInstruction*> expected_outputs(
      pipeline_stage_1_bwd_root->operands().begin(),
      std::next(pipeline_stage_1_bwd_root->operands().begin(), 2));

  CallOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, optimizer.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(ShapeUtil::TupleElementCount(pipeline_stage_1_bwd->shape()), 2);
  pipeline_stage_1_bwd_root =
      pipeline_stage_1_bwd->to_apply()->root_instruction();
  std::vector<HloInstruction*> outputs(
      pipeline_stage_1_bwd_root->operands().begin(),
      pipeline_stage_1_bwd_root->operands().end());
  EXPECT_THAT(outputs, expected_outputs);
}

TEST_F(CallOptimizerTest, TestRemoveUnusedInputs) {
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
  stage_0_fwd_input = f32[1,4,4,2] get-tuple-element(stage_0_fwd_feed),
  index=0 stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0) stage_0_fwd_acts_0
  = f32[1,4,4,2] add(stage_0_fwd_input, stage_0_fwd_weights0) ROOT
  stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_0_fwd_acts_0,
  stage_0_fwd_input)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_1_fwd_acts_1 = f32[1,4,4,2] add(stage_1_fwd_acts_0,
  stage_1_fwd_weights1) stage_1_fwd_zero = f32[] constant(0)
  stage_1_fwd_reduce = f32[] reduce(stage_1_fwd_acts_1, stage_1_fwd_zero),
  dimensions={0,1,2,3}, to_apply=add_float ROOT stage_1_fwd_tuple = (f32[],
  f32[1,4,4,2]) tuple(stage_1_fwd_reduce, stage_1_fwd_acts_0)
}

stage_1_bwd {
  stage_1_bwd_reduce = f32[] parameter(0)
  stage_1_bwd_bcast1 = f32[1,4,4,2] broadcast(stage_1_bwd_reduce),
  dimensions={} stage_1_bwd_acts_0 = f32[1,4,4,2] parameter(1)
  stage_1_bwd_acts_0_bwd = f32[1,4,4,2] add(stage_1_bwd_acts_0,
  stage_1_bwd_bcast1) stage_1_bwd_lr = f32[] constant(0.01)
  stage_1_bwd_lr_bcast = f32[1,4,4,2] broadcast(stage_1_bwd_lr),
  dimensions={} stage_1_bwd_update = f32[1,4,4,2]
  multiply(stage_1_bwd_acts_0_bwd, stage_1_bwd_lr_bcast) stage_1_bwd_weights1
  = f32[1,4,4,2] parameter(2) stage_1_bwd_weights1_new = f32[1,4,4,2]
  subtract(stage_1_bwd_weights1, stage_1_bwd_update) ROOT stage_1_bwd_tuple =
  (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_bwd_acts_0_bwd,
  stage_1_bwd_weights1_new, stage_1_bwd_update)
}

stage_0_bwd {
  stage_0_bwd_weights0_unused = f32[1,4,4,2] parameter(1)
  stage_0_bwd_acts_0_bwd = f32[1,4,4,2] parameter(0)
  stage_0_bwd_input = f32[1,4,4,2] parameter(2)
  stage_0_bwd_input_bwd = f32[1,4,4,2] add(stage_0_bwd_input,
  stage_0_bwd_acts_0_bwd) stage_0_bwd_lr = f32[] constant(0.01)
  stage_0_bwd_lr_bcast = f32[1,4,4,2] broadcast(stage_0_bwd_lr),
  dimensions={} stage_0_bwd_update = f32[1,4,4,2]
  multiply(stage_0_bwd_input_bwd, stage_0_bwd_lr_bcast) stage_0_bwd_weights0
  = f32[1,4,4,2] parameter(3) stage_0_bwd_weights0_new = f32[1,4,4,2]
  subtract(stage_0_bwd_weights0, stage_0_bwd_update) ROOT stage_0_bwd_tuple =
  (f32[1,4,4,2]) tuple(stage_0_bwd_weights0_new)
}

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0, arg1)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0),
  to_apply=stage_0_fwd,
  backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  pipeline_acts_0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_input = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[], f32[1,4,4,2]) call(pipeline_acts_0,
  pipeline_weights1), to_apply=stage_1_fwd,
  backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  pipeline_reduce = f32[] get-tuple-element(pipeline_stage_1), index=0
  pipeline_acts_0_local = f32[1,4,4,2] get-tuple-element(pipeline_stage_1),
  index=1 pipeline_stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2])
  call(pipeline_reduce, pipeline_acts_0_local, pipeline_weights1),
  to_apply=stage_1_bwd,
  backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  pipeline_acts_0_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd),
  index=0 pipeline_weights1_new = f32[1,4,4,2]
  get-tuple-element(pipeline_stage_1_bwd), index=1 pipeline_stage_0_bwd =
  (f32[1,4,4,2]) call(pipeline_acts_0_bwd, pipeline_weights0, pipeline_input,
  pipeline_weights0), to_apply=stage_0_bwd,
  backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  pipeline_weights0_new = f32[1,4,4,2]
  get-tuple-element(pipeline_stage_0_bwd), index=0
  call_ru = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0_new, pipeline_weights1_new), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1),
  to_apply=pipeline,
  backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  auto pipeline_stage_0_bwd = stages.backward[0];
  EXPECT_THAT(pipeline_stage_0_bwd->operand_count(), 4);
  EXPECT_THAT(
      pipeline_stage_0_bwd->to_apply()->parameter_instruction(1)->user_count(),
      0);
  CallOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, optimizer.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(stages, GetPipelineStages(pipeline_computation));
  pipeline_stage_0_bwd = stages.backward[0];
  EXPECT_THAT(pipeline_stage_0_bwd->operand_count(), 3);
}

TEST_F(CallOptimizerTest, TestDuplicateInputs) {
  std::string hlo = R"(
HloModule cluster

stage_0 {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,2] parameter(1)
  ROOT tuple.8 = (f32[1,4,4,2], f32[1,2]) tuple(arg0, arg1)
}

stage_1 {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  arg2 = f32[1,2] parameter(2)
  arg3 = f32[1,2] parameter(3)
  tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,2], f32[1,2]) tuple(arg0, arg1, arg2, arg3)
  after-all = token[] after-all()
  outfeed = token[] outfeed(tuple, after-all), outfeed_config="\010\001\022\005feed0\"\002\001\001(\001"
  ROOT tuple.1 = () tuple()
}

pipeline {
  ROOT tuple.27 = () tuple()
  arg0.19 = f32[1,4,4,2] parameter(0)
  arg1.20 = f32[1,2] parameter(1)
  call.21 = (f32[1,4,4,2], f32[1,2]) call(arg0.19, arg1.20), to_apply=stage_0, frontend_attributes={CALL_CONFIG_TYPE=PipelineStage}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  get-tuple-element.1 = f32[1,2] get-tuple-element(call.21), index=1
  get-tuple-element.0 = f32[1,4,4,2] get-tuple-element(call.21), index=0
  call = () call(get-tuple-element.0, get-tuple-element.0, get-tuple-element.1, get-tuple-element.1), to_apply=stage_1, frontend_attributes={CALL_CONFIG_TYPE=PipelineStage}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
}

ENTRY cluster {
  arg0.1 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  arg1.2 = f32[1,2] parameter(1), parameter_replication={false}
  call.28 = () call(f32[1,4,4,2] arg0.1, f32[1,2] arg1.2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE=Pipeline}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  ROOT tuple.29 = () tuple()
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  auto stage_1 = stages.forward[1];
  auto find_outfeed = [](HloInstruction* stage) {
    return *std::find_if(stage->to_apply()->instructions().begin(),
                         stage->to_apply()->instructions().end(),
                         [](const HloInstruction* inst) {
                           return inst->opcode() == HloOpcode::kOutfeed;
                         });
  };
  auto outfeed = find_outfeed(stage_1);
  EXPECT_TRUE(
      Match(outfeed, m::Outfeed(m::Tuple(m::Parameter(0), m::Parameter(1),
                                         m::Parameter(2), m::Parameter(3)),
                                m::Op())));

  CallOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, optimizer.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(stages, GetPipelineStages(pipeline_computation));
  stage_1 = stages.forward[1];
  outfeed = find_outfeed(stage_1);
  EXPECT_TRUE(
      Match(outfeed, m::Outfeed(m::Tuple(m::Parameter(0), m::Parameter(0),
                                         m::Parameter(1), m::Parameter(1)),
                                m::Op())));
}

TEST_F(CallOptimizerTest, TestDuplicateOutputs) {
  std::string hlo = R"(
HloModule cluster

stage_0 {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,2] parameter(1)
  ROOT tuple.8 = (f32[1,4,4,2], f32[1,2], f32[1,4,4,2], f32[1,2], f32[1,2]) tuple(arg0, arg1, arg0, arg1, arg1)
}

stage_1 {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  arg2 = f32[1,2] parameter(2)
  arg3 = f32[1,2] parameter(3)
  tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,2], f32[1,2]) tuple(arg0, arg1, arg2, arg3)
  after-all = token[] after-all()
  outfeed = token[] outfeed(tuple, after-all), outfeed_config="\010\001\022\005feed0\"\002\001\001(\001"
  ROOT tuple.1 = () tuple()
}

pipeline {
  ROOT tuple.27 = () tuple()
  arg0.19 = f32[1,4,4,2] parameter(0)
  arg1.20 = f32[1,2] parameter(1)
  call.21 = (f32[1,4,4,2], f32[1,2], f32[1,4,4,2], f32[1,2], f32[1,2]) call(arg0.19, arg1.20), to_apply=stage_0, frontend_attributes={CALL_CONFIG_TYPE=PipelineStage}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  get-tuple-element.3 = f32[1,2] get-tuple-element(call.21), index=3
  get-tuple-element.2 = f32[1,4,4,2] get-tuple-element(call.21), index=2
  get-tuple-element.1 = f32[1,2] get-tuple-element(call.21), index=1
  get-tuple-element.0 = f32[1,4,4,2] get-tuple-element(call.21), index=0
  call = () call(get-tuple-element.0, get-tuple-element.2, get-tuple-element.1, get-tuple-element.3), to_apply=stage_1, frontend_attributes={CALL_CONFIG_TYPE=PipelineStage}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
}

ENTRY cluster {
  arg0.1 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  arg1.2 = f32[1,2] parameter(1), parameter_replication={false}
  call.28 = () call(f32[1,4,4,2] arg0.1, f32[1,2] arg1.2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE=Pipeline}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  ROOT tuple.29 = () tuple()
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  auto stage_1 = stages.forward[1];
  auto find_outfeed = [](HloInstruction* stage) {
    return *std::find_if(stage->to_apply()->instructions().begin(),
                         stage->to_apply()->instructions().end(),
                         [](const HloInstruction* inst) {
                           return inst->opcode() == HloOpcode::kOutfeed;
                         });
  };
  auto outfeed = find_outfeed(stage_1);
  EXPECT_TRUE(
      Match(outfeed, m::Outfeed(m::Tuple(m::Parameter(0), m::Parameter(1),
                                         m::Parameter(2), m::Parameter(3)),
                                m::Op())));

  CallOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, optimizer.Run(module.get()));
  EXPECT_TRUE(changed);
  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK_AND_ASSIGN(changed, optimizer.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(stages, GetPipelineStages(pipeline_computation));
  stage_1 = stages.forward[1];
  outfeed = find_outfeed(stage_1);
  EXPECT_TRUE(
      Match(outfeed, m::Outfeed(m::Tuple(m::Parameter(0), m::Parameter(0),
                                         m::Parameter(1), m::Parameter(1)),
                                m::Op())));
}

TEST_F(CallOptimizerTest, TestPropagateConstantAndBroadcast) {
  std::string hlo = R"(
HloModule cluster

stage_0 {
  arg0 = f32[1,4,4,2] parameter(0)
  c = f32[] constant(1)
  b = f32[1,4,4,2] broadcast(c), dimensions={}
  a = f32[1,4,4,2] add(b, arg0)
  c2 = f32[2] constant({10, 10})
  c3 = f32[2] constant({10, 11})
  ROOT tuple.8 = (f32[1,4,4,2], f32[1,4,4,2], f32[2], f32[2]) tuple(a, b, c2, c3)
}

stage_1 {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  a = f32[1,4,4,2] add(arg0, arg1)
  arg2 = f32[2] parameter(2)
  tuple = (f32[1,4,4,2], f32[2]) tuple(a, arg2)
  after-all = token[] after-all()
  outfeed = token[] outfeed(tuple, after-all), outfeed_config="\010\001\022\005feed0\"\002\001\001(\001"
  ROOT tuple.1 = () tuple()
}

pipeline {
  ROOT tuple.27 = () tuple()
  arg0.19 = f32[1,4,4,2] parameter(0)
  call.21 = (f32[1,4,4,2], f32[1,4,4,2], f32[2], f32[2]) call(arg0.19), to_apply=stage_0, frontend_attributes={CALL_CONFIG_TYPE=PipelineStage}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  get-tuple-element.0 = f32[1,4,4,2] get-tuple-element(call.21), index=0
  get-tuple-element.1 = f32[1,4,4,2] get-tuple-element(call.21), index=1
  get-tuple-element.2 = f32[2] get-tuple-element(call.21), index=2
  call = () call(get-tuple-element.0, get-tuple-element.1, get-tuple-element.2), to_apply=stage_1, frontend_attributes={CALL_CONFIG_TYPE=PipelineStage}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
}

ENTRY cluster {
  arg0.1 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  call.28 = () call(f32[1,4,4,2] arg0.1), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE=Pipeline}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  ROOT tuple.29 = () tuple()
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  auto stage_0 = stages.forward[0];
  EXPECT_TRUE(
      Match(stage_0->to_apply()->root_instruction(),
            m::Tuple(m::Add(m::Broadcast(m::ConstantScalar()), m::Parameter(0)),
                     m::Broadcast(m::ConstantScalar()), m::Constant(),
                     m::Constant())));

  auto stage_1 = stages.forward[1];
  auto find_outfeed = [](HloInstruction* stage) {
    return *std::find_if(stage->to_apply()->instructions().begin(),
                         stage->to_apply()->instructions().end(),
                         [](const HloInstruction* inst) {
                           return inst->opcode() == HloOpcode::kOutfeed;
                         });
  };
  auto outfeed = find_outfeed(stage_1);
  EXPECT_TRUE(Match(
      outfeed, m::Outfeed(m::Tuple(m::Add(m::Parameter(0), m::Parameter(1)),
                                   m::Parameter(2)),
                          m::Op())));

  CallOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, optimizer.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(changed, optimizer.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(stages, GetPipelineStages(pipeline_computation));
  stage_0 = stages.forward[0];
  // Only the non constant output is in the pipeline stage output now.
  EXPECT_TRUE(Match(
      stage_0->to_apply()->root_instruction(),
      m::Tuple(m::Add(m::Broadcast(m::ConstantScalar()), m::Parameter(0)))));

  stage_1 = stages.forward[1];
  outfeed = find_outfeed(stage_1);
  EXPECT_TRUE(Match(
      outfeed, m::Outfeed(m::Tuple(m::Add(m::Parameter(0),
                                          m::Broadcast(m::ConstantScalar())),
                                   m::Constant()),
                          m::Op())));
}

TEST_F(CallOptimizerTest, TestPropagatePad) {
  std::string hlo = R"(
HloModule cluster

stage_0 {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,1] parameter(1)
  c = f32[] constant(0)
  p_arg1 = f32[1,4,4,2] pad(arg1, c), padding=0_0x0_0x0_0x1_0
  add = f32[1,4,4,2] add(arg0, p_arg1)
  ROOT t1 = (f32[1,4,4,2], f32[1,4,4,2]) tuple(p_arg1, add)
}

stage_1 {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  add = f32[1,4,4,2] add(arg0, arg1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(add, arg1)
}

pipeline {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,1] parameter(1)
  call = (f32[1,4,4,2], f32[1,4,4,2]) call(arg0, arg1), to_apply=stage_0, frontend_attributes={CALL_CONFIG_TYPE=PipelineStage}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  get-tuple-element_0 = f32[1,4,4,2] get-tuple-element(call), index=0
  get-tuple-element_1 = f32[1,4,4,2] get-tuple-element(call), index=1
  ROOT call2 = (f32[1,4,4,2], f32[1,4,4,2]) call(get-tuple-element_0, get-tuple-element_1), to_apply=stage_1, frontend_attributes={CALL_CONFIG_TYPE=PipelineStage}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
}

ENTRY cluster {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,1] parameter(1)
  ROOT call = (f32[1,4,4,2], f32[1,4,4,2]) call(arg0, arg1), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE=Pipeline}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();

  CallOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, optimizer.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(changed, optimizer.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(changed, optimizer.Run(module.get()));
  EXPECT_FALSE(changed);
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  const HloInstruction* stage_0 = stages.forward[0];
  // Check that the pad has been removed from the stage output.
  EXPECT_TRUE(
      Match(stage_0->to_apply()->root_instruction(),
            m::Tuple(m::Add(m::Parameter(0),
                            m::Pad(m::Parameter(1), m::ConstantScalar(0))),
                     m::Parameter(1))));

  // Check that the pad has been propagated to the next stage.
  auto stage_1 = stages.forward[1];
  EXPECT_TRUE(
      Match(stage_1->to_apply()->root_instruction(),
            m::Tuple(m::Add(m::Pad(m::Parameter(1), m::ConstantScalar(0)),
                            m::Parameter(0)),
                     m::Parameter(0))));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
