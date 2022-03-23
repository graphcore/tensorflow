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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_feed_hoisting.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inter_ipu_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {
HloInstruction* FindOutfeed(HloComputation* comp) {
  return *std::find_if(comp->instructions().begin(), comp->instructions().end(),
                       [](const HloInstruction* inst) {
                         return inst->opcode() == HloOpcode::kOutfeed;
                       });
}

using PipelineFeedHoistingTest = HloTestBase;

TEST_F(PipelineFeedHoistingTest, HoistInfeed) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_weights1 = f32[1,4,4,2] parameter(1)
  add = f32[1,4,4,2] add(stage_0_fwd_weights0, stage_0_fwd_weights1)
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="01234567"
  input = f32[1,4,4,2] get-tuple-element(infeed), index=0
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(add, input)
}

stage_1_fwd {
  stage_1_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights2 = f32[1,4,4,2] parameter(1)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_fwd_weights0, stage_1_fwd_weights2)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_stage_0_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_stage_0_w0, pipeline_stage_0_w1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  pipeline_stage_1_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_0_w0, pipeline_stage_0_w1)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ShardingPass sharding;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, sharding.Run(module.get()));
  EXPECT_TRUE(changed);

  InterIpuCopyInserter inter_inserter;
  TF_ASSERT_OK_AND_ASSIGN(changed, inter_inserter.Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  EXPECT_THAT(stages.forward[0]->operand_count(), 2);
  EXPECT_THAT(stages.forward[0]->to_apply()->instruction_count(), 7);

  PipelineFeedHoisting hoister;
  TF_ASSERT_OK_AND_ASSIGN(changed, hoister.Run(module.get()));
  EXPECT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(stages, GetPipelineStages(pipeline_computation));
  EXPECT_THAT(stages.forward[0]->operand_count(), 3);
  EXPECT_TRUE(Match(stages.forward[0]->operand(2),
                    m::GetTupleElement(m::Infeed(m::AfterAll()), 0)));
  EXPECT_THAT(stages.forward[0]->to_apply()->instruction_count(), 5);
}

TEST_F(PipelineFeedHoistingTest, HoistTupleInfeed) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_weights1 = f32[1,4,4,2] parameter(1)
  add = f32[1,4,4,2] add(stage_0_fwd_weights0, stage_0_fwd_weights1)
  after-all = token[] after-all()
  infeed = ((f32[1,4,4,2], f32[]), token[]) infeed(after-all), infeed_config="01234567"
  tuple = (f32[1,4,4,2], f32[]) get-tuple-element(infeed), index=0
  input = f32[1,4,4,2] get-tuple-element(tuple), index=0
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(add, input)
}

stage_1_fwd {
  stage_1_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights2 = f32[1,4,4,2] parameter(1)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_fwd_weights0, stage_1_fwd_weights2)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_stage_0_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_stage_0_w0, pipeline_stage_0_w1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  pipeline_stage_1_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_0_w0, pipeline_stage_0_w1)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ShardingPass sharding;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, sharding.Run(module.get()));
  EXPECT_TRUE(changed);

  InterIpuCopyInserter inter_inserter;
  TF_ASSERT_OK_AND_ASSIGN(changed, inter_inserter.Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  EXPECT_THAT(stages.forward[0]->operand_count(), 2);
  EXPECT_THAT(stages.forward[0]->to_apply()->instruction_count(), 8);

  PipelineFeedHoisting hoister;
  TF_ASSERT_OK_AND_ASSIGN(changed, hoister.Run(module.get()));
  EXPECT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(stages, GetPipelineStages(pipeline_computation));
  EXPECT_THAT(stages.forward[0]->operand_count(), 3);
  EXPECT_TRUE(Match(stages.forward[0]->operand(2),
                    m::GetTupleElement(m::Infeed(m::AfterAll()), 0)));
  EXPECT_TRUE(stages.forward[0]->operand(2)->shape().IsTuple());
  EXPECT_THAT(stages.forward[0]->to_apply()->instruction_count(), 6);
}

TEST_F(PipelineFeedHoistingTest, CantHoistInfeed) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_weights1 = f32[1,4,4,2] parameter(1)
  add = f32[1,4,4,2] add(stage_0_fwd_weights0, stage_0_fwd_weights1)
  after-all = token[] after-all()
  infeed = ((f32[1,4,4,2], f32[]), token[]) infeed(after-all), infeed_config="01234567"
  tuple = (f32[1,4,4,2], f32[]) get-tuple-element(infeed), index=0
  tuple1 = (f32[1,4,4,2], f32[]) get-tuple-element(infeed), index=0
  input = f32[1,4,4,2] get-tuple-element(tuple), index=0
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(add, input)
}

stage_1_fwd {
  stage_1_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights2 = f32[1,4,4,2] parameter(1)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_fwd_weights0, stage_1_fwd_weights2)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_stage_0_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_stage_0_w0, pipeline_stage_0_w1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  pipeline_stage_1_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_0_w0, pipeline_stage_0_w1)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ShardingPass sharding;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, sharding.Run(module.get()));
  EXPECT_TRUE(changed);

  InterIpuCopyInserter inter_inserter;
  TF_ASSERT_OK_AND_ASSIGN(changed, inter_inserter.Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  EXPECT_THAT(stages.forward[0]->operand_count(), 2);

  PipelineFeedHoisting hoister;
  TF_ASSERT_OK_AND_ASSIGN(changed, hoister.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(PipelineFeedHoistingTest, CantHoistInfeed2) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_weights1 = f32[1,4,4,2] parameter(1)
  add = f32[1,4,4,2] add(stage_0_fwd_weights0, stage_0_fwd_weights1)
  after-all = token[] after-all(stage_0_fwd_weights1)
  infeed = ((f32[1,4,4,2], f32[]), token[]) infeed(after-all), infeed_config="01234567"
  tuple = (f32[1,4,4,2], f32[]) get-tuple-element(infeed), index=0
  input = f32[1,4,4,2] get-tuple-element(tuple), index=0
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(add, input)
}

stage_1_fwd {
  stage_1_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights2 = f32[1,4,4,2] parameter(1)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_fwd_weights0, stage_1_fwd_weights2)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_stage_0_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_stage_0_w0, pipeline_stage_0_w1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  pipeline_stage_1_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_0_w0, pipeline_stage_0_w1)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ShardingPass sharding;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, sharding.Run(module.get()));
  EXPECT_TRUE(changed);

  InterIpuCopyInserter inter_inserter;
  TF_ASSERT_OK_AND_ASSIGN(changed, inter_inserter.Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  EXPECT_THAT(stages.forward[0]->operand_count(), 2);

  PipelineFeedHoisting hoister;
  TF_ASSERT_OK_AND_ASSIGN(changed, hoister.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(PipelineFeedHoistingTest, HoistOutfeed) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_weights1 = f32[1,4,4,2] parameter(1)
  add = f32[1,4,4,2] add(stage_0_fwd_weights0, stage_0_fwd_weights1)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(add, stage_0_fwd_weights1)
}

stage_1_fwd {
  stage_1_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights2 = f32[1,4,4,2] parameter(1)
  after-all = token[] after-all()
  outfeed = token[] outfeed(stage_1_fwd_weights2, after-all), outfeed_config="\010\001\022\005feed0\"\002\001\001(\001"
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_fwd_weights0, stage_1_fwd_weights2)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_stage_0_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_stage_0_w0, pipeline_stage_0_w1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  pipeline_stage_1_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_0_w0, pipeline_stage_0_w1)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ShardingPass sharding;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, sharding.Run(module.get()));
  EXPECT_TRUE(changed);

  InterIpuCopyInserter inter_inserter;
  TF_ASSERT_OK_AND_ASSIGN(changed, inter_inserter.Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  EXPECT_THAT(ShapeUtil::TupleElementCount(stages.forward[1]->shape()), 2);
  EXPECT_THAT(stages.forward[1]->to_apply()->instruction_count(), 5);

  PipelineFeedHoisting hoister;
  TF_ASSERT_OK_AND_ASSIGN(changed, hoister.Run(module.get()));
  EXPECT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(stages, GetPipelineStages(pipeline_computation));
  EXPECT_THAT(ShapeUtil::TupleElementCount(stages.forward[1]->shape()), 3);
  HloInstruction* outfeed = FindOutfeed(pipeline_computation);
  EXPECT_TRUE(Match(
      outfeed, m::Outfeed(m::GetTupleElement(m::Op().Is(stages.forward[1]), 2),
                          m::AfterAll())));
  EXPECT_THAT(stages.forward[1]->to_apply()->instruction_count(), 3);
}

TEST_F(PipelineFeedHoistingTest, HoistOutfeedTuple) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_weights1 = f32[1,4,4,2] parameter(1)
  add = f32[1,4,4,2] add(stage_0_fwd_weights0, stage_0_fwd_weights1)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(add, stage_0_fwd_weights1)
}

stage_1_fwd {
  stage_1_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights2 = f32[1,4,4,2] parameter(1)
  after-all = token[] after-all()
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_fwd_weights0, stage_1_fwd_weights2)
  outfeed = token[] outfeed(stage_1_fwd_tuple, after-all), outfeed_config="\010\001\022\005feed0\"\002\001\001(\001"
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_stage_0_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_stage_0_w0, pipeline_stage_0_w1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  pipeline_stage_1_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_0_w0, pipeline_stage_0_w1)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ShardingPass sharding;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, sharding.Run(module.get()));
  EXPECT_TRUE(changed);

  InterIpuCopyInserter inter_inserter;
  TF_ASSERT_OK_AND_ASSIGN(changed, inter_inserter.Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  EXPECT_THAT(ShapeUtil::TupleElementCount(stages.forward[1]->shape()), 2);
  EXPECT_THAT(stages.forward[1]->to_apply()->instruction_count(), 5);

  PipelineFeedHoisting hoister;
  TF_ASSERT_OK_AND_ASSIGN(changed, hoister.Run(module.get()));
  EXPECT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(stages, GetPipelineStages(pipeline_computation));
  EXPECT_THAT(ShapeUtil::TupleElementCount(stages.forward[1]->shape()), 3);
  HloInstruction* outfeed = FindOutfeed(pipeline_computation);
  EXPECT_TRUE(Match(
      outfeed, m::Outfeed(m::GetTupleElement(m::Op().Is(stages.forward[1]), 2),
                          m::AfterAll())));
  EXPECT_THAT(stages.forward[1]->to_apply()->instruction_count(), 4);
}

TEST_F(PipelineFeedHoistingTest, CantHoistOutfeed) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_weights1 = f32[1,4,4,2] parameter(1)
  add = f32[1,4,4,2] add(stage_0_fwd_weights0, stage_0_fwd_weights1)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(add, stage_0_fwd_weights1)
}

stage_1_fwd {
  stage_1_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights2 = f32[1,4,4,2] parameter(1)
  after-all = token[] after-all(stage_1_fwd_weights2)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_fwd_weights0, stage_1_fwd_weights2)
  outfeed = token[] outfeed(stage_1_fwd_tuple, after-all), outfeed_config="\010\001\022\005feed0\"\002\001\001(\001"
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_stage_0_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_stage_0_w0, pipeline_stage_0_w1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  pipeline_stage_1_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_0_w0, pipeline_stage_0_w1)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ShardingPass sharding;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, sharding.Run(module.get()));
  EXPECT_TRUE(changed);

  InterIpuCopyInserter inter_inserter;
  TF_ASSERT_OK_AND_ASSIGN(changed, inter_inserter.Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  EXPECT_THAT(ShapeUtil::TupleElementCount(stages.forward[1]->shape()), 2);

  PipelineFeedHoisting hoister;
  TF_ASSERT_OK_AND_ASSIGN(changed, hoister.Run(module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
