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

#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"

#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using PipelineUtilTest = HloTestBase;

std::string GetCorrectPipelineStages() {
  // A "network" simulating the expected flow.
  return R"(
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
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_0_fwd_acts_0, stage_0_fwd_input)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_1_fwd_acts_1 = f32[1,4,4,2] add(stage_1_fwd_acts_0, stage_1_fwd_weights1)
  stage_1_fwd_zero = f32[] constant(0)
  stage_1_fwd_reduce = f32[] reduce(stage_1_fwd_acts_1, stage_1_fwd_zero), dimensions={0,1,2,3}, to_apply=add_float
  ROOT stage_1_fwd_tuple = (f32[], f32[1,4,4,2]) tuple(stage_1_fwd_reduce, stage_1_fwd_acts_0)
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

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_acts_0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_input = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[], f32[1,4,4,2]) call(pipeline_acts_0, pipeline_weights1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_reduce = f32[] get-tuple-element(pipeline_stage_1), index=0
  pipeline_acts_0_local = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  pipeline_stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_reduce, pipeline_acts_0_local, pipeline_weights1), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\"}}"
  pipeline_acts_0_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd), index=0
  pipeline_weights1_new = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd), index=1
  pipeline_stage_0_bwd = (f32[1,4,4,2]) call(pipeline_acts_0_bwd, pipeline_input, pipeline_weights0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\"}}"
  pipeline_weights0_new = f32[1,4,4,2] get-tuple-element(pipeline_stage_0_bwd), index=0
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_weights0_new, pipeline_weights1_new)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
}

std::string GetNeedToLowerPipelineStages() {
  // A "network" simulating the expected flow.
  // Note that "weight updates" need to be lowered.
  // Note how lr is thread through the forward stages.
  return R"(
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
  stage_0_fwd_lr = f32[] parameter(1)
  ROOT stage_0_fwd_tuple = (f32[], f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_0_fwd_lr, stage_0_fwd_acts_0, stage_0_fwd_input)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_1_fwd_acts_1 = f32[1,4,4,2] add(stage_1_fwd_acts_0, stage_1_fwd_weights1)
  stage_1_fwd_zero = f32[] constant(0)
  stage_1_fwd_reduce = f32[] reduce(stage_1_fwd_acts_1, stage_1_fwd_zero), dimensions={0,1,2,3}, to_apply=add_float
  stage_1_fwd_lr = f32[] parameter(2)
  ROOT stage_1_fwd_tuple = (f32[], f32[], f32[1,4,4,2]) tuple(stage_1_fwd_lr, stage_1_fwd_reduce, stage_1_fwd_acts_0)
}

stage_1_bwd {
  stage_1_bwd_reduce = f32[] parameter(0)
  stage_1_bwd_bcast1 = f32[1,4,4,2] broadcast(stage_1_bwd_reduce), dimensions={}
  stage_1_bwd_acts_0 = f32[1,4,4,2] parameter(1)
  stage_1_bwd_acts_0_bwd = f32[1,4,4,2] add(stage_1_bwd_acts_0, stage_1_bwd_bcast1)
  ROOT stage_1_bwd_tuple = (f32[1,4,4,2]) tuple(stage_1_bwd_acts_0_bwd)
}

stage_0_bwd {
  stage_0_bwd_acts_0_bwd = f32[1,4,4,2] parameter(0)
  stage_0_bwd_input = f32[1,4,4,2] parameter(1)
  stage_0_bwd_input_bwd = f32[1,4,4,2] add(stage_0_bwd_input, stage_0_bwd_acts_0_bwd)
  ROOT stage_0_bwd_tuple = (f32[1,4,4,2]) tuple(stage_0_bwd_input_bwd)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_lr = f32[] parameter(2)
  pipeline_stage_0 = (f32[], f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_lr), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_stage_0_lr = f32[] get-tuple-element(pipeline_stage_0), index=0
  pipeline_acts_0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_input = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=2
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[], f32[], f32[1,4,4,2]) call(pipeline_acts_0, pipeline_weights1, pipeline_stage_0_lr), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_stage_1_lr = f32[] get-tuple-element(pipeline_stage_1), index=0
  pipeline_reduce = f32[] get-tuple-element(pipeline_stage_1), index=1
  pipeline_acts_0_local = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=2
  pipeline_stage_1_bwd = (f32[1,4,4,2]) call(pipeline_reduce, pipeline_acts_0_local), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\"}}"
  pipeline_acts_0_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd), index=0
  pipeline_lr_bcast1 = f32[1,4,4,2] broadcast(pipeline_stage_1_lr), dimensions={}
  pipeline_weights1_update = f32[1,4,4,2] multiply(pipeline_acts_0_bwd, pipeline_lr_bcast1)
  pipeline_weights1_apply = f32[1,4,4,2] subtract(pipeline_weights1, pipeline_weights1_update)
  pipeline_stage_0_bwd = (f32[1,4,4,2]) call(pipeline_acts_0_bwd, pipeline_input), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\"}}"
  pipeline_input_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_0_bwd), index=0
  pipeline_lr_bcast2 = f32[1,4,4,2] broadcast(pipeline_stage_1_lr), dimensions={}
  pipeline_weights0_update = f32[1,4,4,2] multiply(pipeline_input_bwd, pipeline_lr_bcast2)
  pipeline_weights0_apply = f32[1,4,4,2] subtract(pipeline_weights0, pipeline_weights0_update)
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_weights0_apply, pipeline_weights1_apply)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.lr = f32[] parameter(2), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1, e.lr), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
}

std::string GetInputsNeedToBeLowered() {
  // Note inputs need to be lowered.
  return R"(
HloModule top

add_float {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT a = f32[] add(f32[] x, f32[] y)
}

stage_0_fwd {
  stage_0_fwd_t = token[] after-all()
  stage_0_fwd_infeed = (f32[1,4,4,2], token[]) infeed(stage_0_fwd_t)
  stage_0_fwd_input = f32[1,4,4,2] get-tuple-element(stage_0_fwd_infeed), index=0
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_acts_0 = f32[1,4,4,2] add(stage_0_fwd_input, stage_0_fwd_weights0)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2]) tuple(stage_0_fwd_acts_0)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_1_fwd_acts_1 = f32[1,4,4,2] add(stage_1_fwd_acts_0, stage_1_fwd_weights1)
  stage_1_fwd_zero = f32[] parameter(2)
  stage_1_fwd_reduce = f32[] reduce(stage_1_fwd_acts_1, stage_1_fwd_zero), dimensions={0,1,2,3}, to_apply=add_float
  stage_1_fwd_t = token[] after-all()
  stage_1_fwd_outfeed = () outfeed(stage_1_fwd_reduce, stage_1_fwd_t)
  ROOT stage_1_fwd_tuple = () tuple()
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_acts_0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_zero = f32[] constant(0)
  pipeline_stage_1 = () call(pipeline_acts_0, pipeline_weights1, pipeline_zero), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  ROOT pipeline_tuple = () tuple()
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = () call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
}

HloValueSet GetValueSet(PipelineDataflowAnalysis* analysis,
                        std::vector<HloInstruction*> insts) {
  std::vector<const HloValueSet*> value_sets(insts.size());
  absl::c_transform(insts, value_sets.begin(),
                    [&analysis](HloInstruction* inst) {
                      return &analysis->GetValueSet(inst);
                    });
  HloValueSet value_set;
  value_set.AssignUnionOf(value_sets);
  return value_set;
}

TEST_F(PipelineUtilTest, IsPiplineStageOrBackwardOpTest) {
  std::string hlo = GetCorrectPipelineStages();

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  EXPECT_TRUE(
      IsPiplineStageOrBackwardOp(FindInstruction(module0, "pipeline_stage_0")));
  EXPECT_TRUE(
      IsPiplineStageOrBackwardOp(FindInstruction(module0, "pipeline_stage_1")));
  EXPECT_TRUE(IsPiplineStageOrBackwardOp(
      FindInstruction(module0, "pipeline_stage_1_bwd")));
  EXPECT_TRUE(IsPiplineStageOrBackwardOp(
      FindInstruction(module0, "pipeline_stage_0_bwd")));
  EXPECT_FALSE(IsPiplineStageOrBackwardOp(FindInstruction(module0, "e.call")));
}

TEST_F(PipelineUtilTest, IsProducerOpTest) {
  std::string hlo = GetCorrectPipelineStages();

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  EXPECT_TRUE(IsProducerOp(FindInstruction(module0, "pipeline_weights0")));
  EXPECT_TRUE(IsProducerOp(FindInstruction(module0, "pipeline_stage_0")));
  EXPECT_FALSE(IsProducerOp(FindInstruction(module0, "pipeline_acts_0")));
  EXPECT_FALSE(IsProducerOp(FindInstruction(module0, "pipeline_input")));
  EXPECT_TRUE(IsProducerOp(FindInstruction(module0, "pipeline_weights1")));
  EXPECT_TRUE(IsProducerOp(FindInstruction(module0, "pipeline_stage_1")));
  EXPECT_FALSE(IsProducerOp(FindInstruction(module0, "pipeline_reduce")));
  EXPECT_FALSE(IsProducerOp(FindInstruction(module0, "pipeline_acts_0_local")));
  EXPECT_TRUE(IsProducerOp(FindInstruction(module0, "pipeline_stage_1_bwd")));
  EXPECT_FALSE(IsProducerOp(FindInstruction(module0, "pipeline_acts_0_bwd")));
  EXPECT_FALSE(IsProducerOp(FindInstruction(module0, "pipeline_weights1_new")));
  EXPECT_TRUE(IsProducerOp(FindInstruction(module0, "pipeline_stage_0_bwd")));
  EXPECT_FALSE(IsProducerOp(FindInstruction(module0, "pipeline_weights0_new")));
  EXPECT_FALSE(IsProducerOp(FindInstruction(module0, "pipeline_tuple")));
}

TEST_F(PipelineUtilTest, GetPipelineStagesFwdBwdTest) {
  std::string hlo = GetCorrectPipelineStages();

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  EXPECT_THAT(
      stages.forward,
      ::testing::ElementsAre(FindInstruction(module0, "pipeline_stage_0"),
                             FindInstruction(module0, "pipeline_stage_1")));
  EXPECT_THAT(
      stages.backward,
      ::testing::ElementsAre(FindInstruction(module0, "pipeline_stage_0_bwd"),
                             FindInstruction(module0, "pipeline_stage_1_bwd")));
}

TEST_F(PipelineUtilTest, GetPipelineStagesFwdBwdMismatchTest) {
  // A "network" simulating the expected flow.
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
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_0_fwd_acts_0, stage_0_fwd_input)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_1_fwd_acts_1 = f32[1,4,4,2] add(stage_1_fwd_acts_0, stage_1_fwd_weights1)
  stage_1_fwd_zero = f32[] constant(0)
  stage_1_fwd_reduce = f32[] reduce(stage_1_fwd_acts_1, stage_1_fwd_zero), dimensions={0,1,2,3}, to_apply=add_float
  ROOT stage_1_fwd_tuple = (f32[], f32[1,4,4,2]) tuple(stage_1_fwd_reduce, stage_1_fwd_acts_1)
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

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_acts_0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_input = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[], f32[1,4,4,2]) call(pipeline_acts_0, pipeline_weights1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_reduce = f32[] get-tuple-element(pipeline_stage_1), index=0
  pipeline_acts_1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  pipeline_stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_reduce, pipeline_acts_1, pipeline_weights1), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\"}}"
  pipeline_acts_0_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd), index=0
  pipeline_weights1_new = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_weights0, pipeline_weights1_new)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");
  auto stages_or = GetPipelineStages(pipeline_computation);
  EXPECT_FALSE(stages_or.ok());
}

TEST_F(PipelineUtilTest, PipelineDataflowAnalysisNoOpsToLower) {
  std::string hlo = GetCorrectPipelineStages();
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  TF_ASSERT_OK_AND_ASSIGN(auto analysis,
                          PipelineDataflowAnalysis::GetAnalysis(stages));

  EXPECT_TRUE(absl::c_all_of(
      pipeline_computation->instructions(), [&analysis](HloInstruction* inst) {
        return !analysis->HasToBeLowered(inst).ValueOrDie();
      }));
  HloInstruction* pipeline_weights0 =
      FindInstruction(module0, "pipeline_weights0");
  {
    auto& value_set = analysis->GetValueSet(pipeline_weights0);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_weights0);
  }
  HloInstruction* pipeline_stage_0 =
      FindInstruction(module0, "pipeline_stage_0");
  {
    EXPECT_THAT(analysis->GetStageID(pipeline_stage_0).ValueOrDie(),
                StageID(true, 0));
    auto& value_set = analysis->GetValueSet(pipeline_stage_0);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_0);
  }
  HloInstruction* pipeline_acts_0 = FindInstruction(module0, "pipeline_acts_0");
  {
    auto& value_set = analysis->GetValueSet(pipeline_acts_0);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_0);
  }
  HloInstruction* pipeline_input = FindInstruction(module0, "pipeline_input");
  {
    auto& value_set = analysis->GetValueSet(pipeline_input);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_0);
  }
  HloInstruction* pipeline_stage_1 =
      FindInstruction(module0, "pipeline_stage_1");
  {
    EXPECT_THAT(analysis->GetStageID(pipeline_stage_1).ValueOrDie(),
                StageID(true, 1));
    EXPECT_THAT(analysis->GetPreviousStageID(pipeline_stage_1).ValueOrDie(),
                StageID(true, 0));
    auto& value_set = analysis->GetValueSet(pipeline_stage_1);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_1);
  }
  HloInstruction* pipeline_reduce = FindInstruction(module0, "pipeline_reduce");
  {
    auto& value_set = analysis->GetValueSet(pipeline_reduce);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_1);
  }
  HloInstruction* pipeline_acts_0_local =
      FindInstruction(module0, "pipeline_acts_0_local");
  {
    auto& value_set = analysis->GetValueSet(pipeline_acts_0_local);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_1);
  }
  HloInstruction* pipeline_stage_1_bwd =
      FindInstruction(module0, "pipeline_stage_1_bwd");
  {
    EXPECT_THAT(analysis->GetStageID(pipeline_stage_1_bwd).ValueOrDie(),
                StageID(false, 1));
    EXPECT_THAT(analysis->GetPreviousStageID(pipeline_stage_1_bwd).ValueOrDie(),
                StageID(true, 1));
    auto& value_set = analysis->GetValueSet(pipeline_stage_1_bwd);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_1_bwd);
  }
  HloInstruction* pipeline_acts_0_bwd =
      FindInstruction(module0, "pipeline_acts_0_bwd");
  {
    auto& value_set = analysis->GetValueSet(pipeline_acts_0_bwd);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_1_bwd);
  }
  HloInstruction* pipeline_weights1_new =
      FindInstruction(module0, "pipeline_weights1_new");
  {
    auto& value_set = analysis->GetValueSet(pipeline_weights1_new);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_1_bwd);
  }
  HloInstruction* pipeline_stage_0_bwd =
      FindInstruction(module0, "pipeline_stage_0_bwd");
  {
    EXPECT_THAT(analysis->GetStageID(pipeline_stage_0_bwd).ValueOrDie(),
                StageID(false, 0));
    EXPECT_THAT(analysis->GetPreviousStageID(pipeline_stage_0_bwd).ValueOrDie(),
                StageID(false, 1));
    auto& value_set = analysis->GetValueSet(pipeline_stage_0_bwd);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_0_bwd);
  }
  HloInstruction* pipeline_weights0_new =
      FindInstruction(module0, "pipeline_weights0_new");
  {
    auto& value_set = analysis->GetValueSet(pipeline_weights0_new);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_0_bwd);
  }
  HloInstruction* pipeline_tuple = FindInstruction(module0, "pipeline_tuple");
  {
    auto& value_set = analysis->GetValueSet(pipeline_tuple);
    HloValueSet expected_value_set = GetValueSet(
        analysis.get(), {pipeline_stage_0_bwd, pipeline_stage_1_bwd});
    EXPECT_THAT(value_set.values(), expected_value_set.values());
  }
}

TEST_F(PipelineUtilTest, DuplicateGTEEdgesTestNoDuplication) {
  std::string hlo = GetCorrectPipelineStages();
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  auto duplicated_or = DuplicateGTEEdges(stages);
  EXPECT_TRUE(duplicated_or.ok());
  EXPECT_FALSE(duplicated_or.ValueOrDie());
}

TEST_F(PipelineUtilTest, DuplicateGTEEdgesTestDuplication) {
  std::string hlo = GetNeedToLowerPipelineStages();
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");
  HloInstruction* pipeline_stage_1 =
      FindInstruction(module0, "pipeline_stage_1");
  HloInstruction* pipeline_stage_1_bwd =
      FindInstruction(module0, "pipeline_stage_1_bwd");

  auto num_gtes_at_tuple_index = [&](HloInstruction* inst,
                                     const int64 tuple_index) {
    int64 count = 0;
    for (HloInstruction* user : inst->users()) {
      if (user->opcode() == HloOpcode::kGetTupleElement &&
          user->tuple_index() == tuple_index) {
        count++;
      }
    }
    return count;
  };

  EXPECT_THAT(num_gtes_at_tuple_index(pipeline_stage_1, 0), 1);
  EXPECT_THAT(num_gtes_at_tuple_index(pipeline_stage_1_bwd, 0), 1);

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  auto duplicated_or = DuplicateGTEEdges(stages);
  EXPECT_TRUE(duplicated_or.ok());
  EXPECT_TRUE(duplicated_or.ValueOrDie());

  EXPECT_THAT(num_gtes_at_tuple_index(pipeline_stage_1, 0), 2);
  EXPECT_THAT(num_gtes_at_tuple_index(pipeline_stage_1_bwd, 0), 2);
}

TEST_F(PipelineUtilTest, UniquifyPipelineStageCallsitesTest) {
  std::string hlo = R"(
HloModule top

stage {
  param = f32[1,4,4,2] parameter(0)
  ROOT t = (f32[1,4,4,2]) tuple(param)
}

pipeline {
  weights0 = f32[1,4,4,2] parameter(0)
  stage_0 = (f32[1,4,4,2]) call(weights0), to_apply=stage, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  stage_0_0 = f32[1,4,4,2] get-tuple-element(stage_0), index=0
  stage_1 = (f32[1,4,4,2]) call(stage_0_0), to_apply=stage, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  ROOT t = () tuple()
}

ENTRY e {
  weights = f32[1,4,4,2] parameter(0), parameter_replication={false}
  ROOT t = () call(weights), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");
  HloInstruction* stage_0 = FindInstruction(module0, "stage_0");
  HloInstruction* stage_1 = FindInstruction(module0, "stage_1");
  EXPECT_EQ(stage_0->to_apply(), stage_1->to_apply());

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));

  auto unique_or = UniquifyPipelineStageCallsites(stages);
  EXPECT_TRUE(unique_or.ok());
  EXPECT_TRUE(unique_or.ValueOrDie());
  EXPECT_NE(stage_0->to_apply(), stage_1->to_apply());
}

TEST_F(PipelineUtilTest, VerifyPipelineStagesBeforeLoweringOK) {
  std::string hlo = GetNeedToLowerPipelineStages();
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  auto valid_or = VerifyPipelineStagesBeforeLowering(stages);
  EXPECT_TRUE(valid_or.ok());
}

TEST_F(PipelineUtilTest, VerifyPipelineStagesBeforeLoweringNotOK) {
  std::string hlo_non_tuple_output = R"(
HloModule top

stage_0_fwd {
  ROOT stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = f32[1,4,4,2] call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  ROOT pipeline_tuple = (f32[1,4,4,2]) tuple(pipeline_stage_0)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";

  std::string hlo_pipeline_non_gte_use = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2]) tuple(stage_0_fwd_weights0)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  ROOT pipeline_stage_0 = (f32[1,4,4,2]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";

  std::string hlo_pipeline_stage_root = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2]) tuple(stage_0_fwd_weights0)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  p1 = pred[] constant(1)
  pipeline_stage_0 = (f32[1,4,4,2]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_weights0_tuple = (f32[1,4,4,2]) tuple(pipeline_weights0)
  ROOT tuple-select = (f32[1,4,4,2]) tuple-select(p1, pipeline_stage_0, pipeline_weights0_tuple)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  for (const std::string& hlo : {hlo_non_tuple_output, hlo_pipeline_non_gte_use,
                                 hlo_pipeline_stage_root}) {
    auto config = GetModuleConfigForTest();
    auto module = ParseAndReturnVerifiedModule(hlo, config);
    EXPECT_TRUE(module.ok());
    auto* module0 = module.ValueOrDie().get();
    HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

    auto stages_or = GetPipelineStages(pipeline_computation);
    EXPECT_TRUE(stages_or.ok());
    auto stages = stages_or.ValueOrDie();
    auto valid = VerifyPipelineStagesBeforeLowering(stages);
    EXPECT_FALSE(valid.ok());
  }
}

TEST_F(PipelineUtilTest, PipelineDataflowAnalysisOpsToLower) {
  std::string hlo = GetNeedToLowerPipelineStages();
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  auto duplicated_or = DuplicateGTEEdges(stages);
  EXPECT_TRUE(duplicated_or.ok());
  EXPECT_TRUE(duplicated_or.ValueOrDie());

  auto analysis_or = PipelineDataflowAnalysis::GetAnalysis(stages);
  EXPECT_TRUE(analysis_or.ok());
  auto analysis = std::move(analysis_or.ValueOrDie());
  HloInstruction* pipeline_weights0 =
      FindInstruction(module0, "pipeline_weights0");
  {
    auto& value_set = analysis->GetValueSet(pipeline_weights0);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_weights0);
    EXPECT_FALSE(analysis->HasToBeLowered(pipeline_weights0).ValueOrDie());
  }
  HloInstruction* pipeline_lr = FindInstruction(module0, "pipeline_lr");
  {
    auto& value_set = analysis->GetValueSet(pipeline_lr);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_lr);
    EXPECT_FALSE(analysis->HasToBeLowered(pipeline_lr).ValueOrDie());
  }
  HloInstruction* pipeline_stage_0 =
      FindInstruction(module0, "pipeline_stage_0");
  {
    EXPECT_THAT(analysis->GetStageID(pipeline_stage_0).ValueOrDie(),
                StageID(true, 0));
    auto& value_set = analysis->GetValueSet(pipeline_stage_0);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_0);
    EXPECT_FALSE(analysis->HasToBeLowered(pipeline_stage_0).ValueOrDie());
  }
  HloInstruction* pipeline_stage_0_lr =
      FindInstruction(module0, "pipeline_stage_0_lr");
  {
    auto& value_set = analysis->GetValueSet(pipeline_stage_0_lr);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_0);
    EXPECT_FALSE(analysis->HasToBeLowered(pipeline_stage_0_lr).ValueOrDie());
  }
  HloInstruction* pipeline_acts_0 = FindInstruction(module0, "pipeline_acts_0");
  {
    auto& value_set = analysis->GetValueSet(pipeline_acts_0);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_0);
    EXPECT_FALSE(analysis->HasToBeLowered(pipeline_acts_0).ValueOrDie());
  }
  HloInstruction* pipeline_input = FindInstruction(module0, "pipeline_input");
  {
    auto& value_set = analysis->GetValueSet(pipeline_input);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_0);
    EXPECT_FALSE(analysis->HasToBeLowered(pipeline_input).ValueOrDie());
  }
  HloInstruction* pipeline_weights1 =
      FindInstruction(module0, "pipeline_weights1");
  {
    auto& value_set = analysis->GetValueSet(pipeline_weights1);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_weights1);
    EXPECT_FALSE(analysis->HasToBeLowered(pipeline_weights1).ValueOrDie());
  }
  HloInstruction* pipeline_stage_1 =
      FindInstruction(module0, "pipeline_stage_1");
  {
    EXPECT_THAT(analysis->GetStageID(pipeline_stage_1).ValueOrDie(),
                StageID(true, 1));
    EXPECT_THAT(analysis->GetPreviousStageID(pipeline_stage_1).ValueOrDie(),
                StageID(true, 0));
    auto& value_set = analysis->GetValueSet(pipeline_stage_1);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_1);
    EXPECT_FALSE(analysis->HasToBeLowered(pipeline_stage_1).ValueOrDie());
  }
  for (HloInstruction* user : pipeline_stage_1->users()) {
    EXPECT_THAT(user->opcode(), HloOpcode::kGetTupleElement);
    if (user->tuple_index() == 0) {
      auto& value_set = analysis->GetValueSet(user);
      EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_1);
      EXPECT_FALSE(analysis->HasToBeLowered(user).ValueOrDie());
    }
  }
  HloInstruction* pipeline_reduce = FindInstruction(module0, "pipeline_reduce");
  {
    auto& value_set = analysis->GetValueSet(pipeline_reduce);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_1);
    EXPECT_FALSE(analysis->HasToBeLowered(pipeline_reduce).ValueOrDie());
  }
  HloInstruction* pipeline_acts_0_local =
      FindInstruction(module0, "pipeline_acts_0_local");
  {
    auto& value_set = analysis->GetValueSet(pipeline_acts_0_local);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_1);
    EXPECT_FALSE(analysis->HasToBeLowered(pipeline_acts_0_local).ValueOrDie());
  }
  HloInstruction* pipeline_stage_1_bwd =
      FindInstruction(module0, "pipeline_stage_1_bwd");
  {
    EXPECT_THAT(analysis->GetStageID(pipeline_stage_1_bwd).ValueOrDie(),
                StageID(false, 1));
    EXPECT_THAT(analysis->GetPreviousStageID(pipeline_stage_1_bwd).ValueOrDie(),
                StageID(true, 1));
    auto& value_set = analysis->GetValueSet(pipeline_stage_1_bwd);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_1_bwd);
    EXPECT_FALSE(analysis->HasToBeLowered(pipeline_stage_1_bwd).ValueOrDie());
  }
  for (HloInstruction* user : pipeline_stage_1_bwd->users()) {
    EXPECT_THAT(user->opcode(), HloOpcode::kGetTupleElement);
    if (user->tuple_index() == 0) {
      auto& value_set = analysis->GetValueSet(user);
      EXPECT_THAT(value_set.GetUniqueValue().instruction(),
                  pipeline_stage_1_bwd);
      EXPECT_FALSE(analysis->HasToBeLowered(user).ValueOrDie());
    }
  }
  HloInstruction* pipeline_lr_bcast1 =
      FindInstruction(module0, "pipeline_lr_bcast1");
  {
    auto& value_set = analysis->GetValueSet(pipeline_lr_bcast1);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_1);
    EXPECT_TRUE(analysis->HasToBeLowered(pipeline_lr_bcast1).ValueOrDie());
  }
  HloInstruction* pipeline_weights1_update =
      FindInstruction(module0, "pipeline_weights1_update");
  {
    auto& value_set = analysis->GetValueSet(pipeline_weights1_update);
    HloValueSet expected_value_set =
        GetValueSet(analysis.get(), {pipeline_stage_1, pipeline_stage_1_bwd});
    EXPECT_THAT(value_set.values(), expected_value_set.values());
    EXPECT_TRUE(
        analysis->HasToBeLowered(pipeline_weights1_update).ValueOrDie());
  }
  HloInstruction* pipeline_weights1_apply =
      FindInstruction(module0, "pipeline_weights1_apply");
  {
    auto& value_set = analysis->GetValueSet(pipeline_weights1_apply);
    HloValueSet expected_value_set = GetValueSet(
        analysis.get(),
        {pipeline_weights1, pipeline_stage_1, pipeline_stage_1_bwd});
    EXPECT_THAT(value_set.values(), expected_value_set.values());
    EXPECT_TRUE(analysis->HasToBeLowered(pipeline_weights1_apply).ValueOrDie());
  }
  HloInstruction* pipeline_stage_0_bwd =
      FindInstruction(module0, "pipeline_stage_0_bwd");
  {
    EXPECT_THAT(analysis->GetStageID(pipeline_stage_0_bwd).ValueOrDie(),
                StageID(false, 0));
    EXPECT_THAT(analysis->GetPreviousStageID(pipeline_stage_0_bwd).ValueOrDie(),
                StageID(false, 1));
    auto& value_set = analysis->GetValueSet(pipeline_stage_0_bwd);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_0_bwd);
    EXPECT_FALSE(analysis->HasToBeLowered(pipeline_stage_0_bwd).ValueOrDie());
  }
  HloInstruction* pipeline_input_bwd =
      FindInstruction(module0, "pipeline_input_bwd");
  {
    auto& value_set = analysis->GetValueSet(pipeline_input_bwd);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_0_bwd);
    EXPECT_FALSE(analysis->HasToBeLowered(pipeline_input_bwd).ValueOrDie());
  }
  HloInstruction* pipeline_lr_bcast2 =
      FindInstruction(module0, "pipeline_lr_bcast2");
  {
    auto& value_set = analysis->GetValueSet(pipeline_lr_bcast2);
    EXPECT_THAT(value_set.GetUniqueValue().instruction(), pipeline_stage_1);
    EXPECT_TRUE(analysis->HasToBeLowered(pipeline_lr_bcast2).ValueOrDie());
  }
  HloInstruction* pipeline_weights0_update =
      FindInstruction(module0, "pipeline_weights0_update");
  {
    auto& value_set = analysis->GetValueSet(pipeline_weights0_update);
    HloValueSet expected_value_set =
        GetValueSet(analysis.get(), {pipeline_stage_1, pipeline_stage_0_bwd});
    EXPECT_THAT(value_set.values(), expected_value_set.values());
    EXPECT_TRUE(
        analysis->HasToBeLowered(pipeline_weights0_update).ValueOrDie());
  }
  HloInstruction* pipeline_weights0_apply =
      FindInstruction(module0, "pipeline_weights0_apply");
  {
    auto& value_set = analysis->GetValueSet(pipeline_weights0_apply);
    HloValueSet expected_value_set = GetValueSet(
        analysis.get(),
        {pipeline_weights0, pipeline_stage_1, pipeline_stage_0_bwd});
    EXPECT_THAT(value_set.values(), expected_value_set.values());
    EXPECT_TRUE(analysis->HasToBeLowered(pipeline_weights0_apply).ValueOrDie());
  }
  HloInstruction* pipeline_tuple = FindInstruction(module0, "pipeline_tuple");
  {
    auto& value_set = analysis->GetValueSet(pipeline_tuple);
    HloValueSet expected_value_set = GetValueSet(
        analysis.get(), {pipeline_weights0, pipeline_weights1, pipeline_stage_1,
                         pipeline_stage_0_bwd, pipeline_stage_1_bwd});
    EXPECT_THAT(value_set.values(), expected_value_set.values());
    EXPECT_FALSE(analysis->HasToBeLowered(pipeline_tuple).ValueOrDie());
  }
}

TEST_F(PipelineUtilTest, DataFlowViolations) {
  std::string hlo_skipped_connections = R"(
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
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_0_fwd_acts_0, stage_0_fwd_input)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_1_fwd_acts_1 = f32[1,4,4,2] add(stage_1_fwd_acts_0, stage_1_fwd_weights1)
  stage_1_fwd_zero = f32[] constant(0)
  stage_1_fwd_reduce = f32[] reduce(stage_1_fwd_acts_1, stage_1_fwd_zero), dimensions={0,1,2,3}, to_apply=add_float
  ROOT stage_1_fwd_tuple = (f32[], f32[1,4,4,2]) tuple(stage_1_fwd_reduce, stage_1_fwd_acts_0)
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

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_acts_0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_input = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[], f32[1,4,4,2]) call(pipeline_acts_0, pipeline_weights1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_reduce = f32[] get-tuple-element(pipeline_stage_1), index=0
  pipeline_acts_0_local = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  pipeline_stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_reduce, pipeline_acts_0, pipeline_weights1), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\"}}"
  pipeline_acts_0_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd), index=0
  pipeline_weights1_new = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd), index=1
  pipeline_stage_0_bwd = (f32[1,4,4,2]) call(pipeline_acts_0_bwd, pipeline_input, pipeline_weights0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\"}}"
  pipeline_weights0_new = f32[1,4,4,2] get-tuple-element(pipeline_stage_0_bwd), index=0
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_weights0_new, pipeline_weights1_new)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";

  // Weights0 is passed to two stages.
  std::string hlo_same_param = R"(
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
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_0_fwd_acts_0, stage_0_fwd_input)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_1_fwd_acts_1 = f32[1,4,4,2] add(stage_1_fwd_acts_0, stage_1_fwd_weights1)
  stage_1_fwd_zero = f32[] constant(0)
  stage_1_fwd_reduce = f32[] reduce(stage_1_fwd_acts_1, stage_1_fwd_zero), dimensions={0,1,2,3}, to_apply=add_float
  ROOT stage_1_fwd_tuple = (f32[], f32[1,4,4,2]) tuple(stage_1_fwd_reduce, stage_1_fwd_acts_1)
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

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_acts_0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_input = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_stage_1 = (f32[], f32[1,4,4,2]) call(pipeline_acts_0, pipeline_weights0), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_reduce = f32[] get-tuple-element(pipeline_stage_1), index=0
  pipeline_acts_1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  pipeline_stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_reduce, pipeline_acts_1, pipeline_weights0), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\"}}"
  pipeline_acts_0_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd), index=0
  pipeline_weights0_new = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd), index=1
  pipeline_stage_0_bwd = (f32[1,4,4,2]) call(pipeline_acts_0_bwd, pipeline_input, pipeline_weights0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\"}}"
  pipeline_weights0_new_new = f32[1,4,4,2] get-tuple-element(pipeline_stage_0_bwd), index=0
  ROOT pipeline_tuple = (f32[1,4,4,2]) tuple(pipeline_weights0_new_new)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";

  for (const std::string& hlo : {hlo_skipped_connections, hlo_same_param}) {
    auto config = GetModuleConfigForTest();
    auto module = ParseAndReturnVerifiedModule(hlo, config);
    EXPECT_TRUE(module.ok());
    auto* module0 = module.ValueOrDie().get();
    HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

    auto stages_or = GetPipelineStages(pipeline_computation);
    EXPECT_TRUE(stages_or.ok());
    auto stages = stages_or.ValueOrDie();
    auto duplicated_or = DuplicateGTEEdges(stages);
    EXPECT_TRUE(duplicated_or.ok());

    auto analysis_or = PipelineDataflowAnalysis::GetAnalysis(stages);
    EXPECT_FALSE(analysis_or.ok());
  }
}

TEST_F(PipelineUtilTest, VerifyPipelineStagesAfterLoweringOK) {
  std::string hlo = GetCorrectPipelineStages();
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  HloInstruction* pipeline_inst = FindInstruction(module0, "e.call");
  auto valid = VerifyPipelineStagesAfterLowering(pipeline_inst);
  EXPECT_TRUE(valid.ok());
}

TEST_F(PipelineUtilTest, VerifyPipelineStagesAfterLoweringNotOK) {
  for (const std::string& hlo :
       {GetInputsNeedToBeLowered(), GetNeedToLowerPipelineStages()}) {
    auto config = GetModuleConfigForTest();
    auto module = ParseAndReturnVerifiedModule(hlo, config);
    EXPECT_TRUE(module.ok());
    auto* module0 = module.ValueOrDie().get();
    HloInstruction* pipeline_inst = FindInstruction(module0, "e.call");
    auto valid = VerifyPipelineStagesAfterLowering(pipeline_inst);
    EXPECT_FALSE(valid.ok());
  }
}

TEST_F(PipelineUtilTest, TestGetUnusedParametersInPipelineStage) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_0_fwd_weights2 = f32[1,4,4,2] parameter(2)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_0_fwd_weights0, stage_0_fwd_weights2)
}

stage_1_fwd {
  stage_1_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights2 = f32[1,4,4,2] parameter(1)
  ROOT stage_1_fwd_tuple = (f32[], f32[1,4,4,2]) tuple(stage_1_fwd_weights0, stage_1_fwd_weights2)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_weights2 = f32[1,4,4,2] parameter(2)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1, pipeline_weights2), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_stage_0_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_stage_0_w2 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_stage_0_w0, pipeline_stage_0_w2), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_stage_1_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  pipeline_stage_1_w2 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_1_w0, pipeline_stage_1_w2)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.weights2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1, e.weights2), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  HloInstruction* pipeline_stage_0 =
      FindInstruction(module0, "pipeline_stage_0");
  {
    auto unused_or = GetUnusedParametersInPipelineStage(pipeline_stage_0);
    EXPECT_TRUE(unused_or.ok());
    auto unused = unused_or.ValueOrDie();
    EXPECT_THAT(unused, ::testing::ElementsAre(1));
  }
  HloInstruction* pipeline_stage_1 =
      FindInstruction(module0, "pipeline_stage_1");
  {
    auto unused_or = GetUnusedParametersInPipelineStage(pipeline_stage_1);
    EXPECT_TRUE(unused_or.ok());
    auto unused = unused_or.ValueOrDie();
    EXPECT_TRUE(unused.empty());
  }
}

TEST_F(PipelineUtilTest, TestRemoveParametersFromStage) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_const = f32[] parameter(1)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2]) tuple(stage_0_fwd_weights0)
}

stage_1_fwd {
  stage_1_const1 = f32[] parameter(0)
  stage_1_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_1_const2 = f32[] parameter(2)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2]) tuple(stage_1_fwd_weights1)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_const1 = f32[] constant(0.01)
  pipeline_const2 = f32[] constant(0.01)
  pipeline_stage_0 = (f32[1,4,4,2]) call(pipeline_weights0, pipeline_const1), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_stage_0_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[1,4,4,2]) call(pipeline_const1, pipeline_weights1, pipeline_const2), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_stage_1_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_0_w0, pipeline_stage_1_w1)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  HloInstruction* pipeline_stage_0 =
      FindInstruction(module0, "pipeline_stage_0");
  HloInstruction* pipeline_stage_1 =
      FindInstruction(module0, "pipeline_stage_1");

  HloInstruction* pipeline_const1 = FindInstruction(module0, "pipeline_const1");
  EXPECT_THAT(
      pipeline_const1->users(),
      ::testing::UnorderedElementsAre(pipeline_stage_0, pipeline_stage_1));

  HloInstruction* pipeline_const2 = FindInstruction(module0, "pipeline_const2");
  EXPECT_THAT(pipeline_const2->users(),
              ::testing::UnorderedElementsAre(pipeline_stage_1));

  HloInstruction* pipeline_weights0 =
      FindInstruction(module0, "pipeline_weights0");
  EXPECT_THAT(pipeline_stage_0->operands(),
              ::testing::ElementsAre(pipeline_weights0, pipeline_const1));
  auto new_stage_0_or = RemoveParametersFromStage(pipeline_stage_0, {1});
  EXPECT_TRUE(new_stage_0_or.ok());
  auto new_stage_0 = new_stage_0_or.ValueOrDie();
  EXPECT_THAT(new_stage_0->operands(),
              ::testing::ElementsAre(pipeline_weights0));

  EXPECT_THAT(pipeline_const1->users(),
              ::testing::UnorderedElementsAre(pipeline_stage_1));
  HloInstruction* pipeline_weights1 =
      FindInstruction(module0, "pipeline_weights1");
  EXPECT_THAT(pipeline_stage_1->operands(),
              ::testing::ElementsAre(pipeline_const1, pipeline_weights1,
                                     pipeline_const2));
  auto new_stage_1_or = RemoveParametersFromStage(pipeline_stage_1, {0, 2});
  EXPECT_TRUE(new_stage_1_or.ok());
  auto new_stage_1 = new_stage_1_or.ValueOrDie();
  EXPECT_THAT(new_stage_1->operands(),
              ::testing::ElementsAre(pipeline_weights1));
}

TEST_F(PipelineUtilTest, AddInstructionsToPipelineStageTest1) {
  // Test that we can lower outputs of a stage back into a stage.
  std::string hlo = GetNeedToLowerPipelineStages();

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  HloInstruction* pipeline_weights1 =
      FindInstruction(module0, "pipeline_weights1");

  HloInstruction* pipeline_stage_1 =
      FindInstruction(module0, "pipeline_stage_1");
  EXPECT_THAT(pipeline_stage_1->operands(),
              ::testing::Contains(pipeline_weights1));

  HloInstruction* pipeline_acts_0_local =
      FindInstruction(module0, "pipeline_acts_0_local");
  EXPECT_THAT(pipeline_acts_0_local->operands(),
              ::testing::ElementsAre(pipeline_stage_1));

  HloInstruction* pipeline_reduce = FindInstruction(module0, "pipeline_reduce");
  EXPECT_THAT(pipeline_reduce->operands(),
              ::testing::ElementsAre(pipeline_stage_1));

  HloInstruction* pipeline_stage_1_lr =
      FindInstruction(module0, "pipeline_stage_1_lr");
  EXPECT_THAT(pipeline_stage_1_lr->operands(),
              ::testing::ElementsAre(pipeline_stage_1));

  HloInstruction* pipeline_stage_1_bwd =
      FindInstruction(module0, "pipeline_stage_1_bwd");
  EXPECT_THAT(pipeline_stage_1_bwd->operands(),
              ::testing::ElementsAre(pipeline_reduce, pipeline_acts_0_local));

  HloInstruction* pipeline_acts_0_bwd =
      FindInstruction(module0, "pipeline_acts_0_bwd");
  EXPECT_THAT(pipeline_acts_0_bwd->operands(),
              ::testing::ElementsAre(pipeline_stage_1_bwd));

  HloInstruction* pipeline_lr_bcast1 =
      FindInstruction(module0, "pipeline_lr_bcast1");
  EXPECT_THAT(pipeline_lr_bcast1->operands(),
              ::testing::ElementsAre(pipeline_stage_1_lr));

  HloInstruction* pipeline_weights1_update =
      FindInstruction(module0, "pipeline_weights1_update");
  EXPECT_THAT(pipeline_weights1_update->operands(),
              ::testing::ElementsAre(pipeline_acts_0_bwd, pipeline_lr_bcast1));

  HloInstruction* pipeline_weights1_apply =
      FindInstruction(module0, "pipeline_weights1_apply");
  EXPECT_THAT(
      pipeline_weights1_apply->operands(),
      ::testing::ElementsAre(pipeline_weights1, pipeline_weights1_update));

  HloInstruction* pipeline_tuple = FindInstruction(module0, "pipeline_tuple");
  EXPECT_THAT(pipeline_tuple->operands(),
              ::testing::Contains(pipeline_weights1_apply));

  HloComputation* pipeline_computation = pipeline_tuple->parent();
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  TF_ASSERT_OK_AND_ASSIGN(auto duplicated, DuplicateGTEEdges(stages));
  EXPECT_TRUE(duplicated);
  EXPECT_TRUE(
      Match(pipeline_stage_1_bwd->to_apply()->root_instruction(),
            m::Tuple(m::Add(m::Parameter(1), m::Broadcast(m::Parameter(0))))));
  // Lower pipeline_lr_bcast1, pipeline_weights1_update and
  // pipeline_weights1_apply into pipeline_stage_1_bwd.
  TF_ASSERT_OK_AND_ASSIGN(
      pipeline_stage_1_bwd,
      AddInstructionsToPipelineStage(
          pipeline_stage_1_bwd, {pipeline_lr_bcast1, pipeline_weights1_update,
                                 pipeline_weights1_apply}));
  // Check pipeline_weights1 is used by the new stage.
  EXPECT_THAT(pipeline_stage_1_bwd->operands(),
              ::testing::Contains(pipeline_weights1));
  EXPECT_TRUE(Match(
      pipeline_stage_1_bwd->to_apply()->root_instruction(),
      m::Tuple(m::Add(m::Parameter(1), m::Broadcast(m::Parameter(0))),
               m::Subtract(m::Parameter(),
                           m::Multiply(m::Add(m::Parameter(1),
                                              m::Broadcast(m::Parameter(0))),
                                       m::Broadcast(m::Parameter()))))));
}

TEST_F(PipelineUtilTest, AddInstructionsToPipelineStageTest2) {
  // Test that we can lower inputs into a stage.
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2]) tuple(stage_0_fwd_weights0)
}

stage_1_fwd {
  stage_1_bcast1 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights0 = f32[1,4,4,2] parameter(1)
  stage_1_add = f32[1,4,4,2] add(stage_1_bcast1, stage_1_fwd_weights0)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2]) tuple(stage_1_add)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_stage_0_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_const1 = f32[] constant(0.01)
  pipeline_bcast1 = f32[1,4,4,2] broadcast(pipeline_const1), dimensions={}
  pipeline_stage_1 = (f32[1,4,4,2]) call(pipeline_bcast1, pipeline_stage_0_w0), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_stage_1_add = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  ROOT pipeline_tuple = (f32[1,4,4,2]) tuple(pipeline_stage_1_add)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  HloInstruction* pipeline_stage_0_w0 =
      FindInstruction(module0, "pipeline_stage_0_w0");
  HloInstruction* pipeline_const1 = FindInstruction(module0, "pipeline_const1");
  HloInstruction* pipeline_bcast1 = FindInstruction(module0, "pipeline_bcast1");
  EXPECT_THAT(pipeline_bcast1->operands(),
              ::testing::ElementsAre(pipeline_const1));
  HloInstruction* pipeline_stage_1 =
      FindInstruction(module0, "pipeline_stage_1");
  EXPECT_THAT(pipeline_stage_1->operands(),
              ::testing::ElementsAre(pipeline_bcast1, pipeline_stage_0_w0));
  EXPECT_TRUE(Match(pipeline_stage_1->to_apply()->root_instruction(),
                    m::Tuple(m::Add(m::Parameter(0), m::Parameter(1)))));
  // Lower pipeline_const1 and pipeline_bcast1 into pipeline_stage_1 and replace
  // any uses of parameter 0 with lowered version of pipeline_bcast1.
  TF_ASSERT_OK_AND_ASSIGN(
      pipeline_stage_1,
      AddInstructionsToPipelineStage(pipeline_stage_1,
                                     {pipeline_const1, pipeline_bcast1},
                                     {{0, pipeline_bcast1}}));
  EXPECT_TRUE(
      Match(pipeline_stage_1->to_apply()->root_instruction(),
            m::Tuple(m::Add(m::Broadcast(m::Constant()), m::Parameter(1)))));
  EXPECT_THAT(
      pipeline_stage_1->to_apply()->parameter_instruction(0)->user_count(), 0);
}

TEST_F(PipelineUtilTest, AddInstructionsToPipelineStageTest3) {
  // Test that we can force parameters through.
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_input0 = f32[1,4,4,2] parameter(0)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2]) tuple(stage_0_fwd_input0)
}

stage_1_fwd {
  stage_1_fwd_input1 = f32[1,4,4,2] parameter(0)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2]) tuple(stage_1_fwd_input1)
}

pipeline {
  pipeline_input0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2]) call(pipeline_input0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_stage_0_i0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_input1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[1,4,4,2]) call(pipeline_input1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  pipeline_stage_1_i1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_0_i0, pipeline_stage_1_i1)
}

ENTRY e {
  e.input0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.input1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.input0, e.input1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  HloInstruction* pipeline_input0 = FindInstruction(module0, "pipeline_input0");
  HloInstruction* pipeline_stage_0 =
      FindInstruction(module0, "pipeline_stage_0");
  EXPECT_THAT(pipeline_stage_0->operands(),
              ::testing::ElementsAre(pipeline_input0));
  HloInstruction* pipeline_stage_0_i0 =
      FindInstruction(module0, "pipeline_stage_0_i0");
  EXPECT_THAT(pipeline_stage_0_i0->operands(),
              ::testing::ElementsAre(pipeline_stage_0));
  HloInstruction* pipeline_input1 = FindInstruction(module0, "pipeline_input1");
  HloInstruction* pipeline_stage_1 =
      FindInstruction(module0, "pipeline_stage_1");
  EXPECT_THAT(pipeline_stage_1->operands(),
              ::testing::ElementsAre(pipeline_input1));

  EXPECT_TRUE(Match(pipeline_stage_0->to_apply()->root_instruction(),
                    m::Tuple(m::Parameter(0))));
  // Force pipeline_input1 through pipeline_stage_1.
  TF_ASSERT_OK_AND_ASSIGN(
      pipeline_stage_0, AddInstructionsToPipelineStage(pipeline_stage_0, {}, {},
                                                       {pipeline_input1}));
  EXPECT_TRUE(Match(pipeline_stage_0->to_apply()->root_instruction(),
                    m::Tuple(m::Parameter(0), m::Parameter(1))));
  EXPECT_THAT(pipeline_stage_0->operands(),
              ::testing::ElementsAre(pipeline_input0, pipeline_input1));
  EXPECT_THAT(pipeline_stage_1->operand(0)->operand(0), pipeline_stage_0);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
