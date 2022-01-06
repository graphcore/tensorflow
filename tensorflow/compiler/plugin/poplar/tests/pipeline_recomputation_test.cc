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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_recomputation.h"

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/recomputation_checkpoint_remover.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using PipelineRecomputationTest = HloTestBase;

TEST_F(PipelineRecomputationTest, TestNoBackwardStages) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_weights1 = f32[1,4,4,2] parameter(1)
  log = f32[1,4,4,2] log(stage_0_fwd_weights1)
  add = f32[1,4,4,2] add(stage_0_fwd_weights0, log)
  stage_0_fwd_weights2 = f32[1,4,4,2] parameter(2)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(add, stage_0_fwd_weights2)
}

stage_1_fwd {
  stage_1_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights2 = f32[1,4,4,2] parameter(1)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_fwd_weights0, stage_1_fwd_weights2)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_weights2 = f32[1,4,4,2] parameter(2)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1, pipeline_weights2), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_stage_0_w2 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_stage_0_w0, pipeline_stage_0_w2), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  pipeline_stage_1_w2 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_1_w0, pipeline_weights1, pipeline_stage_1_w2)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.weights2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1, e.weights2), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":2}}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  PipelineRecomputation recomputation(true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, recomputation.Run(module.get()));
  EXPECT_FALSE(changed);
}

std::string GetRecomputationHlo() {
  return R"(
HloModule top

stage_0_fwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,2] parameter(1)
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314577"
  in2 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in12 = f32[1,4,4,2] add(in2, in1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0, in12)
}

stage_1_fwd {
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  stage_1_in12 = f32[1,4,4,2] add(in2, in1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_in12, in2)
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
  stage_0_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_bwd_1, stage_0_1), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[1,4,4,2] get-tuple-element(stage_0_bwd), index=0
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_bwd_0, stage_1_bwd_1, stage_1_bwd_2), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.in0, e.in1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":2}}}"
}
)";
}

TEST_F(PipelineRecomputationTest, TestRecomputation) {
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(GetRecomputationHlo(), config));

  HloComputation* pipeline_comp = FindComputation(module.get(), "pipeline");
  PipelineRecomputation recomputation(true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, recomputation.Run(module.get()));
  EXPECT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_comp));

  // Check that the root tuple of stage 0 has two extra outputs.
  auto stage0_fwd = stages.forward[0];
  auto stage0_fwd_root = stage0_fwd->to_apply()->root_instruction();
  ASSERT_EQ(stage0_fwd_root->operand_count(), 4);
  auto output2 = stage0_fwd_root->operand(2);
  auto output3 = stage0_fwd_root->operand(3);

  if (output2->opcode() == HloOpcode::kParameter) {
    EXPECT_TRUE(Match(output2, m::Parameter(1)));
    EXPECT_TRUE(Match(output3, m::GetTupleElement(m::Infeed(), 0)));
  } else {
    EXPECT_TRUE(Match(output2, m::GetTupleElement(m::Infeed(), 0)));
    EXPECT_TRUE(Match(output3, m::Parameter(1)));
  }

  auto stage0_bwd = stages.backward[0];
  ASSERT_EQ(stage0_bwd->operand_count(), 4);
  auto input2 = stage0_bwd->operand(2);
  auto input3 = stage0_bwd->operand(3);
  EXPECT_EQ(input2->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(input3->opcode(), HloOpcode::kGetTupleElement);
  if (input2->tuple_index() == 2) {
    EXPECT_EQ(input3->tuple_index(), 3);
  } else {
    EXPECT_EQ(input3->tuple_index(), 2);
    EXPECT_EQ(input2->tuple_index(), 3);
  }

  // Check that the add is now recomputed.
  auto stage0_bwd_root = stage0_bwd->to_apply()->root_instruction();
  EXPECT_TRUE(
      Match(stage0_bwd_root,
            m::Tuple(m::Parameter(0), m::Add(m::Parameter(), m::Parameter()))));
}

TEST_F(PipelineRecomputationTest, TestGetInstructionsToRecompute) {
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(GetRecomputationHlo(), config));

  TF_ASSERT_OK_AND_ASSIGN(
      auto insts,
      PipelineRecomputation::GetInstructionsToRecompute(module.get()));
  HloInstruction* in12 = FindInstruction(module.get(), "in12");
  EXPECT_THAT(insts, ::testing::ElementsAre(in12));
}

TEST_F(PipelineRecomputationTest, TestRecomputation2) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  in0 = f32[2] parameter(0)
  after-all = token[] after-all()
  infeed = (f32[2], token[]) infeed(after-all), infeed_config="4"
  in1 = f32[2] get-tuple-element(infeed), index=0
  add = f32[2] add(in1, in0)
  ROOT tuple = (f32[2], f32[2]) tuple(in0, add)
}

stage_1_fwd {
  ROOT tuple = () tuple()
}

stage_1_bwd {
  ROOT tuple = () tuple()
}

stage_0_bwd {
  in0 = f32[2] parameter(0)
  in1 = f32[2] parameter(1)
  sub = f32[2] subtract(in0, in1)
  ROOT tuple = (f32[2]) tuple(sub)
}

resource_update {
  arg0 = f32[2] parameter(0)
  arg1 = f32[2] parameter(1)
  add = f32[2] add(arg0, arg1)
  ROOT t = (f32[2]) tuple(add)
}

pipeline {
  in0 = f32[2] parameter(0)
  stage_0 = (f32[2], f32[2]) call(in0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[2] get-tuple-element(stage_0), index=0
  stage_0_1 = f32[2] get-tuple-element(stage_0), index=1
  stage_1 = () call(), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_bwd = () call(), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_0_bwd = (f32[2]) call(stage_0_1, stage_0_0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[2] get-tuple-element(stage_0_bwd), index=0
  call_ru = (f32[2]) call(stage_0_bwd_0, in0), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[2] get-tuple-element(call_ru), index=0
  ROOT tuple = (f32[2], f32[2]) tuple(gte0)
}

ENTRY e {
  e.in0 = f32[2] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[2]) call(e.in0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":2}}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  HloComputation* pipeline_comp = FindComputation(module.get(), "pipeline");
  PipelineRecomputation recomputation(true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, recomputation.Run(module.get()));
  EXPECT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_comp));

  // Check that the root tuple of stage 0 has two extra outputs.
  auto stage0_fwd = stages.forward[0];
  auto stage0_fwd_root = stage0_fwd->to_apply()->root_instruction();
  ASSERT_EQ(stage0_fwd_root->operand_count(), 4);
  auto output2 = stage0_fwd_root->operand(2);
  auto output3 = stage0_fwd_root->operand(3);

  if (output2->opcode() == HloOpcode::kParameter) {
    EXPECT_TRUE(Match(output2, m::Parameter(0)));
    EXPECT_TRUE(Match(output3, m::GetTupleElement(m::Infeed(), 0)));
  } else {
    EXPECT_TRUE(Match(output2, m::GetTupleElement(m::Infeed(), 0)));
    EXPECT_TRUE(Match(output3, m::Parameter(0)));
  }

  auto stage0_bwd = stages.backward[0];
  ASSERT_EQ(stage0_bwd->operand_count(), 4);
  auto input2 = stage0_bwd->operand(2);
  auto input3 = stage0_bwd->operand(3);
  EXPECT_EQ(input2->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(input3->opcode(), HloOpcode::kGetTupleElement);
  if (input2->tuple_index() == 2) {
    EXPECT_EQ(input3->tuple_index(), 3);
  } else {
    EXPECT_EQ(input3->tuple_index(), 2);
    EXPECT_EQ(input2->tuple_index(), 3);
  }

  // Check that the add is now recomputed.
  auto stage0_bwd_root = stage0_bwd->to_apply()->root_instruction();
  EXPECT_TRUE(Match(stage0_bwd_root,
                    m::Tuple(m::Subtract(m::Add(m::Parameter(), m::Parameter()),
                                         m::Parameter()))));
}

TEST_F(PipelineRecomputationTest, TestRecomputationWithCheckpoints) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  in0 = f32[2] parameter(0)
  in1 = f32[2] parameter(1)
  after-all = token[] after-all()
  infeed = (f32[2], token[]) infeed(after-all), infeed_config="4"
  gte = f32[2] get-tuple-element(infeed), index=0
  checkpoint1 = f32[2] custom-call(gte), custom_call_target="RecomputationCheckpoint"
  add = f32[2] add(checkpoint1, in0)
  checkpoint2 = f32[2] custom-call(add), custom_call_target="RecomputationCheckpoint"
  add2 = f32[2] add(checkpoint2, in1)
  ROOT tuple = (f32[2], f32[2]) tuple(in0, add2)
}

stage_1_fwd {
  ROOT tuple = () tuple()
}

stage_1_bwd {
  ROOT tuple = () tuple()
}

stage_0_bwd {
  in0 = f32[2] parameter(0)
  in1 = f32[2] parameter(1)
  in2 = f32[2] parameter(2)
  sub = f32[2] subtract(in0, in1)
  sub2 = f32[2] subtract(sub, in2)
  ROOT tuple = (f32[2]) tuple(sub2)
}

resource_update {
  arg0 = f32[2] parameter(0)
  arg1 = f32[2] parameter(1)
  arg2 = f32[2] parameter(2)
  add = f32[2] add(arg0, arg1)
  add2 = f32[2] add(arg2, add)
  ROOT t = (f32[2], f32[2]) tuple(add, add2)
}

pipeline {
  in0 = f32[2] parameter(0)
  in1 = f32[2] parameter(1)
  stage_0 = (f32[2], f32[2]) call(in0, in1), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[2] get-tuple-element(stage_0), index=0
  stage_0_1 = f32[2] get-tuple-element(stage_0), index=1
  stage_1 = () call(), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_bwd = () call(), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_0_bwd = (f32[2]) call(stage_0_1, stage_0_0, in1), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[2] get-tuple-element(stage_0_bwd), index=0
  call_ru = (f32[2], f32[2]) call(stage_0_bwd_0, in0, in1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[2] get-tuple-element(call_ru), index=0
  gte1 = f32[2] get-tuple-element(call_ru), index=1
  ROOT tuple = (f32[2], f32[2]) tuple(gte0, gte1)
}

ENTRY e {
  e.in0 = f32[2] parameter(0)
  e.in1 = f32[2] parameter(1)
  ROOT e.call = (f32[2],f32[2]) call(e.in0,e.in1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":2}}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());

  HloComputation* pipeline_comp = FindComputation(module.get(), "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          PipelineRecomputation(true).Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_comp));

  // Check that the root tuple of stage 0 has five extra outputs.
  auto stage0_fwd = stages.forward[0];
  auto stage0_fwd_root = stage0_fwd->to_apply()->root_instruction();
  ASSERT_EQ(stage0_fwd_root->operand_count(), 7);

  auto stage0_bwd = stages.backward[0];
  ASSERT_EQ(stage0_bwd->operand_count(), 8);
  for (int64 i = 3; i != 8; ++i) {
    EXPECT_TRUE(Match(stage0_bwd->operand(i),
                      m::GetTupleElement(m::Op().Is(stage0_fwd))));
  }

  // Check the recomputation.
  auto stage0_bwd_root = stage0_bwd->to_apply()->root_instruction();
  HloInstruction* recomputation_input;
  EXPECT_TRUE(Match(
      stage0_bwd_root,
      m::Tuple(m::Subtract(
          m::Subtract(m::Add(m::Op(&recomputation_input), m::Parameter(7)),
                      m::Parameter(1)),
          m::Parameter(2)))));

  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::RecomputationInput, recomputation_input));
  EXPECT_TRUE(Match(
      recomputation_input,
      m::CustomCall(m::Parameter(6),
                    m::Add(m::Op(&recomputation_input), m::Parameter(5)))));

  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::RecomputationInput, recomputation_input));
  EXPECT_TRUE(Match(recomputation_input,
                    m::CustomCall(m::Parameter(3), m::Parameter(4))));
}

TEST_F(PipelineRecomputationTest, TestRecomputationWithCheckpointsInFinal) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  after-all = token[] after-all()
  infeed = (f32[2], token[]) infeed(after-all), infeed_config="4"
  gte = f32[2] get-tuple-element(infeed), index=0
  ROOT tuple = (f32[2]) tuple(gte)
}

stage_1_fwd {
  var0 = f32[2] parameter(0)
  var1 = f32[2] parameter(1)
  x = f32[2] parameter(2)
  checkpoint1 = f32[2] custom-call(x), custom_call_target="RecomputationCheckpoint"
  add = f32[2] add(checkpoint1, var0)
  checkpoint2 = f32[2] custom-call(add), custom_call_target="RecomputationCheckpoint"
  add2 = f32[2] add(checkpoint2, var1)
  ROOT tuple = (f32[2], f32[2], f32[2]) tuple(x, add, add2)
}

stage_1_bwd {
  stage_1_x = f32[2] parameter(0)
  stage_1_add = f32[2] parameter(1)
  stage_1_add2 = f32[2] parameter(2)
  var0 = f32[2] parameter(3)
  var1 = f32[2] parameter(4)
  var1g = f32[2] multiply(stage_1_add, stage_1_add2)
  add1g = f32[2] multiply(var1, stage_1_add2)
  var0g = f32[2] multiply(stage_1_x, add1g)
  addg = f32[2] multiply(var0, add1g)
  ROOT tuple = (f32[2], f32[2]) tuple(var0g, var1g)
}

stage_0_bwd {
  ROOT tuple = () tuple()
}

resource_update {
  var0g = f32[2] parameter(0)
  var1g = f32[2] parameter(1)
  var0 = f32[2] parameter(2)
  var1 = f32[2] parameter(3)
  var0new = f32[2] add(var0, var0g)
  var1new = f32[2] add(var1, var1g)
  ROOT t = (f32[2], f32[2]) tuple(var0new, var1new)
}

pipeline {
  var0 = f32[2] parameter(0)
  var1 = f32[2] parameter(1)
  stage_0 = (f32[2]) call(), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[2] get-tuple-element(stage_0), index=0
  stage_1 = (f32[2], f32[2], f32[2]) call(var0, var1, stage_0_0), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_x = f32[2] get-tuple-element(stage_1), index=0
  stage_1_add = f32[2] get-tuple-element(stage_1), index=1
  stage_1_add2 = f32[2] get-tuple-element(stage_1), index=2
  stage_1_bwd = (f32[2], f32[2]) call(stage_1_x, stage_1_add, stage_1_add2, var0, var1), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  var0g = f32[2] get-tuple-element(stage_1_bwd), index=0
  var1g = f32[2] get-tuple-element(stage_1_bwd), index=1
  stage_0_bwd = () call(), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  call_ru = (f32[2], f32[2]) call(var0g, var1g, var0, var1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  var0new = f32[2] get-tuple-element(call_ru), index=0
  var1new = f32[2] get-tuple-element(call_ru), index=1
  ROOT tuple = (f32[2], f32[2]) tuple(var0new, var1new)
}

ENTRY e {
  e.in0 = f32[2] parameter(0)
  e.in1 = f32[2] parameter(1)
  ROOT e.call = (f32[2],f32[2]) call(e.in0,e.in1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":2}}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());

  HloComputation* pipeline_comp = FindComputation(module.get(), "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          PipelineRecomputation(true).Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_comp));

  // Root tuple of stage 1 contains:
  // 1. its initial outputs x, add, add2
  // 2. three additional outputs checkpoint1, var0 and x to recompute "add"
  auto stage_1_fwd = stages.forward[1];
  auto stage_1_fwd_root = stage_1_fwd->to_apply()->root_instruction();
  ASSERT_EQ(stage_1_fwd_root->operand_count(), 6);

  // Stage 1's bwd pass will take these 3 additional outputs from stage 1.
  auto stage_1_bwd = stages.backward[1];
  ASSERT_EQ(stage_1_bwd->operand_count(), 8);
  for (int64 i = 5; i != 8; ++i) {
    EXPECT_TRUE(Match(stage_1_bwd->operand(i),
                      m::GetTupleElement(m::Op().Is(stage_1_fwd))));
  }

  // To compute var1g in the backward pass, we need stage_1_add, which should
  // now be recomputed with the additional inputs passed.
  // The second input to var1g (stage_2_add) should NOT be recomputed, since
  // it's past the final checkpoint in the forward stage.
  auto stage_1_bwd_root = stage_1_bwd->to_apply()->root_instruction();
  HloInstruction* checkpoint;
  HloInstruction* var0g;
  EXPECT_TRUE(
      Match(stage_1_bwd_root,
            m::Tuple(
                // output0: var0g
                m::Op(&var0g),
                // output1: var1g
                m::Multiply(
                    // Recomputation of forward stage add(ckpt, var0)
                    m::Add(
                        // Checkpointed add input
                        m::Op(&checkpoint),
                        // Forwarded var0 from forward stage will be deduped
                        m::Parameter(6)),
                    // stage_1_add2 NOT recomputed but passed in.
                    m::Parameter(2)))));

  // The checkpoint shouldn't be converted to a RecomputationInput since it's
  // checkpointing a parameter so there's no pivoting to be done.
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::RecomputationCheckpoint, checkpoint));
  EXPECT_TRUE(Match(checkpoint, m::CustomCall(m::Parameter(5))));

  // The final checkpoint will become a recomputation input which we
  // can pivot the operations around later to reduce liveness, but in this case
  // nothing past the final checkpoint is needed in the backward stage since it
  // feeds into the last op of the stage, so it has no users.
}

TEST_F(PipelineRecomputationTest, TestRecomputationNoCheckpointsInFinal) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  after-all = token[] after-all()
  infeed = (f32[2], token[]) infeed(after-all), infeed_config="4"
  gte = f32[2] get-tuple-element(infeed), index=0
  ROOT tuple = (f32[2]) tuple(gte)
}

stage_1_fwd {
  var0 = f32[2] parameter(0)
  var1 = f32[2] parameter(1)
  x = f32[2] parameter(2)
  add = f32[2] add(x, var0)
  add2 = f32[2] add(add, var1)
  ROOT tuple = (f32[2], f32[2], f32[2]) tuple(x, add, add2)
}

stage_1_bwd {
  stage_1_x = f32[2] parameter(0)
  stage_1_add = f32[2] parameter(1)
  stage_1_add2 = f32[2] parameter(2)
  var0 = f32[2] parameter(3)
  var1 = f32[2] parameter(4)
  var1g = f32[2] multiply(stage_1_add, stage_1_add2)
  add1g = f32[2] multiply(var1, stage_1_add2)
  var0g = f32[2] multiply(stage_1_x, add1g)
  addg = f32[2] multiply(var0, add1g)
  ROOT tuple = (f32[2], f32[2]) tuple(var0g, var1g)
}

stage_0_bwd {
  ROOT tuple = () tuple()
}

resource_update {
  var0g = f32[2] parameter(0)
  var1g = f32[2] parameter(1)
  var0 = f32[2] parameter(2)
  var1 = f32[2] parameter(3)
  var0new = f32[2] add(var0, var0g)
  var1new = f32[2] add(var1, var1g)
  ROOT t = (f32[2], f32[2]) tuple(var0new, var1new)
}

pipeline {
  var0 = f32[2] parameter(0)
  var1 = f32[2] parameter(1)
  stage_0 = (f32[2]) call(), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[2] get-tuple-element(stage_0), index=0
  stage_1 = (f32[2], f32[2], f32[2]) call(var0, var1, stage_0_0), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_x = f32[2] get-tuple-element(stage_1), index=0
  stage_1_add = f32[2] get-tuple-element(stage_1), index=1
  stage_1_add2 = f32[2] get-tuple-element(stage_1), index=2
  stage_1_bwd = (f32[2], f32[2]) call(stage_1_x, stage_1_add, stage_1_add2, var0, var1), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  var0g = f32[2] get-tuple-element(stage_1_bwd), index=0
  var1g = f32[2] get-tuple-element(stage_1_bwd), index=1
  stage_0_bwd = () call(), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  call_ru = (f32[2], f32[2]) call(var0g, var1g, var0, var1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  var0new = f32[2] get-tuple-element(call_ru), index=0
  var1new = f32[2] get-tuple-element(call_ru), index=1
  ROOT tuple = (f32[2], f32[2]) tuple(var0new, var1new)
}

ENTRY e {
  e.in0 = f32[2] parameter(0)
  e.in1 = f32[2] parameter(1)
  ROOT e.call = (f32[2],f32[2]) call(e.in0,e.in1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":2}}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  HloComputation* pipeline_comp = FindComputation(module.get(), "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          PipelineRecomputation(true).Run(module.get()));

  // Since there's no checkpoints in the final stage, it should not be
  // recomputed.
  EXPECT_FALSE(changed);
}

TEST_F(PipelineRecomputationTest,
       TestRecomputationWithCheckpointsInputAndOutput) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  in0 = f32[2] parameter(0)
  checkpoint1 = f32[2] custom-call(in0), custom_call_target="RecomputationCheckpoint"
  after-all = token[] after-all()
  infeed = (f32[2], token[]) infeed(after-all), infeed_config="4"
  in1 = f32[2] get-tuple-element(infeed), index=0
  add = f32[2] add(in1, checkpoint1)
  checkpoint2 = f32[2] custom-call(add), custom_call_target="RecomputationCheckpoint"
  checkpoint3 = f32[2] custom-call(add), custom_call_target="RecomputationCheckpoint"
  add2 = f32[2] add(in1, checkpoint3)
  ROOT tuple = (f32[2], f32[2]) tuple(in0, checkpoint2, add2)
}

stage_1_fwd {
  ROOT tuple = () tuple()
}

stage_1_bwd {
  ROOT tuple = () tuple()
}

stage_0_bwd {
  in0 = f32[2] parameter(0)
  in1 = f32[2] parameter(1)
  in2 = f32[2] parameter(2)
  sub = f32[2] subtract(in0, in1)
  sub2 = f32[2] subtract(sub, in2)
  ROOT tuple = (f32[2]) tuple(sub2)
}

resource_update {
  arg0 = f32[2] parameter(0)
  arg1 = f32[2] parameter(1)
  add = f32[2] add(arg0, arg1)
  ROOT t = (f32[2]) tuple(add)
}

pipeline {
  in0 = f32[2] parameter(0)
  stage_0 = (f32[2], f32[2], f32[2]) call(in0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[2] get-tuple-element(stage_0), index=0
  stage_0_1 = f32[2] get-tuple-element(stage_0), index=1
  stage_0_2 = f32[2] get-tuple-element(stage_0), index=2
  stage_1 = () call(), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_bwd = () call(), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_0_bwd = (f32[2]) call(stage_0_1, stage_0_0, stage_0_2), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[2] get-tuple-element(stage_0_bwd), index=0
  call_ru = (f32[2]) call(stage_0_bwd_0, in0), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[2] get-tuple-element(call_ru), index=0
  ROOT tuple = (f32[2], f32[2]) tuple(gte0)
}

ENTRY e {
  e.in0 = f32[2] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[2]) call(e.in0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":2}}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());

  HloComputation* pipeline_comp = FindComputation(module.get(), "pipeline");
  PipelineRecomputation recomputation(true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, recomputation.Run(module.get()));
  EXPECT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_comp));

  // Check that the root tuple of stage 0 has three extra outputs.
  auto stage0_fwd = stages.forward[0];
  auto stage0_fwd_root = stage0_fwd->to_apply()->root_instruction();
  ASSERT_EQ(stage0_fwd_root->operand_count(), 6);
  auto output3 = stage0_fwd_root->operand(3);
  auto output4 = stage0_fwd_root->operand(4);

  if (output3->opcode() == HloOpcode::kParameter) {
    EXPECT_TRUE(Match(output3, m::Parameter(0)));
    EXPECT_TRUE(Match(output4, m::GetTupleElement(m::Infeed(), 0)));
  } else {
    EXPECT_TRUE(Match(output3, m::GetTupleElement(m::Infeed(), 0)));
    EXPECT_TRUE(Match(output4, m::Parameter(0)));
  }

  auto stage0_bwd = stages.backward[0];
  ASSERT_EQ(stage0_bwd->operand_count(), 6);
  auto input2 = stage0_bwd->operand(2);
  auto input3 = stage0_bwd->operand(3);
  EXPECT_EQ(input2->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(input3->opcode(), HloOpcode::kGetTupleElement);
  if (input2->tuple_index() == 2) {
    EXPECT_EQ(input3->tuple_index(), 3);
  } else {
    EXPECT_EQ(input3->tuple_index(), 2);
    EXPECT_EQ(input2->tuple_index(), 3);
  }

  // Check that the adds are now recomputed.
  auto stage0_bwd_root = stage0_bwd->to_apply()->root_instruction();

  HloInstruction* checkpoint3;
  HloInstruction* subtract;
  EXPECT_TRUE(Match(
      stage0_bwd_root,
      m::Tuple(m::Subtract(m::Op(&subtract),
                           m::Add(m::Parameter(), m::Op(&checkpoint3))))));
  // checkpoint3 is converted to a RecomputationInput even though it's a stage
  // output since it has a non-root user.
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::RecomputationInput, checkpoint3));

  // checkpoint1 isn't converted to a RecomputationInput because it's
  // checkpointing a parameter.
  HloInstruction* checkpoint1;
  EXPECT_TRUE(Match(
      checkpoint3, m::CustomCall(m::Parameter(),
                                 m::Add(m::Parameter(), m::Op(&checkpoint1)))));
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::RecomputationCheckpoint, checkpoint1));

  // checkpoint2 isn't converted to a RecomputationInput because its only a
  // stage output.
  HloInstruction* checkpoint2;
  EXPECT_TRUE(
      Match(subtract, m::Subtract(m::Op(&checkpoint2), m::Parameter())));
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::RecomputationCheckpoint, checkpoint2));

  EXPECT_TRUE(RecomputationCheckpointRemover().Run(module.get()).ValueOrDie());

  // It has been removed now.
  EXPECT_TRUE(Match(
      stage0_bwd_root,
      m::Tuple(m::Subtract(
          m::Subtract(m::Add(m::Parameter(), m::Parameter()), m::Parameter()),
          m::Add(m::Parameter(),
                 m::CustomCall(m::Parameter(),
                               m::Add(m::Parameter(), m::Parameter())))))));
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
