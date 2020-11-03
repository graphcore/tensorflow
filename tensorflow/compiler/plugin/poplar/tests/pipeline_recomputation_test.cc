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

TEST_F(PipelineRecomputationTest, TestRecomputation) {
  std::string hlo = R"(
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
  stage_0_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_bwd_1, stage_0_1), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[1,4,4,2] get-tuple-element(stage_0_bwd), index=0
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_bwd_0, stage_1_bwd_1, stage_1_bwd_2), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
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
  call_ru = (f32[2]) call(stage_0_bwd_0, in0), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
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

TEST_F(PipelineRecomputationTest, TestRecomputationWithCheckpoints2) {
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
  call_ru = (f32[2], f32[2]) call(stage_0_bwd_0, in0, in1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
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

  // Check that the root tuple of stage 0 has two extra outputs.
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
  ROOT tuple = (f32[2], f32[2]) tuple(in0, checkpoint2)
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
  call_ru = (f32[2]) call(stage_0_bwd_0, in0), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
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
  VLOG(0) << stage0_bwd->to_apply()->ToString();
  HloInstruction* recomputation_checkpoint;
  EXPECT_TRUE(Match(
      stage0_bwd_root,
      m::Tuple(m::Subtract(m::Op(&recomputation_checkpoint), m::Parameter()))));
  // The recomputation checkpoint is not converted to a recomputation input
  // because it's used by the root in the forward stage.
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::RecomputationCheckpoint,
                                  recomputation_checkpoint));
  EXPECT_TRUE(Match(
      recomputation_checkpoint,
      m::CustomCall(m::Add(m::Parameter(), m::Op(&recomputation_checkpoint)))));
  // The recomputation checkpoint is not converted to a recomputation input
  // because it's checkpointing a parameter.
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::RecomputationCheckpoint,
                                  recomputation_checkpoint));
  EXPECT_TRUE(Match(recomputation_checkpoint, m::CustomCall(m::Parameter())));

  EXPECT_TRUE(RecomputationCheckpointRemover().Run(module.get()).ValueOrDie());

  // It has been removed now.
  EXPECT_TRUE(Match(stage0_bwd_root,
                    m::Tuple(m::Subtract(m::Add(m::Parameter(), m::Parameter()),
                                         m::Parameter()))));
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
