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

#include "tensorflow/compiler/plugin/poplar/driver/passes/copy_inserter.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_information.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using CopyInserterTest = HloTestBase;

TEST_F(CopyInserterTest, DontInsertCopyParams) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT t = (f32[], f32[], f32[], f32[]) tuple(p0, p1, p1, p0)
}

)";
  auto module = ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  EXPECT_FALSE(CopyInserter().Run(module0).ValueOrDie());
}

TEST_F(CopyInserterTest, DontInsertCopyParamsAndConst) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  c = f32[] constant(0)
  ROOT t = (f32[], f32[], f32[], f32[]) tuple(p0, p1, c, p0)
}

)";
  auto module = ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  EXPECT_FALSE(CopyInserter().Run(module0).ValueOrDie());
}

TEST_F(CopyInserterTest, InsertCopy) {
  std::string hlo = R"(
HloModule top

body {
  p = (s32[4],s32[4]) parameter(0)
  p.0 = s32[4] get-tuple-element(p), index=0
  p.1 = s32[4] get-tuple-element(p), index=1
  c.0 = s32[8] concatenate(p.0, p.1), dimensions={0}
  s.0 = s32[4] slice(c.0), slice={[2:6]}
  a.0 = s32[4] add(p.0, p.0)
  ROOT root = (s32[4],s32[4]) tuple(a.0, s.0)
}

condition {
  p_cond = (s32[4],s32[4]) parameter(0)
  p_cond.0 = s32[4] get-tuple-element(p_cond), index=0
  p_s0 = s32[1] slice(p_cond.0), slice={[0:1]}
  p_s1 = s32[] reshape(p_s0)
  p_const = s32[] constant(10)
  ROOT result = pred[] compare(p_s1, p_const), direction=LT
}

ENTRY entry {
  const_0 = s32[4] constant({0, 0, 0, 0})
  repeat_init = (s32[4],s32[4]) tuple(const_0, const_0)
  ROOT while = (s32[4],s32[4]) while(repeat_init), condition=condition, body=body
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  EXPECT_TRUE(CopyInserter().Run(module0).ValueOrDie());
  HloInstruction *op0, *op1;
  EXPECT_TRUE(Match(
      module0->entry_computation()->root_instruction()->mutable_operand(0),
      m::Tuple(m::Op(&op0), m::Copy(m::Op(&op1)))));
  EXPECT_EQ(op0, op1);
}

TEST_F(CopyInserterTest, InsertResourceUpdate) {
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
  stage_1_fwd_acts_1 = f32[1,4,4,2] add(stage_1_fwd_acts_0, stage_1_fwd_weights1) stage_1_fwd_zero = f32[] constant(0)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_fwd_acts_1, stage_1_fwd_acts_0)
}

stage_2_fwd {
  stage_2_fwd_acts_0 = f32[1,4,4,2] parameter(0)
  stage_2_fwd_acts_1 = f32[1,4,4,2] parameter(1)
  stage_2_fwd_acts_2 = f32[1,4,4,2] add(stage_2_fwd_acts_0, stage_2_fwd_acts_1)
  ROOT stage_2_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_2_fwd_acts_2, stage_2_fwd_acts_1)
}

stage_2_bwd {
  stage_2_fwd_acts_2 = f32[1,4,4,2] parameter(0)
  stage_2_fwd_acts_1 = f32[1,4,4,2] parameter(1)
  stage_2_fwd_acts_2_bwd = f32[1,4,4,2] subtract(stage_2_fwd_acts_2, stage_2_fwd_acts_1)
  ROOT stage_2_bwd_tuple = (f32[1,4,4,2]) tuple(stage_2_fwd_acts_2_bwd)
}

stage_1_bwd {
  stage_1_fwd_acts_2_bwd = f32[1,4,4,2] parameter(0)
  stage_1_fwd_acts_1 = f32[1,4,4,2] parameter(1)
  stage_1_fwd_acts_1_bwd = f32[1,4,4,2] subtract(stage_1_fwd_acts_2_bwd, stage_1_fwd_acts_1)
  ROOT stage_1_bwd_tuple = (f32[1,4,4,2]) tuple(stage_1_fwd_acts_1_bwd)
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
  add = f32[1,4,4,2] add(arg0, arg1)
  ROOT t = (f32[1,4,4,2]) tuple(add)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}

  stage_0_fwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_weights1 = f32[1,4,4,2] parameter(1)

  pipeline_stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_fwd, pipeline_weights1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_fwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  stage_0_fwd_x = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1

  pipeline_stage_2 = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_fwd_x, stage_1_fwd), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_fwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_2), index=0
  stage_1_fwd_x = f32[1,4,4,2] get-tuple-element(pipeline_stage_2), index=1

  pipeline_stage_2_bwd = (f32[1,4,4,2]) call(stage_2_fwd, stage_1_fwd_x), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  stage_2_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_2_bwd), index=0

  pipeline_stage_1_bwd = (f32[1,4,4,2]) call(stage_2_bwd, stage_1_fwd), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd), index=0

  pipeline_stage_0_bwd = (f32[1,4,4,2]) call(stage_1_bwd, stage_0_fwd), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_0_bwd), index=0

  call_ru = (f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, pipeline_weights1)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, CopyInserter().Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* call_ru = FindInstruction(module.get(), "call_ru");
  EXPECT_TRUE(Match(call_ru->operand(1), m::Copy(m::Parameter(1))));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
