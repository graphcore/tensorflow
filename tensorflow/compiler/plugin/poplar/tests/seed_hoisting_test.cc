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

#include "tensorflow/compiler/plugin/poplar/driver/passes/seed_hoisting.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_information.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {
class SeedHoistingTest : public HloTestBase {
 protected:
  std::pair<HloInstruction*, HloInstruction*> MatchHashCombine(
      HloInstruction* inst) {
    HloInstruction *seed0, *seed1, *seed2, *counter;
    EXPECT_TRUE(Match(
        inst, m::BitcastConvert(m::Xor(
                  m::BitcastConvert(m::Op(&seed0)),
                  m::Add(m::Add(m::Broadcast(
                                    m::Add(m::BitcastConvert(m::Op(&counter)),
                                           m::Constant())),
                                m::ShiftLeft(m::BitcastConvert(m::Op(&seed1)),
                                             m::Constant())),
                         m::ShiftRightLogical(m::BitcastConvert(m::Op(&seed2)),
                                              m::Constant()))))));
    EXPECT_THAT(seed0, seed1);
    EXPECT_THAT(seed0, seed2);
    return {seed0, counter};
  }
};

TEST_F(SeedHoistingTest, NothingToHoist) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  ROOT s = s32[2] custom-call(), custom_call_target="Seed", backend_config=""
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest()));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_FALSE(SeedHoisting().Run(module.get()).ValueOrDie());
}

TEST_F(SeedHoistingTest, HosistFromFunction) {
  std::string hlo = R"(
HloModule top

func {
  p0 = s32[2] parameter(0)
  p1 = s32[2] parameter(1)
  s = s32[2] custom-call(), custom_call_target="Seed", backend_config=""
  a = s32[2] add(p0, s)
  ROOT b = s32[2] add(a, p1)
}

ENTRY e {
  p0 = s32[2] parameter(0)
  p1 = s32[2] parameter(1)
  ROOT c = s32[2] call(p0, p1), to_apply=func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest()));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(SeedHoisting().Run(module.get()).ValueOrDie());

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root->operand_count(), 3);
  EXPECT_THAT(root->operand(0)->parameter_number(), 0);
  EXPECT_THAT(root->operand(1)->parameter_number(), 1);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::Seed)(root->operand(2)));

  auto func = root->to_apply();
  EXPECT_THAT(func->instruction_count(), 5);
  auto func_root = func->root_instruction();
  ASSERT_TRUE(Match(func_root, m::Add(m::Add(m::Parameter(0), m::Parameter(2)),
                                      m::Parameter(1))));
}

TEST_F(SeedHoistingTest, NestedFunction) {
  std::string hlo = R"(
HloModule top

func2 {
  p0 = s32[2] parameter(0)
  s = s32[2] custom-call(), custom_call_target="Seed", backend_config=""
  ROOT b = s32[2] add(s, p0)
}

func {
  p0 = s32[2] parameter(0)
  p1 = s32[2] parameter(1)
  a = s32[2] call(p0), to_apply=func2, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  ROOT b = s32[2] add(a, p1)
}

ENTRY e {
  p0 = s32[2] parameter(0)
  p1 = s32[2] parameter(1)
  ROOT c = s32[2] call(p0, p1), to_apply=func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest()));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  int64 exec_counter = 0;
  while (SeedHoisting().Run(module.get()).ValueOrDie()) {
    exec_counter++;
  }
  EXPECT_EQ(exec_counter, 2);

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root->operand_count(), 3);
  EXPECT_THAT(root->operand(0)->parameter_number(), 0);
  EXPECT_THAT(root->operand(1)->parameter_number(), 1);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::Seed)(root->operand(2)));

  auto func = root->to_apply();
  EXPECT_THAT(func->instruction_count(), 5);
  auto func_root = func->root_instruction();
  HloInstruction* func2_inst;
  ASSERT_TRUE(Match(func_root, m::Add(m::Op(&func2_inst), m::Parameter(1))));
  EXPECT_THAT(func2_inst->operand(0)->parameter_number(), 0);
  EXPECT_THAT(func2_inst->operand(1)->parameter_number(), 2);

  auto func2 = func2_inst->to_apply();
  EXPECT_THAT(func->instruction_count(), 5);
  auto func2_root = func2->root_instruction();
  ASSERT_TRUE(Match(func2_root, m::Add(m::Parameter(1), m::Parameter(0))));
}

TEST_F(SeedHoistingTest, Pipeliening) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_p0 = s32[2] parameter(0)
  stage_0_p1 = s32[2] parameter(1)
  add = s32[2] add(stage_0_p0, stage_0_p1)
  s = s32[2] custom-call(), custom_call_target="Seed", backend_config=""
  ROOT stage_0_tuple = (s32[2], s32[2]) tuple(add, s)
}

stage_1_fwd {
  stage_1_p0 = s32[2] parameter(0)
  stage_1_p2 = s32[2] parameter(1)
  ROOT stage_1_tuple = (s32[2], s32[2]) tuple(stage_1_p0, stage_1_p2)
}

pipeline {
  pipeline_p0 = s32[2] parameter(0)
  pipeline_p1 = s32[2] parameter(1)
  pipeline_stage_0 = (s32[2], s32[2]) call(pipeline_p0, pipeline_p1), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_0 = s32[2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_stage_0_1 = s32[2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_stage_1 = (s32[2], s32[2]) call(pipeline_stage_0_0, pipeline_stage_0_1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_0 = s32[2] get-tuple-element(pipeline_stage_1), index=0
  pipeline_stage_1_1 = s32[2] get-tuple-element(pipeline_stage_1), index=1
  ROOT pipeline_tuple = (s32[2], s32[2]) tuple(pipeline_stage_0_0, pipeline_stage_0_1)
}

ENTRY e {
  e.p0 = s32[2] parameter(0), parameter_replication={false}
  e.p1 = s32[2] parameter(1), parameter_replication={false}
  e.call = (s32[2], s32[2]) call(e.p0, e.p1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  ROOT t = ((s32[2], s32[2])) tuple(e.call)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest()));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(SeedHoisting().Run(module.get()).ValueOrDie());

  HloInstruction *pipeline0, *pipeline1;
  auto root = module->entry_computation()->root_instruction();
  ASSERT_TRUE(Match(
      root, m::Tuple(m::Tuple(m::GetTupleElement(m::Op(&pipeline0), 0),
                              m::GetTupleElement(m::Op(&pipeline1), 1)))));
  EXPECT_THAT(pipeline0, pipeline1);
  EXPECT_TRUE(IsPipelineOp(pipeline0));
  EXPECT_THAT(pipeline0->operand_count(), 3);
  EXPECT_THAT(pipeline0->operand(0)->parameter_number(), 0);
  EXPECT_THAT(pipeline0->operand(1)->parameter_number(), 1);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::Seed)(pipeline0->operand(2)));

  HloComputation* pipeline_comp = pipeline0->to_apply();
  TF_ASSERT_OK_AND_ASSIGN(PipelineStages stages,
                          GetPipelineStages(pipeline_comp));

  HloInstruction* s0 = stages.forward[0];
  EXPECT_THAT(s0->operand_count(), 4);
  EXPECT_THAT(s0->operand(0)->parameter_number(), 0);
  EXPECT_THAT(s0->operand(1)->parameter_number(), 1);
  // Seed added as parameter 2
  EXPECT_THAT(s0->operand(2)->parameter_number(), 2);
  // Exec counter for the pipeline.
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(s0->operand(3)));
  HloInstruction* s0_root = s0->to_apply()->root_instruction();
  HloInstruction* seed_hash;
  ASSERT_TRUE(Match(s0_root, m::Tuple(m::Add(m::Parameter(0), m::Parameter(1)),
                                      m::Op(&seed_hash))));
  // Seed was hashed with the execution counter of the stage.
  auto pair = MatchHashCombine(seed_hash);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(pair.second));

  // Seed was hashed with the execution counter of the pipeline.
  pair = MatchHashCombine(pair.first);
  EXPECT_THAT(pair.first->parameter_number(), 2);
  EXPECT_THAT(pair.second->parameter_number(), 3);
}

TEST_F(SeedHoistingTest, RepeatLoop) {
  std::string hlo = R"(
HloModule top

loop {
  p0 = s32[2] parameter(0)
  p1 = s32[2] parameter(1)
  s = s32[2] custom-call(), custom_call_target="Seed", backend_config=""
  add1 = s32[2] add(s, p0)
  add2 = s32[2] add(add1, p1)
  ROOT tuple = (s32[2], s32[2]) tuple(add2, p1)
}

ENTRY e {
  e.p0 = s32[2] parameter(0), parameter_replication={false}
  e.p1 = s32[2] parameter(1), parameter_replication={false}
  ROOT e.call = (s32[2], s32[2]) call(e.p0, e.p1), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest()));
  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(SeedHoisting().Run(module.get()).ValueOrDie());

  HloInstruction *loop0, *loop1;
  auto root = module->entry_computation()->root_instruction();
  ASSERT_TRUE(Match(root, m::Tuple(m::GetTupleElement(m::Op(&loop0), 0),
                                   m::GetTupleElement(m::Op(&loop1), 1))));
  EXPECT_THAT(loop0, loop1);
  EXPECT_TRUE(IsRepeatLoop(loop0));
  EXPECT_THAT(loop0->operand_count(), 3);
  EXPECT_THAT(loop0->operand(0)->parameter_number(), 0);
  EXPECT_THAT(loop0->operand(1)->parameter_number(), 1);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::Seed)(loop0->operand(2)));

  HloInstruction* loop_root = loop0->to_apply()->root_instruction();
  HloInstruction* seed_hash;
  ASSERT_TRUE(Match(loop_root,
                    m::Tuple(m::Add(m::Add(m::Op(&seed_hash), m::Parameter(0)),
                                    m::Parameter(1)),
                             m::Parameter(1), m::Parameter(2))));
  // Seed was hashed with the execution counter of the stage.
  auto pair = MatchHashCombine(seed_hash);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(pair.second));
  EXPECT_THAT(pair.first->parameter_number(), 2);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
