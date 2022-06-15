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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_tuple_remover.h"

#include "tensorflow/compiler/plugin/poplar/driver/passes/call_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using PipelineTupleRemoverTest = HloTestBase;

TEST_F(PipelineTupleRemoverTest, TestNoTuples) {
  std::string hlo = R"(
HloModule top

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
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0, arg1)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"

  stage_0_fwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_weights1 = f32[1,4,4,2] parameter(1)

  pipeline_stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_fwd, pipeline_weights1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  stage_1_fwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  stage_0_fwd_x = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1

  pipeline_stage_2 = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_fwd_x, stage_1_fwd), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
  stage_2_fwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_2), index=0
  stage_1_fwd_x = f32[1,4,4,2] get-tuple-element(pipeline_stage_2), index=1

  pipeline_stage_2_bwd = (f32[1,4,4,2]) call(stage_2_fwd, stage_1_fwd_x), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
  stage_2_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_2_bwd), index=0

  pipeline_stage_1_bwd = (f32[1,4,4,2]) call(stage_2_bwd, stage_1_fwd), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  stage_1_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd), index=0

  pipeline_stage_0_bwd = (f32[1,4,4,2]) call(stage_1_bwd, stage_0_fwd), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  stage_0_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_0_bwd), index=0

  call_ru = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  PipelineTupleRemover inserter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(PipelineTupleRemoverTest, TestNestedTuples) {
  std::string hlo = R"(
HloModule top

stage_0 {
  stage_0_t = token[] after-all()
  stage_0_feed = (f32[1,1,1], token[]) infeed(stage_0_t)
  stage_0_input = f32[1,1,1] get-tuple-element(stage_0_feed), index=0
  stage_0_weights0 = f32[1,2] parameter(0)
  stage_0_input_tuple = (f32[1,1,1]) tuple(stage_0_input)
  ROOT stage_0_tuple = ((f32[1,1,1]), f32[1,2]) tuple(stage_0_input_tuple, stage_0_weights0)
}

stage_1 {
  stage_1_p0 = (f32[1,1,1]) parameter(0)
  stage_1_p1 = f32[1,2] parameter(1)
  stage_1_p2 = f32[1024] parameter(2)

  stage_1_p1_tuple = (f32[1,2]) tuple(stage_1_p1)
  stage_1_p0p1_tuple = ((f32[1,1,1]), (f32[1,2])) tuple(stage_1_p0, stage_1_p1_tuple)

  stage_1_p2_tuple = (f32[1024]) tuple(stage_1_p2)
  
  ROOT stage_1_tuple = (((f32[1,1,1]), (f32[1,2])), (f32[1024])) tuple(stage_1_p0p1_tuple, stage_1_p2_tuple)
}

stage_2 {
  stage_2_p0 = ((f32[1,1,1]), (f32[1,2])) parameter(0)
  stage_2_p1 = (f32[1024]) parameter(1)
  stage_2_p2 = (f32[1,1,1]) parameter(2)

  stage_2_p0p1p2 = (((f32[1,1,1]), (f32[1,2])), (f32[1024]), (f32[1,1,1])) tuple(stage_2_p0, stage_2_p1, stage_2_p2)

  stage_2_t = token[] after-all()
  stage_2_outfeed = token[] outfeed(stage_2_p0p1p2, stage_2_t)

  ROOT tuple = () tuple()
}

pipeline {
  pipeline_weights0 = f32[1,2] parameter(0)
  pipeline_stage_0 = ((f32[1,1,1]), f32[1,2]) call(pipeline_weights0), to_apply=stage_0, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"

  s0_0 = (f32[1,1,1]) get-tuple-element(pipeline_stage_0), index=0
  s0_1 = f32[1,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_weights1 = f32[1024] parameter(1)

  pipeline_stage_1 = (((f32[1,1,1]), (f32[1,2])), (f32[1024])) call(s0_0, s0_1, pipeline_weights1), to_apply=stage_1, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  s1_0 = ((f32[1,1,1]), (f32[1,2])) get-tuple-element(pipeline_stage_1), index=0
  s1_1 = (f32[1024]) get-tuple-element(pipeline_stage_1), index=1
  
  pipeline_stage_2 = () call(s1_0, s1_1, s0_0), to_apply=stage_2, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"

  ROOT pipeline_tuple = (f32[1,2], f32[1024]) tuple(pipeline_weights0, pipeline_weights1)
}

ENTRY e {
  e.weights0 = f32[1,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1024] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,2], f32[1024]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  PipelineTupleRemover inserter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module.get()));
  EXPECT_TRUE(changed);

  // Run optimizations passes to remove all the unused operands etc.
  while (CallOptimizer().Run(module.get()).ValueOrDie() ||
         PipelineOptimizer().Run(module.get()).ValueOrDie() ||
         HloDCE().Run(module.get()).ValueOrDie() ||
         HloCSE(false).Run(module.get()).ValueOrDie() ||
         TupleSimplifier(true).Run(module.get()).ValueOrDie()) {
  }
  HloInstruction* pipeline = module->entry_computation()->root_instruction();

  TF_ASSERT_OK_AND_ASSIGN(auto pipeline_stages,
                          GetPipelineStages(pipeline->to_apply()));
  for (auto& stages : {pipeline_stages.forward, pipeline_stages.backward}) {
    for (auto& stage : stages) {
      for (auto p : stage->to_apply()->parameter_instructions()) {
        EXPECT_FALSE(p->shape().IsTuple());
      }
    }
  }

  EXPECT_TRUE(
      Match(pipeline_stages.forward[0]->to_apply()->root_instruction(),
            m::Tuple(m::Parameter(0), m::GetTupleElement(m::Infeed(), 0))));

  // Match inputs for stage 1.
  EXPECT_TRUE(
      Match(pipeline_stages.forward[1]->operand(0),
            m::GetTupleElement(m::Op().Is(pipeline_stages.forward[0]), 0)));
  EXPECT_TRUE(Match(pipeline_stages.forward[1]->operand(1), m::Parameter(1)));
  EXPECT_TRUE(
      Match(pipeline_stages.forward[1]->operand(2),
            m::GetTupleElement(m::Op().Is(pipeline_stages.forward[0]), 1)));
  // Stage 1 just passes the variables through.
  EXPECT_TRUE(
      Match(pipeline_stages.forward[1]->to_apply()->root_instruction(),
            m::Tuple(m::Parameter(2), m::Parameter(0), m::Parameter(1))));
  // Match inputs for stage 2.
  EXPECT_TRUE(
      Match(pipeline_stages.forward[2]->operand(0),
            m::GetTupleElement(m::Op().Is(pipeline_stages.forward[0]), 1)));
  EXPECT_TRUE(
      Match(pipeline_stages.forward[2]->operand(1),
            m::GetTupleElement(m::Op().Is(pipeline_stages.forward[1]), 1)));
  EXPECT_TRUE(
      Match(pipeline_stages.forward[2]->operand(2),
            m::GetTupleElement(m::Op().Is(pipeline_stages.forward[1]), 0)));
  EXPECT_TRUE(
      Match(pipeline_stages.forward[2]->operand(3),
            m::GetTupleElement(m::Op().Is(pipeline_stages.forward[1]), 2)));

  // Find the outfeed.
  auto insts =
      pipeline_stages.forward[2]->to_apply()->MakeInstructionPostOrder();
  auto itr = absl::c_find_if(insts, [](const HloInstruction* inst) {
    return inst->opcode() == HloOpcode::kOutfeed;
  });
  EXPECT_NE(itr, insts.end());
  EXPECT_TRUE(Match(
      *itr,
      m::Outfeed(m::Tuple(m::Tuple(m::Tuple(m::Parameter(2)),
                                   m::Tuple(m::Parameter(1))),
                          m::Tuple(m::Parameter(3)), m::Tuple(m::Parameter(0))),
                 m::AfterAll())));
}

TEST_F(PipelineTupleRemoverTest, TestSimplifyTuples) {
  std::string hlo = R"(
HloModule top

stage_0 {
  stage_0_t = token[] after-all()
  stage_0_feed = (f32[1,1,1], token[]) infeed(stage_0_t)
  stage_0_input = f32[1,1,1] get-tuple-element(stage_0_feed), index=0
  stage_0_weights0 = f32[1,1,1] parameter(0)
  stage_0_input_tuple = (f32[1,1,1]) tuple(stage_0_input)
  ROOT stage_0_tuple = ((f32[1,1,1]), f32[1,1,1]) tuple(stage_0_input_tuple, stage_0_weights0)
}

stage_1 {
  stage_1_p0 = (f32[1,1,1]) parameter(0)
  stage_1_p0_0 = f32[1,1,1] get-tuple-element(stage_1_p0), index=0
  stage_1_p1 = f32[1,1,1] parameter(1)
  add = f32[1,1,1] add(stage_1_p0_0, stage_1_p1)
  ROOT stage_1_tuple = (f32[1,1,1]) tuple(add)
}

pipeline {
  pipeline_weights0 = f32[1,1,1] parameter(0)
  pipeline_stage_0 = ((f32[1,1,1]), f32[1,1,1]) call(pipeline_weights0), to_apply=stage_0, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"

  s0_0 = (f32[1,1,1]) get-tuple-element(pipeline_stage_0), index=0
  s0_1 = f32[1,1,1] get-tuple-element(pipeline_stage_0), index=1

  ROOT pipeline_stage_1 = (f32[1,1,1]) call(s0_0, s0_1), to_apply=stage_1, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
}

ENTRY e {
  e.weights0 = f32[1,1,1] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[1,1,1]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  PipelineTupleRemover inserter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module.get()));
  EXPECT_TRUE(changed);

  // Run optimizations passes to remove all the unused operands etc.
  while (CallOptimizer().Run(module.get()).ValueOrDie() ||
         PipelineOptimizer().Run(module.get()).ValueOrDie() ||
         HloDCE().Run(module.get()).ValueOrDie() ||
         HloCSE(false).Run(module.get()).ValueOrDie()) {
  }
  HloInstruction* pipeline = module->entry_computation()->root_instruction();

  TF_ASSERT_OK_AND_ASSIGN(auto pipeline_stages,
                          GetPipelineStages(pipeline->to_apply()));
  for (auto& stages : {pipeline_stages.forward, pipeline_stages.backward}) {
    for (auto& stage : stages) {
      for (auto p : stage->to_apply()->parameter_instructions()) {
        EXPECT_FALSE(p->shape().IsTuple());
      }
    }
  }

  EXPECT_TRUE(
      Match(pipeline_stages.forward[0]->to_apply()->root_instruction(),
            m::Tuple(m::Parameter(0), m::GetTupleElement(m::Infeed(), 0))));

  // Match inputs for stage 1.
  EXPECT_TRUE(
      Match(pipeline_stages.forward[1]->operand(0),
            m::GetTupleElement(m::Op().Is(pipeline_stages.forward[0]), 0)));
  EXPECT_TRUE(
      Match(pipeline_stages.forward[1]->operand(1),
            m::GetTupleElement(m::Op().Is(pipeline_stages.forward[0]), 1)));

  EXPECT_TRUE(Match(pipeline_stages.forward[1]->to_apply()->root_instruction(),
                    m::Tuple(m::Add(m::Parameter(1), m::Parameter(0)))));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
