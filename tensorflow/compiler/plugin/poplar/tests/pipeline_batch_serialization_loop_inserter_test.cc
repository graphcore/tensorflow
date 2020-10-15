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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_batch_serialization_loop_inserter.h"

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

HloInstruction* FindInstructionPred(
    HloComputation* comp,
    const std::function<bool(HloInstruction*)>& predicate) {
  return *std::find_if(comp->instructions().begin(), comp->instructions().end(),
                       predicate);
}

using PipelineBatchSerializationLoopInserterTest = HloTestBase;

TEST_F(PipelineBatchSerializationLoopInserterTest, Test) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  create-buffer.1 = f32[5,2] parameter(1)
  stage_0_fwd_p0.1 = f32[2] parameter(0)
  log.1 = f32[2] log(f32[2] stage_0_fwd_p0.1)
  reshape.5 = f32[1,2] reshape(f32[2] log.1)
  execution-counter = s32[] custom-call(), custom_call_target="ExecutionCounter", backend_config="{}"
  constant.5 = s32[] constant(0)
  dynamic-update-slice.3 = f32[5,2] dynamic-update-slice(f32[5,2] create-buffer.1, f32[1,2] reshape.5, s32[] execution-counter, s32[] constant.5)
  ROOT tuple.46 = (f32[5,2]) tuple(f32[5,2] dynamic-update-slice.3), sharding={{maximal device=0}}
}

stage_1_fwd {
  create-buffer.13 = f32[5,2] parameter(1)
  dynamic-update-slice.37 = f32[5,2] parameter(0)
  execution-counter = s32[] custom-call(), custom_call_target="ExecutionCounter", backend_config="{}"
  constant.45 = s32[] constant(0)
  dynamic-slice.32 = f32[1,2] dynamic-slice(f32[5,2] dynamic-update-slice.37, s32[] execution-counter, s32[] constant.45), dynamic_slice_sizes={1,2}
  reshape.45 = f32[2] reshape(f32[1,2] dynamic-slice.32)
  log.6 = f32[2] log(f32[2] reshape.45)
  reshape.46 = f32[1,2] reshape(f32[2] log.6)
  dynamic-update-slice.38 = f32[5,2] dynamic-update-slice(f32[5,2] create-buffer.13, f32[1,2] reshape.46, s32[] execution-counter, s32[] constant.45)
  ROOT tuple.45 = (f32[5,2]) tuple(f32[5,2] dynamic-update-slice.38), sharding={{maximal device=0}}
}

stage_2_fwd {
  dynamic-update-slice.35 = f32[5,2] parameter(0)
  execution-counter = s32[] custom-call(), custom_call_target="ExecutionCounter", backend_config="{}"
  constant.43 = s32[] constant(0)
  dynamic-slice.31 = f32[1,2] dynamic-slice(f32[5,2] dynamic-update-slice.35, s32[] execution-counter, s32[] constant.43), dynamic_slice_sizes={1,2}
  reshape.43 = f32[2] reshape(f32[1,2] dynamic-slice.31)
  log = f32[2] log(f32[2] reshape.43)
  after-all = token[] after-all()
  outfeed = token[] outfeed(log, after-all), outfeed_config="g"
  ROOT tuple.43 = () tuple()
}

pipeline {
  pipeline_p0 = f32[2] parameter(0), sharding={maximal device=0}
  
  create-buffer = f32[5,2] custom-call(), custom_call_target="CreateBuffer", backend_config="{\"is_remote\":0}\n"
  call.2 = (f32[5,2]) call(f32[2] pipeline_p0, f32[5,2] create-buffer), to_apply=stage_0_fwd, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  get-tuple-element.1 = f32[5,2] get-tuple-element((f32[5,2]) call.2), index=0
  
  create-buffer.2 = f32[5,2] custom-call(), custom_call_target="CreateBuffer", backend_config="{\"is_remote\":0}\n"
  call.17 = (f32[5,2]) call(f32[5,2] get-tuple-element.1, f32[5,2] create-buffer.2), to_apply=stage_1_fwd, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  get-tuple-element.3 = f32[5,2] get-tuple-element((f32[5,2]) call.17), index=0
  
  call.16 = () call(f32[5,2] get-tuple-element.3), to_apply=stage_2_fwd, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
  
  ROOT pipeline_tuple = (f32[2]) tuple(f32[2] pipeline_p0)
}

ENTRY e (e.weights0: f32[2]) -> (f32[2]) {
  e.weights0 = f32[2] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[2]) call(f32[2] e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0,\"batch_serialization_iterations\":5}}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();

  ASSERT_TRUE(CustomOpReplacer().Run(module0).ValueOrDie());

  PipelineBatchSerializationLoopInserter inserter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module0));
  EXPECT_TRUE(changed);

  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));

  {
    auto stage = stages.forward[0];
    EXPECT_THAT(stage->operand(0)->opcode(), HloOpcode::kParameter);
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::CreateBuffer)(stage->operand(1)));
    HloInstruction* loop;
    EXPECT_TRUE(Match(stage->to_apply()->root_instruction(),
                      m::Tuple(m::GetTupleElement(m::Op(&loop), 1))));
    EXPECT_TRUE(IsRepeatLoop(loop));
    EXPECT_EQ(GetRepeatLoopCount(loop), 5);
    EXPECT_TRUE(Match(loop->operand(0), m::Parameter(0)));
    EXPECT_TRUE(Match(loop->operand(1), m::Parameter(1)));
    HloInstruction* counter;
    EXPECT_TRUE(
        Match(loop->to_apply()->root_instruction(),
              m::Tuple(m::Parameter(0),
                       m::DynamicUpdateSlice(
                           m::Parameter(1), m::Reshape(m::Log(m::Parameter(0))),
                           m::Op(&counter), m::ConstantScalar(0)))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter));
  }
  {
    auto stage = stages.forward[1];
    EXPECT_TRUE(Match(stage->operand(0),
                      m::GetTupleElement(m::Op().Is(stages.forward[0]), 0)));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::CreateBuffer)(stage->operand(1)));
    HloInstruction* loop;
    EXPECT_TRUE(Match(stage->to_apply()->root_instruction(),
                      m::Tuple(m::GetTupleElement(m::Op(&loop), 1))));
    EXPECT_TRUE(IsRepeatLoop(loop));
    EXPECT_EQ(GetRepeatLoopCount(loop), 5);
    EXPECT_TRUE(Match(loop->operand(0), m::Parameter(0)));
    EXPECT_TRUE(Match(loop->operand(1), m::Parameter(1)));
    HloInstruction *counter1, *counter2;
    EXPECT_TRUE(Match(
        loop->to_apply()->root_instruction(),
        m::Tuple(
            m::Parameter(0),
            m::DynamicUpdateSlice(
                m::Parameter(1),
                m::Reshape(m::Log(m::Reshape(m::DynamicSlice(
                    m::Parameter(0), m::Op(&counter1), m::ConstantScalar(0))))),
                m::Op(&counter2), m::ConstantScalar(0)))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter1));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter2));
  }
  {
    auto stage = stages.forward[2];
    EXPECT_TRUE(Match(stage->operand(0),
                      m::GetTupleElement(m::Op().Is(stages.forward[1]), 0)));
    EXPECT_TRUE(Match(stage->to_apply()->root_instruction(), m::Tuple()));
    HloInstruction* loop = FindInstructionPred(
        stage->to_apply(),
        [](const HloInstruction* inst) { return IsRepeatLoop(inst); });
    EXPECT_EQ(GetRepeatLoopCount(loop), 5);
    EXPECT_TRUE(Match(loop->operand(0), m::Parameter(0)));

    HloInstruction* outfeed =
        FindInstructionPred(loop->to_apply(), [](const HloInstruction* inst) {
          return inst->opcode() == HloOpcode::kOutfeed;
        });

    HloInstruction* counter;
    EXPECT_TRUE(Match(outfeed, m::Outfeed(m::Log(m::Reshape(m::DynamicSlice(
                                              m::Parameter(0), m::Op(&counter),
                                              m::ConstantScalar(0)))),
                                          m::AfterAll())));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter));
  }
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
