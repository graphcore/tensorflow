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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_batch_serialization_buffer_inserter.h"

#include "tensorflow/compiler/plugin/poplar/driver/passes/call_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_optimizer.h"
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

using PipelineBatchSerializationBufferInserterTest = HloTestBase;

std::string GetHlo(ThreeState offload_variables) {
  constexpr absl::string_view hlo_format = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_p0 = f32[2] parameter(0)
  l = f32[2] log(stage_0_fwd_p0)
  ROOT stage_0_fwd_tuple = (f32[2]) tuple(l)
}

stage_1_fwd {
  stage_1_fwd_p0 = f32[2] parameter(0)
  l = f32[2] log(stage_1_fwd_p0)
  ROOT stage_1_fwd_tuple = (f32[2]) tuple(l)
}

stage_2_fwd {
  stage_2_fwd_p0 = f32[2] parameter(0)
  l = f32[2] log(stage_2_fwd_p0)
  ROOT stage_2_fwd_tuple = (f32[2]) tuple(l)
}

stage_2_bwd {
  stage_2_bwd_p0 = f32[2] parameter(0)
  stage_2_bwd_p1 = f32[2] parameter(1)
  add = f32[2] add(stage_2_bwd_p0, stage_2_bwd_p1)
  ROOT stage_2_bwd_tuple = (f32[2]) tuple(add)
}

stage_1_bwd {
  stage_1_bwd_p0 = f32[2] parameter(0)
  stage_1_bwd_p1 = f32[2] parameter(1)
  ROOT stage_1_bwd_tuple = (f32[2], f32[2]) tuple(stage_1_bwd_p0, stage_1_bwd_p1)
}

stage_0_bwd {
  stage_0_bwd_p0 = f32[2] parameter(0)
  stage_0_bwd_p1 = f32[2] parameter(1)
  stage_0_bwd_accumulator = f32[2] parameter(2)
  stage_0_bwd_add_grads = f32[2] add(stage_0_bwd_p0, stage_0_bwd_p1)
  stage_0_bwd_accumulator_update = f32[2] custom-call(stage_0_bwd_accumulator, stage_0_bwd_add_grads), custom_call_target="GradientAccumulatorAdd"
  ROOT stage_0_bwd_tuple = (f32[2]) tuple(stage_0_bwd_accumulator_update)
}

resource_update {
  resource_update_p0 = f32[2] parameter(0)
  ROOT t = (f32[2]) tuple(resource_update_p0)
}

pipeline {
  pipeline_p0 = f32[2] parameter(0), sharding={maximal device=0}
  pipeline_stage_0 = (f32[2]) call(pipeline_p0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0.0 = f32[2] get-tuple-element(pipeline_stage_0), index=0

  pipeline_stage_1 = (f32[2]) call(pipeline_stage_0.0), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  pipeline_stage_1.0 = f32[2] get-tuple-element(pipeline_stage_1), index=0

  pipeline_stage_2 = (f32[2]) call(pipeline_stage_1.0), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  pipeline_stage_2.0 = f32[2] get-tuple-element(pipeline_stage_2), index=0

  pipeline_stage_2_bwd = (f32[2]) call(pipeline_stage_2.0, pipeline_stage_1.0), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
  pipeline_stage_2_bwd.0 = f32[2] get-tuple-element(pipeline_stage_2_bwd), index=0

  pipeline_stage_1_bwd = (f32[2], f32[2]) call(pipeline_stage_2_bwd.0, pipeline_stage_1.0), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  pipeline_stage_1_bwd.0 = f32[2] get-tuple-element(pipeline_stage_1_bwd), index=0
  pipeline_stage_1_bwd.1 = f32[2] get-tuple-element(pipeline_stage_1_bwd), index=1

  pipeline_accumulator = f32[2] custom-call(pipeline_p0), custom_call_target="GradientAccumulatorCreate", backend_config="{}"
  pipeline_stage_0_bwd = (f32[2]) call(pipeline_stage_1_bwd.0, pipeline_stage_0.0, pipeline_accumulator), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  pipeline_stage_0_bwd.0 = f32[2] get-tuple-element(pipeline_stage_0_bwd), index=0


  pipeline_accumulator_sink = f32[2] custom-call(pipeline_stage_0_bwd.0), custom_call_target="GradientAccumulatorSink", backend_config="{\"num_mini_batches\":1}\n"

  call_ru = (f32[2]) call(pipeline_accumulator_sink), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  pipeline_p0_updated = f32[2] get-tuple-element(call_ru), index=0
  ROOT pipeline_tuple = (f32[2]) tuple(pipeline_p0_updated)
}

ENTRY e {
  e.weights0 = f32[2] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[2]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":2,\"batch_serialization_iterations\":5,\"offload_activations\":\"%s\"}}}"
}
)";
  return absl::StrFormat(hlo_format, ThreeState_Name(offload_variables));
}

TEST_F(PipelineBatchSerializationBufferInserterTest, TestInMemory) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           GetHlo(THREESTATE_OFF), config));
  auto module0 = module.get();

  ASSERT_TRUE(CustomOpReplacer().Run(module0).ValueOrDie());

  PipelineBatchSerializationBufferInserter inserter(false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module0));
  EXPECT_TRUE(changed);

  // Remove all the unused outputs/inputs in pipeline stages.
  while (CallOptimizer().Run(module0).ValueOrDie() ||
         PipelineOptimizer().Run(module0).ValueOrDie() ||
         HloDCE().Run(module0).ValueOrDie() ||
         HloCSE(true).Run(module0).ValueOrDie()) {
  }
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));

  {
    auto stage = stages.forward[0];
    EXPECT_THAT(stage->operand(0)->opcode(), HloOpcode::kParameter);
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::CreateBuffer)(stage->operand(1)));
    HloInstruction* counter;
    EXPECT_TRUE(Match(stage->to_apply()->root_instruction(),
                      m::Tuple(m::DynamicUpdateSlice(
                          m::Parameter(1), m::Reshape(m::Log(m::Parameter(0))),
                          m::Op(&counter), m::ConstantScalar(0)))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter));
  }
  {
    auto stage = stages.forward[1];
    EXPECT_TRUE(Match(stage->operand(0),
                      m::GetTupleElement(m::Op().Is(stages.forward[0]), 0)));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::CreateBuffer)(stage->operand(1)));
    HloInstruction *counter1, *counter2;
    EXPECT_TRUE(Match(
        stage->to_apply()->root_instruction(),
        m::Tuple(m::DynamicUpdateSlice(
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
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::CreateBuffer)(stage->operand(1)));
    HloInstruction *counter1, *counter2;
    EXPECT_TRUE(Match(
        stage->to_apply()->root_instruction(),
        m::Tuple(m::DynamicUpdateSlice(
            m::Parameter(1),
            m::Reshape(m::Log(m::Reshape(m::DynamicSlice(
                m::Parameter(0), m::Op(&counter1), m::ConstantScalar(0))))),
            m::Op(&counter2), m::ConstantScalar(0)))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter1));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter2));
  }
  {
    auto stage = stages.backward[2];
    EXPECT_TRUE(Match(stage->operand(0),
                      m::GetTupleElement(m::Op().Is(stages.forward[1]), 0)));
    EXPECT_TRUE(Match(stage->operand(1),
                      m::GetTupleElement(m::Op().Is(stages.forward[2]), 0)));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::CreateBuffer)(stage->operand(2)));
    HloInstruction *counter1, *counter2, *counter3;
    EXPECT_TRUE(Match(
        stage->to_apply()->root_instruction(),
        m::Tuple(m::DynamicUpdateSlice(
            m::Parameter(2),
            m::Reshape(m::Add(
                m::Reshape(m::DynamicSlice(m::Parameter(1), m::Op(&counter1),
                                           m::ConstantScalar(0))),
                m::Reshape(m::DynamicSlice(m::Parameter(0), m::Op(&counter2),
                                           m::ConstantScalar(0))))),
            m::Op(&counter3), m::ConstantScalar(0)))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter1));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter2));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter3));
  }
  {
    auto stage = stages.backward[1];
    EXPECT_TRUE(Match(stage->operand(0),
                      m::GetTupleElement(m::Op().Is(stages.backward[2]), 0)));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::CreateBuffer)(stage->operand(1)));
    HloInstruction *counter1, *counter2;
    EXPECT_TRUE(Match(
        stage->to_apply()->root_instruction(),
        m::Tuple(m::DynamicUpdateSlice(
            m::Parameter(1),
            m::Reshape(m::Reshape(m::DynamicSlice(
                m::Parameter(0), m::Op(&counter1), m::ConstantScalar(0)))),
            m::Op(&counter2), m::ConstantScalar(0)))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter1));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter2));
  }
  {
    auto stage = stages.backward[0];
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(
        stage->operand(0)));
    EXPECT_TRUE(Match(stage->operand(1),
                      m::GetTupleElement(m::Op().Is(stages.forward[0]), 0)));
    EXPECT_TRUE(Match(stage->operand(2),
                      m::GetTupleElement(m::Op().Is(stages.backward[1]), 0)));
    HloInstruction* accumulator_add;
    EXPECT_TRUE(Match(stage->to_apply()->root_instruction(),
                      m::Tuple(m::Op(&accumulator_add))));
    EXPECT_TRUE(
        IsPoplarInstruction(PoplarOp::GradientAccumulatorAdd)(accumulator_add));
    EXPECT_TRUE(Match(accumulator_add->operand(0), m::Parameter(0)));
    HloInstruction *counter1, *counter2;
    EXPECT_TRUE(Match(
        accumulator_add->mutable_operand(1),
        m::Add(m::Reshape(m::DynamicSlice(m::Parameter(2), m::Op(&counter1),
                                          m::ConstantScalar(0))),
               m::Reshape(m::DynamicSlice(m::Parameter(1), m::Op(&counter2),
                                          m::ConstantScalar(0))))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter1));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter2));
  }

  const HloInstruction* resource_update = *stages.resource_update;
  const HloInstruction* gradient_accumulator_sink = resource_update->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(
      gradient_accumulator_sink));
  EXPECT_EQ(gradient_accumulator_sink->operand_count(), 1);
  // Make sure the sink shape doesn't change.
  EXPECT_THAT(gradient_accumulator_sink->shape().dimensions(),
              ::testing::ElementsAre(2));
}

TEST_F(PipelineBatchSerializationBufferInserterTest, TestOffloaded) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(GetHlo(THREESTATE_ON), config));
  auto module0 = module.get();

  ASSERT_TRUE(CustomOpReplacer().Run(module0).ValueOrDie());

  PipelineBatchSerializationBufferInserter inserter(true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module0));
  EXPECT_TRUE(changed);

  // Remove all the unused outputs/inputs in pipeline stages.
  while (CallOptimizer().Run(module0).ValueOrDie() ||
         PipelineOptimizer().Run(module0).ValueOrDie() ||
         HloDCE().Run(module0).ValueOrDie() ||
         HloCSE(true).Run(module0).ValueOrDie()) {
  }
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));

  {
    auto stage = stages.forward[0];
    EXPECT_THAT(stage->operand(0)->opcode(), HloOpcode::kParameter);
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::CreateBuffer)(stage->operand(1)));
    HloInstruction* store;
    EXPECT_TRUE(
        Match(stage->to_apply()->root_instruction(), m::Tuple(m::Op(&store))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::BufferStoreSlice)(store));
    HloInstruction* counter;
    EXPECT_TRUE(Match(store, m::CustomCall(m::Parameter(1),
                                           m::Reshape(m::Log(m::Parameter(0))),
                                           m::Op(&counter))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter));
  }
  {
    auto stage = stages.forward[1];
    EXPECT_TRUE(Match(stage->operand(0),
                      m::GetTupleElement(m::Op().Is(stages.forward[0]), 0)));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::CreateBuffer)(stage->operand(1)));
    HloInstruction *store, *load;
    EXPECT_TRUE(
        Match(stage->to_apply()->root_instruction(), m::Tuple(m::Op(&store))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::BufferStoreSlice)(store));
    HloInstruction* counter;
    EXPECT_TRUE(
        Match(store, m::CustomCall(m::Parameter(1),
                                   m::Reshape(m::Log(m::Reshape(m::Op(&load)))),
                                   m::Op(&counter))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::BufferLoadSlice)(load));
    EXPECT_TRUE(Match(load, m::CustomCall(m::Parameter(0), m::Op(&counter))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter));
  }
  {
    auto stage = stages.forward[2];
    EXPECT_TRUE(Match(stage->operand(0),
                      m::GetTupleElement(m::Op().Is(stages.forward[1]), 0)));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::CreateBuffer)(stage->operand(1)));
    HloInstruction *store, *load;
    EXPECT_TRUE(
        Match(stage->to_apply()->root_instruction(), m::Tuple(m::Op(&store))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::BufferStoreSlice)(store));
    HloInstruction* counter;
    EXPECT_TRUE(
        Match(store, m::CustomCall(m::Parameter(1),
                                   m::Reshape(m::Log(m::Reshape(m::Op(&load)))),
                                   m::Op(&counter))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::BufferLoadSlice)(load));
    EXPECT_TRUE(Match(load, m::CustomCall(m::Parameter(0), m::Op(&counter))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter));
  }
  {
    auto stage = stages.backward[2];
    EXPECT_TRUE(Match(stage->operand(0),
                      m::GetTupleElement(m::Op().Is(stages.forward[1]), 0)));
    EXPECT_TRUE(Match(stage->operand(1),
                      m::GetTupleElement(m::Op().Is(stages.forward[2]), 0)));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::CreateBuffer)(stage->operand(2)));
    HloInstruction *store, *load0, *load1;
    EXPECT_TRUE(
        Match(stage->to_apply()->root_instruction(), m::Tuple(m::Op(&store))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::BufferStoreSlice)(store));
    HloInstruction* counter;
    EXPECT_TRUE(Match(
        store, m::CustomCall(m::Parameter(2),
                             m::Reshape(m::Add(m::Reshape(m::Op(&load0)),
                                               m::Reshape(m::Op(&load1)))),
                             m::Op(&counter))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::BufferLoadSlice)(load0));
    EXPECT_TRUE(Match(load0, m::CustomCall(m::Parameter(1), m::Op(&counter))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::BufferLoadSlice)(load1));
    EXPECT_TRUE(Match(load1, m::CustomCall(m::Parameter(0), m::Op(&counter))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter));
  }
  {
    auto stage = stages.backward[1];
    EXPECT_TRUE(Match(stage->operand(0),
                      m::GetTupleElement(m::Op().Is(stages.backward[2]), 0)));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::CreateBuffer)(stage->operand(1)));
    HloInstruction *store, *load;
    EXPECT_TRUE(
        Match(stage->to_apply()->root_instruction(), m::Tuple(m::Op(&store))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::BufferStoreSlice)(store));
    HloInstruction* counter;
    EXPECT_TRUE(Match(store, m::CustomCall(m::Parameter(1),
                                           m::Reshape(m::Reshape(m::Op(&load))),
                                           m::Op(&counter))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::BufferLoadSlice)(load));
    EXPECT_TRUE(Match(load, m::CustomCall(m::Parameter(0), m::Op(&counter))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter));
  }
  {
    auto stage = stages.backward[0];
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(
        stage->operand(0)));
    EXPECT_TRUE(Match(stage->operand(1),
                      m::GetTupleElement(m::Op().Is(stages.forward[0]), 0)));
    EXPECT_TRUE(Match(stage->operand(2),
                      m::GetTupleElement(m::Op().Is(stages.backward[1]), 0)));
    HloInstruction* accumulator_add;
    EXPECT_TRUE(Match(stage->to_apply()->root_instruction(),
                      m::Tuple(m::Op(&accumulator_add))));
    EXPECT_TRUE(
        IsPoplarInstruction(PoplarOp::GradientAccumulatorAdd)(accumulator_add));
    HloInstruction *counter, *load0, *load1;
    EXPECT_TRUE(Match(
        accumulator_add,
        m::CustomCall(m::Parameter(0), m::Add(m::Reshape(m::Op(&load0)),
                                              m::Reshape(m::Op(&load1))))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::BufferLoadSlice)(load0));
    EXPECT_TRUE(Match(load0, m::CustomCall(m::Parameter(2), m::Op(&counter))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::BufferLoadSlice)(load1));
    EXPECT_TRUE(Match(load1, m::CustomCall(m::Parameter(1), m::Op(&counter))));
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ExecutionCounter)(counter));
  }

  const HloInstruction* resource_update = *stages.resource_update;
  const HloInstruction* gradient_accumulator_sink = resource_update->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(
      gradient_accumulator_sink));
  EXPECT_EQ(gradient_accumulator_sink->operand_count(), 1);
  // Make sure the sink shape doesn't change.
  EXPECT_THAT(gradient_accumulator_sink->shape().dimensions(),
              ::testing::ElementsAre(2));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
