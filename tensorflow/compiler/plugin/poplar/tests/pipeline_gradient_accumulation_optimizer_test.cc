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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_gradient_accumulation_optimizer.h"

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/fifo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
namespace {

using PipelineGradientAccumulationOptimizerTest = HloTestBase;

TEST_F(PipelineGradientAccumulationOptimizerTest, TestLowerToANewStage) {
  std::string hlo = R"(
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
  ROOT stage_2_bwd_tuple = (f32[2]) tuple(stage_2_bwd_p0)
}

stage_1_bwd {
  stage_1_bwd_p0 = f32[2] parameter(0)
  stage_1_bwd_p1 = f32[2] parameter(1)
  ROOT stage_1_bwd_tuple = (f32[2], f32[2]) tuple(stage_1_bwd_p0, stage_1_bwd_p1)
}

stage_0_bwd {
  stage_0_bwd_p0 = f32[2] parameter(0)
  stage_0_bwd_p1 = f32[2] parameter(1)
  acc_scale = f32[] constant(1)
  stage_0_bwd_accumulator = f32[2] parameter(2)
  stage_0_bwd_add_grads = f32[2] add(stage_0_bwd_p0, stage_0_bwd_p1)
  stage_0_bwd_accumulator_update = f32[2] custom-call(stage_0_bwd_accumulator, stage_0_bwd_add_grads, acc_scale), custom_call_target="GradientAccumulatorAddWithScale"
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

  pipeline_stage_1 = (f32[2]) call(pipeline_stage_0.0), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1.0 = f32[2] get-tuple-element(pipeline_stage_1), index=0

  pipeline_stage_2 = (f32[2]) call(pipeline_stage_1.0), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  pipeline_stage_2.0 = f32[2] get-tuple-element(pipeline_stage_2), index=0

  pipeline_stage_2_bwd = (f32[2]) call(pipeline_stage_2.0), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
  pipeline_stage_2_bwd.0 = f32[2] get-tuple-element(pipeline_stage_2_bwd), index=0

  pipeline_stage_1_bwd = (f32[2], f32[2]) call(pipeline_stage_2_bwd.0, pipeline_stage_1.0), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  pipeline_stage_1_bwd.0 = f32[2] get-tuple-element(pipeline_stage_1_bwd), index=0
  pipeline_stage_1_bwd.1 = f32[2] get-tuple-element(pipeline_stage_1_bwd), index=1

  pipeline_accumulator = f32[2] custom-call(pipeline_p0), custom_call_target="GradientAccumulatorCreate", backend_config="{}"
  pipeline_stage_0_bwd = (f32[2]) call(pipeline_stage_1_bwd.0, pipeline_stage_0.0, pipeline_accumulator), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  pipeline_stage_0_bwd.0 = f32[2] get-tuple-element(pipeline_stage_0_bwd), index=0


  pipeline_accumulator_sink = f32[2] custom-call(pipeline_stage_0_bwd.0), custom_call_target="GradientAccumulatorSink", backend_config="{\"num_mini_batches\":1}\n"

  call_ru = (f32[2]) call(pipeline_accumulator_sink), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  pipeline_p0_updated = f32[2] get-tuple-element(call_ru), index=0
  ROOT pipeline_tuple = (f32[2]) tuple(pipeline_p0_updated)
}

ENTRY e {
  e.weights0 = f32[2] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[2]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();

  ASSERT_TRUE(CustomOpReplacer().Run(module0).ValueOrDie());

  PipelineGradientAccumulationOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, optimizer.Run(module.get()));
  // Expect changes.
  EXPECT_TRUE(changed);
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  EXPECT_TRUE(stages.resource_update);
  const HloInstruction* resource_update = *stages.resource_update;
  const HloInstruction* gradient_accumulator_sink = resource_update->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(
      gradient_accumulator_sink));
  EXPECT_EQ(gradient_accumulator_sink->operand_count(), 2);

  std::set<const HloInstruction*> accumulator_creators;
  {
    // Check that the stage 0 has an add to the accumulater, and where the
    // gradient previously was is now just a zero.
    const HloInstruction* grad_pipeline_stage_0 =
        gradient_accumulator_sink->operand(0);
    EXPECT_EQ(grad_pipeline_stage_0->opcode(), HloOpcode::kGetTupleElement);
    HloInstruction* bwd_stage_0 = stages.backward[0];
    EXPECT_EQ(grad_pipeline_stage_0->operand(0), bwd_stage_0);
    HloComputation* bwd_stage_0_comp = bwd_stage_0->to_apply();
    auto root = bwd_stage_0_comp->root_instruction();
    auto accumulator_add = root->operand(0);
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::GradientAccumulatorAddWithScale)(
        accumulator_add));
    auto lhs = accumulator_add->operand(0);
    EXPECT_EQ(lhs->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(lhs->parameter_number(), 2);
    accumulator_creators.insert(
        bwd_stage_0->operand(accumulator_add->operand(0)->parameter_number()));
    auto rhs = accumulator_add->operand(1);
    EXPECT_EQ(rhs->opcode(), HloOpcode::kAdd);
    EXPECT_TRUE(IsWideConstantZero(rhs->operand(0)));
    EXPECT_EQ(rhs->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(rhs->operand(1)->parameter_number(), 1);
  }

  {
    // Check that stage 2 has now a gradient accumulator.
    const HloInstruction* grad_pipeline_stage_2 =
        gradient_accumulator_sink->operand(1);
    EXPECT_EQ(grad_pipeline_stage_2->opcode(), HloOpcode::kGetTupleElement);
    HloInstruction* bwd_stage_2 = stages.backward[2];
    EXPECT_EQ(grad_pipeline_stage_2->operand(0), bwd_stage_2);
    HloComputation* bwd_stage_2_comp = bwd_stage_2->to_apply();
    auto root = bwd_stage_2_comp->root_instruction();
    auto accumulator_add = root->operand(1);
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::GradientAccumulatorAddWithScale)(
        accumulator_add));
    auto lhs = accumulator_add->operand(0);
    EXPECT_EQ(lhs->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(lhs->parameter_number(), 1);
    accumulator_creators.insert(
        bwd_stage_2->operand(accumulator_add->operand(0)->parameter_number()));
    auto rhs = accumulator_add->operand(1);
    EXPECT_EQ(rhs->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(rhs->parameter_number(), 0);
  }

  EXPECT_EQ(accumulator_creators.size(), 1);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(
      *accumulator_creators.begin()));
}

TEST_F(PipelineGradientAccumulationOptimizerTest, TestLowerToAnExistingStage) {
  std::string hlo = R"(
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
  acc_scale = f32[] constant(1)
  stage_2_bwd_p0 = f32[2] parameter(0)
  grad = f32[2] constant({2.9, 2.1})
  stage_2_bwd_accumulator = f32[2] parameter(1)
  stage_2_bwd_accumulator_update = f32[2] custom-call(stage_2_bwd_accumulator, grad, acc_scale), custom_call_target="GradientAccumulatorAddWithScale"
  ROOT stage_2_bwd_tuple = (f32[2], f32[2]) tuple(stage_2_bwd_p0, stage_2_bwd_accumulator_update)
}

stage_1_bwd {
  stage_1_bwd_p0 = f32[2] parameter(0)
  stage_1_bwd_p1 = f32[2] parameter(1)
  ROOT stage_1_bwd_tuple = (f32[2], f32[2]) tuple(stage_1_bwd_p0, stage_1_bwd_p1)
}

stage_0_bwd {
  acc_scale = f32[] constant(1)
  stage_0_bwd_p0 = f32[2] parameter(0)
  stage_0_bwd_p1 = f32[2] parameter(1)
  stage_0_bwd_accumulator = f32[2] parameter(2)
  stage_0_bwd_add_grads = f32[2] add(stage_0_bwd_p0, stage_0_bwd_p1)
  stage_0_bwd_accumulator_update = f32[2] custom-call(stage_0_bwd_accumulator, stage_0_bwd_add_grads, acc_scale), custom_call_target="GradientAccumulatorAddWithScale"
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

  pipeline_stage_1 = (f32[2]) call(pipeline_stage_0.0), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1.0 = f32[2] get-tuple-element(pipeline_stage_1), index=0

  pipeline_stage_2 = (f32[2]) call(pipeline_stage_1.0), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  pipeline_stage_2.0 = f32[2] get-tuple-element(pipeline_stage_2), index=0

  pipeline_accumulator = f32[2] custom-call(pipeline_p0), custom_call_target="GradientAccumulatorCreate", backend_config="{}"

  pipeline_stage_2_bwd = (f32[2], f32[2]) call(pipeline_stage_2.0, pipeline_accumulator), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
  pipeline_stage_2_bwd.0 = f32[2] get-tuple-element(pipeline_stage_2_bwd), index=0
  pipeline_stage_2_bwd.1 = f32[2] get-tuple-element(pipeline_stage_2_bwd), index=1

  pipeline_stage_1_bwd = (f32[2], f32[2]) call(pipeline_stage_2_bwd.0, pipeline_stage_1.0), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  pipeline_stage_1_bwd.0 = f32[2] get-tuple-element(pipeline_stage_1_bwd), index=0
  pipeline_stage_1_bwd.1 = f32[2] get-tuple-element(pipeline_stage_1_bwd), index=1

  pipeline_stage_0_bwd = (f32[2]) call(pipeline_stage_1_bwd.0, pipeline_stage_0.0, pipeline_accumulator), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  pipeline_stage_0_bwd.0 = f32[2] get-tuple-element(pipeline_stage_0_bwd), index=0


  pipeline_accumulator_sink = f32[2] custom-call(pipeline_stage_0_bwd.0, pipeline_stage_2_bwd.1), custom_call_target="GradientAccumulatorSink", backend_config="{\"num_mini_batches\":1}\n"

  call_ru = (f32[2]) call(pipeline_accumulator_sink), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  pipeline_p0_updated = f32[2] get-tuple-element(call_ru), index=0
  ROOT pipeline_tuple = (f32[2]) tuple(pipeline_p0_updated)
}

ENTRY e {
  e.weights0 = f32[2] parameter(0), parameter_replication={false}
  ROOT e.call = (f32[2]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  auto module0 = module.get();

  ASSERT_TRUE(CustomOpReplacer().Run(module0).ValueOrDie());

  PipelineGradientAccumulationOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, optimizer.Run(module.get()));
  // Expect changes.
  EXPECT_TRUE(changed);
  HloComputation* pipeline_computation = FindComputation(module0, "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  EXPECT_TRUE(stages.resource_update);
  const HloInstruction* resource_update = *stages.resource_update;
  const HloInstruction* gradient_accumulator_sink = resource_update->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(
      gradient_accumulator_sink));
  EXPECT_EQ(gradient_accumulator_sink->operand_count(), 2);

  std::set<const HloInstruction*> accumulator_creators;
  {
    // Check that the stage 0 has an add to the accumulater, and where the
    // gradient previously was is now just a zero.
    const HloInstruction* grad_pipeline_stage_0 =
        gradient_accumulator_sink->operand(0);
    EXPECT_EQ(grad_pipeline_stage_0->opcode(), HloOpcode::kGetTupleElement);
    HloInstruction* bwd_stage_0 = stages.backward[0];
    EXPECT_EQ(grad_pipeline_stage_0->operand(0), bwd_stage_0);
    HloComputation* bwd_stage_0_comp = bwd_stage_0->to_apply();
    auto root = bwd_stage_0_comp->root_instruction();
    auto accumulator_add = root->operand(0);
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::GradientAccumulatorAddWithScale)(
        accumulator_add));
    auto lhs = accumulator_add->operand(0);
    EXPECT_EQ(lhs->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(lhs->parameter_number(), 2);
    accumulator_creators.insert(
        bwd_stage_0->operand(accumulator_add->operand(0)->parameter_number()));
    auto rhs = accumulator_add->operand(1);
    EXPECT_EQ(rhs->opcode(), HloOpcode::kAdd);
    EXPECT_TRUE(IsWideConstantZero(rhs->operand(0)));
    EXPECT_EQ(rhs->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(rhs->operand(1)->parameter_number(), 1);
  }

  {
    // Check that stage 2 gradient accumulation has been extended with an extra
    // add.
    const HloInstruction* grad_pipeline_stage_2 =
        gradient_accumulator_sink->operand(1);
    EXPECT_EQ(grad_pipeline_stage_2->opcode(), HloOpcode::kGetTupleElement);
    HloInstruction* bwd_stage_2 = stages.backward[2];
    EXPECT_EQ(grad_pipeline_stage_2->operand(0), bwd_stage_2);
    HloComputation* bwd_stage_2_comp = bwd_stage_2->to_apply();
    auto root = bwd_stage_2_comp->root_instruction();
    auto accumulator_add = root->operand(1);
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::GradientAccumulatorAddWithScale)(
        accumulator_add));
    auto lhs = accumulator_add->operand(0);
    EXPECT_EQ(lhs->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(lhs->parameter_number(), 1);
    accumulator_creators.insert(
        bwd_stage_2->operand(accumulator_add->operand(0)->parameter_number()));
    auto rhs = accumulator_add->operand(1);
    // Check that an extra add was added.
    EXPECT_EQ(rhs->opcode(), HloOpcode::kAdd);
    EXPECT_EQ(rhs->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(rhs->operand(1)->parameter_number(), 0);
  }

  EXPECT_EQ(accumulator_creators.size(), 1);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(
      *accumulator_creators.begin()));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
