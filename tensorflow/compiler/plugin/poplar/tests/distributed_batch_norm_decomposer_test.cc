/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/distributed_batch_norm_decomposer.h"

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/recomputation_checkpoint_remover.h"
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

using DistributedBatchNormDecomposerTest = HloTestBase;

std::string GetRecomputationHlo() {
  return R"(
HloModule top

stage_0_fwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,2] parameter(1)
  after-all = token[] after-all()

  infeed = ((f32[1,4,4,2], f32[2], f32[2]), token[]) infeed(after-all), infeed_config="140121807314577"
  infeed_tuple = (f32[1,4,4,2], f32[2], f32[2]) get-tuple-element(infeed), index=0
  activations = f32[1,4,4,2] get-tuple-element(infeed_tuple), index=0
  scale = f32[2] get-tuple-element(infeed_tuple), index=1
  offset = f32[2] get-tuple-element(infeed_tuple), index=2
  batch-norm-training = (f32[1,4,4,2], f32[2], f32[2]) batch-norm-training(activations, scale, offset), epsilon=0.001, feature_index=3
  normalised = f32[1,4,4,2] get-tuple-element(batch-norm-training), index=0
  in12 = f32[1,4,4,2] add(normalised, in1)
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

TEST_F(DistributedBatchNormDecomposerTest, TestDecompose) {
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(GetRecomputationHlo(), config));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, DistributedBatchNormDecomposer(/*allow_recomputation=*/true,
                                                   /*replica_group_size=*/2)
                        .Run(module.get()));
  EXPECT_TRUE(changed);

  // Check the decomposition and that recomputation checkpoints have been
  // inserted.
  HloInstruction* in12 = FindInstruction(module.get(), "in12");
  HloInstruction *batch_norm_inference, *mean_recomp, *var_recomp;
  EXPECT_TRUE(
      Match(in12, m::Add(m::GetTupleElement(
                             m::Tuple(m::Op(&batch_norm_inference),
                                      m::Op(&mean_recomp), m::Op(&var_recomp)),
                             0),
                         m::Parameter(1))));
  EXPECT_EQ(batch_norm_inference->opcode(), HloOpcode::kBatchNormInference);
  HloInstruction* infeed;
  EXPECT_TRUE(
      Match(batch_norm_inference->mutable_operand(0),
            m::GetTupleElement(m::GetTupleElement(m::Op(&infeed), 0), 0)));
  EXPECT_TRUE(
      Match(batch_norm_inference->operand(1),
            m::GetTupleElement(m::GetTupleElement(m::Op().Is(infeed), 0), 1)));
  EXPECT_TRUE(
      Match(batch_norm_inference->operand(2),
            m::GetTupleElement(m::GetTupleElement(m::Op().Is(infeed), 0), 2)));
  EXPECT_EQ(batch_norm_inference->operand(3), mean_recomp);
  EXPECT_EQ(batch_norm_inference->operand(4), var_recomp);

  // Check mean and variance come from the stats through the checkpoints.
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::RecomputationCheckpoint, mean_recomp));
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::RecomputationCheckpoint, var_recomp));
  HloInstruction* batch_norm_stats;
  EXPECT_TRUE(Match(mean_recomp->mutable_operand(0),
                    m::GetTupleElement(m::Op(&batch_norm_stats), 0)));
  EXPECT_TRUE(Match(var_recomp->operand(0),
                    m::GetTupleElement(m::Op().Is(batch_norm_stats), 1)));

  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::BatchNormStatistics, batch_norm_stats));
  EXPECT_TRUE(
      Match(batch_norm_stats->operand(0),
            m::GetTupleElement(m::GetTupleElement(m::Op().Is(infeed), 0), 0)));
}

TEST_F(DistributedBatchNormDecomposerTest, TestNotDecompose) {
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(GetRecomputationHlo(), config));

  HloComputation* pipeline_comp = FindComputation(module.get(), "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, DistributedBatchNormDecomposer(/*allow_recomputation=*/true,
                                                   /*replica_group_size=*/1)
                        .Run(module.get()));
  EXPECT_FALSE(changed);
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
