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
#include "tensorflow/compiler/plugin/poplar/driver/passes/gradient_accumulation_buffers_offload.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
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

using GradientAccumulationBuffersOffloadTest = HloTestBase;

std::string GetHlo(int64_t schedule, ThreeState offload_accumulators) {
  constexpr absl::string_view hlo_format = R"(
HloModule top

_pop_op_serialized_gradient_accumulation_1 {
  p0 = f32[128] parameter(0)
  p1 = f16[128] parameter(1)
  p2 = f16[128] parameter(2)
  p1_convert = f32[128] convert(p1)
  a0 = f32[128] add(p0, p1_convert)
  p2_convert = f32[128] convert(p2)
  ROOT a1 = f32[128] add(a0, p2_convert)
}

stage_0_fwd {
  stage_0_fwd_p0 = f16[128] parameter(0)
  l = f16[128] log(stage_0_fwd_p0)
  ROOT stage_0_fwd_tuple = (f16[128]) tuple(l)
}

stage_1_fwd {
  stage_1_fwd_p0 = f16[128] parameter(0)
  l = f16[128] log(stage_1_fwd_p0)
  ROOT stage_1_fwd_tuple = (f16[128]) tuple(l)
}

stage_2_fwd {
  stage_2_fwd_p0 = f16[128] parameter(0)
  l = f16[128] log(stage_2_fwd_p0)
  ROOT stage_2_fwd_tuple = (f16[128]) tuple(l)
}

stage_2_bwd {
  stage_2_bwd_p0 = f16[128] parameter(0)
  stage_2_bwd_p1 = f16[128] parameter(1)
  add = f16[128] add(stage_2_bwd_p0, stage_2_bwd_p1)
  ROOT stage_2_bwd_tuple = (f16[128]) tuple(add)
}

stage_1_bwd {
  stage_1_bwd_p0 = f16[128] parameter(0)
  stage_1_bwd_p1 = f16[128] parameter(1)
  stage_1_bwd_accumulator = f32[128] parameter(2)
  ROOT stage_1_bwd_tuple = (f16[128], f16[128], f32[128]) tuple(stage_1_bwd_p0, stage_1_bwd_p1, stage_1_bwd_accumulator)
}

stage_0_bwd {
  stage_0_bwd_p0 = f16[128] parameter(0)
  stage_0_bwd_p1 = f16[128] parameter(1)
  stage_0_bwd_accumulator = f32[128] parameter(2)
  stage_0_bwd_accumulator_update = f32[128] fusion(stage_0_bwd_accumulator, stage_0_bwd_p0, stage_0_bwd_p1), kind=kCustom, calls=_pop_op_serialized_gradient_accumulation_1
  ROOT stage_0_bwd_tuple = (f32[128]) tuple(stage_0_bwd_accumulator_update)
}

resource_update {
  resource_update_p0 = f32[128] parameter(0)
  resource_update_p1 = f32[128] parameter(1)
  a0 = f32[128] add(resource_update_p0, resource_update_p1)
  a0_convert = f16[128] convert(a0)
  ROOT t = (f16[128]) tuple(a0_convert)
}

pipeline {
  pipeline_p0 = f16[128] parameter(0), sharding={maximal device=0}
  pipeline_stage_0 = (f16[128]) call(pipeline_p0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0.0 = f16[128] get-tuple-element(pipeline_stage_0), index=0

  pipeline_stage_1 = (f16[128]) call(pipeline_stage_0.0), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  pipeline_stage_1.0 = f16[128] get-tuple-element(pipeline_stage_1), index=0

  pipeline_stage_2 = (f16[128]) call(pipeline_stage_1.0), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  pipeline_stage_2.0 = f16[128] get-tuple-element(pipeline_stage_2), index=0

  pipeline_stage_2_bwd = (f16[128]) call(pipeline_stage_2.0, pipeline_stage_1.0), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
  pipeline_stage_2_bwd.0 = f16[128] get-tuple-element(pipeline_stage_2_bwd), index=0

  pipeline_accumulator_1 = f32[128] custom-call(pipeline_p0), custom_call_target="GradientAccumulatorCreate", backend_config="{}"
  pipeline_stage_1_bwd = (f16[128], f16[128], f32[128]) call(pipeline_stage_2_bwd.0, pipeline_stage_1.0, pipeline_accumulator_1), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  pipeline_stage_1_bwd.0 = f16[128] get-tuple-element(pipeline_stage_1_bwd), index=0
  pipeline_stage_1_bwd.1 = f16[128] get-tuple-element(pipeline_stage_1_bwd), index=1
  pipeline_stage_1_bwd.2 = f32[128] get-tuple-element(pipeline_stage_1_bwd), index=2

  pipeline_accumulator_0 = f32[128] custom-call(pipeline_p0), custom_call_target="GradientAccumulatorCreate", backend_config="{}"
  pipeline_stage_0_bwd = (f32[128]) call(pipeline_stage_1_bwd.0, pipeline_stage_0.0, pipeline_accumulator_0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  pipeline_stage_0_bwd.0 = f32[128] get-tuple-element(pipeline_stage_0_bwd), index=0

  pipeline_accumulator_1_sink = f32[128] custom-call(pipeline_stage_1_bwd.2), custom_call_target="GradientAccumulatorSink", backend_config="{\"num_mini_batches\":1}\n"
  pipeline_accumulator_0_sink = f32[128] custom-call(pipeline_stage_0_bwd.0), custom_call_target="GradientAccumulatorSink", backend_config="{\"num_mini_batches\":1}\n"

  call_ru = (f16[128]) call(pipeline_accumulator_0_sink, pipeline_accumulator_1_sink), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  pipeline_p0_updated = f16[128] get-tuple-element(call_ru), index=0
  ROOT pipeline_tuple = (f16[128]) tuple(pipeline_p0_updated)
}

ENTRY e {
  e.weights0 = f16[128] parameter(0), parameter_replication={false}
  ROOT e.call = (f16[128]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":%d,\"batch_serialization_iterations\":5,\"offload_gradient_accumulation_buffers\":\"%s\"}}}"
}
)";
  return absl::StrFormat(hlo_format, schedule,
                         ThreeState_Name(offload_accumulators));
}

TEST_F(GradientAccumulationBuffersOffloadTest, TestOn) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           GetHlo(0, THREESTATE_ON), config));

  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());

  EXPECT_TRUE(GradientAccumulationBuffersOffload(true, 0)
                  .Run(module.get())
                  .ValueOrDie());

  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(PipelineStages stages,
                          GetPipelineStages(pipeline_computation));
  {
    // Accumulator in stage 1 has no users - don't offload it.
    HloInstruction* stage_1_bwd = stages.backward[1];
    const HloGradientAccumulatorCreate* create =
        Cast<HloGradientAccumulatorCreate>(stage_1_bwd->operand(2));
    EXPECT_FALSE(create->IsRemote());
  }

  {
    // Check accumulator in stage 0 is offloaded.
    HloInstruction* stage_0_bwd = stages.backward[0];
    const HloGradientAccumulatorCreate* create =
        Cast<HloGradientAccumulatorCreate>(stage_0_bwd->operand(2));
    EXPECT_TRUE(create->IsRemote());
    EXPECT_EQ(create->shape().element_type(), F32);
    EXPECT_EQ(create->operand_count(), 1);
    HloComputation* stage_comp = stage_0_bwd->to_apply();
    HloInstruction* param = stage_comp->parameter_instruction(2);
    HloInstruction *load, *store;
    TF_ASSERT_OK(GetRemoteLoadStoreUsers(param, &load, &store));
  }

  {
    // Check the resource update loads in the accumulator.
    HloInstruction* resource_update = *stages.resource_update;
    HloComputation* resource_update_comp = resource_update->to_apply();
    HloInstruction* param = resource_update_comp->parameter_instruction(0);
    EXPECT_EQ(param->user_count(), 1);
    EXPECT_TRUE(
        IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(param->users()[0]));
  }
}

TEST_F(GradientAccumulationBuffersOffloadTest, TestOnTooSmall) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           GetHlo(0, THREESTATE_ON), config));

  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());

  EXPECT_FALSE(GradientAccumulationBuffersOffload(true, 1024)
                   .Run(module.get())
                   .ValueOrDie());
}

TEST_F(GradientAccumulationBuffersOffloadTest, TestOff) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           GetHlo(0, THREESTATE_OFF), config));

  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());

  EXPECT_FALSE(GradientAccumulationBuffersOffload(true, 0)
                   .Run(module.get())
                   .ValueOrDie());
}

TEST_F(GradientAccumulationBuffersOffloadTest, TestNoRemoteMemory) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           GetHlo(0, THREESTATE_ON), config));

  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  auto status_or =
      GradientAccumulationBuffersOffload(false, 0).Run(module.get());
  EXPECT_FALSE(status_or.ok());
  EXPECT_THAT(status_or.status().error_message(),
              ::testing::StartsWith(
                  "Gradient accumulation buffer offloading has been enabled, "
                  "however the current configuration of the IPU devices does "
                  "not support"));
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
