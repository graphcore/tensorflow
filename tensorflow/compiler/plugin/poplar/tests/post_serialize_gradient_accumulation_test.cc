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
#include "tensorflow/compiler/plugin/poplar/driver/passes/post_serialize_gradient_accumulation.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/slice_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using PostSerializeGradientAccumulationTest = HloTestBase;

TEST_F(PostSerializeGradientAccumulationTest, LowerGradientAccumulationFusion) {
  const string& hlo_string = R"(
HloModule main

_pop_op_serialized_gradient_accumulation {
  p0 = f32[2] parameter(0)
  p1 = f32[2] parameter(1)
  a0 = f32[2] add(p0, p1)
  p2 = f32[2] parameter(2)
  a1 = f32[2] add(a0, p2)
  p3 = f32[2] parameter(3)
  ROOT a2 = f32[2] add(a1, p3)
}

ENTRY main {
  p0 = f32[2] parameter(0)
  p1 = f32[2] parameter(1)
  p2 = f32[2] parameter(2)
  p3 = f32[2] parameter(3)
  grad = f32[2] fusion(p0, p1, p2, p3), kind=kCustom, calls=_pop_op_serialized_gradient_accumulation
  ROOT t = (f32[2]) tuple(grad)
}
)";
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  PostSerializeGradientAccumulation psga;
  EXPECT_TRUE(psga.Run(module).ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(
      root->operand(0),
      m::Add(m::Add(m::Add(m::Parameter(0), m::Parameter(1)), m::Parameter(2)),
             m::Parameter(3))));
  HloInstruction* p0 = module->entry_computation()->parameter_instruction(0);
  HloInstruction* p1 = module->entry_computation()->parameter_instruction(1);
  EXPECT_THAT(p0->control_predecessors(), ::testing::ElementsAre(p1));
}

TEST_F(PostSerializeGradientAccumulationTest,
       LowerGradientAccumulationFusionUsers) {
  const string& hlo_string = R"(
HloModule main

_pop_op_serialized_gradient_accumulation {
  p0 = f32[2] parameter(0)
  p1 = f32[2] parameter(1)
  a0 = f32[2] add(p0, p1)
  p2 = f32[2] parameter(2)
  a1 = f32[2] add(a0, p2)
  p3 = f32[2] parameter(3)
  ROOT a2 = f32[2] add(a1, p3)
}

_pop_op_serialized_gradient_accumulation_2 {
  p0 = f32[2] parameter(0)
  p1 = f32[2] parameter(1)
  ROOT a0 = f32[2] add(p0, p1)
}

ENTRY main {
  p0 = f32[2] parameter(0)
  p1 = f32[2] parameter(1)
  p2 = f32[2] parameter(2)
  p3 = f32[2] parameter(3)
  p4 = f32[2] parameter(4)
  grad = f32[2] fusion(p0, p1, p2, p3), kind=kCustom, calls=_pop_op_serialized_gradient_accumulation
  add = f32[2] add(p2, p3)
  grad2 = f32[2] fusion(p4, add), kind=kCustom, calls=_pop_op_serialized_gradient_accumulation_2
  ROOT t = (f32[2], f32[2]) tuple(grad, grad2)
}
)";
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  PostSerializeGradientAccumulation psga;
  EXPECT_TRUE(psga.Run(module).ValueOrDie());

  HloComputation* comp = module->entry_computation();
  HloInstruction* root = comp->root_instruction();
  EXPECT_TRUE(Match(
      root, m::Tuple(m::Add(m::Add(m::Add(m::Parameter(0), m::Parameter(1)),
                                   m::Parameter(2)),
                            m::Parameter(3)),
                     m::Add(m::Parameter(4),
                            m::Add(m::Parameter(2), m::Parameter(3))))));

  HloInstruction* p0 = comp->parameter_instruction(0);
  HloInstruction* p1 = comp->parameter_instruction(1);
  EXPECT_THAT(p0->control_predecessors(), ::testing::ElementsAre(p1));

  auto grad = root->operand(0);
  auto grad_input0 = grad->operand(0);
  auto grad2 = root->operand(1);
  auto grad_2_input = grad2->operand(1);
  EXPECT_THAT(grad_2_input->control_predecessors(),
              ::testing::UnorderedElementsAre(grad, grad_input0));
}

TEST_F(PostSerializeGradientAccumulationTest, AddCreatorControlDeps) {
  const string& hlo_string = R"(
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

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  HloComputation* pipeline_computation = FindComputation(module, "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(PipelineStages stages,
                          GetPipelineStages(pipeline_computation));
  auto creator = stages.backward[0]->operand(2);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(creator));

  PostSerializeGradientAccumulation psga;
  EXPECT_TRUE(psga.Run(module).ValueOrDie());

  // Expect the forward stage to be a control dependency.
  EXPECT_THAT(creator->control_predecessors(),
              ::testing::ElementsAre(stages.forward[0]));
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
