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

#include "tensorflow/compiler/plugin/poplar/driver/passes/resource_update_copy_inserter.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/fifo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using ResourceUpdateCopyInserterTest = HloTestBase;

std::string GetHlo(const std::string& inplace_descriptions) {
  constexpr absl::string_view hlo_format = R"(
HloModule top

stage_0_fwd {
  t = token[] after-all()
  feed = (f32[], token[]) infeed(t)
  input = f32[] get-tuple-element(feed), index=0
  ROOT stage_0_fwd_tuple = (f32[]) tuple(input)
}

stage_1_fwd {
  ROOT tuple = () tuple()
}

stage_1_bwd {
  ROOT tuple = () tuple()
}

stage_0_bwd {
  value = f32[] parameter(0)
  ROOT stage_0_bwd_tuple = (f32[]) tuple(value)
}

_pop_op_add_comp {
  arg0 = f32[] parameter(0)
  arg1 = f32[] parameter(1)
  ROOT add = f32[] add(arg0, arg1)
}

resource_update {
  ru_arg0 = f32[] parameter(0)
  ru_arg1 = f32[] parameter(1)
  ru_arg2 = f32[] parameter(2)
  fusion = f32[] fusion(ru_arg0, ru_arg1), kind=kCustom, calls=_pop_op_add_comp, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[%s]}}"
  add_fusion = f32[] add(ru_arg2, fusion)
  ROOT t = (f32[],f32[]) tuple(add_fusion, fusion)
}

pipeline {
  param0 = f32[] parameter(0), parameter_replication={false}
  param1 = f32[] parameter(1), parameter_replication={false}
  pipeline_stage_0 = (f32[]) call(), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}


  pipeline_stage_1 = () call(), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_bwd = () call(), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}

  stage_0_fwd.0 = f32[] get-tuple-element(pipeline_stage_0), index=0

  pipeline_stage_0_bwd = (f32[]) call(stage_0_fwd.0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd.0 = f32[] get-tuple-element(pipeline_stage_0_bwd), index=0

  call_ru = (f32[],f32[]) call(stage_0_bwd.0, param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  gte1 = f32[] get-tuple-element(call_ru), index=1
  ROOT pipeline_tuple = (f32[], f32[]) tuple(gte1, gte0)
}

ENTRY e {
  e.weights0 = f32[] parameter(0)
  e.weights1 = f32[] parameter(1)
  ROOT e.call = (f32[], f32[]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";
  return absl::StrFormat(hlo_format, inplace_descriptions);
}

TEST_F(ResourceUpdateCopyInserterTest, TestAddCopy) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(
                              GetHlo(/*inplace_descriptions=*/
                                     R"({\"kind\":\"USE_ALIAS_READ_WRITE\"})"),
                              config));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ResourceUpdateCopyInserter().Run(module.get()));
  EXPECT_TRUE(changed);
  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  HloInstruction* resource_update = *stages.resource_update;
  HloInstruction* resource_update_root =
      resource_update->to_apply()->root_instruction();

  const HloInstruction* ru_arg1 = FindInstruction(module.get(), "ru_arg1");
  const HloInstruction* fusion = FindInstruction(module.get(), "fusion");
  const HloInstruction* add_fusion =
      FindInstruction(module.get(), "add_fusion");

  const HloInstruction* copy = resource_update_root->operand(1);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::CopyInto)(copy));
  EXPECT_THAT(copy->operands(), ::testing::ElementsAre(ru_arg1, fusion));

  EXPECT_THAT(resource_update_root->operands(),
              ::testing::ElementsAre(add_fusion, copy));
}

TEST_F(ResourceUpdateCopyInserterTest, TestNoCopy) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(
          GetHlo(/*inplace_descriptions=*/
                 R"({\"kind\":\"USE_ALIAS_READ_WRITE\",\"operand_number\":\"1\"})"),
          config));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ResourceUpdateCopyInserter().Run(module.get()));
  EXPECT_FALSE(changed);
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
