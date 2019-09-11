/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using PipelineCopyInserterTest = HloTestBase;

TEST_F(PipelineCopyInserterTest, TestParameterUsedInplace) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_weights1 = f32[1,4,4,2] parameter(1)
  add = f32[1,4,4,2] add(stage_0_fwd_weights0, stage_0_fwd_weights1), backend_config="{\"isInplace\":true}"
  stage_0_fwd_weights2 = f32[1,4,4,2] parameter(2)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(add, stage_0_fwd_weights2)
}

stage_1_fwd {
  stage_1_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights2 = f32[1,4,4,2] parameter(1)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_fwd_weights0, stage_1_fwd_weights2)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_weights2 = f32[1,4,4,2] parameter(2)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1, pipeline_weights2), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_stage_0_w2 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_stage_0_w0, pipeline_stage_0_w2), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  pipeline_stage_1_w2 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_1_w0, pipeline_stage_1_w2)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.weights2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1, e.weights2), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  PipelineCopyInserter inserter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* pipeline_stage_0 =
      FindInstruction(module.get(), "pipeline_stage_0");
  EXPECT_TRUE(Match(pipeline_stage_0->to_apply()->root_instruction(),
                    m::Tuple(m::Add(m::Copy(m::Parameter(0)), m::Parameter(1)),
                             m::Parameter(2))));
}

TEST_F(PipelineCopyInserterTest, TestParameterUsedNotInplace) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_weights1 = f32[1,4,4,2] parameter(1)
  add = f32[1,4,4,2] add(stage_0_fwd_weights0, stage_0_fwd_weights1)
  stage_0_fwd_weights2 = f32[1,4,4,2] parameter(2)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(add, stage_0_fwd_weights2)
}

stage_1_fwd {
  stage_1_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights2 = f32[1,4,4,2] parameter(1)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_fwd_weights0, stage_1_fwd_weights2)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_weights2 = f32[1,4,4,2] parameter(2)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1, pipeline_weights2), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_stage_0_w2 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_stage_0_w0, pipeline_stage_0_w2), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  pipeline_stage_1_w2 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_1_w0, pipeline_stage_1_w2)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.weights2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1, e.weights2), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  PipelineCopyInserter inserter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(PipelineCopyInserterTest, TestParameterUsedInplaceThroughReshape) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_weights1 = f32[32] parameter(1)
  reshape = f32[32] reshape(stage_0_fwd_weights0), backend_config="{\"isInplace\":true}"
  add = f32[32] add(reshape, stage_0_fwd_weights1), backend_config="{\"isInplace\":true}"
  reshape2 = f32[1,4,4,2] reshape(add), backend_config="{\"isInplace\":true}"
  stage_0_fwd_weights2 = f32[1,4,4,2] parameter(2)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(reshape2, stage_0_fwd_weights2)
}

stage_1_fwd {
  stage_1_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights2 = f32[1,4,4,2] parameter(1)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_fwd_weights0, stage_1_fwd_weights2)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[32] parameter(1)
  pipeline_weights2 = f32[1,4,4,2] parameter(2)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1, pipeline_weights2), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_stage_0_w2 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_stage_0_w0, pipeline_stage_0_w2), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  pipeline_stage_1_w2 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_1_w0, pipeline_stage_1_w2)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[32] parameter(1), parameter_replication={false}
  e.weights2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1, e.weights2), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  PipelineCopyInserter inserter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* pipeline_stage_0 =
      FindInstruction(module.get(), "pipeline_stage_0");
  EXPECT_TRUE(
      Match(pipeline_stage_0->to_apply()->root_instruction(),
            m::Tuple(m::Reshape(m::Add(m::Reshape(m::Copy(m::Parameter(0))),
                                       m::Parameter(1))),
                     m::Parameter(2))));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
