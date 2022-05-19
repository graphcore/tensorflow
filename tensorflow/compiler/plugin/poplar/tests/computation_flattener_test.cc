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
#include "tensorflow/compiler/plugin/poplar/driver/passes/computation_flattener.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using pipeline_config = PoplarBackendConfig::CallConfig::PipelineConfig;

struct ComputationFlattenerTestSpec {
  pipeline_config::RecomputationMode recomputation_mode;
  int64_t expected_funcs;
};

std::ostream& operator<<(std::ostream& os,
                         const ComputationFlattenerTestSpec& spec) {
  return os << "{ recomputation_mode: "
            << pipeline_config::RecomputationMode_Name(spec.recomputation_mode)
            << ", expected funcs: " << spec.expected_funcs << "}";
}

class ComputationFlattenerTest
    : public HloTestBase,
      public ::testing::WithParamInterface<ComputationFlattenerTestSpec> {};

INSTANTIATE_TEST_SUITE_P(
    ComputationFlattenerTestCases, ComputationFlattenerTest,
    ::testing::ValuesIn(std::vector<ComputationFlattenerTestSpec>{
        {pipeline_config::Recompute_then_backpropagate, 0},
        {pipeline_config::Recompute_and_backpropagate_interleaved, 1},
    }));

std::string GetHlo(pipeline_config::RecomputationMode recomputation_mode) {
  constexpr absl::string_view hlo_format = R"(
HloModule top

func {
  p0 = s32[2] parameter(0)
  p1 = s32[2] parameter(1)
  a = s32[2] add(p0, p1)
  ROOT t = (s32[2]) tuple(a)
}

func1 {
  p0 = s32[2] parameter(0)
  p1 = s32[2] parameter(1)
  a = s32[2] subtract(p0, p1)
  ROOT t = (s32[2]) tuple(a)
}

stage_0_fwd {
  const = s32[2] constant({1, 2})
  c = (s32[2]) call(const, const), to_apply=func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  ROOT stage_0_fwd_tuple = () tuple()
}

stage_1_fwd {
  ROOT stage_1_fwd_tuple = () tuple()
}

stage_1_bwd {
  ROOT stage_1_bwd_tuple = () tuple()
}

stage_0_bwd {
  const = s32[2] constant({1, 2})
  c = (s32[2]) call(const, const), to_apply=func1, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  ROOT stage_0_bwd_tuple = () tuple()
}

resource_update {
  resource_update_p0 = f16[128] parameter(0)
  ROOT t = (f16[128]) tuple(resource_update_p0)
}

pipeline {
  pipeline_p0 = f16[128] parameter(0), sharding={maximal device=0}
  pipeline_stage_0 = () call(), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_1 = () call(), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  pipeline_stage_1_bwd = () call(), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  pipeline_stage_0_bwd = () call(), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"

  call_ru = (f16[128]) call(pipeline_p0), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  pipeline_p0_updated = f16[128] get-tuple-element(call_ru), index=0
  ROOT pipeline_tuple = (f16[128]) tuple(pipeline_p0_updated)
}

ENTRY e {
  e.weights0 = f16[128] parameter(0), parameter_replication={false}
  ROOT e.call = (f16[128]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":\"Grouped\",\"recomputationMode\":\"%s\"}}}"
}
)";
  return absl::StrFormat(
      hlo_format, pipeline_config::RecomputationMode_Name(recomputation_mode));
}

int64_t CountFunctions(const HloModule* module) {
  int64_t count = 0;
  for (auto comp : module->computations()) {
    count += absl::c_count_if(comp->instructions(), IsFunction);
  }
  return count;
}
TEST_P(ComputationFlattenerTest, DoTest) {
  auto params = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           GetHlo(params.recomputation_mode)));
  TF_ASSERT_OK(ComputationFlattener().Run(module.get()).status());
  EXPECT_THAT(CountFunctions(module.get()), params.expected_funcs);
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
