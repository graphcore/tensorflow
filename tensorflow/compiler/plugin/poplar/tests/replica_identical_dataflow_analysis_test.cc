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

#include <gtest/gtest.h>

#include "absl/strings/substitute.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/replica_identical_dataflow_analysis.h"

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/user_op_hlo.h"
#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"

#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

#define ASSERT_CATEGORY_EQ(STATUSOR_CATEGORY, EXPECTED_CATEGORY) \
  {                                                              \
    TF_ASSERT_OK_AND_ASSIGN(auto category, STATUSOR_CATEGORY);   \
    ASSERT_EQ(category, EXPECTED_CATEGORY);                      \
  }

se::Platform* GetPlatform(const std::string& name) {
  auto result = PlatformUtil::GetPlatform(name);
  TF_CHECK_OK(result.status()) << "Could not get " << name << " platform";
  return result.ValueOrDie();
}

// Test fixtures for creating flattened HLO, which is what
// the ReplicaIdenticalDataflowAnalysis class expects.
struct ReplicaIdenticalDataflowAnalysisTest : HloTestFixture {
  ReplicaIdenticalDataflowAnalysisTest()
      :  // Using the platform constructor to remove noisy platform messages
         // from the test output
        HloTestFixture(/*test platform*/ GetPlatform("Poplar"),
                       /*reference platform*/ GetPlatform("interpreter")) {}

  ::testing::AssertionResult SetUpHloFlattenedModule(const std::string& hlo,
                                                     int64 replica_count = 1) {
    auto module_setup_result =
        HloTestFixture::SetUpHloModule(hlo, replica_count);
    if (module_setup_result == ::testing::AssertionSuccess()) {
      FlattenCallGraph flatten_call_graph;
      flatten_call_graph.Run(hlo_module_);
    }

    return module_setup_result;
  }
};

struct ParameterizedReplicaDataflowAnalysisTest
    : ParameterizedHloTestFixture<ReplicaIdenticalDataflowAnalysisTest> {
  void SetUp() override {
    ASSERT_TRUE(
        SetUpHloFlattenedModule(GetParam().hlo, GetParam().replica_count));
  }
};

using ReplicaIdenticalInstructionTest =
    ParameterizedReplicaDataflowAnalysisTest;

static const HloTestCase simple_parameters = {"parameters", R"(
HloModule test
ENTRY test {
   param0 = f32[1,1,2,4] parameter(0)
   param1 = f32[1,1,2,4] parameter(1)
   ROOT identical_root = f32[1,1,2,4] add(param0, param1)
}
)"};
static const HloTestCase simple_constants{"constants", R"(
HloModule test
ENTRY test {
   const1 = f32[] constant(0)
   const2 = f32[] constant(4)
   ROOT identical_root = f32[] multiply(const1, const2)
}
)"};
static const HloTestCase simple_wide_const = {"wide_const", R"(
HloModule test
_pop_op_wide_const {
  constant = f32[] constant(0.1)
  ROOT broadcast = f32[3,3,4,12] broadcast(constant), dimensions={}
}

ENTRY test {
 identical0 = f32[3,3,4,12] parameter(0)
 wide_const = f32[3,3,4,12] fusion(), kind=kCustom, calls=_pop_op_wide_const
 ROOT identical_root = f32[3,3,4,12] add(identical0, wide_const)
}
)"};
static const HloTestCase global_all_reduce = {"global_all_reduce", R"(
HloModule test
add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

ENTRY test {
  identical0 = f32[4] parameter(0)
  after-all = token[] after-all()
  infeed = (f32[4], token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed1\"\002\001\001"
  differing_value = f32[4] get-tuple-element((f32[4], token[]) infeed), index=0
  global_all_reduce = f32[4] all-reduce(differing_value), to_apply=add, replica_groups={}
  ROOT identical_root = f32[4] add(identical0, global_all_reduce)
}
)"};
static const HloTestCase global_all_gather = {"global_all_gather", R"(
HloModule test
ENTRY test {
  identical0 = f32[4] parameter(0)
  after-all = token[] after-all()
  infeed = (f32[2], token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed1\"\002\001\001"
  differing_value = f32[2] get-tuple-element((f32[2], token[]) infeed), index=0
  global_all_gather = f32[4] custom-call(differing_value), custom_call_target="AllGather", backend_config="{\"replica_group_size\":0}"
  ROOT identical_root = f32[4] add(identical0, global_all_gather)
}
)"};
static const HloTestCase repeat_with_identical_io = {"repeat_identical_io", R"(
HloModule test
repeat {
  x = f32[] parameter(0)
  increment = f32[] constant(1)
  ROOT count = f32[] add(x, increment)
}

ENTRY test {
  identical0 = f32[] parameter(0)
  loop_count = f32[] call(identical0), to_apply=repeat, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"20\"}}}"
  ROOT identical_root = f32[] add(identical0, loop_count)
}
)"};
static const HloTestCase repeat_single_element_tuple = {
    "repeat_single_element_tuple", R"(
HloModule test
repeat {
  x = f32[] parameter(0)
  increment = f32[] constant(1)
  count = f32[] add(x, increment)
  ROOT tuple = (f32[]) tuple(count)
}

ENTRY test {
  identical0 = f32[] parameter(0)
  loop_tuple = (f32[]) call(identical0), to_apply=repeat, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"20\"}}}"
  loop_count = f32[] get-tuple-element(loop_tuple), index=0
  ROOT identical_root = f32[] add(identical0, loop_count)
}
)"};

static const HloTestCase while_with_identical_body_and_condition = {
    "while_identical_body_and_condition", R"(
HloModule test
body {
  x = s32[] parameter(0)
  increment = s32[] constant(1)
  ROOT count = s32[] add(x, increment)
}

condition {
  x = s32[] parameter(0)
  const = s32[] constant(10)
  ROOT result = pred[] compare(x, const), direction=LT
}

ENTRY test {
  identical0 = s32[] constant(0)
  loop_count = s32[] while(identical0), condition=condition, body=body
  ROOT identical_root = s32[] add(identical0, loop_count)
}
)"};
static const HloTestCase conditional_with_identical_branches_and_pred = {
    "conditional_with_identical_branches_and_pred", R"(
HloModule test
cond_false {
  x = f32[] parameter(0)
  increment = f32[] constant(1)
  ROOT add = f32[] add(x, increment)
}

cond_true {
  x = f32[] parameter(0)
  increment = f32[] constant(-1)
  ROOT add = f32[] add(x, increment)
}

ENTRY test {
  identical_pred = pred[] parameter(0)
  identical_true_param = f32[] parameter(1)
  identical_false_param = f32[] parameter(2)
  conditional = f32[] conditional(identical_pred, identical_true_param, identical_false_param), true_computation=cond_true, false_computation=cond_false
  ROOT identical_root = f32[] add(identical_true_param, conditional)
}
)"};
static const HloTestCase switch_with_identical_branches_and_index = {
    "switch_with_identical_branches_and_index", R"(
HloModule test
branchA {
  x = f32[] parameter(0)
  increment = f32[] constant(1)
  ROOT add = f32[] add(x, increment)
}

branchB {
  x = f32[] parameter(0)
  increment = f32[] constant(-1)
  ROOT add = f32[] add(x, increment)
}

branchC {
  x = f32[] parameter(0)
  increment = f32[] constant(10)
  ROOT add = f32[] add(x, increment)
}

ENTRY test {
  identical_index = s32[] parameter(0)
  identical_branchA_param = f32[] parameter(1)
  identical_branchB_param = f32[] parameter(2)
  identical_branchC_param = f32[] parameter(3)
  conditional = f32[] conditional(identical_index, identical_branchA_param, identical_branchB_param, identical_branchC_param), branch_computations={branchA, branchB, branchC}
  ROOT identical_root = f32[] add(identical_branchA_param, conditional)
}
)"};
static const HloTestCase simple_pipeline = {"simple_pipeline", R"(
HloModule test
stage_0_fwd {
  x = f32[1,4,4,2] parameter(0)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,42]) tuple(x, x)
}

stage_1_fwd {
  x = f32[1,4,4,2] parameter(0)
  ROOT tuple = (f32[1,4,4,2]) tuple(x)
}

stage_0_bwd {
  x = f32[1,4,4,2] parameter(0)
  ROOT tuple = (f32[1,4,4,2]) tuple(x)
}

resource_update {
 x = f32[1,4,4,2] parameter(0)
 y = f32[1,4,4,2] parameter(1)
 ROOT xy = (f32[1,4,4,2], f32[1,4,4,2]) tuple(x, y)
}

pipeline {
  pipeline_input0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_input0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=-1}
  pipeline_stage_0_i0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_input1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[1,4,4,2]) call(pipeline_input1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_i0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0

  pipeline_stage_bwd = (f32[1,4,4,2]) call(pipeline_stage_1_i0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=1}
  pipeline_stage_bwd_i0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_bwd), index=0

  updated_params = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_stage_0_i0, pipeline_stage_bwd_i0), to_apply=resource_update, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  updated_param0 = f32[1,4,4,2] get-tuple-element(updated_params), index=0
  updated_param1 = f32[1,4,4,2] get-tuple-element(updated_params), index=1

  gradient_accumulation_count = s32[] parameter(2)

  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2], s32[]) tuple(updated_param0, updated_param1, gradient_accumulation_count)
}

ENTRY test {
  input0 = f32[1,4,4,2] parameter(0)
  input1 = f32[1,4,4,2] parameter(1)
  gradient_accumulation_count = s32[] constant(8)
  ROOT call = (f32[1,4,4,2], f32[1,4,4,2], s32[]) call(input0, input1, gradient_accumulation_count), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipeline_config\":{\"repeatCount\":\"4\"}}}"
}
)"};
static const HloTestCase simple_select = {"simple_select", R"(
HloModule test

ENTRY test {
  identical_pred = pred[] parameter(0)
  identical_on_value = f32[] parameter(1)
  identical_off_value = f32[] parameter(2)
  select = f32[] select(identical_pred, identical_on_value, identical_off_value)
  ROOT identical_root = f32[] add(identical_on_value, select)
}
)"};
static const HloTestCase simple_tuple_select = {"simple_tuple_select", R"(
HloModule test

ENTRY test {
  identical_pred = pred[] parameter(0)
  identical_on_value = (f32[]) parameter(1)
  identical_off_value = (f32[]) parameter(2)
  ROOT select = (f32[]) tuple-select(identical_pred, identical_on_value, identical_off_value)
}
)"};
static const HloTestCase compare_with_identical_operands = {
    "compare_with_identical_operands", R"(
HloModule test

ENTRY test {
  identical0 = s32[] parameter(0)
  constant = s32[] constant(10)
  ROOT compare = pred[] compare(identical0, constant), direction=LT
}
)"};
static const HloTestCase assume_equal_identical_operands = {
    "assume_equal_identical_operands", R"(
HloModule test

ENTRY test {
  identical0 = f32[] constant(1)
  ROOT assumeEqual = f32[] custom-call(identical0), custom_call_target="AssumeEqualAcrossReplicas"
}
)"};
static const HloTestCase assume_equal_differing_operands = {
    "assume_equal_differing_operands", R"(
HloModule test

ENTRY test {
  const = f32[] constant(1)
  differing0 = f32[] rng(const, const), distribution=rng_uniform
  ROOT assumeEqual = f32[] custom-call(differing0), custom_call_target="AssumeEqualAcrossReplicas"
}
)"};
TEST_P(ReplicaIdenticalInstructionTest, ValueCategory) {
  CustomOpReplacer custom_op_replacer;
  // We dont assert against the return value of this since it's not relevent
  // for all hlo being tested.
  custom_op_replacer.Run(hlo_module_);

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  // The root should be replica identical since all the other instructions in
  // the computation that contribute to it are as well.
  auto* root = FindRootInstruction();
  TF_ASSERT_OK_AND_ASSIGN(bool is_identical,
                          analysis.IsValueIdenticalAcrossReplicas(root));
  ASSERT_TRUE(is_identical) << "Root instruction should be replica identical";
}

using ReplicaDifferingInstructionTest =
    ParameterizedReplicaDataflowAnalysisTest;

static const HloTestCase simple_infeed = {"infeed", R"(
HloModule test
ENTRY test {
   identical0 = f32[1, 1, 2, 4] parameter(0)
   after-all = token[] after-all()
   infeed = (f32[1,1,2,4], token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed1\"\002\001\001"
   infeed_value = f32[1,1,2,4] get-tuple-element((f32[1,1,2,4], token[]) infeed), index=0
   ROOT differing_root = f32[1,1,2,4] add(identical0, infeed_value)
}
)"};
static const HloTestCase partial_all_reduce = {"partial_all_reduce", R"(
HloModule test
add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

ENTRY test {
  identical0 = f32[4] parameter(0)
  partial_all_reduce = f32[4] all-reduce(identical0), to_apply=add, replica_groups={{0, 2}, {1, 3}}
  ROOT differing_root = f32[4] add(identical0, partial_all_reduce)
}
)",
                                               /*replica_count=*/4};
static const HloTestCase partial_all_gather = {"partial_all_gather", R"(
HloModule test
ENTRY test {
  identical0 = f32[4] parameter(0)
  identical1 = f32[2] parameter(1)
  partial_all_gather = f32[4] custom-call(identical1), custom_call_target="AllGather", backend_config="{\"replica_group_size\":2}"
  ROOT differing_root = f32[4] add(identical0, partial_all_gather)
}
)"};
static const HloTestCase uniform_rng = {"uniform_rng", R"(
HloModule test
ENTRY test {
  identical0 = f32[] constant(0)
  identical1 = f32[] constant(1)
  identical2 = f32[8]  broadcast(f32[] identical0), dimensions={}, backend_config="{}"
  rng_uniform = f32[8] rng(identical0, identical1), distribution=rng_uniform
  ROOT differing_root = f32[8] add(identical2, rng_uniform)
}
)"};
static const HloTestCase normal_rng = {"normal_rng", R"(
HloModule test
ENTRY test {
  identical0 = f32[] constant(0)
  identical1 = f32[] constant(1)
  identical2 = f32[8]  broadcast(f32[] identical0), dimensions={}, backend_config="{}"
  rng_normal = f32[8] rng(identical0, identical1), distribution=rng_normal
  ROOT differing_root = f32[8] add(identical2, rng_normal)
}
)"};
static const HloTestCase conditional_with_differing_pred = {
    "conditional_differing_pred", R"(
HloModule test
identical_func {
  ROOT x = f32[] parameter(0)
}

ENTRY test {
  after-all = token[] after-all()
  infeed = (pred[], token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed1\"\002\001\001"
  differing_pred = pred[] get-tuple-element((pred[], token[]) infeed), index=0

  identical1 = f32[] parameter(0)
  identical2 = f32[] parameter(1)
  conditional = f32[] conditional(differing_pred, identical1, identical2), true_computation=identical_func, false_computation=identical_func
  ROOT differing_root = f32[] add(identical2, conditional)
}
)"};
static const HloTestCase conditional_with_differing_branches = {
    "array_conditional_differing_branches", R"(
HloModule test
cond_true {
  ROOT x = f32[] parameter(0)
}

cond_false {
  x = f32[] parameter(0)
  constant = f32[] constant(1)
  ROOT differing = f32[] rng(constant, constant), distribution=rng_uniform
}

ENTRY test {
  identical_pred = pred[] parameter(0)
  identical1 = f32[] parameter(1)
  identical2 = f32[] parameter(2)
  conditional = f32[] conditional(identical_pred, identical1, identical2), true_computation=cond_true, false_computation=cond_false
  ROOT differing_root = f32[] add(identical2, conditional)
}
)"};
static const HloTestCase switch_with_differing_index = {
    "switch_with_differing_index", R"(
HloModule test
branchA {
  x = f32[] parameter(0)
  increment = f32[] constant(1)
  ROOT add = f32[] add(x, increment)
}

branchB {
  x = f32[] parameter(0)
  increment = f32[] constant(-1)
  ROOT add = f32[] add(x, increment)
}

branchC {
  x = f32[] parameter(0)
  increment = f32[] constant(10)
  ROOT add = f32[] add(x, increment)
}

ENTRY test {
  after-all = token[] after-all()
  infeed = (s32[], token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed1\"\002\001\001"
  differing_index = s32[] get-tuple-element(infeed), index=0

  identical_branchA_param = f32[] parameter(0)
  identical_branchB_param = f32[] parameter(1)
  identical_branchC_param = f32[] parameter(2)
  conditional = f32[] conditional(differing_index, identical_branchA_param, identical_branchB_param, identical_branchC_param), branch_computations={branchA, branchB, branchC}
  ROOT differing_root = f32[] add(identical_branchA_param, conditional)
}
)"};
static const HloTestCase switch_with_differing_branches = {
    "switch_with_differing_branches", R"(
HloModule test
branchA {
  x = f32[] parameter(0)
  increment = f32[] constant(1)
  ROOT add = f32[] add(x, increment)
}

branchB {
  x = f32[] parameter(0)
  constant = f32[] constant(1)
  ROOT differing = f32[] rng(constant, constant), distribution=rng_uniform
}

branchC {
  x = f32[] parameter(0)
  increment = f32[] constant(10)
  ROOT add = f32[] add(x, increment)
}

ENTRY test {
  identical_index = s32[] parameter(0)
  identical_branchA_param = f32[] parameter(1)
  identical_branchB_param = f32[] parameter(2)
  identical_branchC_param = f32[] parameter(3)
  conditional = f32[] conditional(identical_index, identical_branchA_param, identical_branchB_param, identical_branchC_param), branch_computations={branchA, branchB, branchC}
  ROOT differing_root = f32[] add(identical_branchA_param, conditional)
}
)"};
static const HloTestCase repeat_with_differing_inputs = {
    "repeat_differing_inputs", R"(
HloModule test
repeat {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  count = f32[] add(x, y)
  ROOT tuple = (f32[], f32[]) tuple(count, y)
}

ENTRY test {
  identical0 = f32[] parameter(0)
  constant = f32[] constant(2)
  differing0 = f32[] rng(constant, constant), distribution=rng_uniform
  loop = (f32[], f32[]) call(identical0, differing0), to_apply=repeat, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"20\"}}}"
  loop_count = f32[] get-tuple-element(loop), index=0
  ROOT differing_root = f32[] add(identical0, loop_count)
}
)"};
static const HloTestCase repeat_with_differing_outputs = {
    "repeat_differing_outputs", R"(
HloModule test
differing_repeat {
  x = f32[] parameter(0)
  constant = f32[] constant(2)
  differing0 = f32[] rng(constant, constant), distribution=rng_uniform
  ROOT differing1 = f32[] add(x, differing0)
}

ENTRY test {
  identical0 = f32[] parameter(0)
  loop_value = f32[] call(identical0), to_apply=differing_repeat, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"20\"}}}"
  ROOT differing_root = f32[] add(identical0, loop_value)
}
)"};
static const HloTestCase while_with_differing_condition = {
    "while_differing_condition", R"(
HloModule test
body {
  x = s32[] parameter(0)
  ROOT identical = s32[] constant(1)
}

condition {
  x = s32[] parameter(0)
  const = s32[] constant(10)
  ROOT result = pred[] compare(x, const), direction=LT
}

ENTRY test {
  identical0 = s32[] constant(2)
  after-all = token[] after-all()
  infeed = (s32[], token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed1\"\002\001\001"
  differing0 = s32[] get-tuple-element((s32[], token[]) infeed), index=0

  loop_count = s32[] while(differing0), condition=condition, body=body
  ROOT differing_root = s32[] add(identical0, loop_count)
}
)"};
static const HloTestCase while_with_differing_body = {"while_differing_body",
                                                      R"(
HloModule test
differing_body {
  x = s32[] parameter(0)
  constant = s32[] constant(1)
  differing0 = s32[] rng(constant, constant), distribution=rng_uniform
  ROOT differing1 = s32[] add(x, differing0)
}

condition {
  x = s32[] parameter(0)
  ROOT true_pred = pred[] constant(true)
}

ENTRY test {
  identical0 = s32[] parameter(0)
  loop_count = s32[] while(identical0), condition=condition, body=differing_body
  ROOT differing_root = s32[] add(identical0, loop_count)
}
)"};
static const HloTestCase select_with_differing_pred = {
    "select_with_differing_pred", R"(
HloModule test

ENTRY test {
  after-all = token[] after-all()
  infeed = (pred[], token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed1\"\002\001\001"
  differing_pred = pred[] get-tuple-element((pred[], token[]) infeed), index=0

  identical_on_value = f32[] parameter(0)
  identical_off_value = f32[] parameter(1)
  select = f32[] select(differing_pred, identical_on_value, identical_off_value)
  ROOT differing_root = f32[] add(identical_on_value, select)
}
)"};
static const HloTestCase select_with_differing_values = {
    "select_with_differing_values", R"(
HloModule test

ENTRY test {
  identical_pred = pred[] parameter(0)

  constant = f32[] constant(1)
  differing_on_value = f32[] rng(constant, constant), distribution=rng_uniform

  identical_off_value = f32[] parameter(1)

  select = f32[] select(identical_pred, differing_on_value, identical_off_value)
  ROOT differing_root = f32[] add(identical_off_value, select)
}
)"};
static const HloTestCase tuple_select_with_differing_pred = {
    "tuple_select_with_differing_pred", R"(
HloModule test

ENTRY test {
  after-all = token[] after-all()
  infeed = (pred[], token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed1\"\002\001\001"
  differing_pred = pred[] get-tuple-element((pred[], token[]) infeed), index=0

  identical_on_value = (f32[]) parameter(0)
  identical_off_value = (f32[]) parameter(1)
  ROOT select = (f32[]) tuple-select(differing_pred, identical_on_value, identical_off_value)
}
)"};
static const HloTestCase tuple_select_with_differing_values = {
    "tuple_select_with_differing_values", R"(
HloModule test

ENTRY test {
  identical_pred = pred[] parameter(0)

  after-all = token[] after-all()
  infeed = ((f32[]), token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed1\"\002\001\001"
  differing_on_value = (f32[]) get-tuple-element(((f32[]), token[]) infeed), index=0

  identical_off_value = (f32[]) parameter(1)
  ROOT select = (f32[]) tuple-select(identical_pred, differing_on_value, identical_off_value)
}
)"};
static const HloTestCase compare_with_differing_operands = {
    "compare_with_differing_operands", R"(
HloModule test

ENTRY test {
  after-all = token[] after-all()
  infeed = (s32[], token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed1\"\002\001\001"
  differing0 = s32[] get-tuple-element(infeed), index=0
  constant = s32[] constant(10)
  ROOT compare = pred[] compare(differing0, constant), direction=LT
}
)"};
static const HloTestCase pipeline_with_differing_gradient_accumulation_count = {
    "pipeline_with_differing_gradient_accumulation_count", R"(
HloModule test
stage_0_fwd {
  x = f32[1,4,4,2] parameter(0)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,42]) tuple(x, x)
}

stage_1_fwd {
  x = f32[1,4,4,2] parameter(0)
  ROOT tuple = (f32[1,4,4,2]) tuple(x)
}

stage_0_bwd {
  x = f32[1,4,4,2] parameter(0)
  ROOT tuple = (f32[1,4,4,2]) tuple(x)
}

resource_update {
 x = f32[1,4,4,2] parameter(0)
 y = f32[1,4,4,2] parameter(1)
 ROOT xy = (f32[1,4,4,2], f32[1,4,4,2]) tuple(x, y)
}

pipeline {
  pipeline_input0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_input0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=-1}
  pipeline_stage_0_i0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_input1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[1,4,4,2]) call(pipeline_input1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_i0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0

  pipeline_stage_bwd = (f32[1,4,4,2]) call(pipeline_stage_1_i0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=1}
  pipeline_stage_bwd_i0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_bwd), index=0

  updated_params = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_stage_0_i0, pipeline_stage_bwd_i0), to_apply=resource_update, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  updated_param0 = f32[1,4,4,2] get-tuple-element(updated_params), index=0
  updated_param1 = f32[1,4,4,2] get-tuple-element(updated_params), index=1

  gradient_accumulation_count = s32[] parameter(2)

  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2], s32[]) tuple(updated_param0, updated_param1, gradient_accumulation_count)
}

ENTRY test {
  input0 = f32[1,4,4,2] parameter(0)
  input1 = f32[1,4,4,2] parameter(1)

  after-all = token[] after-all()
  infeed = (s32[], token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed1\"\002\001\001"
  differing_gradient_accumulation_count = s32[] get-tuple-element(infeed), index=0

  ROOT call = (f32[1,4,4,2], f32[1,4,4,2], s32[]) call(input0, input1, differing_gradient_accumulation_count), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipeline_config\":{\"repeatCount\":\"4\"}}}"
}
)"};
static const HloTestCase dropout_with_seed_op = {"dropout_with_seed_op", R"(
HloModule test

ENTRY test {
  param0 = f32[1024,1024,4] parameter(0)
  reshape = f32[1024,1024,4] reshape(param0)
  seed = s32[2] custom-call(), custom_call_target="Seed"
  ROOT dropout = (f32[1024,1024,4], s32[2], opaque[]) custom-call(reshape, seed), custom_call_target="Dropout", backend_config="{\n\t\"rate\" : 0.5,\n\t\"scale\" : 2.0\n}"
}
)"};
static const HloTestCase truncated_normal = {"truncated_normal", R"(
HloModule test

ENTRY test {
  constant = f32[] constant(2)
  truncated_normal = f32[] custom-call(), custom_call_target="TruncatedNormal"
  ROOT differing_root = f32[] multiply(truncated_normal, constant)
}
)"};
TEST_P(ReplicaDifferingInstructionTest, ValueCategory) {
  CustomOpReplacer custom_op_replacer;
  // We dont assert against the return value of this since it's not relevent
  // for all hlo being tested.
  custom_op_replacer.Run(hlo_module_);

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  // The root should not be replica identical since there are instructions in
  // the graph which are not. Although sub-values of the root shape could still
  // be identical.
  auto* root = FindRootInstruction();
  TF_ASSERT_OK_AND_ASSIGN(bool is_identical,
                          analysis.IsValueIdenticalAcrossReplicas(root));
  ASSERT_FALSE(is_identical)
      << "Root instruction should not be replica identical";
}

using ReplicaValueCategoryGTETest = ParameterizedReplicaDataflowAnalysisTest;

const char* get_tuple_elements = R"(
HloModule test
ENTRY test {
   identical0 = f32[] parameter(0)
   after-all = token[] after-all()
   differing0 = (f32[], token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed1\"\002\001\001"

   mixed_tuple = ((f32[], token[]), f32[]) tuple(differing0, identical0)

   gte0 = (f32[], token[]) get-tuple-element(mixed_tuple), index=0
   ROOT gte1 = f32[] get-tuple-element(mixed_tuple), index=1
}
)";
TEST_F(ReplicaIdenticalDataflowAnalysisTest, GTEValueCategory) {
  ASSERT_TRUE(SetUpHloFlattenedModule(get_tuple_elements));

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  auto* tuple = FindInstruction(hlo_module_, "mixed_tuple");

  TF_ASSERT_OK_AND_ASSIGN(auto tuple0_category,
                          analysis.ValueCategory(tuple, {0}));
  TF_ASSERT_OK_AND_ASSIGN(auto tuple1_category,
                          analysis.ValueCategory(tuple, {1}));

  // The get-tuple-element value category should be the same as its
  // tuple index
  auto* gte0 = FindInstruction(hlo_module_, "gte0");
  auto* gte1 = FindInstruction(hlo_module_, "gte1");
  ASSERT_TRUE(gte0);
  ASSERT_TRUE(gte1);

  ASSERT_CATEGORY_EQ(analysis.ValueCategory(gte0), tuple0_category);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(gte1), tuple1_category);
}

using ReplicaValueCategoryTupleForwardingTest =
    ParameterizedReplicaDataflowAnalysisTest;

static const HloTestCase simple_tuple_forwarding = {"simple_tuple", R"(
HloModule test
add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

ENTRY test {
   identical0 = f32[1, 2] parameter(0)
   identical1 = f32[1, 2] parameter(1)
   differing0 = f32[1, 2] all-reduce(identical0), to_apply=add, replica_groups={{0, 2}, {1, 3}}
   ROOT tuple = (f32[1, 2], f32[1, 2], f32[1, 2]) tuple(identical0, differing0, identical1)
}
)",
                                                    /*replica_count=*/4};
static const HloTestCase nested_tuple_forwarding = {"nested_tuple", R"(
HloModule test
add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

ENTRY test {
   identical0 = f32[1, 2] parameter(0)
   identical1 = f32[1, 2] parameter(1)
   differing0 = f32[1, 2] all-reduce(identical0), to_apply=add, replica_groups={{0, 2}, {1, 3}}

   tuple1 = (f32[1, 2], f32[1, 2], f32[1, 2]) tuple(identical0, differing0, identical1)
   ROOT nested_tuple = ((f32[1, 2], f32[1, 2], f32[1, 2])) tuple(tuple1)
}
)",
                                                    /*replica_count=*/4};
TEST_P(ReplicaValueCategoryTupleForwardingTest,
       TupleForwardsCategoryOfOperands) {
  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  auto* tuple = FindRootInstruction();
  auto leaf_shapes = ShapeUtil::GetLeafShapes(tuple->shape());
  ASSERT_EQ(leaf_shapes.size(), 3);

  // Each of the leaf values of the tuple correspond to the value of a leaf
  // instruction in the HLO.
  //
  // The category of leaf0 should match instruction identical0
  TF_ASSERT_OK_AND_ASSIGN(ValueReplicaCategory leaf0_category,
                          analysis.ValueCategory(tuple, leaf_shapes[0].index));
  ASSERT_EQ(leaf0_category, ValueReplicaCategory::Identical);

  // The category of leaf1 should match instruction differing0
  TF_ASSERT_OK_AND_ASSIGN(ValueReplicaCategory leaf1_category,
                          analysis.ValueCategory(tuple, leaf_shapes[1].index));
  ASSERT_EQ(leaf1_category, ValueReplicaCategory::Differing);

  // The category of leaf2 should match instruction identical1
  TF_ASSERT_OK_AND_ASSIGN(ValueReplicaCategory leaf2_category,
                          analysis.ValueCategory(tuple, leaf_shapes[2].index));
  ASSERT_EQ(leaf2_category, ValueReplicaCategory::Identical);
}

using ReplicaValueCategoryTupleOutputTest =
    ParameterizedReplicaDataflowAnalysisTest;

static const HloTestCase identical_tuple_output = {"identical_tuple", R"(
HloModule test
ENTRY test {
   identical0 = f32[] parameter(0)
   identical1 = f32[] parameter(1)
   ROOT identical_tuple = (f32[1, 1, 2, 4], f32[1, 1, 2, 4]) tuple(identical0, identical1)
}
)"};
static const HloTestCase differing_tuple_output = {"differing_tuple", R"(
HloModule test
ENTRY test {
   identical0 = f32[] parameter(0)
   after-all = token[] after-all()
   differing = (f32[], token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed1\"\002\001\001"
   ROOT mixed_tuple = (f32[1, 1, 2, 4], f32[1, 1, 2, 4]) tuple(identical0, differing)
}
)"};
TEST_P(ReplicaValueCategoryTupleOutputTest, ValueIsIdenticalIfSubShapesAre) {
  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  auto* tuple = FindRootInstruction();
  TF_ASSERT_OK_AND_ASSIGN(bool tuple_elem0_is_identical,
                          analysis.IsValueIdenticalAcrossReplicas(tuple, {0}));
  TF_ASSERT_OK_AND_ASSIGN(bool tuple_elem1_is_identical,
                          analysis.IsValueIdenticalAcrossReplicas(tuple, {1}));
  const auto identical_sub_shapes =
      tuple_elem0_is_identical && tuple_elem1_is_identical;

  TF_ASSERT_OK_AND_ASSIGN(bool tuple_is_identical,
                          analysis.IsValueIdenticalAcrossReplicas(tuple));

  ASSERT_EQ(tuple_is_identical, identical_sub_shapes);
}

const char* multi_computation_hlo = R"(
HloModule test
add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

ENTRY test {
  identical0 = f32[4] parameter(0)
  ROOT global_all_reduce = f32[4] all-reduce(identical0), to_apply=add, replica_groups={}
}
)";
TEST_F(ReplicaIdenticalDataflowAnalysisTest, UnvisitedInstructionErrors) {
  ASSERT_TRUE(SetUpHloFlattenedModule(multi_computation_hlo));

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  auto* unvisited_comp = FindComputation(hlo_module_, "add");
  ASSERT_FALSE(analysis.Analysed(unvisited_comp));

  const auto status_or_value_category =
      analysis.ValueCategory(FindInstruction(hlo_module_, "x"));
  ASSERT_FALSE(status_or_value_category.ok());

  const auto status_or_identical = analysis.IsValueIdenticalAcrossReplicas(
      FindInstruction(hlo_module_, "y"));
  ASSERT_FALSE(status_or_identical.ok());
}

TEST_F(ReplicaIdenticalDataflowAnalysisTest, CanOverrideValueCategory) {
  ASSERT_TRUE(SetUpHloFlattenedModule(multi_computation_hlo));

  auto* root = FindRootInstruction();

  ValuesIdenticalAcrossReplicasVisitor visitor;
  root->Accept(&visitor);

  auto& original_category_mapping = visitor.ValueCategoryMapping();
  const auto original_value_category =
      original_category_mapping.at(root).element(RootShapeIndex());

  ValuesIdenticalAcrossReplicasVisitor visitor_with_override(
      {{root,
        ValueCategoryTree(root->shape(), ValueReplicaCategory::Differing)}});
  root->Accept(&visitor_with_override);

  auto& overriden_category_mapping =
      visitor_with_override.ValueCategoryMapping();
  const auto overriden_value_category =
      overriden_category_mapping.at(root).element(RootShapeIndex());

  ASSERT_EQ(overriden_value_category, ValueReplicaCategory::Differing);
  ASSERT_NE(overriden_value_category, original_value_category);
}

const char* paramless_callables_hlo = R"(
HloModule test
identical_func {
  x = f32[] constant(1)
  y = f32[] constant(2)
  ROOT identical_root = f32[] add(x, y)
}

differing_func {
  constant = f32[] constant(2)
  rng = f32[] rng(constant, constant), distribution=rng_uniform
  ROOT differing_root = f32[] add(constant, rng)
}

ENTRY test {
  identical_call = f32[] call(), to_apply=identical_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  differing_call = f32[] call(), to_apply=differing_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"

  ROOT tuple = (f32[], f32[], f32[]) tuple(identical_call, differing_call)
}
)";
TEST_F(ReplicaIdenticalDataflowAnalysisTest, ParamlessCallValueCategory) {
  ASSERT_TRUE(SetUpHloFlattenedModule(paramless_callables_hlo));

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  auto* identical_call = FindInstruction(hlo_module_, "identical_call");
  auto* differing_call = FindInstruction(hlo_module_, "differing_call");
  ASSERT_TRUE(identical_call);
  ASSERT_TRUE(differing_call);

  // Check that a parameterless call has the value category
  // of the computation its calling
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(identical_call),
                     ValueReplicaCategory::Identical);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(differing_call),
                     ValueReplicaCategory::Differing);
}

const char* calls_with_mixed_params_hlo = R"(
HloModule test
identical_func {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

ENTRY test {
  identical0 = f32[] parameter(0)
  identical1 = f32[] constant(3)
  identical_call0 = f32[] call(identical0, identical1), to_apply=identical_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"

  constant = f32[] constant(2)
  differing0 = f32[] rng(constant, constant), distribution=rng_uniform
  differing1 = f32[] rng(constant, constant), distribution=rng_uniform

  differing_call0 = f32[] call(differing0, differing1), to_apply=identical_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  mixed_call0 = f32[] call(identical0, differing1), to_apply=identical_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"

  ROOT tuple = (f32[], f32[], f32[]) tuple(identical_call0, differing_call0, mixed_call0)
}
)";
TEST_F(ReplicaIdenticalDataflowAnalysisTest, CallableValueCategory) {
  ASSERT_TRUE(SetUpHloFlattenedModule(calls_with_mixed_params_hlo));

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  auto* identical_call0 = FindInstruction(hlo_module_, "identical_call0");
  auto* differing_call0 = FindInstruction(hlo_module_, "differing_call0");
  auto* mixed_call0 = FindInstruction(hlo_module_, "mixed_call0");
  ASSERT_TRUE(identical_call0);
  ASSERT_TRUE(differing_call0);
  ASSERT_TRUE(mixed_call0);

  // Since the function being called is replica identical the value category of
  // the caller should just depend on the params it passes in.
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(identical_call0),
                     ValueReplicaCategory::Identical);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(differing_call0),
                     ValueReplicaCategory::Differing);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(mixed_call0),
                     ValueReplicaCategory::Differing);
}

const char* chained_calls_hlo = R"(
HloModule test
identical_func {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT flipped_tuple = (f32[], f32[]) tuple(y, x)
}

intermediate_func {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT call = (f32[], f32[]) call(x, y), to_apply=identical_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
}

ENTRY test {
  identical0 = f32[] parameter(0)

  constant = f32[] constant(2)
  differing0 = f32[] rng(constant, constant), distribution=rng_uniform

  ROOT call = (f32[], f32[]) call(identical0, differing0), to_apply=intermediate_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
}
)";
TEST_F(ReplicaIdenticalDataflowAnalysisTest, ChainedCallsValueCategory) {
  ASSERT_TRUE(SetUpHloFlattenedModule(chained_calls_hlo));

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  // We expect the root call to have the same value category
  // as if calling identical_func directly, since intermediate_func
  // should just pass the params though.
  auto* root_call = FindRootInstruction();
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(root_call, {0}),
                     ValueReplicaCategory::Differing);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(root_call, {1}),
                     ValueReplicaCategory::Identical);
}

const char* tuple_conditional_with_mixed_branches = R"(
HloModule test
cond_false {
  x = f32[] parameter(0)

  identical0 = f32[] constant(1)
  identical1 = f32[] add(x, identical0)
  ROOT tuple = (f32[], f32[]) tuple(identical0, identical1)
}

cond_true {
  x = f32[] parameter(0)
  increment = f32[] constant(-1)
  identical = f32[] add(x, increment)

  after-all = token[] after-all()
  infeed = (f32[], token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed1\"\002\001\001"
  differing = f32[] get-tuple-element(infeed), index=0

  ROOT tuple = (f32[], f32[]) tuple(differing, identical)
}

ENTRY test {
  identical_pred = pred[] parameter(0)
  identical_true_param = f32[] parameter(1)
  identical_false_param = f32[] parameter(2)
  ROOT conditional = (f32[], f32[]) conditional(identical_pred, identical_true_param, identical_false_param), true_computation=cond_true, false_computation=cond_false
}
)";
TEST_F(ReplicaIdenticalDataflowAnalysisTest, TupleConditionalValueCategory) {
  ASSERT_TRUE(SetUpHloFlattenedModule(tuple_conditional_with_mixed_branches));

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  // Only element1 is a replica identical value in both branches
  // of our conditional, so its category should be identical and
  // element0s differing.
  auto* root = FindRootInstruction();
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(root, {0}),
                     ValueReplicaCategory::Differing);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(root, {1}),
                     ValueReplicaCategory::Identical);
}
const char* tuple_select_with_mixed_values = R"(
HloModule test
ENTRY test {
  constant = f32[] constant(1)
  differing0 = f32[] rng(constant, constant), distribution=rng_uniform

  identical_pred = pred[] parameter(0)
  identical0 = f32[] parameter(1)
  identical1 = f32[] parameter(2)

  mixed_on_value = (f32[], f32[], f32[]) tuple(differing0, identical0, identical1)
  mixed_off_value = (f32[], f32[], f32[]) tuple(identical1, identical0, differing0)

  ROOT select = (f32[], f32[], f32[]) tuple-select(identical_pred, mixed_on_value, mixed_off_value)
}
)";
TEST_F(ReplicaIdenticalDataflowAnalysisTest, TupleSelectcValueCategory) {
  ASSERT_TRUE(SetUpHloFlattenedModule(tuple_select_with_mixed_values));

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  // Values should be identical if they're in both branches, so
  // element1 should be replica identical element0 not.
  auto* root = FindRootInstruction();
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(root, {0}),
                     ValueReplicaCategory::Differing);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(root, {1}),
                     ValueReplicaCategory::Identical);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(root, {2}),
                     ValueReplicaCategory::Differing);
}

TEST_F(ReplicaIdenticalDataflowAnalysisTest, CanQuerySubComputations) {
  ASSERT_TRUE(SetUpHloFlattenedModule(chained_calls_hlo));

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  // Make sure that we can query instructions outside of the entry computation.
  auto* flipped_tuple = FindInstruction(hlo_module_, "flipped_tuple");
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(flipped_tuple, {0}),
                     ValueReplicaCategory::Differing);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(flipped_tuple, {1}),
                     ValueReplicaCategory::Identical);
}
const char* differing_conditional_subcomputations = R"(
HloModule test
other_func {
  x = f32[] parameter(0)
  constant = f32[] constant(-1)
  ROOT multiply = f32[] multiply(x, constant)
}

true_func {
  y = f32[] parameter(0)
  constant = f32[] constant(1)
  ROOT add = f32[] add(y, constant)
}

false_func {
  z = f32[] parameter(0)
  call = f32[] call(z), to_apply=other_func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  constant = f32[] constant(-1)
  ROOT add = f32[] add(call, constant)
}

ENTRY test {
  after-all = token[] after-all()
  infeed = (pred[], token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed1\"\002\001\001"
  differing_pred = pred[] get-tuple-element((pred[], token[]) infeed), index=0

  identical1 = f32[] parameter(0)
  identical2 = f32[] parameter(1)
  conditional = f32[] conditional(differing_pred, identical1, identical2), true_computation=true_func, false_computation=false_func
  ROOT root = f32[] add(identical2, conditional)
}
)";
TEST_F(ReplicaIdenticalDataflowAnalysisTest, ConditionalBranchesValueCategory) {
  ASSERT_TRUE(SetUpHloFlattenedModule(differing_conditional_subcomputations));

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  // Since the predicate is differing we expect all instructions in the
  // conditionals true/false branches to also be differing.
  const auto* true_func = hlo_module_->GetComputationWithName("true_func");
  ASSERT_TRUE(true_func);
  for (auto* instruction : true_func->instructions()) {
    ASSERT_CATEGORY_EQ(analysis.ValueCategory(instruction),
                       ValueReplicaCategory::Differing);
  }

  const auto* false_func = hlo_module_->GetComputationWithName("false_func");
  ASSERT_TRUE(false_func);
  for (auto* instruction : false_func->instructions()) {
    ASSERT_CATEGORY_EQ(analysis.ValueCategory(instruction),
                       ValueReplicaCategory::Differing);
  }

  // Make sure that we recurse through the branches and so reach computations
  // called by them
  const auto* other_func = hlo_module_->GetComputationWithName("other_func");
  ASSERT_TRUE(other_func);
  for (auto* instruction : other_func->instructions()) {
    ASSERT_CATEGORY_EQ(analysis.ValueCategory(instruction),
                       ValueReplicaCategory::Differing);
  }
}
TEST_F(ReplicaIdenticalDataflowAnalysisTest, WhileBodyValueCategory) {
  ASSERT_TRUE(SetUpHloFlattenedModule(while_with_differing_condition.hlo));

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  // Since the while condition is differing then so should the body.
  auto* while_body = hlo_module_->GetComputationWithName("body");
  ASSERT_TRUE(while_body);
  for (auto* instruction : while_body->instructions()) {
    // The instruction bodies should all be differing since
    // the while condition is, as this means that the body
    // can be executed a different number of times by each
    // replica.
    ASSERT_CATEGORY_EQ(analysis.ValueCategory(instruction),
                       ValueReplicaCategory::Differing);
  }
}

const char* unflattened_hlo = R"(
HloModule test
func {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

ENTRY test {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  call1 = f32[] call(param0, param1), to_apply=func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"

  param2 = f32[] parameter(2)
  constant = f32[] constant(2)
  call2 = f32[] call(param2, constant), to_apply=func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"

  ROOT tuple = (f32[], f32[]) tuple(call1, call2)
}
)";
TEST_F(ReplicaIdenticalDataflowAnalysisTest, RequiresFlattenedHLO) {
  ASSERT_TRUE(SetUpHloModule(unflattened_hlo));

  ReplicaIdenticalDataflowAnalysis analysis;

  // Since the HLO graph isn't flattened at this point
  // we should get an error when running the analysis
  const auto analysis_status = analysis.Run(hlo_module_);
  ASSERT_FALSE(analysis_status.ok());

  FlattenCallGraph flatten_call_graph;
  flatten_call_graph.Run(hlo_module_);

  // After flattening it should work as usual
  TF_ASSERT_OK(analysis.Run(hlo_module_));
}

const char* while_mixed_caller_params = R"(
HloModule test
body {
  args = (s32[], s32[]) parameter(0)
  x = s32[] get-tuple-element(args), index=0
  y = s32[] get-tuple-element(args), index=1

  identical0 = s32[] constant(1)
  identical1 = s32[] add(x, identical0)

  ROOT tuple = (s32[], s32[]) tuple(identical0, identical1)
}

condition {
  args = (s32[], s32[]) parameter(0)
  ROOT true_cond = pred[] constant(true)
}

ENTRY test {
  identical0 = s32[] constant(2)

  constant = s32[] constant(2)
  differing0 = s32[] rng(constant, constant), distribution=rng_uniform

  mixed_args = (s32[], s32[]) tuple(identical0, differing0)

  ROOT loop = (s32[], s32[]) while(mixed_args), condition=condition, body=body
}
)";
TEST_F(ReplicaIdenticalDataflowAnalysisTest,
       WhilelIsIdenticalWhereCallerParamsAre) {
  ASSERT_TRUE(SetUpHloFlattenedModule(while_mixed_caller_params));

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  // We only expect the first element of the while loop to be replica
  // identical, even though the loop body produces 2 replica identical
  // values, since the second element of the while argument tuple
  // is replica dependent. This means that depending on the number
  // of iterations we may get differing values.
  auto* while_loop = FindRootInstruction();
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(while_loop, {0}),
                     ValueReplicaCategory::Identical);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(while_loop, {1}),
                     ValueReplicaCategory::Differing);
}

const char* while_mixed_loop_body = R"(
HloModule test
body {
  args = (s32[], s32[]) parameter(0)
  x = s32[] get-tuple-element(args), index=0
  y = s32[] get-tuple-element(args), index=1

  increment = s32[] constant(1)
  identical = s32[] add(y, increment)

  constant = s32[] constant(1)
  differing = s32[] rng(constant, constant), distribution=rng_uniform

  ROOT tuple = (s32[], s32[]) tuple(differing, identical)
}

condition {
  args = (s32[], s32[]) parameter(0)
  ROOT true_cond = pred[] constant(true)
}

ENTRY test {
  identical0 = s32[] constant(2)
  identical1 = s32[] parameter(0)
  identical_args = (s32[], s32[]) tuple(identical0, identical1)

  ROOT loop = (s32[], s32[]) while(identical_args), condition=condition, body=body
}
)";
TEST_F(ReplicaIdenticalDataflowAnalysisTest, WhilelIsIdenticalWhereLoopBodyIs) {
  ASSERT_TRUE(SetUpHloFlattenedModule(while_mixed_loop_body));

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  // We only expect the second element of the while loop to be replica
  // identical, since this is the only replica identical value the loop
  // body produces. Eventhough the initial params are all identical we
  // cant say its identical as we don't know how many iterations it'll
  // run for.
  auto* while_loop = FindRootInstruction();
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(while_loop, {0}),
                     ValueReplicaCategory::Differing);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(while_loop, {1}),
                     ValueReplicaCategory::Identical);
}

const char* while_differing_fixed_point = R"(
HloModule test
body {
  args = (s32[], s32[]) parameter(0)
  x = s32[] get-tuple-element(args), index=0
  y = s32[] get-tuple-element(args), index=1

  increment = s32[] constant(1)
  varying = s32[] add(y, increment)

  constant = s32[] constant(1)
  differing = s32[] rng(constant, constant), distribution=rng_uniform

  ROOT tuple = (s32[], s32[]) tuple(varying, differing)
}

condition {
  args = (s32[], s32[]) parameter(0)
  ROOT true_cond = pred[] constant(true)
}

ENTRY test {
  identical0 = s32[] constant(2)
  identical1 = s32[] parameter(0)
  identical_args = (s32[], s32[]) tuple(identical0, identical1)

  ROOT loop = (s32[], s32[]) while(identical_args), condition=condition, body=body
}
)";
TEST_F(ReplicaIdenticalDataflowAnalysisTest,
       WhilelIsIdenticalWhereLoopFixedPointIs) {
  ASSERT_TRUE(SetUpHloFlattenedModule(while_differing_fixed_point));

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  // We don't expect any elements to be identical since at our loop
  // fixed point (2nd iteration) we produce 2 differing values. We may produce
  // identical values before then but without knowing the number of iterations
  // we cant be sure they'll stay identical.
  auto* root = FindRootInstruction();
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(root, {0}),
                     ValueReplicaCategory::Differing);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(root, {1}),
                     ValueReplicaCategory::Differing);
}

using ReplicaValueCategoryRepeatTest = ParameterizedReplicaDataflowAnalysisTest;

static const HloTestCase repeat_loop_template = {"repeat_loop_template", R"(
HloModule test
repeat {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  constant = f32[] constant(1)
  differing0 = f32[] rng(constant, constant), distribution=rng_uniform
  ROOT tuple = (f32[], f32[]) tuple(y, differing0)
}

ENTRY test {
  identical0 = f32[] parameter(0)
  identical1 = f32[] parameter(1)
  ROOT loop_count = (f32[], f32[]) call(identical0, identical1), to_apply=repeat, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"$0\"}}}"
}
)"};
static const HloTestCase repeat_pipeline_template = {"repeat_pipeline_template",
                                                     R"(
HloModule test
stage_0_fwd {
  x = f32[1,4,4,2] parameter(0)
  constant = f32[] constant(1)
  differing0 = f32[1,4,4,2] rng(constant, constant), distribution=rng_uniform
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(x, differing0)
}

stage_1_fwd {
  x = f32[1,4,4,2] parameter(0)
  ROOT tuple = (f32[1,4,4,2]) tuple(x)
}

pipeline {
  pipeline_input0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_input0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=-1}
  pipeline_stage_0_i1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1

  pipeline_input1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[1,4,4,2]) call(pipeline_input1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_i0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0

  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_1_i0, pipeline_stage_0_i1)
}

ENTRY test {
  input0 = f32[1,4,4,2] parameter(0)
  input1 = f32[1,4,4,2] parameter(1)
  ROOT call = (f32[1,4,4,2], f32[1,4,4,2]) call(input0, input1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipeline_config\":{\"repeatCount\":\"$0\"}}}"
}
)"};
TEST_P(ReplicaValueCategoryRepeatTest, SingleIterationRepeatValueCategory) {
  auto repeat_template = GetParam().hlo;
  auto single_iteration_hlo = absl::Substitute(repeat_template, "1");
  ASSERT_TRUE(SetUpHloFlattenedModule(single_iteration_hlo));

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  // Check that we're driving the call category from the initial
  // caller params (as we're only doing 1 repeat) and not the ROOT of the
  // repeated body, which we would do when iterating more than once.
  // Expects that the HLO be setup so that the value category of element
  // 0 is identical when its generated with the caller params and differing
  // otherwise.
  auto* root = FindRootInstruction();
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(root, {0}),
                     ValueReplicaCategory::Identical);
}

TEST_P(ReplicaValueCategoryRepeatTest, MultiIterationRepeatValueCategory) {
  auto repeat_template = GetParam().hlo;
  auto multi_iteration_hlo = absl::Substitute(repeat_template, "4");
  ASSERT_TRUE(SetUpHloFlattenedModule(multi_iteration_hlo));

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  // Check that we're driving the call category from the repeated
  // body. So the category comes from calling the body with the
  // results of the previous iteration.
  auto* root = FindRootInstruction();
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(root, {0}),
                     ValueReplicaCategory::Differing);
}

struct ReplicaIdenticalUserOpTest : ReplicaIdenticalDataflowAnalysisTest {
  HloModule* BuildModule() {
    auto computation = builder_.Build();
    hlo_module_owner_ = CreateNewVerifiedModule();
    hlo_module_owner_->AddEntryComputation(std::move(computation));
    return hlo_module_owner_.get();
  }

  HloInstruction* AddUserOpInstruction(
      const Shape& shape, HloUserOpInstruction::MetadataFn metadata_fn) {
    return builder_.AddInstruction(CreateUserOp(
        {}, shape, "", nullptr, reinterpret_cast<void*>(metadata_fn), nullptr,
        0, 0, false, ""));
  }

  HloComputation::Builder builder_ = HloComputation::Builder(TestName());
};
TEST_F(ReplicaIdenticalUserOpTest, UserOpArrayDifferingValueCategory) {
  HloUserOpInstruction::MetadataFn empty_indices_metadata{
      [](std::vector<std::int64_t>&,
         std::vector<std::int64_t>& replica_identical_output_indices,
         std::map<std::int64_t, std::int64_t>&, bool&, bool&, bool&,
         std::uint32_t) {
        // an empty replica_identical_output_indices vector should mean that
        // there are no replica identical outputs.
      }};
  HloUserOpInstruction::MetadataFn bad_index_metadata{
      [](std::vector<std::int64_t>&,
         std::vector<std::int64_t>& replica_identical_output_indices,
         std::map<std::int64_t, std::int64_t>&, bool&, bool&, bool&,
         std::uint32_t) { replica_identical_output_indices.push_back(2); }};

  Shape shape = ShapeUtil::MakeShape(F32, {});
  auto* empty_indicies_inst =
      AddUserOpInstruction(shape, empty_indices_metadata);
  auto* bad_index_inst = AddUserOpInstruction(shape, bad_index_metadata);
  auto* default_inst = AddUserOpInstruction(shape, nullptr);
  HloModule* module = BuildModule();

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(module));

  ASSERT_CATEGORY_EQ(analysis.ValueCategory(empty_indicies_inst),
                     ValueReplicaCategory::Differing);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(bad_index_inst),
                     ValueReplicaCategory::Differing);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(default_inst),
                     ValueReplicaCategory::Differing);
}

TEST_F(ReplicaIdenticalUserOpTest, UserOpArrayIdenticalValueCategory) {
  HloUserOpInstruction::MetadataFn valid_index_metadata{
      [](std::vector<std::int64_t>&,
         std::vector<std::int64_t>& replica_identical_output_indices,
         std::map<std::int64_t, std::int64_t>&, bool&, bool&, bool&,
         std::uint32_t) { replica_identical_output_indices.push_back(0); }};

  Shape shape = ShapeUtil::MakeShape(F32, {});
  auto* valid_output_inst = AddUserOpInstruction(shape, valid_index_metadata);
  HloModule* module = BuildModule();

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(module));

  ASSERT_CATEGORY_EQ(analysis.ValueCategory(valid_output_inst),
                     ValueReplicaCategory::Identical);
}

TEST_F(ReplicaIdenticalUserOpTest, UserOpTupleDifferingValueCategory) {
  HloUserOpInstruction::MetadataFn differing_output_metadata{
      [](std::vector<std::int64_t>&,
         std::vector<std::int64_t>& replica_identical_output_indices,
         std::map<std::int64_t, std::int64_t>&, bool&, bool&, bool&,
         std::uint32_t) {
        // an empty replica_identical_output_indices vector should mean that
        // there are no replica identical outputs.
      }};

  Shape inner_shape = ShapeUtil::MakeShape(F32, {});
  Shape outer_shape =
      ShapeUtil::MakeTupleShape({inner_shape, inner_shape, inner_shape});
  auto* differing_user_inst =
      AddUserOpInstruction(outer_shape, differing_output_metadata);
  auto* default_user_inst = AddUserOpInstruction(outer_shape, nullptr);
  HloModule* module = BuildModule();

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(module));

  ASSERT_CATEGORY_EQ(analysis.ValueCategory(differing_user_inst),
                     ValueReplicaCategory::Differing);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(default_user_inst),
                     ValueReplicaCategory::Differing);
}

TEST_F(ReplicaIdenticalUserOpTest, UserOpTupleMixedValueCategory) {
  HloUserOpInstruction::MetadataFn mixed_output_metadata{
      [](std::vector<std::int64_t>&,
         std::vector<std::int64_t>& replica_identical_output_indices,
         std::map<std::int64_t, std::int64_t>&, bool&, bool&, bool&,
         std::uint32_t) {
        replica_identical_output_indices.push_back(0);
        replica_identical_output_indices.push_back(2);
      }};

  Shape inner_shape = ShapeUtil::MakeShape(F32, {});
  Shape outer_shape =
      ShapeUtil::MakeTupleShape({inner_shape, inner_shape, inner_shape});
  auto* mixed_user_inst =
      AddUserOpInstruction(outer_shape, mixed_output_metadata);
  HloModule* module = BuildModule();

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(module));

  ASSERT_CATEGORY_EQ(analysis.ValueCategory(mixed_user_inst),
                     ValueReplicaCategory::Differing);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(mixed_user_inst, {0}),
                     ValueReplicaCategory::Identical);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(mixed_user_inst, {1}),
                     ValueReplicaCategory::Differing);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(mixed_user_inst, {2}),
                     ValueReplicaCategory::Identical);
}

TEST_F(ReplicaIdenticalUserOpTest, UserOpTupleIdenticalValueCategory) {
  HloUserOpInstruction::MetadataFn identical_output_metadata{
      [](std::vector<std::int64_t>&,
         std::vector<std::int64_t>& replica_identical_output_indices,
         std::map<std::int64_t, std::int64_t>&, bool&, bool&, bool&,
         std::uint32_t) {
        replica_identical_output_indices.push_back(0);
        replica_identical_output_indices.push_back(1);
        replica_identical_output_indices.push_back(2);
      }};

  Shape inner_shape = ShapeUtil::MakeShape(F32, {});
  Shape outer_shape =
      ShapeUtil::MakeTupleShape({inner_shape, inner_shape, inner_shape});
  auto* identical_user_inst =
      AddUserOpInstruction(outer_shape, identical_output_metadata);
  HloModule* module = BuildModule();

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(module));

  ASSERT_CATEGORY_EQ(analysis.ValueCategory(identical_user_inst),
                     ValueReplicaCategory::Identical);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(identical_user_inst, {0}),
                     ValueReplicaCategory::Identical);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(identical_user_inst, {1}),
                     ValueReplicaCategory::Identical);
  ASSERT_CATEGORY_EQ(analysis.ValueCategory(identical_user_inst, {2}),
                     ValueReplicaCategory::Identical);
}

INSTANTIATE_TEST_SUITE_P(
    ReplicaIdenticalDataflowHLO, ReplicaIdenticalInstructionTest,
    ::testing::Values(
        simple_parameters, simple_constants, simple_wide_const,
        global_all_reduce, global_all_gather, repeat_with_identical_io,
        repeat_single_element_tuple, while_with_identical_body_and_condition,
        conditional_with_identical_branches_and_pred, simple_pipeline,
        simple_select, simple_tuple_select, compare_with_identical_operands,
        switch_with_identical_branches_and_index,
        assume_equal_identical_operands, assume_equal_differing_operands),
    HloTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    ReplicaIdenticalDataflowHLO, ReplicaDifferingInstructionTest,
    ::testing::Values(
        simple_infeed, partial_all_reduce, partial_all_gather, uniform_rng,
        normal_rng, conditional_with_differing_pred,
        conditional_with_differing_branches, repeat_with_differing_inputs,
        repeat_with_differing_outputs, while_with_differing_condition,
        while_with_differing_body, select_with_differing_pred,
        select_with_differing_values, tuple_select_with_differing_pred,
        tuple_select_with_differing_values, compare_with_differing_operands,
        pipeline_with_differing_gradient_accumulation_count,
        switch_with_differing_index, switch_with_differing_branches,
        dropout_with_seed_op, truncated_normal),
    HloTestCaseName);

INSTANTIATE_TEST_SUITE_P(ReplicaIdenticalDataflowHLO,
                         ReplicaValueCategoryTupleForwardingTest,
                         ::testing::Values(simple_tuple_forwarding,
                                           nested_tuple_forwarding),
                         HloTestCaseName);

INSTANTIATE_TEST_SUITE_P(ReplicaIdenticalDataflowHLO,
                         ReplicaValueCategoryTupleOutputTest,
                         ::testing::Values(identical_tuple_output,
                                           differing_tuple_output),
                         HloTestCaseName);

INSTANTIATE_TEST_SUITE_P(ReplicaIdenticalDataflowHLO,
                         ReplicaValueCategoryRepeatTest,
                         ::testing::Values(repeat_loop_template,
                                           repeat_pipeline_template),
                         HloTestCaseName);
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
