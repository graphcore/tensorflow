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

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/replica_identical_dataflow_analysis.h"
#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"

#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

// Test fixtures for creating flattened HLO, which is what
// the ReplicaIdenticalDataflowAnalysis class expects.
struct ReplicaIdenticalDataflowAnalysisTest : HloTestFixture {
  ::testing::AssertionResult SetUpHloFlattenedModule(const std::string& hlo) {
    auto module_setup_result = HloTestFixture::SetUpHloModule(hlo);
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
    ASSERT_TRUE(SetUpHloFlattenedModule(GetParam().hlo));
  }
};

using ReplicaIdenticalInstructionTest =
    ParameterizedReplicaDataflowAnalysisTest;

static const HloTestCase parameters_are_replica_identical = {"parameters", R"(
HloModule test
ENTRY test {
   param0 = f32[1,1,2,4] parameter(0)
   param1 = f32[1,1,2,4] parameter(1)
   ROOT identical_root = f32[1,1,2,4] add(param0, param1)
}
)"};
static const HloTestCase constants_are_replica_identical{"constants", R"(
HloModule test
ENTRY test {
   const1 = f32[] constant(0)
   const2 = f32[] constant(4)
   ROOT identical_root = f32[] multiply(const1, const2)
}
)"};
static const HloTestCase consumers_become_replica_identical = {"consumers", R"(
HloModule test
ENTRY test {
   identical0 = f32[] parameter(0)
   identical1 = f32[] parameter(1)
   consumer0 = f32[1, 1, 2, 4]  broadcast(f32[] identical0), dimensions={}, backend_config="{}"
   consumer1 = f32[1, 1, 2, 4]  broadcast(f32[] identical1), dimensions={}, backend_config="{}"
   ROOT identical_root = f32[1,1,2,4] add(consumer0, consumer1)
}
)"};
static const HloTestCase wide_consts_are_replica_identical = {"wide_const", R"(
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
static const HloTestCase global_all_reduce_is_replica_identical = {
    "global_all_reduce", R"(
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
static const HloTestCase global_all_gather_is_replica_identical = {
    "global_all_gather", R"(
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
TEST_P(ReplicaIdenticalInstructionTest,
       UsingReplicaIdenticalInstructionsMakesConsumerReplicaIdentical) {
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
  ASSERT_TRUE(is_identical);
}

using ReplicaDifferingInstructionTest =
    ParameterizedReplicaDataflowAnalysisTest;

static const HloTestCase infeed_is_replica_differing = {"infeed", R"(
HloModule test
ENTRY test {
   identical0 = f32[1, 1, 2, 4] parameter(0)
   after-all = token[] after-all()
   infeed = (f32[1,1,2,4], token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed1\"\002\001\001"
   infeed_value = f32[1,1,2,4] get-tuple-element((f32[1,1,2,4], token[]) infeed), index=0
   ROOT differing_root = f32[1,1,2,4] add(identical0, infeed_value)
}
)"};
static const HloTestCase partial_all_reduce_is_replica_differing = {
    "partial_all_reduce", R"(
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
)"};
static const HloTestCase partial_all_gather_is_replica_differing = {
    "partial_all_gather", R"(
HloModule test
ENTRY test {
  identical0 = f32[4] parameter(0)
  identical1 = f32[2] parameter(1)
  partial_all_gather = f32[4] custom-call(identical1), custom_call_target="AllGather", backend_config="{\"replica_group_size\":2}"
  ROOT differing_root = f32[4] add(identical0, partial_all_gather)
}
)"};
static const HloTestCase uniform_rng_is_replica_differing = {"uniform_rng", R"(
HloModule test
ENTRY test {
  identical0 = f32[] constant(0)
  identical1 = f32[] constant(1)
  identical2 = f32[8]  broadcast(f32[] identical0), dimensions={}, backend_config="{}"
  rng_uniform = f32[8] rng(identical0, identical1), distribution=rng_uniform
  ROOT differing_root = f32[8] add(identical2, rng_uniform)
}
)"};
static const HloTestCase normal_rng_is_replica_differing = {"normal_rng", R"(
HloModule test
ENTRY test {
  identical0 = f32[] constant(0)
  identical1 = f32[] constant(1)
  identical2 = f32[8]  broadcast(f32[] identical0), dimensions={}, backend_config="{}"
  rng_normal = f32[8] rng(identical0, identical1), distribution=rng_normal
  ROOT differing_root = f32[8] add(identical2, rng_normal)
}
)"};
static const HloTestCase conditional_is_replica_differing = {"conditional", R"(
HloModule test
identical_func {
  ROOT x = f32[] parameter(0)
}

ENTRY test {
  identical0 = pred[] constant(0)
  identical1 = f32[] parameter(0)
  identical2 = f32[] parameter(1)
  conditional = f32[] conditional(identical0, identical1, identical2), true_computation=identical_func, false_computation=identical_func
  ROOT differing_root = f32[] add(identical2, conditional)
}
)"};
TEST_P(ReplicaDifferingInstructionTest,
       UsingNonReplicaIndenticalInstructionsMakesConsumersNonIdentical) {
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
  ASSERT_FALSE(is_identical);
}

using ReplicaValueCategoryGTETest = ParameterizedReplicaDataflowAnalysisTest;

static const HloTestCase get_differing_element = {"get_tuple_differing", R"(
HloModule test
ENTRY test {
   identical0 = f32[] parameter(0)
   after-all = token[] after-all()
   differing0 = (f32[], token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed1\"\002\001\001"
   mixed_tuple = ((f32[], token[]), f32[]) tuple(differing0, identical0)
   ROOT differing_root = (f32[], token[]) get-tuple-element(mixed_tuple), index=0
}
)"};
static const HloTestCase get_identical_element = {"get_tuple_identical", R"(
HloModule test
ENTRY test {
   identical0 = f32[] parameter(0)
   after-all = token[] after-all()
   differing0 = (f32[], token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed1\"\002\001\001"
   mixed_tuple = ((f32[], token[]), f32[]) tuple(differing0, identical0)
   ROOT identical_root = f32[] get-tuple-element(mixed_tuple), index=1
}
)"};
TEST_P(ReplicaValueCategoryGTETest, GetTupleElementIsIdenticalIfTupleIndexIs) {
  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  auto* gte = FindRootInstruction();
  auto tuple = gte->mutable_operand(0);

  TF_ASSERT_OK_AND_ASSIGN(
      bool tuple_value_is_identical,
      analysis.IsValueIdenticalAcrossReplicas(tuple, {gte->tuple_index()}));
  TF_ASSERT_OK_AND_ASSIGN(bool gte_is_identical,
                          analysis.IsValueIdenticalAcrossReplicas(gte));

  ASSERT_EQ(gte_is_identical, tuple_value_is_identical);
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
)"};
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
)"};
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

  TF_ASSERT_OK_AND_ASSIGN(auto original_value_category,
                          visitor.ValueCategory(root, RootShapeIndex()));

  ValuesIdenticalAcrossReplicasVisitor visitor_with_override(
      {{root,
        ValueCategoryTree(root->shape(), ValueReplicaCategory::Differing)}});
  root->Accept(&visitor_with_override);

  TF_ASSERT_OK_AND_ASSIGN(
      auto override_value_category,
      visitor_with_override.ValueCategory(root, RootShapeIndex()));
  ASSERT_EQ(override_value_category, ValueReplicaCategory::Differing);
  ASSERT_NE(override_value_category, original_value_category);
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
TEST_F(ReplicaIdenticalDataflowAnalysisTest,
       ParamlessCallHasCategoryOfCalledComputation) {
  ASSERT_TRUE(SetUpHloFlattenedModule(paramless_callables_hlo));

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  auto* identical_call = FindInstruction(hlo_module_, "identical_call");
  auto* differing_call = FindInstruction(hlo_module_, "differing_call");
  ASSERT_TRUE(identical_call);
  ASSERT_TRUE(differing_call);

  TF_ASSERT_OK_AND_ASSIGN(auto identical_call_category,
                          analysis.ValueCategory(identical_call));
  TF_ASSERT_OK_AND_ASSIGN(auto differing_call_category,
                          analysis.ValueCategory(differing_call));

  ASSERT_EQ(identical_call_category, ValueReplicaCategory::Identical);
  ASSERT_EQ(differing_call_category, ValueReplicaCategory::Differing);
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
TEST_F(ReplicaIdenticalDataflowAnalysisTest, CallableCategoryDependsOnParams) {
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
  TF_ASSERT_OK_AND_ASSIGN(auto call_with_identical_params_category,
                          analysis.ValueCategory(identical_call0));
  TF_ASSERT_OK_AND_ASSIGN(auto call_with_differing_params_category,
                          analysis.ValueCategory(differing_call0));
  TF_ASSERT_OK_AND_ASSIGN(auto call_with_mixed_params_category,
                          analysis.ValueCategory(mixed_call0));

  ASSERT_EQ(call_with_identical_params_category,
            ValueReplicaCategory::Identical);
  ASSERT_EQ(call_with_differing_params_category,
            ValueReplicaCategory::Differing);
  ASSERT_EQ(call_with_mixed_params_category, ValueReplicaCategory::Differing);
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
TEST_F(ReplicaIdenticalDataflowAnalysisTest, CallableVisitedWithParams) {
  ASSERT_TRUE(SetUpHloFlattenedModule(chained_calls_hlo));

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  auto* root = FindRootInstruction();
  // Since the function being called is replica identical the categories of
  // the callers output values depend on the params its called with.
  TF_ASSERT_OK_AND_ASSIGN(auto differing0_param_category,
                          analysis.ValueCategory(root, {0}));
  TF_ASSERT_OK_AND_ASSIGN(auto identical0_param_category,
                          analysis.ValueCategory(root, {1}));

  ASSERT_EQ(differing0_param_category, ValueReplicaCategory::Differing);
  ASSERT_EQ(identical0_param_category, ValueReplicaCategory::Identical);
}

TEST_F(ReplicaIdenticalDataflowAnalysisTest, CanQuerySubComputations) {
  ASSERT_TRUE(SetUpHloFlattenedModule(chained_calls_hlo));

  ReplicaIdenticalDataflowAnalysis analysis;
  TF_ASSERT_OK(analysis.Run(hlo_module_));

  auto* flipped_tuple = FindInstruction(hlo_module_, "flipped_tuple");

  TF_ASSERT_OK_AND_ASSIGN(auto differing0_param_category,
                          analysis.ValueCategory(flipped_tuple, {0}));
  TF_ASSERT_OK_AND_ASSIGN(auto identical0_param_category,
                          analysis.ValueCategory(flipped_tuple, {1}));

  ASSERT_EQ(differing0_param_category, ValueReplicaCategory::Differing);
  ASSERT_EQ(identical0_param_category, ValueReplicaCategory::Identical);
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

INSTANTIATE_TEST_SUITE_P(
    ReplicaIdenticalDataflowHLO, ReplicaIdenticalInstructionTest,
    ::testing::Values(parameters_are_replica_identical,
                      constants_are_replica_identical,
                      consumers_become_replica_identical,
                      wide_consts_are_replica_identical,
                      global_all_reduce_is_replica_identical,
                      global_all_gather_is_replica_identical),
    HloTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    ReplicaIdenticalDataflowHLO, ReplicaDifferingInstructionTest,
    ::testing::Values(infeed_is_replica_differing,
                      partial_all_reduce_is_replica_differing,
                      partial_all_gather_is_replica_differing,
                      uniform_rng_is_replica_differing,
                      normal_rng_is_replica_differing,
                      conditional_is_replica_differing),
    HloTestCaseName);

INSTANTIATE_TEST_SUITE_P(ReplicaIdenticalDataflowHLO,
                         ReplicaValueCategoryGTETest,
                         ::testing::Values(get_differing_element,
                                           get_identical_element),
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

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
