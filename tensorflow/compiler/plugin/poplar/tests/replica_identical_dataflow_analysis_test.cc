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

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
namespace {

using ReplicaIdenticalInstructionTest = ParameterizedHloTestFixture;

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

  auto* root = FindRootInstruction();
  ASSERT_TRUE(root);

  ReplicaIdenticalDataflowAnalysis analysis(root);

  // The root should be replica identical since all the other instructions in
  // the computation that contribute to it are as well.
  TF_ASSERT_OK_AND_ASSIGN(bool is_identical,
                          analysis.IsValueIdenticalAcrossReplicas(root));
  ASSERT_TRUE(is_identical);
}

using ReplicaDifferingInstructionTest = ParameterizedHloTestFixture;

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
TEST_P(ReplicaDifferingInstructionTest,
       UsingNonReplicaIndenticalInstructionsMakesConsumersNonIdentical) {
  CustomOpReplacer custom_op_replacer;
  // We dont assert against the return value of this since it's not relevent
  // for all hlo being tested.
  custom_op_replacer.Run(hlo_module_);

  auto* root = FindRootInstruction();
  ASSERT_TRUE(root);

  ReplicaIdenticalDataflowAnalysis analysis(root);

  // The root should not be replica identical since there are instructions in
  // the graph which are not. Although sub-values of the root shape could still
  // be identical.
  TF_ASSERT_OK_AND_ASSIGN(bool is_identical,
                          analysis.IsValueIdenticalAcrossReplicas(root));
  ASSERT_FALSE(is_identical);
}

using ReplicaValueCategoryGTETest = ParameterizedHloTestFixture;

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
  auto* gte = FindRootInstruction();
  ASSERT_TRUE(gte);
  auto tuple = gte->mutable_operand(0);

  ReplicaIdenticalDataflowAnalysis analysis(gte);

  TF_ASSERT_OK_AND_ASSIGN(
      bool tuple_value_is_identical,
      analysis.IsValueIdenticalAcrossReplicas(tuple, {gte->tuple_index()}));
  TF_ASSERT_OK_AND_ASSIGN(bool gte_is_identical,
                          analysis.IsValueIdenticalAcrossReplicas(gte));

  ASSERT_EQ(gte_is_identical, tuple_value_is_identical);
}

using ReplicaValueCategoryTupleForwardingTest = ParameterizedHloTestFixture;

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
  auto* tuple = FindRootInstruction();
  ASSERT_TRUE(tuple);

  auto leaf_shapes = ShapeUtil::GetLeafShapes(tuple->shape());
  ASSERT_EQ(leaf_shapes.size(), 3);

  ReplicaIdenticalDataflowAnalysis analysis(tuple);

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

using ReplicaValueCategoryTupleTest = HloTestFixture;
using ReplicaValueCategoryTupleOutputTest = ParameterizedHloTestFixture;

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
  auto* tuple = FindRootInstruction();
  ASSERT_TRUE(tuple);

  ReplicaIdenticalDataflowAnalysis analysis(tuple);

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

using ReplicaIdenticalDataflowAnalysisTest = HloTestFixture;
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
  ASSERT_TRUE(SetUpHloModule(multi_computation_hlo));

  auto* root = FindRootInstruction();
  ASSERT_TRUE(root);

  ReplicaIdenticalDataflowAnalysis analysis(root);

  const auto status_or_value_category =
      analysis.ValueCategory(FindInstruction(hlo_module_, "x"));
  ASSERT_FALSE(status_or_value_category.ok());

  const auto status_or_identical = analysis.IsValueIdenticalAcrossReplicas(
      FindInstruction(hlo_module_, "y"));
  ASSERT_FALSE(status_or_identical.ok());
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
                      normal_rng_is_replica_differing),
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
