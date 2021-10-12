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

#include "tensorflow/compiler/plugin/poplar/driver/passes/replicated_resource_update_elementwise_clustering.h"

#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fusion_inliner.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/variables_offload_and_partition.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/reduce_scatter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/elementwise_cluster.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_replica_groups.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {

using tu = HloPoplarTestUtil;

namespace {

StatusOr<HloInstruction*> GetNextUser(HloInstruction* inst) {
  if (inst->user_count() != 1) {
    return FailedPrecondition("Expected single user.");
  }
  return inst->users()[0];
}

StatusOr<HloInstruction*> LookThroughReshape(HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kReshape ? GetNextUser(inst) : inst;
}

HloInstruction* MaybeReshapeOf(HloInstruction* inst) {
  for (HloInstruction* user : inst->users()) {
    if (user->opcode() == HloOpcode::kReshape) {
      return user;
    }
  }
  return inst;
}

StatusOr<HloInstruction*> GetRewrittenInput(HloInstruction* inst, bool padded) {
  TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
  TF_ASSIGN_OR_RETURN(inst, LookThroughReshape(inst));
  if (padded) {
    EXPECT_THAT(inst->opcode(), HloOpcode::kPad);
    TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
  }
  TF_ASSIGN_OR_RETURN(inst, LookThroughReshape(inst));
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::CollectiveReorder)(inst));
  TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ReduceScatter)(inst));
  return inst;
}

StatusOr<HloInstruction*> GetScatterInput(HloInstruction* inst, bool padded) {
  TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
  TF_ASSIGN_OR_RETURN(inst, LookThroughReshape(inst));
  if (padded) {
    EXPECT_THAT(inst->opcode(), HloOpcode::kPad);
    TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
  }
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::CollectiveReorder)(inst));
  TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ReduceScatter)(inst));
  return inst;
}

StatusOr<HloInstruction*> GetAllReduceInput(HloInstruction* inst) {
  TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
  TF_ASSIGN_OR_RETURN(inst, LookThroughReshape(inst));
  EXPECT_THAT(inst->opcode(), HloOpcode::kAllReduce);
  return inst;
}

StatusOr<HloInstruction*> GetRewrittenOutput(HloInstruction* inst,
                                             bool sliced) {
  TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::AllGather)(inst));
  TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::UndoCollectiveReorder)(inst));
  TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
  EXPECT_THAT(inst->opcode(), HloOpcode::kReshape);
  if (sliced) {
    TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
    EXPECT_THAT(inst->opcode(), HloOpcode::kSlice);
  }
  return inst;
}

class ReplicatedResourceUpdateElementwiseClusteringBasicTest
    : public HloTestBase,
      public ::testing::WithParamInterface<bool> {};

TEST_P(ReplicatedResourceUpdateElementwiseClusteringBasicTest,
       TestElementwiseComputations) {
  const string& hlo_string = R"(
HloModule main

_comp0 {
  p0 = f32[10] parameter(0)
  p1 = f32[10] parameter(1)
  a0 = f32[10] add(p0, p1)
  p2 = f32[10] parameter(2)
  a1 = f32[10] add(a0, p2)
  p3 = f32[10] parameter(3)
  ROOT a2 = f32[10] add(a1, p3)
}

_comp1 {
  arg_0.1 = f16[1024,3000] parameter(0)
  arg_1.1 = f16[3000] parameter(1)
  broadcast.18.clone = f16[1024,3000] broadcast(arg_1.1), dimensions={1}
  ROOT add.19.clone = f16[1024,3000] add(arg_0.1, broadcast.18.clone)
}

_comp2 {
  arg_0.2 = f32[128,3000] parameter(0)
  arg_1.2 = f32[128,3000] parameter(1)
  constant.103.clone = f32[] constant(0.001)
  broadcast.138.clone = f32[128,3000] broadcast(constant.103.clone), dimensions={}
  multiply.175.clone = f32[128,3000] multiply(arg_1.2, broadcast.138.clone)
  ROOT subtract.176.clone = f32[128,3000] subtract(arg_0.2, multiply.175.clone)
}

ENTRY main {
  ROOT t = () tuple()
}
)";
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  CompilerAnnotations annotations(module.get());
  ReplicatedResourceUpdateElementwiseClustering pass(annotations, 1);
  auto elementwise_comps =
      ElementwiseCluster::GetElementwiseClusterableComputations(module.get());
  TF_ASSERT_OK(pass.RunDataflowAnalysis(module.get()));
  CHECK_EQ(elementwise_comps.size(), 2);
  absl::flat_hash_set<std::string> elementwise_comp_names = {"_comp0",
                                                             "_comp2"};
  for (auto comp : elementwise_comps) {
    EXPECT_TRUE(elementwise_comp_names.contains(comp->name()));
  }
}

TEST_P(ReplicatedResourceUpdateElementwiseClusteringBasicTest,
       TestMultipleClustersSharedScalar) {
  const bool replica_partition = GetParam();
  const std::string hlo = R"(
  HloModule main

  sum {
    y = f16[] parameter(1)
    x = f16[] parameter(0), control-predecessors={y}
    ROOT add = f16[] add(x, y), backend_config="{\"isInplace\":true}"
  }

  resource_update {
    arg0 = f16[128] parameter(0)
    arg1 = f16[128] parameter(1)
    arg0_r = f16[128] all-reduce(arg0), to_apply=sum
    arg1_r = f16[128] all-reduce(arg1), to_apply=sum

    arg2 = f16[128] parameter(2)
    arg3 = f16[128] parameter(3)

    arg4 = f16[128] parameter(4)
    arg5 = f16[128] parameter(5)

    arg6 = f16[] parameter(6)

    arg4_new = f16[128] add(arg0_r, arg4)
    arg5_new = f16[128] add(arg1_r, arg5)

    bcast = f16[128] broadcast(arg6), dimensions={}

    arg4_mul = f16[128] multiply(bcast, arg4_new)
    arg5_mul = f16[128] multiply(bcast, arg5_new)

    arg2_new = f16[128] add(arg2, arg4_mul)
    arg3_new = f16[128] add(arg3, arg5_mul)

    ROOT t = (f16[128],f16[128],f16[128],f16[128]) tuple(arg2_new, arg3_new, arg4_new, arg5_new)

    counter_0 = s32[] constant(4)
    gac = () custom-call(s32[] counter_0), custom_call_target="GradientAccumulationCount"
  }

  loop {
    after-all = token[] after-all()
    infeed = (f16[128], token[]) infeed(after-all), infeed_config="140121807314576"
    input = f16[128] get-tuple-element(infeed), index=0

    l.arg0 = f16[128] parameter(0)
    l.arg1 = f16[128] parameter(1)
    l.arg2 = f16[128] parameter(2)
    l.arg3 = f16[128] parameter(3)
    l.arg4 = f16[] parameter(4)

    add.1 = f16[128] add(input, l.arg0)
    add.2 = f16[128] add(add.1, l.arg1)

    call = (f16[128],f16[128],f16[128],f16[128]) call(add.1, add.2, l.arg0, l.arg1, l.arg2, l.arg3, l.arg4), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\", \"partitionOffloadedVariables\":\"THREESTATE_OFF\"}}}"
    gte0 = f16[128] get-tuple-element(call), index=0
    gte1 = f16[128] get-tuple-element(call), index=1
    gte2 = f16[128] get-tuple-element(call), index=2
    gte3 = f16[128] get-tuple-element(call), index=3
    ROOT r = (f16[128],f16[128],f16[128],f16[128],f16[]) tuple(gte0, gte1, gte2, gte3, l.arg4)
  }

  ENTRY e {
    e.in0 = f16[128] parameter(0)
    e.in1 = f16[128] parameter(1)
    e.in2 = f16[128] parameter(2)
    e.in3 = f16[128] parameter(3)
    e.in4 = f16[] parameter(4)
    loop_call = (f16[128],f16[128],f16[128],f16[128],f16[]) call(e.in0, e.in1, e.in2, e.in3, e.in4), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
    gte0 = f16[128] get-tuple-element(loop_call), index=0
    gte1 = f16[128] get-tuple-element(loop_call), index=1
    gte2 = f16[128] get-tuple-element(loop_call), index=2
    gte3 = f16[128] get-tuple-element(loop_call), index=3
    ROOT r = (f16[128],f16[128],f16[128],f16[128]) tuple(gte0, gte1, gte2, gte3)
  }
  )";

  auto config = GetModuleConfigForTest();
  config.set_argument_input_indices({4});
  config.set_resource_input_indices({0, 1, 2, 3});
  config.set_resource_input_initialized({true, true, true, true});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  HloInstruction* loop = FindInstruction(module.get(), "loop_call");

  HloInstruction* bcast = FindInstruction(module.get(), "bcast");

  std::vector<HloInstruction*> all_reduces = {
      FindInstruction(module.get(), "arg0_r"),
      FindInstruction(module.get(), "arg1_r")};
  std::vector<HloInstruction*> inputs = {FindInstruction(module.get(), "arg2"),
                                         FindInstruction(module.get(), "arg3")};
  std::vector<HloInstruction*> offloaded_inputs = {
      FindInstruction(module.get(), "arg4"),
      FindInstruction(module.get(), "arg5")};
  std::vector<HloInstruction*> offloaded_inputs_new = {
      FindInstruction(module.get(), "arg4_new"),
      FindInstruction(module.get(), "arg5_new")};
  std::vector<HloInstruction*> offloaded_inputs_mul = {
      FindInstruction(module.get(), "arg4_mul"),
      FindInstruction(module.get(), "arg5_mul")};
  std::vector<HloInstruction*> inputs_new = {
      FindInstruction(module.get(), "arg2_new"),
      FindInstruction(module.get(), "arg3_new")};

  CompilerAnnotations annotations(module.get());
  TF_ASSERT_OK_AND_ASSIGN(
      bool offloaded,
      VariablesOffloadAndPartition(annotations, true, 4, 2).Run(module.get()));
  EXPECT_TRUE(offloaded);

  std::unique_ptr<ResourceUpdateElementwiseClustering> pass_ptr(
      replica_partition
          ? new ReplicatedResourceUpdateElementwiseClustering(annotations, 2)
          : new ResourceUpdateElementwiseClustering);
  auto& pass = *pass_ptr;
  auto elementwise_comps =
      ElementwiseCluster::GetElementwiseClusterableComputations(module.get());
  TF_ASSERT_OK(pass.RunDataflowAnalysis(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto clusters,
                          pass.GetClustersIn(loop, elementwise_comps));
  ASSERT_THAT(clusters.size(), 2);

  for (int64 i = 0; i != 2; i++) {
    auto& cluster = clusters[1 - i];
    HloInstruction* offloaded_reshape =
        offloaded_inputs_new[i]->mutable_operand(1);
    EXPECT_THAT(cluster.GetInputs(),
                ::testing::UnorderedElementsAre(all_reduces[i], inputs[i],
                                                bcast, offloaded_reshape));
    EXPECT_THAT(
        cluster.GetPostOrder(),
        ::testing::UnorderedElementsAre(
            offloaded_inputs_new[i], offloaded_inputs_mul[i], inputs_new[i]));
    EXPECT_THAT(cluster.GetOutputs(),
                ::testing::UnorderedElementsAre(offloaded_inputs_new[i],
                                                inputs_new[i]));
  }
}

TEST_P(ReplicatedResourceUpdateElementwiseClusteringBasicTest,
       TestScalarConstant) {
  const bool replica_partition = GetParam();
  constexpr absl::string_view hlo_format = R"(
  HloModule main

  sum {
    y = f16[] parameter(1)
    x = f16[] parameter(0), control-predecessors={y}
    ROOT add = f16[] add(x, y), backend_config="{\"isInplace\":true}"
  }

  resource_update {
    arg0 = f16[128] parameter(0)
    arg1 = f16[128] parameter(1)
    arg0_r = f16[128] all-reduce(arg0), to_apply=sum
    arg2 = f16[128] parameter(2)
    arg2_new = f16[128] add(arg0_r, arg2)
    arg3 = f16[] parameter(3)
    bcast = f16[128] broadcast(arg3), dimensions={}
    arg3_arg2_mul = f16[128] multiply(bcast, arg2_new)
    arg1_new = f16[128] add(arg1, arg3_arg2_mul)
    ROOT t = (f16[128],f16[128]) tuple(arg1_new, arg2_new)
    counter_0 = s32[] constant(4)
    gac = () custom-call(s32[] counter_0), custom_call_target="GradientAccumulationCount"
  }

  loop {
    after-all = token[] after-all()
    infeed = (f16[128], token[]) infeed(after-all), infeed_config="140121807314576"
    input = f16[128] get-tuple-element(infeed), index=0

    arg0 = f16[128] parameter(0)
    arg1 = f16[128] parameter(1)
    arg2 = f16[] parameter(2)

    add.1 = f16[128] add(input, arg0)
    call = (f16[128],f16[128]) call(add.1, arg0, arg1, arg2), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\", \"partitionOffloadedVariables\":\"%s\"}}}"
    gte0 = f16[128] get-tuple-element(call), index=0
    gte1 = f16[128] get-tuple-element(call), index=1
    ROOT r = (f16[128],f16[128],f16[]) tuple(gte0, gte1, arg2)
  }

  ENTRY e {
    e.in0 = f16[128] parameter(0)
    e.in1 = f16[128] parameter(1)
    e.in2 = f16[] parameter(2)
    loop_call = (f16[128],f16[128],f16[]) call(e.in0, e.in1, e.in2), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
    gte0 = f16[128] get-tuple-element(loop_call), index=0
    gte1 = f16[128] get-tuple-element(loop_call), index=1
    ROOT r = (f16[128],f16[128]) tuple(gte0, gte1)
  }
  )";
  const std::string hlo =
      absl::StrFormat(hlo_format, ThreeState_Name(replica_partition));

  auto config = GetModuleConfigForTest();
  config.set_argument_input_indices({2});
  config.set_resource_input_indices({0, 1});
  config.set_resource_input_initialized({true, true});
  config.set_resource_update_to_input_index({0, 1});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  HloInstruction* loop = FindInstruction(module.get(), "loop_call");

  HloInstruction* arg0 = FindInstruction(module.get(), "arg0");
  HloInstruction* arg0_r = FindInstruction(module.get(), "arg0_r");
  HloInstruction* arg1 = FindInstruction(module.get(), "arg1");
  HloInstruction* arg2 = FindInstruction(module.get(), "arg2");
  HloInstruction* arg2_new = FindInstruction(module.get(), "arg2_new");
  HloInstruction* arg3 = FindInstruction(module.get(), "arg3");
  HloInstruction* bcast = FindInstruction(module.get(), "bcast");
  HloInstruction* arg3_arg2_mul =
      FindInstruction(module.get(), "arg3_arg2_mul");
  HloInstruction* arg1_new = FindInstruction(module.get(), "arg1_new");

  CompilerAnnotations annotations(module.get());
  TF_ASSERT_OK_AND_ASSIGN(
      bool offloaded,
      VariablesOffloadAndPartition(annotations, true, 4, 2).Run(module.get()));
  EXPECT_TRUE(offloaded);

  std::unique_ptr<ResourceUpdateElementwiseClustering> pass_ptr(
      replica_partition
          ? new ReplicatedResourceUpdateElementwiseClustering(annotations, 2)
          : new ResourceUpdateElementwiseClustering);
  auto& pass = *pass_ptr;
  auto elementwise_comps =
      ElementwiseCluster::GetElementwiseClusterableComputations(module.get());
  TF_ASSERT_OK(pass.RunDataflowAnalysis(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto clusters,
                          pass.GetClustersIn(loop, elementwise_comps));
  ASSERT_THAT(clusters.size(), 1);

  const int64 shard_size = replica_partition ? 64 : 128;
  auto& cluster = *std::begin(clusters);
  EXPECT_THAT(cluster.GetClusterSize(), 128);
  EXPECT_THAT(cluster.GetAlignedClusterSize(), 128);
  EXPECT_THAT(cluster.GetShardSize(), shard_size);
  HloInstruction* arg2_new_reshape = arg2_new->mutable_operand(1);
  EXPECT_THAT(cluster.GetInputs(), ::testing::UnorderedElementsAre(
                                       arg0_r, arg1, arg2_new_reshape, bcast));
  EXPECT_THAT(cluster.GetPostOrder(), ::testing::UnorderedElementsAre(
                                          arg2_new, arg3_arg2_mul, arg1_new));
  EXPECT_THAT(cluster.GetOutputs(),
              ::testing::UnorderedElementsAre(arg1_new, arg2_new));

  // Convert the cluster.
  TF_ASSERT_OK(pass.OutlineCluster(cluster).status());
  TF_ASSERT_OK_AND_ASSIGN(bool eliminated, HloDCE().Run(module.get()));

  HloInstruction *arg2_load, *arg2_store;
  TF_ASSERT_OK(GetRemoteLoadStoreUsers(arg2, &arg2_load, &arg2_store));

  EXPECT_THAT(arg0->user_count(), 1);
  HloInstruction* cluster_call = GetNextUser(arg0).ValueOrDie();
  EXPECT_TRUE(IsFunction(cluster_call));
  HloComputation* cluster_comp = cluster_call->to_apply();
  EXPECT_THAT(cluster_call->operands(),
              ::testing::ElementsAre(arg1, arg0, arg3, arg2_load));

  arg1 = cluster_comp->parameter_instruction(0);
  if (replica_partition) {
    TF_ASSERT_OK_AND_ASSIGN(arg1, GetRewrittenInput(arg1, false));
  }
  EXPECT_THAT(arg1->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  HloInstruction* collective = cluster_comp->parameter_instruction(1);
  if (replica_partition) {
    TF_ASSERT_OK_AND_ASSIGN(collective, GetScatterInput(collective, false));
  } else {
    TF_ASSERT_OK_AND_ASSIGN(collective, GetAllReduceInput(collective));
  }
  EXPECT_THAT(collective->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  arg3 = cluster_comp->parameter_instruction(2);
  EXPECT_THAT(arg3->shape(), ShapeUtil::MakeShape(F16, {}));

  arg2_load = cluster_comp->parameter_instruction(3);
  EXPECT_THAT(arg2_load->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  EXPECT_THAT(arg2_load->user_count(), 1);
  arg2_new = arg2_load->users()[0];
  EXPECT_THAT(arg2_new->operands(),
              ::testing::ElementsAre(collective, arg2_load));
  EXPECT_THAT(arg2_new->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  EXPECT_THAT(arg3->user_count(), 1);
  bcast = arg3->users()[0];
  EXPECT_THAT(bcast->operands(), ::testing::ElementsAre(arg3));
  EXPECT_THAT(bcast->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  EXPECT_THAT(bcast->user_count(), 1);
  arg3_arg2_mul = bcast->users()[0];
  EXPECT_THAT(arg3_arg2_mul->operands(),
              ::testing::ElementsAre(bcast, arg2_new));
  EXPECT_THAT(arg3_arg2_mul->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  EXPECT_THAT(arg1->user_count(), 1);
  arg1_new = arg1->users()[0];
  EXPECT_THAT(arg1_new->operands(),
              ::testing::ElementsAre(arg1, arg3_arg2_mul));
  EXPECT_THAT(arg1_new->shape(), ShapeUtil::MakeShape(F16, {shard_size}));
  if (replica_partition) {
    TF_ASSERT_OK_AND_ASSIGN(arg1_new, GetRewrittenOutput(arg1_new, false));
  }
  EXPECT_THAT(arg1_new->shape(), ShapeUtil::MakeShape(F16, {128}));

  if (replica_partition) {
    arg1_new = MaybeReshapeOf(arg1_new);
    arg2_new = MaybeReshapeOf(arg2_new);
    EXPECT_TRUE(Match(cluster_comp->root_instruction(),
                      m::Tuple(m::Op().Is(arg2_new), m::Op().Is(arg1_new))));
    EXPECT_TRUE(
        Match(arg2_store,
              m::CustomCall(m::Op().Is(arg2),
                            m::GetTupleElement(m::Op().Is(cluster_call), 0))));
  } else {
    EXPECT_TRUE(Match(cluster_comp->root_instruction(),
                      m::Tuple(m::Op().Is(arg2_new), m::Op().Is(arg1_new))));
    EXPECT_TRUE(
        Match(arg2_store,
              m::CustomCall(m::Op().Is(arg2),
                            m::GetTupleElement(m::Op().Is(cluster_call), 0))));
  }

  HloComputation* resource_update =
      FindComputation(module.get(), "resource_update");
  EXPECT_TRUE(Match(resource_update->root_instruction(),
                    m::Tuple(m::GetTupleElement(m::Op().Is(cluster_call), 1),
                             m::Op().Is(arg2_store))));
}

TEST_P(ReplicatedResourceUpdateElementwiseClusteringBasicTest,
       TestScalarConstantUpdated) {
  const bool replica_partition = GetParam();
  constexpr absl::string_view hlo_format = R"(
  HloModule main

  sum {
    y = f16[] parameter(1)
    x = f16[] parameter(0), control-predecessors={y}
    ROOT add = f16[] add(x, y), backend_config="{\"isInplace\":true}"
  }

  resource_update {
    arg0 = f16[128] parameter(0)
    arg1 = f16[128] parameter(1)
    arg0_r = f16[128] all-reduce(arg0), to_apply=sum
    arg2 = f16[128] parameter(2)
    arg2_new = f16[128] add(arg0_r, arg2)
    arg3 = f16[] parameter(3)
    const = f16[] constant(0.1)
    arg3_new = f16[] multiply(arg3, const)
    bcast = f16[128] broadcast(arg3_new), dimensions={}
    arg3_arg2_mul = f16[128] multiply(bcast, arg2_new)
    arg1_new = f16[128] add(arg1, arg3_arg2_mul)
    ROOT t = (f16[128],f16[128],f16[]) tuple(arg1_new, arg2_new, arg3_new)
    counter_0 = s32[] constant(4)
    gac = () custom-call(s32[] counter_0), custom_call_target="GradientAccumulationCount"
  }

  loop {
    after-all = token[] after-all()
    infeed = (f16[128], token[]) infeed(after-all),
    infeed_config="140121807314576" input = f16[128]
    get-tuple-element(infeed), index=0

    arg0 = f16[128] parameter(0)
    arg1 = f16[128] parameter(1)
    arg2 = f16[] parameter(2)

    add.1 = f16[128] add(input, arg0)
    call = (f16[128],f16[128],f16[]) call(add.1, arg0, arg1, arg2), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\", \"partitionOffloadedVariables\":\"%s\"}}}"
    gte0 = f16[128] get-tuple-element(call), index=0
    gte1 = f16[128] get-tuple-element(call), index=1
    gte2 = f16[] get-tuple-element(call), index=2
    ROOT r = (f16[128],f16[128],f16[]) tuple(gte0, gte1, gte2)
  }

  ENTRY e {
    e.in0 = f16[128] parameter(0)
    e.in1 = f16[128] parameter(1)
    e.in2 = f16[] parameter(2)
    loop_call = (f16[128],f16[128],f16[]) call(e.in0, e.in1, e.in2), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
    gte0 = f16[128] get-tuple-element(loop_call), index=0
    gte1 = f16[128] get-tuple-element(loop_call), index=1
    ROOT r = (f16[128],f16[128]) tuple(gte0, gte1)
  }
  )";
  const std::string hlo =
      absl::StrFormat(hlo_format, ThreeState_Name(replica_partition));

  auto config = GetModuleConfigForTest();
  config.set_argument_input_indices({2});
  config.set_resource_input_indices({0, 1});
  config.set_resource_input_initialized({true, true});
  config.set_resource_update_to_input_index({0, 1});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  HloInstruction* loop = FindInstruction(module.get(), "loop_call");

  HloInstruction* arg1 = FindInstruction(module.get(), "arg1");
  HloInstruction* arg0 = FindInstruction(module.get(), "arg0");
  HloInstruction* arg0_r = FindInstruction(module.get(), "arg0_r");
  HloInstruction* arg2 = FindInstruction(module.get(), "arg2");
  HloInstruction* arg2_new = FindInstruction(module.get(), "arg2_new");
  HloInstruction* arg3 = FindInstruction(module.get(), "arg3");
  HloInstruction* constant = FindInstruction(module.get(), "const");
  HloInstruction* arg3_new = FindInstruction(module.get(), "arg3_new");
  HloInstruction* bcast = FindInstruction(module.get(), "bcast");
  HloInstruction* arg3_arg2_mul =
      FindInstruction(module.get(), "arg3_arg2_mul");
  HloInstruction* arg1_new = FindInstruction(module.get(), "arg1_new");

  CompilerAnnotations annotations(module.get());
  TF_ASSERT_OK_AND_ASSIGN(
      bool offloaded,
      VariablesOffloadAndPartition(annotations, true, 4, 2).Run(module.get()));
  EXPECT_TRUE(offloaded);

  std::unique_ptr<ResourceUpdateElementwiseClustering> pass_ptr(
      replica_partition
          ? new ReplicatedResourceUpdateElementwiseClustering(annotations, 2)
          : new ResourceUpdateElementwiseClustering);
  auto& pass = *pass_ptr;
  auto elementwise_comps =
      ElementwiseCluster::GetElementwiseClusterableComputations(module.get());
  TF_ASSERT_OK(pass.RunDataflowAnalysis(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto clusters,
                          pass.GetClustersIn(loop, elementwise_comps));
  ASSERT_THAT(clusters.size(), 1);

  auto& cluster = *std::begin(clusters);
  const int64 shard_size = replica_partition ? 64 : 128;
  EXPECT_THAT(cluster.GetClusterSize(), 128);
  EXPECT_THAT(cluster.GetAlignedClusterSize(), 128);
  EXPECT_THAT(cluster.GetShardSize(), shard_size);
  HloInstruction* arg2_new_reshape = arg2_new->mutable_operand(1);
  EXPECT_THAT(cluster.GetInputs(), ::testing::UnorderedElementsAre(
                                       arg0_r, arg1, arg2_new_reshape, bcast));
  EXPECT_THAT(cluster.GetPostOrder(), ::testing::UnorderedElementsAre(
                                          arg2_new, arg3_arg2_mul, arg1_new));
  EXPECT_THAT(cluster.GetOutputs(),
              ::testing::UnorderedElementsAre(arg1_new, arg2_new));

  // Convert the cluster.
  TF_ASSERT_OK(pass.OutlineCluster(cluster).status());
  TF_ASSERT_OK_AND_ASSIGN(bool eliminated, HloDCE().Run(module.get()));

  HloInstruction *arg2_load, *arg2_store;
  TF_ASSERT_OK(GetRemoteLoadStoreUsers(arg2, &arg2_load, &arg2_store));

  EXPECT_THAT(arg0->user_count(), 1);
  HloInstruction* cluster_call = GetNextUser(arg0).ValueOrDie();
  EXPECT_TRUE(IsFunction(cluster_call));
  HloComputation* cluster_comp = cluster_call->to_apply();

  EXPECT_THAT(cluster_call->operands(),
              ::testing::ElementsAre(arg1, arg0, arg3_new, arg2_load));

  arg1 = cluster_comp->parameter_instruction(0);
  if (replica_partition) {
    TF_ASSERT_OK_AND_ASSIGN(arg1, GetRewrittenInput(arg1, false));
  }
  EXPECT_THAT(arg1->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  HloInstruction* collective = cluster_comp->parameter_instruction(1);
  if (replica_partition) {
    TF_ASSERT_OK_AND_ASSIGN(collective, GetScatterInput(collective, false));
  } else {
    TF_ASSERT_OK_AND_ASSIGN(collective, GetAllReduceInput(collective));
  }
  EXPECT_THAT(collective->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  HloInstruction* param_arg3_new = cluster_comp->parameter_instruction(2);
  EXPECT_THAT(param_arg3_new->shape(), ShapeUtil::MakeShape(F16, {}));
  EXPECT_THAT(param_arg3_new->user_count(), 1);
  bcast = param_arg3_new->users()[0];
  EXPECT_THAT(bcast->operands(), ::testing::ElementsAre(param_arg3_new));
  EXPECT_THAT(bcast->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  arg2_load = cluster_comp->parameter_instruction(3);
  EXPECT_THAT(arg2_load->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  EXPECT_THAT(arg2_load->user_count(), 1);
  arg2_new = arg2_load->users()[0];
  EXPECT_THAT(arg2_new->operands(),
              ::testing::ElementsAre(collective, arg2_load));
  EXPECT_THAT(arg2_new->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  EXPECT_THAT(bcast->user_count(), 1);
  arg3_arg2_mul = bcast->users()[0];
  EXPECT_THAT(arg3_arg2_mul->operands(),
              ::testing::ElementsAre(bcast, arg2_new));
  EXPECT_THAT(arg3_arg2_mul->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  EXPECT_THAT(arg1->user_count(), 1);
  arg1_new = arg1->users()[0];
  EXPECT_THAT(arg1_new->operands(),
              ::testing::ElementsAre(arg1, arg3_arg2_mul));
  EXPECT_THAT(arg1_new->shape(), ShapeUtil::MakeShape(F16, {shard_size}));
  if (replica_partition) {
    TF_ASSERT_OK_AND_ASSIGN(arg1_new, GetRewrittenOutput(arg1_new, false));
  }
  EXPECT_THAT(arg1_new->shape(), ShapeUtil::MakeShape(F16, {128}));

  if (replica_partition) {
    EXPECT_TRUE(Match(cluster_comp->root_instruction(),
                      m::Tuple(m::Op().Is(arg2_new), m::Op().Is(arg1_new))));
    EXPECT_TRUE(
        Match(arg2_store,
              m::CustomCall(m::Op().Is(arg2),
                            m::GetTupleElement(m::Op().Is(cluster_call), 0))));
  } else {
    EXPECT_TRUE(Match(cluster_comp->root_instruction(),
                      m::Tuple(m::Op().Is(arg2_new), m::Op().Is(arg1_new))));
    EXPECT_TRUE(
        Match(arg2_store,
              m::CustomCall(m::Op().Is(arg2),
                            m::GetTupleElement(m::Op().Is(cluster_call), 0))));
  }

  HloComputation* resource_update =
      FindComputation(module.get(), "resource_update");
  EXPECT_TRUE(Match(resource_update->root_instruction(),
                    m::Tuple(m::GetTupleElement(m::Op().Is(cluster_call), 1),
                             m::Op().Is(arg2_store), m::Op().Is(arg3_new))));
}

TEST_P(ReplicatedResourceUpdateElementwiseClusteringBasicTest,
       TestWideConstant) {
  const bool replica_partition = GetParam();
  constexpr absl::string_view hlo_format = R"(
  HloModule main

  sum {
    y = f16[] parameter(1)
    x = f16[] parameter(0), control-predecessors={y}
    ROOT add = f16[] add(x, y), backend_config="{\"isInplace\":true}"
  }

  _pop_op_wide_const {
    constant = f16[] constant(0.125)
    ROOT bcast = f16[128] broadcast(constant), dimensions={}
  }

  resource_update {
    arg0 = f16[128] parameter(0)
    arg1 = f16[128] parameter(1)
    arg0_r = f16[128] all-reduce(arg0), to_apply=sum
    arg2 = f16[128] parameter(2)
    arg2_new = f16[128] add(arg0_r, arg2)
    fusion = f16[128] fusion(), kind=kCustom, calls=_pop_op_wide_const
    arg3_arg2_mul = f16[128] multiply(fusion, arg2_new)
    arg1_new = f16[128] add(arg1, arg3_arg2_mul)
    ROOT t = (f16[128],f16[128]) tuple(arg1_new, arg2_new)
    counter_0 = s32[] constant(4)
    gac = () custom-call(s32[] counter_0), custom_call_target="GradientAccumulationCount"
  }

  loop {
    after-all = token[] after-all()
    infeed = (f16[128], token[]) infeed(after-all), infeed_config="140121807314576"
    input = f16[128] get-tuple-element(infeed), index=0

    arg0 = f16[128] parameter(0)
    arg1 = f16[128] parameter(1)

    add.1 = f16[128] add(input, arg0)
    call = (f16[128],f16[128]) call(add.1, arg0, arg1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\", \"partitionOffloadedVariables\":\"%s\"}}}"
    gte0 = f16[128] get-tuple-element(call), index=0
    gte1 = f16[128] get-tuple-element(call), index=1
    ROOT r = (f16[128],f16[128]) tuple(gte0, gte1)
  }

  ENTRY e {
    e.in0 = f16[128] parameter(0)
    e.in1 = f16[128] parameter(1)
    loop_call = (f16[128],f16[128]) call(e.in0, e.in1), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
    gte0 = f16[128] get-tuple-element(loop_call), index=0
    gte1 = f16[128] get-tuple-element(loop_call), index=1
    ROOT r = (f16[128],f16[128]) tuple(gte0, gte1)
  }
  )";
  const std::string hlo =
      absl::StrFormat(hlo_format, ThreeState_Name(replica_partition));

  auto config = GetModuleConfigForTest();
  config.set_argument_input_indices({});
  config.set_resource_input_indices({0, 1});
  config.set_resource_input_initialized({true, true});
  config.set_resource_update_to_input_index({0, 1});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  HloInstruction* loop = FindInstruction(module.get(), "loop_call");

  HloInstruction* arg0 = FindInstruction(module.get(), "arg0");
  HloInstruction* arg0_r = FindInstruction(module.get(), "arg0_r");
  HloInstruction* arg1 = FindInstruction(module.get(), "arg1");
  HloInstruction* arg2 = FindInstruction(module.get(), "arg2");
  HloInstruction* arg2_new = FindInstruction(module.get(), "arg2_new");
  HloInstruction* fusion = FindInstruction(module.get(), "fusion");
  HloInstruction* arg3_arg2_mul =
      FindInstruction(module.get(), "arg3_arg2_mul");
  HloInstruction* arg1_new = FindInstruction(module.get(), "arg1_new");

  CompilerAnnotations annotations(module.get());
  TF_ASSERT_OK_AND_ASSIGN(
      bool offloaded,
      VariablesOffloadAndPartition(annotations, true, 4, 2).Run(module.get()));
  EXPECT_TRUE(offloaded);

  std::unique_ptr<ResourceUpdateElementwiseClustering> pass_ptr(
      replica_partition
          ? new ReplicatedResourceUpdateElementwiseClustering(annotations, 2)
          : new ResourceUpdateElementwiseClustering);
  auto& pass = *pass_ptr;
  auto elementwise_comps =
      ElementwiseCluster::GetElementwiseClusterableComputations(module.get());
  TF_ASSERT_OK(pass.RunDataflowAnalysis(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto clusters,
                          pass.GetClustersIn(loop, elementwise_comps));
  ASSERT_THAT(clusters.size(), 1);

  const int64 shard_size = replica_partition ? 64 : 128;
  auto& cluster = *std::begin(clusters);
  EXPECT_THAT(cluster.GetClusterSize(), 128);
  EXPECT_THAT(cluster.GetAlignedClusterSize(), 128);
  EXPECT_THAT(cluster.GetShardSize(), shard_size);
  HloInstruction* arg2_new_reshape = arg2_new->mutable_operand(1);
  EXPECT_THAT(cluster.GetInputs(), ::testing::UnorderedElementsAre(
                                       arg0_r, arg1, arg2_new_reshape, fusion));
  EXPECT_THAT(cluster.GetPostOrder(), ::testing::UnorderedElementsAre(
                                          arg2_new, arg3_arg2_mul, arg1_new));
  EXPECT_THAT(cluster.GetOutputs(),
              ::testing::UnorderedElementsAre(arg1_new, arg2_new));

  // Convert the cluster.
  TF_ASSERT_OK(pass.OutlineCluster(cluster).status());
  TF_ASSERT_OK_AND_ASSIGN(bool eliminated, HloDCE().Run(module.get()));

  HloInstruction *arg2_load, *arg2_store;
  TF_ASSERT_OK(GetRemoteLoadStoreUsers(arg2, &arg2_load, &arg2_store));

  EXPECT_THAT(arg0->user_count(), 1);
  HloInstruction* cluster_call = GetNextUser(arg0).ValueOrDie();
  EXPECT_TRUE(IsFunction(cluster_call));
  HloComputation* cluster_comp = cluster_call->to_apply();

  const HloInstruction* new_const = cluster_call->operand(2);
  EXPECT_TRUE(Match(new_const, m::ConstantScalar(0.125)));
  EXPECT_THAT(cluster_call->operands(),
              ::testing::ElementsAre(arg1, arg0, new_const, arg2_load));

  arg1 = cluster_comp->parameter_instruction(0);
  if (replica_partition) {
    TF_ASSERT_OK_AND_ASSIGN(arg1, GetRewrittenInput(arg1, false));
  }
  EXPECT_THAT(arg1->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  HloInstruction* collective = cluster_comp->parameter_instruction(1);
  if (replica_partition) {
    TF_ASSERT_OK_AND_ASSIGN(collective, GetScatterInput(collective, false));
  } else {
    TF_ASSERT_OK_AND_ASSIGN(collective, GetAllReduceInput(collective));
  }
  EXPECT_THAT(collective->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  fusion = cluster_comp->parameter_instruction(2);
  EXPECT_THAT(fusion->shape(), ShapeUtil::MakeShape(F16, {}));
  EXPECT_THAT(fusion->user_count(), 1);
  fusion = fusion->users()[0];
  EXPECT_THAT(fusion->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  arg2_load = cluster_comp->parameter_instruction(3);
  EXPECT_THAT(arg2_load->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  EXPECT_THAT(arg2_load->user_count(), 1);
  arg2_new = arg2_load->users()[0];
  EXPECT_THAT(arg2_new->operands(),
              ::testing::ElementsAre(collective, arg2_load));
  EXPECT_THAT(arg2_new->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  arg3_arg2_mul = fusion->users()[0];
  EXPECT_THAT(arg3_arg2_mul->operands(),
              ::testing::ElementsAre(fusion, arg2_new));
  EXPECT_THAT(arg3_arg2_mul->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  EXPECT_THAT(arg1->user_count(), 1);
  arg1_new = arg1->users()[0];
  EXPECT_THAT(arg1_new->operands(),
              ::testing::ElementsAre(arg1, arg3_arg2_mul));
  EXPECT_THAT(arg1_new->shape(), ShapeUtil::MakeShape(F16, {shard_size}));
  if (replica_partition) {
    TF_ASSERT_OK_AND_ASSIGN(arg1_new, GetRewrittenOutput(arg1_new, false));
  }
  EXPECT_THAT(arg1_new->shape(), ShapeUtil::MakeShape(F16, {128}));

  if (replica_partition) {
    EXPECT_TRUE(Match(cluster_comp->root_instruction(),
                      m::Tuple(m::Op().Is(arg2_new), m::Op().Is(arg1_new))));
    EXPECT_TRUE(
        Match(arg2_store,
              m::CustomCall(m::Op().Is(arg2),
                            m::GetTupleElement(m::Op().Is(cluster_call), 0))));
  } else {
    EXPECT_TRUE(Match(cluster_comp->root_instruction(),
                      m::Tuple(m::Op().Is(arg2_new), m::Op().Is(arg1_new))));
    EXPECT_TRUE(
        Match(arg2_store,
              m::CustomCall(m::Op().Is(arg2),
                            m::GetTupleElement(m::Op().Is(cluster_call), 0))));
  }

  HloComputation* resource_update =
      FindComputation(module.get(), "resource_update");
  EXPECT_TRUE(Match(resource_update->root_instruction(),
                    m::Tuple(m::GetTupleElement(m::Op().Is(cluster_call), 1),
                             m::Op().Is(arg2_store))));
}

std::string GetHlo(bool partition_offloaded_variables,
                   const std::vector<int64>& dimensions,
                   const PrimitiveType& element_type,
                   const PrimitiveType& remote_buffer_element_type) {
  const std::string hlo = R"(
  HloModule main

  sum {
    y = $element_type[] parameter(1)
    x = $element_type[] parameter(0), control-predecessors={y}
    ROOT add = $element_type[] add(x, y),
    backend_config="{\"isInplace\":true}"
  }

  resource_update {
    arg0 = $element_type$shape parameter(0)
    arg1 = $element_type$shape parameter(1)
    arg0_r = $element_type$shape all-reduce(arg0), to_apply=sum
    arg2 = $remote_buffer_element_type$shape parameter(2)
    arg2_c = $element_type$shape convert(arg2)
    arg2_new = $element_type$shape add(arg0_r, arg2_c)
    arg1_new = $element_type$shape add(arg1, arg2_new)
    arg2_new_c = $remote_buffer_element_type$shape convert(arg2_new)
    ROOT t = ($element_type$shape,$remote_buffer_element_type$shape)
    tuple(arg1_new, arg2_new_c)

    counter_0 = s32[] constant(4)
    gac = () custom-call(s32[] counter_0), custom_call_target="GradientAccumulationCount"
  }

  loop {
    after-all = token[] after-all()
    infeed = ($element_type$shape, token[]) infeed(after-all),
    infeed_config="140121807314576" input = $element_type$shape
    get-tuple-element(infeed), index=0

    arg0 = $element_type$shape parameter(0)
    arg1 = $remote_buffer_element_type$shape parameter(1)

    add.1 = $element_type$shape add(input, arg0)
    call = ($element_type$shape,$remote_buffer_element_type$shape)
    call(add.1, arg0, arg1), to_apply=resource_update,
    frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate},
    backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\",
    \"partitionOffloadedVariables\":\"$partition_offloaded_variables\"}}}" gte0 =
    $element_type$shape get-tuple-element(call), index=0 gte1 =
    $remote_buffer_element_type$shape get-tuple-element(call), index=1 ROOT r
    = ($element_type$shape,$remote_buffer_element_type$shape) tuple(gte0,
    gte1)
  }

  ENTRY e {
    e.in0 = $element_type$shape parameter(0)
    e.in1 = $remote_buffer_element_type$shape parameter(1)
    loop_call = ($element_type$shape,$remote_buffer_element_type$shape)
    call(e.in0, e.in1), to_apply=loop,
    backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
    gte0 = $element_type$shape get-tuple-element(loop_call), index=0
    gte1 = $remote_buffer_element_type$shape get-tuple-element(loop_call),
    index=1 ROOT r = ($element_type$shape,$remote_buffer_element_type$shape)
    tuple(gte0, gte1)
  }
  )";
  std::string hlo_string = tensorflow::str_util::StringReplace(
      hlo, "$shape", absl::StrCat("[", absl::StrJoin(dimensions, ","), "]"),
      true);
  hlo_string = tensorflow::str_util::StringReplace(
      hlo_string, "$partition_offloaded_variables",
      ThreeState_Name(partition_offloaded_variables), true);
  hlo_string = tensorflow::str_util::StringReplace(
      hlo_string, "$element_type",
      primitive_util::LowercasePrimitiveTypeName(element_type), true);
  hlo_string = tensorflow::str_util::StringReplace(
      hlo_string, "$remote_buffer_element_type",
      primitive_util::LowercasePrimitiveTypeName(remote_buffer_element_type),
      true);
  return hlo_string;
}

struct ReplicatedResourceUpdateElementwiseClusteringShapeTestSpec {
  bool partition_offloaded_variables;
  std::vector<int64> dimensions;
  int64 cluster_size;
  int64 aligned_cluster_size;
  int64 shard_size;
  PrimitiveType element_type;
  PrimitiveType remote_buffer_element_type;
  bool padded_and_sliced;
};

std::ostream& operator<<(
    std::ostream& os,
    const ReplicatedResourceUpdateElementwiseClusteringShapeTestSpec& spec) {
  return os << "{ partition_offloaded_variables: "
            << spec.partition_offloaded_variables << ", dimensions: ["
            << absl::StrJoin(spec.dimensions, ",")
            << "], cluster_size: " << spec.cluster_size
            << ", aligned_cluster_size: " << spec.aligned_cluster_size
            << ", shard_size: " << spec.shard_size
            << ", element_type: " << spec.element_type
            << ", remote_buffer_element_type: "
            << spec.remote_buffer_element_type
            << ", padded_and_sliced: " << spec.padded_and_sliced << "}";
}

class ReplicatedResourceUpdateElementwiseClusteringShapeTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          ReplicatedResourceUpdateElementwiseClusteringShapeTestSpec> {};

INSTANTIATE_TEST_SUITE_P(
    ReplicatedResourceUpdateElementwiseClusteringShapeTestCases,
    ReplicatedResourceUpdateElementwiseClusteringShapeTest,
    ::testing::ValuesIn(
        std::vector<ReplicatedResourceUpdateElementwiseClusteringShapeTestSpec>{
            {true, {128, 2, 2}, 512, 512, 256, F32, F32, false},
            {true, {128}, 128, 128, 64, F32, F32, false},
            {true, {128, 2, 2}, 512, 512, 256, F16, F32, false},
            {true, {128, 2, 2}, 512, 512, 256, F32, F16, false},
            {true, {129, 3}, 387, 388, 194, F32, F32, true},
            {true, {1}, 1, 2, 1, F32, F32, true},
            {true, {127, 5}, 635, 636, 318, F16, F32, true},
            {true, {127, 5}, 635, 636, 318, F32, F16, true},
            {false, {128}, 128, 128, 128, F32, F32, false},
            {false, {128, 2, 2}, 512, 512, 512, F16, F32, false},
        }));

TEST_P(ReplicatedResourceUpdateElementwiseClusteringShapeTest, DoTest) {
  auto param = GetParam();

  auto config = GetModuleConfigForTest();
  config.set_argument_input_indices({});
  config.set_resource_input_indices({0, 1});
  config.set_resource_input_initialized({true, true});
  config.set_resource_update_to_input_index({0, 1});
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(
          GetHlo(param.partition_offloaded_variables, param.dimensions,
                 param.element_type, param.remote_buffer_element_type),
          config));

  HloInstruction* loop = FindInstruction(module.get(), "loop_call");

  HloInstruction* arg0 = FindInstruction(module.get(), "arg0");
  HloInstruction* arg0_r = FindInstruction(module.get(), "arg0_r");
  HloInstruction* arg1 = FindInstruction(module.get(), "arg1");
  HloInstruction* arg2 = FindInstruction(module.get(), "arg2");
  HloInstruction* arg2_c = FindInstruction(module.get(), "arg2_c");
  HloInstruction* arg2_new = FindInstruction(module.get(), "arg2_new");
  HloInstruction* arg1_new = FindInstruction(module.get(), "arg1_new");
  HloInstruction* arg2_new_c = FindInstruction(module.get(), "arg2_new_c");

  CompilerAnnotations annotations(module.get());
  TF_ASSERT_OK_AND_ASSIGN(
      bool offloaded,
      VariablesOffloadAndPartition(annotations, true, 0, 2).Run(module.get()));
  EXPECT_TRUE(offloaded);

  std::unique_ptr<ResourceUpdateElementwiseClustering> pass_ptr(
      param.partition_offloaded_variables
          ? new ReplicatedResourceUpdateElementwiseClustering(annotations, 2)
          : new ResourceUpdateElementwiseClustering);
  auto& pass = *pass_ptr;
  auto elementwise_comps =
      ElementwiseCluster::GetElementwiseClusterableComputations(module.get());
  TF_ASSERT_OK(pass.RunDataflowAnalysis(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto clusters,
                          pass.GetClustersIn(loop, elementwise_comps));
  ASSERT_THAT(clusters.size(), 1);
  auto& cluster = *std::begin(clusters);
  EXPECT_THAT(cluster.GetClusterSize(), param.cluster_size);
  EXPECT_THAT(cluster.GetAlignedClusterSize(), param.aligned_cluster_size);
  EXPECT_THAT(cluster.GetShardSize(), param.shard_size);
  HloInstruction* arg2_c_reshape = arg2_c->mutable_operand(0);
  EXPECT_THAT(cluster.GetInputs(),
              ::testing::UnorderedElementsAre(arg0_r, arg1, arg2_c_reshape));
  EXPECT_THAT(
      cluster.GetPostOrder(),
      ::testing::UnorderedElementsAre(arg2_c, arg2_new, arg1_new, arg2_new_c));
  EXPECT_THAT(cluster.GetOutputs(),
              ::testing::UnorderedElementsAre(arg1_new, arg2_new_c));

  // Convert the cluster.
  TF_ASSERT_OK(pass.OutlineCluster(cluster).status());
  TF_ASSERT_OK_AND_ASSIGN(bool eliminated, HloDCE().Run(module.get()));

  HloInstruction *arg2_load, *arg2_store;
  TF_ASSERT_OK(GetRemoteLoadStoreUsers(arg2, &arg2_load, &arg2_store));

  EXPECT_THAT(arg0->user_count(), 1);
  HloInstruction* cluster_call = GetNextUser(arg0).ValueOrDie();
  EXPECT_TRUE(IsFunction(cluster_call));
  HloComputation* cluster_comp = cluster_call->to_apply();

  EXPECT_THAT(cluster_call->operands(),
              ::testing::ElementsAre(arg1, arg0, arg2_load));

  arg1 = cluster_comp->parameter_instruction(0);
  if (param.partition_offloaded_variables) {
    TF_ASSERT_OK_AND_ASSIGN(arg1,
                            GetRewrittenInput(arg1, param.padded_and_sliced));
  }

  HloInstruction* collective = cluster_comp->parameter_instruction(1);
  if (param.partition_offloaded_variables) {
    TF_ASSERT_OK_AND_ASSIGN(
        collective, GetScatterInput(collective, param.padded_and_sliced));
    EXPECT_THAT(collective->shape(),
                ShapeUtil::MakeShape(param.element_type, {param.shard_size}));
  } else {
    TF_ASSERT_OK_AND_ASSIGN(collective, GetAllReduceInput(collective));
    EXPECT_THAT(collective->shape(),
                ShapeUtil::MakeShape(param.element_type, {param.dimensions}));
  }

  arg2_load = cluster_comp->parameter_instruction(2);

  TF_ASSERT_OK_AND_ASSIGN(arg2_c, GetNextUser(arg2_load));
  TF_ASSERT_OK_AND_ASSIGN(arg2_c, LookThroughReshape(arg2_c));
  EXPECT_THAT(arg2_c->opcode(), HloOpcode::kConvert);

  TF_ASSERT_OK_AND_ASSIGN(arg2_new, GetNextUser(arg2_c));
  EXPECT_THAT(arg2_new->operands(),
              ::testing::ElementsAre(MaybeReshapeOf(collective), arg2_c));
  EXPECT_THAT(arg2_new->opcode(), HloOpcode::kAdd);

  TF_ASSERT_OK_AND_ASSIGN(arg1_new, GetNextUser(arg1));
  if (arg1_new->opcode() == HloOpcode::kReshape) {
    arg1 = arg1_new;
    TF_ASSERT_OK_AND_ASSIGN(arg1_new, LookThroughReshape(arg1_new));
  }
  EXPECT_THAT(arg1_new->operands(), ::testing::ElementsAre(arg1, arg2_new));
  if (param.partition_offloaded_variables) {
    EXPECT_THAT(arg1_new->shape(),
                ShapeUtil::MakeShape(param.element_type, {param.shard_size}));
  } else {
    EXPECT_THAT(arg1_new->shape(),
                ShapeUtil::MakeShape(param.element_type, param.dimensions));
  }

  EXPECT_THAT(arg2_new->user_count(), 2);
  auto arg2_new_c_it =
      absl::c_find_if(arg2_new->users(), [](const HloInstruction* i) {
        return i->opcode() == HloOpcode::kConvert;
      });
  CHECK(arg2_new_c_it != arg2_new->users().end());
  arg2_new_c = *arg2_new_c_it;

  EXPECT_THAT(arg2_new_c->operands(), ::testing::ElementsAre(arg2_new));
  if (param.partition_offloaded_variables) {
    EXPECT_THAT(arg2_new_c->shape(),
                ShapeUtil::MakeShape(param.remote_buffer_element_type,
                                     {param.shard_size}));
  } else {
    EXPECT_THAT(arg1_new->shape(),
                ShapeUtil::MakeShape(param.element_type, param.dimensions));
  }

  EXPECT_THAT(arg2_new_c->user_count(), 1);
  if (param.partition_offloaded_variables) {
    EXPECT_THAT(arg2_new_c->shape(),
                ShapeUtil::MakeShape(param.remote_buffer_element_type,
                                     {param.shard_size}));
    TF_ASSERT_OK_AND_ASSIGN(
        arg1_new, GetRewrittenOutput(arg1_new, param.padded_and_sliced));
  }

  arg1_new = MaybeReshapeOf(arg1_new);
  arg2_new_c = MaybeReshapeOf(arg2_new_c);
  EXPECT_TRUE(Match(cluster_comp->root_instruction(),
                    m::Tuple(m::Op().Is(arg1_new), m::Op().Is(arg2_new_c))));

  EXPECT_TRUE(
      Match(arg2_store,
            m::CustomCall(m::Op().Is(arg2),
                          m::GetTupleElement(m::Op().Is(cluster_call), 1))));

  HloComputation* resource_update =
      FindComputation(module.get(), "resource_update");
  EXPECT_TRUE(Match(resource_update->root_instruction(),
                    m::Tuple(m::GetTupleElement(m::Op().Is(cluster_call), 0),
                             m::Op().Is(arg2_store))));
}

struct ReplicatedResourceUpdateElementwiseClusteringTestSpec {
  std::string hlo;
  std::string short_name;
  bool cluster;
  int expected_offloads;
  int expected_all_gathers;
  // These are all-gathers which are not arguments to the root instruction.
  int expected_non_root_all_gathers;
  int expected_elementwise_clusters;
  int expected_all_reduces;
};

std::ostream& operator<<(
    std::ostream& os,
    const ReplicatedResourceUpdateElementwiseClusteringTestSpec& spec) {
  return os << "{ name: " << spec.short_name << ", cluster: " << spec.cluster
            << ", offloads: " << spec.expected_offloads
            << ", all-gathers: " << spec.expected_all_gathers
            << ", non-root-all-gathers: " << spec.expected_non_root_all_gathers
            << ", elementwise-clusters: " << spec.expected_elementwise_clusters
            << ", all-reduces: " << spec.expected_all_reduces << "}";
}

class ReplicatedResourceUpdateElementwiseClusteringTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          ReplicatedResourceUpdateElementwiseClusteringTestSpec> {};

INSTANTIATE_TEST_SUITE_P(
    ReplicatedResourceUpdateElementwiseClusteringTestCases,
    ReplicatedResourceUpdateElementwiseClusteringTest,
    ::testing::ValuesIn(
        std::vector<ReplicatedResourceUpdateElementwiseClusteringTestSpec>{
            // Simple HLO with all types of the inputs
            {tu::GetSimpleHloString(20, 100), "simple", false, 2, 2, 2, 0, 2},
            {tu::GetSimpleHloString(20, 100), "simple", true, 2, 0, 0, 1, 0},
            // Edge case
            {tu::GetSimpleHloString(1, 1), "1x1", false, 2, 2, 2, 0, 2},
            {tu::GetSimpleHloString(1, 1), "1x1", true, 2, 0, 0, 1, 0},
            // Check padded offloading:
            {tu::GetSimpleHloString(11, 13), "simple-padded", false, 2, 2, 2, 0,
             2},
            {tu::GetSimpleHloString(11, 13), "simple-padded", true, 2, 0, 0, 1,
             0},
            // Two cluster share the same inputs
            {tu::GetTwoClustersShareInputHloString(20, 100), "2-clusters",
             false, 2, 2, 2, 0, 2},
            {tu::GetTwoClustersShareInputHloString(20, 100), "2-clusters", true,
             2, 0, 0, 2, 2},
            // Adam-like resource update
            {tu::GetAdamLikeHloString(20, 100), "adam", false, 2, 2, 2, 0, 2},
            {tu::GetAdamLikeHloString(20, 100), "adam", true, 2, 0, 0, 1, 0},
            // Lamb-like resource update
            {tu::GetLambLikeHloString(20, 100), "lamb", false, 2, 2, 2, 0, 2},
            {tu::GetLambLikeHloString(20, 100), "lamb", true, 2, 0, 0, 1, 2},
            // Momentum-like resource update
            {tu::GetMomentumLikeHloString(1000, 20), "momentum", false, 2, 2, 2,
             0, 1},
            {tu::GetMomentumLikeHloString(1000, 20), "momentum", true, 2, 0, 0,
             1, 0},
            // SGD-like resource update
            {tu::GetSGDHloString(1000, 20), "sgd", false, 2, 2, 2, 0, 2},
            {tu::GetSGDHloString(1000, 20), "sgd", true, 2, 0, 0, 2, 0},
            // Test with one of the arguments be non-replicated remote buffer.
            {tu::GetFullRemoteLoadHloString(100, 20), "full-remote-load", false,
             4, 2, 2, 0, 0},
            {tu::GetFullRemoteLoadHloString(19, 7), "full-remote-load", false,
             4, 2, 2, 0, 0},
        }));

int64 GetCount(const HloComputation* comp,
               std::function<bool(const HloInstruction*)> pred) {
  int64 count = 0;
  for (auto inst : comp->instructions()) {
    count += pred(inst);
    for (auto called_comp : inst->called_computations()) {
      count += GetCount(called_comp, pred);
    }
  }
  return count;
}

TEST_P(ReplicatedResourceUpdateElementwiseClusteringTest, DoTest) {
  auto param = GetParam();

  static constexpr int replication_factor = 2;
  auto config = GetModuleConfigForTest();
  config.set_argument_input_indices({});
  config.set_resource_input_indices({0, 1, 2, 3});
  config.set_resource_input_initialized({true, true, true, true});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(param.hlo, config));

  CompilerAnnotations annotations(module.get());
  TF_ASSERT_OK_AND_ASSIGN(bool custom_op_replaced,
                          CustomOpReplacer().Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(
      bool offloaded,
      VariablesOffloadAndPartition(annotations, true, 0, replication_factor)
          .Run(module.get()));
  EXPECT_TRUE(offloaded);

  if (param.cluster) {
    TF_ASSERT_OK_AND_ASSIGN(bool changed,
                            ReplicatedResourceUpdateElementwiseClustering(
                                annotations, replication_factor)
                                .Run(module.get()));
    EXPECT_TRUE(changed);
  }
  TF_ASSERT_OK_AND_ASSIGN(bool inlined,
                          FusionInliner([](const HloInstruction* inst) {
                            return IsReplicatedParameterLoadFusion(inst) ||
                                   IsReplicatedParameterStoreFusion(inst);
                          })
                              .Run(module.get()));
  EXPECT_TRUE(inlined);
  TF_ASSERT_OK_AND_ASSIGN(bool eliminated, HloDCE().Run(module.get()));
  auto root = module->entry_computation()->root_instruction();
  HloComputation* repeat_computation = root->operand(0)->operand(0)->to_apply();
  HloInstruction* repeat_root = repeat_computation->root_instruction();
  HloComputation* resource_update =
      repeat_root->mutable_operand(0)->mutable_operand(0)->to_apply();

  HloInstruction* resource_update_root = resource_update->root_instruction();
  EXPECT_EQ(resource_update->num_parameters(), 4);
  EXPECT_EQ(ShapeUtil::TupleElementCount(resource_update_root->shape()), 4);

  // Check there is 2 stores 2 loads, and all-gathers removed.
  EXPECT_EQ(GetCount(resource_update,
                     IsPoplarInstruction(PoplarOp::RemoteParameterStore)),
            param.expected_offloads);
  EXPECT_EQ(GetCount(resource_update,
                     IsPoplarInstruction(PoplarOp::RemoteParameterLoad)),
            param.expected_offloads);
  EXPECT_EQ(GetCount(resource_update, IsPoplarInstruction(PoplarOp::AllGather)),
            param.expected_all_gathers);
  EXPECT_EQ(param.expected_elementwise_clusters,
            GetCount(resource_update, IsFunction));

  // Check there are the specified number of all-reductions
  EXPECT_EQ(param.expected_all_reduces,
            GetCount(resource_update, [](const HloInstruction* inst) {
              return inst->opcode() == HloOpcode::kAllReduce;
            }));

  auto IsNonRootAllGather = [&](const HloInstruction* inst) {
    if (!IsPoplarInstruction(PoplarOp::AllGather)(inst)) {
      return false;
    }
    if (inst->user_count() != 1) {
      return true;
    }
    auto user = inst->users()[0];
    if (IsPoplarInstruction(PoplarOp::UndoCollectiveReorder)(user) &&
        user->user_count() == 1) {
      user = user->users()[0];
    }
    if (user->opcode() == HloOpcode::kReshape && user->user_count() == 1) {
      user = user->users()[0];
    }
    return user != inst->parent()->root_instruction();
  };
  EXPECT_EQ(GetCount(resource_update, IsNonRootAllGather),
            param.expected_non_root_all_gathers);
}

INSTANTIATE_TEST_CASE_P(
    ReplicatedResourceUpdateElementwiseClusteringBasicTest_Instantiation,
    ReplicatedResourceUpdateElementwiseClusteringBasicTest,
    ::testing::Values(true, false));

using TestPartitionReplicationFactor = HloTestBase;

TEST_F(TestPartitionReplicationFactor, TestCollectiveGroups) {
  const std::string hlo = R"(
  HloModule main

  sum {
    y = f16[] parameter(1)
    x = f16[] parameter(0), control-predecessors={y}
    ROOT add = f16[] add(x, y), backend_config="{\"isInplace\":true}"
  }

  resource_update {
    arg0 = f16[128] parameter(0)
    arg1 = f16[128] parameter(1)
    arg2 = f16[128] parameter(2)

    arg0_r = f16[128] all-reduce(arg0), to_apply=sum
    arg2_new = f16[128] add(arg0_r, arg2)
    arg1_new = f16[128] add(arg1, arg2_new)

    ROOT t = (f16[128],f16[128]) tuple(arg1_new, arg2_new)
    counter_0 = s32[] constant(4)
    gac = () custom-call(s32[] counter_0), custom_call_target="GradientAccumulationCount"
  }

  loop {
    after-all = token[] after-all()
    infeed = (f16[128], token[]) infeed(after-all), infeed_config="140121807314576"
    input = f16[128] get-tuple-element(infeed), index=0

    arg0 = f16[128] parameter(0)
    arg1 = f16[128] parameter(1)

    add.1 = f16[128] add(input, arg0)
    call = (f16[128],f16[128]) call(add.1, arg0, arg1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\", \"partitionOffloadedVariables\":\"THREESTATE_ON\"}}}"
    gte0 = f16[128] get-tuple-element(call), index=0
    gte1 = f16[128] get-tuple-element(call), index=1
    ROOT r = (f16[128],f16[128]) tuple(gte0, gte1)
  }

  ENTRY e {
    e.in0 = f16[128] parameter(0)
    e.in1 = f16[128] parameter(1)
    loop_call = (f16[128],f16[128]) call(e.in0, e.in1), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
    gte0 = f16[128] get-tuple-element(loop_call), index=0
    gte1 = f16[128] get-tuple-element(loop_call), index=1
    ROOT r = (f16[128],f16[128]) tuple(gte0, gte1)
  }
  )";

  auto config = GetModuleConfigForTest();
  config.set_argument_input_indices({});
  config.set_resource_input_indices({0, 1});
  config.set_resource_input_initialized({true, true});
  config.set_resource_update_to_input_index({0, 1});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  HloInstruction* loop =
      CHECK_NOTNULL(FindInstruction(module.get(), "loop_call"));
  HloInstruction* arg0 = CHECK_NOTNULL(FindInstruction(module.get(), "arg0"));
  HloInstruction* arg0_r =
      CHECK_NOTNULL(FindInstruction(module.get(), "arg0_r"));
  HloInstruction* arg1 = CHECK_NOTNULL(FindInstruction(module.get(), "arg1"));
  HloInstruction* arg2 = CHECK_NOTNULL(FindInstruction(module.get(), "arg2"));
  HloInstruction* arg2_new =
      CHECK_NOTNULL(FindInstruction(module.get(), "arg2_new"));
  HloInstruction* arg1_new =
      CHECK_NOTNULL(FindInstruction(module.get(), "arg1_new"));

  const uint64 partition_replication_factor = 2;
  const uint64 global_replication_factor = 8;

  CompilerAnnotations annotations(module.get());
  TF_ASSERT_OK_AND_ASSIGN(
      bool offloaded,
      VariablesOffloadAndPartition(
          annotations, /*remote_memory_supported=*/true,
          /*minimum_remote_tensor_size=*/4, partition_replication_factor)
          .Run(module.get()));
  EXPECT_TRUE(offloaded);

  ReplicatedResourceUpdateElementwiseClustering pass(
      annotations, partition_replication_factor, global_replication_factor,
      global_replication_factor);
  auto elementwise_comps =
      ElementwiseCluster::GetElementwiseClusterableComputations(module.get());
  TF_ASSERT_OK(pass.RunDataflowAnalysis(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto clusters,
                          pass.GetClustersIn(loop, elementwise_comps));
  ASSERT_THAT(clusters.size(), 1);

  const int64 shard_size = 128 / partition_replication_factor;
  auto& cluster = *std::begin(clusters);
  EXPECT_THAT(cluster.GetClusterSize(), 128);
  EXPECT_THAT(cluster.GetAlignedClusterSize(), 128);
  EXPECT_THAT(cluster.GetShardSize(), shard_size);

  HloInstruction* arg2_new_reshape = arg2_new->mutable_operand(1);
  EXPECT_THAT(cluster.GetInputs(),
              ::testing::UnorderedElementsAre(arg0_r, arg1, arg2_new_reshape));
  EXPECT_THAT(cluster.GetPostOrder(),
              ::testing::UnorderedElementsAre(arg2_new, arg1_new));
  EXPECT_THAT(cluster.GetOutputs(),
              ::testing::UnorderedElementsAre(arg1_new, arg2_new));

  // Convert the cluster.
  TF_ASSERT_OK(pass.OutlineCluster(cluster).status());
  TF_ASSERT_OK_AND_ASSIGN(bool eliminated, HloDCE().Run(module.get()));

  HloInstruction *arg2_load, *arg2_store;
  TF_ASSERT_OK(GetRemoteLoadStoreUsers(arg2, &arg2_load, &arg2_store));

  EXPECT_THAT(arg0->user_count(), 1);
  HloInstruction* cluster_call = GetNextUser(arg0).ValueOrDie();
  EXPECT_TRUE(IsFunction(cluster_call));
  HloComputation* cluster_comp = cluster_call->to_apply();
  EXPECT_THAT(cluster_call->operands(),
              ::testing::ElementsAre(arg1, arg0, arg2_load));

  arg1 = cluster_comp->parameter_instruction(0);
  TF_ASSERT_OK_AND_ASSIGN(arg1, GetRewrittenInput(arg1, false));
  EXPECT_THAT(arg1->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  HloInstruction* collective = cluster_comp->parameter_instruction(1);
  TF_ASSERT_OK_AND_ASSIGN(collective, GetScatterInput(collective, false));
  EXPECT_THAT(collective->shape(), ShapeUtil::MakeShape(F16, {shard_size}));

  auto* reduce_scatter = Cast<HloReduceScatterInstruction>(collective);
  const auto reduce_scatter_groups = reduce_scatter->GetPoplarReplicaGroups();
  EXPECT_THAT(reduce_scatter_groups,
              PoplarReplicaGroups::Consecutive(partition_replication_factor));

  EXPECT_THAT(reduce_scatter->user_count(), 1);
  HloInstruction* all_reduce = reduce_scatter->users()[0];
  EXPECT_THAT(all_reduce->opcode(), HloOpcode::kAllReduce);

  TF_ASSERT_OK_AND_ASSIGN(
      const auto all_reduce_groups,
      PoplarReplicaGroups::FromXlaReplicaGroups(all_reduce->replica_groups()));

  EXPECT_THAT(all_reduce_groups,
              PoplarReplicaGroups::Orthogonal(global_replication_factor /
                                              partition_replication_factor));
}

TEST_F(TestPartitionReplicationFactor, TestUnsupportedPartitioning) {
  const std::string hlo = R"(
  HloModule main

  sum {
    y = f16[] parameter(1)
    x = f16[] parameter(0), control-predecessors={y}
    ROOT add = f16[] add(x, y), backend_config="{\"isInplace\":true}"
  }

  resource_update {
    arg0 = f16[128] parameter(0)
    arg1 = f16[128] parameter(1)
    arg2 = f16[128] parameter(2)

    arg0_r = f16[128] all-reduce(arg0), to_apply=sum
    arg2_new = f16[128] add(arg0_r, arg2)
    arg1_new = f16[128] add(arg1, arg2_new)

    ROOT t = (f16[128],f16[128]) tuple(arg1_new, arg2_new)
    counter_0 = s32[] constant(4)
    gac = () custom-call(s32[] counter_0), custom_call_target="GradientAccumulationCount"
  }

  loop {
    after-all = token[] after-all()
    infeed = (f16[128], token[]) infeed(after-all), infeed_config="140121807314576"
    input = f16[128] get-tuple-element(infeed), index=0

    arg0 = f16[128] parameter(0)
    arg1 = f16[128] parameter(1)

    add.1 = f16[128] add(input, arg0)
    call = (f16[128],f16[128]) call(add.1, arg0, arg1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\", \"partitionOffloadedVariables\":\"THREESTATE_ON\"}}}"
    gte0 = f16[128] get-tuple-element(call), index=0
    gte1 = f16[128] get-tuple-element(call), index=1
    ROOT r = (f16[128],f16[128]) tuple(gte0, gte1)
  }

  ENTRY e {
    e.in0 = f16[128] parameter(0)
    e.in1 = f16[128] parameter(1)
    loop_call = (f16[128],f16[128]) call(e.in0, e.in1), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
    gte0 = f16[128] get-tuple-element(loop_call), index=0
    gte1 = f16[128] get-tuple-element(loop_call), index=1
    ROOT r = (f16[128],f16[128]) tuple(gte0, gte1)
  }
  )";

  auto config = GetModuleConfigForTest();
  config.set_argument_input_indices({});
  config.set_resource_input_indices({0, 1});
  config.set_resource_input_initialized({true, true});
  config.set_resource_update_to_input_index({0, 1});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  HloInstruction* loop =
      CHECK_NOTNULL(FindInstruction(module.get(), "loop_call"));

  const uint64 partition_replication_factor = 2;
  const uint64 ipu_link_domain_replication_factor = 4;
  const uint64 global_replication_factor = 8;

  CompilerAnnotations annotations(module.get());
  TF_ASSERT_OK_AND_ASSIGN(
      bool offloaded,
      VariablesOffloadAndPartition(
          annotations, /*remote_memory_supported=*/true,
          /*minimum_remote_tensor_size=*/4, partition_replication_factor)
          .Run(module.get()));
  EXPECT_TRUE(offloaded);

  ReplicatedResourceUpdateElementwiseClustering pass(
      annotations, partition_replication_factor, global_replication_factor,
      ipu_link_domain_replication_factor);
  auto elementwise_comps =
      ElementwiseCluster::GetElementwiseClusterableComputations(module.get());
  TF_ASSERT_OK(pass.RunDataflowAnalysis(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto clusters,
                          pass.GetClustersIn(loop, elementwise_comps));
  ASSERT_THAT(clusters.size(), 1);
  auto& cluster = clusters.front();
  const auto status = pass.OutlineCluster(cluster).status();
  ASSERT_THAT(status.code(), tensorflow::error::UNIMPLEMENTED);
  EXPECT_THAT(status.error_message(),
              ::testing::StartsWith(
                  "Replicated partitioning is not supported when there are "
                  "multiple instances per IPU-link domain."));
}

TEST_F(TestPartitionReplicationFactor, TestNonGlobalAllReduce) {
  const std::string hlo = R"(
  HloModule main

  sum {
    y = f16[] parameter(1)
    x = f16[] parameter(0), control-predecessors={y}
    ROOT add = f16[] add(x, y), backend_config="{\"isInplace\":true}"
  }

  resource_update {
    arg0 = f16[128] parameter(0)
    arg1 = f16[128] parameter(1)
    arg2 = f16[128] parameter(2)

    arg0_r = f16[128] all-reduce(arg0), to_apply=sum, replica_groups={{0}}
    arg2_new = f16[128] add(arg0_r, arg2)
    arg1_new = f16[128] add(arg1, arg2_new)

    ROOT t = (f16[128],f16[128]) tuple(arg1_new, arg2_new)
    counter_0 = s32[] constant(4)
    gac = () custom-call(s32[] counter_0), custom_call_target="GradientAccumulationCount"
  }

  loop {
    after-all = token[] after-all()
    infeed = (f16[128], token[]) infeed(after-all), infeed_config="140121807314576"
    input = f16[128] get-tuple-element(infeed), index=0

    arg0 = f16[128] parameter(0)
    arg1 = f16[128] parameter(1)

    add.1 = f16[128] add(input, arg0)
    call = (f16[128],f16[128]) call(add.1, arg0, arg1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\", \"partitionOffloadedVariables\":\"THREESTATE_ON\"}}}"
    gte0 = f16[128] get-tuple-element(call), index=0
    gte1 = f16[128] get-tuple-element(call), index=1
    ROOT r = (f16[128],f16[128]) tuple(gte0, gte1)
  }

  ENTRY e {
    e.in0 = f16[128] parameter(0)
    e.in1 = f16[128] parameter(1)
    loop_call = (f16[128],f16[128]) call(e.in0, e.in1), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
    gte0 = f16[128] get-tuple-element(loop_call), index=0
    gte1 = f16[128] get-tuple-element(loop_call), index=1
    ROOT r = (f16[128],f16[128]) tuple(gte0, gte1)
  }
  )";

  auto config = GetModuleConfigForTest();
  config.set_argument_input_indices({});
  config.set_resource_input_indices({0, 1});
  config.set_resource_input_initialized({true, true});
  config.set_resource_update_to_input_index({0, 1});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  HloInstruction* loop =
      CHECK_NOTNULL(FindInstruction(module.get(), "loop_call"));
  HloInstruction* arg0 = CHECK_NOTNULL(FindInstruction(module.get(), "arg0"));

  const uint64 partition_replication_factor = 2;
  const uint64 global_replication_factor = 8;

  CompilerAnnotations annotations(module.get());
  TF_ASSERT_OK_AND_ASSIGN(
      bool offloaded,
      VariablesOffloadAndPartition(
          annotations, /*remote_memory_supported=*/true,
          /*minimum_remote_tensor_size=*/4, partition_replication_factor)
          .Run(module.get()));
  EXPECT_TRUE(offloaded);

  ReplicatedResourceUpdateElementwiseClustering pass(
      annotations, partition_replication_factor, global_replication_factor,
      global_replication_factor);
  auto elementwise_comps =
      ElementwiseCluster::GetElementwiseClusterableComputations(module.get());
  TF_ASSERT_OK(pass.RunDataflowAnalysis(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto clusters,
                          pass.GetClustersIn(loop, elementwise_comps));
  // No clusters because the all-reduce is not global.
  ASSERT_THAT(clusters.size(), 0);
}

TEST_F(TestPartitionReplicationFactor, IgnoreImplicit2ScalarBroadcast) {
  static constexpr int replication_factor = 2;
  auto config = GetModuleConfigForTest();
  config.set_argument_input_indices({});
  config.set_resource_input_indices({0, 1, 2, 3});
  config.set_resource_input_initialized({true, true, true, true});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  const std::string hlo(tu::GetBroadcastHloString(100));
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  TF_ASSERT_OK_AND_ASSIGN(bool custom_op_replaced,
                          CustomOpReplacer().Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(
      bool offloaded,
      VariablesOffloadAndPartition(annotations, true, 0, replication_factor)
          .Run(module.get()));
  EXPECT_TRUE(offloaded);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ReplicatedResourceUpdateElementwiseClustering(
                              annotations, replication_factor)
                              .Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* fusion = FindInstruction(module.get(), "fusion");
  HloComputation* resource_update =
      FindComputation(module.get(), "resource_update");

  // Fusion should not be in the cluster, but it could be one of the inputs.
  EXPECT_EQ(resource_update, fusion->parent());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
