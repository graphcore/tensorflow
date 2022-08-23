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

#include "tensorflow/compiler/plugin/poplar/driver/passes/resource_update_elementwise_clustering.h"

#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fusion_inliner.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/variables_offload_and_partition.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/elementwise_cluster.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
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

StatusOr<HloInstruction*> GetAllReduceInput(HloInstruction* inst) {
  TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
  TF_ASSIGN_OR_RETURN(inst, LookThroughReshape(inst));
  EXPECT_THAT(inst->opcode(), HloOpcode::kAllReduce);
  return inst;
}

int64_t GetCount(const HloComputation* comp,
                 std::function<bool(const HloInstruction*)> pred) {
  int64_t count = 0;
  for (auto inst : comp->instructions()) {
    count += pred(inst);
    for (auto called_comp : inst->called_computations()) {
      count += GetCount(called_comp, pred);
    }
  }
  return count;
}

struct ResourceUpdateElementwiseClusteringShapeTestSpec {
  std::vector<int64_t> dimensions;
  int64_t cluster_size;
  PrimitiveType element_type;
  PrimitiveType remote_buffer_element_type;
  bool padded_and_sliced;

  std::string GetHlo() const {
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
    frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"},
    backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\",
    \"partitionOffloadedVariables\":\"THREESTATE_OFF\"}}}" gte0 =
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
        hlo_string, "$element_type",
        primitive_util::LowercasePrimitiveTypeName(element_type), true);
    hlo_string = tensorflow::str_util::StringReplace(
        hlo_string, "$remote_buffer_element_type",
        primitive_util::LowercasePrimitiveTypeName(remote_buffer_element_type),
        true);
    return hlo_string;
  }
};

std::ostream& operator<<(
    std::ostream& os,
    const ResourceUpdateElementwiseClusteringShapeTestSpec& spec) {
  return os << "{ dimensions: [" << absl::StrJoin(spec.dimensions, ",")
            << "], cluster_size: " << spec.cluster_size
            << ", element_type: " << spec.element_type
            << ", remote_buffer_element_type: "
            << spec.remote_buffer_element_type
            << ", padded_and_sliced: " << spec.padded_and_sliced << "}";
}

class ResourceUpdateElementwiseClusteringShapeTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          ResourceUpdateElementwiseClusteringShapeTestSpec> {};

INSTANTIATE_TEST_SUITE_P(
    ResourceUpdateElementwiseClusteringShapeTestCases,
    ResourceUpdateElementwiseClusteringShapeTest,
    ::testing::ValuesIn(
        std::vector<ResourceUpdateElementwiseClusteringShapeTestSpec>{
            {{128, 2, 2}, 512, F32, F32, false},
            {{128}, 128, F32, F32, false},
            {{128, 2, 2}, 512, F16, F32, false},
            {{128, 2, 2}, 512, F32, F16, false},
            {{129, 3}, 387, F32, F32, true},
            {{1}, 1, F32, F32, true},
            {{127, 5}, 635, F16, F32, true},
            {{127, 5}, 635, F32, F16, true},
            {{128}, 128, F32, F32, false},
            {{128, 2, 2}, 512, F16, F32, false},
            {{128}, 128, F32, F32, false},
            {{128, 2, 2}, 512, F16, F32, false},
        }));

TEST_P(ResourceUpdateElementwiseClusteringShapeTest, DoTest) {
  auto param = GetParam();

  auto config = GetModuleConfigForTest();
  config.set_argument_input_indices({});
  config.set_resource_input_indices({0, 1});
  config.set_resource_input_initialized({true, true});
  config.set_resource_update_to_input_index({0, 1});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(param.GetHlo(), config));

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

  ResourceUpdateElementwiseClustering pass;
  TF_ASSERT_OK_AND_ASSIGN(auto clusters, pass.GetClustersIn(loop));
  ASSERT_THAT(clusters.size(), 1);
  auto& cluster = *std::begin(clusters);
  EXPECT_THAT(cluster.GetClusterSize(), param.cluster_size);
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
  HloInstruction* collective = cluster_comp->parameter_instruction(1);
  TF_ASSERT_OK_AND_ASSIGN(collective, GetAllReduceInput(collective));
  EXPECT_THAT(collective->shape(),
              ShapeUtil::MakeShape(param.element_type, {param.dimensions}));

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
  EXPECT_THAT(arg1_new->shape(),
              ShapeUtil::MakeShape(param.element_type, param.dimensions));

  EXPECT_THAT(arg2_new->user_count(), 2);
  auto arg2_new_c_it =
      absl::c_find_if(arg2_new->users(), [](const HloInstruction* i) {
        return i->opcode() == HloOpcode::kConvert;
      });
  CHECK(arg2_new_c_it != arg2_new->users().end());
  arg2_new_c = *arg2_new_c_it;

  EXPECT_THAT(arg2_new_c->operands(), ::testing::ElementsAre(arg2_new));
  EXPECT_THAT(arg1_new->shape(),
              ShapeUtil::MakeShape(param.element_type, param.dimensions));

  EXPECT_THAT(arg2_new_c->user_count(), 1);

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

using ResourceUpdateElementwiseClusteringOutlineTests = HloTestBase;

TEST_F(ResourceUpdateElementwiseClusteringOutlineTests, TestSameCluster) {
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

    resource_update = (f16[128],f16[128],f16[128],f16[128]) call(add.1, add.2, l.arg0, l.arg1, l.arg2, l.arg3, l.arg4), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\", \"partitionOffloadedVariables\":\"THREESTATE_OFF\"}}}"
    gte0 = f16[128] get-tuple-element(resource_update), index=0
    gte1 = f16[128] get-tuple-element(resource_update), index=1
    gte2 = f16[128] get-tuple-element(resource_update), index=2
    gte3 = f16[128] get-tuple-element(resource_update), index=3
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
  int64_t replication_factor = 1;
  CompilerAnnotations annotations(module.get());
  HloInstruction* resource_update =
      FindInstruction(module.get(), "resource_update");
  // Check that there were no functions before.
  ASSERT_EQ(GetCount(resource_update->to_apply(), IsFunction), 0);
  TF_ASSERT_OK_AND_ASSIGN(
      bool offloaded,
      VariablesOffloadAndPartition(annotations, true, 4, replication_factor)
          .Run(module.get()));
  EXPECT_TRUE(offloaded);

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, ResourceUpdateElementwiseClustering().Run(module.get()));
  EXPECT_TRUE(changed);
  // Check that there are functions now due to the elementwise clusters which
  // were outlined.
  ASSERT_EQ(GetCount(resource_update->to_apply(), IsFunction), 2);
}

TEST_F(ResourceUpdateElementwiseClusteringOutlineTests, TestDifferentCluster) {
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
    arg5_div = f16[128] divide(bcast, arg5_new)

    arg2_new = f16[128] add(arg2, arg4_mul)
    arg3_new = f16[128] add(arg3, arg5_div)

    ROOT t = (f16[128],f16[128],f16[128],f16[128]) tuple(arg2_new, arg3_new, arg4_new, arg5_new)
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

    resource_update = (f16[128],f16[128],f16[128],f16[128]) call(add.1, add.2, l.arg0, l.arg1, l.arg2, l.arg3, l.arg4), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\", \"partitionOffloadedVariables\":\"THREESTATE_OFF\"}}}"
    gte0 = f16[128] get-tuple-element(resource_update), index=0
    gte1 = f16[128] get-tuple-element(resource_update), index=1
    gte2 = f16[128] get-tuple-element(resource_update), index=2
    gte3 = f16[128] get-tuple-element(resource_update), index=3
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
  int64_t replication_factor = 1;
  CompilerAnnotations annotations(module.get());
  HloInstruction* resource_update =
      FindInstruction(module.get(), "resource_update");
  // Check that there were no functions before.
  ASSERT_EQ(GetCount(resource_update->to_apply(), IsFunction), 0);
  TF_ASSERT_OK_AND_ASSIGN(
      bool offloaded,
      VariablesOffloadAndPartition(annotations, true, 4, replication_factor)
          .Run(module.get()));
  EXPECT_TRUE(offloaded);

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, ResourceUpdateElementwiseClustering().Run(module.get()));
  EXPECT_TRUE(changed);
  // Check that there are no functions due to the elementwise clusters being
  // different.
  ASSERT_EQ(GetCount(resource_update->to_apply(), IsFunction), 0);
}

TEST_F(ResourceUpdateElementwiseClusteringOutlineTests, TestCycle) {
  const std::string hlo = R"(
  HloModule main

  sum {
    y = f16[] parameter(1)
    x = f16[] parameter(0)
    ROOT add = f16[] add(x, y)
  }

  resource_update {
    arg0 = f16[128] parameter(0)
    arg1 = f16[128] parameter(1)

    arg2 = f16[128] parameter(2)
    arg3 = f16[128] parameter(3)
    arg4 = f16[128] parameter(4)
    arg5 = f16[128] parameter(5)
    arg6 = f16[] parameter(6)

    c0 = f16[] constant(0)
    sum0 = f16[128] add(arg2, arg3)
    sum1 = f16[128] add(sum0, arg5)

    r0 = f16[] reduce(sum0, c0), dimensions={0}, to_apply=sum
    br0 = f16[128] broadcast(r0), dimensions={}
    div0 = f16[128] divide(sum1, br0)

    ROOT t = (f16[128],f16[128],f16[128],f16[128]) tuple(arg2, arg3, sum0, div0)
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

    resource_update = (f16[128],f16[128],f16[128],f16[128]) call(add.1, add.2, l.arg0, l.arg1, l.arg2, l.arg3, l.arg4), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\", \"partitionOffloadedVariables\":\"THREESTATE_OFF\"}}}"
    gte0 = f16[128] get-tuple-element(resource_update), index=0
    gte1 = f16[128] get-tuple-element(resource_update), index=1
    gte2 = f16[128] get-tuple-element(resource_update), index=2
    gte3 = f16[128] get-tuple-element(resource_update), index=3
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
  int64_t replication_factor = 1;
  CompilerAnnotations annotations(module.get());
  HloInstruction* resource_update =
      FindInstruction(module.get(), "resource_update");
  // Check that there were no functions before.
  ASSERT_EQ(GetCount(resource_update->to_apply(), IsFunction), 0);
  TF_ASSERT_OK_AND_ASSIGN(
      bool offloaded,
      VariablesOffloadAndPartition(annotations, true, 4, replication_factor)
          .Run(module.get()));
  EXPECT_TRUE(offloaded);

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, ResourceUpdateElementwiseClustering().Run(module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
