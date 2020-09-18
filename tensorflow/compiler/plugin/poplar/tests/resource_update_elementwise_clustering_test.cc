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
#include "tensorflow/compiler/plugin/poplar/driver/passes/variables_offload_and_partition.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

std::string GetTemplateHloString(const std::string& wu, int n, int m) {
  const std::string hlo = R"(
  HloModule main

  sum {
    y = f32[] parameter(1)
    x = f32[] parameter(0), control-predecessors={y}
    ROOT add = f32[] add(x, y), backend_config="{\"isInplace\":true}"
  }

  scale_xya.1 {
    arg0 = f32[$N,$M] parameter(0)
    arg1 = f32[$N,$M] parameter(1)
    arg2 = f32[] parameter(2)
    broadcast = f32[$N,$M] broadcast(arg2), dimensions={}
    mul.1 = f32[$N,$M] multiply(arg0, broadcast)
    ROOT r = f32[$N,$M] add(mul.1, arg1)
  }

  scale_xya.2 {
    arg0 = f32[$N,$M] parameter(0)
    arg1 = f32[$N,$M] parameter(1)
    arg2 = f32[] parameter(2)
    broadcast = f32[$N,$M] broadcast(arg2), dimensions={}
    mul.1 = f32[$N,$M] multiply(arg0, broadcast)
    ROOT r = f32[$N,$M] add(mul.1, arg1)
  }

  scale_xa {
    arg0 = f32[$N,$M] parameter(0)
    arg1 = f32[] parameter(1)
    broadcast = f32[$N,$M] broadcast(arg1), dimensions={}
    ROOT mul.1 = f32[$N,$M] multiply(arg0, broadcast)
  }

  inplace_xa {
    arg0 = f32[$N,$M] parameter(0)
    const = f32[] constant(1.1)
    broadcast = f32[$N,$M] broadcast(const), dimensions={}
    ROOT mul.1 = f32[$N,$M] multiply(arg0, broadcast)
  }

  inplace_xya.1 {
    arg0 = f32[$N,$M] parameter(0)
    arg1 = f32[$N,$M] parameter(1)
    const = f32[] constant(1.1)
    broadcast = f32[$N,$M] broadcast(const), dimensions={}
    mul.1 = f32[$N,$M] multiply(arg0, broadcast)
    ROOT r = f32[$N,$M] add(mul.1, arg1)
  }
  inplace_xya.2 {
    arg0 = f32[$N,$M] parameter(0)
    arg1 = f32[$N,$M] parameter(1)
    const = f32[] constant(1.1)
    broadcast = f32[$N,$M] broadcast(const), dimensions={}
    mul.1 = f32[$N,$M] multiply(arg0, broadcast)
    ROOT r = f32[$N,$M] add(mul.1, arg1)
  }

  resource_update {
    arg0 = f32[$N,$M] parameter(0)
    arg1 = f32[$N,$M] parameter(1)
    arg2 = f32[$N,$M] parameter(2)
    arg3 = f32[$N,$M] parameter(3)

    $WU
  }

  loop {
    after-all = token[] after-all()
    infeed = (f32[$N,$M], token[]) infeed(after-all), infeed_config="140121807314576"
    input = f32[$N,$M] get-tuple-element(infeed), index=0

    arg0 = f32[$N,$M] parameter(0)
    arg1 = f32[$N,$M] parameter(1)
    arg2 = f32[$N,$M] parameter(2)
    arg3 = f32[$N,$M] parameter(3)
    add.1 = f32[$N,$M] add(input, arg0)
    add.2 = f32[$N,$M] add(input, arg1)
    call = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) call(add.1, add.2, arg2, arg3), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\", \"partitionOffloadedVariables\":\"THREESTATE_ON\"}}}"
    gte0 = f32[$N,$M] get-tuple-element(call), index=0
    gte1 = f32[$N,$M] get-tuple-element(call), index=1
    gte2 = f32[$N,$M] get-tuple-element(call), index=2
    gte3 = f32[$N,$M] get-tuple-element(call), index=3
    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(gte0, gte1, gte2, gte3)
  }

  ENTRY e {
    e.in0 = f32[$N,$M] parameter(0)
    e.in1 = f32[$N,$M] parameter(1)
    e.in2 = f32[$N,$M] parameter(2)
    e.in3 = f32[$N,$M] parameter(3)
    call = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) call(e.in0, e.in1, e.in2, e.in3), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
    gte0 = f32[$N,$M] get-tuple-element(call), index=0
    gte1 = f32[$N,$M] get-tuple-element(call), index=1
    gte2 = f32[$N,$M] get-tuple-element(call), index=2
    gte3 = f32[$N,$M] get-tuple-element(call), index=3
    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(gte0, gte1, gte2, gte3)
  }
  )";
  std::string hlo_string =
      tensorflow::str_util::StringReplace(hlo, "$WU", wu, true);
  hlo_string = tensorflow::str_util::StringReplace(hlo_string, "$N",
                                                   std::to_string(n), true);
  hlo_string = tensorflow::str_util::StringReplace(hlo_string, "$M",
                                                   std::to_string(m), true);
  return hlo_string;
}

std::string GetAdamLikeHloString(int n, int m) {
  const std::string wu = R"(
    beta.1 = f32[] constant(0.1)
    beta.2 = f32[] constant(0.01)
    const.55 = f32[] constant(0.9)

    all-reduce = f32[$N,$M] all-reduce(arg0), to_apply=sum
    replication-normalise = f32[$N,$M] custom-call(all-reduce), custom_call_target="ReplicationNormalise"
    fusion.2 = f32[$N,$M] fusion(arg2, replication-normalise), kind=kCustom, calls=inplace_xya.1, backend_config="{\"fusionConfig\":{\"inplaceOperands\":[\"0\"]}}"

    constant.54 = f32[] constant(0.001)
    constant.58 = f32[] constant(1)
    subtract.60 = f32[] subtract(constant.58, beta.2)
    sqrt.61 = f32[] sqrt(f32[] subtract.60)
    multiply.62 = f32[] multiply(constant.54, sqrt.61)
    subtract.59 = f32[] subtract(constant.58, beta.1)
    divide.63 = f32[] divide(multiply.62, subtract.59)

    fusion.4 = f32[$N,$M] fusion(fusion.2, divide.63), kind=kCustom, calls=scale_xa
    multiply.71 = f32[$N,$M] multiply(replication-normalise, replication-normalise)
    fusion.1 = f32[$N,$M] fusion(arg3, multiply.71), kind=kCustom, calls=inplace_xya.2, backend_config="{\"fusionConfig\":{\"inplaceOperands\":[\"0\"]}}"
    sqrt.77 = f32[$N,$M] sqrt(fusion.1)
    fusion.5 = f32[$N,$M] fusion(sqrt.77), kind=kCustom, calls=inplace_xa, backend_config="{\"fusionConfig\":{\"inplaceOperands\":[\"0\"]}}"
    divide.82 = f32[$N,$M] divide(fusion.4, fusion.5)
    subtract.83 = f32[$N,$M] subtract(arg1, divide.82)
    ROOT root = (f32[$N,$M], f32[$N,$M], f32[$N,$M], f32[$N,$M]) tuple(subtract.83, fusion.2, arg0, arg1)
  )";
  return GetTemplateHloString(wu, n, m);
}

std::string GetMomentumLikeHloString(int n, int m) {
  const std::string wu = R"(
  all-reduce = f32[$N,$M] all-reduce(arg0), to_apply=sum
  replication-normalise = f32[$N,$M] custom-call(all-reduce), custom_call_target="ReplicationNormalise"
  fusion.1 = f32[$N,$M] fusion(arg2, replication-normalise), kind=kCustom, calls=inplace_xya.1, backend_config="{\"fusionConfig\":{\"inplaceOperands\":[\"0\"]}}"
  fusion.2 = f32[$N,$M] fusion(arg3, replication-normalise), kind=kCustom, calls=inplace_xya.2, backend_config="{\"fusionConfig\":{\"inplaceOperands\":[\"0\"]}}"
  ROOT root = (f32[$N,$M], f32[$N,$M], f32[$N,$M], f32[$N,$M]) tuple(fusion.1, fusion.2, arg0, arg1)
)";
  return GetTemplateHloString(wu, n, m);
}

std::string GetSGDHloString(int n, int m) {
  const std::string wu = R"(
    all-reduce.1 = f32[$N,$M] all-reduce(arg0), to_apply=sum
    all-reduce.2 = f32[$N,$M] all-reduce(arg1), to_apply=sum
    replication-normalise.1 = f32[$N,$M] custom-call(all-reduce.1), custom_call_target="ReplicationNormalise"
    replication-normalise.2 = f32[$N,$M] custom-call(all-reduce.2), custom_call_target="ReplicationNormalise"
    fusion.1 = f32[$N,$M] fusion(arg2, replication-normalise.1), kind=kCustom, calls=inplace_xya.1, backend_config="{\"fusionConfig\":{\"inplaceOperands\":[\"0\"]}}"
    fusion.2 = f32[$N,$M] fusion(arg3, replication-normalise.2), kind=kCustom, calls=inplace_xya.2, backend_config="{\"fusionConfig\":{\"inplaceOperands\":[\"0\"]}}"

    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(arg0, arg1, fusion.1, fusion.2)
  )";
  return GetTemplateHloString(wu, n, m);
}

std::string GetSimpleHloString(int n, int m) {
  const std::string wu = R"(
    all-reduce.1 = f32[$N,$M] all-reduce(arg0), to_apply=sum
    all-reduce.2 = f32[$N,$M] all-reduce(arg1), to_apply=sum

    add.1 = f32[$N,$M] add(arg2, all-reduce.1)
    add.2 = f32[$N,$M] add(arg3, all-reduce.2)

    rate.1 = f32[] constant(0.1)
    rate.2 = f32[1] reshape(rate.1)
    rate.3 = f32[] reshape(rate.2)
    fusion.1 = f32[$N,$M] fusion(add.1, add.2, rate.1), kind=kCustom, calls=scale_xya.1
    fusion.2 = f32[$N,$M] fusion(add.1, add.2, rate.3), kind=kCustom, calls=scale_xya.2

    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(arg0, arg1, fusion.1, fusion.2)
  )";
  return GetTemplateHloString(wu, n, m);
}

struct ResourceUpdateElementwiseClusteringTestSpec {
  std::string hlo;
  std::string short_name;
  bool cluster;
  int expected_offloads;
  int expected_all_gathers;
  // These are all-gathers which are not arguments to the root instruction.
  int expected_non_root_all_gathers;
};

std::ostream& operator<<(
    std::ostream& os, const ResourceUpdateElementwiseClusteringTestSpec& spec) {
  return os << "{ name: " << spec.short_name << ", cluster: " << spec.cluster
            << ", offloads: " << spec.expected_offloads
            << ", all-gathers: " << spec.expected_all_gathers
            << ", non-root-all-gathers: " << spec.expected_non_root_all_gathers
            << "}";
}

class ResourceUpdateElementwiseClusteringTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          ResourceUpdateElementwiseClusteringTestSpec> {};

INSTANTIATE_TEST_SUITE_P(
    ResourceUpdateElementwiseClusteringTestCases,
    ResourceUpdateElementwiseClusteringTest,
    ::testing::ValuesIn(std::vector<
                        ResourceUpdateElementwiseClusteringTestSpec>{
        // Simple HLO with all inputs
        {GetSimpleHloString(20, 100), "simple", false, 2, 2, 2},
        {GetSimpleHloString(20, 100), "simple", true, 2, 0, 0},
        // Check padded offloading:
        {GetSimpleHloString(11, 13), "simple-padded", false, 2, 2, 2},
        {GetSimpleHloString(11, 13), "simple-padded", true, 2, 0, 0},
        // Adam-like resource update
        {GetAdamLikeHloString(20, 100), "adam", false, 2, 2, 2},
        // We still have to do all-gathers, but they all are operands to root
        // instruction
        {GetAdamLikeHloString(20, 100), "adam", true, 2, 2, 0},
        // Momentum-like resource update
        {GetMomentumLikeHloString(1000, 20), "momentum", false, 2, 2, 2},
        {GetMomentumLikeHloString(1000, 20), "momentum", true, 2, 2, 0},
        // SGD-like resource update
        {GetSGDHloString(1000, 20), "sgd", false, 2, 2, 2},
        {GetSGDHloString(1000, 20), "sgd", true, 2, 0, 0},
    }));

TEST_P(ResourceUpdateElementwiseClusteringTest, DoTest) {
  auto param = GetParam();

  static constexpr int replication_factor = 2;
  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3});
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
    TF_ASSERT_OK_AND_ASSIGN(
        bool changed, ResourceUpdateElementwiseClustering(replication_factor)
                          .Run(module.get()));
    EXPECT_TRUE(changed);
  }
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
  auto insts = resource_update->instructions();
  EXPECT_EQ(absl::c_count_if(
                insts, IsPoplarInstruction(PoplarOp::RemoteParameterStore)),
            param.expected_offloads);
  EXPECT_EQ(absl::c_count_if(
                insts, IsPoplarInstruction(PoplarOp::RemoteParameterLoad)),
            param.expected_offloads);
  EXPECT_EQ(absl::c_count_if(insts, IsPoplarInstruction(PoplarOp::AllGather)),
            param.expected_all_gathers);

  auto IsNonRootAllGather = [&](HloInstruction* inst) {
    if (!IsPoplarInstruction(PoplarOp::AllGather)(inst)) {
      return false;
    }
    if (inst->user_count() != 1) {
      return true;
    }
    auto user = inst->users()[0];
    if (user->opcode() == HloOpcode::kReshape && user->user_count() == 1) {
      user = user->users()[0];
    }
    return user != resource_update_root;
  };
  EXPECT_EQ(absl::c_count_if(insts, IsNonRootAllGather),
            param.expected_non_root_all_gathers);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
