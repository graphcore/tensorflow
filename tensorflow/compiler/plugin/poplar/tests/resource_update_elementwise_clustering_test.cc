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
    
    convert.1 = f16[$N,$M] convert(fusion.1)
    convert.2 = f32[$N,$M] convert(convert.1)

    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(arg0, arg1, convert.2, fusion.2)
  )";
  return GetTemplateHloString(wu, n, m);
}

std::string GetTwoClustersShareInputHloString(int n, int m) {
  const std::string wu = R"(
    all-reduce.1 = f32[$N,$M] all-reduce(arg0), to_apply=sum
    all-reduce.2 = f32[$N,$M] all-reduce(arg1), to_apply=sum

    add.1 = f32[$N,$M] add(arg2, all-reduce.1)
    add.2 = f32[$N,$M] add(arg3, all-reduce.1)

    rate.1 = f32[] constant(0.1)
    rate.2 = f32[1] reshape(rate.1)
    rate.3 = f32[] reshape(rate.2)
    fusion.1 = f32[$N,$M] fusion(add.1, all-reduce.2, rate.1), kind=kCustom, calls=scale_xya.1
    fusion.2 = f32[$N,$M] fusion(add.2, all-reduce.2, rate.3), kind=kCustom, calls=scale_xya.2

    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(arg0, arg1, fusion.1, fusion.2)
  )";
  return GetTemplateHloString(wu, n, m);
}

std::string GetFullRemoteLoadHloString(int n, int m) {
  const std::string wu = R"(
    buffer.1 = f32[$N,$M] custom-call(arg0), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}"
    buffer.2 = f32[$N,$M] custom-call(arg1), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}"

    add.1 = f32[$N,$M] add(arg2, buffer.1)
    add.2 = f32[$N,$M] add(arg3, buffer.2)

    mul.1 = f32[$N,$M] multiply(add.1, buffer.1)
    mul.2 = f32[$N,$M] multiply(add.2, buffer.2)

    rate.1 = f32[] constant(0.1)
    rate.2 = f32[1] reshape(rate.1)
    rate.3 = f32[] reshape(rate.2)
    fusion.1 = f32[$N,$M] fusion(mul.1, mul.2, rate.1), kind=kCustom, calls=scale_xya.1
    fusion.2 = f32[$N,$M] fusion(mul.1, mul.2, rate.3), kind=kCustom, calls=scale_xya.2

    store.1 = f32[$N,$M] custom-call(arg0, fusion.1), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}"
    store.2 = f32[$N,$M] custom-call(arg1, fusion.2), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}"

    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(store.1, store.2, fusion.1, fusion.2)
  )";
  return GetTemplateHloString(wu, n, m);
}

using ResourceUpdateElementwiseClusteringBasicTest = HloTestBase;

TEST_F(ResourceUpdateElementwiseClusteringBasicTest,
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
  auto elementwise_comps = ResourceUpdateElementwiseClustering::
      GetElementwiseClusterableComputations(module.get());
  CHECK_EQ(elementwise_comps.size(), 2);
  absl::flat_hash_set<std::string> elementwise_comp_names = {"_comp0",
                                                             "_comp2"};
  for (auto comp : elementwise_comps) {
    EXPECT_TRUE(elementwise_comp_names.contains(comp->name()));
  }
}

std::string GetHlo(const std::vector<int64>& dimensions,
                   const PrimitiveType& element_type,
                   const PrimitiveType& remote_buffer_element_type) {
  const std::string hlo = R"(
  HloModule main

  sum {
    y = $element_type[] parameter(1)
    x = $element_type[] parameter(0), control-predecessors={y}
    ROOT add = $element_type[] add(x, y), backend_config="{\"isInplace\":true}"
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
    ROOT t = ($element_type$shape,$remote_buffer_element_type$shape) tuple(arg1_new, arg2_new_c)
  }

  loop {
    after-all = token[] after-all()
    infeed = ($element_type$shape, token[]) infeed(after-all), infeed_config="140121807314576"
    input = $element_type$shape get-tuple-element(infeed), index=0

    arg0 = $element_type$shape parameter(0)
    arg1 = $remote_buffer_element_type$shape parameter(1)

    add.1 = $element_type$shape add(input, arg0)
    call = ($element_type$shape,$remote_buffer_element_type$shape) call(add.1, arg0, arg1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\", \"partitionOffloadedVariables\":\"THREESTATE_ON\"}}}"
    gte0 = $element_type$shape get-tuple-element(call), index=0
    gte1 = $remote_buffer_element_type$shape get-tuple-element(call), index=1
    ROOT r = ($element_type$shape,$remote_buffer_element_type$shape) tuple(gte0, gte1)
  }

  ENTRY e {
    e.in0 = $element_type$shape parameter(0)
    e.in1 = $remote_buffer_element_type$shape parameter(1)
    call = ($element_type$shape,$remote_buffer_element_type$shape) call(e.in0, e.in1), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
    gte0 = $element_type$shape get-tuple-element(call), index=0
    gte1 = $remote_buffer_element_type$shape get-tuple-element(call), index=1
    ROOT r = ($element_type$shape,$remote_buffer_element_type$shape) tuple(gte0, gte1)
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

StatusOr<HloInstruction*> GetNextUser(HloInstruction* inst) {
  if (inst->user_count() != 1) {
    return FailedPrecondition("Expected single user.");
  }
  return inst->users()[0];
}

StatusOr<HloInstruction*> GetRewrittenInput(HloInstruction* inst, bool padded) {
  if (padded) {
    TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
    EXPECT_THAT(inst->opcode(), HloOpcode::kReshape);
    TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
    EXPECT_THAT(inst->opcode(), HloOpcode::kPad);
  }
  TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
  EXPECT_THAT(inst->opcode(), HloOpcode::kReshape);
  TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
  EXPECT_THAT(inst->opcode(), HloOpcode::kDynamicSlice);
  TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
  EXPECT_THAT(inst->opcode(), HloOpcode::kReshape);
  return inst;
}

StatusOr<HloInstruction*> GetRewrittenOutput(HloInstruction* inst,
                                             bool sliced) {
  TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::AllGather)(inst));
  TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
  EXPECT_THAT(inst->opcode(), HloOpcode::kReshape);
  if (sliced) {
    TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
    EXPECT_THAT(inst->opcode(), HloOpcode::kSlice);
    TF_ASSIGN_OR_RETURN(inst, GetNextUser(inst));
    EXPECT_THAT(inst->opcode(), HloOpcode::kReshape);
  }
  return inst;
}

struct ResourceUpdateElementwiseClusteringShapeTestSpec {
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
    const ResourceUpdateElementwiseClusteringShapeTestSpec& spec) {
  return os << "{ dimensions: [" << absl::StrJoin(spec.dimensions, "],")
            << ", cluster_size; " << spec.cluster_size
            << ", aligned_cluster_size; " << spec.aligned_cluster_size
            << ", shard_size; " << spec.shard_size
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
            {{128, 2, 2}, 512, 512, 256, F32, F32, false},
            {{128}, 128, 128, 64, F32, F32, false},
            {{128, 2, 2}, 512, 512, 256, F16, F32, false},
            {{128, 2, 2}, 512, 512, 256, F32, F16, false},
            {{129, 3}, 387, 388, 194, F32, F32, true},
            {{1}, 1, 2, 1, F32, F32, true},
            {{127, 5}, 635, 636, 318, F16, F32, true},
            {{127, 5}, 635, 636, 318, F32, F16, true},
        }));

TEST_P(ResourceUpdateElementwiseClusteringShapeTest, DoTest) {
  auto param = GetParam();

  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1});
  config.set_resource_update_to_input_index({0, 1});
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(GetHlo(param.dimensions, param.element_type,
                                          param.remote_buffer_element_type),
                                   config));

  HloComputation* resource_update =
      FindComputation(module.get(), "resource_update");

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

  auto elementwise_comps = ResourceUpdateElementwiseClustering::
      GetElementwiseClusterableComputations(module.get());
  auto clusters = ResourceUpdateElementwiseClustering::GetClustersIn(
      resource_update, elementwise_comps);
  EXPECT_THAT(clusters.size(), 1);
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
  TF_ASSERT_OK_AND_ASSIGN(
      bool converted,
      ResourceUpdateElementwiseClustering::OutlineCluster(cluster, 2));
  EXPECT_TRUE(converted);

  TF_ASSERT_OK_AND_ASSIGN(arg0_r,
                          GetRewrittenInput(arg0_r, param.padded_and_sliced));
  EXPECT_THAT(arg0_r->shape(),
              ShapeUtil::MakeShape(param.element_type, {param.shard_size}));

  TF_ASSERT_OK_AND_ASSIGN(arg1,
                          GetRewrittenInput(arg1, param.padded_and_sliced));
  EXPECT_THAT(arg1->shape(),
              ShapeUtil::MakeShape(param.element_type, {param.shard_size}));

  HloInstruction *arg2_load, *arg2_store;
  TF_ASSERT_OK(GetRemoteLoadStoreUsers(arg2, &arg2_load, &arg2_store));
  EXPECT_THAT(arg2_load->shape(),
              ShapeUtil::MakeShape(param.remote_buffer_element_type,
                                   {param.shard_size}));

  EXPECT_THAT(arg2_c->operands(), ::testing::ElementsAre(arg2_load));
  EXPECT_THAT(arg2_c->shape(),
              ShapeUtil::MakeShape(param.element_type, {param.shard_size}));

  EXPECT_THAT(arg2_new->operands(), ::testing::ElementsAre(arg0_r, arg2_c));
  EXPECT_THAT(arg2_new->shape(),
              ShapeUtil::MakeShape(param.element_type, {param.shard_size}));

  EXPECT_THAT(arg1_new->operands(), ::testing::ElementsAre(arg1, arg2_new));
  EXPECT_THAT(arg1_new->shape(),
              ShapeUtil::MakeShape(param.element_type, {param.shard_size}));

  EXPECT_THAT(arg2_new_c->operands(), ::testing::ElementsAre(arg2_new));
  EXPECT_THAT(arg2_new_c->shape(),
              ShapeUtil::MakeShape(param.remote_buffer_element_type,
                                   {param.shard_size}));

  EXPECT_THAT(arg2_store->operands(), ::testing::ElementsAre(arg2, arg2_new_c));

  TF_ASSERT_OK_AND_ASSIGN(
      arg1_new, GetRewrittenOutput(arg1_new, param.padded_and_sliced));
  EXPECT_THAT(arg1_new->shape(),
              ShapeUtil::MakeShape(param.element_type, param.dimensions));

  EXPECT_THAT(resource_update->root_instruction()->operands(),
              ::testing::ElementsAre(arg1_new, arg2_store));
  TF_ASSERT_OK_AND_ASSIGN(bool eliminated, HloDCE().Run(module.get()));
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
        // Simple HLO with all types of the inputs
        {GetSimpleHloString(20, 100), "simple", false, 2, 2, 2},
        {GetSimpleHloString(20, 100), "simple", true, 2, 0, 0},
        // Edge case
        {GetSimpleHloString(1, 1), "1x1", false, 2, 2, 2},
        {GetSimpleHloString(1, 1), "1x1", true, 2, 0, 0},
        // Check padded offloading:
        {GetSimpleHloString(11, 13), "simple-padded", false, 2, 2, 2},
        {GetSimpleHloString(11, 13), "simple-padded", true, 2, 0, 0},
        // Two cluster share the same inputs
        {GetTwoClustersShareInputHloString(20, 100), "2-clusters", false, 2, 2,
         2},
        {GetTwoClustersShareInputHloString(20, 100), "2-clusters", true, 2, 0,
         0},
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
        // Test with one of the arguments be non-replicated remote buffer.
        {GetFullRemoteLoadHloString(100, 20), "full-remote-load", false, 4, 2,
         2},
        {GetFullRemoteLoadHloString(100, 20), "full-remote-load", true, 4, 2,
         2},
        // And unaligned
        {GetFullRemoteLoadHloString(19, 7), "full-remote-load", false, 4, 2, 2},
        {GetFullRemoteLoadHloString(19, 7), "full-remote-load", true, 4, 2, 2},
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
