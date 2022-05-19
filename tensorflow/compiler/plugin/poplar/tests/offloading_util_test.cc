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

#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/variables_offload_and_partition.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_util.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {

namespace {

using tu = HloPoplarTestUtil;

struct OffloadingUtilRemoteBufferNumberTestSpec {
  std::string hlo;
  std::string short_name;
  std::vector<int64_t> param_nums;
};

std::ostream& operator<<(std::ostream& os,
                         const OffloadingUtilRemoteBufferNumberTestSpec& spec) {
  return os << "{ name: " << spec.short_name << "}";
}

class OffloadingUtilRemoteBufferNumberTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          OffloadingUtilRemoteBufferNumberTestSpec> {};

INSTANTIATE_TEST_SUITE_P(
    OffloadingUtilRemoteBufferNumberTestCases,
    OffloadingUtilRemoteBufferNumberTest,
    ::testing::ValuesIn(std::vector<OffloadingUtilRemoteBufferNumberTestSpec>{
        {tu::GetSimpleHloString(20, 100), "simple", {2, 3}},
        {tu::GetTwoClustersShareInputHloString(20, 100), "2-clusters", {2, 3}},
        {tu::GetAdamLikeHloString(20, 100), "adam", {2, 3}},
        {tu::GetMomentumLikeHloString(1000, 20), "momentum", {2, 3}},
        {tu::GetSGDHloString(1000, 20), "sgd", {2, 3}},
        {tu::GetFullRemoteLoadHloString(100, 20), "full-remote-load", {2, 3}},
    }));

TEST_P(OffloadingUtilRemoteBufferNumberTest, DoTest) {
  auto& param = GetParam();
  auto config = GetModuleConfigForTest();
  config.set_argument_input_indices({});
  config.set_resource_input_indices({0, 1, 2, 3});
  config.set_resource_input_initialized({true, true, true, true});
  config.set_resource_update_to_input_index({0, 1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(param.hlo, config));
  TF_ASSERT_OK_AND_ASSIGN(bool custom_op_replaced,
                          CustomOpReplacer().Run(module.get()));
  CompilerAnnotations annotations(module.get());
  TF_ASSERT_OK_AND_ASSIGN(
      bool offloaded,
      VariablesOffloadAndPartition(
          annotations, /*remote_memory_supported*/ true,
          /*minimum_remote_tensor_size=*/4, /*partition_replication_factor=*/2)
          .Run(module.get()));
  EXPECT_TRUE(offloaded);

  TF_ASSERT_OK_AND_ASSIGN(auto dfa, HloDataflowAnalysis::Run(*module));

  HloInstruction* resource_update = nullptr;
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (IsResourceUpdate(inst)) {
        resource_update = inst;
        break;
      }
    }
  }
  CHECK_NOTNULL(resource_update);
  HloComputation* resource_update_comp = resource_update->to_apply();
  std::vector<int64_t> param_nums;
  for (int64_t i = 0; i < resource_update_comp->num_parameters(); ++i) {
    auto param = resource_update_comp->parameter_instruction(i);
    auto result = GetRemoteBufferEntryParameterNumber(*dfa, param);
    if (result.ok()) {
      param_nums.push_back(result.ValueOrDie());
    }
  }
  EXPECT_THAT(param_nums, ::testing::Eq(param.param_nums));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
