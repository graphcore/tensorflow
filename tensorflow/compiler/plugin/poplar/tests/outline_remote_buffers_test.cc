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

#include "tensorflow/compiler/plugin/poplar/driver/passes/outline_remote_buffers.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using OutlineRemoteBuffersTest = HloTestBase;

std::string GetValidHlo(int64_t n = 0, int64_t replication_factor = 1,
                        int64_t shard_size = 0) {
  std::string hlo = R"(
HloModule module

comp_0 {
  c0_param_0 = f32[$N] parameter(0)
  c0_param_1 = f32[$RN] parameter(1)
  c0_param_2 = f32[$RN] parameter(2)
  c0_param_3 = f32[$RN] parameter(3)
  c0_add = f32[$RN] add(c0_param_2, c0_param_1)
  c0_subtract = f32[$RN] subtract(c0_param_2, c0_param_1)
  ROOT c0_t = (f32[$N], f32[$RN], f32[$RN]) tuple(c0_param_0, c0_add, c0_subtract)
}

comp_1 {
  c1_param_0 = f32[$N] parameter(0)
  c1_param_1 = f32[$RN] parameter(1)
  c1_param_2 = f32[$RN] parameter(2)
  c1_param_3 = f32[$RN] parameter(3)
  c1_add = f32[$RN] add(c1_param_2, c1_param_1)
  c1_subtract = f32[$RN] subtract(c1_param_2, c1_param_1)
  ROOT c1_t = (f32[$N], f32[$RN], f32[$RN]) tuple(c1_param_0, c1_add, c1_subtract)
}

ENTRY main {
  param_0 = f32[$N] parameter(0)
  param_1 = f32[$N] parameter(1)
  param_2 = f32[$N] parameter(2)
  param_3 = f32[$N] parameter(3)
  param_4 = f32[$N] parameter(4)
  param_5 = f32[$N] parameter(5)
  param_6 = f32[$N] parameter(6)
  param_7 = f32[$N] parameter(7)

  load_0 = f32[$RN] custom-call(param_0), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":$R}\n"
  load_1 = f32[$RN] custom-call(param_1), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":$R}\n"
  load_2 = f32[$RN] custom-call(param_2), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":$R}\n"

  c0 = (f32[$N], f32[$RN], f32[$RN]) call(param_6, load_0, load_1, load_2), to_apply=comp_0, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  c0_gte0 = f32[$N] get-tuple-element(c0), index=0
  c0_gte1 = f32[$RN] get-tuple-element(c0), index=1
  c0_gte2 = f32[$RN] get-tuple-element(c0), index=2

  new_param_0 = f32[$N] custom-call(param_0, c0_gte1), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":$R}\n"
  new_param_1 = f32[$N] custom-call(param_1, c0_gte2), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":$R}\n"

  load_3 = f32[$RN] custom-call(param_3), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":$R}\n"
  load_4 = f32[$RN] custom-call(param_4), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":$R}\n"
  load_5 = f32[$RN] custom-call(param_5), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":$R}\n"

  c1 = (f32[$N], f32[$RN], f32[$RN]) call(param_7, load_3, load_4, load_5), to_apply=comp_1, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  c1_gte0 = f32[$N] get-tuple-element(c1), index=0
  c1_gte1 = f32[$RN] get-tuple-element(c1), index=1
  c1_gte2 = f32[$RN] get-tuple-element(c1), index=2

  new_param_3 = f32[$N] custom-call(param_3, c1_gte1), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":$R}\n"
  new_param_4 = f32[$N] custom-call(param_4, c1_gte2), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":$R}\n"

  ROOT t = (f32[$N], f32[$N], f32[$N], f32[$N], f32[$RN], f32[$RN]) tuple(new_param_0, new_param_1, new_param_3, new_param_4, c0_gte0, c1_gte0)
}
)";
  hlo = tensorflow::str_util::StringReplace(
      hlo, "$RN", shard_size ? std::to_string(shard_size) : std::string(),
      true);
  hlo = tensorflow::str_util::StringReplace(
      hlo, "$R", std::to_string(replication_factor), true);
  hlo = tensorflow::str_util::StringReplace(
      hlo, "$N", n ? std::to_string(n) : std::string(), true);

  return hlo;
}

std::string GetValidHloWithPermutation() {
  const string& hlo = R"(
HloModule module

comp_0 {
  c0_param_0 = f32[] parameter(0)
  c0_param_1 = f32[] parameter(1)
  c0_param_2 = f32[] parameter(2)
  c0_add = f32[] add(c0_param_2, c0_param_1)
  c0_subtract = f32[] subtract(c0_param_2, c0_param_1)
  ROOT c0_t = (f32[], f32[], f32[]) tuple(c0_param_0, c0_add, c0_subtract)
}

comp_1 {
  c1_param_0 = f32[] parameter(0)
  c1_param_1 = f32[] parameter(1)
  c1_param_2 = f32[] parameter(2)
  c1_add = f32[] add(c1_param_2, c1_param_1)
  c1_subtract = f32[] subtract(c1_param_2, c1_param_1)
  ROOT c1_t = (f32[], f32[], f32[]) tuple(c1_param_0, c1_add, c1_subtract)
}

ENTRY main {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  param_2 = f32[] parameter(2)
  param_3 = f32[] parameter(3)
  param_4 = f32[] parameter(4)
  param_5 = f32[] parameter(5)

  load_0 = f32[] custom-call(param_0), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n"
  load_1 = f32[] custom-call(param_1), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n"

  c0 = (f32[], f32[], f32[]) call(param_4, load_0, load_1), to_apply=comp_0, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  c0_gte0 = f32[] get-tuple-element(c0), index=0
  c0_gte1 = f32[] get-tuple-element(c0), index=1
  c0_gte2 = f32[] get-tuple-element(c0), index=2

  new_param_0 = f32[] custom-call(param_0, c0_gte2), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n"
  new_param_1 = f32[] custom-call(param_1, c0_gte1), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n"

  load_2 = f32[] custom-call(param_2), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n"
  load_3 = f32[] custom-call(param_3), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n"

  c1 = (f32[], f32[], f32[]) call(param_5, load_2, load_3), to_apply=comp_1, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  c1_gte0 = f32[] get-tuple-element(c1), index=0
  c1_gte1 = f32[] get-tuple-element(c1), index=1
  c1_gte2 = f32[] get-tuple-element(c1), index=2

  new_param_2 = f32[] custom-call(param_2, c1_gte2), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n"
  new_param_3 = f32[] custom-call(param_3, c1_gte1), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n"

  ROOT t = (f32[], f32[], f32[], f32[], f32[], f32[]) tuple(new_param_0, new_param_1, new_param_2, new_param_3, c0_gte0, c1_gte0)
}
)";
  return hlo;
}

std::string GetElementwiseClusterHlo(int64_t n = 0,
                                     int64_t replication_factor = 1,
                                     int64_t shard_size = 0) {
  std::string hlo = R"(
HloModule module

comp_0 {
  c0_param_0 = f32[$N] parameter(0)
  c0_param_1 = f32[$RN] parameter(1)
  c0_param_2 = f32[$RN] parameter(2)
  c0_param_3 = f32[$RN] parameter(3)
  c0_add = f32[$RN] add(c0_param_2, c0_param_1)
  c0_subtract = f32[$RN] subtract(c0_param_2, c0_param_1)
  ROOT c0_t = (f32[$N], f32[$RN], f32[$RN]) tuple(c0_param_0, c0_add, c0_subtract)
}

ENTRY main {
  param_0 = f32[$N] parameter(0)
  param_1 = f32[$N] parameter(1)
  param_2 = f32[$N] parameter(2)
  param_3 = f32[$N] parameter(3)

  load_0 = f32[$RN] custom-call(param_0), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":$R}\n"
  load_1 = f32[$RN] custom-call(param_1), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":$R}\n"
  load_2 = f32[$RN] custom-call(param_2), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":$R}\n"

  c0 = (f32[$N], f32[$RN], f32[$RN]) call(param_3, load_0, load_1, load_2), to_apply=comp_0, backend_config="{\"callConfig\":{\"type\":\"Function\",\"functionConfig\":{\"keepInputLayouts\":true,\"uniqueSharding\":true,\"partitionedElementwiseCluster\":true}}}"
  c0_gte0 = f32[$N] get-tuple-element(c0), index=0
  c0_gte1 = f32[$RN] get-tuple-element(c0), index=1
  c0_gte2 = f32[$RN] get-tuple-element(c0), index=2

  new_param_0 = f32[$N] custom-call(param_0, c0_gte1), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":$R}\n"
  new_param_1 = f32[$N] custom-call(param_1, c0_gte2), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":$R}\n"

  ROOT t = (f32[$N], f32[$N], f32[$N], f32[$N], f32[$RN], f32[$RN]) tuple(new_param_0, new_param_1)
}
)";
  hlo = tensorflow::str_util::StringReplace(
      hlo, "$RN", shard_size ? std::to_string(shard_size) : std::string(),
      true);
  hlo = tensorflow::str_util::StringReplace(
      hlo, "$R", std::to_string(replication_factor), true);
  hlo = tensorflow::str_util::StringReplace(
      hlo, "$N", n ? std::to_string(n) : std::string(), true);

  return hlo;
}

TEST_F(OutlineRemoteBuffersTest, TestGetFunctions) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(GetValidHlo()));

  ASSERT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  auto all_functions =
      OutlineRemoteBuffers::GetFunctionsForOutlining(module.get());
  EXPECT_EQ(all_functions.size(), 1);

  auto functions = *std::begin(all_functions)->second;
  EXPECT_EQ(functions.size(), 2);

  auto function = *std::begin(functions);
  RemoteBufferInputsOutputsInfos rbioi(function);
  EXPECT_EQ(rbioi.GetNumModifiedLoadStores(), 2);
  EXPECT_EQ(rbioi.GetNumUnmodifiedLoads(), 1);
  EXPECT_THAT(rbioi.GetInputsOldToNewPermutation(),
              ::testing::ElementsAre(3, 0, 1, 2));
  EXPECT_THAT(rbioi.GetInputsNewToOldPermutation(),
              ::testing::ElementsAre(1, 2, 3, 0));

  EXPECT_THAT(rbioi.GetOutputsOldToNewPermutation(),
              ::testing::ElementsAre(2, 0, 1));
  EXPECT_THAT(rbioi.GetOutputsNewToOldPermutation(),
              ::testing::ElementsAre(1, 2, 0));
}

TEST_F(OutlineRemoteBuffersTest, TestGetFunctionsWithPermutation) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(GetValidHloWithPermutation()));

  ASSERT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  auto all_functions =
      OutlineRemoteBuffers::GetFunctionsForOutlining(module.get());
  EXPECT_EQ(all_functions.size(), 1);

  auto functions = *std::begin(all_functions)->second;
  EXPECT_EQ(functions.size(), 2);

  auto function = *std::begin(functions);
  RemoteBufferInputsOutputsInfos rbioi(function);
  EXPECT_EQ(rbioi.GetNumModifiedLoadStores(), 2);
  EXPECT_EQ(rbioi.GetNumUnmodifiedLoads(), 0);
  EXPECT_THAT(rbioi.GetInputsOldToNewPermutation(),
              ::testing::ElementsAre(2, 0, 1));
  EXPECT_THAT(rbioi.GetInputsNewToOldPermutation(),
              ::testing::ElementsAre(1, 2, 0));

  EXPECT_THAT(rbioi.GetOutputsOldToNewPermutation(),
              ::testing::ElementsAre(2, 1, 0));
  EXPECT_THAT(rbioi.GetOutputsNewToOldPermutation(),
              ::testing::ElementsAre(2, 1, 0));
}

TEST_F(OutlineRemoteBuffersTest, TestGetFunctionsInOutPermutationsDontMatch) {
  const string& hlo = R"(
HloModule module

comp_0 {
  c0_param_0 = f32[] parameter(0)
  c0_param_1 = f32[] parameter(1)
  c0_param_2 = f32[] parameter(2)
  c0_add = f32[] add(c0_param_2, c0_param_1)
  c0_subtract = f32[] subtract(c0_param_2, c0_param_1)
  ROOT c0_t = (f32[], f32[], f32[]) tuple(c0_param_0, c0_add, c0_subtract)
}

comp_1 {
  c1_param_0 = f32[] parameter(0)
  c1_param_1 = f32[] parameter(1)
  c1_param_2 = f32[] parameter(2)
  c1_add = f32[] add(c1_param_2, c1_param_1)
  c1_subtract = f32[] subtract(c1_param_2, c1_param_1)
  ROOT c1_t = (f32[], f32[], f32[]) tuple(c1_param_0, c1_add, c1_subtract)
}

ENTRY main {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  param_2 = f32[] parameter(2)
  param_3 = f32[] parameter(3)
  param_4 = f32[] parameter(4)
  param_5 = f32[] parameter(5)

  load_0 = f32[] custom-call(param_0), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n"
  load_1 = f32[] custom-call(param_1), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n"

  c0 = (f32[], f32[], f32[]) call(param_4, load_0, load_1), to_apply=comp_0, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  c0_gte0 = f32[] get-tuple-element(c0), index=0
  c0_gte1 = f32[] get-tuple-element(c0), index=1
  c0_gte2 = f32[] get-tuple-element(c0), index=2

  new_param_0 = f32[] custom-call(param_0, c0_gte2), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n"
  new_param_1 = f32[] custom-call(param_1, c0_gte1), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n"

  load_2 = f32[] custom-call(param_2), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n"
  load_3 = f32[] custom-call(param_3), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n"

  c1 = (f32[], f32[], f32[]) call(param_5, load_2, load_3), to_apply=comp_1, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  c1_gte0 = f32[] get-tuple-element(c1), index=0
  c1_gte1 = f32[] get-tuple-element(c1), index=1
  c1_gte2 = f32[] get-tuple-element(c1), index=2

  new_param_2 = f32[] custom-call(param_2, c1_gte1), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n"
  new_param_3 = f32[] custom-call(param_3, c1_gte2), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n"

  ROOT t = (f32[], f32[], f32[], f32[], f32[], f32[]) tuple(new_param_0, new_param_1, new_param_2, new_param_3, c0_gte0, c1_gte0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  ASSERT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  auto all_functions =
      OutlineRemoteBuffers::GetFunctionsForOutlining(module.get());
  EXPECT_EQ(all_functions.size(), 0);

  HloInstruction* c0 = FindInstruction(module.get(), "c0");
  HloInstruction* c1 = FindInstruction(module.get(), "c1");

  // Outputs permutation differs between the functions.
  RemoteBufferInputsOutputsInfos rbioi_c0(c0);
  EXPECT_THAT(rbioi_c0.GetInputsOldToNewPermutation(),
              ::testing::ElementsAre(2, 0, 1));
  EXPECT_THAT(rbioi_c0.GetInputsNewToOldPermutation(),
              ::testing::ElementsAre(1, 2, 0));

  EXPECT_THAT(rbioi_c0.GetOutputsOldToNewPermutation(),
              ::testing::ElementsAre(2, 1, 0));
  EXPECT_THAT(rbioi_c0.GetOutputsNewToOldPermutation(),
              ::testing::ElementsAre(2, 1, 0));

  RemoteBufferInputsOutputsInfos rbioi_c1(c1);
  EXPECT_THAT(rbioi_c1.GetInputsOldToNewPermutation(),
              ::testing::ElementsAre(2, 0, 1));
  EXPECT_THAT(rbioi_c1.GetInputsNewToOldPermutation(),
              ::testing::ElementsAre(1, 2, 0));

  EXPECT_THAT(rbioi_c1.GetOutputsOldToNewPermutation(),
              ::testing::ElementsAre(2, 0, 1));
  EXPECT_THAT(rbioi_c1.GetOutputsNewToOldPermutation(),
              ::testing::ElementsAre(1, 2, 0));
}

struct OutlinePartitionedRemoteBuffersTestSpec {
  unsigned n;
  unsigned replication_factor;
  unsigned shard_size;
};

std::ostream& operator<<(std::ostream& os,
                         const OutlinePartitionedRemoteBuffersTestSpec& spec) {
  return os << "{ "
            << "n: " << spec.n
            << ", replication factor: " << spec.replication_factor
            << ", shard_size: " << spec.shard_size << " }";
}

class OutlinePartitionedRemoteBuffersTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          OutlinePartitionedRemoteBuffersTestSpec> {};

INSTANTIATE_TEST_SUITE_P(
    OutlinePartitionedRemoteBuffersTestCases,
    OutlinePartitionedRemoteBuffersTest,
    ::testing::ValuesIn(std::vector<OutlinePartitionedRemoteBuffersTestSpec>{
        {0, 1, 0},
        {8, 1, 8},
        {8, 2, 4},
        {10, 3, 4},
    }));

TEST_P(OutlinePartitionedRemoteBuffersTest, TestOutline) {
  auto param = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(GetValidHlo(
                       param.n, param.replication_factor, param.shard_size)));

  ASSERT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  ASSERT_TRUE(OutlineRemoteBuffers().Run(module.get()).ValueOrDie());

  // Check that the remote loads and stores have been outlined into functions.
  HloInstruction* param_0 = FindInstruction(module.get(), "param_0");
  HloInstruction* param_1 = FindInstruction(module.get(), "param_1");
  HloInstruction* param_2 = FindInstruction(module.get(), "param_2");
  HloInstruction* param_3 = FindInstruction(module.get(), "param_3");
  HloInstruction* param_4 = FindInstruction(module.get(), "param_4");
  HloInstruction* param_5 = FindInstruction(module.get(), "param_5");
  HloInstruction* param_6 = FindInstruction(module.get(), "param_6");
  HloInstruction* param_7 = FindInstruction(module.get(), "param_7");
  HloInstruction* c0_outlined = FindInstruction(module.get(), "c0_outlined");
  HloInstruction* c1_outlined = FindInstruction(module.get(), "c1_outlined");
  HloInstruction* t = FindInstruction(module.get(), "t");
  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * c0_0,
                          GetUniqueGTEUser(c0_outlined, 0));
  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * c0_1,
                          GetUniqueGTEUser(c0_outlined, 1));
  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * c0_2,
                          GetUniqueGTEUser(c0_outlined, 2));
  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * c1_0,
                          GetUniqueGTEUser(c1_outlined, 0));
  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * c1_1,
                          GetUniqueGTEUser(c1_outlined, 1));
  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * c1_2,
                          GetUniqueGTEUser(c1_outlined, 2));

  // Check the inputs are re-ordered.
  EXPECT_THAT(c0_outlined->operands(),
              ::testing::ElementsAre(param_0, param_1, param_2, param_6));
  EXPECT_THAT(c1_outlined->operands(),
              ::testing::ElementsAre(param_3, param_4, param_5, param_7));

  // Check that the outputs have been correctly connected.
  EXPECT_THAT(t->operands(),
              ::testing::ElementsAre(c0_0, c0_1, c1_0, c1_1, c0_2, c1_2));
  // Verify that each computation has the inputs connected correctly.
  for (HloInstruction* call : {c0_outlined, c1_outlined}) {
    HloInstruction* root = call->to_apply()->root_instruction();
    HloInstruction *store0, *store1;
    EXPECT_TRUE(
        Match(root, m::Tuple(m::Op(&store0), m::Op(&store1), m::Parameter(3))));
    EXPECT_TRUE(IsPoplarInstruction(RemoteParameterStore, store0));
    EXPECT_TRUE(IsPoplarInstruction(RemoteParameterStore, store1));

    HloInstruction *load0_0, *load0_1;
    EXPECT_TRUE(
        Match(store0, m::CustomCall(m::Parameter(0),
                                    m::Add(m::Op(&load0_1), m::Op(&load0_0)))));
    EXPECT_TRUE(IsPoplarInstruction(RemoteParameterLoad, load0_0));
    EXPECT_TRUE(IsPoplarInstruction(RemoteParameterLoad, load0_1));

    HloInstruction *load1_0, *load1_1;
    EXPECT_TRUE(Match(
        store1, m::CustomCall(m::Parameter(1),
                              m::Subtract(m::Op(&load1_1), m::Op(&load1_0)))));
    EXPECT_TRUE(IsPoplarInstruction(RemoteParameterLoad, load1_0));
    EXPECT_TRUE(IsPoplarInstruction(RemoteParameterLoad, load1_1));

    EXPECT_EQ(load0_0, load1_0);
    EXPECT_EQ(load0_1, load1_1);

    EXPECT_TRUE(Match(load0_0, m::CustomCall(m::Parameter(0))));
    EXPECT_TRUE(Match(load0_1, m::CustomCall(m::Parameter(1))));
  }
}

// Difference between this test and the above one is that the outputs are
// switched, which therefore means that the outputs are permuted.
TEST_F(OutlineRemoteBuffersTest, TestOutlineWithPermutation) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(GetValidHloWithPermutation()));

  ASSERT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  ASSERT_TRUE(OutlineRemoteBuffers().Run(module.get()).ValueOrDie());

  // Check that the remote loads and stores have been outlined into functions.
  HloInstruction* param_0 = FindInstruction(module.get(), "param_0");
  HloInstruction* param_1 = FindInstruction(module.get(), "param_1");
  HloInstruction* param_2 = FindInstruction(module.get(), "param_2");
  HloInstruction* param_3 = FindInstruction(module.get(), "param_3");
  HloInstruction* param_4 = FindInstruction(module.get(), "param_4");
  HloInstruction* param_5 = FindInstruction(module.get(), "param_5");
  HloInstruction* c0_outlined = FindInstruction(module.get(), "c0_outlined");
  HloInstruction* c1_outlined = FindInstruction(module.get(), "c1_outlined");
  HloInstruction* t = FindInstruction(module.get(), "t");
  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * c0_0,
                          GetUniqueGTEUser(c0_outlined, 0));
  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * c0_1,
                          GetUniqueGTEUser(c0_outlined, 1));
  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * c0_2,
                          GetUniqueGTEUser(c0_outlined, 2));
  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * c1_0,
                          GetUniqueGTEUser(c1_outlined, 0));
  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * c1_1,
                          GetUniqueGTEUser(c1_outlined, 1));
  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * c1_2,
                          GetUniqueGTEUser(c1_outlined, 2));

  // Check the inputs are re-ordered.
  EXPECT_THAT(c0_outlined->operands(),
              ::testing::ElementsAre(param_0, param_1, param_4));
  EXPECT_THAT(c1_outlined->operands(),
              ::testing::ElementsAre(param_2, param_3, param_5));

  // Check that the outputs have been correctly connected.
  EXPECT_THAT(t->operands(),
              ::testing::ElementsAre(c0_0, c0_1, c1_0, c1_1, c0_2, c1_2));
  // Verify that each computation has the inputs connected correctly.
  for (HloInstruction* call : {c0_outlined, c1_outlined}) {
    HloInstruction* root = call->to_apply()->root_instruction();
    HloInstruction *store0, *store1;
    EXPECT_TRUE(
        Match(root, m::Tuple(m::Op(&store0), m::Op(&store1), m::Parameter(2))));
    EXPECT_TRUE(IsPoplarInstruction(RemoteParameterStore, store0));
    EXPECT_TRUE(IsPoplarInstruction(RemoteParameterStore, store1));

    HloInstruction *load0_0, *load0_1;
    EXPECT_TRUE(Match(
        store0, m::CustomCall(m::Parameter(0),
                              m::Subtract(m::Op(&load0_1), m::Op(&load0_0)))));
    EXPECT_TRUE(IsPoplarInstruction(RemoteParameterLoad, load0_0));
    EXPECT_TRUE(IsPoplarInstruction(RemoteParameterLoad, load0_1));

    HloInstruction *load1_0, *load1_1;
    EXPECT_TRUE(
        Match(store1, m::CustomCall(m::Parameter(1),
                                    m::Add(m::Op(&load1_1), m::Op(&load1_0)))));
    EXPECT_TRUE(IsPoplarInstruction(RemoteParameterLoad, load1_0));
    EXPECT_TRUE(IsPoplarInstruction(RemoteParameterLoad, load1_1));

    EXPECT_EQ(load0_0, load1_0);
    EXPECT_EQ(load0_1, load1_1);

    EXPECT_TRUE(Match(load0_0, m::CustomCall(m::Parameter(0))));
    EXPECT_TRUE(Match(load0_1, m::CustomCall(m::Parameter(1))));
  }
}

TEST_F(OutlineRemoteBuffersTest, TestElementwiseCluster) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(GetElementwiseClusterHlo()));

  ASSERT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  auto all_functions =
      OutlineRemoteBuffers::GetFunctionsForOutlining(module.get());
  EXPECT_EQ(all_functions.size(), 1);

  auto functions = *std::begin(all_functions)->second;
  EXPECT_EQ(functions.size(), 1);

  auto function = *std::begin(functions);
  RemoteBufferInputsOutputsInfos rbioi(function);
  EXPECT_EQ(rbioi.GetNumModifiedLoadStores(), 2);
  EXPECT_EQ(rbioi.GetNumUnmodifiedLoads(), 1);
  EXPECT_THAT(rbioi.GetInputsOldToNewPermutation(),
              ::testing::ElementsAre(3, 0, 1, 2));
  EXPECT_THAT(rbioi.GetInputsNewToOldPermutation(),
              ::testing::ElementsAre(1, 2, 3, 0));

  EXPECT_THAT(rbioi.GetOutputsOldToNewPermutation(),
              ::testing::ElementsAre(2, 0, 1));
  EXPECT_THAT(rbioi.GetOutputsNewToOldPermutation(),
              ::testing::ElementsAre(1, 2, 0));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
