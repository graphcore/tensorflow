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

#include "tensorflow/compiler/plugin/poplar/driver/passes/function_combiner.h"

#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using FunctionCombinerTest = HloTestBase;

struct FunctionDescription {
  int64_t count;
  int64_t shard;
  std::vector<int64_t> shape;
};

// Function for generating Hlo modules with different functions.
std::string GetHlo(const std::vector<FunctionDescription>& functions) {
  std::stringstream comps_ss, entry_ss;
  comps_ss << "HloModule m\n";
  entry_ss << "ENTRY entry {";

  int64_t next_param_number = 0;
  int64_t next_function_id = 0;
  // Create the functions.

  for (const FunctionDescription& desc : functions) {
    for (int64_t i = 0; i != desc.count; ++i) {
      const std::string func_hlo = R"(
      comp_$counter_$shard {
        c$counter_param_0 = f32[$shape] parameter(0), sharding={maximal device=$shard}
        c$counter_param_1 = f32[$shape] parameter(1), sharding={maximal device=$shard}
        c$counter_param_2 = f32[$shape] parameter(2), sharding={maximal device=$shard}

        c$counter_load_0 = f32[$shape] custom-call(c$counter_param_0), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=$shard}
        c$counter_load_1 = f32[$shape] custom-call(c$counter_param_1), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=$shard}

        c$counter_add = f32[$shape] add(c$counter_load_0, c$counter_load_1), sharding={maximal device=$shard}
        c$counter_new_2 = f32[$shape] subtract(c$counter_param_2, c$counter_load_1), sharding={maximal device=$shard}

        c$counter_new_0 = f32[$shape] custom-call(c$counter_param_0, c$counter_add), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=$shard}
        ROOT c$counter_t = (f32[$shape], f32[$shape]) tuple(c$counter_new_0, c$counter_new_2), sharding={{maximal device=$shard}, {maximal device=$shard}}
      }
    )";
      std::string func_string = absl::StrReplaceAll(
          func_hlo, {{"$counter", std::to_string(next_function_id)},
                     {"$shard", std::to_string(desc.shard)},
                     {"$shape", absl::StrJoin(desc.shape, ",")}});
      comps_ss << func_string;

      const std::string entry_hlo = R"(
        param_$one = f32[$shape] parameter($one), sharding={maximal device=$shard}
        param_$two = f32[$shape] parameter($two), sharding={maximal device=$shard}
        param_$three = f32[$shape] parameter($three), sharding={maximal device=$shard}
        c$counter_$shard = (f32[$shape], f32[$shape]) call(param_$one, param_$two, param_$three), to_apply=comp_$counter_$shard, sharding={{maximal device=$shard}, {maximal device=$shard}}, backend_config="{\"callConfig\":{\"type\":\"Function\", \"functionConfig\":{\"numModifiedRemoteBufferInputs\":\"1\", \"numUnmodifiedRemoteBufferInputs\":\"1\"}}}"
        new_param_$one = f32[$shape] get-tuple-element(c$counter_$shard), index=0, sharding={maximal device=$shard}
        new_param_$three = f32[$shape] get-tuple-element(c$counter_$shard), index=1, sharding={maximal device=$shard}
      )";
      std::string call_string = absl::StrReplaceAll(
          entry_hlo, {{"$counter", std::to_string(next_function_id)},
                      {"$shard", std::to_string(desc.shard)},
                      {"$one", std::to_string(next_param_number++)},
                      {"$two", std::to_string(next_param_number++)},
                      {"$three", std::to_string(next_param_number++)},
                      {"$shape", absl::StrJoin(desc.shape, ",")}});
      entry_ss << call_string;
      next_function_id++;
    }
  }

  // As long as DCE is not run, this is fine.
  entry_ss << "ROOT t = () tuple ()\n";
  entry_ss << "}";

  return comps_ss.str() + entry_ss.str();
}

std::string GetHloDifferentNumberParameters() {
  const string& hlo = R"(
HloModule module

comp_0 {
  c0_param_0 = f32[] parameter(0), sharding={maximal device=0}
  c0_param_1 = f32[] parameter(1), sharding={maximal device=0}
  c0_param_2 = f32[] parameter(2), sharding={maximal device=0}
  c0_load_0 = f32[] custom-call(c0_param_0), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
  c0_load_1 = f32[] custom-call(c0_param_1), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
  c0_add = f32[] add(c0_param_2, c0_load_0), sharding={maximal device=0}
  c0_subtract = f32[] subtract(c0_param_2, c0_load_1), sharding={maximal device=0}
  c0_new_param_0 = f32[] custom-call(c0_param_0, c0_add), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
  c0_new_param_1 = f32[] custom-call(c0_param_1, c0_subtract), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=0}
  ROOT c0_t = (f32[], f32[], f32[]) tuple(c0_new_param_0, c0_new_param_1, c0_subtract), sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}
}

comp_1 {
  c1_param_0 = f32[] parameter(0), sharding={maximal device=1}
  c1_param_1 = f32[] parameter(1), sharding={maximal device=1}
  c1_load_0 = f32[] custom-call(c1_param_0), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=1}
  c1_load_1 = f32[] custom-call(c1_param_1), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=1}
  c1_add = f32[] add(c1_load_0, c1_load_1), sharding={maximal device=1}
  c1_subtract = f32[] subtract(c1_load_0, c1_load_1), sharding={maximal device=1}
  c1_new_param_0 = f32[] custom-call(c1_param_0, c1_add), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=1}
  c1_new_param_1 = f32[] custom-call(c1_param_1, c1_subtract), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n", sharding={maximal device=1}
  ROOT c1_t = (f32[], f32[]) tuple(c1_new_param_0, c1_new_param_1), sharding={{maximal device=1}, {maximal device=1}}
}

ENTRY main {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  param_1 = f32[] parameter(1), sharding={maximal device=0}
  param_2 = f32[] parameter(2), sharding={maximal device=0}
  param_3 = f32[] parameter(3), sharding={maximal device=1}
  param_4 = f32[] parameter(4), sharding={maximal device=1}

  c0 = (f32[], f32[], f32[]) call(param_0, param_1, param_2), to_apply=comp_0, backend_config="{\"callConfig\":{\"type\":\"Function\", \"functionConfig\":{\"numModifiedRemoteBufferInputs\":\"2\", \"numUnmodifiedRemoteBufferInputs\":\"0\"}}}", sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}
  c0_gte0 = f32[] get-tuple-element(c0), index=0
  c0_gte1 = f32[] get-tuple-element(c0), index=1
  c0_gte2 = f32[] get-tuple-element(c0), index=2

  c1 = (f32[], f32[]) call(param_3, param_4), to_apply=comp_1, backend_config="{\"callConfig\":{\"type\":\"Function\", \"functionConfig\":{\"numModifiedRemoteBufferInputs\":\"2\", \"numUnmodifiedRemoteBufferInputs\":\"0\"}}}", sharding={{maximal device=1}, {maximal device=1}}
  c1_gte0 = f32[] get-tuple-element(c1), index=0, sharding={maximal device=1}
  c1_gte1 = f32[] get-tuple-element(c1), index=1, sharding={maximal device=1}

  ROOT t = () tuple()
}
)";
  return hlo;
}

TEST_F(FunctionCombinerTest, GetFunctionsToCombinePerShardMatch) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(
                       GetHlo({{/*count=*/4, /*shard=*/0, /*shape=*/{1}},
                               {/*count=*/4, /*shard=*/1, /*shape=*/{1}}})));

  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  auto funcs =
      FunctionCombiner::GetFunctionsToCombine(module->entry_computation());
  EXPECT_EQ(funcs.size(), 1);
  EXPECT_EQ(funcs[0].size(), 2);
  for (auto f : funcs[0]) {
    EXPECT_EQ(f.size(), 4);
  }
}

TEST_F(FunctionCombinerTest, GetFunctionsToCombinePerShardMismatch) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(
                       GetHlo({{/*count=*/4, /*shard=*/0, /*shape=*/{1}},
                               {/*count=*/5, /*shard=*/2, /*shape=*/{1}},
                               {/*count=*/4, /*shard=*/1, /*shape=*/{1}}})));

  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  auto funcs =
      FunctionCombiner::GetFunctionsToCombine(module->entry_computation());
  EXPECT_EQ(funcs.size(), 1);
  // Only two shards get combined.
  EXPECT_EQ(funcs[0].size(), 2);
  for (auto f : funcs[0]) {
    EXPECT_EQ(f.size(), 4);
  }
}

TEST_F(FunctionCombinerTest, GetFunctionsToCombinePerShardDifferentShapes) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(
                       GetHlo({{/*count=*/4, /*shard=*/0, /*shape=*/{10}},
                               {/*count=*/4, /*shard=*/2, /*shape=*/{11}},
                               {/*count=*/4, /*shard=*/1, /*shape=*/{12}}})));

  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  auto funcs =
      FunctionCombiner::GetFunctionsToCombine(module->entry_computation());
  EXPECT_EQ(funcs.size(), 1);
  EXPECT_EQ(funcs[0].size(), 3);
  for (auto f : funcs[0]) {
    EXPECT_EQ(f.size(), 4);
  }
}

TEST_F(FunctionCombinerTest, GetFunctionsToCombineSortBySize) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(
                       GetHlo({{/*count=*/4, /*shard=*/0, /*shape=*/{1}},
                               {/*count=*/4, /*shard=*/1, /*shape=*/{1}},
                               {/*count=*/5, /*shard=*/0, /*shape=*/{10}},
                               {/*count=*/5, /*shard=*/2, /*shape=*/{10}},
                               {/*count=*/5, /*shard=*/1, /*shape=*/{10}}})));

  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  auto funcs =
      FunctionCombiner::GetFunctionsToCombine(module->entry_computation());
  EXPECT_EQ(funcs.size(), 2);

  EXPECT_EQ(funcs[0].size(), 2);
  for (auto fs : funcs[0]) {
    EXPECT_EQ(fs.size(), 4);
    for (auto f : fs) {
      EXPECT_EQ(GetByteSizeOfTotalShape(f->shape()), 8);
    }
  }

  EXPECT_EQ(funcs[1].size(), 3);
  for (auto fs : funcs[1]) {
    EXPECT_EQ(fs.size(), 5);
    for (auto f : fs) {
      EXPECT_EQ(GetByteSizeOfTotalShape(f->shape()), 80);
    }
  }
}

TEST_F(FunctionCombinerTest, TestPermutations) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(
                       GetHlo({{/*count=*/4, /*shard=*/0, /*shape=*/{1}},
                               {/*count=*/4, /*shard=*/1, /*shape=*/{1}}})));

  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  auto funcs =
      FunctionCombiner::GetFunctionsToCombine(module->entry_computation());
  EXPECT_EQ(funcs.size(), 1);
  EXPECT_EQ(funcs[0].size(), 2);
  for (auto f : funcs[0]) {
    EXPECT_EQ(f.size(), 4);
  }

  std::vector<HloInstruction*> functions(2);
  absl::c_transform(funcs[0], functions.begin(),
                    [](const Functions& shard_functions) {
                      return *std::begin(shard_functions);
                    });

  auto permutations = FunctionCombiner::GetInputsOutputsPermutation(functions);

  EXPECT_THAT(permutations.old_to_new_inputs_permutation,
              ::testing::ElementsAre(0, 2, 4, 1, 3, 5));
  EXPECT_THAT(permutations.old_to_new_outputs_permutation,
              ::testing::ElementsAre(0, 2, 1, 3));
}

TEST_F(FunctionCombinerTest, TestPermutationsDifferentInputs) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           GetHloDifferentNumberParameters()));

  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  auto funcs =
      FunctionCombiner::GetFunctionsToCombine(module->entry_computation());
  EXPECT_EQ(funcs.size(), 1);
  EXPECT_EQ(funcs[0].size(), 2);
  for (auto f : funcs[0]) {
    EXPECT_EQ(f.size(), 1);
  }

  std::vector<HloInstruction*> functions(2);
  absl::c_transform(funcs[0], functions.begin(),
                    [](const Functions& shard_functions) {
                      return *std::begin(shard_functions);
                    });

  auto permutations = FunctionCombiner::GetInputsOutputsPermutation(functions);

  EXPECT_THAT(permutations.old_to_new_inputs_permutation,
              ::testing::ElementsAre(0, 1, 4, 2, 3));
  EXPECT_THAT(permutations.old_to_new_outputs_permutation,
              ::testing::ElementsAre(0, 1, 4, 2, 3));
}

TEST_F(FunctionCombinerTest, TestCombine) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(
                       GetHlo({{/*count=*/4, /*shard=*/0, /*shape=*/{1}},
                               {/*count=*/4, /*shard=*/1, /*shape=*/{2}}})));

  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  auto funcs =
      FunctionCombiner::GetFunctionsToCombine(module->entry_computation());
  EXPECT_EQ(funcs.size(), 1);
  EXPECT_EQ(funcs[0].size(), 2);
  for (auto f : funcs[0]) {
    EXPECT_EQ(f.size(), 4);
  }

  TF_ASSERT_OK_AND_ASSIGN(auto combined,
                          FunctionCombiner::CombineFunctions(funcs[0]));
  EXPECT_EQ(combined.size(), 4);
  for (auto* inst : combined) {
    auto* computation = inst->to_apply();
    auto* root = computation->root_instruction();

    HloInstruction *root0_0, *root0_1, *root1_0, *root1_1;
    EXPECT_TRUE(Match(root, m::Tuple(m::GetTupleElement(m::Op(&root0_0), 0),
                                     m::GetTupleElement(m::Op(&root1_0), 0),
                                     m::GetTupleElement(m::Op(&root0_1), 1),
                                     m::GetTupleElement(m::Op(&root1_1), 1))));
    EXPECT_THAT(root->sharding().tuple_elements(),
                ::testing::ElementsAre(HloSharding::AssignDevice(0),
                                       HloSharding::AssignDevice(1),
                                       HloSharding::AssignDevice(0),
                                       HloSharding::AssignDevice(1)));

    CHECK_EQ(root0_0, root0_1);
    CHECK_EQ(root1_0, root1_1);

    EXPECT_TRUE(Match(
        root0_0, m::Tuple(m::CustomCall(m::Parameter(0),
                                        m::Add(m::CustomCall(m::Parameter(0)),
                                               m::CustomCall(m::Parameter(2)))),
                          m::Subtract(m::Parameter(4),
                                      m::CustomCall(m::Parameter(2))))));

    EXPECT_TRUE(Match(
        root1_0, m::Tuple(m::CustomCall(m::Parameter(1),
                                        m::Add(m::CustomCall(m::Parameter(1)),
                                               m::CustomCall(m::Parameter(3)))),
                          m::Subtract(m::Parameter(5),
                                      m::CustomCall(m::Parameter(3))))));
  }
}

TEST_F(FunctionCombinerTest, TestCombineDifferentInputs) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           GetHloDifferentNumberParameters()));

  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  auto funcs =
      FunctionCombiner::GetFunctionsToCombine(module->entry_computation());
  EXPECT_EQ(funcs.size(), 1);
  EXPECT_EQ(funcs[0].size(), 2);
  for (auto f : funcs[0]) {
    EXPECT_EQ(f.size(), 1);
  }

  TF_ASSERT_OK_AND_ASSIGN(auto combined,
                          FunctionCombiner::CombineFunctions(funcs[0]));
  EXPECT_EQ(combined.size(), 1);
  auto* inst = combined[0];
  auto* computation = inst->to_apply();
  auto* root = computation->root_instruction();

  HloInstruction *root0_0, *root0_1, *root0_2, *root1_0, *root1_1;
  EXPECT_TRUE(Match(root, m::Tuple(m::GetTupleElement(m::Op(&root0_0), 0),
                                   m::GetTupleElement(m::Op(&root0_1), 1),
                                   m::GetTupleElement(m::Op(&root1_0), 0),
                                   m::GetTupleElement(m::Op(&root1_1), 1),
                                   m::GetTupleElement(m::Op(&root0_2), 2))));
  EXPECT_THAT(root->sharding().tuple_elements(),
              ::testing::ElementsAre(
                  HloSharding::AssignDevice(0), HloSharding::AssignDevice(0),
                  HloSharding::AssignDevice(1), HloSharding::AssignDevice(1),
                  HloSharding::AssignDevice(0)));

  CHECK_EQ(root0_0, root0_1);
  CHECK_EQ(root0_0, root0_2);
  CHECK_EQ(root1_0, root1_1);

  EXPECT_TRUE(Match(
      root0_0,
      m::Tuple(m::CustomCall(
                   m::Parameter(0),
                   m::Add(m::Parameter(4), m::CustomCall(m::Parameter(0)))),
               m::CustomCall(m::Parameter(1),
                             m::Subtract(m::Parameter(4),
                                         m::CustomCall(m::Parameter(1)))),
               m::Subtract(m::Parameter(4), m::CustomCall(m::Parameter(1))))));

  EXPECT_TRUE(Match(
      root1_0,
      m::Tuple(m::CustomCall(m::Parameter(2),
                             m::Add(m::CustomCall(m::Parameter(2)),
                                    m::CustomCall(m::Parameter(3)))),
               m::CustomCall(m::Parameter(3),
                             m::Subtract(m::CustomCall(m::Parameter(2)),
                                         m::CustomCall(m::Parameter(3)))))));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
