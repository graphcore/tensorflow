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

#include "tensorflow/compiler/plugin/poplar/driver/passes/mark_replica_identical_instructions.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"

namespace xla {
namespace poplarplugin {
namespace {

using MarkReplicaIdenticalInstructionsTest = HloTestFixture;

const char* simple_flat_hlo = R"(
HloModule test
funcA {
  xA = f32[] parameter(0)
  yA = f32[] parameter(1)
  ROOT addA = f32[] add(xA, yA)
}

funcB {
  xB = f32[] parameter(0)
  yB = f32[] parameter(1)
  ROOT addB = f32[] add(xB, yB)
}

ENTRY test {
  identical0 = f32[] parameter(0)
  identical1 = f32[] constant(3)
  identical_call = f32[] call(identical0, identical1), to_apply=funcA, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"

  constant = f32[] constant(2)
  differing0 = f32[] rng(constant, constant), distribution=rng_uniform
  differing1 = f32[] rng(constant, constant), distribution=rng_uniform

  differing_call = f32[] call(differing0, differing1), to_apply=funcB, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"

  ROOT differing_root = (f32[], f32[], f32[]) tuple(identical_call, differing_call)
}
)";
TEST_F(MarkReplicaIdenticalInstructionsTest, SetsIsReplicaIdenticalOption) {
  ASSERT_TRUE(SetUpHloModule(simple_flat_hlo));

  MarkReplicaIdenticalInstructions mark_instructions;

  TF_ASSERT_OK_AND_ASSIGN(bool modified, mark_instructions.Run(hlo_module_));
  ASSERT_TRUE(modified);

  // Check that MarkReplicaIdenticalInstructions set the is_replica_identical
  // option on the instructions
  // we expect it to.

  const std::set<std::string> identical_instr_names = {"identical_call",
                                                       "identical0",
                                                       "identical1",
                                                       "constant",
                                                       "xA",
                                                       "yA",
                                                       "addA"};
  for (const auto& instruction_name : identical_instr_names) {
    auto* instr = FindInstruction(hlo_module_, instruction_name);
    ASSERT_TRUE(instr);
    ASSERT_TRUE(IsInstructionReplicaIdentical(instr));
  }

  const std::set<std::string> differing_instr_names = {"differing_root",
                                                       "differing_call",
                                                       "differing0",
                                                       "differing1",
                                                       "xB",
                                                       "yB",
                                                       "addB"};
  for (const auto& instruction_name : differing_instr_names) {
    auto* instr = FindInstruction(hlo_module_, instruction_name);
    ASSERT_TRUE(instr);
    ASSERT_FALSE(IsInstructionReplicaIdentical(instr));
  }
}

const char* unreachable_comp_hlo = R"(
HloModule test
func {
  ROOT unvisited = f32[] parameter(0)
}

ENTRY test {
  constant = f32[] constant(2)
  param0 = f32[] parameter(0)
  ROOT add = f32[] add(constant, param0)
}
)";
TEST_F(MarkReplicaIdenticalInstructionsTest, UnvisitedInstructionDefault) {
  ASSERT_TRUE(SetUpHloModule(unreachable_comp_hlo));

  MarkReplicaIdenticalInstructions mark_instructions;

  TF_ASSERT_OK_AND_ASSIGN(bool modified, mark_instructions.Run(hlo_module_));
  ASSERT_TRUE(modified);

  auto* unvisited_instr = FindInstruction(hlo_module_, "unvisited");
  ASSERT_FALSE(IsInstructionReplicaIdentical(unvisited_instr));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
