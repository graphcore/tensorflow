/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/slice_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_base.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"

namespace xla {
namespace poplarplugin {
namespace {

/**
 * Create a trivial schedule using the order of definition.
 */
static void Schedule(HloModule* module) {
  HloSchedule schedule{module};
  for (auto* comp : module->computations()) {
    std::vector<HloInstruction*> sequence;
    for (auto* inst : comp->instructions()) {
      sequence.push_back(inst);
    }
    schedule.set_sequence(comp, sequence);
  }
  module->set_schedule(schedule);
}

class SliceCopyInserterDoesCopyTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          std::pair<std::string, std::vector<std::string>>> {
 public:
  std::string hlo;
  std::vector<std::string> instructions_to_copy_names;

  void SetUp() {
    hlo = GetParam().first;
    instructions_to_copy_names = GetParam().second;
  }
};

TEST_P(SliceCopyInserterDoesCopyTest, DoesCopy) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_OK(CustomOpReplacer().Run(module.get()));
  EXPECT_OK(FlattenCallGraph().Run(module.get()));
  auto annotations = CompilerAnnotations(module.get());
  Schedule(module.get());

  std::vector<HloInstruction*> instructions_to_copy;
  for (const auto& name : instructions_to_copy_names) {
    instructions_to_copy.push_back(FindInstruction(module.get(), name));
  }

  auto copy_count = instructions_to_copy.size();
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
      copied_instructions_users;
  for (auto* instr : instructions_to_copy) {
    copied_instructions_users[instr] = instr->users();
  }

  auto instr_count_before = module->instruction_count();
  TF_ASSERT_OK_AND_ASSIGN(auto changed,
                          SliceCopyInserter(annotations).Run(module.get()));
  auto instr_count_after = module->instruction_count();

  EXPECT_TRUE(changed);
  EXPECT_EQ(instr_count_after, instr_count_before + copy_count);
  for (auto* instr : instructions_to_copy) {
    ASSERT_EQ(instr->user_count(), 1);
    ASSERT_EQ(instr->users()[0]->opcode(), HloOpcode::kCopy);
    ASSERT_EQ(instr->users()[0]->users(), copied_instructions_users[instr]);
  }
}

std::vector<std::pair<std::string, std::vector<std::string>>>
    test_values_does_copy = {
        {// A trivial case where the slice to be copied is in the entry
         // computation.
         R"(
HloModule top

ENTRY main {
  c_0 = s32[1,1] constant({{0}})
  c_1 = s32[1] constant({1})
  c_2 = s32[3] constant({1, 2, 3})

  reshape_0 = s32[1] reshape(c_0)
  add_0 = s32[1] add(c_1, reshape_0)

  slice_0 = s32[1] slice(c_2), slice={[0:1]}
  add_1 = s32[1] add(slice_0, c_1)

  ROOT t = tuple(add_0, add_1)
})",
         {"slice_0"}},
        {// Check that we update the root of a subcomputation. `slice_1` is also
         // copied because the call implicitly copies inputs and outputs.
         R"(
HloModule top

foo {
  p_0 = s32[3] parameter(0)
  ROOT slice_0 = s32[2] slice(p_0), slice={[0:2]}
}

ENTRY main {
  c_0 = s32[3] constant({1, 2, 3})
  call_0 = s32[2] call(c_0), to_apply=foo
  slice_1 = s32[1] slice(c_0), slice={[0:1]}
  ROOT tuple_0 = tuple(slice_1, call_0)
})",
         {"slice_0", "slice_1"}},
        {// Concatenate two buffers and kill them by copying a slice.
         R"(
HloModule top

ENTRY main {
  c_0 = s32[2] constant({0, 1})
  c_1 = s32[2] constant({0, 1})

  tuple_0 = (s32[2], s32[2]) tuple(c_0, c_1)

  gte_0 = s32[2] get-tuple-element(tuple_0), index=0
  gte_1 = s32[2] get-tuple-element(tuple_0), index=1

  cat_0 = s32[4] concatenate(gte_0, gte_1), dimensions={0}

  slice_0 = s32[2] slice(cat_0), slice={[0:2]}
  add_0 = s32[2] add(slice_0, slice_0)

  ROOT t = tuple(add_0)
})",
         {"slice_0"}},
        {// Copy in pipeline stage subcomputation.
         R"(
HloModule top

foo {
  p_1 = s32[3] parameter(0)
  ROOT slice_0 = s32[2] slice(p_1), slice={[0:2]}
}

bar {
  p_2 = s32[3] parameter(0)
  ROOT slice_1 = s32[1] slice(p_2), slice={[0:1]}
}

ENTRY main {
  p_0 = s32[3] parameter(0)
  call_0 = s32[2] call(p_0), to_apply=foo, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  call_1 = s32[1] call(p_0), to_apply=bar
  slice_2 = s32[1] slice(p_0), slice={[0:1]}
  ROOT tuple_0 = tuple(slice_2, call_0, call_1)
})",
         {"slice_1"}},
        {// In this case one of the slices to be copied is inside of a call.
         // It's copied because `p_0` creates a new buffer rather that reuse the
         // buffer from `c_0`.
         R"(
HloModule top

foo {
  p_0 = s32[3] parameter(0)
  ROOT slice_0 = s32[1] slice(p_0), slice={[0:1]}
}

ENTRY main {
  c_0 = s32[3] constant({1, 2, 3})
  call_0 = s32[1] call(c_0), to_apply=foo
  slice_1 = s32[1] slice(c_0), slice={[0:1]}
  ROOT tuple_0 = tuple(call_0, slice_1)
})",
         {"slice_0", "slice_1"}},
        {// Copy in conditional.
         R"(
HloModule module

true_branch {
  p_2 = s32[3] parameter(0)
  slice_1 = s32[1] slice(p_2), slice={[0:1]}
  ROOT tuple_1 = tuple(slice_1)
}

false_branch {
  p_3 = s32[3] parameter(0)
  slice_2 = s32[1] slice(p_3), slice={[1:2]}
  ROOT tuple_2 = tuple(slice_2)
}

ENTRY entry {
  p_0 = s32[3] parameter(0)
  p_1 = s32[3] parameter(1)
  pred_0 = pred[] parameter(2)
  conditional = (s32[1]) conditional(pred_0, p_0, p_1), true_computation=true_branch, false_computation=false_branch
  ROOT tuple_0 = tuple(conditional)
}
)",
         {"slice_1", "slice_2"}},
        {// Test that we don't copy a sequence of slices in a way that will
         // increase liveness.
         R"(
HloModule module

ENTRY entry {
  input = s32[100] parameter(0)
  slice0 = s32[90] slice(input), slice={[0:90]}
  slice1 = s32[89] slice(slice0), slice={[0:89]}
  slice2 = s32[88] slice(slice1), slice={[0:88]}
  slice3 = s32[87] slice(slice1), slice={[0:87]}
  ROOT root = tuple(slice2, slice3)
}
)",
         {"slice0"}},
        {// Copy in while.
         R"(
    HloModule top

    cond {
      p_2 = s32[3] parameter(0)
      ROOT c_0 = pred[] constant(true)
    }

    body {
      p_1 = s32[3] parameter(0)
      slice_1 = s32[1] slice(p_1), slice={[0:1]}
      reshape_0 = s32[] reshape(slice_1)
      ROOT broadcast_0 = s32[3] broadcast(reshape_0), dimensions={}
    }

    ENTRY main {
      p_0 = s32[3] parameter(0)
      while_0 = s32[3] while(p_0), condition=cond, body=body
      ROOT tuple_1 = tuple(while_0)
    })",
         {"slice_1"}},
};

INSTANTIATE_TEST_SUITE_P(, SliceCopyInserterDoesCopyTest,
                         ::testing::ValuesIn(test_values_does_copy));

class SliceCopyInserterDoesNotCopyTest
    : public HloTestBase,
      public ::testing::WithParamInterface<std::string> {
 public:
  std::string hlo;

  void SetUp() { hlo = GetParam(); }
};

TEST_P(SliceCopyInserterDoesNotCopyTest, DoesNotCopy) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_OK(CustomOpReplacer().Run(module.get()));
  EXPECT_OK(FlattenCallGraph().Run(module.get()));
  auto annotations = CompilerAnnotations(module.get());
  Schedule(module.get());

  auto instr_count_before = module->instruction_count();
  TF_ASSERT_OK_AND_ASSIGN(auto changed,
                          SliceCopyInserter(annotations).Run(module.get()));
  auto instr_count_after = module->instruction_count();

  EXPECT_FALSE(changed);
  EXPECT_EQ(instr_count_after, instr_count_before);
}

std::vector<std::string> test_values_does_not_copy = {
    // A trivial case where the slice is in the entry computation but it's not
    // copied because that wouldn't decrease liveness.
    R"(
HloModule top

ENTRY main {
  c_0 = s32[1,1] constant({{0}})
  c_1 = s32[1] constant({1})
  c_2 = s32[1] constant({1})

  reshape_0 = s32[1] reshape(c_0)
  add_0 = s32[1] add(c_1, reshape_0)

  slice_0 = s32[1] slice(c_2), slice={[0:1]}
  add_1 = s32[1] add(slice_0, c_1)

  ROOT t = tuple(add_0, add_1)
})",
    // Check that we don't copy the slice if it's already copied.
    R"(
HloModule top

ENTRY main {
  c_0 = s32[3] constant({1, 2, 3})

  slice_0 = s32[1] slice(c_0), slice={[0:1]}
  copy_0 = s32[1] copy(slice_0)

  ROOT t = tuple(copy_0)
})",
    // Check that we don't copy the root of the entry computation.
    R"(
HloModule top

ENTRY main {
  c_0 = s32[3] constant({1, 2, 3})
  
  ROOT slice_0 = s32[1] slice(c_0), slice={[0:1]}
})",
    // In this case the slice is inside of a call but it's not copied because
    // that would cause two identical subcomputations (foo is cloned when the
    // call graph is flattened) to become different.
    R"(
HloModule top

foo {
  p_0 = s32[3] parameter(0)
  ROOT slice_0 = s32[1] slice(p_0), slice={[0:1]}
}

ENTRY main {
  c_0 = s32[3] constant({1, 2, 3})
  c_1 = s32[3] constant({1, 2, 3})
  call_0 = s32[1] call(c_0), to_apply=foo, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  call_1 = s32[1] call(c_1), to_apply=foo, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  ROOT t = tuple(call_0, call_1, c_1)
})",
    // Don't copy the slice, because a slice of the same buffer is live on a
    // different path.
    R"(
HloModule top

ENTRY main {
  c_0 = s32[2] constant({0, 1})

  slice_0 = s32[1] slice(c_0), slice={[0:1]}
  slice_1 = s32[1] slice(c_0), slice={[1:2]}
  c_1 = s32[1] constant({5})
  add_0 = s32[1] add(slice_0, c_1)

  ROOT t = tuple(add_0, slice_1)
})",
    // In this case the slices won't be copied because the buffer defined in
    // `c_0` is live in 2 paths at the same time (the call uses a reference to
    // the buffer).
    R"(
HloModule top

foo {
  p_0 = s32[3] parameter(0)
  ROOT slice_0 = s32[1] slice(p_0), slice={[0:1]}
}

ENTRY main {
  c_0 = s32[3] constant({1, 2, 3})
  call_0 = s32[1] call(c_0), to_apply=foo, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\"}}"
  slice_1 = s32[1] slice(c_0), slice={[0:1]}
  ROOT tuple_0 = tuple(call_0, slice_1)
})",
};

INSTANTIATE_TEST_SUITE_P(, SliceCopyInserterDoesNotCopyTest,
                         ::testing::ValuesIn(test_values_does_not_copy));

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
