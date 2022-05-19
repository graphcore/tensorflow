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

#include <gtest/gtest.h>

#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_dataflow_analysis.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_liveness.h"

namespace xla {
namespace poplarplugin {
namespace {

InstructionBufferSets MockOutputBufferAssignment(
    std::initializer_list<
        std::pair<HloInstruction*, std::vector<HloPoplarBuffer*>>>
        assignments) {
  InstructionBufferSets instruction_buffer_sets;
  for (auto& assignment : assignments) {
    auto* inst = assignment.first;
    auto& buffer_uses = assignment.second;

    HloPoplarBufferSet buffer_set;
    for (auto* buffer : buffer_uses) {
      buffer_set.AddBuffer(buffer);
    }

    InstructionPoplarBufferSet instruction_buffer_set(inst->shape());
    instruction_buffer_set.SetOutputBufferSet({}, buffer_set);

    instruction_buffer_sets.emplace(
        std::make_pair(inst, instruction_buffer_set));
  }

  return instruction_buffer_sets;
}

HloPoplarBuffer MockHloPoplarBuffer(HloPoplarBuffer::Id id,
                                    HloInstruction* owner) {
  return HloPoplarBuffer(id, HloPoplarPosition{owner, {}},
                         BufferLocality::kDeviceMemory);
}

struct UsedBuffersTest : HloTestFixture {
  const char* basic_outfeed_add_hlo = R"(
 HloModule top

 ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  arg1 = f32[2] parameter(1)
  ROOT add = f32[2] add(arg0, arg1)
  after-all = token[] after-all()
  outfeed = token[] outfeed(add, after-all)
}
)";

  void SetUp() override {
    ASSERT_TRUE(SetUpHloModule(basic_outfeed_add_hlo));

    arg0_ = FindInstruction(hlo_module_, "arg0");
    arg1_ = FindInstruction(hlo_module_, "arg1");
    add_ = FindInstruction(hlo_module_, "add");
    after_all_ = FindInstruction(hlo_module_, "after-all");
    outfeed_ = FindInstruction(hlo_module_, "outfeed");
  }

  HloInstruction* arg0_;
  HloInstruction* arg1_;
  HloInstruction* add_;
  HloInstruction* after_all_;
  HloInstruction* outfeed_;
};

TEST_F(UsedBuffersTest, NoInputs) {
  auto buffer0 = MockHloPoplarBuffer(0, arg0_);
  auto buffer1 = MockHloPoplarBuffer(1, arg1_);
  auto instruction_buffer_sets =
      MockOutputBufferAssignment({{arg0_, {&buffer0}}, {arg1_, {&buffer1}}});

  HloInstructionMap<HloPoplarBufferSet> buffer_usage =
      FindUsedBuffers(hlo_module_, instruction_buffer_sets);

  ASSERT_EQ(buffer_usage[arg0_], HloPoplarBufferSet({&buffer0}));
  ASSERT_EQ(buffer_usage[arg1_], HloPoplarBufferSet({&buffer1}));
}

TEST_F(UsedBuffersTest, NoOutputs) {
  auto buffer0 = MockHloPoplarBuffer(0, arg0_);
  auto buffer1 = MockHloPoplarBuffer(1, arg1_);
  auto instruction_buffer_sets = MockOutputBufferAssignment(
      {{arg0_, {&buffer0}}, {arg1_, {&buffer1}}, {add_, {&buffer0}}});

  HloInstructionMap<HloPoplarBufferSet> buffer_usage =
      FindUsedBuffers(hlo_module_, instruction_buffer_sets);

  ASSERT_EQ(buffer_usage[after_all_].size(), 0);
  ASSERT_EQ(buffer_usage[outfeed_], HloPoplarBufferSet({&buffer0}));
}

TEST_F(UsedBuffersTest, OutOfPlaceInputs) {
  auto buffer0 = MockHloPoplarBuffer(0, arg0_);
  auto buffer1 = MockHloPoplarBuffer(1, arg1_);
  auto buffer2 = MockHloPoplarBuffer(2, add_);
  auto instruction_buffer_sets = MockOutputBufferAssignment(
      {{arg0_, {&buffer0}}, {arg1_, {&buffer1}}, {add_, {&buffer2}}});

  HloInstructionMap<HloPoplarBufferSet> buffer_usage =
      FindUsedBuffers(hlo_module_, instruction_buffer_sets);

  ASSERT_EQ(buffer_usage[add_],
            HloPoplarBufferSet({&buffer1, &buffer2, &buffer0}));
}

TEST_F(UsedBuffersTest, PartialInplaceInputs) {
  auto buffer0 = MockHloPoplarBuffer(0, arg0_);
  auto buffer1 = MockHloPoplarBuffer(1, arg1_);
  auto instruction_buffer_sets = MockOutputBufferAssignment(
      {{arg0_, {&buffer0}}, {arg1_, {&buffer1}}, {add_, {&buffer0}}});

  HloInstructionMap<HloPoplarBufferSet> buffer_usage =
      FindUsedBuffers(hlo_module_, instruction_buffer_sets);

  ASSERT_EQ(buffer_usage[add_], HloPoplarBufferSet({&buffer1, &buffer0}));
}

const char* simple_pad_hlo = R"(
 HloModule top

 ENTRY cluster_1 {
  arg0 = f32[3,4] parameter(0)
  c0 = f32[] constant(0.0)
  ROOT pad = f32[8,10] pad(f32[3,4] arg0, f32[] c0), padding=3_2x1_5
}
)";
TEST_F(UsedBuffersTest, FullyInplaceInputs) {
  ASSERT_TRUE(SetUpHloModule(simple_pad_hlo));

  auto* arg0 = FindInstruction(hlo_module_, "arg0");
  auto* c0 = FindInstruction(hlo_module_, "c0");
  auto* pad = FindInstruction(hlo_module_, "pad");

  auto buffer0 = MockHloPoplarBuffer(0, arg0);
  auto buffer1 = MockHloPoplarBuffer(1, c0);
  auto instruction_buffer_sets = MockOutputBufferAssignment(
      {{arg0, {&buffer0}}, {c0, {&buffer1}}, {pad, {&buffer0, &buffer1}}});

  HloInstructionMap<HloPoplarBufferSet> buffer_usage =
      FindUsedBuffers(hlo_module_, instruction_buffer_sets);

  ASSERT_EQ(buffer_usage[pad], HloPoplarBufferSet({&buffer1, &buffer0}));
}

const char* duplicate_inputs_hlo = R"(
 HloModule top

 ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  ROOT add = f32[2] add(arg0, arg0)
}
)";
TEST_F(UsedBuffersTest, DuplicateInputs) {
  ASSERT_TRUE(SetUpHloModule(duplicate_inputs_hlo));

  auto* arg0 = FindInstruction(hlo_module_, "arg0");
  auto* add = FindInstruction(hlo_module_, "add");

  auto buffer0 = MockHloPoplarBuffer(0, arg0);
  auto buffer1 = MockHloPoplarBuffer(1, add);
  auto instruction_buffer_sets =
      MockOutputBufferAssignment({{arg0, {&buffer0}}, {add, {&buffer1}}});

  HloInstructionMap<HloPoplarBufferSet> buffer_usage =
      FindUsedBuffers(hlo_module_, instruction_buffer_sets);

  ASSERT_EQ(buffer_usage[add], HloPoplarBufferSet({&buffer1, &buffer0}));
}

struct LivenessProperties : HloTestFixture {
  const char* simple_read_write = R"(
 HloModule top

 ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  arg1 = f32[2] parameter(1)
  add = f32[2] add(arg0, arg1)
  arg2 = f32[2] parameter(2)
  ROOT add2 = f32[2] add(add, arg2)
}
)";
  void SetUp() override {
    ASSERT_TRUE(SetUpHloModule(simple_read_write));

    arg0_ = FindInstruction(hlo_module_, "arg0");
    arg1_ = FindInstruction(hlo_module_, "arg1");
    add_ = FindInstruction(hlo_module_, "add");
    arg2_ = FindInstruction(hlo_module_, "arg2");
    add2_ = FindInstruction(hlo_module_, "add2");

    ASSERT_TRUE(arg0_);
    ASSERT_TRUE(arg1_);
    ASSERT_TRUE(add_);
    ASSERT_TRUE(arg2_);
    ASSERT_TRUE(add2_);
  }

  void DefinesBuffer(HloInstruction* inst, HloPoplarBuffer::Id id) {
    auto* buffer = BufferDefinition(id, inst);
    buffer_usage_[inst].AddBuffer(buffer);
  }

  HloPoplarBuffer* BufferDefinition(HloPoplarBuffer::Id id,
                                    HloInstruction* inst) {
    auto result = buffers_.emplace(id, MockHloPoplarBuffer(id, inst));
    auto& buffer = result.first->second;
    return &buffer;
  }

  void ReferencesBuffers(HloInstruction* inst,
                         const std::vector<HloPoplarBuffer::Id>& ids) {
    for (auto id : ids) {
      buffer_usage_[inst].AddBuffer(BufferReference(id));
    }
  }

  HloPoplarBuffer* BufferReference(HloPoplarBuffer::Id id) {
    auto& buffer = buffers_.at(id);
    return &buffer;
  }

  HloInstruction* arg0_;
  HloInstruction* arg1_;
  HloInstruction* add_;
  HloInstruction* arg2_;
  HloInstruction* add2_;

  HloPoplarBuffer::Id buffer0_ = 0;
  HloPoplarBuffer::Id buffer1_ = 1;
  HloPoplarBuffer::Id buffer2_ = 2;
  HloPoplarBuffer::Id buffer3_ = 3;

  std::map<HloPoplarBuffer::Id, HloPoplarBuffer> buffers_;
  HloInstructionMap<HloPoplarBufferSet> buffer_usage_;
};

TEST_F(LivenessProperties, DefinedPerScheduledInstruction) {
  using ::testing::Key;
  using ::testing::UnorderedElementsAre;

  std::vector<HloInstruction*> schedule = {arg0_, arg1_, add_, arg2_, add2_};
  auto program_liveness = GenerateProgramLiveness(schedule, {});

  ASSERT_THAT(program_liveness,
              UnorderedElementsAre(Key(arg0_), Key(arg1_), Key(add_),
                                   Key(arg2_), Key(add2_)));
}

TEST_F(LivenessProperties, DeadAtDefinition) {
  using ::testing::IsEmpty;

  DefinesBuffer(arg0_, buffer0_);
  DefinesBuffer(arg1_, buffer1_);
  ReferencesBuffers(add_, {buffer0_, buffer1_});

  auto program_liveness =
      GenerateProgramLiveness({arg0_, arg1_, add_}, buffer_usage_);

  ASSERT_THAT(program_liveness[arg0_], IsEmpty());
}

TEST_F(LivenessProperties, AliveAfterDefinition) {
  using ::testing::Contains;
  using ::testing::UnorderedElementsAre;

  DefinesBuffer(arg0_, buffer0_);
  DefinesBuffer(arg1_, buffer1_);
  ReferencesBuffers(add_, {buffer0_, buffer1_});

  auto program_liveness =
      GenerateProgramLiveness({arg0_, arg1_, add_}, buffer_usage_);

  ASSERT_THAT(program_liveness[arg1_], UnorderedElementsAre(buffer0_));
  ASSERT_THAT(program_liveness[add_], Contains(buffer0_));
}

TEST_F(LivenessProperties, AliveAtReadWrite) {
  using ::testing::UnorderedElementsAre;

  DefinesBuffer(arg0_, buffer0_);
  DefinesBuffer(arg1_, buffer1_);
  ReferencesBuffers(add_, {buffer0_, buffer1_});

  auto program_liveness =
      GenerateProgramLiveness({arg0_, arg1_, add_}, buffer_usage_);

  ASSERT_THAT(program_liveness[add_], UnorderedElementsAre(buffer0_, buffer1_));
}

TEST_F(LivenessProperties, DeadAfterLastRead) {
  using ::testing::Contains;
  using ::testing::Not;

  DefinesBuffer(arg0_, buffer0_);
  DefinesBuffer(arg1_, buffer1_);
  ReferencesBuffers(add_, {buffer1_, buffer0_});
  DefinesBuffer(arg2_, buffer2_);
  DefinesBuffer(add2_, buffer3_);
  ReferencesBuffers(add2_, {buffer2_, buffer0_});

  auto program_liveness = GenerateProgramLiveness(
      {arg0_, arg1_, add_, arg2_, add2_}, buffer_usage_);

  ASSERT_THAT(program_liveness[arg2_], Not(Contains(buffer1_)));
  ASSERT_THAT(program_liveness[add2_], Not(Contains(buffer1_)));
}

TEST_F(LivenessProperties, FinalResultAliveAtDefinition) {
  using ::testing::Contains;

  DefinesBuffer(arg0_, buffer0_);
  DefinesBuffer(arg1_, buffer1_);
  ReferencesBuffers(add_, {buffer1_, buffer0_});
  DefinesBuffer(arg2_, buffer2_);
  DefinesBuffer(add2_, buffer3_);
  ReferencesBuffers(add2_, {buffer2_, buffer0_});

  auto program_liveness = GenerateProgramLiveness(
      {arg0_, arg1_, add_, arg2_, add2_}, buffer_usage_);

  ASSERT_THAT(program_liveness[add_], Contains(buffer3_));
}
const char* unused_param_hlo = R"(
 HloModule top

 ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  arg1 = f32[2] parameter(1)
  arg2 = f32[2] parameter(2)
  ROOT add = f32[2] add(arg2, arg1)
}
)";
TEST_F(LivenessProperties, UnusedBufferIsAlwaysDead) {
  using ::testing::Contains;
  using ::testing::Not;

  ASSERT_TRUE(SetUpHloModule(unused_param_hlo));

  auto* arg0 = FindInstruction(hlo_module_, "arg0");
  auto* arg1 = FindInstruction(hlo_module_, "arg1");
  auto* arg2 = FindInstruction(hlo_module_, "arg2");
  auto* add = FindInstruction(hlo_module_, "add");

  DefinesBuffer(arg0, buffer0_);
  DefinesBuffer(arg1, buffer1_);
  DefinesBuffer(arg2, buffer2_);
  ReferencesBuffers(add, {buffer2_, buffer1_});

  auto program_liveness =
      GenerateProgramLiveness({arg0, arg1, arg2, add}, buffer_usage_);

  for (auto& item : program_liveness) {
    const auto& live_set = item.second;
    ASSERT_THAT(live_set, Not(Contains(buffer0_)));
  }
}

using LivenessTest = HloTestFixture;

const char* multiply_add_hlo = R"(
 HloModule top

 ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  mul0 = f32[2] multiply(arg0, arg0)
  arg1 = f32[2] parameter(1)
  mul1 = f32[2] multiply(arg1, arg1)
  add = f32[2] add(mul0, mul1)
  arg2 = f32[100] parameter(2)
  ROOT result = (f32[2], f32[100]) tuple(add, arg2)
}
)";
TEST_F(LivenessTest, DataflowAnalysisIntegration) {
  using ::testing::IsEmpty;
  using ::testing::UnorderedElementsAre;

  ASSERT_TRUE(SetUpHloModule(multiply_add_hlo));

  auto* arg0 = FindInstruction(hlo_module_, "arg0");
  auto* mul0 = FindInstruction(hlo_module_, "mul0");
  auto* arg1 = FindInstruction(hlo_module_, "arg1");
  auto* mul1 = FindInstruction(hlo_module_, "mul1");
  auto* add = FindInstruction(hlo_module_, "add");
  auto* arg2 = FindInstruction(hlo_module_, "arg2");
  auto* result = FindInstruction(hlo_module_, "result");

  auto annotations = CompilerAnnotations(hlo_module_);
  TF_ASSERT_OK_AND_ASSIGN(
      auto analysis, HloPoplarDataflowAnalysis::Run(hlo_module_, annotations));

  auto instruction_buffer_sets = analysis->GetInstructionBufferSets();
  auto buffer_uses = FindUsedBuffers(hlo_module_, instruction_buffer_sets);

  std::vector<HloInstruction*> schedule = {arg0, mul0, arg1,  mul1,
                                           add,  arg2, result};
  auto program_liveness = GenerateProgramLiveness(schedule, buffer_uses);

  // Expecting 3 buffers - one for each param. Multiply/add should be
  // inplace on their first operand.
  const HloPoplarBuffer::Id buffer0 = 0;
  const HloPoplarBuffer::Id buffer1 = 1;
  const HloPoplarBuffer::Id buffer2 = 2;

  ASSERT_THAT(program_liveness[arg0], IsEmpty());
  ASSERT_THAT(program_liveness[mul0], UnorderedElementsAre(buffer0));
  ASSERT_THAT(program_liveness[arg1], UnorderedElementsAre(buffer0));
  ASSERT_THAT(program_liveness[mul1], UnorderedElementsAre(buffer0, buffer1));
  ASSERT_THAT(program_liveness[add], UnorderedElementsAre(buffer0, buffer1));
  ASSERT_THAT(program_liveness[arg2], UnorderedElementsAre(buffer0));
  ASSERT_THAT(program_liveness[result], UnorderedElementsAre(buffer0, buffer2));
}

using MemoryUsageTest = HloTestFixture;

const char* stub_hlo = R"(
 HloModule top

 ENTRY cluster_1 {
  inst0 = f32[] parameter(0)
  inst1 = f32[] parameter(1)
  inst2 = f32[] parameter(2)
  ROOT inst3 = f32[2] constant(1)
}
)";
TEST_F(MemoryUsageTest, SumOfLiveBuffers) {
  ASSERT_TRUE(SetUpHloModule(stub_hlo));

  auto* inst = FindInstruction(hlo_module_, "inst3");

  HloInstructionMap<HloPoplarBufferIdSet> program_liveness;
  program_liveness[inst] = {0, 1, 2, 3, 5};

  absl::flat_hash_map<HloPoplarBuffer::Id, int64_t> buffer_sizes_in_bytes;
  buffer_sizes_in_bytes[0] = 100;
  buffer_sizes_in_bytes[1] = 100;
  buffer_sizes_in_bytes[2] = 300;
  buffer_sizes_in_bytes[3] = 10;
  buffer_sizes_in_bytes[5] = 5000;
  buffer_sizes_in_bytes[6] = 99999;

  const auto memory_estimate =
      EstimateMinimumLiveMemory(program_liveness, buffer_sizes_in_bytes);
  // 5510, since we're only using buffer 0,1,2,3,5.
  ASSERT_EQ(memory_estimate, 5510);
}

TEST_F(MemoryUsageTest, MaxOfInstructionUsage) {
  ASSERT_TRUE(SetUpHloModule(stub_hlo));

  auto* inst0 = FindInstruction(hlo_module_, "inst0");
  auto* inst1 = FindInstruction(hlo_module_, "inst1");
  auto* inst2 = FindInstruction(hlo_module_, "inst2");
  auto* inst3 = FindInstruction(hlo_module_, "inst3");

  HloInstructionMap<HloPoplarBufferIdSet> program_liveness;
  program_liveness[inst0] = {0};
  program_liveness[inst1] = {2};
  program_liveness[inst2] = {1};
  program_liveness[inst3] = {4};

  absl::flat_hash_map<HloPoplarBuffer::Id, int64_t> buffer_sizes_in_bytes;
  buffer_sizes_in_bytes[0] = 5451;
  buffer_sizes_in_bytes[1] = 230;
  buffer_sizes_in_bytes[2] = 7500;
  buffer_sizes_in_bytes[3] = 999999;
  buffer_sizes_in_bytes[4] = 499;

  const auto memory_estimate =
      EstimateMinimumLiveMemory(program_liveness, buffer_sizes_in_bytes);
  // 7500, since this is the max of 5451, 230, 7, 500, 499.
  ASSERT_EQ(memory_estimate, 7500);
}

TEST_F(MemoryUsageTest, DataflowAnalysisIntegration) {
  ASSERT_TRUE(SetUpHloModule(multiply_add_hlo));

  auto* arg0 = FindInstruction(hlo_module_, "arg0");
  auto* mul0 = FindInstruction(hlo_module_, "mul0");
  auto* arg1 = FindInstruction(hlo_module_, "arg1");
  auto* mul1 = FindInstruction(hlo_module_, "mul1");
  auto* add = FindInstruction(hlo_module_, "add");
  auto* arg2 = FindInstruction(hlo_module_, "arg2");
  auto* result = FindInstruction(hlo_module_, "result");

  auto annotations = CompilerAnnotations(hlo_module_);
  TF_ASSERT_OK_AND_ASSIGN(
      auto analysis, HloPoplarDataflowAnalysis::Run(hlo_module_, annotations));

  auto instruction_buffer_sets = analysis->GetInstructionBufferSets();
  auto buffer_uses = FindUsedBuffers(hlo_module_, instruction_buffer_sets);

  absl::flat_hash_map<HloPoplarBuffer::Id, int64_t> buffer_sizes;
  for (auto& item : buffer_uses) {
    buffer_sizes.merge(DeviceMemoryBufferSizesInBytes(item.second));
  }

  std::vector<HloInstruction*> schedule;

  schedule = {arg0, mul0, arg1, mul1, add, arg2, result};
  auto program_liveness = GenerateProgramLiveness(schedule, buffer_uses);
  auto memory_estimate =
      EstimateMinimumLiveMemory(program_liveness, buffer_sizes);
  // The root instruction depends on f32[2] add and f32[100] arg2, so this
  // schedule uses at least 4*2 + 4*100 = 408 bytes of device memory.
  ASSERT_EQ(memory_estimate, 408);

  // By moving arg2 to the front of the schedule we extend its life time, so at
  // mul1 we have arg2 and mul0 and arg1 alive, and 4*100 + 4*2 + 4*2 = 416.
  schedule = {arg2, arg0, mul0, arg1, mul1, add, result};
  program_liveness = GenerateProgramLiveness(schedule, buffer_uses);
  memory_estimate = EstimateMinimumLiveMemory(program_liveness, buffer_sizes);
  ASSERT_EQ(memory_estimate, 416);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
