/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/schedulers/shortest_path_scheduler.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_information.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/ipu_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using ShortestPathSchedulerTest = HloTestBase;

TEST_F(ShortestPathSchedulerTest, TestShortestPathScheduler0) {
  //   p0
  //   |
  // cos0  p1
  //   |  /
  // add0   p2
  //   |  /
  //  add1
  //   |
  //  tuple
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  p2 = f32[] parameter(2)
  cos0 = f32[] cosine(p0)
  add0 = f32[] add(cos0, p1)
  add1 = f32[] add(add0, p2)
  ROOT tuple = (f16[]) tuple(add1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  IpuScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateShortestPathScheduler(CompilerInformation()));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_EQ(seq.size(), 7);

  EXPECT_EQ(seq[0]->name(), "p0");
  EXPECT_EQ(seq[1]->name(), "cos0");
  EXPECT_EQ(seq[2]->name(), "p1");
  EXPECT_EQ(seq[3]->name(), "add0");
  EXPECT_EQ(seq[4]->name(), "p2");
  EXPECT_EQ(seq[5]->name(), "add1");
  EXPECT_EQ(seq[6]->name(), "tuple");
}

TEST_F(ShortestPathSchedulerTest, TestShortestPathScheduler1) {
  //  p0
  //   |
  //  c0
  //   |
  //  c1
  //   |\  
  //   |  \ 
  //  c2   c3
  //   |   |
  //  c4   |
  //   |   |
  //   tuple
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  p0 = f32[] parameter(0)
  cos0 = f32[] cosine(p0)
  cos1 = f32[] cosine(cos0)
  cos2 = f32[] cosine(cos1)  
  cos3 = f32[] cosine(cos1)
  cos4 = f32[] cosine(cos2)  

  ROOT tuple = (f16[]) tuple(cos4, cos3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  IpuScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateShortestPathScheduler(CompilerInformation()));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_EQ(seq.size(), 7);

  EXPECT_EQ(seq[0]->name(), "p0");
  EXPECT_EQ(seq[1]->name(), "cos0");
  EXPECT_EQ(seq[2]->name(), "cos1");
  EXPECT_EQ(seq[3]->name(), "cos3");
  EXPECT_EQ(seq[4]->name(), "cos2");
  EXPECT_EQ(seq[5]->name(), "cos4");
  EXPECT_EQ(seq[6]->name(), "tuple");
}

// from old scheduler test
TEST_F(ShortestPathSchedulerTest, TestShortestPathScheduler2) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  %arg0 = f16[4] parameter(0)
  %arg1 = f16[4] parameter(1)
  %arg2 = f16[4] parameter(2)
  %sin.0 = f16[4] sine(f16[4] %arg0)
  %mul.0 = f16[4] multiply(f16[4] %sin.0, f16[4] %arg1)
  %mul.1 = f16[4] multiply(f16[4] %mul.0, f16[4] %arg2)
  ROOT %tuple = (f16[4], f16[4]) tuple(f16[4] %mul.0, f16[4] %mul.1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  IpuScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateShortestPathScheduler(CompilerInformation()));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_EQ(seq.size(), 7);
  EXPECT_EQ(seq[0]->name(), "arg0");
  EXPECT_EQ(seq[1]->name(), "sin.0");
  if (seq[2]->name() == "arg1") {
    EXPECT_EQ(seq[3]->name(), "mul.0");
    EXPECT_EQ(seq[4]->name(), "arg2");
    EXPECT_EQ(seq[5]->name(), "mul.1");
  } else {
    EXPECT_EQ(seq[2]->name(), "arg2");
    EXPECT_EQ(seq[3]->name(), "arg1");
    EXPECT_EQ(seq[4]->name(), "mul.0");
    EXPECT_EQ(seq[5]->name(), "mul.1");
  }
  EXPECT_EQ(seq[6]->name(), "tuple");
}

// from old scheduler test
TEST_F(ShortestPathSchedulerTest, TestShortestPathScheduler3) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  %arg0 = f16[4] parameter(0)
  %arg1 = f16[4] parameter(1)
  ROOT %tuple = (f16[4], f16[4]) tuple(f16[4] %arg0, f16[4] %arg1)
  %const = f16[] constant(1)
  %sum = f16[] add(f16[] const, f16[] const)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  IpuScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateShortestPathScheduler(CompilerInformation()));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_EQ(seq.size(), 5);
  EXPECT_EQ(seq[0]->name(), "const");
  EXPECT_EQ(seq[1]->name(), "sum");

  if (seq[2]->name() == "arg0") {
    EXPECT_EQ(seq[2]->name(), "arg0");
    EXPECT_EQ(seq[3]->name(), "arg1");
  } else {
    EXPECT_EQ(seq[2]->name(), "arg1");
    EXPECT_EQ(seq[3]->name(), "arg0");
  }
  EXPECT_EQ(seq[4]->name(), "tuple");
}

TEST_F(ShortestPathSchedulerTest, TestUnusedParameter) {
  //   p0
  //   |
  // cos0  p1
  //   |  /
  // add0   p2
  //   |  /
  //  add1
  //   |
  //  tuple
  std::string hlo_string = R"(
HloModule top
%cluster_1  {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  p2 = f32[] parameter(2)
  p_unused = f32[] parameter(3)
  cos0 = f32[] cosine(p0)
  add0 = f32[] add(cos0, p1)
  add1 = f32[] add(add0, p2)
  ROOT tuple = (f16[]) tuple(add1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  IpuScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateShortestPathScheduler(CompilerInformation()));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_EQ(seq.size(), 8);

  EXPECT_EQ(seq[0]->name(), "p0");
  EXPECT_EQ(seq[1]->name(), "cos0");
  EXPECT_EQ(seq[2]->name(), "p1");
  EXPECT_EQ(seq[3]->name(), "add0");
  EXPECT_EQ(seq[4]->name(), "p2");
  EXPECT_EQ(seq[5]->name(), "add1");
  EXPECT_EQ(seq[6]->name(), "tuple");
  EXPECT_EQ(seq[7]->name(), "p_unused");
}

TEST_F(ShortestPathSchedulerTest, TestReachableInfeedOutfeed) {
  //  after-all
  //   |
  //  infeed
  //   |
  //  tuple
  //   |
  //  input  after-all.36
  //   |    /
  //  outfeed.37
  std::string hlo_string = R"(
HloModule top
%cluster_1  {
  after-all = token[] after-all()
  infeed = ((f32[1,4,4,2], f32[]), token[]) infeed(after-all), infeed_config="01234567"
  tuple = (f32[1,4,4,2], f32[]) get-tuple-element(infeed), index=0
  input = f32[1,4,4,2] get-tuple-element(tuple), index=0
  after-all.36 = token[] after-all()
  outfeed.37 = token[] outfeed(input, after-all.36), outfeed_config="\010\001\022\005feed3\"\001\001(\001"
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  IpuScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateShortestPathScheduler(CompilerInformation()));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_EQ(seq.size(), 6);

  EXPECT_EQ(seq[0]->name(), "after-all.36");
  EXPECT_EQ(seq[1]->name(), "after-all");
  EXPECT_EQ(seq[2]->name(), "infeed");
  EXPECT_EQ(seq[3]->name(), "tuple");
  EXPECT_EQ(seq[4]->name(), "input");
  EXPECT_EQ(seq[5]->name(), "outfeed.37");
}

TEST_F(ShortestPathSchedulerTest, TestShortestPathScheduler2lines) {
  //   p0
  //   |
  //   c0
  //   |
  //   c1  p1
  //   |   |
  //   c2  c3
  //   |   |
  //   tuple
  std::string hlo_string = R"(
HloModule top

cluster_1  {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  cos0 = f32[] cosine(p0)
  cos1 = f32[] cosine(cos0)
  cos2 = f32[] cosine(cos1)
  cos3 = f32[] cosine(p1)  
  ROOT tuple = (f16[]) tuple(cos2, cos3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  IpuScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateShortestPathScheduler(CompilerInformation()));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_EQ(seq.size(), 7);
  EXPECT_EQ(seq[0]->name(), "p0");
  EXPECT_EQ(seq[1]->name(), "cos0");
  EXPECT_EQ(seq[2]->name(), "cos1");
  EXPECT_EQ(seq[3]->name(), "cos2");
  EXPECT_EQ(seq[4]->name(), "p1");
  EXPECT_EQ(seq[5]->name(), "cos3");
  EXPECT_EQ(seq[6]->name(), "tuple");
}

TEST_F(ShortestPathSchedulerTest, TestShortestPathScheduler2linesNoParameters) {
  // const0
  //   |
  //   c0
  //   |
  //   c1  const1
  //   |   |
  //   c2  c3
  //   |   |
  //   tuple
  std::string hlo_string = R"(
HloModule top

cluster_1  {

  const0 = f32[] constant(4)
  const1 = f32[] constant(3)
  cos0 = f32[] cosine(const0)
  cos1 = f32[] cosine(cos0)
  cos2 = f32[] cosine(cos1)
  cos3 = f32[] cosine(const1)  
  ROOT tuple = (f16[]) tuple(cos2, cos3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  IpuScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateShortestPathScheduler(CompilerInformation()));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_EQ(seq.size(), 7);
  EXPECT_EQ(seq[0]->name(), "const1");
  EXPECT_EQ(seq[1]->name(), "cos3");
  EXPECT_EQ(seq[2]->name(), "const0");
  EXPECT_EQ(seq[3]->name(), "cos0");
  EXPECT_EQ(seq[4]->name(), "cos1");
  EXPECT_EQ(seq[5]->name(), "cos2");
  EXPECT_EQ(seq[6]->name(), "tuple");
}

TEST_F(ShortestPathSchedulerTest, TestShortestPathSchedulerLarger) {
  //
  //         p1   p2
  //          \  /
  //      p0   a0    p3   p4
  //        \  |      \  /
  //          a1       a2
  //           |      /
  //          c0    c1
  //           |  /
  //           a3
  //           |
  //           c2
  //           |
  //         tuple
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  p2 = f32[] parameter(2)
  p3 = f32[] parameter(3)
  p4 = f32[] parameter(4)
  a0 = f32[] add(p1, p2)
  a1 = f32[] add(p0, a0)
  a2 = f32[] add(p3, p4)
  c0 = f32[] cosine(a1)
  c1 = f32[] cosine(a2) 
  a3 = f32[] add(c0, c1)  
  c2 = f32[] cosine(a3) 
  ROOT tuple = (f16[]) tuple(c2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  IpuScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateShortestPathScheduler(CompilerInformation()));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_EQ(seq.size(), 13);

  if (seq[0]->name() == "p1") {
    EXPECT_EQ(seq[1]->name(), "p2");
  } else {
    EXPECT_EQ(seq[0]->name(), "p2");
    EXPECT_EQ(seq[1]->name(), "p1");
  }
  EXPECT_EQ(seq[2]->name(), "a0");
  if (seq[3]->name() == "p0") {
    EXPECT_EQ(seq[3]->name(), "p0");
    EXPECT_EQ(seq[4]->name(), "a1");
    EXPECT_EQ(seq[5]->name(), "c0");
    if (seq[6]->name() == "p3") {
      EXPECT_EQ(seq[7]->name(), "p4");
    } else {
      EXPECT_EQ(seq[6]->name(), "p4");
      EXPECT_EQ(seq[7]->name(), "p3");
    }
    EXPECT_EQ(seq[8]->name(), "a2");
    EXPECT_EQ(seq[9]->name(), "c1");
  } else {
    if (seq[3]->name() == "p3") {
      EXPECT_EQ(seq[4]->name(), "p4");
    } else {
      EXPECT_EQ(seq[3]->name(), "p4");
      EXPECT_EQ(seq[4]->name(), "p3");
    }
    EXPECT_EQ(seq[5]->name(), "a2");
    EXPECT_EQ(seq[6]->name(), "c1");
    EXPECT_EQ(seq[7]->name(), "p0");
    EXPECT_EQ(seq[8]->name(), "a1");
    EXPECT_EQ(seq[9]->name(), "c0");
  }
  EXPECT_EQ(seq[10]->name(), "a3");
  EXPECT_EQ(seq[11]->name(), "c2");
  EXPECT_EQ(seq[12]->name(), "tuple");
}

TEST_F(ShortestPathSchedulerTest, TestShortestPathSchedulerWithOutFeed) {
  //
  //         p1   p2
  //          \  /
  //      p0   a0    p3   p4
  //        \  |      \  /
  //          a1       a2
  //           |      /  \
  //          c0    c1   s0
  //           |  /       |
  //           a3       tuple1
  //           |
  //           c2
  //           |
  //         tuple0
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  p2 = f32[] parameter(2)
  p3 = f32[] parameter(3)
  p4 = f32[] parameter(4)
  a0 = f32[] add(p1, p2)
  a1 = f32[] add(p0, a0)
  a2 = f32[] add(p3, p4)
  c0 = f32[] cosine(a1)
  c1 = f32[] cosine(a2) 
  a3 = f32[] add(c0, c1)  
  c2 = f32[] cosine(a3)
  s0 = f32[] sine(a2)
  tuple1 = (f16[]) tuple(s0)
  ROOT tuple0 = (f16[]) tuple(c2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  IpuScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateShortestPathScheduler(CompilerInformation()));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_EQ(seq.size(), 15);

  if (seq[0]->name() == "p1") {
    EXPECT_EQ(seq[1]->name(), "p2");
  } else {
    EXPECT_EQ(seq[0]->name(), "p2");
    EXPECT_EQ(seq[1]->name(), "p1");
  }
  EXPECT_EQ(seq[2]->name(), "a0");
  EXPECT_EQ(seq[3]->name(), "p0");
  EXPECT_EQ(seq[4]->name(), "a1");
  EXPECT_EQ(seq[5]->name(), "c0");

  if (seq[6]->name() == "p3") {
    EXPECT_EQ(seq[7]->name(), "p4");
  } else {
    EXPECT_EQ(seq[6]->name(), "p4");
    EXPECT_EQ(seq[7]->name(), "p3");
  }
  EXPECT_EQ(seq[8]->name(), "a2");
  EXPECT_EQ(seq[9]->name(), "s0");
  EXPECT_EQ(seq[10]->name(), "tuple1");

  EXPECT_EQ(seq[11]->name(), "c1");
  EXPECT_EQ(seq[12]->name(), "a3");
  EXPECT_EQ(seq[13]->name(), "c2");
  EXPECT_EQ(seq[14]->name(), "tuple0");
}

TEST_F(ShortestPathSchedulerTest, TestShortestPathSchedulerDisconnect) {
  //
  //         p1   p2
  //          \  /
  //      p0   a0    p3   p4
  //        \  |      \  /
  //          a1       a2
  //           |      /  \
  //          c0    c1   s0
  //           |  /       |
  //           a3       tuple1       p5
  //           |                     |
  //           c2                    s1
  //           |                     |
  //         tuple0                tuple2
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  p2 = f32[] parameter(2)
  p3 = f32[] parameter(3)
  p4 = f32[] parameter(4)
  p5 = f32[] parameter(5)  
  a0 = f32[] add(p1, p2)
  a1 = f32[] add(p0, a0)
  a2 = f32[] add(p3, p4)
  c0 = f32[] cosine(a1)
  c1 = f32[] cosine(a2) 
  a3 = f32[] add(c0, c1)  
  c2 = f32[] cosine(a3)
  s0 = f32[] sine(a2)
  tuple1 = (f16[]) tuple(s0)
  s1 = f32[] sine(p5)
  tuple2 = (f16[]) tuple(s1)  
  ROOT tuple0 = (f16[]) tuple(c2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  IpuScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateShortestPathScheduler(CompilerInformation()));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_EQ(seq.size(), 18);

  if (seq[0]->name() == "p1") {
    EXPECT_EQ(seq[1]->name(), "p2");
  } else {
    EXPECT_EQ(seq[0]->name(), "p2");
    EXPECT_EQ(seq[1]->name(), "p1");
  }
  EXPECT_EQ(seq[2]->name(), "a0");
  EXPECT_EQ(seq[3]->name(), "p0");
  EXPECT_EQ(seq[4]->name(), "a1");
  EXPECT_EQ(seq[5]->name(), "c0");

  if (seq[6]->name() == "p3") {
    EXPECT_EQ(seq[7]->name(), "p4");
  } else {
    EXPECT_EQ(seq[6]->name(), "p4");
    EXPECT_EQ(seq[7]->name(), "p3");
  }
  EXPECT_EQ(seq[8]->name(), "a2");
  EXPECT_EQ(seq[9]->name(), "s0");
  EXPECT_EQ(seq[10]->name(), "tuple1");

  EXPECT_EQ(seq[11]->name(), "c1");
  EXPECT_EQ(seq[12]->name(), "a3");
  EXPECT_EQ(seq[13]->name(), "c2");
  EXPECT_EQ(seq[14]->name(), "tuple0");

  EXPECT_EQ(seq[15]->name(), "p5");
  EXPECT_EQ(seq[16]->name(), "s1");
  EXPECT_EQ(seq[17]->name(), "tuple2");
}

TEST_F(ShortestPathSchedulerTest, TestShortestPathSchedulerWithLoops) {
  //     p0
  //     |
  //     c0
  //     |
  //     c1
  //     |
  //     c2--------|
  //     |         |
  //     c3-----|  |
  //     |      |  |
  //     c4     |  |
  //     |      |  |
  //     c5--|  |  |
  //     |   |  |  |
  //     c6  |  |  |
  //     |   |  |  |
  //     c7  |  |  |
  //     |   |  |  |
  //    s0  s1  s2 s3
  //      \  |  | /
  //        tuple0
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  p0 = f32[] parameter(0)
  c0 = f32[] cosine(p0)
  c1 = f32[] cosine(c0)
  c2 = f32[] cosine(c1)
  c3 = f32[] cosine(c2)
  c4 = f32[] cosine(c3)
  c5 = f32[] cosine(c4)
  c6 = f32[] cosine(c5)
  c7 = f32[] cosine(c6)
  s0 = f32[] sine(c7)
  s1 = f32[] sine(c5)
  s2 = f32[] sine(c3)
  s3 = f32[] sine(c2)      
  ROOT tuple0 = (f16[]) tuple(s0, s1, s2, s3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  IpuScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateShortestPathScheduler(CompilerInformation()));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_EQ(seq.size(), 14);
  EXPECT_EQ(seq[0]->name(), "p0");
  EXPECT_EQ(seq[1]->name(), "c0");
  EXPECT_EQ(seq[2]->name(), "c1");
  EXPECT_EQ(seq[3]->name(), "c2");
  EXPECT_EQ(seq[4]->name(), "s3");
  EXPECT_EQ(seq[5]->name(), "c3");
  EXPECT_EQ(seq[6]->name(), "s2");
  EXPECT_EQ(seq[7]->name(), "c4");
  EXPECT_EQ(seq[8]->name(), "c5");
  EXPECT_EQ(seq[9]->name(), "s1");
  EXPECT_EQ(seq[10]->name(), "c6");
  EXPECT_EQ(seq[11]->name(), "c7");
  EXPECT_EQ(seq[12]->name(), "s0");
  EXPECT_EQ(seq[13]->name(), "tuple0");
}

TEST_F(ShortestPathSchedulerTest, TestOnlyArg) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  ROOT p0 = f32[] parameter(0)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  IpuScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateShortestPathScheduler(CompilerInformation()));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_EQ(seq.size(), 1);
}

TEST_F(ShortestPathSchedulerTest, TestOnlyArgsAndTuple) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  p2 = f32[] parameter(2)
  ROOT t = (f32[], f32[], f32[]) tuple(p0, p1, p2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  IpuScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateShortestPathScheduler(CompilerInformation()));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_EQ(seq.size(), 4);
}

TEST_F(ShortestPathSchedulerTest, TestArgsWithControlPredecessors) {
  //   p0(-->p2)
  //   |
  // cos0  p1
  //   |  /
  // add0   p2
  //   |  /
  //  add1
  //   |
  //  tuple
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  p1 = f32[] parameter(1)
  p2 = f32[] parameter(2)
  p0 = f32[] parameter(0), control-predecessors={p2}
  cos0 = f32[] cosine(p0)
  add0 = f32[] add(cos0, p1)
  add1 = f32[] add(add0, p2)
  ROOT tuple = (f16[]) tuple(add1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  IpuScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateShortestPathScheduler(CompilerInformation()));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_EQ(seq.size(), 7);

  EXPECT_EQ(seq[0]->name(), "p1");
  EXPECT_EQ(seq[1]->name(), "p2");
  EXPECT_EQ(seq[2]->name(), "p0");
  EXPECT_EQ(seq[3]->name(), "cos0");
  EXPECT_EQ(seq[4]->name(), "add0");
  EXPECT_EQ(seq[5]->name(), "add1");
  EXPECT_EQ(seq[6]->name(), "tuple");
}

TEST_F(ShortestPathSchedulerTest, TestArgsWithControlDepsOnSameInst) {
  //   p0(-->p1)
  //   |
  // add0----p1
  //   |
  // cos0   p2
  //   |  /
  //  add1
  //   |
  //  tuple
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  p1 = f32[] parameter(1)
  p0 = f32[] parameter(0), control-predecessors={p1}
  add0 = f32[] add(p0, p1)
  cos0 = f32[] cosine(add0)
  p2 = f32[] parameter(2)
  add1 = f32[] add(cos0, p2)
  ROOT tuple = (f16[]) tuple(add1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  IpuScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateShortestPathScheduler(CompilerInformation()));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_EQ(seq.size(), 7);

  LOG(INFO) << "seq";
  for (auto* i : seq) {
    LOG(INFO) << i->name();
  }

  EXPECT_EQ(seq[0]->name(), "p1");
  EXPECT_EQ(seq[1]->name(), "p0");
  EXPECT_EQ(seq[2]->name(), "add0");
  EXPECT_EQ(seq[3]->name(), "cos0");
  EXPECT_EQ(seq[4]->name(), "p2");
  EXPECT_EQ(seq[5]->name(), "add1");
  EXPECT_EQ(seq[6]->name(), "tuple");
}

TEST_F(ShortestPathSchedulerTest, TestShortestPathParamsNoUsers) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  ROOT tuple = () tuple()
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  p2 = f32[] parameter(2)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  IpuScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateShortestPathScheduler(CompilerInformation()));

  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_EQ(seq.size(), 4);

  EXPECT_EQ(seq[0]->name(), "tuple");
  EXPECT_EQ(seq[1]->name(), "p0");
  EXPECT_EQ(seq[2]->name(), "p1");
  EXPECT_EQ(seq[3]->name(), "p2");
}

TEST_F(ShortestPathSchedulerTest, TestUnusedParametersControlDeps) {
  std::string hlo_string = R"(
HloModule top

comp {
  p0 = f32[10] parameter(0)
  p1 = f32[10] parameter(1)
  ROOT add = f32[10] add(f32[10] p0, f32[10] p1)
  p2 = f32[10] parameter(2), control-predecessors={p0, p1, add}
  p3 = f32[32,28,28,1] parameter(3), control-predecessors={p0, p1, add}
  p4 = f32[3,3,1,10] parameter(4), control-predecessors={p0, p1, add}
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  IpuScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateShortestPathScheduler(CompilerInformation()));

  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_EQ(seq.size(), 6);

  EXPECT_EQ(seq[0]->name(), "p0");
  EXPECT_EQ(seq[1]->name(), "p1");
  EXPECT_EQ(seq[2]->name(), "add");
  EXPECT_EQ(seq[3]->name(), "p2");
  EXPECT_EQ(seq[4]->name(), "p3");
  EXPECT_EQ(seq[5]->name(), "p4");
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
