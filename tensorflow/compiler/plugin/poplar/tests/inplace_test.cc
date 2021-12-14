/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_information.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/forward_allocation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_into_poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_late.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/while_loop_to_repeat_simplify.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/shortest_path_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using HloInplaceDependencyTest = HloTestBase;

void CheckInstructionCanBeLoweredNonInplace(const HloInstruction* inst,
                                            bool inplace,
                                            bool allow_non_inplace) {
  auto desc = GetInplaceDescription(inst);
  EXPECT_EQ(desc.IsInplaceType(), inplace);
  EXPECT_EQ(desc.AllowNonInplaceLowering(), allow_non_inplace);
}

TEST_F(HloInplaceDependencyTest, ResourceUpdate) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = s32[20] parameter(0)
  p1 = s32[20] parameter(1)

  s = s32[20] subtract(p0, p1),
  metadata={op_type="ResourceApplyGradientDescent" op_name="name"}

  ROOT t = (s32[20]) tuple(s)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  CompilerAnnotations annotations(module0);
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(module0);

  EXPECT_THAT(inplace_instructions.size(), 2);
  std::set<std::string> in_place_ops = {"s", "t"};
  for (const auto* inst : inplace_instructions) {
    EXPECT_THAT(in_place_ops.count(inst->name()), 1);
  }
}

TEST_F(HloInplaceDependencyTest, DynamicSliceUpdateInWhile) {
  std::string hlo = R"(
HloModule top

body {
  p_b = (s32[], s32[20], s32[20]) parameter(0)
  p0_b = s32[] get-tuple-element(p_b), index=0
  p1_b = s32[20] get-tuple-element(p_b), index=1
  p2_b = s32[20] get-tuple-element(p_b), index=2
  i_b = s32[1] reshape(p0_b)
  a_b = s32[1] dynamic-slice(p1_b, i_b), dynamic_slice_sizes={1}
  t1_b = s32[1] dynamic-slice(p2_b, a_b), dynamic_slice_sizes={1}
  t2_b = s32[1] dynamic-slice(p2_b, i_b), dynamic_slice_sizes={1}
  u0_b = s32[20] dynamic-update-slice(p2_b, t2_b, a_b)
  u1_b = s32[20] dynamic-update-slice(u0_b, t1_b, i_b)
  ROOT root_b = (s32[], s32[20], s32[20]) tuple(p0_b, p1_b, u1_b)
}

cond {
  p_c = (s32[], s32[20], s32[20]) parameter(0)
  p0_c = s32[] get-tuple-element(p_c), index=0
  z_c = s32[] constant(0)
  ROOT eq_c = pred[] compare(p0_c, z_c), direction=EQ
}

ENTRY c1 {
  p0 = s32[] parameter(0)
  p1 = s32[20] parameter(1)
  p2 = s32[20] parameter(2)
  t = (s32[], s32[20], s32[20]) tuple(p0, p1, p2)
  ROOT while = (s32[], s32[20], s32[20]) while(t), condition=cond, body=body
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  CompilerAnnotations annotations(module0);
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  std::set<std::string> in_place_ops = {"p0_b",  "p1_b",   "p2_b", "u0_b",
                                        "u1_b",  "root_b", "p0_c", "t",
                                        "while", "i_b"};
  auto inplace_instructions = GetInplaceInstructions(module0);

  EXPECT_THAT(inplace_instructions.size(), 10);
  for (const auto* inst : inplace_instructions) {
    EXPECT_THAT(in_place_ops.count(inst->name()), 1);
  }
}

TEST_F(HloInplaceDependencyTest, DynamicUpdateSlice) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = s32[] parameter(0)
  p1 = s32[20] parameter(1)
  p2 = s32[20] parameter(2)
  i = s32[1] reshape(s32[] p0)
  a = s32[1] dynamic-slice(p1, i), dynamic_slice_sizes={1}
  t1 = s32[1] dynamic-slice(p2, a), dynamic_slice_sizes={1}
  t2 = s32[1] dynamic-slice(p2, i), dynamic_slice_sizes={1}
  u0 = s32[20] dynamic-update-slice(p2, t2, a)
  u1 = s32[20] dynamic-update-slice(u0, t1, i)
  ROOT root = (s32[20]) tuple(u1)
 }
)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  auto* entry = module0->entry_computation();
  CompilerAnnotations annoations(module0);
  InplaceFinder inplaceFinder{annoations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  std::set<std::string> in_place_ops = {"u0", "u1", "root", "i"};
  auto inplace_instructions = GetInplaceInstructions(module0);

  EXPECT_THAT(inplace_instructions.size(), 4);
  for (const auto* inst : inplace_instructions) {
    EXPECT_THAT(in_place_ops.count(inst->name()), 1);
  }

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(IpuToMemorySchedulerAlgorithm(
          CreateShortestPathScheduler(CompilerInformation()))));

  EXPECT_TRUE(scheduler.Run(module0).ValueOrDie());

  auto instruction_order = module0->schedule().sequence(entry).instructions();

  EXPECT_THAT(instruction_order.size(), 10);

  std::map<std::string, unsigned int> order;
  for (unsigned int i = 0; i < instruction_order.size(); i++) {
    order[instruction_order[i]->name()] = i;
  }

  // Normal ordering
  EXPECT_TRUE(order.at("p0") < order.at("i"));
  EXPECT_TRUE(order.at("i") < order.at("a"));
  EXPECT_TRUE(order.at("i") < order.at("t2"));
  EXPECT_TRUE(order.at("i") < order.at("u1"));
  EXPECT_TRUE(order.at("a") < order.at("t1"));
  EXPECT_TRUE(order.at("a") < order.at("u0"));
  EXPECT_TRUE(order.at("t1") < order.at("u1"));
  EXPECT_TRUE(order.at("t2") < order.at("u0"));
  EXPECT_TRUE(order.at("u0") < order.at("u1"));

  // All updates need to occur after all reads
  EXPECT_TRUE(order.at("t1") < order.at("u0"));
  EXPECT_TRUE(order.at("t2") < order.at("u1"));
}

TEST_F(HloInplaceDependencyTest, MultipleUpdateInPlacePeers) {
  std::string hlo = R"(
  HloModule top

  ENTRY c1 {
    p0 = s32[20] parameter(0)
    p1 = s32[20] parameter(1)
    u0 = s32[20] add(p0, p1),
    metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
    u1 = s32[20] subtract(p0, p1),
    metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
    ROOT root = (s32[20], s32[20]) tuple(u0, u1)
   }
  )";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  auto* entry = module0->entry_computation();
  CompilerAnnotations annotations(module0);
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  std::set<std::string> either_in_place_ops = {"u0", "u1"};
  std::set<std::string> in_place_ops = {"root"};
  auto inplace_instructions = GetInplaceInstructions(module0);
  // Only one of the binary ops can be update in place
  EXPECT_THAT(inplace_instructions.size(), 2);
  for (const auto* inst : inplace_instructions) {
    EXPECT_TRUE(either_in_place_ops.count(inst->name()) == 1 ||
                in_place_ops.count(inst->name()) == 1);
  }

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(IpuToMemorySchedulerAlgorithm(
          CreateShortestPathScheduler(CompilerInformation()))));

  EXPECT_TRUE(scheduler.Run(module0).ValueOrDie());

  auto instruction_order = module0->schedule().sequence(entry).instructions();
  EXPECT_THAT(instruction_order.size(), 5);

  std::map<std::string, unsigned int> order;
  for (unsigned int i = 0; i < instruction_order.size(); i++) {
    order[instruction_order[i]->name()] = i;
  }

  // Normal ordering
  EXPECT_TRUE(order.at("p0") < order.at("u0"));
  EXPECT_TRUE(order.at("p1") < order.at("u0"));
  EXPECT_TRUE(order.at("p0") < order.at("u1"));
  EXPECT_TRUE(order.at("p1") < order.at("u1"));
  EXPECT_TRUE(order.at("u0") < order.at("root"));
  EXPECT_TRUE(order.at("u1") < order.at("root"));
}

TEST_F(HloInplaceDependencyTest, MultipleInplaceWithInterdependency) {
  std::string hlo = R"(
    HloModule top

    ENTRY c1 {
      p0 = s32[20] parameter(0)
      p1 = s32[20] parameter(1)
      u0 = s32[20] add(p0, p1),
      metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
      u1 = s32[20] subtract(u0, p0),
      metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
      ROOT root = (s32[20]) tuple(u1)
     }
    )";

  auto module =
      HloRunner::CreateModuleFromString(hlo, GetDebugOptionsForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  auto* entry = module0->entry_computation();
  CompilerAnnotations annotations(module0);
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(module0);

  // Only one of the binary ops can be update in place
  std::set<std::string> in_place_ops = {"u1", "root"};
  EXPECT_THAT(inplace_instructions.size(), 2);
  for (const auto* inst : inplace_instructions) {
    EXPECT_THAT(in_place_ops.count(inst->name()), 1);
  }

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(IpuToMemorySchedulerAlgorithm(
          CreateShortestPathScheduler(CompilerInformation()))));

  EXPECT_TRUE(scheduler.Run(module0).ValueOrDie());

  auto instruction_order = module0->schedule().sequence(entry).instructions();

  EXPECT_THAT(instruction_order.size(), 5);

  std::map<std::string, unsigned int> order;
  for (unsigned int i = 0; i < instruction_order.size(); i++) {
    order[instruction_order[i]->name()] = i;
  }

  // Normal ordering
  EXPECT_TRUE(order.at("p0") < order.at("u0"));
  EXPECT_TRUE(order.at("p1") < order.at("u0"));
  EXPECT_TRUE(order.at("u0") < order.at("u1"));
  EXPECT_TRUE(order.at("u1") < order.at("root"));
}

TEST_F(HloInplaceDependencyTest, MultipleInplaceWithRightOrder) {
  std::string hlo = R"(
    HloModule top

    ENTRY c1 {
      p0 = s32[20] parameter(0)
      p1 = s32[20] parameter(1)
      p2 = s32[20] parameter(2)
      u0 = s32[20] add(p0, p1),
      metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
      u1 = s32[20] add(p1, p2),
      metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
      ROOT root = (s32[20], s32[20]) tuple(u0, u1)
     }
    )";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  auto* entry = module0->entry_computation();
  CompilerAnnotations annotations(module0);
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(module0);

  std::set<std::string> in_place_ops = {"u0", "u1", "root"};
  EXPECT_THAT(inplace_instructions.size(), 3);
  for (const auto* inst : inplace_instructions) {
    EXPECT_THAT(in_place_ops.count(inst->name()), 1);
  }

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(IpuToMemorySchedulerAlgorithm(
          CreateShortestPathScheduler(CompilerInformation()))));

  EXPECT_TRUE(scheduler.Run(module0).ValueOrDie());

  auto instruction_order = module0->schedule().sequence(entry).instructions();

  EXPECT_THAT(instruction_order.size(), 6);

  std::map<std::string, unsigned int> order;
  for (unsigned int i = 0; i < instruction_order.size(); i++) {
    order[instruction_order[i]->name()] = i;
  }

  // Normal ordering
  EXPECT_TRUE(order.at("p0") < order.at("u0"));
  EXPECT_TRUE(order.at("p1") < order.at("u0"));
  EXPECT_TRUE(order.at("p2") < order.at("u1"));
  EXPECT_TRUE(order.at("u0") < order.at("u1"));
  EXPECT_TRUE(order.at("u1") < order.at("root"));
}

TEST_F(HloInplaceDependencyTest, InplaceCorrectDependencies) {
  std::string hlo = R"(
    HloModule top

    ENTRY c1 {
      p0 = s32[20] parameter(0)
      p1 = s32[20] parameter(1)
      u0 = s32[20] add(p0, p1),
      metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
      u1 = s32[20] add(p0, u0),
      metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
      ROOT root = (s32[20], s32[20]) tuple(u0, u1)
     }
    )";

  auto module =
      HloRunner::CreateModuleFromString(hlo, GetDebugOptionsForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  auto* entry = module0->entry_computation();
  CompilerAnnotations annotations(module0);
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());
  auto inplace_instructions = GetInplaceInstructions(module0);

  std::set<std::string> in_place_ops = {"u1", "root"};
  EXPECT_THAT(inplace_instructions.size(), 2);
  for (const auto* inst : inplace_instructions) {
    EXPECT_THAT(in_place_ops.count(inst->name()), 1);
  }

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(IpuToMemorySchedulerAlgorithm(
          CreateShortestPathScheduler(CompilerInformation()))));

  EXPECT_TRUE(scheduler.Run(module0).ValueOrDie());

  auto instruction_order = module0->schedule().sequence(entry).instructions();

  EXPECT_THAT(instruction_order.size(), 5);

  std::map<std::string, unsigned int> order;
  for (unsigned int i = 0; i < instruction_order.size(); i++) {
    order[instruction_order[i]->name()] = i;
  }

  // Normal ordering
  EXPECT_TRUE(order.at("p0") < order.at("u0"));
  EXPECT_TRUE(order.at("p1") < order.at("u0"));
  EXPECT_TRUE(order.at("u0") < order.at("u1"));
  EXPECT_TRUE(order.at("u1") < order.at("root"));
}

TEST_F(HloInplaceDependencyTest, InplaceInputOuputStreamedAndResourceVariable) {
  std::string hlo = R"(
    HloModule top

    ENTRY c1 {
      p0 = s32[20] parameter(0)
      p1 = s32[20] parameter(1)
      u0 = s32[20] add(p1, p0)
      u1 = s32[20] add(p0, u0),
      metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
      ROOT root = (s32[20], s32[20]) tuple(u0, u1)
     }
    )";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  CompilerAnnotations annotations(module0);
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(module0);

  EXPECT_THAT(inplace_instructions.size(), 3);
  std::set<std::string> in_place_ops = {"u0", "u1", "root"};
  for (auto i : inplace_instructions) {
    EXPECT_TRUE(in_place_ops.count(i->name()));
  }
}

TEST_F(HloInplaceDependencyTest, InplaceAddCopyForInplaceReadOnly) {
  std::string hlo = R"(
    HloModule top

    ENTRY c1 {
      p0 = f32[20] parameter(0)
      a = f32[40] concatenate(p0, p0), dimensions={0}
      b = f32[20] negate(p0)
      c = f32[40] reshape(a)
      d = f32[40] log(a)
      ROOT t = (f32[20], f32[40], f32[40]) tuple(b, c, d)
     }
    )";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  CompilerAnnotations annotations(module0);
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());
  auto inplace_instructions = GetInplaceInstructions(module0);

  EXPECT_THAT(inplace_instructions.size(), 3);
  // Neither a or c can be inplace.
  std::set<std::string> in_place_ops = {"b", "d", "t"};
  for (auto i : inplace_instructions) {
    EXPECT_TRUE(in_place_ops.count(i->name()));
  }
}

TEST_F(HloInplaceDependencyTest, InplaceDontAddCopyForInplaceReadOnly) {
  std::string hlo = R"(
    HloModule top

    ENTRY c1 {
      p0 = f32[20] parameter(0)
      a = f32[40] concatenate(p0, p0), dimensions={0}
      b = f32[20] negate(p0)
      c = f32[40] add(a, a)
      ROOT t = (f32[20], f32[40]) tuple(b, c)
     }
    )";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  auto* comp = module0->entry_computation();
  CompilerAnnotations annotations(module0);
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());
  auto inplace_instructions = GetInplaceInstructions(module0);

  EXPECT_THAT(inplace_instructions.size(), 3);
  // b can be inplace as long as a is executed after b and c.
  std::set<std::string> in_place_ops = {"a", "b", "t"};
  for (auto i : inplace_instructions) {
    EXPECT_TRUE(in_place_ops.count(i->name()));
  }
  auto* a = comp->GetInstructionWithName("a");
  auto* b = comp->GetInstructionWithName("b");
  auto* c = comp->GetInstructionWithName("c");
  EXPECT_THAT(b->control_predecessors(), ::testing::UnorderedElementsAre(a, c));
}

TEST_F(HloInplaceDependencyTest, InplaceElementwiseBinary) {
  std::string hlo = R"(
    HloModule top

    ENTRY c1 {
      p0 = s32[20] parameter(0)
      p1 = s32[20] parameter(1)
      ROOT u0 = s32[20] add(p1, p0)
     }
    )";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  CompilerAnnotations annotations(module0);
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(module0);
  EXPECT_THAT(inplace_instructions.size(), 1);

  auto* inst = *(inplace_instructions.begin());
  EXPECT_THAT(inst->name(), "u0");
}

TEST_F(HloInplaceDependencyTest, ScaledInplaceHighPriority) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  a = f32[20] parameter(0)
  b = f32[20] parameter(1)
  c = f32[] constant(2)
  c_bcast = f32[20] broadcast(f32[] c), dimensions={}
  bc = f32[20] multiply(b, c_bcast)
  ROOT res = f32[20] add(a, bc),
  metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
}

)";

  auto module =
      HloRunner::CreateModuleFromString(hlo, GetDebugOptionsForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);

  FuseOpsIntoPoplarOps foipo(annotations);
  EXPECT_TRUE(foipo.Run(module0).ValueOrDie());
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  // Make sure that the only inplace instruction is a call to scaled add to.
  auto inplace_instructions = GetInplaceInstructions(module0);
  EXPECT_THAT(inplace_instructions.size(), 1);
  auto inplace_description =
      GetInplaceDescription(*inplace_instructions.begin());
  EXPECT_TRUE(inplace_description.GetType() ==
              HloInstructionType::kInplaceReadWrite);

  EXPECT_TRUE(module0->entry_computation()->root_instruction() ==
              *inplace_instructions.begin());
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ScaledInplaceXbY)(
      module0->entry_computation()->root_instruction()));
}

TEST_F(HloInplaceDependencyTest, PadZeroInplaceType) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  a = f32[20] parameter(0)
  b = f32[10] parameter(1)
  c = f32[] constant(0)
  ROOT pad = f32[20] pad(b, c), padding=0_0x0_0x0_0x0_10
}

)";

  auto module =
      HloRunner::CreateModuleFromString(hlo, GetDebugOptionsForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);

  FuseOpsLate fol(annotations);
  EXPECT_TRUE(fol.Run(module0).ValueOrDie());
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(module0);
  EXPECT_THAT(inplace_instructions.size(), 1);
  auto inplace_description =
      GetInplaceDescription(*inplace_instructions.begin());
  EXPECT_TRUE(inplace_description.GetType() ==
              HloInstructionType::kInplaceReadOnly);

  EXPECT_TRUE(module0->entry_computation()->root_instruction() ==
              *inplace_instructions.begin());
  EXPECT_TRUE(IsPopOpsFusion(module0->entry_computation()->root_instruction(),
                             "zero_pad"));
}

TEST_F(HloInplaceDependencyTest, InplaceInsideWhile) {
  const char* const hlo = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element((s32[],s32[]) p_body), index=0
  const = s32[] constant(1)
  add = s32[] add(p_body.0, const)
  p_body.1 = s32[] get-tuple-element((s32[],s32[]) p_body), index=1
  ROOT root = (s32[],s32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element((s32[],s32[]) p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[] constant(10)
  while_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(while_init), condition=condition, body=body
}
)";

  auto module =
      HloRunner::CreateModuleFromString(hlo, GetDebugOptionsForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  CompilerAnnotations annotations(module0);
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(module0);
  EXPECT_THAT(inplace_instructions.size(), 7);
  std::set<std::string> in_place_ops = {
      "p_body.2", "root", "p_body.1", "add", "p_cond.1", "while_init", "while"};
  for (auto i : inplace_instructions) {
    EXPECT_TRUE(in_place_ops.count(i->name()));
  }
}

TEST_F(HloInplaceDependencyTest, CustomPoplarOpNotInplace) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = s32[20] parameter(0)
  p1 = s32[20] parameter(1)

  c = s32[20] custom-call(p0, p1), custom_call_target="LstmLayerFwd", backend_config="{\"num_channels\":4, \"is_training\":false, \"partials_dtype\":\"DT_FLOAT\", \"activation\":\"tanh\", \"recurrent_activation\":\"sigmoid\", \"options\":\"{}\"}\n"

  ROOT t = (s32[20]) tuple(c)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(module0);

  EXPECT_THAT(inplace_instructions.size(), 1);
  auto* inst = *(inplace_instructions.begin());
  EXPECT_THAT(inst->name(), "t");
}

TEST_F(HloInplaceDependencyTest, TestAllGTEsInplace) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = (s32[20], s32[20], s32[20], s32[20]) parameter(0)
  p0_0 = s32[20] get-tuple-element(p0), index=0
  p0_1 = s32[20] get-tuple-element(p0), index=1
  p0_2 = s32[20] get-tuple-element(p0), index=2
  a = s32[20] add(p0_0, p0_1)
  ROOT a2 = s32[20] add(a, p0_2)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  CompilerAnnotations annotations(module0);
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(module0);
  EXPECT_THAT(inplace_instructions.size(), 5);
  std::set<std::string> in_place_ops = {"p0_0", "p0_1", "p0_2", "a", "a2"};
  for (auto i : inplace_instructions) {
    EXPECT_TRUE(in_place_ops.count(i->name()));
  }
}

TEST_F(HloInplaceDependencyTest, TestGTENotInplace) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = (s32[20], s32[20], s32[20], s32[20]) parameter(0)
  p0_0 = s32[20] get-tuple-element(p0), index=0

  c = s32[20] custom-call(p0), custom_call_target="LstmLayerFwd", backend_config="{\"num_channels\":4, \"is_training\":false, \"partials_dtype\":\"DT_FLOAT\", \"activation\":\"tanh\", \"recurrent_activation\":\"sigmoid\", \"options\":\"{}\"}\n"

  ROOT a = s32[20] add(p0_0, c)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(module0);
  EXPECT_THAT(inplace_instructions.size(), 1);
  std::set<std::string> in_place_ops = {"a"};
  for (auto i : inplace_instructions) {
    EXPECT_TRUE(in_place_ops.count(i->name()));
  }
}

TEST_F(HloInplaceDependencyTest, TestInplaceReadOnly) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = s32[20] parameter(0)
  ROOT c = s32[10,2] reshape(p0)
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  CompilerAnnotations annotations(module0);
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(module0);
  EXPECT_THAT(inplace_instructions.size(), 1);
  std::set<std::string> in_place_ops = {"c"};
  for (auto i : inplace_instructions) {
    EXPECT_TRUE(in_place_ops.count(i->name()));
  }
}

TEST_F(HloInplaceDependencyTest, TestInplaceReadOnlyBeforeReadWrite) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f32[20] parameter(0)
  a = f32[20,1] reshape(p0)
  ROOT b = f32[20] log(p0)
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  auto* comp = module0->entry_computation();
  CompilerAnnotations annotations(module0);
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(module0);
  EXPECT_THAT(inplace_instructions.size(), 2);
  std::set<std::string> in_place_ops = {"a", "b"};
  for (auto i : inplace_instructions) {
    EXPECT_TRUE(in_place_ops.count(i->name()));
  }
  auto* a = comp->GetInstructionWithName("a");
  auto* b = comp->GetInstructionWithName("b");
  EXPECT_EQ(b->control_predecessors()[0], a);
}

TEST_F(HloInplaceDependencyTest, TestInplaceMultipleReadOnlyClusters) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f32[20] parameter(0)
  a = f32[10, 2] reshape(p0)
  b = f32[40] concatenate(p0, p0), dimensions={0}
  c = f32[20] negate(p0)

  d = f32[20] reshape(a)
  e = f32[10,2] log(a)

  f = f32[20] slice(b), slice={[0:20]}
  g = f32[20] slice(b), slice={[20:40]}

  h = f16[20] convert(d)
  i = f16[20] convert(f)
  j = f16[20] convert(g)
  k = f16[20] add(h, i)
  l = f16[20] add(j, k)
  ROOT t = (f32[20], f16[10,2]) tuple(l, e)
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  auto* comp = module0->entry_computation();
  CompilerAnnotations annotations(module0);
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());
  auto inplace_instructions = GetInplaceInstructions(module0);
  EXPECT_THAT(inplace_instructions.size(), 8);
  std::set<std::string> in_place_ops = {"c", "d", "e", "f", "g", "k", "l", "t"};
  for (auto i : inplace_instructions) {
    EXPECT_TRUE(in_place_ops.count(i->name()));
  }
  auto* e = comp->GetInstructionWithName("e");
  auto* h = comp->GetInstructionWithName("h");
  EXPECT_EQ(h->control_successors()[0], e);
}

TEST_F(HloInplaceDependencyTest, TestUpdateScalarInRowsInplace) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  softmax = f16[20,30] parameter(0)
	labels = s32[20] parameter(1)

  ROOT c = f16[20, 30] custom-call(softmax, labels), custom_call_target="UpdateScalarInRows"
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(module0);
  EXPECT_THAT(inplace_instructions.size(), 1);
  std::set<std::string> in_place_ops = {"update-scalar-in-rows"};
  for (auto i : inplace_instructions) {
    EXPECT_TRUE(in_place_ops.count(i->name()));
  }
}

TEST_F(HloInplaceDependencyTest, TestRootIsAlwaysChosen) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f32[] parameter(0)

  l = f32[] add(p0, p0)

  l2 = f32[] add(l, l)

  ROOT t = (f32[]) tuple(l)

  l3 = f32[] add(l, l)
}

)";

  auto config = GetModuleConfigForTest();

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);

  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(module0);
  EXPECT_THAT(inplace_instructions.size(), 1);
  std::set<std::string> in_place_ops = {"t"};
  for (auto i : inplace_instructions) {
    EXPECT_TRUE(in_place_ops.count(i->name()));
  }
}

TEST_F(HloInplaceDependencyTest, OperandIsRootInstruction) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = s32[20] parameter(0)
  p1 = s32[20] parameter(1)

  ROOT s = s32[20] subtract(p0, p1)
  c = token[] custom-call(s), custom_call_target="CustomOp"
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  CompilerAnnotations annotations(module0);
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(module0);

  // 'c' cannot be inplace because the output of 's' must be preserved until
  // after the computation completes.
  EXPECT_THAT(inplace_instructions.size(), 1);
  std::set<std::string> in_place_ops = {"s"};
  for (const auto* inst : inplace_instructions) {
    LOG(INFO) << inst->name();
    EXPECT_THAT(in_place_ops.count(inst->name()), 1);
  }
}

TEST_F(HloInplaceDependencyTest, OperandIsRootTupleInstruction) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = s32[20] parameter(0)
  p1 = s32[20] parameter(1)

  s = s32[20] subtract(p0, p1)
  ROOT t = (s32[20]) tuple(s)
  c = token[] custom-call(t), custom_call_target="CustomOp"
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  CompilerAnnotations annotations(module0);
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(module0);

  // 'c' cannot be inplace because the output of 's' must be preserved until
  // after the computation completes.
  EXPECT_THAT(inplace_instructions.size(), 2);
  std::set<std::string> in_place_ops = {"s", "t"};
  for (const auto* inst : inplace_instructions) {
    EXPECT_THAT(in_place_ops.count(inst->name()), 1);
  }
}

TEST_F(HloInplaceDependencyTest, ReadOnlyAndReadWrite) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f32[7,8] parameter(0)
  p1 = f32[7,7] parameter(1)
  slice = f32[7,7] slice(p0), slice={[0:7], [0:7]}
  add = f32[7,7] add(slice, p1)
  dot = f32[7,8] dot(add, p0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT r = f32[56] reshape(dot)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  CompilerAnnotations annotations(module0);
  InplaceFinder inplaceFinder{annotations};
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(module0);
  EXPECT_THAT(inplace_instructions.size(), 2);

  HloInstruction* add = FindInstruction(module0, "add");
  HloInstruction* r = FindInstruction(module0, "r");
  // 'slice' cannot be inplace because p0 is used in `dot` and `add` is inplace.
  EXPECT_TRUE(inplace_instructions.contains(add));
  EXPECT_TRUE(inplace_instructions.contains(r));
}

void EnableLoopAliasAnalysis(HloInstruction* repeat_loop) {
  auto backend_config =
      repeat_loop->backend_config<PoplarBackendConfig>().ValueOrDie();
  auto* call_config = backend_config.mutable_call_config();
  call_config->set_type(PoplarBackendConfig::CallConfig::RepeatLoop);
  auto* repeat_cfg = call_config->mutable_repeat_config();
  repeat_cfg->set_allow_finer_alias_analysis(true);
  TF_CHECK_OK(repeat_loop->set_backend_config(backend_config));
}

TEST_F(HloInplaceDependencyTest, InplaceRepeatLoop) {
  const char* const hlo = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[],f32[10],f32[10]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  const = s32[] constant(1)
  add = s32[] add(p_body.0, const)
  p_body.1 = s32[] get-tuple-element(p_body), index=1
  p_body.2 = f32[10] get-tuple-element(p_body), index=2
  p_body.3 = f32[10] get-tuple-element(p_body), index=3
  add2 = f32[10] add(p_body.2, p_body.3)
  ROOT root = (s32[],s32[],f32[10],f32[10]) tuple(add, p_body.1, add2, p_body.3)
}

condition {
  p_cond = (s32[],s32[],f32[10],f32[10]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[] constant(10)
  p0 = f32[10] parameter(0)
  p1 = f32[10] parameter(1)
  while_init = (s32[],s32[],f32[10],f32[10]) tuple(const_0, const_1, p0, p1)
  ROOT while = (s32[],s32[],f32[10],f32[10]) while(while_init), condition=condition, body=body
}
)";

  auto module =
      HloRunner::CreateModuleFromString(hlo, GetDebugOptionsForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  EXPECT_TRUE(WhileLoopToRepeatSimplify().Run(module0).ValueOrDie());
  EXPECT_TRUE(HloDCE().Run(module0).ValueOrDie());
  auto entry = module0->entry_computation();
  EXPECT_TRUE(IsRepeatLoop(entry->root_instruction()));
  EnableLoopAliasAnalysis(entry->root_instruction());
  EXPECT_TRUE(
      InplaceFinder(CompilerAnnotations(module0)).Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(entry);
  EXPECT_THAT(inplace_instructions.size(), 1);
  EXPECT_THAT(GetInplaceDescription(entry->root_instruction())
                  .GetInplaceOperandIndices(),
              ::testing::ElementsAre(2));
}

TEST_F(HloInplaceDependencyTest, InplaceRepeatLoop2) {
  const char* const hlo = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[],f32[10],f32[10]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  const = s32[] constant(1)
  add = s32[] add(p_body.0, const)
  p_body.1 = s32[] get-tuple-element(p_body), index=1
  p_body.2 = f32[10] get-tuple-element(p_body), index=2
  p_body.3 = f32[10] get-tuple-element(p_body), index=3
  add2 = f32[10] add(p_body.2, p_body.3)
  add3 = f32[10] add(p_body.2, p_body.3)
  ROOT root = (s32[],s32[],f32[10],f32[10]) tuple(add, p_body.1, add2, add3)
}

condition {
  p_cond = (s32[],s32[],f32[10],f32[10]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[] constant(10)
  p0 = f32[10] parameter(0)
  p1 = f32[10] parameter(1)
  while_init = (s32[],s32[],f32[10],f32[10]) tuple(const_0, const_1, p0, p1)
  ROOT while = (s32[],s32[],f32[10],f32[10]) while(while_init), condition=condition, body=body
}
)";

  auto module =
      HloRunner::CreateModuleFromString(hlo, GetDebugOptionsForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  EXPECT_TRUE(WhileLoopToRepeatSimplify().Run(module0).ValueOrDie());
  EXPECT_TRUE(HloDCE().Run(module0).ValueOrDie());
  auto entry = module0->entry_computation();
  EXPECT_TRUE(IsRepeatLoop(entry->root_instruction()));
  EnableLoopAliasAnalysis(entry->root_instruction());
  EXPECT_TRUE(
      InplaceFinder(CompilerAnnotations(module0)).Run(module0).ValueOrDie());

  auto inplace_instructions = GetInplaceInstructions(entry);
  EXPECT_THAT(inplace_instructions.size(), 1);
  EXPECT_THAT(GetInplaceDescription(entry->root_instruction())
                  .GetInplaceOperandIndices(),
              ::testing::ElementsAre(2, 3));
}

TEST_F(HloInplaceDependencyTest, InplaceNoInplace) {
  const char* const hlo = R"(
HloModule top

body {
  p_b = (s32[], s32[20], s32[20]) parameter(0)
  p0_b = s32[] get-tuple-element(p_b), index=0
  p1_b = s32[20] get-tuple-element(p_b), index=1
  p2_b = s32[20] get-tuple-element(p_b), index=2
  i_b = s32[1] reshape(p0_b)
  a_b = s32[1] dynamic-slice(p1_b, i_b), dynamic_slice_sizes={1}
  t1_b = s32[1] dynamic-slice(p2_b, a_b), dynamic_slice_sizes={1}
  t2_b = s32[1] dynamic-slice(p2_b, i_b), dynamic_slice_sizes={1}
  u0_b = s32[20] dynamic-update-slice(p2_b, t2_b, a_b)
  u1_b = s32[20] dynamic-update-slice(u0_b, t1_b, i_b)
  concat = f32[40] concatenate(u0_b, u1_b), dimensions={0}
  slice = f32[20] slice(concat), slice={[10:30]}
  add = f32[20] add(u1_b, slice)
  ROOT root_b = (s32[], s32[20], s32[20]) tuple(p0_b, p1_b, add)
}

cond {
  p_c = (s32[], s32[20], s32[20]) parameter(0)
  p0_c = s32[] get-tuple-element(p_c), index=0
  z_c = s32[] constant(0)
  ROOT eq_c = pred[] compare(p0_c, z_c), direction=EQ
}

ENTRY c1 {
  p0 = s32[] parameter(0)
  p1 = s32[20] parameter(1)
  p2 = s32[20] parameter(2)
  t = (s32[], s32[20], s32[20]) tuple(p0, p1, p2)
  ROOT while = (s32[], s32[20], s32[20]) while(t), condition=cond, body=body
  }
)";

  auto module =
      HloRunner::CreateModuleFromString(hlo, GetDebugOptionsForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CheckInstructionCanBeLoweredNonInplace(FindInstruction(module0, "while"),
                                         true, false);
  CheckInstructionCanBeLoweredNonInplace(FindInstruction(module0, "add"), true,
                                         true);
  CheckInstructionCanBeLoweredNonInplace(FindInstruction(module0, "slice"),
                                         true, true);
  CheckInstructionCanBeLoweredNonInplace(FindInstruction(module0, "concat"),
                                         true, false);
  CheckInstructionCanBeLoweredNonInplace(FindInstruction(module0, "u1_b"), true,
                                         false);
  CheckInstructionCanBeLoweredNonInplace(FindInstruction(module0, "t1_b"),
                                         false, true);
  CheckInstructionCanBeLoweredNonInplace(FindInstruction(module0, "i_b"), true,
                                         false);
  CheckInstructionCanBeLoweredNonInplace(FindInstruction(module0, "root_b"),
                                         true, false);
  CheckInstructionCanBeLoweredNonInplace(FindInstruction(module0, "p0"), false,
                                         true);
  CheckInstructionCanBeLoweredNonInplace(FindInstruction(module0, "eq_c"),
                                         false, true);
}

struct InplaceFinderCopyTestSpec {
  bool convert_to_repeat;
};

void PrintTo(const InplaceFinderCopyTestSpec& spec, std::ostream* os) {
  *os << "{ convert_to_repeat: " << spec.convert_to_repeat << " }";
}

struct InplaceFinderCopyTest
    : public HloTestBase,
      public ::testing::WithParamInterface<InplaceFinderCopyTestSpec> {};

TEST_P(InplaceFinderCopyTest, TestCopyDeduce) {
  const char* const hlo = R"(
HloModule top

body {
  p_b = (s32[], s32[20], s32[20]) parameter(0)
  p0_b = s32[] get-tuple-element(p_b), index=0
  p1_b = s32[20] get-tuple-element(p_b), index=1
  p2_b = s32[20] get-tuple-element(p_b), index=2
  i_b = s32[1] reshape(p0_b)
  a_b = s32[1] dynamic-slice(p1_b, i_b), dynamic_slice_sizes={1}
  t1_b = s32[1] dynamic-slice(p2_b, a_b), dynamic_slice_sizes={1}
  t2_b = s32[1] dynamic-slice(p2_b, i_b), dynamic_slice_sizes={1}
  u0_b = s32[20] dynamic-update-slice(p2_b, t2_b, a_b)
  u1_b = s32[20] dynamic-update-slice(u0_b, t1_b, i_b)
  concat = s32[40] concatenate(u0_b, u1_b), dimensions={0}
  slice = s32[20] slice(concat), slice={[10:30]}
  add = s32[20] add(u1_b, slice)
  inc = s32[] constant(1)
  p0_b_next = s32[] add(p0_b, inc)
  ROOT root_b = (s32[], s32[20], s32[20]) tuple(p0_b_next, p1_b, add)
}

cond {
  p_c = (s32[], s32[20], s32[20]) parameter(0)
  p0_c = s32[] get-tuple-element(p_c), index=0
  z_c = s32[] constant(20)
  ROOT eq_c = pred[] compare(p0_c, z_c), direction=LE
}

ENTRY c1 {
  c0 = s32[] constant(0)
  p0 = s32[20] parameter(0)
  t = (s32[], s32[20], s32[20]) tuple(c0, p0, p0)
  ROOT while = (s32[], s32[20], s32[20]) while(t), condition=cond, body=body
  }
)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  CompilerAnnotations annotations(module0);
  const bool convert_to_repeat = GetParam().convert_to_repeat;
  EXPECT_TRUE(CopyInserter().Run(module0).ValueOrDie());
  if (convert_to_repeat) {
    EXPECT_TRUE(WhileLoopToRepeatSimplify().Run(module0).ValueOrDie());
  }
  EXPECT_TRUE(InplaceFinder(annotations).Run(module0).ValueOrDie());
  EXPECT_TRUE(AllocationFinder(annotations).Run(module0).ValueOrDie());
  EXPECT_TRUE(ForwardAllocation(annotations).Run(module0).ValueOrDie());
  EXPECT_EQ(annotations.tensor_allocation_map.size(),
            convert_to_repeat ? 4 : 5);
  HloInstruction* copy = FindInstruction(module0, "copy");
  auto clone_methods = GetCopyCloneMethod(copy).ConsumeValueOrDie();
  for (auto& clone_method : clone_methods.leaves()) {
    EXPECT_EQ(clone_method.second, CloneMethod_DeduceNewOrderOrPreserveAliases);
  }
  // Allocation target is dynamic slice, so this copy will actually convert
  // layout to sliceable.
  auto t =
      annotations.tensor_allocation_map.at(TensorLocation{copy->operand(0), 0});
  EXPECT_EQ(t.tgt->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_EQ(t.input_index, 0);
}

INSTANTIATE_TEST_SUITE_P(InplaceFinderCopyTestCases, InplaceFinderCopyTest,
                         ::testing::Values(InplaceFinderCopyTestSpec{true},
                                           InplaceFinderCopyTestSpec{false}));

TEST_P(InplaceFinderCopyTest, TestNonInplaceLoopReallocatingCopies) {
  const char* const hlo = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[],f32[10],f32[10]) parameter(0)
  p_body.0 = s32[] get-tuple-element(p_body), index=0
  const = s32[] constant(1)
  add = s32[] add(p_body.0, const)
  p_body.1 = s32[] get-tuple-element(p_body), index=1
  p_body.2 = f32[10] get-tuple-element(p_body), index=2
  p_body.3 = f32[10] get-tuple-element(p_body), index=3
  add2 = f32[10] add(p_body.2, p_body.3)
  ROOT root = (s32[],s32[],f32[10],f32[10]) tuple(add, p_body.1, add2, p_body.3)
}

condition {
  p_cond = (s32[],s32[],f32[10],f32[10]) parameter(0)
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] compare(p_cond.0, const), direction=LT
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[] constant(10)
  add_1 = s32[] add(const_0, const_1)
  p0 = f32[10] parameter(0)
  p1 = f32[10] parameter(1)
  while_init = (s32[],s32[],f32[10],f32[10]) tuple(const_0, add_1, p0, p1)
  while_output = (s32[],s32[],f32[10],f32[10]) while(while_init), condition=condition, body=body
  ROOT root = (s32[], s32[], (s32[],s32[],f32[10],f32[10]), (s32[],s32[],f32[10],f32[10])) tuple(p0, while_init, while_output)
}
)";

  auto module =
      HloRunner::CreateModuleFromString(hlo, GetDebugOptionsForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const bool convert_to_repeat = GetParam().convert_to_repeat;
  if (convert_to_repeat) {
    EXPECT_TRUE(WhileLoopToRepeatSimplify().Run(module0).ValueOrDie());
  }

  auto* entry = module0->entry_computation();
  auto* root = entry->root_instruction();
  auto* loop = root->operand(2);
  CHECK_NOTNULL(loop);
  EXPECT_TRUE(
      InplaceFinder(CompilerAnnotations(module0)).Run(module0).ValueOrDie());

  // Make sure both while and repeat wasn't lowered inplace.
  EXPECT_TRUE(!IsLoweredInplace(loop));
  int64 operand_n;
  if (convert_to_repeat) {
    EXPECT_TRUE(IsRepeatLoop(loop));
  } else {
    EXPECT_EQ(loop->opcode(), HloOpcode::kWhile);
  }
  for (const HloInstruction* op : loop->operands()) {
    EXPECT_EQ(op->opcode(), HloOpcode::kCopy);
    TF_ASSERT_OK_AND_ASSIGN(auto clone_method, GetCopyCloneMethod(op));
    EXPECT_TRUE(absl::c_all_of(
        clone_method.leaves(),
        [](const std::pair<ShapeIndex, CloneMethod>& method) {
          return method.second == CloneMethod_DeduceNewOrderOrExpandAliases;
        }));
  }
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
