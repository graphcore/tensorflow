/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/gather_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using GatherSimplifierTest = HloTestBase;

int64 GetNumMultiSlice(const HloComputation* comp) {
  return absl::c_count_if(comp->instructions(),
                          IsPoplarInstruction(PoplarOp::MultiSlice));
}

int64 GetNumGather(const HloComputation* comp) {
  return absl::c_count_if(comp->instructions(), [](const HloInstruction* inst) {
    return inst->opcode() == HloOpcode::kGather;
  });
}

TEST_F(GatherSimplifierTest, TestMultiSlice0) {
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
    operand = s32[3,4] constant({{1,2,3,4},{1,2,3,4},{1,2,3,4}})
    indices = s32[5] parameter(0)
    ROOT gather = s32[5,4] gather(operand, indices),
         offset_dims={1},
         collapsed_slice_dims={0},
         start_index_map={0},
         index_vector_dim=1,
         slice_sizes={1, 4}
    }
    )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();
  auto* module = module0.get();

  CompilerAnnotations annotations(module);
  GatherSimplifier gs;
  EXPECT_TRUE(gs.Run(module).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(GetNumMultiSlice(module->entry_computation()), 1);
  EXPECT_EQ(GetNumGather(module->entry_computation()), 0);
}

TEST_F(GatherSimplifierTest, TestMultiSlice1) {
  // Will fail collapsed slice dims must be {0}
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
    operand = s32[3,3] parameter(0)
    indices = s32[2] parameter(1)
    ROOT gather = s32[3,2] gather(operand, indices),
        offset_dims={0},
        collapsed_slice_dims={1},
        start_index_map={1},
        index_vector_dim=1,
        slice_sizes={3, 1}
    }
    )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();
  auto* module = module0.get();

  CompilerAnnotations annotations(module);
  GatherSimplifier gs;
  EXPECT_FALSE(gs.Run(module).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(GetNumMultiSlice(module->entry_computation()), 0);
  EXPECT_EQ(GetNumGather(module->entry_computation()), 1);
}

TEST_F(GatherSimplifierTest, TestMultiSlice2) {
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
    operand = s32[3,3,2] parameter(0)
    indices = s32[100] parameter(1)
    gather = s32[100,3,2] gather(operand, indices),
        offset_dims={1,2},
        collapsed_slice_dims={0},
        start_index_map={0},
        index_vector_dim=1,
        slice_sizes={1,3,2}
    }
    )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();
  auto* module = module0.get();

  CompilerAnnotations annotations(module);
  GatherSimplifier gs;
  EXPECT_TRUE(gs.Run(module).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(GetNumMultiSlice(module->entry_computation()), 1);
  EXPECT_EQ(GetNumGather(module->entry_computation()), 0);
  EXPECT_TRUE(Match(root, m::Reshape(m::CustomCall(m::Reshape(m::Parameter(0)),
                                                   m::Parameter(1)))));
}

TEST_F(GatherSimplifierTest, TestMultiSlice3) {
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
    operand = s32[3,0] parameter(0)
    indices = s32[2] parameter(1)
    ROOT gather = s32[2,0] gather(operand, indices),
        offset_dims={1},
        collapsed_slice_dims={0},
        start_index_map={0},
        index_vector_dim=1,
        slice_sizes={1, 0}
  }
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();
  auto* module = module0.get();

  CompilerAnnotations annotations(module);
  GatherSimplifier gs;
  EXPECT_TRUE(gs.Run(module).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(GetNumMultiSlice(module->entry_computation()), 1);
  EXPECT_EQ(GetNumGather(module->entry_computation()), 0);
}

TEST_F(GatherSimplifierTest, TestMultiSlice4) {
  // Will fail as operand_shape.rank() is 0.
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
    operand = f32[] parameter(0)
    indices = s32[0]{0} parameter(1)
     ROOT gather = f32[] gather(operand, indices),
       offset_dims={},
       collapsed_slice_dims={},
       start_index_map={},
       index_vector_dim=0,
       slice_sizes={}
  }
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();
  auto* module = module0.get();

  CompilerAnnotations annotations(module);
  GatherSimplifier gs;
  EXPECT_FALSE(gs.Run(module).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(GetNumMultiSlice(module->entry_computation()), 0);
  EXPECT_EQ(GetNumGather(module->entry_computation()), 1);
}

TEST_F(GatherSimplifierTest, TestMultiSlice5) {
  // TODO(T14037): enable this by adding transposes.
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
    operand = s32[3,8] parameter(0)
    indices = s32[2] parameter(1)
    ROOT gather = s32[3,2] gather(operand, indices),
        offset_dims={0},
        collapsed_slice_dims={1},
        start_index_map={0},
        index_vector_dim=1,
        slice_sizes={3, 1}
  }
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();
  auto* module = module0.get();

  CompilerAnnotations annotations(module);
  EXPECT_FALSE(GatherSimplifier().Run(module).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(GetNumMultiSlice(module->entry_computation()), 0);
  EXPECT_EQ(GetNumGather(module->entry_computation()), 1);
}

TEST_F(GatherSimplifierTest, TestMultiSlice6) {
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
    operand = s32[6] parameter(0)
    indices = s32[] parameter(1)
    gather = s32[] gather(operand, indices),
        offset_dims={},
        collapsed_slice_dims={0},
        start_index_map={0},
        index_vector_dim=0,
        slice_sizes={1}
    }
    )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();
  auto* module = module0.get();

  CompilerAnnotations annotations(module);
  GatherSimplifier gs;
  EXPECT_TRUE(gs.Run(module).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(GetNumMultiSlice(module->entry_computation()), 1);
  EXPECT_EQ(GetNumGather(module->entry_computation()), 0);
  EXPECT_TRUE(Match(root, m::Reshape(m::CustomCall(m::Reshape(m::Parameter(0)),
                                                   m::Parameter(1)))));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
