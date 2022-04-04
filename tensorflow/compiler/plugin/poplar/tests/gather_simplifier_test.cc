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

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

int64 GetNumMultiSlice(const HloComputation* comp) {
  return absl::c_count_if(comp->instructions(),
                          IsPoplarInstruction(PoplarOp::MultiSlice));
}

int64 GetNumGather(const HloComputation* comp) {
  return absl::c_count_if(comp->instructions(), [](const HloInstruction* inst) {
    return inst->opcode() == HloOpcode::kGather;
  });
}

template <typename NativeT>
Literal CreateIotaLiteral(absl::Span<const int64> dims) {
  auto shape = ShapeUtil::MakeShape(
      primitive_util::NativeToPrimitiveType<NativeT>(), dims);
  Literal literal(shape);

  int i = 0;
  literal.Populate<NativeT>(
      [&](const xla::DimensionVector& index) { return i++; });
  return literal;
}

class GatherSimplifierTest : public HloTestBase {
 protected:
  void RunAndCompare(const string& hlo_string, Literal& operand,
                     Literal& indices) {
    RunAndCompare(hlo_string, {&operand, &indices});
  }

  void RunAndCompare(const string& hlo_string,
                     absl::Span<Literal* const> args) {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module0,
                            ParseAndReturnVerifiedModule(hlo_string, config));

    EXPECT_TRUE(
        HloTestBase::RunAndCompare(std::move(module0), args, absl::nullopt));
  }

  void AssertSimplified(const string& hlo_string) {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());

    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module0,
                            ParseAndReturnVerifiedModule(hlo_string, config));
    auto* module = module0.get();
    CompilerAnnotations annotations(module);

    EXPECT_TRUE(GatherSimplifier().Run(module).ValueOrDie());
    EXPECT_EQ(GetNumMultiSlice(module->entry_computation()), 1);
    EXPECT_EQ(GetNumGather(module->entry_computation()), 0);
  }

  void AssertNotSimplified(const string& hlo_string) {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());

    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module0,
                            ParseAndReturnVerifiedModule(hlo_string, config));
    auto* module = module0.get();
    CompilerAnnotations annotations(module);

    EXPECT_FALSE(GatherSimplifier().Run(module).ValueOrDie());
    EXPECT_EQ(GetNumMultiSlice(module->entry_computation()), 0);
    EXPECT_EQ(GetNumGather(module->entry_computation()), 1);
  }
};

TEST_F(GatherSimplifierTest, TestGatherWithScalarIndicesAndData) {
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
  AssertNotSimplified(hlo_string);
}

TEST_F(GatherSimplifierTest, TestTrivialGather) {
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
  AssertSimplified(hlo_string);
  Literal operand = LiteralUtil::CreateR2<int32>({{}, {}, {}});
  Literal indices = LiteralUtil::CreateR1<int32>({2, 1});
  RunAndCompare(hlo_string, operand, indices);
}

TEST_F(GatherSimplifierTest, TestTrivialGatherOnConstant) {
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
      operand = s32[3,4] constant({{0,1,2,3},{4,5,6,7},{8,9,10,11}})
      indices = s32[5] parameter(0)
      ROOT gather = s32[5,4] gather(operand, indices),
          offset_dims={1},
          collapsed_slice_dims={0},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={1, 4}
    }
    )";
  AssertSimplified(hlo_string);
  Literal indices = LiteralUtil::CreateR1<int32>({2, 1, 2, 2, 0});
  RunAndCompare(hlo_string, {&indices});
}

TEST_F(GatherSimplifierTest, TestRankOneData) {
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
      operand = s32[3] parameter(0)
      indices = s32[2] parameter(1)
      ROOT gather = s32[2] gather(operand, indices),
          offset_dims={},
          collapsed_slice_dims={0},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={1}
  }
  )";
  AssertSimplified(hlo_string);
  Literal operand = CreateIotaLiteral<int32>({3});
  Literal indices = LiteralUtil::CreateR1<int32>({0, 2});
  RunAndCompare(hlo_string, operand, indices);
}

TEST_F(GatherSimplifierTest, TestGatherWithFloatDtype) {
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
      operand = f16[3,0] parameter(0)
      indices = s32[2] parameter(1)
      ROOT gather = f16[2,0] gather(operand, indices),
          offset_dims={1},
          collapsed_slice_dims={0},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={1, 0}
  }
  )";
  AssertSimplified(hlo_string);
  Literal operand = LiteralUtil::CreateR2<half>({{}, {}, {}});
  Literal indices = LiteralUtil::CreateR1<int32>({1, 0});
  RunAndCompare(hlo_string, operand, indices);
}

TEST_F(GatherSimplifierTest, TestGatherWithDoubleDtype) {
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
      operand = f32[3,0] parameter(0)
      indices = s32[2] parameter(1)
      ROOT gather = f32[2,0] gather(operand, indices),
          offset_dims={1},
          collapsed_slice_dims={0},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={1, 0}
  }
  )";
  AssertSimplified(hlo_string);
  Literal operand = LiteralUtil::CreateR2<float>({{}, {}, {}});
  Literal indices = LiteralUtil::CreateR1<int32>({1, 2});
  RunAndCompare(hlo_string, operand, indices);
}

TEST_F(GatherSimplifierTest, TestMultipleUnslicedDims) {
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
      operand = s32[3,3,2] parameter(0)
      indices = s32[7] parameter(1)
      gather = s32[7,3,2] gather(operand, indices),
          offset_dims={1,2},
          collapsed_slice_dims={0},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={1,3,2}
    }
    )";
  AssertSimplified(hlo_string);
  Literal operand = CreateIotaLiteral<int32>({3, 3, 2});
  Literal indices = LiteralUtil::CreateR1<int32>({2, 2, 1, 0, 1, 1, 0});
  RunAndCompare(hlo_string, operand, indices);
}

TEST_F(GatherSimplifierTest, TestMultidimensionalIndicesShape) {
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
      operand = s32[5, 6] parameter(0)
      indices = s32[2, 3, 4] parameter(1)
      gather = s32[2, 3, 4, 6] gather(operand, indices),
          offset_dims={3},
          collapsed_slice_dims={0},
          start_index_map={0},
          index_vector_dim=3,
          slice_sizes={1, 6}
    }
    )";
  AssertSimplified(hlo_string);
  Literal operand = CreateIotaLiteral<int32>({5, 6});
  Literal indices = LiteralUtil::CreateR3<int32>(
      {{{4, 3, 2, 0}, {3, 2, 1, 2}, {0, 0, 3, 4}},
       {{3, 0, 0, 4}, {1, 0, 3, 2}, {2, 1, 4, 2}}});
  RunAndCompare(hlo_string, operand, indices);
}

TEST_F(GatherSimplifierTest,
       TestMultidimensionalIndicesShapeSplitInOutputShape) {
  // The index shape (2, 8) is split either side of the slice shape (6) in the
  // output shape (2, 6, 8). This is decided by the offset_dims.
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
      operand = s32[5, 6] parameter(0)
      indices = s32[2, 8] parameter(1)
      gather = s32[2, 6, 8] gather(operand, indices),
          offset_dims={1},
          collapsed_slice_dims={0},
          start_index_map={0},
          index_vector_dim=2,
          slice_sizes={1, 6}
    }
    )";
  AssertSimplified(hlo_string);
  Literal operand = CreateIotaLiteral<int32>({5, 6});
  Literal indices = LiteralUtil::CreateR2<int32>(
      {{4, 3, 2, 0, 0, 0, 3, 4}, {3, 0, 0, 4, 2, 1, 4, 2}});
  RunAndCompare(hlo_string, operand, indices);
}

TEST_F(GatherSimplifierTest, TestMultipleSliceDimensions) {
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
      operand = s32[3,5,8] parameter(0)
      indices = s32[5,2] parameter(1)
      ROOT gather = s32[5,8] gather(operand, indices),
          offset_dims={1},
          collapsed_slice_dims={0, 1},
          start_index_map={0, 1},
          index_vector_dim=1,
          slice_sizes={1, 1, 8}
  }
  )";
  AssertSimplified(hlo_string);
  Literal operand = CreateIotaLiteral<int32>({3, 5, 8});
  Literal indices =
      LiteralUtil::CreateR2<int32>({{0, 3}, {2, 4}, {1, 1}, {0, 2}, {2, 0}});
  RunAndCompare(hlo_string, operand, indices);
}

TEST_F(GatherSimplifierTest, TestIndexDimInMiddleOfIndicesShape) {
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
      operand = s32[3,5,8] parameter(0)
      indices = s32[1,2,5] parameter(1)
      ROOT gather = s32[1,8,5] gather(operand, indices),
          offset_dims={1},
          collapsed_slice_dims={0, 1},
          start_index_map={0, 1},
          index_vector_dim=1,
          slice_sizes={1, 1, 8}
  }
  )";
  AssertSimplified(hlo_string);
  Literal operand = CreateIotaLiteral<int32>({3, 5, 8});
  Literal indices =
      LiteralUtil::CreateR3<int32>({{{0, 2, 1, 0, 2}, {3, 4, 1, 2, 0}}});
  RunAndCompare(hlo_string, operand, indices);
}

TEST_F(GatherSimplifierTest, TestSliceOnNonZeroDim) {
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
  AssertSimplified(hlo_string);
  Literal operand = CreateIotaLiteral<int32>({3, 3});
  Literal indices = LiteralUtil::CreateR1<int32>({0, 2});
  RunAndCompare(hlo_string, operand, indices);
}

TEST_F(GatherSimplifierTest, TestSliceNotAllSliceDimsCollapsed) {
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
      operand = s32[3,5,8] parameter(0)
      indices = s32[5,2] parameter(1)
      ROOT gather = s32[5,1,8] gather(operand, indices),
          offset_dims={1, 2},
          collapsed_slice_dims={0},
          start_index_map={0, 1},
          index_vector_dim=1,
          slice_sizes={1, 1, 8}
    }
    )";
  AssertSimplified(hlo_string);
  Literal operand = CreateIotaLiteral<int32>({3, 5, 8});
  Literal indices =
      LiteralUtil::CreateR2<int32>({{0, 3}, {2, 4}, {1, 1}, {0, 2}, {2, 0}});
  RunAndCompare(hlo_string, operand, indices);
}

TEST_F(GatherSimplifierTest, TestOffsetDimsInterleavedWithBatchDims) {
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
      operand = s32[3,5,8] parameter(0)
      indices = s32[2,3,1,2] parameter(1)
      ROOT gather = s32[2,1,3,1,1,8] gather(operand, indices),
          offset_dims={1,3,5},
          collapsed_slice_dims={},
          start_index_map={0, 1},
          index_vector_dim=3,
          slice_sizes={1, 1, 8}
    }
    )";
  AssertSimplified(hlo_string);
  Literal operand = CreateIotaLiteral<int32>({3, 5, 8});
  Literal indices = LiteralUtil::CreateR4<int32>(
      {{{{0, 3}}, {{2, 4}}, {{1, 1}}}, {{{1, 0}}, {{0, 2}}, {{2, 0}}}});
  RunAndCompare(hlo_string, operand, indices);
}

TEST_F(GatherSimplifierTest, TestMultipleNonZeroSliceDims) {
  // Technically this kind of gather with a non-zero axis and multiple slice
  // dims cannot be created. It is not fundementally invalid though, just
  // disabled as it is rare and untested.
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
      operand = s32[8, 5, 3] parameter(0)
      indices = s32[6, 2] parameter(1)
      ROOT gather = s32[8, 6] gather(operand, indices),
          offset_dims={0},
          collapsed_slice_dims={1, 2},
          start_index_map={1, 2},
          index_vector_dim=1,
          slice_sizes={8, 1, 1}
  }
  )";
  AssertSimplified(hlo_string);
  Literal operand = CreateIotaLiteral<int32>({8, 5, 3});
  Literal indices = LiteralUtil::CreateR2<int32>(
      {{0, 1}, {2, 2}, {1, 1}, {0, 2}, {3, 0}, {4, 0}});
  RunAndCompare(hlo_string, operand, indices);
}

TEST_F(GatherSimplifierTest, TestComplexGather) {
  // Multiple slice and non-slice dims, multidimensional indices, and slice dims
  // don't start at 0.
  // Technically this kind of gather with a non-zero axis and multiple slice
  // dims cannot be created. It is not fundementally invalid though, just
  // disabled as it is rare and untested.
  std::string hlo_string = R"(
    HloModule top

    ENTRY main {
      operand = s32[3, 5, 7, 8] parameter(0)
      indices = s32[2, 4, 1, 2] parameter(1)
      ROOT gather = s32[3, 2, 1, 4, 1, 8] gather(operand, indices),
          offset_dims={0, 2, 5},
          collapsed_slice_dims={2},
          start_index_map={1, 2},
          index_vector_dim=3,
          slice_sizes={3, 1, 1, 8}
  }
  )";
  AssertSimplified(hlo_string);
  Literal operand = CreateIotaLiteral<int32>({3, 5, 7, 8});
  Literal indices =
      LiteralUtil::CreateR4<int32>({{{{4, 6}}, {{1, 3}}, {{2, 4}}, {{3, 2}}},
                                    {{{0, 4}}, {{2, 5}}, {{1, 0}}, {{0, 1}}}});
  RunAndCompare(hlo_string, operand, indices);
}

TEST_F(GatherSimplifierTest, TestLargerSliceSize) {
  std::string hlo_string = R"(
  HloModule top

  ENTRY main {
    operand = f32[3,3] parameter(0)
    indices = s32[2,2] parameter(1)
    ROOT gather = f32[2,1,2] gather(operand, indices),
        offset_dims={1,2},
        collapsed_slice_dims={},
        start_index_map={0,1},
        index_vector_dim=1,
        slice_sizes={1,2}
  }
  )";
  // Multi-slice does not currently support non-standard slice sizes.
  AssertNotSimplified(hlo_string);
}

TEST_F(GatherSimplifierTest, TestSmallerSliceSize) {
  std::string hlo_string = R"(
  HloModule top

  ENTRY main {
    operand = f32[3,3] parameter(0)
    indices = s32[2,2] parameter(1)
    ROOT gather = f32[2,2,2] gather(operand, indices),
        offset_dims={2},
        collapsed_slice_dims={0},
        start_index_map={0},
        index_vector_dim=2,
        slice_sizes={1,2}
  }
  )";
  // Multi-slice does not currently support non-standard slice sizes.
  AssertNotSimplified(hlo_string);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
