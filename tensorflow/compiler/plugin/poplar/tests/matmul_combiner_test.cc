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
#include "tensorflow/compiler/plugin/poplar/driver/passes/matmul_combiner.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/scatter_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xp = ::xla::poplarplugin;
namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

se::Platform* GetReferencePlatform() {
  auto result = PlatformUtil::GetPlatform("interpreter");
  return result.ValueOrDie();
}

se::Platform* GetTestPlatform() {
  auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
  EXPECT_TRUE(platform.ok());

  auto* p = static_cast<xp::PoplarPlatform*>(platform.ValueOrDie());

  xla::poplarplugin::IpuOptions options;
  options.set_creator_id(IpuOptionsCreator::IPU_UTILS);
  options.set_enable_matmul_combiner(true);

  EXPECT_EQ(p->ConfigurePoplarDevices(options), Status::OK());
  return p;
}

class MatmulCombinerTest : public HloTestBase {
 public:
  MatmulCombinerTest()
      : HloTestBase(GetTestPlatform(), GetReferencePlatform()) {}
};

auto GetNumMatmul = GetNumInstructions<HloDotInstruction>;
auto GetNumSlice = GetNumInstructions<HloSliceInstruction>;
auto GetNumConcatenate = GetNumInstructions<HloConcatenateInstruction>;
auto GetNumReshape = GetNumInstructions<HloReshapeInstruction>;
auto GetNumTranspose = GetNumInstructions<HloTransposeInstruction>;

float ComputeMatMulValue2D(const Literal& lhs, const Literal& rhs,
                           absl::Span<const int64_t> output_index) {
  EXPECT_EQ(output_index.size(), 2);
  float value = 0.0f;
  auto M = output_index[0];
  auto N = output_index[1];
  auto K = lhs.shape().dimensions(1);
  for (int64_t k = 0; k < K; k++) {
    float lhs_value = lhs.Get<float>({M, k});
    float rhs_value = rhs.Get<float>({k, N});
    value += lhs_value * rhs_value;
  }
  return value;
}

float ComputeMatMulValue3D(const Literal& lhs, const Literal& rhs,
                           absl::Span<const int64_t> output_index) {
  EXPECT_EQ(output_index.size(), 3);
  float value = 0.0f;
  auto batch = output_index[0];
  auto M = output_index[1];
  auto N = output_index[2];
  auto K = lhs.shape().dimensions(2);
  for (int64_t k = 0; k < K; k++) {
    float lhs_value = lhs.Get<float>({batch, M, k});
    float rhs_value = rhs.Get<float>({batch, k, N});
    value += lhs_value * rhs_value;
  }
  return value;
}

TEST_F(MatmulCombinerTest, MatmulSharedLHS) {
  std::string hlo_string = R"(
HloModule main

ENTRY main {
  lhs = f32[2,4,3] parameter(0)
  rhs = f32[2,3,5] parameter(1)
  rhs2 = f32[2,3,2] parameter(2)
  matmul1 = f32[2,4,5] dot(lhs, rhs), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
  matmul2 = f32[2,4,2] dot(lhs, rhs2), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
  ROOT t = (f32[2,4,5], f32[2,4,2]) tuple(matmul1, matmul2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();
  auto* module = module0.get();

  CompilerAnnotations annotations(module);

  ScatterSimplifier sc;
  EXPECT_FALSE(sc.Run(module).ValueOrDie());

  EXPECT_EQ(GetNumMatmul(module->entry_computation()), 2);
  EXPECT_EQ(GetNumSlice(module->entry_computation()), 0);
  EXPECT_EQ(GetNumConcatenate(module->entry_computation()), 0);
  EXPECT_EQ(GetNumReshape(module->entry_computation()), 0);
  EXPECT_EQ(GetNumTranspose(module->entry_computation()), 0);

  HloPassFix<MatmulCombiner> combiner(annotations);
  EXPECT_TRUE(combiner.Run(module).ValueOrDie());

  EXPECT_EQ(GetNumMatmul(module->entry_computation()), 1);
  EXPECT_EQ(GetNumSlice(module->entry_computation()), 2);
  EXPECT_EQ(GetNumConcatenate(module->entry_computation()), 1);
  EXPECT_EQ(GetNumReshape(module->entry_computation()), 5);
  EXPECT_EQ(GetNumTranspose(module->entry_computation()), 3);

  auto root = module->entry_computation()->root_instruction();

  EXPECT_TRUE(
      Match(root->operand(0),
            m::Reshape(m::Slice(m::Dot(m::Reshape(m::Transpose(m::Parameter())),
                                       m::Concatenate())))));
  EXPECT_TRUE(
      Match(root->operand(1),
            m::Reshape(m::Slice(m::Dot(m::Reshape(m::Transpose(m::Parameter())),
                                       m::Concatenate())))));
  // Check they both point at the same dot product instruction.
  EXPECT_EQ(root->operand(0)->operand(0)->operand(0),
            root->operand(1)->operand(0)->operand(0));

  // Check the expected value.
  auto lhs_shape = ShapeUtil::MakeShape(F32, {2, 4, 3});
  Literal lhs(lhs_shape);
  lhs.Populate<float>([](const xla::DimensionVector& index) {
    return index[0] + index[1] * 10.0f + index[2] * 100.0f;
  });

  auto rhs_shape = ShapeUtil::MakeShape(F32, {2, 3, 5});
  Literal rhs(rhs_shape);
  rhs.Populate<float>([](const xla::DimensionVector& index) {
    return index[0] * 0.1f + index[1] + index[2] * 0.01f;
  });
  auto rhs_shape2 = ShapeUtil::MakeShape(F32, {2, 3, 2});
  Literal rhs2(rhs_shape2);
  rhs2.Populate<float>([](const xla::DimensionVector& index) {
    return (index[0] + index[1] * 2 + index[2] * 6);
  });

  Literal result =
      Execute(
          std::move(
              ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie()),
          {&lhs, &rhs, &rhs2})
          .ValueOrDie();
  ASSERT_TRUE(result.shape().IsTuple());

  const Shape& slice1 = ShapeUtil::GetSubshape(result.shape(), {0});
  ShapeUtil::ForEachIndex(slice1, [&](absl::Span<const int64_t> output_index) {
    float expected_value = ComputeMatMulValue3D(lhs, rhs, output_index);
    float value = result.Get<float>(output_index, {0});
    EXPECT_FLOAT_EQ(value, expected_value);
    return true;
  });
  const Shape& slice2 = ShapeUtil::GetSubshape(result.shape(), {1});
  ShapeUtil::ForEachIndex(slice2, [&](absl::Span<const int64_t> output_index) {
    float expected_value = ComputeMatMulValue3D(lhs, rhs2, output_index);
    float value = result.Get<float>(output_index, {1});
    EXPECT_FLOAT_EQ(value, expected_value);
    return true;
  });
}

TEST_F(MatmulCombinerTest, MatmulSharedRHS) {
  std::string hlo_string = R"(
HloModule main

ENTRY main {
  lhs = f32[2,4,3] parameter(0)
  lhs2 = f32[2,6,3] parameter(1)
  rhs = f32[2,3,5] parameter(2)
  matmul1 = f32[2,4,5] dot(lhs, rhs), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
  matmul2 = f32[2,6,5] dot(lhs2, rhs), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
  ROOT t = (f32[2,4,5], f32[2,6,5]) tuple(matmul1, matmul2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();
  auto* module = module0.get();

  CompilerAnnotations annotations(module);

  ScatterSimplifier sc;
  EXPECT_FALSE(sc.Run(module).ValueOrDie());

  EXPECT_EQ(GetNumMatmul(module->entry_computation()), 2);
  EXPECT_EQ(GetNumSlice(module->entry_computation()), 0);
  EXPECT_EQ(GetNumConcatenate(module->entry_computation()), 0);
  EXPECT_EQ(GetNumReshape(module->entry_computation()), 0);
  EXPECT_EQ(GetNumTranspose(module->entry_computation()), 0);

  HloPassFix<MatmulCombiner> combiner(annotations);
  EXPECT_TRUE(combiner.Run(module).ValueOrDie());

  EXPECT_EQ(GetNumMatmul(module->entry_computation()), 1);
  EXPECT_EQ(GetNumSlice(module->entry_computation()), 2);
  EXPECT_EQ(GetNumConcatenate(module->entry_computation()), 1);
  EXPECT_EQ(GetNumReshape(module->entry_computation()), 5);
  EXPECT_EQ(GetNumTranspose(module->entry_computation()), 3);

  auto root = module->entry_computation()->root_instruction();

  EXPECT_TRUE(
      Match(root->operand(0),
            m::Reshape(m::Slice(m::Dot(
                m::Concatenate(), m::Reshape(m::Transpose(m::Parameter())))))));
  EXPECT_TRUE(
      Match(root->operand(1),
            m::Reshape(m::Slice(m::Dot(
                m::Concatenate(), m::Reshape(m::Transpose(m::Parameter())))))));

  // Check they both point at the same dot product instruction.
  EXPECT_EQ(root->operand(0)->operand(0)->operand(0),
            root->operand(1)->operand(0)->operand(0));

  // Check the expected value.
  auto lhs_shape = ShapeUtil::MakeShape(F32, {2, 4, 3});
  Literal lhs(lhs_shape);
  lhs.Populate<float>([](const xla::DimensionVector& index) {
    return index[0] + index[1] * 10.0f + index[2] * 100.0f;
  });

  auto lhs_shape2 = ShapeUtil::MakeShape(F32, {2, 6, 3});
  Literal lhs2(lhs_shape2);
  lhs2.Populate<float>([](const xla::DimensionVector& index) {
    return index[0] * 0.1f + index[1] + index[2] * 0.01f;
  });
  auto rhs_shape = ShapeUtil::MakeShape(F32, {2, 3, 5});
  Literal rhs(rhs_shape);
  rhs.Populate<float>([](const xla::DimensionVector& index) {
    return (index[0] + index[1] * 2 + index[2] * 6);
  });

  Literal result =
      Execute(
          std::move(
              ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie()),
          {&lhs, &lhs2, &rhs})
          .ValueOrDie();
  ASSERT_TRUE(result.shape().IsTuple());

  const Shape& slice1 = ShapeUtil::GetSubshape(result.shape(), {0});
  ShapeUtil::ForEachIndex(slice1, [&](absl::Span<const int64_t> output_index) {
    float expected_value = ComputeMatMulValue3D(lhs, rhs, output_index);
    float value = result.Get<float>(output_index, {0});
    EXPECT_FLOAT_EQ(value, expected_value);
    return true;
  });
  const Shape& slice2 = ShapeUtil::GetSubshape(result.shape(), {1});
  ShapeUtil::ForEachIndex(slice2, [&](absl::Span<const int64_t> output_index) {
    float expected_value = ComputeMatMulValue3D(lhs2, rhs, output_index);
    float value = result.Get<float>(output_index, {1});
    EXPECT_FLOAT_EQ(value, expected_value);
    return true;
  });
}

TEST_F(MatmulCombinerTest, Matmul2D) {
  std::string hlo_string = R"(
HloModule main

ENTRY main {
  lhs = f32[4,3] parameter(0)
  lhs2 = f32[6,3] parameter(1)
  rhs = f32[3,5] parameter(2)
  matmul1 = f32[4,5] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  matmul2 = f32[6,5] dot(lhs2, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT t = (f32[4,5], f32[6,5]) tuple(matmul1, matmul2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();
  auto* module = module0.get();

  CompilerAnnotations annotations(module);

  ScatterSimplifier sc;
  EXPECT_FALSE(sc.Run(module).ValueOrDie());

  EXPECT_EQ(GetNumMatmul(module->entry_computation()), 2);
  EXPECT_EQ(GetNumSlice(module->entry_computation()), 0);
  EXPECT_EQ(GetNumConcatenate(module->entry_computation()), 0);
  EXPECT_EQ(GetNumReshape(module->entry_computation()), 0);
  EXPECT_EQ(GetNumTranspose(module->entry_computation()), 0);

  HloPassFix<MatmulCombiner> combiner(annotations);
  EXPECT_TRUE(combiner.Run(module).ValueOrDie());

  EXPECT_EQ(GetNumMatmul(module->entry_computation()), 1);
  EXPECT_EQ(GetNumSlice(module->entry_computation()), 2);
  EXPECT_EQ(GetNumConcatenate(module->entry_computation()), 1);
  EXPECT_EQ(GetNumReshape(module->entry_computation()), 5);
  EXPECT_EQ(GetNumTranspose(module->entry_computation()), 3);

  auto root = module->entry_computation()->root_instruction();

  EXPECT_TRUE(
      Match(root->operand(0),
            m::Reshape(m::Slice(m::Dot(
                m::Concatenate(), m::Reshape(m::Transpose(m::Parameter())))))));
  EXPECT_TRUE(
      Match(root->operand(1),
            m::Reshape(m::Slice(m::Dot(
                m::Concatenate(), m::Reshape(m::Transpose(m::Parameter())))))));

  // Check they both point at the same dot product instruction.
  EXPECT_EQ(root->operand(0)->operand(0)->operand(0),
            root->operand(1)->operand(0)->operand(0));

  // Check the expected value.
  auto lhs_shape = ShapeUtil::MakeShape(F32, {4, 3});
  Literal lhs(lhs_shape);
  lhs.Populate<float>([](const xla::DimensionVector& index) {
    return index[0] + index[1] * 10.0f;
  });

  auto lhs_shape2 = ShapeUtil::MakeShape(F32, {6, 3});
  Literal lhs2(lhs_shape2);
  lhs2.Populate<float>([](const xla::DimensionVector& index) {
    return index[0] * 0.1f + index[1];
  });
  auto rhs_shape = ShapeUtil::MakeShape(F32, {3, 5});
  Literal rhs(rhs_shape);
  rhs.Populate<float>([](const xla::DimensionVector& index) {
    return (index[0] + index[1] * 3);
  });

  Literal result =
      Execute(
          std::move(
              ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie()),
          {&lhs, &lhs2, &rhs})
          .ValueOrDie();
  ASSERT_TRUE(result.shape().IsTuple());

  const Shape& slice1 = ShapeUtil::GetSubshape(result.shape(), {0});
  ShapeUtil::ForEachIndex(slice1, [&](absl::Span<const int64_t> output_index) {
    float expected_value = ComputeMatMulValue2D(lhs, rhs, output_index);
    float value = result.Get<float>(output_index, {0});
    EXPECT_FLOAT_EQ(value, expected_value);
    return true;
  });
  const Shape& slice2 = ShapeUtil::GetSubshape(result.shape(), {1});
  ShapeUtil::ForEachIndex(slice2, [&](absl::Span<const int64_t> output_index) {
    float expected_value = ComputeMatMulValue2D(lhs2, rhs, output_index);
    float value = result.Get<float>(output_index, {1});
    EXPECT_FLOAT_EQ(value, expected_value);
    return true;
  });
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
