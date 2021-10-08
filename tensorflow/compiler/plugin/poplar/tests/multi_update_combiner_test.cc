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
#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_update_combiner.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/scatter_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using MultiUpdateCombinerTest = HloTestBase;

int64 GetNumMultiUpdateAdds(const HloComputation* comp) {
  return absl::c_count_if(comp->instructions(),
                          IsPoplarInstruction(PoplarOp::MultiUpdateAdd));
}

TEST_F(MultiUpdateCombinerTest, TestTwoLookups) {
  std::string hlo_string = R"(
HloModule top
scatter-combiner {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

main {
  arg0 = s32[15] parameter(0)
  arg1 = f32[15,10] parameter(1)
  arg2 = s32[75] parameter(2)
  arg3 = f32[75,10] parameter(3)
	zero = f32[] constant(0)
	big_zero = f32[2000,10] broadcast(zero), dimensions={}
  s1 = f32[2000,10] scatter(big_zero, arg0, arg1), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=scatter-combiner
  s2 = f32[2000,10] scatter(big_zero, arg2, arg3), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=scatter-combiner
  ROOT add = f32[2000,10] add(s2, s1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();
  auto* module = module0.get();

  CompilerAnnotations annotations(module);

  ScatterSimplifier sc;
  EXPECT_TRUE(sc.Run(module).ValueOrDie());
  EXPECT_TRUE(PoplarAlgebraicSimplifier().Run(module).ValueOrDie());
  EXPECT_EQ(GetNumMultiUpdateAdds(module->entry_computation()), 2);
  HloCSE cse(false);
  EXPECT_TRUE(cse.Run(module).ValueOrDie());
  MultiUpdateCombiner mu_combiner(annotations);
  EXPECT_TRUE(mu_combiner.Run(module).ValueOrDie());

  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(GetNumMultiUpdateAdds(module->entry_computation()), 1);

  EXPECT_TRUE(Match(root, m::CustomCall(m::Broadcast(), m::Concatenate(),
                                        m::Concatenate(), m::Constant())));

  // Check the expected value.
  std::vector<int32> arg0_vals(15);
  std::iota(arg0_vals.begin(), arg0_vals.end(), 100);
  Literal arg0 = LiteralUtil::CreateR1<int32>(arg0_vals);

  std::vector<int32> arg2_vals(75);
  std::iota(arg2_vals.begin(), arg2_vals.end(), 110);
  Literal arg2 = LiteralUtil::CreateR1<int32>(arg2_vals);

  auto arg1_shape = ShapeUtil::MakeShape(F32, {15, 10});
  Literal arg1 = DataInitializer::GetDataInitializer("2")
                     ->GetData(arg1_shape)
                     .ValueOrDie();

  auto arg3_shape = ShapeUtil::MakeShape(F32, {75, 10});
  Literal arg3 = DataInitializer::GetDataInitializer("5")
                     ->GetData(arg3_shape)
                     .ValueOrDie();
  Literal result =
      Execute(
          std::move(
              ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie()),
          {&arg0, &arg1, &arg2, &arg3})
          .ValueOrDie();

  ShapeUtil::ForEachIndex(result.shape(),
                          [&](absl::Span<const int64> output_index) {
                            EXPECT_EQ(output_index.size(), 2);
                            float value = result.Get<float>(output_index);
                            int64 y = output_index[0];
                            if (y < 100) {
                              EXPECT_EQ(value, 0);
                            } else if (100 <= y && y < 110) {
                              EXPECT_EQ(value, 2);
                            } else if (110 <= y && y < 115) {
                              EXPECT_EQ(value, 7);
                            } else if (115 <= y && y < 185) {
                              EXPECT_EQ(value, 5);
                            } else {
                              EXPECT_EQ(value, 0);
                            }
                            return true;
                          });
}

TEST_F(MultiUpdateCombinerTest, TestThreeLookups) {
  std::string hlo_string = R"(
HloModule top
scatter-combiner {
  p0 = f16[] parameter(0)
  p1 = f16[] parameter(1)
  ROOT add = f16[] add(p0, p1)
}

main {
  arg0 = s32[15] parameter(0)
  arg1 = f16[15,1200] parameter(1)
  arg2 = s32[75] parameter(2)
  arg3 = f16[75,1200] parameter(3)
  arg4 = s32[75] parameter(4)
  arg5 = f16[75,1200] parameter(5)
	zero = f16[] constant(0)
	big_zero = f16[2000,1200] broadcast(zero), dimensions={}
  s1 = f16[2000,1200] scatter(big_zero, arg0, arg1), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=scatter-combiner
  s2 = f16[2000,1200] scatter(big_zero, arg2, arg3), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=scatter-combiner
  s3 = f16[2000,1200] scatter(big_zero, arg4, arg5), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=scatter-combiner
  add = f16[2000,1200] add(s2, s1)
  ROOT add2 = f16[2000,1200] add(add, s3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  ScatterSimplifier sc;
  EXPECT_TRUE(sc.Run(module).ValueOrDie());
  EXPECT_TRUE(PoplarAlgebraicSimplifier().Run(module).ValueOrDie());
  EXPECT_EQ(GetNumMultiUpdateAdds(module->entry_computation()), 3);
  MultiUpdateCombiner mu_combiner(annotations);
  int64 execution_count = -1;
  bool changed = false;
  do {
    HloCSE cse(false);
    cse.Run(module).ValueOrDie();
    changed = mu_combiner.Run(module).ValueOrDie();
    execution_count++;
  } while (changed);
  EXPECT_TRUE(execution_count);

  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(GetNumMultiUpdateAdds(module->entry_computation()), 1);

  EXPECT_TRUE(Match(root, m::CustomCall(m::Broadcast(), m::Concatenate(),
                                        m::Concatenate(), m::Constant())));
}

TEST_F(MultiUpdateCombinerTest, TestThreeLookupsDifferentCompNames) {
  std::string hlo_string = R"(
HloModule top
scatter-combiner1 {
  p0 = f16[] parameter(0)
  p1 = f16[] parameter(1)
  ROOT add = f16[] add(p0, p1)
}
scatter-combiner2 {
  p0 = f16[] parameter(0)
  p1 = f16[] parameter(1)
  ROOT add = f16[] add(p0, p1)
}
scatter-combiner3 {
  p0 = f16[] parameter(0)
  p1 = f16[] parameter(1)
  ROOT add = f16[] add(p0, p1)
}

main {
  arg0 = s32[15] parameter(0)
  arg1 = f16[15,1200] parameter(1)
  arg2 = s32[75] parameter(2)
  arg3 = f16[75,1200] parameter(3)
  arg4 = s32[75] parameter(4)
  arg5 = f16[75,1200] parameter(5)
	zero = f16[] constant(0)
	big_zero = f16[2000,1200] broadcast(zero), dimensions={}
  s1 = f16[2000,1200] scatter(big_zero, arg0, arg1), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=scatter-combiner1
  s2 = f16[2000,1200] scatter(big_zero, arg2, arg3), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=scatter-combiner2
  s3 = f16[2000,1200] scatter(big_zero, arg4, arg5), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=scatter-combiner3
  add = f16[2000,1200] add(s2, s1)
  ROOT add2 = f16[2000,1200] add(add, s3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  ScatterSimplifier sc;
  EXPECT_TRUE(sc.Run(module).ValueOrDie());
  EXPECT_TRUE(PoplarAlgebraicSimplifier().Run(module).ValueOrDie());
  EXPECT_EQ(GetNumMultiUpdateAdds(module->entry_computation()), 3);
  MultiUpdateCombiner mu_combiner(annotations);
  int64 execution_count = -1;
  bool changed = false;
  do {
    HloCSE cse(false);
    cse.Run(module).ValueOrDie();
    changed = mu_combiner.Run(module).ValueOrDie();
    execution_count++;
  } while (changed);
  EXPECT_TRUE(execution_count);

  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(GetNumMultiUpdateAdds(module->entry_computation()), 1);

  EXPECT_TRUE(Match(root, m::CustomCall(m::Broadcast(), m::Concatenate(),
                                        m::Concatenate(), m::Constant())));

  EXPECT_THAT(root->operand(0)->opcode(), HloOpcode::kBroadcast);
  EXPECT_THAT(root->operand(1)->opcode(), HloOpcode::kConcatenate);
  EXPECT_THAT(root->operand(2)->opcode(), HloOpcode::kConcatenate);
}

TEST_F(MultiUpdateCombinerTest, CompNotAdd) {
  std::string hlo_string = R"(
HloModule top
scatter-combiner1 {
  p0 = f16[] parameter(0)
  p1 = f16[] parameter(1)
  ROOT add = f16[] add(p0, p1)
}
scatter-combiner2 {
  p0 = f16[] parameter(0)
  p1 = f16[] parameter(1)
  ROOT add = f16[] add(p0, p1)
}
scatter-combiner3 {
  p0 = f16[] parameter(0)
  p1 = f16[] parameter(1)
  ROOT add = f16[] subtract(p0, p1)
}

main {
  arg0 = s32[15] parameter(0)
  arg1 = f16[15,1200] parameter(1)
  arg2 = s32[75] parameter(2)
  arg3 = f16[75,1200] parameter(3)
  arg4 = s32[75] parameter(4)
  arg5 = f16[75,1200] parameter(5)
	zero = f16[] constant(0)
	big_zero = f16[2000,1200] broadcast(zero), dimensions={}
  s1 = f16[2000,1200] scatter(big_zero, arg0, arg1), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=scatter-combiner1
  s2 = f16[2000,1200] scatter(big_zero, arg2, arg3), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=scatter-combiner2
  s3 = f16[2000,1200] scatter(big_zero, arg4, arg5), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=scatter-combiner3
  add = f16[2000,1200] add(s2, s1)
  ROOT add2 = f16[2000,1200] add(add, s3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  ScatterSimplifier sc;
  EXPECT_TRUE(sc.Run(module).ValueOrDie());
  EXPECT_TRUE(PoplarAlgebraicSimplifier().Run(module).ValueOrDie());
  EXPECT_EQ(GetNumMultiUpdateAdds(module->entry_computation()), 2);
  MultiUpdateCombiner mu_combiner(annotations);
  int64 execution_count = -1;
  bool changed = false;
  do {
    HloCSE cse(false);
    cse.Run(module).ValueOrDie();
    changed = mu_combiner.Run(module).ValueOrDie();
    execution_count++;
  } while (changed);
  EXPECT_TRUE(execution_count);

  auto root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(root, m::Add(m::CustomCall(m::Broadcast(), m::Concatenate(),
                                               m::Concatenate(), m::Constant()),
                                 m::Scatter())));
}

TEST_F(MultiUpdateCombinerTest, CompatibleMultiUpdateAddIndexTypes) {
  std::string hlo_string = R"(
HloModule top
scatter-combiner1 {
  p0 = f16[] parameter(0)
  p1 = f16[] parameter(1)
  ROOT add = f16[] add(p0, p1)
}
scatter-combiner2 {
  p0 = f16[] parameter(0)
  p1 = f16[] parameter(1)
  ROOT add = f16[] add(p0, p1)
}
scatter-combiner3 {
  p0 = f16[] parameter(0)
  p1 = f16[] parameter(1)
  ROOT add = f16[] add(p0, p1)
}

main {
  arg0 = s32[15] parameter(0)
  arg1 = f16[15,1200] parameter(1)
  arg2 = s64[75] parameter(2)
  arg3 = f16[75,1200] parameter(3)
	zero = f16[] constant(0)
	big_zero = f16[2000,1200] broadcast(zero), dimensions={}
  s1 = f16[2000,1200] scatter(big_zero, arg0, arg1), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=scatter-combiner1
  s2 = f16[2000,1200] scatter(big_zero, arg2, arg3), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=scatter-combiner2
  ROOT add = f16[2000,1200] add(s2, s1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  ScatterSimplifier sc;
  EXPECT_TRUE(sc.Run(module).ValueOrDie());
  EXPECT_TRUE(PoplarAlgebraicSimplifier().Run(module).ValueOrDie());
  EXPECT_EQ(GetNumMultiUpdateAdds(module->entry_computation()), 2);
  HloCSE cse(false);
  cse.Run(module).ValueOrDie();
  MultiUpdateCombiner mu_combiner(annotations);
  bool changed = mu_combiner.Run(module).ValueOrDie();
  EXPECT_TRUE(changed);
  EXPECT_EQ(GetNumMultiUpdateAdds(module->entry_computation()), 1);
}

TEST_F(MultiUpdateCombinerTest, IncompatibleMultiUpdateAddIndexTypes) {
  std::string hlo_string = R"(
HloModule top
scatter-combiner1 {
  p0 = f16[] parameter(0)
  p1 = f16[] parameter(1)
  ROOT add = f16[] add(p0, p1)
}
scatter-combiner2 {
  p0 = f16[] parameter(0)
  p1 = f16[] parameter(1)
  ROOT add = f16[] add(p0, p1)
}
scatter-combiner3 {
  p0 = f16[] parameter(0)
  p1 = f16[] parameter(1)
  ROOT add = f16[] add(p0, p1)
}

main {
  arg0 = s32[15] parameter(0)
  arg1 = f16[15,1200] parameter(1)
  arg2 = s16[75] parameter(2)
  arg3 = f16[75,1200] parameter(3)
	zero = f16[] constant(0)
	big_zero = f16[2000,1200] broadcast(zero), dimensions={}
  s1 = f16[2000,1200] scatter(big_zero, arg0, arg1), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=scatter-combiner1
  s2 = f16[2000,1200] scatter(big_zero, arg2, arg3), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=scatter-combiner2
  ROOT add = f16[2000,1200] add(s2, s1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  ScatterSimplifier sc;
  EXPECT_TRUE(sc.Run(module).ValueOrDie());
  EXPECT_TRUE(PoplarAlgebraicSimplifier().Run(module).ValueOrDie());
  EXPECT_EQ(GetNumMultiUpdateAdds(module->entry_computation()), 2);
  HloCSE cse(false);
  cse.Run(module).ValueOrDie();
  MultiUpdateCombiner mu_combiner(annotations);
  bool changed = mu_combiner.Run(module).ValueOrDie();
  EXPECT_FALSE(changed);
  EXPECT_EQ(GetNumMultiUpdateAdds(module->entry_computation()), 2);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
