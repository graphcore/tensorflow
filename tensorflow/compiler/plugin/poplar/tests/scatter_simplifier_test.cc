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

#include "tensorflow/compiler/plugin/poplar/driver/passes/scatter_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using ScatterSimplifierTest = HloTestBase;

int64 GetNumMultiUpdates(const HloComputation* comp) {
  return absl::c_count_if(comp->instructions(),
                          IsPoplarInstruction(PoplarOp::MultiUpdate));
}

int64 GetNumMultiUpdateAdds(const HloComputation* comp) {
  return absl::c_count_if(comp->instructions(),
                          IsPoplarInstruction(PoplarOp::MultiUpdateAdd));
}

TEST_F(ScatterSimplifierTest, TestMultiUpdateAdds) {
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
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(GetNumMultiUpdateAdds(module->entry_computation()), 2);
}

TEST_F(ScatterSimplifierTest, TestMultiUpdates) {
  std::string hlo_string = R"(
HloModule top
scatter-combiner {
  p0 = f32[] parameter(0)
  ROOT p1 = f32[] parameter(1)
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
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(GetNumMultiUpdates(module->entry_computation()), 2);
  auto mu0 = Cast<HloMultiUpdateInstruction>(root->operand(0));
}

TEST_F(ScatterSimplifierTest, TestNotValid) {
  std::string hlo_string = R"(
HloModule top
scatter-combiner {
  p0 = f32[] parameter(0)
  ROOT p1 = f32[] parameter(1)
}

main {
  arg0 = s32[15] parameter(0)
  arg1 = f32[15,15] parameter(1)
	zero = f32[] constant(0)
	big_zero = f32[2000,10] broadcast(zero), dimensions={}
  ROOT s1 = f32[2000,10] scatter(big_zero, arg0, arg1), update_window_dims={0}, inserted_window_dims={1}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=scatter-combiner
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();
  auto* module = module0.get();

  CompilerAnnotations annotations(module);
  ScatterSimplifier sc;
  EXPECT_FALSE(sc.Run(module).ValueOrDie());
  EXPECT_EQ(GetNumMultiUpdates(module->entry_computation()), 0);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
