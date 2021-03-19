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
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/add_block_recompute.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/apply_recompute_suggestion.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/matmul_combiner.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/remove_blocked_recompute_suggestions.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/remove_recompute_suggestions.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/scatter_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/suggest_recompute.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using RemoveBlockedRecomputeTest = HloTestBase;

/**
 * Check that recompute(block(x)) is the same as x
 */
TEST_F(RemoveBlockedRecomputeTest, SingleIteration) {
  std::string hlo_string = R"(
HloModule main

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  c = f32[] parameter(2)
  d = f32[] add(a, b)
  d_b = f32[] custom-call(f32[] d), custom_call_target="BlockRecompute", backend_config="{}"
  d_r = f32[] custom-call(f32[] d_b), custom_call_target="SuggestRecompute", backend_config="{}"
  e = f32[] add(d_r, c)
  f = f32[] add(e, c)
  ROOT g = f32[] add(f, d_r)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();

  HloPassPipeline pipeline("test");
  pipeline.AddPass<CustomOpReplacer>();
  pipeline.AddPass<ApplyRecomputeSuggestion>();
  pipeline.AddPass<RemoveBlockedRecomputeSuggestions>();
  pipeline.AddPass<RemoveRecomputeSuggestions>();

  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  EXPECT_EQ(module->entry_computation()->instruction_count(), 7);

  auto a = m::Parameter();
  auto b = m::Parameter();
  auto c = m::Parameter();
  auto d = m::Add(a, b);  // Not recomputed because of blocker
  auto e = m::Add(d, c);
  auto f = m::Add(e, c);
  auto h = m::Add(f, d);  // ROOT

  auto root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(root, h));
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
