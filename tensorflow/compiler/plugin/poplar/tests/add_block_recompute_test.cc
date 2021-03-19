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
#include "tensorflow/compiler/plugin/poplar/driver/passes/add_block_recompute.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
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

using AddBlockRecomputeTest = HloTestBase;

/**
 * Check that the parameters are mark as not being recomputable.
 */
TEST_F(AddBlockRecomputeTest, BlockParameters) {
  std::string hlo_string = R"(
HloModule main

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  c = f32[] constant(2)
  d = f32[] add(a, b)
  ROOT e = f32[] add(d, c)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();

  HloPassPipeline pipeline("test");
  pipeline.AddPass<AddBlockRecompute>();

  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  EXPECT_EQ(module->entry_computation()->instruction_count(), 8);

  auto root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(root, m::Add(m::Add(m::CustomCall(m::Parameter()),
                                        m::CustomCall(m::Parameter())),
                                 m::CustomCall(m::Constant()))));
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
