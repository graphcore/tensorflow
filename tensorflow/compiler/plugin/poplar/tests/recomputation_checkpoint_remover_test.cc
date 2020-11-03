/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/recomputation_checkpoint_remover.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using RecomputationCheckpointRemoverTest = HloTestBase;

TEST_F(RecomputationCheckpointRemoverTest, TestRemove) {
  const std::string hlo = R"(
HloModule top

comp {
  weights0 = f32[2] parameter(0)
  log = f32[2] log(weights0)
  checkpoint = f32[2] custom-call(log), custom_call_target="RecomputationCheckpoint"
  weights1 = f32[2] parameter(1)
  add = f32[2] add(checkpoint, weights1)
  ROOT tuple = (f32[2], f32[2]) tuple(log, add)
}

ENTRY e {
  e.weights0 = f32[2] parameter(0)
  e.weights1 = f32[2] parameter(1)
  checkpoint = f32[2] custom-call(e.weights1), custom_call_target="RecomputationCheckpoint"
  ROOT e.call = (f32[2], f32[2]) call(e.weights0, checkpoint), to_apply=comp
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  auto module = ParseAndReturnVerifiedModule(hlo, config).ValueOrDie();

  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());

  auto get_num_checkpoints = [&module]() {
    int64 num_values = 0;
    for (auto comp : module->computations()) {
      num_values += absl::c_count_if(
          comp->instructions(),
          IsPoplarInstruction(PoplarOp::RecomputationCheckpoint));
    }
    return num_values;
  };

  EXPECT_EQ(get_num_checkpoints(), 2);
  EXPECT_TRUE(RecomputationCheckpointRemover().Run(module.get()).ValueOrDie());
  EXPECT_EQ(get_num_checkpoints(), 0);
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
