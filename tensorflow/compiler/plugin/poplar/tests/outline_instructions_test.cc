/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/passes/outline_instructions.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/human_readable_json.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using OutlineInstructionsTest = HloTestBase;

bool BEConfigEqual(const PoplarBackendConfig& a, const PoplarBackendConfig& b) {
  PoplarBackendConfig aa(a);
  PoplarBackendConfig bb(b);
  aa.set_is_inplace(false);
  bb.set_is_inplace(false);
  // Reset the MLType if only one of the operations doesn't have an MLType
  // associated with it.
  if (aa.ml_type() != bb.ml_type() &&
      (aa.ml_type() == MLType::NONE || bb.ml_type() == MLType::NONE)) {
    aa.set_ml_type(MLType::NONE);
    bb.set_ml_type(MLType::NONE);
  }
  return protobuf_util::ProtobufEquals(aa, bb);
}

TEST_F(OutlineInstructionsTest, MoveInstructionIntoComputation) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  %input = f32[1,1,1,4]{3,2,1,0} parameter(0)
  %weights = f32[1,1,4,16]{3,2,1,0} parameter(1)
  ROOT %convolution = f32[1,1,1,16]{3,2,1,0} convolution(f32[1,1,1,4]{3,2,1,0} %input, f32[1,1,4,16]{3,2,1,0} weights), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="cnv5_1"}, backend_config="{\"is_inplace\":false,\"hash_of_custom_attributes\":\"0\",\"stochastic_rounding\":\"THREESTATE_OFF\",\"ml_type\":\"INFERENCE_FWD\",\"partials_type\":\"PRIMITIVE_TYPE_INVALID\",\"convolution_options\":[],\"matmul_options\":[],\"slice_options\":[],\"tileset\":\"TILESET_COMPUTE_TILES\",\"is_replica_identical\":false,\"stochastic_rounding_method\":\"StochasticRoundingMethod_Undefined\"}"
}
)";
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(root->opcode() == HloOpcode::kConvolution);

  auto original_conv_config_or_status =
      root->backend_config<PoplarBackendConfig>();
  EXPECT_TRUE(original_conv_config_or_status.ok());
  auto original_conv_config = original_conv_config_or_status.ValueOrDie();

  const auto* p0 = root->operand(0);
  const auto* p1 = root->operand(1);

  OutlineInstructions oi;
  EXPECT_TRUE(oi.Run(module).ValueOrDie());

  root = module->entry_computation()->root_instruction();
  auto* comp = root->to_apply();
  auto* conv = comp->root_instruction();

  auto call_config_or_status = root->backend_config<PoplarBackendConfig>();
  EXPECT_TRUE(call_config_or_status.ok());
  auto call_config = call_config_or_status.ValueOrDie();

  auto new_conv_config_or_status = conv->backend_config<PoplarBackendConfig>();
  EXPECT_TRUE(new_conv_config_or_status.ok());
  auto new_conv_config = new_conv_config_or_status.ValueOrDie();

  EXPECT_TRUE(Match(root, m::Op()
                              .WithOpcode(HloOpcode::kCall)
                              .WithOperand(0, m::Op().Is(p0))
                              .WithOperand(1, m::Op().Is(p1))));

  EXPECT_TRUE(Match(conv, m::Convolution(m::Parameter(0), m::Parameter(1))));

  EXPECT_TRUE(absl::StrContains(comp->name(), "instruction_cache"));

  EXPECT_TRUE(call_config.has_call_config());
  EXPECT_TRUE(call_config.call_config().type() ==
              PoplarBackendConfig::CallConfig::Function);
  EXPECT_TRUE(call_config.call_config().has_function_config());
  EXPECT_TRUE(call_config.call_config().function_config().keep_input_layouts());

  EXPECT_TRUE(BEConfigEqual(original_conv_config, new_conv_config));
  original_conv_config.mutable_call_config()->set_type(
      PoplarBackendConfig::CallConfig::Function);
  original_conv_config.mutable_call_config()
      ->mutable_function_config()
      ->set_keep_input_layouts(true);
  EXPECT_TRUE(BEConfigEqual(original_conv_config, call_config));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
