/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <google/protobuf/util/message_differencer.h>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_early.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/conv_with_reverse.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
namespace {

MATCHER_P(EqualsProto, expected, "") {
  return google::protobuf::util::MessageDifferencer::Equals(arg, expected);
}

using ConvWithReverseEarlyFuseTest = HloTestBase;

const char* convolution_with_reverse_hlo = R"(
HloModule top

ENTRY c1 {
   arg_0 = f16[1,1,2,4] parameter(0)
   arg_1 = f16[3,3,4,4] parameter(1)
   reverse = f16[3,3,4,4] reverse(arg_1), dimensions={0,1}
   ROOT root = f16[1,4,5,4] convolution(arg_0, reverse), window={size=3x3 pad=2_3x2_2 lhs_dilate=2x2}, dim_labels=b01f_01oi->b01f, operand_precision={HIGH, HIGHEST}
}
)";
TEST_F(ConvWithReverseEarlyFuseTest,
       ConvolutionWithReverseGetsFusedIntoCustomCall) {
  auto module = ParseAndReturnVerifiedModule(convolution_with_reverse_hlo,
                                             GetModuleConfigForTest());
  ASSERT_TRUE(module.ok());

  auto* hlo_module = module.ValueOrDie().get();
  CompilerAnnotations annotations(hlo_module);

  FuseOpsEarly pass(annotations);
  ASSERT_TRUE(pass.Run(hlo_module).ValueOrDie());

  auto* comp = hlo_module->entry_computation();
  auto* custom_op = comp->root_instruction();
  ASSERT_TRUE(custom_op);
  ASSERT_EQ(custom_op->opcode(), HloOpcode::kCustomCall);
  ASSERT_EQ("ConvWithReverse", custom_op->custom_call_target());
}

TEST_F(ConvWithReverseEarlyFuseTest,
       CustomInstHasSameAttributesAsOriginalConvolution) {
  auto module = ParseAndReturnVerifiedModule(convolution_with_reverse_hlo,
                                             GetModuleConfigForTest());
  ASSERT_TRUE(module.ok());

  auto* hlo_module = module.ValueOrDie().get();
  CompilerAnnotations annotations(hlo_module);

  auto* comp = hlo_module->entry_computation();

  auto* conv_inst = comp->root_instruction();
  auto original_feature_group_count = conv_inst->feature_group_count();
  auto original_batch_group_count = conv_inst->batch_group_count();
  auto original_window = conv_inst->window();
  auto original_dim_labels = conv_inst->convolution_dimension_numbers();
  auto original_precision_config = conv_inst->precision_config();

  FuseOpsEarly pass(annotations);
  ASSERT_TRUE(pass.Run(hlo_module).ValueOrDie());

  auto* custom_op = DynCast<HloConvWithReverse>(comp->root_instruction());
  ASSERT_TRUE(custom_op);

  ASSERT_EQ(custom_op->feature_group_count(), original_feature_group_count);
  ASSERT_EQ(custom_op->batch_group_count(), original_batch_group_count);
  ASSERT_THAT(custom_op->window(), EqualsProto(original_window));
  ASSERT_THAT(custom_op->convolution_dimension_numbers(),
              EqualsProto(original_dim_labels));
  ASSERT_THAT(custom_op->GetPrecisionConfig(),
              EqualsProto(original_precision_config));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
