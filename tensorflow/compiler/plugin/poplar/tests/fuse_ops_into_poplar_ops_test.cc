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

#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_into_poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/non_linearity.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;

namespace poplarplugin {
namespace {

using FuseOpsIntoPoplarOpsTest = HloTestBase;

TEST_F(FuseOpsIntoPoplarOpsTest, MatchCustomOpRelu) {
  const unsigned int look_through_depth = 0;
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f32[10] parameter(0)
  c = f32[] constant(0)
  zero = f32[10] broadcast(c), dimensions={}
  relu = f32[10] maximum(p0, zero)
  ROOT root = (f32[10]) tuple(relu)
 }
)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* hlo_module = module.ValueOrDie().get();

  CompilerAnnotations annotations(hlo_module);
  FuseOpsIntoPoplarOps pass(annotations);
  EXPECT_TRUE(pass.Run(hlo_module).ValueOrDie());

  auto* comp = hlo_module->entry_computation();
  auto* custom_op = comp->root_instruction()->operand(0);
  EXPECT_EQ("Relu", custom_op->custom_call_target());
}

TEST_F(FuseOpsIntoPoplarOpsTest, MatchCustomOpReluGrad) {
  const unsigned int look_through_depth = 0;
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f32[1,1,2,2] parameter(0)
  p1 = f32[1,1,2,2] parameter(1)
  c0 = f32[] constant(0)
  c1 = f32[] constant(1)
  zero = f32[1,1,2,2] broadcast(c0), dimensions={}
  one = f32[1,1,2,2] broadcast(c1), dimensions={}
  compare = pred[1,1,2,2] compare(p0, zero), direction=GT
  relugrad = f32[1,1,2,2] select(compare, one, zero)
  conv = f32[1,1,2,2] convolution(p1, relugrad), window={size=1x1}, dim_labels=b01f_01io->b01f
  ROOT root = (f32[1,1,2,2]) tuple(conv)
 }
)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* hlo_module = module.ValueOrDie().get();

  CompilerAnnotations annotations(hlo_module);
  FuseOpsIntoPoplarOps pass(annotations);
  EXPECT_TRUE(pass.Run(hlo_module).ValueOrDie());

  auto* comp = hlo_module->entry_computation();
  auto* custom_op = comp->root_instruction()->operand(0)->operand(1);
  EXPECT_EQ("ReluGrad", custom_op->custom_call_target());
}

TEST_F(FuseOpsIntoPoplarOpsTest, MatchScaledInplaceXbyTest) {
  const unsigned int look_through_depth = 0;
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f32[10] parameter(0)
  p1 = f32[10] parameter(1)
  p2 = f32[10] parameter(2)
  c0 = f32[] constant(0.1)
  b0 = f32[10] broadcast(c0), dimensions={}
  c1 = f32[] constant(0.2)
  b1 = f32[10] broadcast(c1), dimensions={}

  divide = f32[10] divide(p0, p1)
  multiply = f32[10] multiply(divide, b0)
  add = f32[10] add(multiply, b1)

  ROOT root = f32[10] add(add, p2)
 }
)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* hlo_module = module.ValueOrDie().get();

  CompilerAnnotations annotations(hlo_module);
  FuseOpsIntoPoplarOps pass(annotations);
  EXPECT_TRUE(pass.Run(hlo_module).ValueOrDie());

  auto* root = hlo_module->entry_computation()->root_instruction();
  CHECK(Match(
      root, m::Add(m::CustomCall(m::Divide(m::Parameter(), m::Parameter()),
                                 m::Parameter(), m::Constant(), m::Constant()),
                   m::Broadcast(m::Constant()))));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
