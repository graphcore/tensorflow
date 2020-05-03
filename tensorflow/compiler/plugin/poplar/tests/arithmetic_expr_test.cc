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

#include "absl/types/optional.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using ArithmeticExprTest = HloTestBase;

//  i i   i
//  \ /   |
//   a    b
//   |    |
//   c    |
//    \  /
//     d
//     |
TEST_F(ArithmeticExprTest, TestArithmeticExpr) {
  HloComputation::Builder builder = HloComputation::Builder("BuilderHloComp0");

  Shape s1 = ShapeUtil::MakeShape(F32, {2, 2});
  Shape s2 = ShapeUtil::MakeShape(F16, {2, 2});

  auto i1 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, s1, "i1"));
  auto i2 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, s1, "i2"));
  auto i3 =
      builder.AddInstruction(HloInstruction::CreateParameter(2, s1, "i3"));

  auto add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(s1, HloOpcode::kAdd, i1, i2));
  auto cast1 = builder.AddInstruction(HloInstruction::CreateConvert(s2, add1));
  auto cast2 = builder.AddInstruction(HloInstruction::CreateConvert(s2, i3));

  builder.AddInstruction(
      HloInstruction::CreateBinary(s2, HloOpcode::kMultiply, cast1, cast2));

  auto computation = builder.Build();
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(std::move(computation));

  EXPECT_THAT(module->computation_count(), 1);
  EXPECT_THAT(module->entry_computation()->instruction_count(), 7);

  Literal params1 = LiteralUtil::CreateR2<float>({{1.1, 1}, {1, 1}});
  Literal params2 = LiteralUtil::CreateR2<float>({{2, 2}, {2, 2}});
  Literal params3 = LiteralUtil::CreateR2<float>({{3, 3}, {3, 3.01}});

  Literal result =
      Execute(std::move(module), {&params1, &params2, &params3}).ValueOrDie();

  half h00(9.29688f);
  half h01(9.0f);
  half h10(9.0f);
  half h11(9.02344f);

  Literal expected = LiteralUtil::CreateR2<half>({{h00, h01}, {h10, h11}});
  EXPECT_TRUE(
      LiteralTestUtil::NearOrEqual(expected, result, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(ArithmeticExprTest, TestArithmeticExpr2) {
  //   p0   p1
  //   |    |
  //   |   cast
  //   |    |
  //    \  /
  //   divide
  //     |
  HloComputation::Builder builder = HloComputation::Builder("BuilderHloComp0");

  Shape s1 = ShapeUtil::MakeShape(F32, {2, 2});
  Shape s2 = ShapeUtil::MakeShape(S32, {2, 2});

  auto p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, s1, "p0"));
  auto p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, s2, "p1"));

  auto cast = builder.AddInstruction(HloInstruction::CreateConvert(s1, p1));
  auto div = builder.AddInstruction(
      HloInstruction::CreateBinary(s1, HloOpcode::kDivide, p0, cast));
  builder.AddInstruction(HloInstruction::CreateTuple({p0, p1, div}));

  auto computation = builder.Build();
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(std::move(computation));

  Literal param0 = LiteralUtil::CreateR2<float>({{1.1, 1}, {1, 1}});
  Literal param1 = LiteralUtil::CreateR2<int32>({{3, 3}, {3, 3}});
  std::vector<Literal*> inputs = {&param0, &param1};
  EXPECT_TRUE(RunAndCompare(std::move(module), inputs, ErrorSpec{1e-4, 1e-4}));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
