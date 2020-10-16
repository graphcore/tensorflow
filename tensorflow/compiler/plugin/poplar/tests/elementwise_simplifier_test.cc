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

#include "tensorflow/compiler/plugin/poplar/driver/passes/elementwise_simplifier.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using ElementwiseSimplifierTest = HloTestBase;

// A * A => Square(A)
TEST_F(ElementwiseSimplifierTest, ReplaceWithSquare) {
  const std::string hlo = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      ROOT mul = f32[] multiply(p0, p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  HloInstruction* p0 = FindInstruction(m.get(), "p0");

  ASSERT_TRUE(ElementwiseSimplifier().Run(m.get()).ValueOrDie());
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root->operands(), ::testing::ElementsAre(p0));
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::Square, root));
}

// 1 / A => Inverse(A)
TEST_F(ElementwiseSimplifierTest, ReplaceWithInverse) {
  const std::string hlo = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      one = f32[] constant(1)
      ROOT mul = f32[] divide(one, p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  HloInstruction* p0 = FindInstruction(m.get(), "p0");

  ASSERT_TRUE(ElementwiseSimplifier().Run(m.get()).ValueOrDie());
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root->operands(), ::testing::ElementsAre(p0));
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::Inverse, root));
}

// 1 / A => Inverse(A)
TEST_F(ElementwiseSimplifierTest, ReplaceWithInverseBroadcast) {
  const std::string hlo = R"(
    HloModule m
    test {
      p0 = f32[2] parameter(0)
      one = f32[] constant(1)
      b_one = f32[2] broadcast(one), dimensions={}
      ROOT mul = f32[2] divide(b_one, p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  HloInstruction* p0 = FindInstruction(m.get(), "p0");

  ASSERT_TRUE(ElementwiseSimplifier().Run(m.get()).ValueOrDie());
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root->operands(), ::testing::ElementsAre(p0));
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::Inverse, root));
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
