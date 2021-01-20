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

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using HloPoplarBufferTest = HloTestBase;

TEST_F(HloPoplarBufferTest, DifferentComputationOrder) {
  std::string hlo = R"(
 HloModule top

 ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  arg1 = f32[2] parameter(1)
  tuple = (f32[2], f32[2]) tuple(arg0, arg1)
  c = f32[4] concatenate(arg0, arg1), dimensions={0}
  ROOT tuple2 = ((f32[2], f32[2]), f32[4]) tuple(tuple, c)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  HloInstruction* arg0 = FindInstruction(m.get(), "arg0");
  HloInstruction* arg1 = FindInstruction(m.get(), "arg1");
  HloInstruction* tuple = FindInstruction(m.get(), "tuple");
  HloInstruction* c = FindInstruction(m.get(), "c");
  HloInstruction* tuple2 = FindInstruction(m.get(), "tuple2");

  auto tuple_descriptions = UseDescriptionsForwardsBuffers(
      tuple, 2, BufferUseKind::USE_ALIAS_READ_WRITE);
  EXPECT_THAT(tuple_descriptions.size(), 2);
  // Aliasing for combined shape.
  EXPECT_EQ(tuple_descriptions[0],
            HloPoplarUseDescription(0, ShapeIndex{}, ShapeIndex{0},
                                    BufferUseKind::USE_ALIAS_READ_WRITE));
  EXPECT_EQ(tuple_descriptions[1],
            HloPoplarUseDescription(1, ShapeIndex{}, ShapeIndex{1},
                                    BufferUseKind::USE_ALIAS_READ_WRITE));

  auto c_descriptions = UseDescriptionsSimpleNoTupleAliasing(
      c, 2, BufferUseKind::USE_ALIAS_READ_ONLY);
  EXPECT_THAT(tuple_descriptions.size(), 2);
  // Both alias the output buffer.
  EXPECT_EQ(c_descriptions[0],
            HloPoplarUseDescription(0, ShapeIndex{}, ShapeIndex{},
                                    BufferUseKind::USE_ALIAS_READ_ONLY));
  EXPECT_EQ(c_descriptions[1],
            HloPoplarUseDescription(1, ShapeIndex{}, ShapeIndex{},
                                    BufferUseKind::USE_ALIAS_READ_ONLY));

  // Test filling out missing shapes.
  HloPoplarUseDescriptions tuple2_aliases{HloPoplarUseDescription(
      0, ShapeIndex{1}, ShapeIndex{0, 1}, BufferUseKind::USE_ALIAS_READ_ONLY)};
  auto tuple2_buffers = BufferDescriptionsAllocatesAllUnaliasedBuffers(
      tuple2, tuple2_aliases, BufferLocality::kRemoteMemory);
  EXPECT_THAT(tuple2_buffers.size(), 2);
  EXPECT_EQ(tuple2_buffers[0],
            HloPoplarBufferDescription(ShapeIndex{0, 0},
                                       BufferLocality::kRemoteMemory));
  EXPECT_EQ(
      tuple2_buffers[1],
      HloPoplarBufferDescription(ShapeIndex{1}, BufferLocality::kRemoteMemory));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
