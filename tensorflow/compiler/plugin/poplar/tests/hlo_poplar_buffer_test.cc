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

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer.h"

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
  arg1 = (f32[2], f32[2]) parameter(1)
  gte1.0 = f32[2] get-tuple-element(arg1), index=0
  gte1.1 = f32[2] get-tuple-element(arg1), index=1
  c = f32[4] concatenate(arg0, gte1.0), dimensions={0}
  slice = f32[2] slice(c), slice={[1:3]}
  ROOT add = f32[2] add(gte1.1, slice)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  HloInstruction* arg0 = FindInstruction(m.get(), "arg0");
  HloInstruction* arg1 = FindInstruction(m.get(), "arg1");
  HloInstruction* gte1_0 = FindInstruction(m.get(), "gte1.0");
  HloInstruction* gte1_1 = FindInstruction(m.get(), "gte1.1");
  HloInstruction* c = FindInstruction(m.get(), "c");
  HloInstruction* slice = FindInstruction(m.get(), "slice");
  HloInstruction* add = FindInstruction(m.get(), "add");

  auto shape_2 = ShapeUtil::MakeShape(F32, {2});
  auto shape_4 = ShapeUtil::MakeShape(F32, {4});

  HloPoplarPosition position_arg0{arg0, ShapeIndex{}};
  EXPECT_THAT(position_arg0.shape(), shape_2);

  HloPoplarPosition position_arg1_0{arg1, ShapeIndex{0}};
  EXPECT_THAT(position_arg1_0.shape(), shape_2);

  HloPoplarPosition position_arg1_1{arg1, ShapeIndex{1}};
  EXPECT_THAT(position_arg1_0.shape(), shape_2);

  HloPoplarPosition position_gte1_0{gte1_0, ShapeIndex{}};
  EXPECT_THAT(position_gte1_0.shape(), shape_2);

  HloPoplarPosition position_gte1_1{gte1_1, ShapeIndex{}};
  EXPECT_THAT(position_gte1_1.shape(), shape_2);

  HloPoplarPosition position_c{c, ShapeIndex{}};
  EXPECT_THAT(position_c.shape(), shape_4);

  HloPoplarPosition position_slice{slice, ShapeIndex{}};
  EXPECT_THAT(position_slice.shape(), shape_2);

  HloPoplarPosition position_add{add, ShapeIndex{}};
  EXPECT_THAT(position_add.shape(), shape_2);

  HloPoplarBuffer buffer_arg0{0, position_arg0, BufferLocality::kDeviceMemory};
  HloPoplarBuffer buffer_arg1_0{1, position_arg1_0,
                                BufferLocality::kDeviceMemory};
  HloPoplarBuffer buffer_arg1_1{2, position_arg1_1,
                                BufferLocality::kDeviceMemory};

  InstructionPoplarBufferSet buffers_arg1(arg1->shape());
  buffers_arg1.SetOutputBufferSet(ShapeIndex{0},
                                  HloPoplarBufferSet({&buffer_arg1_0}));
  buffers_arg1.SetOutputBufferSet(ShapeIndex{1},
                                  HloPoplarBufferSet({&buffer_arg1_1}));
  EXPECT_EQ(buffers_arg1.GetOutputBufferSet(ShapeIndex{0}),
            HloPoplarBufferSet({&buffer_arg1_0}));
  EXPECT_EQ(buffers_arg1.GetOutputBufferSet(ShapeIndex{1}),
            HloPoplarBufferSet({&buffer_arg1_1}));

  HloPoplarBufferSet buffer_set_c{{&buffer_arg0}};
  EXPECT_FALSE(buffer_set_c.AddBuffer(&buffer_arg0));
  EXPECT_TRUE(buffer_set_c.AddBuffer(&buffer_arg1_0));
  EXPECT_FALSE(buffer_set_c.AddBuffer(&buffer_arg1_0));
  EXPECT_THAT(buffer_set_c.size(), 2);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
