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

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

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

struct AllBufferSetTest : HloPoplarBufferTest {
  HloPoplarBuffer CreateHloPoplarBuffer(
      HloInstruction* inst,
      BufferLocality locality = BufferLocality::kDeviceMemory) {
    return HloPoplarBuffer(++buffer_id_, {inst, {}}, locality);
  }

  HloPoplarBuffer::Id buffer_id_ = 0;
};

TEST_F(AllBufferSetTest, IsOpCode) {
  std::string hlo = R"(
 HloModule top
  _pop_op_wide_const {
    constant = f32[] constant(0.1)
    ROOT broadcast = f32[3,3,4,12] broadcast(constant), dimensions={}
  }

 ENTRY test {
  non_const1 = f32[] parameter(0)
  const1 = f32[] constant(0)
  const2 = f32[] constant(4)
  ROOT wide_const1 = f32[3,3,4,12] fusion(), kind=kCustom, calls=_pop_op_wide_const
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  HloInstruction* non_const1 = FindInstruction(m.get(), "non_const1");
  HloInstruction* const1 = FindInstruction(m.get(), "const1");
  HloInstruction* const2 = FindInstruction(m.get(), "const2");
  HloInstruction* wide_const1 = FindInstruction(m.get(), "wide_const1");

  // Check that we can detect HloPoplarBufferSets which contain only constants.
  auto non_const1_buffer = CreateHloPoplarBuffer(non_const1);
  auto const1_buffer = CreateHloPoplarBuffer(const1);
  auto const2_buffer = CreateHloPoplarBuffer(const2);
  auto wide_const1_buffer = CreateHloPoplarBuffer(wide_const1);

  HloPoplarBufferSet buffer_set(
      {&const1_buffer, &const2_buffer, &wide_const1_buffer});
  ASSERT_TRUE(AllOfBufferSet(buffer_set, IsAnyConstant));

  buffer_set.AddBuffer(&non_const1_buffer);
  ASSERT_FALSE(AllOfBufferSet(buffer_set, IsAnyConstant));
}

TEST_F(AllBufferSetTest, IsPoplarOp) {
  std::string hlo = R"(
 HloModule top
 ENTRY test {
  param0 = f32[2] parameter(0)
  non_gradient_accu1 = f32[] parameter(1)
  gradient_accu1 = f32[2] custom-call(param0), custom_call_target="GradientAccumulatorCreate", backend_config="{}"
  gradient_accu2 = f32[2] custom-call(param0), custom_call_target="GradientAccumulatorCreate", backend_config="{}"
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  ASSERT_TRUE(CustomOpReplacer().Run(m.get()).ValueOrDie());

  // Check that we can detect HloPoplarBufferSets which contain only gradient
  // accumulator create.
  HloInstruction* non_gradient_accu1 =
      FindInstruction(m.get(), "non_gradient_accu1");
  HloInstruction* gradient_accu1 =
      FindInstruction(m.get(), "gradient-accumulator-create");
  HloInstruction* gradient_accu2 =
      FindInstruction(m.get(), "gradient-accumulator-create.1");

  ASSERT_TRUE(gradient_accu1);
  ASSERT_TRUE(gradient_accu2);

  auto non_gradient_accu1_buffer = CreateHloPoplarBuffer(non_gradient_accu1);
  auto gradient_accu1_buffer = CreateHloPoplarBuffer(gradient_accu1);
  auto gradient_accu2_buffer = CreateHloPoplarBuffer(gradient_accu2);

  HloPoplarBufferSet buffer_set(
      {&gradient_accu1_buffer, &gradient_accu2_buffer});
  ASSERT_TRUE(AllOfBufferSet(
      buffer_set, IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)));

  buffer_set.AddBuffer(&non_gradient_accu1_buffer);
  ASSERT_FALSE(AllOfBufferSet(
      buffer_set, IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)));
}

TEST_F(AllBufferSetTest, IsBufferType) {
  std::string hlo = R"(
 HloModule top
 ENTRY test {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  ROOT param2 = f32[] parameter(2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  HloInstruction* param0 = FindInstruction(m.get(), "param0");
  HloInstruction* param1 = FindInstruction(m.get(), "param1");
  HloInstruction* param2 = FindInstruction(m.get(), "param2");

  // Check that we can detect HloPoplarBufferSets which contain only remote
  // buffers
  auto remote_buffer1 =
      CreateHloPoplarBuffer(param0, BufferLocality::kRemoteMemory);
  auto remote_buffer2 =
      CreateHloPoplarBuffer(param1, BufferLocality::kRemoteMemory);
  auto local_buffer1 =
      CreateHloPoplarBuffer(param2, BufferLocality::kDeviceMemory);

  auto is_remote_pred = [](const HloPoplarBuffer* buffer) {
    return buffer->locality() == BufferLocality::kRemoteMemory;
  };

  HloPoplarBufferSet buffer_set({&remote_buffer1, &remote_buffer2});
  ASSERT_TRUE(AllOfBufferSet(buffer_set, is_remote_pred));

  buffer_set.AddBuffer(&local_buffer1);
  ASSERT_FALSE(AllOfBufferSet(buffer_set, is_remote_pred));
}

struct BufferUseKindPropogationTest : HloPoplarBufferTest {
  HloPoplarBuffer buffer0_ =
      HloPoplarBuffer(0, {nullptr, {}}, BufferLocality::kRemoteMemory);
  HloPoplarBuffer buffer1_ =
      HloPoplarBuffer(1, {nullptr, {}}, BufferLocality::kRemoteMemory);
  HloPoplarBuffer buffer2_ =
      HloPoplarBuffer(2, {nullptr, {}}, BufferLocality::kRemoteMemory);
};

TEST_F(BufferUseKindPropogationTest, BufferSetConstruction) {
  HloPoplarBufferSet read_only_buffer_set({&buffer0_, &buffer1_},
                                          BufferUseKind::USE_ALIAS_READ_ONLY);

  ASSERT_EQ(buffer0_.use_kind(), BufferUseKind::USE_ALIAS_READ_ONLY);
  ASSERT_EQ(buffer1_.use_kind(), BufferUseKind::USE_ALIAS_READ_ONLY);

  HloPoplarBufferSet no_alias_buffer_set({&buffer0_, &buffer1_, &buffer2_},
                                         BufferUseKind::USE_NO_ALIAS);

  // Make sure that higher USE_KINDS aren't overwritten.
  ASSERT_EQ(buffer0_.use_kind(), BufferUseKind::USE_ALIAS_READ_ONLY);
  ASSERT_EQ(buffer1_.use_kind(), BufferUseKind::USE_ALIAS_READ_ONLY);
  ASSERT_EQ(buffer2_.use_kind(), BufferUseKind::USE_NO_ALIAS);
}

TEST_F(BufferUseKindPropogationTest, BufferSetAddUseKind) {
  HloPoplarBufferSet read_only_buffer_set({&buffer0_, &buffer1_});
  ASSERT_EQ(buffer0_.use_kind(), BufferUseKind::USE_NO_ALIAS);
  ASSERT_EQ(buffer1_.use_kind(), BufferUseKind::USE_NO_ALIAS);

  read_only_buffer_set.AddNewBufferUse(BufferUseKind::USE_ALIAS_READ_ONLY);
  ASSERT_EQ(buffer0_.use_kind(), BufferUseKind::USE_ALIAS_READ_ONLY);
  ASSERT_EQ(buffer1_.use_kind(), BufferUseKind::USE_ALIAS_READ_ONLY);

  // Make sure that higher USE_KINDS aren't overwritten.
  read_only_buffer_set.AddNewBufferUse(BufferUseKind::USE_NO_ALIAS);
  ASSERT_EQ(buffer0_.use_kind(), BufferUseKind::USE_ALIAS_READ_ONLY);
  ASSERT_EQ(buffer1_.use_kind(), BufferUseKind::USE_ALIAS_READ_ONLY);
}

TEST_F(BufferUseKindPropogationTest, BufferSetAddBuffer) {
  ASSERT_NE(buffer0_.use_kind(), BufferUseKind::USE_ALIAS_READ_WRITE);

  HloPoplarBufferSet read_write_buffer_set(BufferUseKind::USE_ALIAS_READ_WRITE);
  read_write_buffer_set.AddBuffer(&buffer0_);
  ASSERT_EQ(buffer0_.use_kind(), BufferUseKind::USE_ALIAS_READ_WRITE);
}

TEST_F(BufferUseKindPropogationTest, BufferSetUnion) {
  HloPoplarBufferSet read_write_buffer_set({&buffer1_, &buffer2_},
                                           BufferUseKind::USE_ALIAS_READ_WRITE);
  HloPoplarBufferSet no_alias_buffer_set({&buffer0_},
                                         BufferUseKind::USE_NO_ALIAS);

  HloPoplarBufferSet union_set;
  auto result =
      union_set.AssignUnionOf({&read_write_buffer_set, &no_alias_buffer_set},
                              BufferUseKind::USE_NO_ALIAS);
  ASSERT_TRUE(result);

  ASSERT_EQ(buffer0_.use_kind(), BufferUseKind::USE_ALIAS_READ_WRITE);
  ASSERT_EQ(buffer1_.use_kind(), BufferUseKind::USE_ALIAS_READ_WRITE);
  ASSERT_EQ(buffer2_.use_kind(), BufferUseKind::USE_ALIAS_READ_WRITE);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
