/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/elementwise_broadcast_converter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/forward_allocation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_into_poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_wide_const.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/module_flatten.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/while_loop_to_repeat_simplify.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_passes/embedding_plans_preplanning.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/ml_type_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/literal_comparison.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using AllocationFinderTest = HloTestBase;

static Window GetConv1Window() {
  Window window;
  for (int i = 0; i < 2; ++i) {
    auto dim = window.add_dimensions();
    dim->set_size(3);
    dim->set_stride(1);
    dim->set_padding_low(1);
    dim->set_padding_high(1);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }
  return window;
}

static Window GetConv2Window() {
  Window window;
  for (int i = 0; i < 2; ++i) {
    auto dim = window.add_dimensions();
    dim->set_size(3);
    dim->set_stride(2);
    dim->set_padding_low(1);
    dim->set_padding_high(1);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }
  return window;
}

static ConvolutionDimensionNumbers GetConvDimensions() {
  ConvolutionDimensionNumbers dimension;
  dimension.set_input_batch_dimension(0);
  dimension.add_input_spatial_dimensions(1);
  dimension.add_input_spatial_dimensions(2);
  dimension.set_input_feature_dimension(3);

  dimension.set_output_batch_dimension(0);
  dimension.add_output_spatial_dimensions(1);
  dimension.add_output_spatial_dimensions(2);
  dimension.set_output_feature_dimension(3);

  dimension.add_kernel_spatial_dimensions(0);
  dimension.add_kernel_spatial_dimensions(1);
  dimension.set_kernel_input_feature_dimension(2);
  dimension.set_kernel_output_feature_dimension(3);
  return dimension;
}

// Check basic parameter matching
TEST_F(AllocationFinderTest, FindBasicTensorAllocations) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[1,16,16,2] parameter(1)
  p2 = f16[3,3,2,4] parameter(2)

  add = f16[1,16,16,2] add(p0, p1)

  conv = f16[1,16,16,4] convolution(p0, p2), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f

  ROOT t = (f16[1,16,16,4], f16[1,16,16,2]) tuple(conv, add)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* conv = root->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip2 = conv->operand(1);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

TEST_F(AllocationFinderTest, FindAllocationTargetWithPriority) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[1,16,16,2] parameter(1)
  p2 = f16[3,3,2,4] parameter(2)

  add = f16[1,16,16,2] add(p0, p1)

  zero = s32[] constant(0)
  ds = f16[1,16,16,1] dynamic-slice(p0, zero, zero, zero, zero), dynamic_slice_sizes={1,16,16,1}
  conv = f16[1,16,16,4] convolution(p0, p2), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f

  ROOT t = (f16[1,16,16,4], f16[1,16,16,2], f16[1,16,16,1]) tuple(conv, add, ds)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* conv = root->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip2 = conv->operand(1);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ((*t.sliceable_dimension), 3);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

TEST_F(AllocationFinderTest, FindAllocationTargetWithPriority2) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f16[2, 64] parameter(0)
  p1 = f16[1024, 64] parameter(1)
  p2 = s32[2] parameter(2)
  slice = f16[2, 64] custom-call(p1, p2), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  p1_t = f16[64, 1024] transpose(p1), dimensions={1, 0}
  dot = f16[2, 1024] dot(p0, p1_t), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT t = (f16[2, 64], f16[2, 1024]) tuple(slice, dot)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* slice = root->operand(0);
  const auto* p1 = slice->operand(0);
  const auto* p2 = slice->operand(1);
  const auto* dot = root->operand(1);
  const auto* p0 = dot->operand(0);
  const auto* transpose = dot->operand(1);
  EXPECT_EQ(p1, transpose->operand(0));

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{p0, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{p1, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], transpose);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(1, 0));
  EXPECT_EQ((*t.sliceable_dimension), 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{p2, 0});
  EXPECT_EQ(t.tgt, slice);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

TEST_F(AllocationFinderTest, FindAllocationTargetWithPriority3) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f16[2, 64] parameter(0)
  p1 = f16[64, 1024] parameter(1)
  p2 = s32[2] parameter(2)
  p1_t = f16[1024, 64] transpose(p1), dimensions={1, 0}
  slice = f16[2, 64] custom-call(p1_t, p2), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  dot = f16[2, 1024] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT t = (f16[2, 64], f16[2, 1024]) tuple(slice, dot)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* slice = root->operand(0);
  const auto* transpose = slice->operand(0);
  const auto* p1 = transpose->operand(0);
  const auto* p2 = slice->operand(1);
  const auto* dot = root->operand(1);
  const auto* p0 = dot->operand(0);
  EXPECT_EQ(p1, dot->operand(1));

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{p0, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{p1, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ((*t.sliceable_dimension), 1);

  t = annotations.tensor_allocation_map.at(TensorLocation{p2, 0});
  EXPECT_EQ(t.tgt, slice);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

TEST_F(AllocationFinderTest, FindAllocationTargetWithPriority4) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f16[2, 64] parameter(0)
  p1 = f16[64, 1024] parameter(1)
  p2 = s32[2, 1] parameter(2)
  p1_t = f16[1024, 64] transpose(p1), dimensions={1, 0}
  mu = f16[1024, 64] custom-call(p1_t, p2, p0), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
  dot = f16[2, 1024] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT t = (f16[2, 64], f16[2, 1024]) tuple(mu, dot)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* mu = root->operand(0);
  const auto* transpose = mu->operand(0);
  const auto* p1 = transpose->operand(0);
  const auto* p2 = mu->operand(1);
  const auto* dot = root->operand(1);
  const auto* p0 = dot->operand(0);
  EXPECT_EQ(p1, dot->operand(1));

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{p0, 0});
  EXPECT_EQ(t.tgt, mu);
  EXPECT_EQ(t.input_index, 2ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{p1, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ((*t.sliceable_dimension), 1);

  t = annotations.tensor_allocation_map.at(TensorLocation{p2, 0});
  EXPECT_EQ(t.tgt, mu);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

// Check it goes through call sites
TEST_F(AllocationFinderTest, FindSubCompTensorAllocations) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 10, 10, 2});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {3, 3, 2, 1});

  Shape conv_shape =
      ShapeInference::InferConvolveShape(
          input_shape, weight_shape, /*feature_group_count=*/1,
          /*batch_group_count*/ 1, GetConv1Window(), GetConvDimensions())
          .ConsumeValueOrDie();

  /* Create convolution sub-computation */
  auto builder_sub = HloComputation::Builder(TestName());
  auto op0_sub = builder_sub.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "input"));
  auto op1_sub = builder_sub.AddInstruction(
      HloInstruction::CreateParameter(1, weight_shape, "weights"));

  auto conv = builder_sub.AddInstruction(HloInstruction::CreateConvolve(
      conv_shape, op0_sub, op1_sub, /*feature_group_count=*/1,
      /*batch_group_count*/ 1, GetConv1Window(), GetConvDimensions(),
      DefaultPrecisionConfig(2)));

  auto computation_sub = builder_sub.Build();

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto op0 = builder_main.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "op0"));
  auto op1 = builder_main.AddInstruction(
      HloInstruction::CreateParameter(1, input_shape, "op1"));
  auto op2 = builder_main.AddInstruction(
      HloInstruction::CreateParameter(2, weight_shape, "op2"));

  auto add = builder_main.AddInstruction(
      HloInstruction::CreateBinary(input_shape, HloOpcode::kAdd, op0, op1));

  auto call = builder_main.AddInstruction(HloInstruction::CreateCall(
      conv_shape, {op1, op2}, computation_sub.get()));

  builder_main.AddInstruction(HloInstruction::CreateTuple({add, call}));

  auto computation_main = builder_main.Build();

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEmbeddedComputation(std::move(computation_sub));
  hlo_module->AddEntryComputation(std::move(computation_main));

  CompilerAnnotations annotations(hlo_module.get());

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(hlo_module.get()).ValueOrDie());

  const HloInstruction* c_conv = conv;

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);
  auto t = annotations.tensor_allocation_map.at(TensorLocation{op1, 0});
  EXPECT_EQ(t.tgt, c_conv);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{op2, 0});
  EXPECT_EQ(t.tgt, c_conv);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{op0_sub, 0});
  EXPECT_EQ(t.tgt, c_conv);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{op1_sub, 0});
  EXPECT_EQ(t.tgt, c_conv);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

// Check it works for multiple valid destinations (perferred one first)
TEST_F(AllocationFinderTest, FindMultiCompTensorAllocations1) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 10, 10, 2});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {3, 3, 2, 1});

  Shape conv1_shape =
      ShapeInference::InferConvolveShape(
          input_shape, weight_shape, /*feature_group_count=*/1,
          /*batch_group_count*/ 1, GetConv1Window(), GetConvDimensions())
          .ConsumeValueOrDie();

  Shape conv2_shape =
      ShapeInference::InferConvolveShape(
          input_shape, weight_shape, /*feature_group_count=*/1,
          /*batch_group_count*/ 1, GetConv2Window(), GetConvDimensions())
          .ConsumeValueOrDie();

  /* Create convolution sub-computation 1 */
  auto builder_sub1 = HloComputation::Builder(TestName());
  auto op0_sub1 = builder_sub1.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "input"));
  auto op1_sub1 = builder_sub1.AddInstruction(
      HloInstruction::CreateParameter(1, weight_shape, "weights"));

  auto conv1 = builder_sub1.AddInstruction(HloInstruction::CreateConvolve(
      conv1_shape, op0_sub1, op1_sub1, /*feature_group_count=*/1,
      /*batch_group_count*/ 1, GetConv1Window(), GetConvDimensions(),
      DefaultPrecisionConfig(2)));

  auto computation_sub1 = builder_sub1.Build();

  /* Create convolution sub-computation 2 */
  auto builder_sub2 = HloComputation::Builder(TestName());
  auto op0_sub2 = builder_sub2.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "input"));
  auto op1_sub2 = builder_sub2.AddInstruction(
      HloInstruction::CreateParameter(1, weight_shape, "weights"));

  auto conv2 = builder_sub2.AddInstruction(HloInstruction::CreateConvolve(
      conv2_shape, op0_sub2, op1_sub2, /*feature_group_count=*/1,
      /*batch_group_count*/ 1, GetConv2Window(), GetConvDimensions(),
      DefaultPrecisionConfig(2)));

  auto computation_sub2 = builder_sub2.Build();

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto op0 = builder_main.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "op0"));
  auto op1 = builder_main.AddInstruction(
      HloInstruction::CreateParameter(1, input_shape, "op1"));
  auto op2 = builder_main.AddInstruction(
      HloInstruction::CreateParameter(2, weight_shape, "op2"));

  auto add = builder_main.AddInstruction(
      HloInstruction::CreateBinary(input_shape, HloOpcode::kAdd, op0, op1));

  auto call1 = builder_main.AddInstruction(HloInstruction::CreateCall(
      conv1_shape, {op1, op2}, computation_sub1.get()));

  auto call2 = builder_main.AddInstruction(HloInstruction::CreateCall(
      conv2_shape, {op1, op2}, computation_sub2.get()));

  builder_main.AddInstruction(HloInstruction::CreateTuple({add, call1, call2}));

  auto computation_main = builder_main.Build();

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEmbeddedComputation(std::move(computation_sub1));
  hlo_module->AddEmbeddedComputation(std::move(computation_sub2));
  hlo_module->AddEntryComputation(std::move(computation_main));

  CompilerAnnotations annotations(hlo_module.get());

  TF_ASSERT_OK(SetInstructionMLType(conv1, MLType::TRAINING_FWD));
  TF_ASSERT_OK(SetInstructionMLType(conv2, MLType::TRAINING_BWD));

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(hlo_module.get()).ValueOrDie());

  const HloInstruction* c_conv1 = conv1;
  const HloInstruction* c_conv2 = conv2;

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 6);
  auto t = annotations.tensor_allocation_map.at(TensorLocation{op1, 0});
  EXPECT_EQ(t.tgt, c_conv1);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{op2, 0});
  EXPECT_EQ(t.tgt, c_conv1);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{op0_sub1, 0});
  EXPECT_EQ(t.tgt, c_conv1);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{op1_sub1, 0});
  EXPECT_EQ(t.tgt, c_conv1);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{op0_sub2, 0});
  EXPECT_EQ(t.tgt, c_conv2);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{op1_sub2, 0});
  EXPECT_EQ(t.tgt, c_conv2);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

// Check it works for multiple valid destinations (perferred one second)
TEST_F(AllocationFinderTest, FindMultiCompTensorAllocations2) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 10, 10, 2});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {3, 3, 2, 1});

  Shape conv1_shape =
      ShapeInference::InferConvolveShape(
          input_shape, weight_shape, /*feature_group_count=*/1,
          /*batch_group_count*/ 1, GetConv1Window(), GetConvDimensions())
          .ConsumeValueOrDie();

  Shape conv2_shape =
      ShapeInference::InferConvolveShape(
          input_shape, weight_shape, /*feature_group_count=*/1,
          /*batch_group_count*/ 1, GetConv2Window(), GetConvDimensions())
          .ConsumeValueOrDie();

  /* Create convolution sub-computation 1 */
  auto builder_sub1 = HloComputation::Builder(TestName());
  auto op0_sub1 = builder_sub1.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "input"));
  auto op1_sub1 = builder_sub1.AddInstruction(
      HloInstruction::CreateParameter(1, weight_shape, "weights"));

  auto conv1 = builder_sub1.AddInstruction(HloInstruction::CreateConvolve(
      conv1_shape, op0_sub1, op1_sub1, /*feature_group_count=*/1,
      /*batch_group_count*/ 1, GetConv1Window(), GetConvDimensions(),
      DefaultPrecisionConfig(2)));

  auto computation_sub1 = builder_sub1.Build();

  /* Create convolution sub-computation 2 */
  auto builder_sub2 = HloComputation::Builder(TestName());
  auto op0_sub2 = builder_sub2.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "input"));
  auto op1_sub2 = builder_sub2.AddInstruction(
      HloInstruction::CreateParameter(1, weight_shape, "weights"));

  auto conv2 = builder_sub2.AddInstruction(HloInstruction::CreateConvolve(
      conv2_shape, op0_sub2, op1_sub2, /*feature_group_count=*/1,
      /*batch_group_count*/ 1, GetConv2Window(), GetConvDimensions(),
      DefaultPrecisionConfig(2)));

  auto computation_sub2 = builder_sub2.Build();

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto op0 = builder_main.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "op0"));
  auto op1 = builder_main.AddInstruction(
      HloInstruction::CreateParameter(1, input_shape, "op1"));
  auto op2 = builder_main.AddInstruction(
      HloInstruction::CreateParameter(2, weight_shape, "op2"));

  auto add = builder_main.AddInstruction(
      HloInstruction::CreateBinary(input_shape, HloOpcode::kAdd, op0, op1));

  auto call1 = builder_main.AddInstruction(HloInstruction::CreateCall(
      conv1_shape, {op1, op2}, computation_sub1.get()));

  auto call2 = builder_main.AddInstruction(HloInstruction::CreateCall(
      conv2_shape, {op1, op2}, computation_sub2.get()));

  builder_main.AddInstruction(HloInstruction::CreateTuple({add, call1, call2}));

  auto computation_main = builder_main.Build();

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEmbeddedComputation(std::move(computation_sub1));
  hlo_module->AddEmbeddedComputation(std::move(computation_sub2));
  hlo_module->AddEntryComputation(std::move(computation_main));

  CompilerAnnotations annotations(hlo_module.get());
  TF_ASSERT_OK(SetInstructionMLType(conv1, MLType::TRAINING_BWD));
  TF_ASSERT_OK(SetInstructionMLType(conv2, MLType::TRAINING_FWD));

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(hlo_module.get()).ValueOrDie());

  const HloInstruction* c_conv1 = conv1;
  const HloInstruction* c_conv2 = conv2;

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 6);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{op1, 0});
  EXPECT_EQ(t.tgt, c_conv2);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{op2, 0});
  EXPECT_EQ(t.tgt, c_conv2);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{op0_sub1, 0});
  EXPECT_EQ(t.tgt, c_conv1);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{op1_sub1, 0});
  EXPECT_EQ(t.tgt, c_conv1);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{op0_sub2, 0});
  EXPECT_EQ(t.tgt, c_conv2);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{op1_sub2, 0});
  EXPECT_EQ(t.tgt, c_conv2);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

// Check it works for constants
TEST_F(AllocationFinderTest, FindConstantTensorAllocations) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[1,16,16,2] parameter(1)
  p2 = f16[1,1,2,4] constant({{{{1,0,0,0},{1,0,0,0}}}})

  add = f16[1,16,16,2] add(p0, p1)

  conv = f16[1,16,16,4] convolution(p0, p2), window={size=1x1}, dim_labels=b01f_01io->b01f

  ROOT t = (f16[1,16,16,4], f16[1,16,16,2]) tuple(conv, add)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* conv = root->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip2 = conv->operand(1);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

// Check it goes through Tuple/Detuple pairs
TEST_F(AllocationFinderTest, CanTraverseTuples) {
  auto hlo_module = CreateNewVerifiedModule();

  Shape lhs_shape = ShapeUtil::MakeShape(F32, {2});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 2});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({lhs_shape, rhs_shape});

  auto b = HloComputation::Builder(TestName());
  auto in =
      b.AddInstruction(HloInstruction::CreateParameter(0, lhs_shape, "in"));
  auto w =
      b.AddInstruction(HloInstruction::CreateParameter(1, rhs_shape, "weight"));

  auto tuple = b.AddInstruction(HloInstruction::CreateTuple({in, w}));

  auto in1 = b.AddInstruction(
      HloInstruction::CreateGetTupleElement(lhs_shape, tuple, 0));
  auto w1 = b.AddInstruction(
      HloInstruction::CreateGetTupleElement(rhs_shape, tuple, 1));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot_inst = b.AddInstruction(HloInstruction::CreateDot(
      lhs_shape, in1, w1, dot_dnums, DefaultPrecisionConfig(2)));

  hlo_module->AddEntryComputation(b.Build());

  CompilerAnnotations annotations(hlo_module.get());

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(hlo_module.get()).ValueOrDie());

  const HloInstruction* dot = dot_inst;

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{in, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{w, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

// Check it can start from tuple subshapes
TEST_F(AllocationFinderTest, CanStartOnTuples) {
  auto hlo_module = CreateNewVerifiedModule();

  Shape lhs_shape = ShapeUtil::MakeShape(F32, {2});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 2});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({lhs_shape, rhs_shape});

  auto b = HloComputation::Builder(TestName());
  auto in = b.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "tuple"));

  auto in1 =
      b.AddInstruction(HloInstruction::CreateGetTupleElement(lhs_shape, in, 0));
  auto w1 =
      b.AddInstruction(HloInstruction::CreateGetTupleElement(rhs_shape, in, 1));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot_inst = b.AddInstruction(HloInstruction::CreateDot(
      lhs_shape, in1, w1, dot_dnums, DefaultPrecisionConfig(2)));

  hlo_module->AddEntryComputation(b.Build());

  CompilerAnnotations annotations(hlo_module.get());

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(hlo_module.get()).ValueOrDie());

  const HloInstruction* dot = dot_inst;

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{in, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{in, 1});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

// Check it goes through while instructions
TEST_F(AllocationFinderTest, FindWhileTensorAllocations) {
  auto hlo_module = CreateNewVerifiedModule();

  Shape counter_shape = ShapeUtil::MakeShape(S32, {});
  Shape input_shape = ShapeUtil::MakeShape(F32, {2});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {2, 2});
  Shape tuple_shape =
      ShapeUtil::MakeTupleShape({counter_shape, input_shape, weight_shape});

  /* Create while condition */
  HloComputation* comp_cond;
  {
    auto builder_cond = HloComputation::Builder(TestName());
    auto tuple = builder_cond.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "cond_tuple"));
    auto limit = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(10)));
    auto c = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 0));
    builder_cond.AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::MakeShape(PRED, {}), c, limit, ComparisonDirection::kLt));

    comp_cond = hlo_module->AddEmbeddedComputation(builder_cond.Build());
  }

  /* Create while body */
  HloComputation* comp_body;
  auto builder_body = HloComputation::Builder(TestName());
  auto tuple_body = builder_body.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "body_tuple"));
  auto c_body = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(counter_shape, tuple_body, 0));
  auto in_body = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(input_shape, tuple_body, 1));
  auto w_body = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(weight_shape, tuple_body, 2));
  auto one = builder_body.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
  auto new_c_body = builder_body.AddInstruction(HloInstruction::CreateBinary(
      c_body->shape(), HloOpcode::kAdd, c_body, one));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot_inst = builder_body.AddInstruction(HloInstruction::CreateDot(
      input_shape, in_body, w_body, dot_dnums, DefaultPrecisionConfig(2)));

  builder_body.AddInstruction(
      HloInstruction::CreateTuple({new_c_body, dot_inst, w_body}));

  comp_body = hlo_module->AddEmbeddedComputation(builder_body.Build());

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto c = builder_main.AddInstruction(
      HloInstruction::CreateParameter(0, counter_shape, "counter"));
  auto in = builder_main.AddInstruction(
      HloInstruction::CreateParameter(1, input_shape, "in"));
  auto w = builder_main.AddInstruction(
      HloInstruction::CreateParameter(2, weight_shape, "weight"));

  auto init =
      builder_main.AddInstruction(HloInstruction::CreateTuple({c, in, w}));

  auto main = builder_main.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));

  builder_main.AddInstruction(HloInstruction::CreateTuple({main}));

  hlo_module->AddEntryComputation(builder_main.Build());

  CompilerAnnotations annotations(hlo_module.get());

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(hlo_module.get()).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{in, 0});
  EXPECT_EQ(t.tgt, dot_inst);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 3);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{w, 0});
  EXPECT_EQ(t.tgt, dot_inst);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 3);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{tuple_body, 1});
  EXPECT_EQ(t.tgt, dot_inst);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{tuple_body, 2});
  EXPECT_EQ(t.tgt, dot_inst);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

// Check it goes through repeat instructions
TEST_F(AllocationFinderTest, FindRepeatTensorAllocations) {
  auto hlo_module = CreateNewVerifiedModule();

  Shape counter_shape = ShapeUtil::MakeShape(S32, {});
  Shape input_shape = ShapeUtil::MakeShape(F32, {2});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {2, 2});
  Shape tuple_shape =
      ShapeUtil::MakeTupleShape({counter_shape, input_shape, weight_shape});

  /* Create while condition */
  HloComputation* comp_cond;
  {
    auto builder_cond = HloComputation::Builder(TestName());
    auto tuple = builder_cond.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "cond_tuple"));
    auto limit = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(10)));
    auto c = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 0));
    builder_cond.AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::MakeShape(PRED, {}), c, limit, ComparisonDirection::kLt));

    comp_cond = hlo_module->AddEmbeddedComputation(builder_cond.Build());
  }

  /* Create while body */
  HloComputation* comp_body;
  {
    auto builder_body = HloComputation::Builder(TestName());
    auto tuple = builder_body.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "body_tuple"));
    auto c = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(counter_shape, tuple, 0));
    auto in = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(input_shape, tuple, 1));
    auto w = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(weight_shape, tuple, 2));
    auto one = builder_body.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
    auto new_c = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c->shape(), HloOpcode::kAdd, c, one));

    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(0);
    dot_dnums.add_rhs_contracting_dimensions(1);
    auto new_in = builder_body.AddInstruction(HloInstruction::CreateDot(
        input_shape, in, w, dot_dnums, DefaultPrecisionConfig(2)));

    builder_body.AddInstruction(
        HloInstruction::CreateTuple({new_c, new_in, w}));

    comp_body = hlo_module->AddEmbeddedComputation(builder_body.Build());
  }

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto c = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  auto in = builder_main.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "in"));
  auto w = builder_main.AddInstruction(
      HloInstruction::CreateParameter(1, weight_shape, "weight"));

  auto init =
      builder_main.AddInstruction(HloInstruction::CreateTuple({c, in, w}));

  builder_main.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));

  hlo_module->AddEntryComputation(builder_main.Build());

  // Simplify the while loop to a repeat (need to also run DCE)
  WhileLoopToRepeatSimplify wltrs;
  EXPECT_TRUE(wltrs.Run(hlo_module.get()).ValueOrDie());
  HloDCE hdce;
  EXPECT_TRUE(hdce.Run(hlo_module.get()).ValueOrDie());

  CompilerAnnotations annotations(hlo_module.get());

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(hlo_module.get()).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  // Get the dot and tuple instruction from the new repeat body.
  const HloComputation* repeat_body =
      hlo_module->entry_computation()->root_instruction()->to_apply();
  const HloInstruction* dot_inst = repeat_body->root_instruction()->operand(1);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{in, 0});
  EXPECT_EQ(t.tgt, dot_inst);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{w, 0});
  EXPECT_EQ(t.tgt, dot_inst);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(
      TensorLocation{repeat_body->parameter_instruction(1), 0});
  EXPECT_EQ(t.tgt, dot_inst);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(
      TensorLocation{repeat_body->parameter_instruction(2), 0});
  EXPECT_EQ(t.tgt, dot_inst);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

// Check basic parameter matching
TEST_F(AllocationFinderTest, TraverseDimShuffleAndReshapeAllocations) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[3,3,4,2] parameter(1)

  p1_t = f16[3,3,2,4] transpose(p1), dimensions={0, 1, 3, 2}

  conv = f16[1,16,16,4] convolution(p0, p1_t), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f

  ROOT t = (f16[1,16,16,4]) tuple(conv)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* conv = root->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* trans = conv->operand(1);
  const auto* ip1 = trans->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], trans);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 3, 2));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

// Check it goes through call sites
TEST_F(AllocationFinderTest, FindDoesntTraceThroughInvalidCalls) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 10, 10, 2});
  Shape half_shape = ShapeUtil::MakeShape(F32, {1, 10, 10, 1});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {3, 3, 2, 1});

  Shape conv_shape =
      ShapeInference::InferConvolveShape(
          input_shape, weight_shape, /*feature_group_count=*/1,
          /*batch_group_count*/ 1, GetConv1Window(), GetConvDimensions())
          .ConsumeValueOrDie();

  /* Create sub-computation which contains an unacceptable op */
  auto builder_sub = HloComputation::Builder(TestName());
  HloInstruction* op0_sub = builder_sub.AddInstruction(
      HloInstruction::CreateParameter(0, half_shape, "input"));
  HloInstruction* op1_sub = builder_sub.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(half_shape)));
  HloInstruction* concat = builder_sub.AddInstruction(
      HloInstruction::CreateConcatenate(input_shape, {op0_sub, op1_sub}, 3));
  auto computation_sub = builder_sub.Build();

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  HloInstruction* op0 = builder_main.AddInstruction(
      HloInstruction::CreateParameter(0, half_shape, "op0"));
  HloInstruction* op1 = builder_main.AddInstruction(
      HloInstruction::CreateParameter(1, weight_shape, "op1"));
  HloInstruction* call = builder_main.AddInstruction(
      HloInstruction::CreateCall(input_shape, {op0}, computation_sub.get()));
  HloInstruction* conv =
      builder_main.AddInstruction(HloInstruction::CreateConvolve(
          conv_shape, call, op1, /*feature_group_count=*/1,
          /*batch_group_count*/ 1, GetConv1Window(), GetConvDimensions(),
          DefaultPrecisionConfig(2)));

  builder_main.AddInstruction(HloInstruction::CreateTuple({conv}));

  auto computation_main = builder_main.Build();

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEmbeddedComputation(std::move(computation_sub));
  hlo_module->AddEntryComputation(std::move(computation_main));

  CompilerAnnotations annotations(hlo_module.get());

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(hlo_module.get()).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  auto t0 = annotations.tensor_allocation_map.at(TensorLocation{op0, 0});
  EXPECT_EQ(t0.tgt, conv);
  EXPECT_EQ(t0.input_index, 0ll);
  EXPECT_EQ(t0.backward_path.size(), 3);
  EXPECT_EQ(t0.backward_path[0], op0_sub);
  EXPECT_EQ(t0.backward_path[1], concat);
  EXPECT_EQ(t0.backward_path[2], call);
  EXPECT_THAT((*t0.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t0.sliceable_dimension, absl::nullopt);

  auto t1 = annotations.tensor_allocation_map.at(TensorLocation{op1, 0});
  EXPECT_EQ(t1.tgt, conv);
  EXPECT_EQ(t1.input_index, 1ll);
  EXPECT_EQ(t1.backward_path.size(), 0);
  EXPECT_THAT((*t1.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t1.sliceable_dimension, absl::nullopt);

  auto t2 = annotations.tensor_allocation_map.at(TensorLocation{op0_sub, 0});
  EXPECT_EQ(t2.tgt, conv);
  EXPECT_EQ(t2.input_index, 0ll);
  EXPECT_EQ(t2.backward_path.size(), 2);
  EXPECT_EQ(t2.backward_path[0], concat);
  EXPECT_EQ(t2.backward_path[1], call);
  EXPECT_THAT((*t2.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t2.sliceable_dimension, absl::nullopt);

  auto t3 = annotations.tensor_allocation_map.at(TensorLocation{op1_sub, 0});
  EXPECT_EQ(t3.tgt, conv);
  EXPECT_EQ(t3.input_index, 0ll);
  EXPECT_EQ(t3.backward_path.size(), 2);
  EXPECT_EQ(t3.backward_path[0], concat);
  EXPECT_EQ(t3.backward_path[1], call);
  EXPECT_THAT((*t3.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t3.sliceable_dimension, absl::nullopt);
}

TEST_F(AllocationFinderTest, BiasAdd1) {
  std::string hlo = R"(
HloModule top

_pop_op_conv_biasadd {
  arg_0 = f16[1,16,16,4] parameter(0)
  arg_1 = f16[4] parameter(1)
  bcast = f16[1,16,16,4] broadcast(arg_1), dimensions={3}
  ROOT add = f16[1,16,16,4] add(arg_0, bcast)
}

ENTRY c1 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[3,3,2,4] parameter(1)
  p2 = f16[4] parameter(2)

  conv = f16[1,16,16,4] convolution(p0, p1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  call = f16[1,16,16,4] fusion(conv, p2), kind=kCustom, calls=_pop_op_conv_biasadd

  ROOT t = (f16[1,16,16,4]) tuple(call)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* call = root->operand(0);
  const auto* conv = call->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* ip2 = call->operand(1);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, call);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
}

TEST_F(AllocationFinderTest, BiasAddATwice) {
  std::string hlo = R"(
HloModule top

_pop_op_conv_biasadd {
  arg_0 = f16[1,16,16,4] parameter(0)
  arg_1 = f16[4] parameter(1)
  bcast = f16[1,16,16,4] broadcast(arg_1), dimensions={3}
  ROOT add = f16[1,16,16,4] add(arg_0, bcast)
}

_pop_op_conv_biasadd.1 {
  arg_0 = f16[1,16,16,4] parameter(0)
  arg_1 = f16[4] parameter(1)
  bcast = f16[1,16,16,4] broadcast(arg_1), dimensions={3}
  ROOT add = f16[1,16,16,4] add(arg_0, bcast)
}

ENTRY c1 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[3,3,2,4] parameter(1)
  p2 = f16[4] parameter(2)
  p3 = f16[4] parameter(3)

  conv = f16[1,16,16,4] convolution(p0, p1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  call = f16[1,16,16,4] fusion(conv, p2), kind=kCustom, calls=_pop_op_conv_biasadd
  call.1 = f16[1,16,16,4] fusion(call, p3), kind=kCustom, calls=_pop_op_conv_biasadd.1

  ROOT t = (f16[1,16,16,4]) tuple(call.1)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto& call1 = root->operand(0);
  const auto* call = call1->operand(0);
  const auto* conv = call->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* ip2 = call->operand(1);
  const auto* ip3 = call1->operand(1);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added two new entries to the map for the 2 bias add ops
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, call);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip3, 0});
  EXPECT_EQ(t.tgt, call1);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 1);
  EXPECT_EQ(t.forward_path[0], call);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, BiasAddWithPath) {
  std::string hlo = R"(
HloModule top

_pop_op_conv_biasadd {
  arg_0 = f16[1,16,16,4] parameter(0)
  arg_1 = f16[4] parameter(1)
  bcast = f16[1,16,16,4] broadcast(arg_1), dimensions={3}
  ROOT add = f16[1,16,16,4] add(arg_0, bcast)
}

ENTRY c1 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[3,3,2,4] parameter(1)
  p2 = f16[2,2] parameter(2)

  p2_r = f16[4] reshape(p2)

  conv = f16[1,16,16,4] convolution(p0, p1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  call = f16[1,16,16,4] fusion(conv, p2_r), kind=kCustom, calls=_pop_op_conv_biasadd

  ROOT t = (f16[1,16,16,4]) tuple(call)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* call = root->operand(0);
  const auto* conv = call->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* reshape = call->operand(1);
  const auto* ip2 = reshape->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, call);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape);
}

TEST_F(AllocationFinderTest, MatMulBiasAdd) {
  std::string hlo = R"(
HloModule top

 _pop_op_matmul_biasadd (arg_0: f32[2,2], arg_1: f32[2]) -> f32[2,2] {
   arg_1 = f32[2] parameter(1)
   broadcast.12.7.clone = f32[2,2] broadcast(arg_1), dimensions={1}
   arg_0 = f32[2,2] parameter(0)
   ROOT add.12.8.clone = f32[2,2] add(f32[2,2] arg_0, f32[2,2] broadcast.12.7.clone)
 }

 ENTRY c (arg0.12.0: f32[2,2], arg1.12.1: f32[2,2], arg2.12.2: f32[2]) -> f32[2,2] {
   arg0.12.0 = f32[2,2] parameter(0)
   arg1.12.1 = f32[2,2] parameter(1)
   dot.12.6 = f32[2,2] dot(f32[2,2] arg0.12.0, f32[2,2] arg1.12.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
   arg2.12.2 = f32[2] parameter(2), control-predecessors={dot.12.6}
   ROOT call = f32[2,2] fusion(f32[2,2] dot.12.6, arg2.12.2), kind=kCustom, calls=_pop_op_matmul_biasadd
 }

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* call = root;
  const auto* dot = call->operand(0);
  const auto* ip0 = dot->operand(0);
  const auto* ip1 = dot->operand(1);
  const auto* ip2 = call->operand(1);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the dot parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, call);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, dot);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest,
       NoTargetBecauseOfDepenencyFromLayoutCreatorToSource) {
  // arg2.12.2 layout cannot depend on the layout of the dot.12.6, because the
  // former is a predecessor of the latter.
  std::string hlo = R"(
HloModule top

 _pop_op_matmul_biasadd (arg_0: f32[2,2], arg_1: f32[2]) -> f32[2,2] {
   arg_1 = f32[2] parameter(1)
   broadcast.12.7.clone = f32[2,2] broadcast(arg_1), dimensions={1}
   arg_0 = f32[2,2] parameter(0)
   ROOT add.12.8.clone = f32[2,2] add(arg_0, broadcast.12.7.clone)
 }

 ENTRY c {
   arg0.12.0 = f32[2,2] parameter(0)
   arg1.12.1 = f32[2,2] parameter(1)
   arg2.12.2 = f32[2] parameter(2)
   dot.12.6 = f32[2,2] dot(arg0.12.0, arg1.12.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, control-predecessors={arg2.12.2}
   ROOT call = f32[2,2] fusion(dot.12.6, arg2.12.2), kind=kCustom, calls=_pop_op_matmul_biasadd
 }
)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* call = root;
  const auto* dot = call->operand(0);
  const auto* ip0 = dot->operand(0);
  const auto* ip1 = dot->operand(1);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the dot parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_FALSE(fwd_finder.Run(module0).ValueOrDie());
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);
}

TEST_F(AllocationFinderTest, MatMulBiasAddWithPath) {
  std::string hlo = R"(
HloModule top

 _pop_op_matmul_biasadd  {
   arg_1 = f32[2] parameter(1)
   broadcast.12.7.clone = f32[2,2] broadcast(arg_1), dimensions={1}
   arg_0 = f32[2,2] parameter(0)
   ROOT add.12.8.clone = f32[2,2] add(arg_0, broadcast.12.7.clone)
 }

 ENTRY c {
   arg0.12.0 = f32[2,2] parameter(0)
   arg1.12.1 = f32[2,2] parameter(1)
   dot.12.6 = f32[2,2] dot(arg0.12.0, arg1.12.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
   arg2.12.2 = f32[1,2] parameter(2), control-predecessors={dot.12.6}
   p2_r = f32[2] reshape(arg2.12.2)
   ROOT call = f32[2,2] fusion(dot.12.6, p2_r), kind=kCustom, calls=_pop_op_matmul_biasadd
 }

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* call = root;
  const auto* dot = call->operand(0);
  const auto* ip0 = dot->operand(0);
  const auto* ip1 = dot->operand(1);
  const auto* reshape = call->operand(1);
  const auto* ip2 = reshape->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the dot parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, call);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, dot);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape);
}

TEST_F(AllocationFinderTest, BatchNormInfParams) {
  std::string hlo = R"(
HloModule top

ENTRY top {
 arg0.36.22 = f32[1,4,4,2] parameter(0)
 arg1.36.24 = f32[2] parameter(1)
 arg2.36.25 = f32[2] parameter(2)
 arg3.36.26 = f32[2] parameter(3)
 arg4.36.27 = f32[2] parameter(4)
 ROOT batch-norm-inference.36.31 = f32[1,4,4,2] batch-norm-inference(arg0.36.22, arg1.36.24, arg2.36.25, arg3.36.26, arg4.36.27), epsilon=0.001, feature_index=3
}

 )";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* bn = root;
  const auto* ip0 = bn->operand(0);
  const auto* ip1 = bn->operand(1);
  const auto* ip2 = bn->operand(2);

  CompilerAnnotations annotations(module0);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, ip0);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, ip0);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, ConstantInput) {
  std::string hlo = R"(
HloModule top

ENTRY top  {
 arg0.36.22 = f32[1,4,4,2] parameter(0)
 arg1.36.23 = f32[1,1,2,2] parameter(1)
 convolution.36.29 = f32[1,4,4,2] convolution(arg0.36.22, arg1.36.23), window={size=1x1}, dim_labels=b01f_01io->b01f
 arg2.36.24 = f32[2] constant({0.0, 1.1})
 arg3.36.25 = f32[2] constant({0.0, 1.1})
 arg4.36.26 = f32[2] parameter(2)
 arg5.36.27 = f32[2] parameter(3)
 ROOT batch-norm-inference.36.31 = f32[1,4,4,2] batch-norm-inference(convolution.36.29, arg2.36.24, arg3.36.25, arg4.36.26, arg5.36.27), epsilon=0.001, feature_index=3
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* bn = root;
  const auto* conv = bn->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* ip2 = bn->operand(1);
  const auto* ip3 = bn->operand(2);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip3, 0});
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, BatchNormInfParamsWithPath) {
  std::string hlo = R"(
HloModule top

ENTRY top {
 arg0.36.22 = f32[1,4,4,2] parameter(0)
 arg1.36.23 = f32[1,1,2,2] parameter(1)
 convolution.36.29 = f32[1,4,4,2] convolution(arg0.36.22, arg1.36.23), window={size=1x1}, dim_labels=b01f_01io->b01f
 arg2.36.24 = f32[1,2] parameter(2)
 arg2.36.24_r = f32[2] reshape(arg2.36.24)
 arg3.36.25 = f32[1,2] parameter(3)
 arg3.36.25_r = f32[2] reshape(arg3.36.25)
 arg4.36.26 = f32[2] parameter(4)
 arg5.36.27 = f32[2] parameter(5)
 ROOT batch-norm-inference.36.31 = f32[1,4,4,2] batch-norm-inference(convolution.36.29, arg2.36.24_r, arg3.36.25_r, arg4.36.26, arg5.36.27), epsilon=0.001, feature_index=3
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* bn = root;
  const auto* conv = bn->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* reshape1 = bn->operand(1);
  const auto* reshape2 = bn->operand(2);
  const auto* ip2 = reshape1->operand(0);
  const auto* ip3 = reshape2->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape1);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip3, 0});
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape2);
}

TEST_F(AllocationFinderTest, BatchNormTrainingParams) {
  std::string hlo = R"(
HloModule top
Sum-reduction48 {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

_pop_op_conv_scaled_inplace {
  arg_1 = f32[1,4,4,2] parameter(1)
  arg_2 = f32[1,4,4,2] parameter(2)
  convolution.78.67.clone = f32[1,1,2,2] convolution(arg_1, arg_2), window={size=4x4}, dim_labels=f01b_i01o->01bf
  constant.78.28.clone = f32[] constant(0.1)
  broadcast.78.68.clone = f32[1,1,2,2] broadcast(f32[] constant.78.28.clone), dimensions={}
  multiply.78.69.clone = f32[1,1,2,2] multiply(convolution.78.67.clone, broadcast.78.68.clone)
  arg_0 = f32[1,1,2,2] parameter(0)
  ROOT subtract.78.70.clone = f32[1,1,2,2] subtract(arg_0, multiply.78.69.clone)
}

_pop_op_wide_const {
  constant.78.29.clone = f32[] constant(1)
  ROOT broadcast.2.clone = f32[1,4,4,2] broadcast(constant.78.29.clone), dimensions={}
}

_pop_op_wide_const.1 {
  constant.78.28.clone.1 = f32[] constant(0.1)
  ROOT broadcast.78.64.clone = f32[2] broadcast(constant.78.28.clone.1), dimensions={}
}

ENTRY top {
  constant.78.43 = f32[] constant(0)
  arg0.78.22 = f32[1,4,4,2] parameter(0)
  arg1.78.23 = f32[1,1,2,2] parameter(1)
  convolution.78.33 = f32[1,4,4,2] convolution(arg0.78.22, arg1.78.23), window={size=1x1}, dim_labels=b01f_01io->b01f
  arg2.78.24 = f32[2] parameter(2)
  arg3.78.25 = f32[2] parameter(3)
  batch-norm-training.78.35 = (f32[1,4,4,2], f32[2], f32[2]) batch-norm-training(convolution.78.33, arg2.78.24, arg3.78.25), epsilon=0.001, feature_index=3
  get-tuple-element.78.36 = f32[1,4,4,2] get-tuple-element(batch-norm-training.78.35), index=0
  reduce.78.49 = f32[] reduce(get-tuple-element.78.36, constant.78.43), dimensions={0,1,2,3}, to_apply=Sum-reduction48
  call.1 = f32[1,4,4,2] fusion(), kind=kCustom, calls=_pop_op_wide_const
  get-tuple-element.78.38 = f32[2] get-tuple-element(batch-norm-training.78.35), index=1
  get-tuple-element.78.39 = f32[2] get-tuple-element(batch-norm-training.78.35), index=2
  batch-norm-grad.78.54 = (f32[1,4,4,2], f32[2], f32[2]) batch-norm-grad(convolution.78.33, arg2.78.24, get-tuple-element.78.38, get-tuple-element.78.39, call.1), epsilon=0.001, feature_index=3
  get-tuple-element.78.55 = f32[1,4,4,2] get-tuple-element(batch-norm-grad.78.54), index=0
  call = f32[1,1,2,2] fusion(arg1.78.23, arg0.78.22, get-tuple-element.78.55), kind=kCustom, calls=_pop_op_conv_scaled_inplace
  call.2 = f32[2] fusion(), kind=kCustom, calls=_pop_op_wide_const.1
  get-tuple-element.78.56 = f32[2] get-tuple-element(batch-norm-grad.78.54), index=1
  multiply.78.65 = f32[2] multiply(call.2, get-tuple-element.78.56)
  subtract.78.66 = f32[2] subtract(arg2.78.24, multiply.78.65)
  get-tuple-element.78.57 = f32[2] get-tuple-element(batch-norm-grad.78.54), index=2
  multiply.78.62 = f32[2] multiply(call.2, get-tuple-element.78.57)
  subtract.78.63 = f32[2] subtract(arg3.78.25, multiply.78.62)
  ROOT tuple.78.77 = (f32[], f32[1,1,2,2], f32[2], f32[2]) tuple(reduce.78.49, call, subtract.78.66, subtract.78.63)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  auto* entry_computation = module0->entry_computation();
  auto* arg0 = entry_computation->parameter_instruction(0);
  const auto* conv = arg0->users()[0]->opcode() == HloOpcode::kConvolution
                         ? arg0->users()[0]
                         : arg0->users()[1];
  const auto* conv_ip0 = conv->operand(0);
  const auto* conv_ip1 = conv->operand(1);
  const auto* bn_tr =
      conv->users()[0]->opcode() == HloOpcode::kBatchNormTraining
          ? conv->users()[0]
          : conv->users()[1];
  const auto* bn_ip1 = bn_tr->operand(1);
  const auto* bn_ip2 = bn_tr->operand(2);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{conv_ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{conv_ip1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  ASSERT_EQ(annotations.tensor_allocation_map.size(), 5);

  t = annotations.tensor_allocation_map.at(TensorLocation{bn_ip1, 0});
  EXPECT_EQ(t.tgt, bn_tr);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{bn_ip2, 0});
  EXPECT_EQ(t.tgt, bn_tr);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);

  // The add in the reduce computation can also have a layout on either
  // operand.
  HloInstruction* x = FindInstruction(module0, "x");
  HloInstruction* y = FindInstruction(module0, "y");
  if (annotations.tensor_allocation_map.find(TensorLocation{x, 0}) ==
      annotations.tensor_allocation_map.end()) {
    EXPECT_NE(annotations.tensor_allocation_map.find(TensorLocation{y, 0}),
              annotations.tensor_allocation_map.end());
  }
}

TEST_F(AllocationFinderTest, ForwardAllocationMultipleUsesOneTarget) {
  // In this test we check that arg2.36.24 still has a layout even though
  // it has two targets but only one is a layout sensitive target.
  std::string hlo = R"(
HloModule top
Sum-reduction48 {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

ENTRY top {
 arg0.36.22 = f32[1,4,4,2] parameter(0)
 arg1.36.23 = f32[1,1,2,2] parameter(1)
 convolution.36.29 = f32[1,4,4,2] convolution(arg0.36.22, arg1.36.23), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="vs/conv2d/Conv2D"}
 arg2.36.24 = f32[1,2] parameter(2)
 arg2.36.24_r = f32[2] reshape(arg2.36.24)
 arg3.36.25 = f32[1,2] parameter(3)
 arg3.36.25_r = f32[2] reshape(arg3.36.25)
 arg4.36.26 = f32[2] parameter(4)
 arg5.36.27 = f32[2] parameter(5)
 s0 = f32[1] slice(arg2.36.24_r), slice={[0:1]}
 s1 = f32[] reshape(s0)
 batch-norm-inference.36.31 = f32[1,4,4,2] batch-norm-inference(convolution.36.29, arg2.36.24_r, arg3.36.25_r, arg4.36.26, arg5.36.27), epsilon=0.001, feature_index=3, metadata={op_type="FusedBatchNorm" op_name="vs/batch_normalization/FusedBatchNorm"}
 ROOT reduce.78.49 = f32[2] reduce(batch-norm-inference.36.31, s1), dimensions={0, 1, 2}, to_apply=Sum-reduction48, metadata={op_type="Sum" op_name="Sum"}
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* reduce = root;
  const auto* bn = reduce->operand(0);
  const auto* conv = bn->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* reshape1 = bn->operand(1);
  const auto* reshape2 = bn->operand(2);
  const auto* ip2 = reshape1->operand(0);
  const auto* ip3 = reshape2->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  unsigned num_succesful_runs = 0;
  while (fwd_finder.Run(module0).ValueOrDie()) {
    num_succesful_runs++;
  }

  // Depending on the order we either expect this to be executed successfully
  // 1 or 2 times.
  EXPECT_TRUE(num_succesful_runs == 1 || num_succesful_runs == 2);

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 5);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape1);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip3, 0});
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape2);

  // The add in the reduce computation can also have a layout on either
  // operand.
  HloInstruction* x = FindInstruction(module0, "x");
  HloInstruction* y = FindInstruction(module0, "y");
  if (annotations.tensor_allocation_map.find(TensorLocation{x, 0}) ==
      annotations.tensor_allocation_map.end()) {
    EXPECT_NE(annotations.tensor_allocation_map.find(TensorLocation{y, 0}),
              annotations.tensor_allocation_map.end());
  }
}

TEST_F(AllocationFinderTest,
       ForwardAllocationMultipleUsesMultipleTargetsSamePriority) {
  // In this test we check that arg2.36.24 and arg3.36.25 get a layout even
  // though they have multiple targets.
  std::string hlo = R"(
HloModule top
Sum-reduction48 {
  x = f32[2] parameter(0)
  y = f32[2] parameter(1)
  ROOT add = f32[2] add(x, y)
}

ENTRY top {
 arg0.36.22 = f32[1,4,4,2] parameter(0)
 arg1.36.23 = f32[1,1,2,2] parameter(1)
 convolution.36.29 = f32[1,4,4,2] convolution(arg0.36.22, arg1.36.23), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="vs/conv2d/Conv2D"}
 arg2.36.24 = f32[1,2] parameter(2)
 arg2.36.24_r = f32[2] reshape(arg2.36.24)
 arg3.36.25 = f32[1,2] parameter(3)
 arg3.36.25_r = f32[2] reshape(arg3.36.25)
 arg4.36.26 = f32[2] parameter(4)
 arg5.36.27 = f32[2] parameter(5)
 arg6.36.28 = f32[1,4,4,2] parameter(6)
 batch-norm-inference.36.31 = f32[1,4,4,2] batch-norm-inference(convolution.36.29, arg2.36.24_r, arg3.36.25_r, arg4.36.26, arg5.36.27), epsilon=0.001, feature_index=3, metadata={op_type="FusedBatchNorm" op_name="vs/batch_normalization/FusedBatchNorm"}
 batch-norm-inference.36.32 = f32[1,4,4,2] batch-norm-inference(convolution.36.29, arg2.36.24_r, arg3.36.25_r, arg4.36.26, arg5.36.27), epsilon=0.001, feature_index=3, metadata={op_type="FusedBatchNorm" op_name="vs/batch_normalization/FusedBatchNorm"}
 ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(batch-norm-inference.36.31, batch-norm-inference.36.32)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* bn1 = root->operand(0);
  const auto* bn2 = root->operand(1);
  const auto* conv = bn1->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* reshape1 = bn1->operand(1);
  const auto* reshape2 = bn1->operand(2);
  const auto* ip2 = reshape1->operand(0);
  const auto* ip3 = reshape2->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  unsigned num_succesful_runs = 0;
  ForwardAllocation fwd_finder(annotations);
  while (fwd_finder.Run(module0).ValueOrDie()) {
    num_succesful_runs++;
  }

  // Depending on the order we either expect this to be executed successfully
  // 1 or 2 times.
  EXPECT_TRUE(num_succesful_runs == 1 || num_succesful_runs == 2);

  // We have added two new entires for the layer norms.
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 5);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  auto target_bn = t.tgt == bn1 ? bn1 : bn2;
  // It was allocated for one of the batch norms.
  EXPECT_EQ(t.tgt, target_bn);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape1);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip3, 0});
  // It was allocated for same batch norm due to control dependencies.
  EXPECT_EQ(t.tgt, target_bn);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape2);

  // The add in the reduce computation can also have a layout on either
  // operand.
  HloInstruction* x = FindInstruction(module0, "x");
  HloInstruction* y = FindInstruction(module0, "y");
  if (annotations.tensor_allocation_map.find(TensorLocation{x, 0}) ==
      annotations.tensor_allocation_map.end()) {
    EXPECT_NE(annotations.tensor_allocation_map.find(TensorLocation{y, 0}),
              annotations.tensor_allocation_map.end());
  }
}

TEST_F(AllocationFinderTest,
       ForwardAllocationMultipleUsesMultipleTargetsDifferentPriority) {
  // In this test we check that arg2.36.24 and arg3.36.25 get a layout even
  // though they have multiple targets - layer norms have higher priority than
  // elementwise ops.
  std::string hlo = R"(
HloModule top
Sum-reduction48 {
  x = f32[2] parameter(0)
  y = f32[2] parameter(1)
  ROOT add = f32[2] add(x, y)
}

ENTRY top {
 arg0.36.22 = f32[1,4,4,2] parameter(0)
 arg1.36.23 = f32[1,1,2,2] parameter(1)
 convolution.36.29 = f32[1,4,4,2] convolution(arg0.36.22, arg1.36.23), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="vs/conv2d/Conv2D"}
 arg2.36.24 = f32[1,2] parameter(2)
 arg2.36.24_r = f32[2] reshape(arg2.36.24)
 arg3.36.25 = f32[1,2] parameter(3)
 arg3.36.25_r = f32[2] reshape(arg3.36.25)
 arg4.36.26 = f32[2] parameter(4)
 arg5.36.27 = f32[2] parameter(5)
 batch-norm-inference.36.31 = f32[1,4,4,2] batch-norm-inference(convolution.36.29, arg2.36.24_r, arg3.36.25_r, arg4.36.26, arg5.36.27), epsilon=0.001, feature_index=3, metadata={op_type="FusedBatchNorm" op_name="vs/batch_normalization/FusedBatchNorm"}
 add = f32[2] add(arg2.36.24_r, arg3.36.25_r)
 ROOT tuple = (f32[1,4,4,2], f32[2]) tuple(batch-norm-inference.36.31, add)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* bn = root->operand(0);
  const auto* conv = bn->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* reshape1 = bn->operand(1);
  const auto* reshape2 = bn->operand(2);
  const auto* ip2 = reshape1->operand(0);
  const auto* ip3 = reshape2->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  unsigned num_succesful_runs = 0;
  ForwardAllocation fwd_finder(annotations);
  while (fwd_finder.Run(module0).ValueOrDie()) {
    num_succesful_runs++;
  }

  // Depending on the order we either expect this to be executed successfully
  // 1 or 2 times.
  EXPECT_TRUE(num_succesful_runs == 1 || num_succesful_runs == 2);

  // We have added two new entires for the layer norms.
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 5);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  // Layer norm has priority over elementwise ops.
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape1);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip3, 0});
  // Layer norm has priority over elementwise ops.
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape2);

  // The add in the reduce computation can also have a layout on either
  // operand.
  HloInstruction* x = FindInstruction(module0, "x");
  HloInstruction* y = FindInstruction(module0, "y");
  if (annotations.tensor_allocation_map.find(TensorLocation{x, 0}) ==
      annotations.tensor_allocation_map.end()) {
    EXPECT_NE(annotations.tensor_allocation_map.find(TensorLocation{y, 0}),
              annotations.tensor_allocation_map.end());
  }
}

TEST_F(AllocationFinderTest, ForwardAllocationElementwiseGetsALayout) {
  // Check the layout is forwarded to the element wise op argument.
  std::string hlo = R"(
HloModule top

_pop_op_conv_biasadd {
  arg_0 = f16[1,16,16,4] parameter(0)
  arg_1 = f16[4] parameter(1)
  bcast = f16[1,16,16,4] broadcast(arg_1), dimensions={3}
  ROOT add = f16[1,16,16,4] add(arg_0, bcast)
}

ENTRY c1 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[3,3,2,4] parameter(1)
  p2 = f16[2,2] parameter(2)
  p2_r = f16[4] reshape(p2)

  conv = f16[1,16,16,4] convolution(p0, p1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  call = f16[1,16,16,4] fusion(conv, p2_r), kind=kCustom, calls=_pop_op_conv_biasadd
  p3 = f16[1,16,16,4] parameter(3)
  ROOT add = f16[1,16,16,4] add(p3, call)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* add = root;
  const auto* ip3 = add->operand(0);
  const auto* call = add->operand(1);
  const auto* conv = call->operand(0);
  const auto* ip2_r = call->operand(1);
  const auto* ip2 = ip2_r->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* ip0 = conv->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  unsigned num_succesful_runs = 0;
  while (fwd_finder.Run(module0).ValueOrDie()) {
    num_succesful_runs++;
  }

  // Depending on the order we either expect this to be executed successfully
  // 1 or 2 times.
  EXPECT_TRUE(num_succesful_runs == 1 || num_succesful_runs == 2);

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, call);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], ip2_r);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip3, 0});
  EXPECT_EQ(t.tgt, add);
  EXPECT_EQ(t.input_index, 0);
  EXPECT_EQ(t.layout, call);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, ForwardAllocationDontLookThroughCasts) {
  // Check the layout is not forwarded to the element wise op argument as it's
  // casted.
  std::string hlo = R"(
HloModule top

_pop_op_conv_biasadd {
  arg_0 = f16[1,16,16,4] parameter(0)
  arg_1 = f16[4] parameter(1)
  bcast = f16[1,16,16,4] broadcast(arg_1), dimensions={3}
  ROOT add = f16[1,16,16,4] add(arg_0, bcast)
}

ENTRY c1 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[3,3,2,4] parameter(1)
  p2 = f16[2,2] parameter(2)
  p2_r = f16[4] reshape(p2)

  conv = f16[1,16,16,4] convolution(p0, p1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  call = f16[1,16,16,4] fusion(conv, p2_r), kind=kCustom, calls=_pop_op_conv_biasadd
  p3 = f32[1,16,16,4] parameter(3)
  p3.c = f16[1,16,16,4] convert(p3)
  ROOT add = f16[1,16,16,4] add(p3.c, call)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* add = root;
  const auto* call = add->operand(1);
  const auto* conv = call->operand(0);
  const auto* ip2_r = call->operand(1);
  const auto* ip2 = ip2_r->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* ip0 = conv->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  unsigned num_succesful_runs = 0;
  while (fwd_finder.Run(module0).ValueOrDie()) {
    num_succesful_runs++;
  }

  // We expect this to be executed successfully 1 time.
  EXPECT_EQ(num_succesful_runs, 1);

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, call);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], ip2_r);
}

TEST_F(AllocationFinderTest, ForwardAllocationElementwiseGetsALayoutWithGTE) {
  // Check the layout is forwarded to the element wise op argument with a GTE.
  std::string hlo = R"(
HloModule top
ENTRY top {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,1,2,2] parameter(1)
  convolution = f32[1,4,4,2] convolution(arg0, arg1), window={size=1x1}, dim_labels=b01f_01io->b01f
  arg2 = f32[2] parameter(2)
  arg3 = f32[2] parameter(3)
  batch-norm-training = (f32[1,4,4,2], f32[2], f32[2]) batch-norm-training(convolution, arg2, arg3), epsilon=0.001, feature_index=3
  get-tuple-element = f32[2] get-tuple-element(batch-norm-training), index=2
  arg4 = f32[2] parameter(4)
  ROOT subtract = f32[2] subtract(get-tuple-element, arg4)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* subtract = root;
  const auto* ip4 = subtract->operand(1);
  const auto* gte = subtract->operand(0);
  const auto* bn = gte->operand(0);
  const auto* conv = bn->operand(0);
  const auto* ip3 = bn->operand(2);
  const auto* ip2 = bn->operand(1);
  const auto* ip1 = conv->operand(1);
  const auto* ip0 = conv->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  unsigned num_succesful_runs = 0;
  while (fwd_finder.Run(module0).ValueOrDie()) {
    num_succesful_runs++;
  }

  // Depending on the order we either expect this to be executed successfully
  // 1 or 2 times.
  EXPECT_TRUE(num_succesful_runs == 1 || num_succesful_runs == 2);

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 5);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip3, 0});
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip4, 0});
  EXPECT_EQ(t.tgt, subtract);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, gte);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, ForwardAllocationCustomPoplarOp) {
  // Check that the layout gets forwarded to a custom op.
  std::string hlo = R"(
HloModule top
ENTRY top {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,1,2,2] parameter(1)
  convolution = f32[1,4,4,2] convolution(arg0, arg1), window={size=1x1}, dim_labels=b01f_01io->b01f
  arg2 = f32[2] parameter(2)
  arg3 = f32[2] parameter(3)
  ROOT cc = (f32[1,4,4,2], f32[2], f32[2]) custom-call(convolution, arg2, arg3), custom_call_target="GroupNormTraining", backend_config="{\"num_groups\":1,\"epsilon\":0.001,\"feature_index\":3,\"strided_channel_grouping\":0}\n"
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());

  const auto* custom_op = module0->entry_computation()->root_instruction();
  const auto* conv = custom_op->operand(0);
  const auto* ip2 = custom_op->operand(1);
  const auto* ip3 = custom_op->operand(2);
  const auto* ip1 = conv->operand(1);
  const auto* ip0 = conv->operand(0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  unsigned num_succesful_runs = 0;
  while (fwd_finder.Run(module0).ValueOrDie()) {
    num_succesful_runs++;
  }

  // Depending on the order we either expect this to be executed successfully
  // 1 or 2 times.
  EXPECT_TRUE(num_succesful_runs == 1 || num_succesful_runs == 2);

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, custom_op);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip3, 0});
  EXPECT_EQ(t.tgt, custom_op);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, FindInfeedAllocation) {
  // Check that allocation finder works with infeed non-tuple.
  std::string hlo = R"(
HloModule top

Sum-reduction.7 {
  x.8 = f32[] parameter(0)
  y.9 = f32[] parameter(1)
  ROOT add.10 = f32[] add(x.8, y.9)
}

_body  {
  arg_tuple.0 = (s32[], f32[], f32[1,1,2,2]) parameter(0)
  get-tuple-element.3 = s32[] get-tuple-element(arg_tuple.0), index=0
  get-tuple-element.4 = f32[1,1,2,2] get-tuple-element(arg_tuple.0), index=2
  constant.6 = f32[] constant(0)
  after-all = token[] after-all()
  infeed = ((f32[2,4,4,2]), token[]) infeed(after-all), infeed_config="140121807314576"
  get-tuple-element.5 = (f32[2,4,4,2]) get-tuple-element(infeed), index=0
  get-tuple-element.6 = f32[2,4,4,2] get-tuple-element(get-tuple-element.5), index=0
  convolution = f32[2,4,4,2] convolution(get-tuple-element.6, get-tuple-element.4), window={size=1x1}, dim_labels=b01f_01io->b01f
  reduce = f32[] reduce(convolution, constant.6), dimensions={0,1,2,3}, to_apply=Sum-reduction.7
  ROOT tuple.1 = (s32[], f32[], f32[1,1,2,2]) tuple(get-tuple-element.3, reduce, get-tuple-element.4)
}

ENTRY top {
  constant.7 = s32[] constant(100)
  constant.5 = f32[] constant(0)
  arg0.1 = f32[1,1,2,2] parameter(0)
  tuple.6.clone = (s32[], f32[], f32[1,1,2,2]) tuple(constant.7, constant.5, arg0.1)
  call = (s32[], f32[], f32[1,1,2,2]) call(tuple.6.clone), to_apply=_body, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
  ROOT get-tuple-element.45 = f32[] get-tuple-element(call), index=1
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* repeat_loop = root->operand(0);
  const auto* input_tuple = repeat_loop->operand(0);
  const auto* ip_weights = repeat_loop->operand(0)->operand(2);
  const auto* repeat_body = repeat_loop->to_apply();
  const auto* repeat_root = repeat_body->root_instruction();
  const auto* reduce = repeat_root->operand(1);
  const auto* convolution = reduce->operand(0);
  const auto* conv_input = convolution->operand(0);
  const auto* conv_weights = convolution->operand(1);
  const auto* infeed_gte = conv_input->operand(0);
  const auto* infeed_tuple = infeed_gte->operand(0);
  const auto* repeat_tuple = conv_weights->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  auto t =
      annotations.tensor_allocation_map.at(TensorLocation{infeed_tuple, 0});
  EXPECT_EQ(t.tgt, convolution);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip_weights, 0});
  EXPECT_EQ(t.tgt, convolution);
  EXPECT_EQ(t.input_index, 1);

  t = annotations.tensor_allocation_map.at(TensorLocation{repeat_tuple, 2});
  EXPECT_EQ(t.tgt, convolution);
  EXPECT_EQ(t.input_index, 1);
}

TEST_F(AllocationFinderTest, FindInfeedAllocationTuple) {
  // Check that allocation finder works with infeed tuple.
  std::string hlo = R"(
HloModule top

Sum-reduction.6 {
  x.7 = f32[] parameter(0)
  y.8 = f32[] parameter(1)
  ROOT add.9 = f32[] add(x.7, y.8)
}

_body {
  arg_tuple.0 = (s32[], f32[]) parameter(0)
  get-tuple-element.2 = s32[] get-tuple-element(arg_tuple.0), index=0
  constant.5 = f32[] constant(0)
  after-all = token[] after-all()
  infeed = ((f32[2,4,4,2], f32[1,1,2,2]), token[]) infeed(after-all), infeed_config="140227418928528"
  get-tuple-element.3 = (f32[2,4,4,2], f32[1,1,2,2]) get-tuple-element(infeed), index=0
  get-tuple-element.4 = f32[2,4,4,2] get-tuple-element(get-tuple-element.3), index=0
  get-tuple-element.5 = f32[1,1,2,2] get-tuple-element(get-tuple-element.3), index=1
  convolution = f32[2,4,4,2] convolution(get-tuple-element.4, get-tuple-element.5), window={size=1x1}, dim_labels=b01f_01io->b01f
  reduce = f32[] reduce(convolution, constant.5), dimensions={0,1,2,3}, to_apply=Sum-reduction.6
  ROOT tuple.1 = (s32[], f32[]) tuple(get-tuple-element.2, f32[] reduce)
}

ENTRY top {
  constant.7 = s32[] constant(100)
  constant.4 = f32[] constant(0)
  tuple.5.clone = (s32[], f32[]) tuple(constant.7, constant.4)
  call = (s32[], f32[]) call(tuple.5.clone), to_apply=_body, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
  ROOT get-tuple-element.41 = f32[] get-tuple-element(call), index=1
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* repeat_loop = root->operand(0);
  const auto* repeat_body = repeat_loop->to_apply();
  const auto* repeat_root = repeat_body->root_instruction();
  const auto* reduce = repeat_root->operand(1);
  const auto* convolution = reduce->operand(0);
  const auto* conv_input = convolution->operand(0);
  const auto* conv_weights = convolution->operand(1);
  const auto* infeed_gte = conv_input->operand(0);
  const auto* infeed_tuple = infeed_gte->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t =
      annotations.tensor_allocation_map.at(TensorLocation{infeed_tuple, 0});
  EXPECT_EQ(t.tgt, convolution);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{infeed_tuple, 1});
  EXPECT_EQ(t.tgt, convolution);
  EXPECT_EQ(t.input_index, 1);
}

TEST_F(AllocationFinderTest, FindRecvFromHostAllocation) {
  // Check that allocation finder works with tensors from RecvFromHost.
  std::string hlo = R"(
HloModule top

ENTRY %top (arg: f32[1,1,2,2]) -> f32[1,1,2,2] {
  %recv-from-host = f32[1,1,2,2]{3,2,1,0} custom-call(), custom_call_target="RecvFromHost", backend_config="{\"rendezvous_key\":\"key\"}"
  %arg = f32[1,1,2,2]{3,2,1,0} parameter(0), parameter_replication={false}
  ROOT %convolution = f32[1,1,2,2]{3,2,1,0} convolution(f32[1,1,2,2]{3,2,1,0} %recv-from-host, f32[1,1,2,2]{3,2,1,0} %arg), window={size=1x1}, dim_labels=b01f_01io->b01f
}

)";

  auto config = GetModuleConfigForTest();
  auto status_module = ParseAndReturnVerifiedModule(hlo);
  ASSERT_TRUE(status_module.ok()) << status_module.status().error_message();

  auto* module = status_module.ValueOrDie().get();
  ASSERT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());

  const auto* convolution = module->entry_computation()->root_instruction();
  const auto* recv_from_host = convolution->operand(0);

  CompilerAnnotations annotations(module);
  ASSERT_TRUE(AllocationFinder(annotations).Run(module).ValueOrDie());

  // Will have both of the convolution arguments (input and weights)
  ASSERT_EQ(annotations.tensor_allocation_map.size(), 2);

  // The RecvFromHost should have the convolution as allocation target
  auto t =
      annotations.tensor_allocation_map.at(TensorLocation{recv_from_host, 0});
  ASSERT_EQ(t.tgt, convolution);
  ASSERT_EQ(t.input_index, 0);
}

TEST_F(AllocationFinderTest, FindRecvFromHostForwardAllocation) {
  std::string hlo = R"(
HloModule top

ENTRY top {
 p0 = f32[1,4,4,8] parameter(0)
 p1 = f32[1,1,8,8] parameter(1)
 conv = f32[1,4,4,8] convolution(p0, p1), window={size=1x1}, dim_labels=b01f_01io->b01f
 p2 = f32[1,8] parameter(2)
 p2_r = f32[8] reshape(p2)
 c0 = f32[] constant(1)
 recv-from-host = f32[8] custom-call(), custom_call_target="RecvFromHost", backend_config="{\"rendezvous_key\":\"key\"}"
 p4 = f32[8] parameter(3)
 p5 = f32[8] parameter(4)
 ROOT batch-norm-inf = f32[1,4,4,8] batch-norm-inference(conv, p2_r, recv-from-host, p4, p5), epsilon=0.001, feature_index=3
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  EXPECT_TRUE(CustomOpReplacer().Run(module0).ValueOrDie());

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* bn = root;
  const auto* conv = bn->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* p2_r = bn->operand(1);
  const auto* ip2 = p2_r->operand(0);
  const auto* recv_from_host = bn->operand(2);

  CompilerAnnotations annotations(module0);

  InplaceFinder inplace_finder;
  ASSERT_TRUE(inplace_finder.Run(module0).ValueOrDie());

  AllocationFinder finder(annotations);
  ASSERT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  ASSERT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  ASSERT_EQ(t.tgt, conv);
  ASSERT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  ASSERT_EQ(t.tgt, conv);
  ASSERT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  ASSERT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  ASSERT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  ASSERT_EQ(t.tgt, bn);
  ASSERT_EQ(t.input_index, 1);
  ASSERT_EQ(t.layout, conv);
  ASSERT_EQ(t.layout_output_idx, 0);
  ASSERT_EQ(t.forward_path.size(), 0);
  ASSERT_EQ(t.backward_path.size(), 1);
  ASSERT_EQ(t.backward_path[0], p2_r);

  t = annotations.tensor_allocation_map.at(TensorLocation{recv_from_host, 0});
  ASSERT_EQ(t.tgt, bn);
  ASSERT_EQ(t.input_index, 2);
  ASSERT_EQ(t.layout, conv);
  ASSERT_EQ(t.layout_output_idx, 0);
  ASSERT_EQ(t.forward_path.size(), 0);
  ASSERT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, InputTupleBiasAdd) {
  std::string hlo = R"(
HloModule top

 _pop_op_matmul_biasadd {
   arg_1 = f32[2] parameter(1)
   broadcast.12.7.clone = f32[2,2] broadcast(arg_1), dimensions={1}
   arg_0 = f32[2,2] parameter(0)
   ROOT add.12.8.clone = f32[2,2] add(arg_0, broadcast.12.7.clone)
 }

 ENTRY c {
   arg0 = (f32[2,2], f32[2,2], f32[2]) parameter(0)
   gte0 = f32[2,2] get-tuple-element(arg0), index=0
   gte1 = f32[2,2] get-tuple-element(arg0), index=1
   dot.12.6 = f32[2,2] dot(gte0, gte1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
   gte2 = f32[2] get-tuple-element(arg0), index=2
   ROOT call = f32[2,2] fusion(dot.12.6, gte2), kind=kCustom, calls=_pop_op_matmul_biasadd
 }

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* call = root;
  const auto* dot = call->operand(0);
  const auto* gte0 = dot->operand(0);
  const auto* gte1 = dot->operand(1);
  const auto* gte2 = call->operand(1);
  const auto* ip_tuple = gte0->operand(0);
  EXPECT_EQ(ip_tuple, gte1->operand(0));
  EXPECT_EQ(ip_tuple, gte2->operand(0));

  CompilerAnnotations annotations(module0);

  InplaceFinder inplace_finder;
  EXPECT_TRUE(inplace_finder.Run(module0).ValueOrDie());

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the dot parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip_tuple, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip_tuple, 1});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  t = annotations.tensor_allocation_map.at(TensorLocation{gte2, 0});
  EXPECT_EQ(t.tgt, call);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, dot);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, InputTupleInfeedBiasAdd) {
  std::string hlo = R"(
HloModule top

 _pop_op_matmul_biasadd {
   arg_1 = f32[2] parameter(1)
   broadcast.12.7.clone = f32[2,2] broadcast(arg_1), dimensions={1}
   arg_0 = f32[2,2] parameter(0)
   ROOT add.12.8.clone = f32[2,2] add(arg_0, broadcast.12.7.clone)
 }

 ENTRY c {
   after-all = token[] after-all()
   infeed = ((f32[2,2], f32[2,2], f32[2]), token[]) infeed(after-all), infeed_config="140227418928528"
   arg0 = (f32[2,2], f32[2,2], f32[2]) get-tuple-element(infeed), index=0
   gte0 = f32[2,2] get-tuple-element(arg0), index=0
   gte1 = f32[2,2] get-tuple-element(arg0), index=1
   dot.12.6 = f32[2,2] dot(gte0, gte1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
   gte2 = f32[2] get-tuple-element(arg0), index=2
   ROOT call = f32[2,2] fusion(dot.12.6, gte2), kind=kCustom, calls=_pop_op_matmul_biasadd
 }

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* call = root;
  const auto* dot = call->operand(0);
  const auto* gte0 = dot->operand(0);
  const auto* gte1 = dot->operand(1);
  const auto* gte2 = call->operand(1);
  const auto* ip_tuple = gte0->operand(0);
  EXPECT_EQ(ip_tuple, gte1->operand(0));
  EXPECT_EQ(ip_tuple, gte2->operand(0));
  const auto* infeed = ip_tuple->operand(0);

  CompilerAnnotations annotations(module0);

  InplaceFinder inplace_finder;
  EXPECT_TRUE(inplace_finder.Run(module0).ValueOrDie());

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the dot parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);
  auto t = annotations.tensor_allocation_map.at(TensorLocation{infeed, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{infeed, 1});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  t = annotations.tensor_allocation_map.at(TensorLocation{gte2, 0});
  EXPECT_EQ(t.tgt, call);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, dot);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, NestedInputTupleBatchNormInfParamsWithPath) {
  std::string hlo = R"(
HloModule top

ENTRY top {
 arg0 = (f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) parameter(0)
 gte0 = f32[1,4,4,2] get-tuple-element(arg0), index=0
 gte1 = f32[1,1,2,2] get-tuple-element(arg0), index=1
 convolution.36.29 = f32[1,4,4,2] convolution(gte0, gte1), window={size=1x1}, dim_labels=b01f_01io->b01f
 gte2 = (f32[1,2], f32[1,2]) get-tuple-element(arg0), index=2
 gte2.0 = f32[1,2] get-tuple-element(gte2), index=0
 gte2.0_r = f32[2] reshape(gte2.0)
 gte2.1 = f32[1,2] get-tuple-element(gte2), index=1
 gte2.1_r = f32[2] reshape(gte2.1)
 gte3 = f32[2] get-tuple-element(arg0), index=3
 gte4 = f32[2] get-tuple-element(arg0), index=4
 ROOT batch-norm-inference.36.31 = f32[1,4,4,2] batch-norm-inference(convolution.36.29, gte2.0_r, gte2.1_r, gte3, gte4), epsilon=0.001, feature_index=3
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* bn = root;
  const auto* conv = bn->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* reshape1 = bn->operand(1);
  const auto* reshape2 = bn->operand(2);
  const auto* ip2_0 = reshape1->operand(0);
  const auto* ip2_1 = reshape2->operand(0);
  const auto* nested_tuple = ip2_0->operand(0);
  CHECK_EQ(nested_tuple, ip2_1->operand(0));
  const auto* arg_tuple = ip0->operand(0);
  CHECK_EQ(arg_tuple, ip1->operand(0));
  CHECK_EQ(arg_tuple, nested_tuple->operand(0));

  CompilerAnnotations annotations(module0);

  InplaceFinder inplace_finder;
  EXPECT_TRUE(inplace_finder.Run(module0).ValueOrDie());

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{arg_tuple, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{arg_tuple, 1});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2_0, 0});
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape1);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2_1, 0});
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape2);
}

TEST_F(AllocationFinderTest, ForwardAllocationTupleHasTupleSharding) {
  // Check that the layout gets forwarded to a custom op.
  std::string hlo = R"(
HloModule top
ENTRY top {
  arg0 = (f32[1,4,4,2], f32[2], f32[2]) parameter(0),
      sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}
  gte0 = f32[1,4,4,2] get-tuple-element(arg0), index=0,
      sharding={maximal device=0}
  gte1 = f32[2] get-tuple-element(arg0), index=1,
      sharding={maximal device=0}
  gte2 = f32[2] get-tuple-element(arg0), index=2,
      sharding={maximal device=0}
  ROOT cc = (f32[1,4,4,2], f32[2], f32[2]) custom-call(gte0, gte1, gte2),
       custom_call_target="GroupNormTraining",
       backend_config="{\"num_groups\":1,\"epsilon\":0.001,\"feature_index\":3,\"strided_channel_grouping\":0}\n"}
)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());

  const auto* custom_op = module0->entry_computation()->root_instruction();
  const auto* ip0 = custom_op->operand(0);
  const auto* ip1 = custom_op->operand(1);
  const auto* ip2 = custom_op->operand(2);
  const auto* arg0 = ip0->operand(0);

  InplaceFinder inplace_finder;
  EXPECT_TRUE(inplace_finder.Run(module0).ValueOrDie());

  ForwardAllocation fwd_finder(annotations);
  unsigned num_succesful_runs = 0;
  while (fwd_finder.Run(module0).ValueOrDie()) {
    num_succesful_runs++;
  }

  // Depending on the order we either expect this to be executed successfully
  // 1 or 2 times.
  EXPECT_TRUE(num_succesful_runs == 1 || num_succesful_runs == 2);

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, custom_op);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, ip0);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, custom_op);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, ip0);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, ForwardAllocationTupleHasNonTupleSharding) {
  // Check that the layout gets forwarded to a custom op.
  std::string hlo = R"(
HloModule top
ENTRY top {
  arg0 = (f32[1,4,4,2], f32[2], f32[2]) parameter(0),
      sharding={maximal device=0}
  gte0 = f32[1,4,4,2] get-tuple-element(arg0), index=0,
      sharding={maximal device=0}
  gte1 = f32[2] get-tuple-element(arg0), index=1,
      sharding={maximal device=0}
  gte2 = f32[2] get-tuple-element(arg0), index=2,
      sharding={maximal device=0}
  ROOT cc = (f32[1,4,4,2], f32[2], f32[2]) custom-call(gte0, gte1, gte2),
       custom_call_target="GroupNormTraining",
       backend_config="{\"num_groups\":1,\"epsilon\":0.001,\"feature_index\":3,\"strided_channel_grouping\":0}\n"
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());

  const auto* custom_op = module0->entry_computation()->root_instruction();
  const auto* ip0 = custom_op->operand(0);
  const auto* ip1 = custom_op->operand(1);
  const auto* ip2 = custom_op->operand(2);
  const auto* arg0 = ip0->operand(0);

  InplaceFinder inplace_finder;
  EXPECT_TRUE(inplace_finder.Run(module0).ValueOrDie());

  ForwardAllocation fwd_finder(annotations);
  unsigned num_succesful_runs = 0;
  while (fwd_finder.Run(module0).ValueOrDie()) {
    num_succesful_runs++;
  }

  // Depending on the order we either expect this to be executed successfully
  // 1 or 2 times.
  EXPECT_TRUE(num_succesful_runs == 1 || num_succesful_runs == 2);

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, custom_op);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, ip0);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, custom_op);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, ip0);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, ForwardAllocationTupleHasTupleShardingDoesnMatch) {
  // Check that the layout gets forwarded to a custom op.
  std::string hlo = R"(
HloModule top
ENTRY top {
  arg0 = (f32[1,4,4,2], f32[2], f32[2]) parameter(0),
      sharding={{maximal device=0}, {maximal device=0}, {maximal device=1}}
  gte0 = f32[1,4,4,2] get-tuple-element(arg0), index=0,
      sharding={maximal device=0}
  gte1 = f32[2] get-tuple-element(arg0), index=1,
      sharding={maximal device=0}
  gte2 = f32[2] get-tuple-element(arg0), index=2,
      sharding={maximal device=0}
  ROOT cc = (f32[1,4,4,2], f32[2], f32[2]) custom-call(gte0, gte1, gte2),
       custom_call_target="GroupNormTraining",
       backend_config="{\"num_groups\":1,\"epsilon\":0.001,\"feature_index\":3,\"strided_channel_grouping\":0}\n"
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());

  InplaceFinder inplace_finder;
  EXPECT_TRUE(inplace_finder.Run(module0).ValueOrDie());

  ForwardAllocation fwd_finder(annotations);
  EXPECT_FALSE(fwd_finder.Run(module0).ValueOrDie());
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 0);
}

// Check cast for find path
TEST_F(AllocationFinderTest, AllocationsWithCast1) {
  std::string hlo = R"(
HloModule top

ENTRY cast1 {
  %p0 = f32[1,16,16,2] parameter(0)
  %p1 = f16[1,16,16,2] parameter(1)
  %p2 = f16[3,3,2,4] parameter(2)
  %add = f16[1,16,16,2] add(%p0, %p1)
  %p3 = f16[1,16,16,2] convert(%p0)
  %conv = f16[1,16,16,4] convolution(%p3, %p2), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  ROOT %t = (f16[1,16,16,4], f16[1,16,16,2]) tuple(%conv, %add)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* conv = root->operand(0);
  const auto* ip0 = conv->operand(0)->operand(0);
  const auto* ip2 = conv->operand(1);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

// Check cast for find path
TEST_F(AllocationFinderTest, AllocationsWithCast2) {
  std::string hlo = R"(
HloModule top

ENTRY cast2 {
  %a = f16[1,16,16,2] parameter(0)
  %b = f32[3,3,2,4] parameter(1)
  %c_cast = f16[3,3,2,4] convert(%b)
  %d_conv = f16[1,16,16,4] convolution(%a, %c_cast), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  %e = f32[1,16,16,4] parameter(2)
  %f_cast = f16[1,16,16,4] convert(%e)
  %g_add = f16[1,16,16,4] add(%d_conv, %f_cast)
  ROOT %t = (f16[1,16,16,4]) tuple(%g_add)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* ig_add = root->operand(0);
  const auto* id_conv = ig_add->operand(0);
  const auto* if_cast = ig_add->operand(1);
  const auto* ie = if_cast->operand(0);
  const auto* ic_cast = id_conv->operand(1);
  const auto* ib = ic_cast->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ie, 0});
  EXPECT_EQ(t.tgt, ig_add);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], if_cast);
  EXPECT_EQ(t.permutation, absl::nullopt);
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{ib, 0});
  EXPECT_EQ(t.tgt, id_conv);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], ic_cast);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

TEST_F(AllocationFinderTest, AllocationsWithCast3) {
  // Check the layout is forwarded to the element wise op argument with a GTE.
  std::string hlo = R"(
HloModule top
ENTRY cast3 (arg0.78.22: f32[1,4,4,2], arg1: f32[1,1,2,2], arg2: f32[2], arg3: f32[2], arg3: f32[2]) -> f32[2] {
  %arg0 = f32[1,4,4,2] parameter(0)
  %arg1 = f32[1,1,2,2] parameter(1)
  %convolution = f32[1,4,4,2] convolution(f32[1,4,4,2] %arg0, f32[1,1,2,2] %arg1), window={size=1x1}, dim_labels=b01f_01io->b01f
  %arg2 = f32[2] parameter(2)
  %arg3 = f32[2] parameter(3)
  %batch-norm-training = (f32[1,4,4,2], f32[2], f32[2]) batch-norm-training(f32[1,4,4,2] %convolution, f32[2] %arg2, f32[2] %arg3), epsilon=0.001, feature_index=3
  %get-tuple-element = f32[2] get-tuple-element((f32[1,4,4,2], f32[2], f32[2]) %batch-norm-training), index=2
  %arg4 = f16[2] parameter(4)
  %cast_arg4 = f32[2] convert(%arg4)
  ROOT %subtract = f32[2] subtract(%get-tuple-element, %cast_arg4)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* subtract = root;
  const auto* cast_arg4 = subtract->operand(1);
  const auto* p4 = cast_arg4->operand(0);
  const auto* gte = subtract->operand(0);
  const auto* bn = gte->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  ForwardAllocation fwd_finder(annotations);
  unsigned num_succesful_runs = 0;
  while (fwd_finder.Run(module0).ValueOrDie()) {
    num_succesful_runs++;
  }

  // Depending on the order we either expect this to be executed successfully
  // 1 or 2 times.
  EXPECT_TRUE(num_succesful_runs == 1 || num_succesful_runs == 2);
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 5);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{p4, 0});
  EXPECT_EQ(t.tgt, subtract);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, gte);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], cast_arg4);
  EXPECT_EQ(t.permutation, absl::nullopt);
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

TEST_F(AllocationFinderTest, AllocationsWithCast4) {
  std::string hlo = R"(
HloModule top

ENTRY cast4 {
  %a = f16[1,16,16,2] parameter(0)
  %b = f32[3,3,2,4] parameter(1)
  %c_cast = f16[3,3,2,4] convert(%b)
  %d_conv = f16[1,16,16,4] convolution(%a, %c_cast), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  %d_conv_cast = f32[1,16,16,4] convert(%d_conv)
  %e = f16[1,16,16,4] parameter(2)
  %f_cast = f32[1,16,16,4] convert(%e)
  %g_add = f32[1,16,16,4] add(%d_conv_cast, %f_cast)
  ROOT %t = (f32[1,16,16,4]) tuple(%g_add)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* g_add = root->operand(0);
  const auto* d_conv_cast = g_add->operand(0);
  const auto* f_cast = g_add->operand(1);
  const auto* e = f_cast->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{e, 0});
  EXPECT_EQ(t.tgt, g_add);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], f_cast);
  EXPECT_EQ(t.permutation, absl::nullopt);
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

TEST_F(AllocationFinderTest, AllocationsWithCast5) {
  std::string hlo = R"(
HloModule top

ENTRY cast4 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f32[1,16,16,2] parameter(1)
  p2 = f32[3,3,2,4] parameter(2)
  cast = f32[1,16,16,2] convert(p0)
  add = f32[1,16,16,2] add(cast, p1)
  ROOT conv = f32[1,16,16,4] convolution(add, p2), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const HloInstruction* p0 = FindInstruction(module0, "p0");
  const HloInstruction* p1 = FindInstruction(module0, "p1");
  const HloInstruction* p2 = FindInstruction(module0, "p2");
  const HloInstruction* cast = FindInstruction(module0, "cast");
  const HloInstruction* add = FindInstruction(module0, "add");
  const HloInstruction* conv = FindInstruction(module0, "conv");

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{p0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);
  EXPECT_EQ(t.backward_path[0], cast);
  EXPECT_EQ(t.backward_path[1], add);

  t = annotations.tensor_allocation_map.at(TensorLocation{p1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], add);

  t = annotations.tensor_allocation_map.at(TensorLocation{p2, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, AllocationsWithIpuRemapDeduce) {
  std::string hlo = R"(
HloModule top
ENTRY cast4 {
  %a = f16[1,16,16,2] parameter(0)
  %b = f32[3,3,2,4] parameter(1)
  %c_cast = f16[3,3,2,4] convert(%b)
  %remap_deduce = f16[3,3,2,4] custom-call(c_cast), custom_call_target="RemapDeduce"
  %d_conv = f16[1,16,16,4] convolution(%a, %remap_deduce), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  %d_conv_cast = f32[1,16,16,4] convert(%d_conv)
  %e = f16[1,16,16,4] parameter(2)
  %f_cast = f32[1,16,16,4] convert(%e)
  %g_add = f32[1,16,16,4] add(%d_conv_cast, %f_cast)
  ROOT %t = (f32[1,16,16,4]) tuple(%g_add)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* g_add = root->operand(0);
  const auto* d_conv_cast = g_add->operand(0);
  const auto* f_cast = g_add->operand(1);
  const auto* e = f_cast->operand(0);

  const auto* d_conv = d_conv_cast->operand(0);
  const auto* a = d_conv->operand(0);
  const auto* remap_deduce = d_conv->operand(1);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  auto allocation_map = annotations.tensor_allocation_map;
  EXPECT_EQ(allocation_map.size(), 3);
  EXPECT_EQ(allocation_map.count(TensorLocation{remap_deduce, 0}), 1);
  EXPECT_EQ(allocation_map.count(TensorLocation{a, 0}), 1);
  EXPECT_EQ(allocation_map.count(TensorLocation{e, 0}), 1);
}

TEST_F(AllocationFinderTest, AllocationsWithConcat) {
  std::string hlo = R"(
HloModule top

ENTRY main {
  %p0 = f32[1,16,16,2] parameter(0)
  %p1 = f16[1,16,16,2] parameter(1)
  %p2 = f16[3,3,2,2] parameter(2)
  %add = f16[1,16,16,2] add(%p0, %p1)
  %c0 = f16[3,3,2,4] concatenate(%p2, %p2), dimensions={3}
  %conv = f16[1,16,16,4] convolution(%p0, %c0), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  ROOT %t = (f16[1,16,16,4], f16[1,16,16,2]) tuple(%conv, %add)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* conv = root->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip2 = conv->operand(1)->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

TEST_F(AllocationFinderTest, AllocationsWithConcat2) {
  std::string hlo = R"(
HloModule top

ENTRY main {
  p0 = f32[4] parameter(0)
  p1 = f32[2] parameter(1)
  c0 = f32[4] concatenate(p1, p1), dimensions={0}, control-predecessors={p0}
  ROOT a0 = f32[4] add(c0, p0)
}

)";
  auto module = ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 1);

  const HloInstruction* p0 = FindInstruction(module0, "p0");
  const HloInstruction* p1 = FindInstruction(module0, "p1");
  const HloInstruction* c0 = FindInstruction(module0, "c0");
  const HloInstruction* a0 = FindInstruction(module0, "a0");

  auto t = annotations.tensor_allocation_map.at(TensorLocation{p1, 0});
  EXPECT_EQ(t.tgt, a0);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], c0);
}

TEST_F(AllocationFinderTest, AllocationsAvoidSlice) {
  std::string hlo = R"(
HloModule top

ENTRY main {
  p0 = f16[1,16,16,3] parameter(0)
  p1 = f16[1,1,4,2] parameter(1)
  slice = f16[1,16,16,1] slice(p0), slice={[0:1],[0:16],[0:16],[0:1]}
  concat = f16[1,16,16,4] concatenate(p0, slice), dimensions={3}
  ROOT conv = f16[1,16,16,2] convolution(concat, p1), window={size=1x1}, dim_labels=b01f_01io->b01f
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  const HloInstruction* p0 = FindInstruction(module0, "p0");
  const HloInstruction* p1 = FindInstruction(module0, "p1");
  const HloInstruction* slice = FindInstruction(module0, "slice");
  const HloInstruction* concat = FindInstruction(module0, "concat");
  const HloInstruction* conv = FindInstruction(module0, "conv");

  CompilerAnnotations annotations(module0);
  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{p0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], concat);

  t = annotations.tensor_allocation_map.at(TensorLocation{p1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, AllocationsWithPad) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f32[1,4,6,1] parameter(0)
  c0 = f32[] constant(0)
  padded_p0 = f32[1,16,16,2] pad(p0, c0), padding=0_0x2_1_3x0_0_2x0_1
  p1 = f32[3,3,1,2] parameter(1)

  p1_t = f32[3,3,2,1] transpose(p1), dimensions={0, 1, 3, 2}

  conv = f32[1,16,16,1] convolution(padded_p0, p1_t), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f

  ROOT t = (f32[1,16,16,1]) tuple(conv)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* conv = root->operand(0);
  const auto* padded_ip0 = conv->operand(0);
  const auto* trans = conv->operand(1);
  const auto* ip0 = padded_ip0->operand(0);
  const auto* ip1 = trans->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], padded_ip0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], trans);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 3, 2));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  Literal p0_data =
      LiteralUtil::CreateR4<float>({{{{10}, {20}, {30}, {40}, {50}, {60}},
                                     {{11}, {21}, {31}, {41}, {51}, {61}},
                                     {{12}, {22}, {32}, {42}, {52}, {62}},
                                     {{13}, {23}, {33}, {43}, {53}, {63}}}});
  Literal p1_data =
      LiteralUtil::CreateR4<float>({{{
                                         {{0.1}, {0.2}, {0.3}},  // NOLINT
                                         {{0.4}, {0.5}, {0.6}},  // NOLINT
                                         {{0.7}, {0.8}, {0.9}}   // NOLINT
                                     },
                                     {
                                         {{0}, {1}, {2}},  // NOLINT
                                         {{5}, {4}, {3}},  // NOLINT
                                         {{6}, {7}, {8}}   // NOLINT
                                     }}});
  std::vector<Literal*> inputs(2);
  inputs[0] = &p0_data;
  inputs[1] = &p1_data;
  std::vector<Literal> results = Execute(std::move(module.ValueOrDie()), inputs)
                                     .ValueOrDie()
                                     .DecomposeTuple();

  // Reference values
  // clang-format off
  Literal expected = LiteralUtil::CreateR4<float>({{
  { {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0},
    {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}},
  { {30}, {50}, {140}, {60}, {100}, {210}, {90}, {150},
    {280}, {120}, {200}, {350}, {150}, {250}, {420}, {180}},
  { {9}, {7}, {20}, {18}, {14}, {30}, {27}, {21},
    {40}, {36}, {28}, {50}, {45}, {35}, {60}, {54}},
  { {3}, {1}, {10}, {6}, {2}, {15}, {9}, {3},
    {20}, {12}, {4}, {25}, {15}, {5}, {30}, {18}},
  { {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0},
    {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}},
  { {33}, {55}, {147}, {63}, {105}, {217}, {93}, {155},
    {287}, {123}, {205}, {357}, {153}, {255}, {427}, {183}},
  { {9.9},  {7.7}, {21}, {18.9}, {14.7}, {31}, {27.9}, {21.7},
    {41}, {36.9}, {28.7}, {51}, {45.9}, {35.7}, {61}, {54.9}},
  { {3.3},  {1.1}, {10.5}, {6.3}, {2.1}, {15.5}, {9.3}, {3.1},
    {20.5}, {12.3}, {4.1}, {25.5}, {15.3}, {5.1}, {30.5}, {18.3}},
  { {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0},
    {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}},
  { {36}, {60}, {154}, {66}, {110}, {224}, {96}, {160},
    {294}, {126}, {210}, {364}, {156}, {260}, {434}, {186}},
  { {10.8}, {8.4}, {22}, {19.8}, {15.4}, {32}, {28.8}, {22.4},
    {42}, {37.8}, {29.4}, {52}, {46.8}, {36.4}, {62}, {55.8}},
  { {3.6},  {1.2}, {11}, {6.6}, {2.2}, {16}, {9.6}, {3.2},
    {21}, {12.6}, {4.2}, {26}, {15.6}, {5.2}, {31}, {18.6}},
  { {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0},
    {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}},
  { {39}, {65}, {161}, {69}, {115}, {231}, {99}, {165},
    {301}, {129}, {215}, {371}, {159}, {265}, {441}, {189}},
  { {11.7}, {9.1}, {23}, {20.7}, {16.1}, {33}, {29.7}, {23.1},
    {43}, {38.7}, {30.1}, {53}, {47.7}, {37.1}, {63}, {56.7}},
  { {3.9}, {1.3}, {11.5}, {6.9}, {2.3}, {16.5}, {9.9}, {3.3},
    {21.5}, {12.9}, {4.3}, {26.5}, {15.9}, {5.3}, {31.5}, {18.9}}
  }});
  // clang-format on

  EXPECT_TRUE(LiteralTestUtil::NearOrEqual(expected, results[0],
                                           ErrorSpec{1e-3, 1e-3}));
}

TEST_F(AllocationFinderTest, AllocationsWithPadZero) {
  std::string hlo = R"(
HloModule top

_pop_op_zero_pad {
  p0 = f32[1,4,6,1] parameter(0)
  c0 = f32[] constant(0)
  ROOT padded = f32[1,16,16,2] pad(p0, c0), padding=0_0x2_1_3x0_0_2x0_1
}

ENTRY c1 {
  p0 = f32[1,4,6,1] parameter(0)
  padded_p0 = f32[1,16,16,2] fusion(p0), kind=kCustom, calls=_pop_op_zero_pad
  p1 = f32[3,3,1,2] parameter(1)

  p1_t = f32[3,3,2,1] transpose(p1), dimensions={0, 1, 3, 2}

  conv = f32[1,16,16,1] convolution(padded_p0, p1_t), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f

  ROOT t = (f32[1,16,16,1]) tuple(conv)
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* conv = root->operand(0);
  const auto* padded_ip0 = conv->operand(0);
  const auto* trans = conv->operand(1);
  const auto* ip0 = padded_ip0->operand(0);
  const auto* ip1 = trans->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], padded_ip0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], trans);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1, 3, 2));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  Literal p0_data =
      LiteralUtil::CreateR4<float>({{{{10}, {20}, {30}, {40}, {50}, {60}},
                                     {{11}, {21}, {31}, {41}, {51}, {61}},
                                     {{12}, {22}, {32}, {42}, {52}, {62}},
                                     {{13}, {23}, {33}, {43}, {53}, {63}}}});
  Literal p1_data =
      LiteralUtil::CreateR4<float>({{{
                                         {{0.1}, {0.2}, {0.3}},  // NOLINT
                                         {{0.4}, {0.5}, {0.6}},  // NOLINT
                                         {{0.7}, {0.8}, {0.9}}   // NOLINT
                                     },
                                     {
                                         {{0}, {1}, {2}},  // NOLINT
                                         {{5}, {4}, {3}},  // NOLINT
                                         {{6}, {7}, {8}}   // NOLINT
                                     }}});
  std::vector<Literal*> inputs(2);
  inputs[0] = &p0_data;
  inputs[1] = &p1_data;
  std::vector<Literal> results = Execute(std::move(module.ValueOrDie()), inputs)
                                     .ValueOrDie()
                                     .DecomposeTuple();

  // Reference values
  // clang-format off
  Literal expected = LiteralUtil::CreateR4<float>({{
  { {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0},
    {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}},
  { {30}, {50}, {140}, {60}, {100}, {210}, {90}, {150},
    {280}, {120}, {200}, {350}, {150}, {250}, {420}, {180}},
  { {9}, {7}, {20}, {18}, {14}, {30}, {27}, {21},
    {40}, {36}, {28}, {50}, {45}, {35}, {60}, {54}},
  { {3}, {1}, {10}, {6}, {2}, {15}, {9}, {3},
    {20}, {12}, {4}, {25}, {15}, {5}, {30}, {18}},
  { {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0},
    {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}},
  { {33}, {55}, {147}, {63}, {105}, {217}, {93}, {155},
    {287}, {123}, {205}, {357}, {153}, {255}, {427}, {183}},
  { {9.9},  {7.7}, {21}, {18.9}, {14.7}, {31}, {27.9}, {21.7},
    {41}, {36.9}, {28.7}, {51}, {45.9}, {35.7}, {61}, {54.9}},
  { {3.3},  {1.1}, {10.5}, {6.3}, {2.1}, {15.5}, {9.3}, {3.1},
    {20.5}, {12.3}, {4.1}, {25.5}, {15.3}, {5.1}, {30.5}, {18.3}},
  { {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0},
    {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}},
  { {36}, {60}, {154}, {66}, {110}, {224}, {96}, {160},
    {294}, {126}, {210}, {364}, {156}, {260}, {434}, {186}},
  { {10.8}, {8.4}, {22}, {19.8}, {15.4}, {32}, {28.8}, {22.4},
    {42}, {37.8}, {29.4}, {52}, {46.8}, {36.4}, {62}, {55.8}},
  { {3.6},  {1.2}, {11}, {6.6}, {2.2}, {16}, {9.6}, {3.2},
    {21}, {12.6}, {4.2}, {26}, {15.6}, {5.2}, {31}, {18.6}},
  { {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0},
    {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}},
  { {39}, {65}, {161}, {69}, {115}, {231}, {99}, {165},
    {301}, {129}, {215}, {371}, {159}, {265}, {441}, {189}},
  { {11.7}, {9.1}, {23}, {20.7}, {16.1}, {33}, {29.7}, {23.1},
    {43}, {38.7}, {30.1}, {53}, {47.7}, {37.1}, {63}, {56.7}},
  { {3.9}, {1.3}, {11.5}, {6.9}, {2.3}, {16.5}, {9.9}, {3.3},
    {21.5}, {12.9}, {4.3}, {26.5}, {15.9}, {5.3}, {31.5}, {18.9}}
  }});
  // clang-format on

  EXPECT_TRUE(LiteralTestUtil::NearOrEqual(expected, results[0],
                                           ErrorSpec{1e-3, 1e-3}));
}

TEST_F(AllocationFinderTest, BatchNormInfParamsWithPad) {
  std::string hlo = R"(
HloModule top

ENTRY top {
 p0 = f32[1,4,4,8] parameter(0)
 p1 = f32[1,1,8,8] parameter(1)
 conv = f32[1,4,4,8] convolution(p0, p1), window={size=1x1}, dim_labels=b01f_01io->b01f
 p2 = f32[1,8] parameter(2)
 p2_r = f32[8] reshape(p2)
 p3 = f32[1,3] parameter(3)
 p3_r = f32[3] reshape(p3)
 c0 = f32[] constant(1)
 padded_p3 = f32[8] pad(p3_r, c0), padding=1_2_1
 p4 = f32[8] parameter(4)
 p5 = f32[8] parameter(5)
 ROOT batch-norm-inf = f32[1,4,4,8] batch-norm-inference(conv, p2_r, padded_p3, p4, p5), epsilon=0.001, feature_index=3
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* bn = root;
  const auto* conv = bn->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* p2_r = bn->operand(1);
  const auto* ip2 = p2_r->operand(0);
  const auto* padded_p3 = bn->operand(2);
  const auto* p3_r = padded_p3->operand(0);
  const auto* ip3 = p3_r->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], p2_r);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip3, 0});
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);
  EXPECT_EQ(t.backward_path[0], p3_r);
  EXPECT_EQ(t.backward_path[1], padded_p3);
}

TEST_F(AllocationFinderTest, BatchNormInfParamsWithZeroPad) {
  std::string hlo = R"(
HloModule top

_pop_op_zero_pad {
  p0 = f32[3] parameter(0)
  c0 = f32[] constant(0)
  ROOT padded = f32[8] pad(p0, c0), padding=1_2_1
}

ENTRY top {
 p0 = f32[1,4,4,8] parameter(0)
 p1 = f32[1,1,8,8] parameter(1)
 conv = f32[1,4,4,8] convolution(p0, p1), window={size=1x1}, dim_labels=b01f_01io->b01f
 p2 = f32[1,8] parameter(2)
 p2_r = f32[8] reshape(p2)
 p3 = f32[1,3] parameter(3)
 p3_r = f32[3] reshape(p3)
 padded_p3 = f32[8] fusion(p3_r), kind=kCustom, calls=_pop_op_zero_pad
 p4 = f32[8] parameter(4)
 p5 = f32[8] parameter(5)
 ROOT batch-norm-inf = f32[1,4,4,8] batch-norm-inference(conv, p2_r, padded_p3, p4, p5), epsilon=0.001, feature_index=3
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* bn = root;
  const auto* conv = bn->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* p2_r = bn->operand(1);
  const auto* ip2 = p2_r->operand(0);
  const auto* padded_p3 = bn->operand(2);
  const auto* p3_r = padded_p3->operand(0);
  const auto* ip3 = p3_r->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{ip0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip2, 0});
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], p2_r);

  t = annotations.tensor_allocation_map.at(TensorLocation{ip3, 0});
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);
  EXPECT_EQ(t.backward_path[0], p3_r);
  EXPECT_EQ(t.backward_path[1], padded_p3);
}

TEST_F(AllocationFinderTest, PipelineScopes) {
  const string& hlo = R"(
HloModule module

_stage_0 {
  s0_param0 = f32[] parameter(0), sharding={maximal device=0}
  ROOT t = (f32[]) tuple(s0_param0), sharding={{maximal device=0}}
}

_stage_1 {
  s1_param0 = f32[] parameter(0), sharding={maximal device=1}
  s1_param1 = f32[] parameter(1), sharding={maximal device=1}
  add_1 = f32[] add(s1_param0, s1_param1), sharding={maximal device=1}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=1}}
}

_stage_2 {
  s2_param0 = f32[] parameter(0), sharding={maximal device=2}
  s2_param1 = f32[] parameter(1), sharding={maximal device=2}
  add_2 = f32[] add(s2_param1, s2_param0), sharding={maximal device=2}
  ROOT t = (f32[]) tuple(add_2), sharding={{maximal device=2}}
}

ENTRY pipeline {
  arg0 = f32[] parameter(0), sharding={maximal device=0}
  arg1 = f32[] parameter(1), sharding={maximal device=1}
  arg2 = f32[] parameter(2), sharding={maximal device=2}

  a0 = (f32[]) call(arg0), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = f32[] get-tuple-element(a0), index=0, sharding={maximal device=0}
  b0 = (f32[]) call(gte_a, arg1), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[] get-tuple-element(b0), index=0, sharding={maximal device=1}
  ROOT c0 = (f32[]) call(gte_b, arg2), to_apply=_stage_2, sharding={{maximal device=2}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  HloInstruction* s1_param1 = FindInstruction(module0, "s1_param1");
  HloInstruction* add_1 = FindInstruction(module0, "add_1");
  auto t = annotations.tensor_allocation_map.at(TensorLocation{s1_param1, 0});
  EXPECT_EQ(t.tgt, add_1);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);

  HloInstruction* s2_param1 = FindInstruction(module0, "s2_param1");
  HloInstruction* add_2 = FindInstruction(module0, "add_2");
  t = annotations.tensor_allocation_map.at(TensorLocation{s2_param1, 0});
  EXPECT_EQ(t.tgt, add_2);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, FindAllocationTargetScalar) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p = (f32[]) parameter(0)
  ROOT c = f32[] custom-call(p), custom_call_target="LstmLayerFwd", backend_config="{\"num_channels\":1, \"is_training\":false, \"partials_dtype\":\"DT_FLOAT\", \"activation\":\"tanh\", \"recurrent_activation\":\"sigmoid\", \"available_memory_proportion\":-1.0}\n"
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 1);

  const auto* param = module0->entry_computation()->parameter_instruction(0);
  const auto* lstm = module0->entry_computation()->root_instruction();

  auto t = annotations.tensor_allocation_map.at(TensorLocation{param, 0});
  EXPECT_EQ(t.tgt, lstm);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_TRUE(t.permutation.has_value());
  EXPECT_EQ(t.permutation->size(), 0);
}

TEST_F(AllocationFinderTest, FindAllocationTargetWithSendToHost) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[3,3,2,4] parameter(1)

  merged = (f16[1,16,16,2], f16[3,3,2,4]) tuple(p0, p1)
  send = () custom-call((f16[1,16,16,2], f16[3,3,2,4]) merged), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"\"}"
  ROOT conv = f16[1,16,16,4] convolution(p0, p1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());

  const auto* conv = module0->entry_computation()->root_instruction();
  const auto* p0 = module0->entry_computation()->parameter_instruction(0);
  const auto* p1 = module0->entry_computation()->parameter_instruction(1);

  for (const bool always_rearrange_on_host : {false, true}) {
    CompilerAnnotations annotations(module0);
    AllocationFinder finder(annotations, always_rearrange_on_host);
    EXPECT_TRUE(finder.Run(module0).ValueOrDie());
    EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

    const auto t0 = annotations.tensor_allocation_map.at(TensorLocation{p0, 0});
    const auto t1 = annotations.tensor_allocation_map.at(TensorLocation{p1, 0});

    if (always_rearrange_on_host) {
      // Conv should have highest priority.
      EXPECT_EQ(t0.tgt, conv);
      EXPECT_EQ(t1.tgt, conv);
      EXPECT_EQ(t0.input_index, 0ll);
      EXPECT_EQ(t1.input_index, 1ll);
    } else {
      // SendToHost should have highest priority.
      EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SendToHost)(t0.tgt));
      EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SendToHost)(t1.tgt));
      EXPECT_EQ(t0.input_index, 0ll);
      EXPECT_EQ(t1.input_index, 0ll);
    }
  }
}

TEST_F(AllocationFinderTest, LookThrough) {
  std::string hlo = R"(
HloModule top

ENTRY top {
 p0 = f32[1,4,4,8] parameter(0)
 p1 = f32[1,1,8,8] parameter(1)
 conv = f32[1,4,4,8] convolution(p0, p1), window={size=1x1}, dim_labels=b01f_01io->b01f
 p2 = f32[1,4,4,8] parameter(2)
 p3 = f32[4,8] parameter(3)
 bcast_p3 = f32[1,4,4,8] broadcast(p3), dimensions={2, 3}
 add = f32[1,4,4,8] add(p2, bcast_p3)
 ROOT add2 = f32[1,4,4,8] add(add, conv)
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  ElementwiseBroadcastConverter ebc;
  EXPECT_TRUE(ebc.Run(module0).ValueOrDie());

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* add2 = root;
  const auto* conv = add2->operand(1);
  const auto* p0 = conv->operand(0);
  const auto* p1 = conv->operand(1);
  const auto* add = add2->operand(0);
  const auto* p2 = add->operand(0);
  const auto* p3 = add->operand(1);
  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{p0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{p1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(TensorLocation{p2, 0});
  EXPECT_EQ(t.tgt, add2);
  EXPECT_EQ(t.input_index, 0);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], add);

  t = annotations.tensor_allocation_map.at(TensorLocation{p3, 0});
  EXPECT_EQ(t.tgt, add);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, p2);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, LookThroughScalar) {
  // A test which makes sure p3 doesn't get a layout.
  std::string hlo = R"(
HloModule top

ENTRY top {
 p0 = f32[1,4,4,8] parameter(0)
 p1 = f32[1,1,8,8] parameter(1)
 conv = f32[1,4,4,8] convolution(p0, p1), window={size=1x1}, dim_labels=b01f_01io->b01f
 p2 = f32[1,4,4,8] parameter(2)
 p3 = f32[] parameter(3)
 p4 = f32[1,4,4,8] parameter(4)
 bcast_p3 = f32[1,4,4,8] broadcast(p3), dimensions={}
 mul = f32[1,4,4,8] multiply(p2, bcast_p3)
 add = f32[1,4,4,8] add(conv, mul)
 add2 = f32[1,4,4,8] add(conv, mul)
 ROOT div = f32[1,4,4,8] divide(add, p4)
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  CompilerAnnotations annotations(module0);

  FuseOpsIntoPoplarOps foipo(annotations);
  EXPECT_TRUE(foipo.Run(module0).ValueOrDie());

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* div = root;
  const auto* add = div->operand(0);
  const auto* conv = add->operand(0);
  const auto* p0 = conv->operand(0);
  const auto* p1 = conv->operand(1);
  const auto* p2 = add->operand(1);
  const auto* p3 = add->operand(2);
  const auto* p4 = div->operand(1);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{p0, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{p1, 0});
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(TensorLocation{p2, 0});
  EXPECT_EQ(t.tgt, add);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);

  t = annotations.tensor_allocation_map.at(TensorLocation{p4, 0});
  EXPECT_EQ(t.tgt, div);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, add);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, LookThroughMultiUpdateAdd) {
  // A test that makes sure that multiple MultiUpdateAdd instructions with
  // equivalent plans may use those plans.
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f16[2, 64] parameter(0)
  p1 = f16[64, 1024] parameter(1)
  p2 = s32[2, 1] parameter(2)
  p1_t = f16[1024, 64] transpose(p1), dimensions={1, 0}
  mu1 = f16[1024, 64] custom-call(p1_t, p2, p0), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
  mu2 = f16[1024, 64] custom-call(mu1, p2, p0), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
  ROOT t = (f16[1024, 64], f16[1024, 64]) tuple(mu1, mu2)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  auto resources = CompilerResources::CreateTestDefault(module0);
  resources->main_graph = absl::make_unique<poplar::Graph>(
      poplar::Device::createCPUDevice(), poplar::replication_factor(1));

  auto& res = *resources;
  auto& annotations = res.annotations;

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ok());

  EXPECT_TRUE(ModuleFlatten(res.annotations).Run(module0).ValueOrDie());
  EXPECT_FALSE(EmbeddingPlansPreplanning(res).Run(module0).ValueOrDie());

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* mu1 = root->operand(0);
  const auto* transpose = mu1->operand(0);
  const auto* p1 = transpose->operand(0);
  const auto* p2 = mu1->operand(1);
  const auto* mu2 = root->operand(1);
  const auto* p0 = mu1->operand(2);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{p0, 0});
  EXPECT_EQ(t.tgt, mu1);
  EXPECT_EQ(t.input_index, 2ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{p1, 0});
  EXPECT_EQ(t.tgt, mu1);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(1, 0));
  EXPECT_EQ((*t.sliceable_dimension), 1);
  EXPECT_EQ(t.compatible_slice_plans.size(), 1);
  EXPECT_EQ(*t.compatible_slice_plans.begin(), mu2);

  NotifySlicePlanAllocation(
      res, t);  // Notify allocation on the first MultiUpdateAdd

  t = annotations.tensor_allocation_map.at(TensorLocation{p2, 0});
  EXPECT_EQ(t.tgt, mu1);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  // Check that we actually can use those plans
  EXPECT_TRUE(SlicePlanHasAllocation(res, mu1).ValueOrDie());
  EXPECT_TRUE(SlicePlanHasAllocation(res, mu2).ValueOrDie());
}

TEST_F(AllocationFinderTest, LookThroughCall) {
  std::string hlo = R"(
HloModule module

c0 {
  a = f32[3, 3] parameter(0)
  b = f32[3, 3] parameter(1)
  ROOT c = f32[3, 3] add(a, b)
}

main {
  p0 = f32[3, 3] parameter(0)
  p1 = f32[3, 3] parameter(1)
  p2 = f32[3, 3] parameter(2)
  x = f32[3, 3] call(p1, p2), to_apply=c0
  ROOT y = f32[3, 3] dot(p0, x), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto config = GetModuleConfigForTest();
  auto module0 = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module0.ok());
  auto* module_ptr = module0.ValueOrDie().get();

  CompilerAnnotations annotations(module_ptr);

  EXPECT_TRUE(AllocationFinder(annotations).Run(module_ptr).ValueOrDie());

  const auto* y = module_ptr->entry_computation()->root_instruction();
  const auto* p0 = y->operand(0);
  const auto* x = y->operand(1);
  const auto* p1 = x->operand(0);
  const auto* p2 = x->operand(1);
  const auto* c = x->to_apply()->root_instruction();
  const auto* a = c->operand(0);
  const auto* b = c->operand(1);

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 5);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{p0, 0});
  EXPECT_EQ(t.tgt, y);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{p1, 0});
  EXPECT_EQ(t.tgt, y);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 3);
  EXPECT_EQ(t.backward_path[0], a);
  EXPECT_EQ(t.backward_path[1], c);
  EXPECT_EQ(t.backward_path[2], x);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{p2, 0});
  EXPECT_EQ(t.tgt, y);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 3);
  EXPECT_EQ(t.backward_path[0], b);
  EXPECT_EQ(t.backward_path[1], c);
  EXPECT_EQ(t.backward_path[2], x);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{a, 0});
  EXPECT_EQ(t.tgt, y);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);
  EXPECT_EQ(t.backward_path[0], c);
  EXPECT_EQ(t.backward_path[1], x);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{b, 0});
  EXPECT_EQ(t.tgt, y);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);
  EXPECT_EQ(t.backward_path[0], c);
  EXPECT_EQ(t.backward_path[1], x);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

TEST_F(AllocationFinderTest, LookThroughCallTuple) {
  std::string hlo = R"(
HloModule module

c0 {
  a = f32[3, 3] parameter(0)
  b = f32[3, 3] parameter(1)
  c = f32[3, 3] add(a, b)

  ROOT e = (f32[3, 3], f32[3, 3]) tuple(c, c)
}

main {
  p0 = f32[3, 3] parameter(0)
  p1 = f32[3, 3] parameter(1)
  p2 = f32[3, 3] parameter(2)

  v = (f32[3, 3], f32[3, 3]) call(p1, p2), to_apply=c0
  x = f32[3, 3] get-tuple-element(v), index=0
  ROOT y = f32[3, 3] dot(p0, x), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto config = GetModuleConfigForTest();
  auto module0 = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module0.ok());
  auto* module_ptr = module0.ValueOrDie().get();

  CompilerAnnotations annotations(module_ptr);

  EXPECT_TRUE(AllocationFinder(annotations).Run(module_ptr).ValueOrDie());

  const auto* y = module_ptr->entry_computation()->root_instruction();
  const auto* p0 = y->operand(0);
  const auto* x = y->operand(1);
  const auto* v = x->operand(0);
  const auto* p1 = v->operand(0);
  const auto* p2 = v->operand(1);
  const auto* e = v->to_apply()->root_instruction();
  const auto* c = e->operand(0);
  const auto* a = c->operand(0);
  const auto* b = c->operand(1);

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 5);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{p0, 0});
  EXPECT_EQ(t.tgt, y);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{p1, 0});
  EXPECT_EQ(t.tgt, y);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 5);
  EXPECT_EQ(t.backward_path[0], a);
  EXPECT_EQ(t.backward_path[1], c);
  EXPECT_EQ(t.backward_path[2], e);
  EXPECT_EQ(t.backward_path[3], v);
  EXPECT_EQ(t.backward_path[4], x);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{p2, 0});
  EXPECT_EQ(t.tgt, y);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 5);
  EXPECT_EQ(t.backward_path[0], b);
  EXPECT_EQ(t.backward_path[1], c);
  EXPECT_EQ(t.backward_path[2], e);
  EXPECT_EQ(t.backward_path[3], v);
  EXPECT_EQ(t.backward_path[4], x);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{a, 0});
  EXPECT_EQ(t.tgt, y);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 4);
  EXPECT_EQ(t.backward_path[0], c);
  EXPECT_EQ(t.backward_path[1], e);
  EXPECT_EQ(t.backward_path[2], v);
  EXPECT_EQ(t.backward_path[3], x);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{b, 0});
  EXPECT_EQ(t.tgt, y);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 4);
  EXPECT_EQ(t.backward_path[0], c);
  EXPECT_EQ(t.backward_path[1], e);
  EXPECT_EQ(t.backward_path[2], v);
  EXPECT_EQ(t.backward_path[3], x);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

TEST_F(AllocationFinderTest, LookThroughCallConcat) {
  std::string hlo = R"(
HloModule module

c0 {
  a = f32[3, 3] parameter(0)
  b = f32[3, 3] parameter(1)
  ROOT c = f32[3, 3] add(a, b)
}

main {
  p0 = f32[3, 3] parameter(0)
  p1 = f32[3, 3] parameter(1)
  p2 = f32[3, 3] parameter(2)
  x = f32[3, 3] call(p1, p2), to_apply=c0
  v = f16[6, 3] concatenate(p0, x), dimensions={0}
  ROOT y = f32[6, 3] dot(v, p0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto config = GetModuleConfigForTest();
  auto module0 = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module0.ok());
  auto* module_ptr = module0.ValueOrDie().get();

  CompilerAnnotations annotations(module_ptr);

  EXPECT_TRUE(AllocationFinder(annotations).Run(module_ptr).ValueOrDie());
  const auto* y = module_ptr->entry_computation()->root_instruction();
  const auto* p0 = y->operand(1);
  const auto* v = y->operand(0);
  const auto* x = v->operand(1);
  const auto* p1 = x->operand(0);
  const auto* p2 = x->operand(1);
  const auto* c = x->to_apply()->root_instruction();
  const auto* a = c->operand(0);
  const auto* b = c->operand(1);

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 5);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{p0, 0});
  EXPECT_EQ(t.tgt, y);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], v);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{p1, 0});
  EXPECT_EQ(t.tgt, y);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 4);
  EXPECT_EQ(t.backward_path[0], a);
  EXPECT_EQ(t.backward_path[1], c);
  EXPECT_EQ(t.backward_path[2], x);
  EXPECT_EQ(t.backward_path[3], v);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{p2, 0});
  EXPECT_EQ(t.tgt, y);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 4);
  EXPECT_EQ(t.backward_path[0], b);
  EXPECT_EQ(t.backward_path[1], c);
  EXPECT_EQ(t.backward_path[2], x);
  EXPECT_EQ(t.backward_path[3], v);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{a, 0});
  EXPECT_EQ(t.tgt, y);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 3);
  EXPECT_EQ(t.backward_path[0], c);
  EXPECT_EQ(t.backward_path[1], x);
  EXPECT_EQ(t.backward_path[2], v);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{b, 0});
  EXPECT_EQ(t.tgt, y);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 3);
  EXPECT_EQ(t.backward_path[0], c);
  EXPECT_EQ(t.backward_path[1], x);
  EXPECT_EQ(t.backward_path[2], v);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

TEST_F(AllocationFinderTest, LookThroughCallAddTensorForTarget) {
  std::string hlo = R"(
HloModule module

c0 {
  a = f32[3, 3] parameter(0)
  b = f32[3, 3] parameter(1)
  ROOT c = f32[3, 3] add(a, b)
}

main {
  p0 = f32[3, 3] parameter(0)
  p1 = f32[3, 3] parameter(1)
  p2 = f32[3, 3] parameter(2)
  x = f32[3, 3] call(p1, p2), to_apply=c0
  v = f16[6, 3] concatenate(p0, x), dimensions={0}
  ROOT y = f32[6, 3] dot(v, p0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto config = GetModuleConfigForTest();
  auto module0 = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module0.ok());
  auto* module_ptr = module0.ValueOrDie().get();

  auto resources = CompilerResources::CreateTestDefault(module_ptr);
  resources->main_graph = absl::make_unique<poplar::Graph>(
      poplar::Device::createCPUDevice(), poplar::replication_factor(1));
  auto& annotations = resources->annotations;
  auto* graph = resources->main_graph.get();

  EXPECT_TRUE(AllocationFinder(annotations).Run(module_ptr).ValueOrDie());

  const auto* entry_comp = module_ptr->entry_computation();
  const auto* y = entry_comp->root_instruction();
  const auto* p0 = y->operand(1);
  const auto* v = y->operand(0);
  const auto* x = v->operand(1);
  const auto* p1 = x->operand(0);
  const auto* p2 = x->operand(1);

  auto& alloc_map = annotations.tensor_allocation_map;
  TensorMap tensor_map;

  // Add tensor for p0.
  TensorLocation p0_loc{p0, 0};
  TF_ASSERT_OK(AddTensorForTarget(*graph, p0_loc, alloc_map.at(p0_loc),
                                  *resources, tensor_map, {})
                   .status());

  // Add tensor for p1.
  TensorLocation p1_loc{p1, 0};
  TF_ASSERT_OK(AddTensorForTarget(*graph, p1_loc, alloc_map.at(p1_loc),
                                  *resources, tensor_map, {})
                   .status());

  // Add tensor for p2.
  TensorLocation p2_loc{p2, 0};
  TF_ASSERT_OK(AddTensorForTarget(*graph, p2_loc, alloc_map.at(p2_loc),
                                  *resources, tensor_map, {})
                   .status());
}

TEST_F(AllocationFinderTest, LookThroughSequenceSlice) {
  std::string hlo = R"(
HloModule module

loop_body {
  dst = f16[128, 1024] parameter(0)
  src = f16[128, 1024] parameter(1)
  num_elems = s32[8] parameter(2)
  src_offsets = s32[8] parameter(3)
  dst_offsets = s32[8] parameter(4)

  out = f16[128, 1024] custom-call(dst, src, num_elems, src_offsets, dst_offsets), custom_call_target="SequenceSlice", backend_config="{\"zero_unused\":1}\n"
  ROOT t = (f16[128, 1024]) tuple(out)
}

ENTRY main {
  p0 = f16[128, 1024] parameter(0)
  p1 = f16[128, 1024] parameter(1)
  p2 = s32[8] parameter(2)
  p3 = s32[8] parameter(3)
  p4 = s32[8] parameter(4)
  
  w = f16[] constant(2)
  weights = f16[1024, 3] broadcast(w), dimensions={}

  repeat = (f16[128, 1024]) call(p0, p1, p2, p3, p4), to_apply=loop_body, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"16\"}}}"
  gte = f16[128, 1024] get-tuple-element(repeat), index=0
  ROOT dot = f16[128, 3] dot(gte, weights), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto config = GetModuleConfigForTest();
  auto module0 = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module0.ok());
  auto* module_ptr = module0.ValueOrDie().get();

  auto resources = CompilerResources::CreateTestDefault(module_ptr);

  CompilerAnnotations annotations(module_ptr);

  EXPECT_TRUE(CustomOpReplacer().Run(module_ptr).ValueOrDie());
  EXPECT_TRUE(FuseWideConst(annotations).Run(module_ptr).ValueOrDie());
  EXPECT_TRUE(AllocationFinder(annotations).Run(module_ptr).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  const auto* entry = module_ptr->entry_computation();
  const auto* dot = entry->root_instruction();
  const auto* gte = dot->operand(0);
  const auto* weights = dot->operand(1);
  const auto* repeat = gte->operand(0);
  const auto* p0 = repeat->operand(0);
  const auto* t_loop = repeat->to_apply()->root_instruction();
  const auto* seq_slice = t_loop->operand(0);
  const auto* dst = seq_slice->operand(0);

  auto t = annotations.tensor_allocation_map.at(TensorLocation{p0, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 5);
  EXPECT_EQ(t.backward_path[0], dst);
  EXPECT_EQ(t.backward_path[1], seq_slice);
  EXPECT_EQ(t.backward_path[2], t_loop);
  EXPECT_EQ(t.backward_path[3], repeat);
  EXPECT_EQ(t.backward_path[4], gte);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{weights, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);

  t = annotations.tensor_allocation_map.at(TensorLocation{dst, 0});
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 4);
  EXPECT_EQ(t.backward_path[0], seq_slice);
  EXPECT_EQ(t.backward_path[1], t_loop);
  EXPECT_EQ(t.backward_path[2], repeat);
  EXPECT_EQ(t.backward_path[3], gte);
  EXPECT_THAT((*t.permutation), ::testing::ElementsAre(0, 1));
  EXPECT_EQ(t.sliceable_dimension, absl::nullopt);
}

TEST_F(AllocationFinderTest, LookThroughConditional) {
  std::string hlo = R"(
HloModule module
cond_true_9__.10 (arg_tuple.11: (f32[2,2], f32[2,2], f32[2,2])) -> (f32[2,2]) {
  arg_tuple.11 = (f32[2,2], f32[2,2], f32[2,2]) parameter(0)
  get-tuple-element.12 = f32[2,2] get-tuple-element(arg_tuple.11), index=0
  get-tuple-element.13 = f32[2,2] get-tuple-element(arg_tuple.11), index=1
  dot.15 = f32[2,2] dot(get-tuple-element.12, get-tuple-element.13), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT tuple.16 = (f32[2,2]) tuple(dot.15)
}

cond_false_10__.17 (arg_tuple.18: (f32[2,2], f32[2,2], f32[2,2])) -> (f32[2,2]) {
  arg_tuple.18 = (f32[2,2], f32[2,2], f32[2,2]) parameter(0)
  get-tuple-element.19 = f32[2,2] get-tuple-element(arg_tuple.18), index=0
  get-tuple-element.21 = f32[2,2] get-tuple-element(arg_tuple.18), index=2
  dot.22 = f32[2,2] dot(get-tuple-element.19, get-tuple-element.21), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT tuple.23 = (f32[2,2]) tuple(dot.22)
}

ENTRY cluster_14155272572172615694__.29 (arg0.1: f32[2,2], arg1.2: f32[2,2], arg2.3: f32[2,2], arg3.4: pred[]) -> f32[2,2] {
  arg3.4 = pred[] parameter(3), parameter_replication={false}
  arg0.1 = f32[2,2] parameter(0), parameter_replication={false}
  arg1.2 = f32[2,2] parameter(1), parameter_replication={false}
  arg2.3 = f32[2,2] parameter(2), parameter_replication={false}
  tuple.9 = (f32[2,2], f32[2,2], f32[2,2]) tuple(arg0.1, arg1.2, arg2.3)
  conditional.24 = (f32[2,2]) conditional(arg3.4, tuple.9, tuple.9), true_computation=cond_true_9__.10, false_computation=cond_false_10__.17
  ROOT get-tuple-element.25 = f32[2,2] get-tuple-element(conditional.24), index=0
}
)";

  auto config = GetModuleConfigForTest();
  auto module0 = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module0.ok());
  auto* module_ptr = module0.ValueOrDie().get();

  auto resources = CompilerResources::CreateTestDefault(module_ptr);

  CompilerAnnotations annotations(module_ptr);

  EXPECT_TRUE(AllocationFinder(annotations).Run(module_ptr).ValueOrDie());
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 7);

  const auto entry = module_ptr->entry_computation();
  const auto arg0_1 = entry->parameter_instruction(0);
  const auto arg1_2 = entry->parameter_instruction(1);
  const auto arg2_3 = entry->parameter_instruction(2);

  const auto true_fn = module_ptr->GetComputationWithName("cond_true_9__.10");
  const auto dot_15 = true_fn->GetInstructionWithName("dot.15");

  const auto false_fn =
      module_ptr->GetComputationWithName("cond_false_10__.17");
  const auto dot_22 = false_fn->GetInstructionWithName("dot.22");

  auto t1 = annotations.tensor_allocation_map.at(TensorLocation{arg0_1, 0});
  EXPECT_EQ(t1.tgt, dot_15);
  EXPECT_EQ(t1.input_index, 0ll);
  EXPECT_EQ(t1.sliceable_dimension, absl::nullopt);

  auto t2 = annotations.tensor_allocation_map.at(TensorLocation{arg1_2, 0});
  EXPECT_EQ(t2.tgt, dot_15);
  EXPECT_EQ(t2.input_index, 1ll);
  EXPECT_EQ(t2.sliceable_dimension, absl::nullopt);

  auto t3 = annotations.tensor_allocation_map.at(TensorLocation{arg2_3, 0});
  EXPECT_EQ(t3.tgt, dot_22);
  EXPECT_EQ(t3.input_index, 1ll);
  EXPECT_EQ(t3.sliceable_dimension, absl::nullopt);
}

// // TODO:
// // - can forward path traverse in-place ops
// // - is forward path rejected when going through non-layout preserving
// inputs

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
