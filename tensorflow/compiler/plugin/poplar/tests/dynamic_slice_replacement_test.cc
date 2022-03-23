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

#include <gtest/gtest.h>

#include "absl/strings/substitute.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_base.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/slice_util.h"

#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/dynamic_slice_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/module_flatten.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_passes/embedding_plans_preplanning.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

#include <poplar/Device.hpp>
#include <poplar/Engine.hpp>

namespace xla {
namespace poplarplugin {
namespace {

void ConnectStream(poplar::Engine& engine, const std::string& name,
                   std::vector<int>& values) {
  engine.connectStream(name, values.data(), values.data() + values.size());
}

PoplarOp MultiSliceType(HloOpcode opcode) {
  return opcode == HloOpcode::kDynamicSlice ? PoplarOp::MultiSlice
                                            : PoplarOp::MultiUpdate;
}

using HloAndSliceOffsets = std::pair<std::string, std::vector<int>>;
struct DynamicSliceHloTest : HloPoplarTestBase,
                             ::testing::WithParamInterface<HloAndSliceOffsets> {
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(auto ipu_count, GetMaxIpuCount());
    if (ipu_count == 0) {
      GTEST_SKIP() << "Skipping tests as we need 1 ipu but have none."
                   << "Make sure TF_IPU_COUNT is set.";
    }

    TF_ASSERT_OK_AND_ASSIGN(device_, CreateIpuDevice(1, /*num_tiles*/ 1200));
    TF_ASSERT_OK_AND_ASSIGN(module_,
                            ParseAndReturnVerifiedModule(GetParam().first));

    auto input = FindInstruction(module_.get(), "input_tensor");
    ASSERT_TRUE(input);
    input_matrix_ = std::vector<int>(ShapeUtil::ElementsIn(input->shape()), 0);

    auto offsets = FindInstruction(module_.get(), "offsets");
    ASSERT_TRUE(offsets);

    offsets_vector_ = GetParam().second;
    ASSERT_EQ(offsets_vector_.size(), ShapeUtil::ElementsIn(offsets->shape()));
  }

  std::pair<std::vector<int>, std::vector<int>> RunEngine(
      poplar::Engine& engine) {
    engine.load(device_);

    // Setup matrix with incrementing values.
    std::iota(input_matrix_.begin(), input_matrix_.end(), 0);
    ConnectStream(engine, "0.0", input_matrix_);

    ConnectStream(engine, "1.0", offsets_vector_);

    const auto root = module_->entry_computation()->root_instruction();

    const auto& root_output0 = root->shape().tuple_shapes(0);
    std::vector<int> multislice_out(ShapeUtil::ElementsIn(root_output0), 0);
    ConnectStream(engine, "out_0.0", multislice_out);

    const auto& root_output1 = root->shape().tuple_shapes(1);
    std::vector<int> dynamicslice_out(ShapeUtil::ElementsIn(root_output1), 0);
    ConnectStream(engine, "out_1.0", dynamicslice_out);

    engine.run(0);

    return {multislice_out, dynamicslice_out};
  }

  poplar::Device device_;
  std::unique_ptr<VerifiedHloModule> module_;

  std::vector<int> input_matrix_;
  std::vector<int> offsets_vector_;

  std::string slice_being_replaced_ = "replace_dynamic_slice";
};

using DynamicSliceSupportedHloTest = DynamicSliceHloTest;

TEST_P(DynamicSliceSupportedHloTest, CheckCanReplaceSlice) {
  auto instr = FindInstruction(module_.get(), slice_being_replaced_);
  ASSERT_TRUE(instr);

  auto dynamic_slice = Cast<HloDynamicIndexInstruction>(instr);
  const auto expected_parent = dynamic_slice->parent();
  const auto expected_poplar_op = MultiSliceType(dynamic_slice->opcode());
  TF_ASSERT_OK_AND_ASSIGN(auto multi_slice,
                          TryReplaceDynamicWithMultiSlice(dynamic_slice));
  ASSERT_TRUE(multi_slice);

  // Check that the slice has been removed.
  ASSERT_FALSE(FindInstruction(module_.get(), slice_being_replaced_));

  ASSERT_TRUE(IsPoplarInstruction(expected_poplar_op, multi_slice));
  ASSERT_EQ(multi_slice->parent(), expected_parent);
  if (expected_poplar_op == PoplarOp::MultiSlice) {
    ASSERT_EQ(multi_slice->shape().dimensions(0), 1);
  }
}

TEST_P(DynamicSliceSupportedHloTest, CompareReplacedSlice) {
  auto instr = FindInstruction(module_.get(), slice_being_replaced_);
  ASSERT_TRUE(instr);

  auto dynamic_slice = Cast<HloDynamicIndexInstruction>(instr);
  TryReplaceDynamicWithMultiSlice(dynamic_slice);

  // EmbeddingPlansPreplanning requires flattened modules.
  auto resources = GetMockResources(device_, module_.get(), 1);
  ASSERT_TRUE(
      ModuleFlatten(resources->annotations).Run(module_.get()).ValueOrDie());
  // Not asserting anything since EmbeddingPlansPreplanning always returns
  // False.
  EmbeddingPlansPreplanning(*resources).Run(module_.get());

  TF_ASSERT_OK_AND_ASSIGN(poplar::Engine engine,
                          Compile(*resources, module_.get()));

  // The hlo ouputs 2 equal dynamic slices. Check that the output of our
  // replacement matches the remaining one.
  auto results = RunEngine(engine);
  auto multislice_out = results.first;
  auto dynamicslice_out = results.second;
  ASSERT_EQ(multislice_out, dynamicslice_out);
}

using DynamicSliceUnsupportedHloTest = DynamicSliceHloTest;

TEST_P(DynamicSliceUnsupportedHloTest, CantReplace) {
  auto instr = FindInstruction(module_.get(), slice_being_replaced_);
  ASSERT_TRUE(instr);

  auto dynamic_slice = Cast<HloDynamicIndexInstruction>(instr);
  TF_ASSERT_OK_AND_ASSIGN(auto multi_slice,
                          TryReplaceDynamicWithMultiSlice(dynamic_slice));
  ASSERT_FALSE(multi_slice);
  // Check that the original slice hasn't been removed.
  ASSERT_TRUE(FindInstruction(module_.get(), slice_being_replaced_));
}

HloAndSliceOffsets Slice3DInputTestCase(const std::string& tensor_size,
                                        const std::string& slice_size,
                                        const std::vector<int>& offsets = {0, 0,
                                                                           0}) {
  const char* template_hlo = R"(
HloModule test
ENTRY test {
  input_tensor = s32[$0] parameter(0)
  offsets = s32[3] parameter(1)

  slice.5 = s32[1] slice(offsets), slice={[0:1]}, metadata={op_type="Slice" op_name="Slice"}
  xOffset = s32[] reshape(slice.5), metadata={op_type="Slice" op_name="Slice"}

  slice.7 = s32[1] slice(offsets), slice={[1:2]}, metadata={op_type="Slice" op_name="Slice"}
  yOffset = s32[] reshape(slice.7), metadata={op_type="Slice" op_name="Slice"}

  slice.9 = s32[1] slice(offsets), slice={[2:3]}, metadata={op_type="Slice" op_name="Slice"}
  zOffset = s32[] reshape(slice.9), metadata={op_type="Slice" op_name="Slice"}

  replace_dynamic_slice = s32[$1] dynamic-slice(input_tensor, xOffset, yOffset, zOffset), dynamic_slice_sizes={$1}, metadata={op_type="Slice" op_name="Slice"}
  ref_dynamic_slice = s32[$1] dynamic-slice(input_tensor, xOffset, yOffset, zOffset), dynamic_slice_sizes={$1}, metadata={op_type="Slice" op_name="Slice"}

  ROOT result = (s32[$1], s32[$1]) tuple(ref_dynamic_slice, replace_dynamic_slice)
}
)";
  return std::make_pair(absl::Substitute(template_hlo, tensor_size, slice_size),
                        offsets);
}

HloAndSliceOffsets Slice2DInputTestCase(const std::string& tensor_size,
                                        const std::string& slice_size,
                                        const std::vector<int>& offsets = {0,
                                                                           0}) {
  const char* template_hlo = R"(
HloModule test

ENTRY test {
  input_tensor = s32[$0] parameter(0)
  offsets = s32[2] parameter(1)

  slice.5 = s32[1] slice(offsets), slice={[0:1]}, metadata={op_type="Slice" op_name="Slice"}
  xOffset = s32[] reshape(slice.5), metadata={op_type="Slice" op_name="Slice"}

  slice.7 = s32[1] slice(offsets), slice={[1:2]}, metadata={op_type="Slice" op_name="lice"}
  yOffset = s32[] reshape(slice.7), metadata={op_type="Slice" op_name="Slice"}

  replace_dynamic_slice = s32[$1] dynamic-slice(input_tensor, xOffset, yOffset), dynamic_slice_sizes={$1}, metadata={op_type="Slice" op_name="Slice"}
  ref_dynamic_slice = s32[$1] dynamic-slice(input_tensor, xOffset, yOffset), dynamic_slice_sizes={$1}, metadata={op_type="Slice" op_name="Slice"}

  ROOT result = (s32[$1], s32[$1]) tuple(ref_dynamic_slice, replace_dynamic_slice)
}
)";
  return std::make_pair(absl::Substitute(template_hlo, tensor_size, slice_size),
                        offsets);
}

HloAndSliceOffsets Slice1DInputTestCase(const std::string& tensor_size,
                                        const std::vector<int>& offsets = {0}) {
  const char* template_hlo = R"(
HloModule test

ENTRY test {
  input_tensor = s32[$0] parameter(0)
  offsets = s32[1] parameter(1)

  slice.5 = s32[1] slice(offsets), slice={[0:1]}, metadata={op_type="Slice" op_name="Slice"}
  xOffset = s32[] reshape(slice.5), metadata={op_type="Slice" op_name="Slice"}

  replace_dynamic_slice = s32[1] dynamic-slice(input_tensor, xOffset), dynamic_slice_sizes={1}, metadata={op_type="Slice" op_name="Slice"}
  ref_dynamic_slice = s32[1] dynamic-slice(input_tensor, xOffset), dynamic_slice_sizes={1}, metadata={op_type="Slice" op_name="Slice"}

  ROOT result = (s32[1], s32[1]) tuple(ref_dynamic_slice, replace_dynamic_slice)
}
)";
  return std::make_pair(absl::Substitute(template_hlo, tensor_size), offsets);
}

HloAndSliceOffsets UpdateSlice3DInputTestCase(
    const std::string& tensor_size, const std::string& slice_size,
    const std::vector<int>& offsets = {0, 0, 0}) {
  const char* template_hlo = R"(
HloModule test
ENTRY test {
  input_tensor = s32[$0] parameter(0)
  offsets = s32[3] parameter(1)

  slice.5 = s32[1] slice(offsets), slice={[0:1]}, metadata={op_type="Slice" op_name="Slice"}
  xOffset = s32[] reshape(slice.5), metadata={op_type="Slice" op_name="Slice"}

  slice.7 = s32[1] slice(offsets), slice={[1:2]}, metadata={op_type="Slice" op_name="Slice"}
  yOffset = s32[] reshape(slice.7), metadata={op_type="Slice" op_name="Slice"}

  slice.9 = s32[1] slice(offsets), slice={[2:3]}, metadata={op_type="Slice" op_name="Slice"}
  zOffset = s32[] reshape(slice.9), metadata={op_type="Slice" op_name="Slice"}

  update_base = s32[] constant(1)
  update = s32[$1] broadcast(update_base), dimensions={}

  replace_dynamic_slice = s32[$0] dynamic-update-slice(input_tensor, update, xOffset, yOffset, zOffset)

  input_tensor_copy = s32[$0] copy(input_tensor)
  ref_dynamic_slice = s32[$0] dynamic-update-slice(input_tensor_copy, update, xOffset, yOffset, zOffset)

  ROOT result = (s32[$0], s32[$0]) tuple(ref_dynamic_slice, replace_dynamic_slice)
}
)";
  return std::make_pair(absl::Substitute(template_hlo, tensor_size, slice_size),
                        offsets);
}
HloAndSliceOffsets UpdateSlice2DInputTestCase(
    const std::string& tensor_size, const std::string& slice_size,
    const std::vector<int>& offsets = {0, 0}) {
  const char* template_hlo = R"(
HloModule test

ENTRY test {
  input_tensor = s32[$0] parameter(0)
  offsets = s32[2] parameter(1)

  slice.5 = s32[1] slice(offsets), slice={[0:1]}, metadata={op_type="Slice" op_name="Slice"}
  xOffset = s32[] reshape(slice.5), metadata={op_type="Slice" op_name="Slice"}

  slice.7 = s32[1] slice(offsets), slice={[1:2]}, metadata={op_type="Slice" op_name="lice"}
  yOffset = s32[] reshape(slice.7), metadata={op_type="Slice" op_name="Slice"}

  update_base = s32[] constant(1)
  update = s32[$1] broadcast(update_base), dimensions={}

  replace_dynamic_slice = s32[$0] dynamic-update-slice(input_tensor, update, xOffset, yOffset)

  input_tensor_copy = s32[$0] copy(input_tensor)
  ref_dynamic_slice = s32[$0] dynamic-update-slice(input_tensor_copy, update, xOffset, yOffset)

  ROOT result = (s32[$0], s32[$0]) tuple(ref_dynamic_slice, replace_dynamic_slice)
}
)";
  return std::make_pair(absl::Substitute(template_hlo, tensor_size, slice_size),
                        offsets);
}

HloAndSliceOffsets UpdateSlice1DInputTestCase(
    const std::string& tensor_size, const std::vector<int>& offsets = {0}) {
  const char* template_hlo = R"(
HloModule test

ENTRY test {
  input_tensor = s32[$0] parameter(0)
  offsets = s32[1] parameter(1)

  slice.5 = s32[1] slice(offsets), slice={[0:1]}, metadata={op_type="Slice" op_name="Slice"}
  xOffset = s32[] reshape(slice.5), metadata={op_type="Slice" op_name="Slice"}

  update = s32[1] constant(2)

  replace_dynamic_slice = s32[$0] dynamic-update-slice(input_tensor, update, xOffset)

  input_tensor_copy = s32[$0] copy(input_tensor)
  ref_dynamic_slice = s32[$0] dynamic-update-slice(input_tensor_copy, update, xOffset)

  ROOT result = (s32[$0], s32[$0]) tuple(ref_dynamic_slice, replace_dynamic_slice)
}
)";
  return std::make_pair(absl::Substitute(template_hlo, tensor_size), offsets);
}

std::vector<HloAndSliceOffsets> SupportedDynamicSliceTestCases() {
  std::vector<HloAndSliceOffsets> test_cases = {
      // These slices are of size 1 in a single dimension.
      //
      // 3d/2d/1d tensors without offsets
      Slice3DInputTestCase("3,4,5", "1,4,5"),
      Slice3DInputTestCase("3,4,5", "3,1,5"),
      Slice3DInputTestCase("3,4,5", "3,4,1"),
      Slice2DInputTestCase("9, 2", "1,2"),
      Slice2DInputTestCase("11, 3", "11,1"),
      Slice1DInputTestCase("5"),
      Slice1DInputTestCase("2"),
      // 3d/2d/1d tensors with offsets. 1d offsets since the
      // slices are also 1d.
      Slice3DInputTestCase("3,4,5", "1,4,5", {1, 0, 0}),
      Slice3DInputTestCase("3,4,5", "3,1,5", {0, 2, 0}),
      Slice3DInputTestCase("3,4,5", "3,4,1", {0, 0, 2}),
      Slice2DInputTestCase("11, 3", "11,1", {0, 1}),
      Slice1DInputTestCase("5", {4}),
  };

  return test_cases;
}

INSTANTIATE_TEST_SUITE_P(DynamicSliceReplacements, DynamicSliceSupportedHloTest,
                         ::testing::ValuesIn(SupportedDynamicSliceTestCases()));

std::vector<HloAndSliceOffsets> SupportedDynamicUpdateTestCases() {
  std::vector<HloAndSliceOffsets> test_cases = {
      // These updates are of size 1 in a single dimension.
      //
      // 3d/2d/1d tensors without offsets
      UpdateSlice3DInputTestCase("10, 5, 8", "1, 5, 8"),
      UpdateSlice3DInputTestCase("10, 5, 8", "10, 1, 8"),
      UpdateSlice3DInputTestCase("10, 5, 8", "10, 5, 1"),
      UpdateSlice2DInputTestCase("10, 5", "10, 1"),
      UpdateSlice2DInputTestCase("10, 10", "1, 10"),
      UpdateSlice2DInputTestCase("3, 1", "1, 1"),
      UpdateSlice1DInputTestCase("5"),
      // 3d/2d/1d tensors with offsets. 1d offsets since the
      // slices are also 1d.
      UpdateSlice1DInputTestCase("5", {3}),
      UpdateSlice2DInputTestCase("3, 1", "1, 1", {2, 0}),
      UpdateSlice2DInputTestCase("3, 4", "3, 1", {0, 3}),
      UpdateSlice3DInputTestCase("10, 5, 8", "1, 5, 8", {8, 0, 0}),
      UpdateSlice3DInputTestCase("10, 5, 8", "10, 1, 8", {0, 3, 0}),
      UpdateSlice3DInputTestCase("10, 5, 8", "10, 5, 1", {0, 0, 5}),
  };

  return test_cases;
}

INSTANTIATE_TEST_SUITE_P(
    DynamicUpdateReplacements, DynamicSliceSupportedHloTest,
    ::testing::ValuesIn(SupportedDynamicUpdateTestCases()));

std::vector<HloAndSliceOffsets> UnsupportedTestCases() {
  std::vector<HloAndSliceOffsets> test_cases = {
      // Slice that are across multiple dimensions.
      Slice3DInputTestCase("3,4,5", "1,1,5"),
      Slice3DInputTestCase("3,4,5", "1,1,1"),
      Slice2DInputTestCase("4,5", "1,1"),
      UpdateSlice3DInputTestCase("10, 5, 8", "1, 1, 8"),
      UpdateSlice3DInputTestCase("10, 5, 8", "1, 1, 1"),
      UpdateSlice2DInputTestCase("10, 8", "1, 1"),
      // Slices with a size > 1.
      Slice3DInputTestCase("3,4,5", "2,4,5"),
      Slice3DInputTestCase("3,4,5", "1,1,1"),
      Slice2DInputTestCase("4,5", "2,1"),
      UpdateSlice3DInputTestCase("10, 5, 8", "2, 5, 8"),
      UpdateSlice2DInputTestCase("10, 8", "2, 1"),
  };

  return test_cases;
}

INSTANTIATE_TEST_SUITE_P(DynamicSliceReplacements,
                         DynamicSliceUnsupportedHloTest,
                         ::testing::ValuesIn(UnsupportedTestCases()));

using DynamicSliceReplacedByPassTest = DynamicSliceHloTest;
TEST_P(DynamicSliceReplacedByPassTest, Replaces) {
  TF_ASSERT_OK_AND_ASSIGN(bool replaced,
                          DynamicSliceReplacer().Run(module_.get()));
  ASSERT_TRUE(replaced);

  auto resources = GetMockResources(device_, module_.get(), 1);
  ASSERT_TRUE(
      ModuleFlatten(resources->annotations).Run(module_.get()).ValueOrDie());
  // Not asserting anything since EmbeddingPlansPreplanning always returns
  // False.
  EmbeddingPlansPreplanning(*resources).Run(module_.get());

  // Throws if we go OOM.
  ASSERT_NO_THROW(Compile(*resources, module_.get()));
}

using DynamicSliceSkippedByPassTest = DynamicSliceHloTest;
TEST_P(DynamicSliceSkippedByPassTest, Skips) {
  TF_ASSERT_OK_AND_ASSIGN(bool replaced,
                          DynamicSliceReplacer().Run(module_.get()));
  ASSERT_FALSE(replaced);

  auto resources = GetMockResources(device_, module_.get(), 1);

  // Setup the dynamicSlices so they get allocated with
  // popops::createSliceTensor.
  TF_ASSERT_OK_AND_ASSIGN(
      bool success, AllocationFinder(resources->annotations,
                                     resources->always_rearrange_copies_on_host)
                        .Run(module_.get()));
  ASSERT_TRUE(success);

  // Throws if we go OOM.
  ASSERT_NO_THROW(Compile(*resources, module_.get()));
}

INSTANTIATE_TEST_SUITE_P(
    DynamicSliceReplacements, DynamicSliceReplacedByPassTest,
    ::testing::Values(Slice2DInputTestCase("20000, 5", "1,5"),
                      Slice3DInputTestCase("20000, 2, 5", "1,2,5"),
                      UpdateSlice2DInputTestCase("20000, 10", "1,10"),
                      UpdateSlice3DInputTestCase("20000, 2, 5", "1,2,5"),
                      Slice2DInputTestCase("100,2", "1,2"),
                      Slice3DInputTestCase("5,2,16", "1,2,16"),
                      Slice3DInputTestCase("5,2,512", "1,2,512"),
                      Slice3DInputTestCase("5,512,512", "1,512,512"),
                      Slice1DInputTestCase("10"),
                      UpdateSlice3DInputTestCase("5,2,512", "1,2,512"),
                      UpdateSlice3DInputTestCase("5,512,512", "1,512,512"),
                      UpdateSlice1DInputTestCase("10")));

// We want these to be skipped since they're not a 1d slice.
INSTANTIATE_TEST_SUITE_P(
    DynamicSliceReplacements, DynamicSliceSkippedByPassTest,
    ::testing::Values(Slice3DInputTestCase("5,2,16", "2,2,16"),
                      Slice3DInputTestCase("20000,1,16", "1,0,16"),
                      UpdateSlice2DInputTestCase("100,5", "1,2")));

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
