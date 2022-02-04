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

using HloAndSliceOffsets = std::pair<std::string, std::vector<int>>;
struct DynamicSliceHloTest : HloPoplarTestBase,
                             ::testing::WithParamInterface<HloAndSliceOffsets> {
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(device_, CreateIpuDevice(1));
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
      poplar::Engine& engine, const Shape& multi_slice,
      const Shape& dynamic_slice) {
    engine.load(device_);

    // Setup matrix with incrementing values.
    std::iota(input_matrix_.begin(), input_matrix_.end(), 0);
    ConnectStream(engine, "0.0", input_matrix_);

    ConnectStream(engine, "1.0", offsets_vector_);

    std::vector<int> multislice_out(ShapeUtil::ElementsIn(multi_slice), 0);
    ConnectStream(engine, "out_0.0", multislice_out);

    std::vector<int> dynamicslice_out(ShapeUtil::ElementsIn(dynamic_slice), 0);
    ConnectStream(engine, "out_1.0", dynamicslice_out);

    engine.run(0);

    return {multislice_out, dynamicslice_out};
  }

  poplar::Device device_;
  std::unique_ptr<VerifiedHloModule> module_;

  std::vector<int> input_matrix_;
  std::vector<int> offsets_vector_;
};

using DynamicSliceSupportedHloTest = DynamicSliceHloTest;

TEST_P(DynamicSliceSupportedHloTest, CheckCanReplaceSlice) {
  auto instr = FindInstruction(module_.get(), "replace_dynamic_slice");
  ASSERT_TRUE(instr);

  auto dynamic_slice = Cast<HloDynamicSliceInstruction>(instr);
  TF_ASSERT_OK_AND_ASSIGN(auto multi_slice,
                          TryReplaceDynamicWithMultiSlice(dynamic_slice));
  ASSERT_TRUE(multi_slice);
  // Even though it's been removed dynamic_slice is valid until
  // HloComputation::Cleanup is called.
  ASSERT_EQ(dynamic_slice->parent(), nullptr);

  ASSERT_TRUE(IsPoplarInstruction(PoplarOp::MultiSlice, multi_slice));
  ASSERT_EQ(multi_slice->shape(), dynamic_slice->shape());
}

TEST_P(DynamicSliceSupportedHloTest, CompareReplacedSlice) {
  auto instr = FindInstruction(module_.get(), "replace_dynamic_slice");
  ASSERT_TRUE(instr);

  auto dynamic_slice = Cast<HloDynamicSliceInstruction>(instr);
  TF_ASSERT_OK_AND_ASSIGN(auto multi_slice,
                          TryReplaceDynamicWithMultiSlice(dynamic_slice));

  // EmbeddingPlansPreplanning requires flattened modules.
  auto resources = GetMockResources(device_, module_.get(), 1);
  ASSERT_TRUE(
      ModuleFlatten(resources->annotations).Run(module_.get()).ValueOrDie());
  // Not asserting anything since EmbeddingPlansPreplanning always returns
  // False.
  EmbeddingPlansPreplanning(*resources).Run(module_.get());

  TF_ASSERT_OK_AND_ASSIGN(poplar::Engine engine,
                          Compile(*resources, module_.get()));
  auto results =
      RunEngine(engine, multi_slice->shape(), dynamic_slice->shape());
  auto multislice_out = results.first;
  auto dynamicslice_out = results.second;

  ASSERT_EQ(multislice_out, dynamicslice_out);
}

using DynamicSliceUnsupportedHloTest = DynamicSliceHloTest;

TEST_P(DynamicSliceUnsupportedHloTest, CantReplace) {
  auto instr = FindInstruction(module_.get(), "replace_dynamic_slice");
  ASSERT_TRUE(instr);

  auto dynamic_slice = Cast<HloDynamicSliceInstruction>(instr);
  TF_ASSERT_OK_AND_ASSIGN(auto multi_slice,
                          TryReplaceDynamicWithMultiSlice(dynamic_slice));
  ASSERT_FALSE(multi_slice);
  ASSERT_EQ(dynamic_slice->parent(), module_->entry_computation());
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

std::vector<HloAndSliceOffsets> SupportedTestCases() {
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
                         ::testing::ValuesIn(SupportedTestCases()));

std::vector<HloAndSliceOffsets> UnsupportedTestCases() {
  std::vector<HloAndSliceOffsets> test_cases = {
      // Slice that are across multiple dimensions.
      Slice3DInputTestCase("3,4,5", "1,1,5"),
      Slice3DInputTestCase("3,4,5", "1,1,1"),
      Slice2DInputTestCase("4,5", "1,1"),
      // Slices with a size > 1.
      Slice3DInputTestCase("3,4,5", "2,4,5"),
      Slice3DInputTestCase("3,4,5", "1,1,1"),
      Slice2DInputTestCase("4,5", "2,1"),
  };

  return test_cases;
}

INSTANTIATE_TEST_SUITE_P(DynamicSliceReplacements,
                         DynamicSliceUnsupportedHloTest,
                         ::testing::ValuesIn(UnsupportedTestCases()));

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
