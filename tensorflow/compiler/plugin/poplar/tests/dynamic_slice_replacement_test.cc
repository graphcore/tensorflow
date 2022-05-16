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

#include "tensorflow/compiler/plugin/poplar/driver/passes/dynamic_slice_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/module_flatten.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_passes/embedding_plans_preplanning.h"

#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"

#include <poplar/Device.hpp>
#include <poplar/Engine.hpp>

namespace xla {
namespace poplarplugin {
namespace {

void ConnectStream(poplar::Engine& engine, const std::string& name,
                   std::vector<int>& values) {
  engine.connectStream(name, values.data(), values.data() + values.size());
}

StatusOr<std::vector<int>> RunModule(HloModule* module,
                                     std::vector<int> offsets) {
  auto device = HloPoplarTestBase::CreateIpuModel(1, /*num_tiles*/ 8);
  auto resources = HloPoplarTestBase::GetMockResources(device, module);
  ModuleFlatten(resources->annotations).Run(module);
  // EmbeddingPlansPreplanning requires flattened modules.
  EmbeddingPlansPreplanning(*resources).Run(module);

  poplar::Engine engine =
      HloPoplarTestBase::Compile(*resources, module).ValueOrDie();
  HloDescheduler().Run(module);

  engine.load(device);

  auto* entry = module->entry_computation();

  auto* input = entry->parameter_instruction(0);
  auto input_tensor =
      std::vector<int>(ShapeUtil::ElementsIn(input->shape()), 0);
  std::iota(input_tensor.begin(), input_tensor.end(), 0);

  ConnectStream(engine, "0.0", input_tensor);
  ConnectStream(engine, "1.0", offsets);

  auto* root = entry->root_instruction();
  auto out_element_count = ShapeUtil::ElementsIn(root->shape());
  std::vector<int> out(out_element_count, 0);

  ConnectStream(engine, "out_0.0", out);

  engine.run(0);

  return out;
}

using HloAndSliceOffsets = std::pair<std::string, std::vector<int>>;
struct DynamicSliceReplacementHloTest
    : HloTestFixture,
      ::testing::WithParamInterface<HloAndSliceOffsets> {
  void SetUp() override {
    const auto hlo = GetParam().first;
    ASSERT_TRUE(SetUpHloModule(hlo, 1));

    offsets_ = GetParam().second;
  }

  ::testing::AssertionResult InstructionRemoved(const std::string& inst_name) {
    if (FindInstruction(hlo_module_, inst_name)) {
      return ::testing::AssertionFailure()
             << "'" << inst_name
             << "' instruction should be removed but it's still present in "
                "module.";
    }

    return ::testing::AssertionSuccess();
  }

  std::vector<int> offsets_;
};

using DynamicSliceSupportedHloTest = DynamicSliceReplacementHloTest;
TEST_P(DynamicSliceSupportedHloTest, CanReplace) {
  auto instr = FindRootInstruction();
  auto dynamic_slice = Cast<HloDynamicSliceInstruction>(instr);
  auto dynamic_slice_name = dynamic_slice->name();

  const auto expected_parent = dynamic_slice->parent();
  TF_ASSERT_OK_AND_ASSIGN(auto multi_slice,
                          TryReplaceDynamicSliceWithMultiSlice(dynamic_slice));
  ASSERT_TRUE(multi_slice);
  ASSERT_TRUE(IsPoplarInstruction(PoplarOp::MultiSlice, multi_slice));
  ASSERT_EQ(multi_slice->parent(), expected_parent);
  ASSERT_EQ(multi_slice->shape().dimensions(0), 1);

  ASSERT_TRUE(InstructionRemoved(dynamic_slice_name));
}

TEST_P(DynamicSliceSupportedHloTest, CompareReplacedSlice) {
  auto instr = FindRootInstruction();
  auto dynamic_slice = Cast<HloDynamicSliceInstruction>(instr);

  TF_ASSERT_OK_AND_ASSIGN(auto expected_output,
                          RunModule(hlo_module_, offsets_));

  TryReplaceDynamicSliceWithMultiSlice(dynamic_slice);
  TF_ASSERT_OK_AND_ASSIGN(auto replaced_output,
                          RunModule(hlo_module_, offsets_));

  ASSERT_EQ(replaced_output, expected_output);
}

using DynamicSliceUnsupportedHloTest = DynamicSliceReplacementHloTest;
TEST_P(DynamicSliceUnsupportedHloTest, CantReplace) {
  auto instr = FindRootInstruction();
  auto dynamic_slice = Cast<HloDynamicSliceInstruction>(instr);

  TF_ASSERT_OK_AND_ASSIGN(auto multi_slice,
                          TryReplaceDynamicSliceWithMultiSlice(dynamic_slice));
  ASSERT_FALSE(multi_slice);
  ASSERT_EQ(FindRootInstruction(), dynamic_slice);
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

  ROOT replace_dynamic_slice = s32[$1] dynamic-slice(input_tensor, xOffset, yOffset, zOffset), dynamic_slice_sizes={$1}, metadata={op_type="Slice" op_name="Slice"}
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

  ROOT replace_dynamic_slice = s32[$1] dynamic-slice(input_tensor, xOffset, yOffset), dynamic_slice_sizes={$1}, metadata={op_type="Slice" op_name="Slice"}
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

  ROOT replace_dynamic_slice = s32[1] dynamic-slice(input_tensor, xOffset), dynamic_slice_sizes={1}, metadata={op_type="Slice" op_name="Slice"}
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

std::vector<HloAndSliceOffsets> UnsupportedDynamicSliceTestCases() {
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
INSTANTIATE_TEST_SUITE_P(
    DynamicSliceReplacements, DynamicSliceUnsupportedHloTest,
    ::testing::ValuesIn(UnsupportedDynamicSliceTestCases()));

using DynamicUpdateSupportedHloTest = DynamicSliceReplacementHloTest;
TEST_P(DynamicUpdateSupportedHloTest, CanReplace) {
  auto instr = FindRootInstruction();
  auto dynamic_update = Cast<HloDynamicUpdateSliceInstruction>(instr);
  auto dynamic_update_name = dynamic_update->name();

  const auto expected_parent = dynamic_update->parent();
  TF_ASSERT_OK_AND_ASSIGN(
      auto multi_update,
      TryReplaceDynamicUpdateWithMultiUpdate(dynamic_update));
  ASSERT_TRUE(multi_update);
  ASSERT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdate, multi_update));
  ASSERT_EQ(multi_update->parent(), expected_parent);

  ASSERT_TRUE(InstructionRemoved(dynamic_update_name));
}

TEST_P(DynamicUpdateSupportedHloTest, CompareReplacedUpdate) {
  auto instr = FindRootInstruction();
  auto dynamic_update = Cast<HloDynamicUpdateSliceInstruction>(instr);

  TF_ASSERT_OK_AND_ASSIGN(auto expected_output,
                          RunModule(hlo_module_, offsets_));

  TryReplaceDynamicUpdateWithMultiUpdate(dynamic_update);
  TF_ASSERT_OK_AND_ASSIGN(auto replaced_output,
                          RunModule(hlo_module_, offsets_));

  ASSERT_EQ(replaced_output, expected_output);
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

  ROOT replace_dynamic_slice = s32[$0] dynamic-update-slice(input_tensor, update, xOffset, yOffset, zOffset)
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

  ROOT replace_dynamic_slice = s32[$0] dynamic-update-slice(input_tensor, update, xOffset, yOffset)
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

  ROOT replace_dynamic_slice = s32[$0] dynamic-update-slice(input_tensor, update, xOffset)
}
)";
  return std::make_pair(absl::Substitute(template_hlo, tensor_size), offsets);
}

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
    DynamicUpdateReplacements, DynamicUpdateSupportedHloTest,
    ::testing::ValuesIn(SupportedDynamicUpdateTestCases()));

using DynamicUpdateUnsupportedHloTest = DynamicSliceReplacementHloTest;
TEST_P(DynamicUpdateUnsupportedHloTest, CantReplace) {
  auto instr = FindRootInstruction();
  auto dynamic_update = Cast<HloDynamicUpdateSliceInstruction>(instr);

  TF_ASSERT_OK_AND_ASSIGN(
      auto multi_update,
      TryReplaceDynamicUpdateWithMultiUpdate(dynamic_update));
  ASSERT_FALSE(multi_update);
  ASSERT_EQ(FindRootInstruction(), dynamic_update);
}

std::vector<HloAndSliceOffsets> UnsupportedDynamicUpdateTestCases() {
  std::vector<HloAndSliceOffsets> test_cases = {
      // Slice that are across multiple dimensions.
      UpdateSlice3DInputTestCase("10, 5, 8", "1, 1, 8"),
      UpdateSlice3DInputTestCase("10, 5, 8", "1, 1, 1"),
      UpdateSlice2DInputTestCase("10, 8", "1, 1"),
      // Slices with a size > 1.
      UpdateSlice3DInputTestCase("10, 5, 8", "2, 5, 8"),
      UpdateSlice2DInputTestCase("10, 8", "2, 1"),
  };

  return test_cases;
}
INSTANTIATE_TEST_SUITE_P(
    DynamicUpdateReplacements, DynamicUpdateUnsupportedHloTest,
    ::testing::ValuesIn(UnsupportedDynamicUpdateTestCases()));

using DynamicUpdateAddSuportedHloTest = DynamicSliceReplacementHloTest;
TEST_P(DynamicUpdateAddSuportedHloTest, CanReplace) {
  auto instr = FindRootInstruction();
  auto dynamic_update = Cast<HloDynamicUpdateSliceInstruction>(instr);
  auto dynamic_update_name = dynamic_update->name();
  auto expected_parent = dynamic_update->parent();

  auto dynamic_update_add = DynamicUpdateAdd(dynamic_update);

  Cast<HloDynamicSliceInstruction>(FindInstruction(hlo_module_, "slice"));
  TF_ASSERT_OK_AND_ASSIGN(
      auto multi_update_add,
      TryReplaceDynamicUpdateAddWithMultiUpdateAdd(dynamic_update_add));
  ASSERT_TRUE(multi_update_add);
  ASSERT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd, multi_update_add));
  ASSERT_EQ(multi_update_add->parent(), expected_parent);
  ASSERT_NE(multi_update_add->operand(1), dynamic_update_add.add);

  ASSERT_TRUE(InstructionRemoved(dynamic_update_name));
  ASSERT_TRUE(InstructionRemoved("add"));
  ASSERT_TRUE(InstructionRemoved("slice"));
}

TEST_P(DynamicUpdateAddSuportedHloTest, CompareReplacedUpdateAdd) {
  auto instr = FindRootInstruction();
  auto dynamic_update = Cast<HloDynamicUpdateSliceInstruction>(instr);

  TF_ASSERT_OK_AND_ASSIGN(auto expected_output,
                          RunModule(hlo_module_, offsets_));

  auto dynamic_update_add = DynamicUpdateAdd(dynamic_update);
  TF_ASSERT_OK_AND_ASSIGN(
      auto multi_update_add,
      TryReplaceDynamicUpdateAddWithMultiUpdateAdd(dynamic_update_add));
  ASSERT_TRUE(multi_update_add);

  TF_ASSERT_OK_AND_ASSIGN(auto replaced_output,
                          RunModule(hlo_module_, offsets_));

  ASSERT_EQ(expected_output, replaced_output);
}

HloAndSliceOffsets UpdateAdd3DInputTestCase(
    const std::string& tensor_size, const std::string& slice_size,
    bool lhs_add = true, const std::vector<int>& offsets = {0, 0, 0}) {
  const char* template_hlo = R"(
HloModule test

ENTRY test {
  input_tensor = s32[$0] parameter(0)
  offsets = s32[3] parameter(1)

  slice.5 = s32[1] slice(offsets), slice={[0:1]}, metadata={op_type="Slice" op_name="Slice"}
  xOffset = s32[] reshape(slice.5), metadata={op_type="Slice" op_name="Slice"}

  slice.7 = s32[1] slice(offsets), slice={[1:2]}, metadata={op_type="Slice" op_name="Slice"}
  yOffset = s32[] reshape(slice.7), metadata={op_type="Slice" op_name="Slice"}

  slice.8 = s32[1] slice(offsets), slice={[2:3]}, metadata={op_type="Slice" op_name="Slice"}
  zOffset = s32[] reshape(slice.8), metadata={op_type="Slice" op_name="Slice"}

  ones_base = s32[] constant(1)
  ones = s32[$1] broadcast(ones_base), dimensions={}

  slice = s32[$1] dynamic-slice(input_tensor, xOffset, yOffset, zOffset), dynamic_slice_sizes={$1}, metadata={op_type="Slice" op_name="Slice"}
  add = s32[$1] add($2)
  ROOT update = s32[$0] dynamic-update-slice(input_tensor, add, xOffset, yOffset, zOffset)
}
)";
  const auto add_operands = lhs_add ? "slice, ones" : "ones, slice";
  return std::make_pair(
      absl::Substitute(template_hlo, tensor_size, slice_size, add_operands),
      offsets);
}

HloAndSliceOffsets UpdateAdd2DInputTestCase(
    const std::string& tensor_size, const std::string& slice_size,
    bool lhs_add = true, const std::vector<int>& offsets = {0, 0}) {
  const char* template_hlo = R"(
HloModule test

ENTRY test {
  input_tensor = s32[$0] parameter(0)
  offsets = s32[2] parameter(1)

  slice.5 = s32[1] slice(offsets), slice={[0:1]}, metadata={op_type="Slice" op_name="Slice"}
  xOffset = s32[] reshape(slice.5), metadata={op_type="Slice" op_name="Slice"}

  slice.7 = s32[1] slice(offsets), slice={[1:2]}, metadata={op_type="Slice" op_name="Slice"}
  yOffset = s32[] reshape(slice.7), metadata={op_type="Slice" op_name="Slice"}

  ones_base = s32[] constant(1)
  ones = s32[$1] broadcast(ones_base), dimensions={}

  slice = s32[$1] dynamic-slice(input_tensor, xOffset, yOffset), dynamic_slice_sizes={$1}, metadata={op_type="Slice" op_name="Slice"}
  add = s32[$1] add($2)
  ROOT update = s32[$0] dynamic-update-slice(input_tensor, add, xOffset, yOffset)
}
)";
  const auto add_operands = lhs_add ? "slice, ones" : "ones, slice";
  return std::make_pair(
      absl::Substitute(template_hlo, tensor_size, slice_size, add_operands),
      offsets);
}

HloAndSliceOffsets UpdateAdd1DInputTestCase(const std::string& tensor_size,
                                            bool lhs_add = true,
                                            const std::vector<int>& offsets = {
                                                0}) {
  const char* template_hlo = R"(
HloModule test

ENTRY test {
  input_tensor = s32[$0] parameter(0)
  offsets = s32[1] parameter(1)

  slice.5 = s32[1] slice(offsets), slice={[0:1]}, metadata={op_type="Slice" op_name="Slice"}
  xOffset = s32[] reshape(slice.5), metadata={op_type="Slice" op_name="Slice"}

  ones_base = s32[] constant(1)
  ones = s32[$1] broadcast(ones_base), dimensions={}

  slice = s32[1] dynamic-slice(input_tensor, xOffset), dynamic_slice_sizes={1}, metadata={op_type="Slice" op_name="Slice"}
  add = s32[1] add($1)
  ROOT update = s32[$0] dynamic-update-slice(input_tensor, add, xOffset)
}
)";
  const auto add_operands = lhs_add ? "slice, ones" : "ones, slice";
  return std::make_pair(
      absl::Substitute(template_hlo, tensor_size, add_operands), offsets);
}

std::vector<HloAndSliceOffsets> SupportedDynamicUpdateAddTestCases() {
  std::vector<HloAndSliceOffsets> test_cases;

  for (int i = 0; i < 2; ++i) {
    const bool lhs_add = i;
    test_cases.push_back(
        UpdateAdd3DInputTestCase("10, 5, 2", "1, 5, 2", lhs_add));
    test_cases.push_back(
        UpdateAdd3DInputTestCase("10, 5, 2", "10, 1, 2", lhs_add));
    test_cases.push_back(
        UpdateAdd3DInputTestCase("10, 5, 2", "10, 5, 1", lhs_add));
    test_cases.push_back(UpdateAdd2DInputTestCase("10, 5", "1, 5", lhs_add));
    test_cases.push_back(UpdateAdd2DInputTestCase("5, 5", "5, 1", lhs_add));
    test_cases.push_back(
        UpdateAdd3DInputTestCase("10, 5, 2", "1, 5, 2", lhs_add, {4, 0, 0}));
    test_cases.push_back(
        UpdateAdd3DInputTestCase("10, 5, 2", "10, 1, 2", lhs_add, {0, 3, 0}));
    test_cases.push_back(
        UpdateAdd3DInputTestCase("10, 5, 2", "10, 5, 1", lhs_add, {0, 0, 1}));
    test_cases.push_back(
        UpdateAdd2DInputTestCase("10, 5", "1, 5", lhs_add, {5, 0}));
    test_cases.push_back(
        UpdateAdd2DInputTestCase("5, 5", "5, 1", lhs_add, {0, 2}));
  }

  return test_cases;
}
INSTANTIATE_TEST_SUITE_P(
    DynamicUpdateAddReplacements, DynamicUpdateAddSuportedHloTest,
    ::testing::ValuesIn(SupportedDynamicUpdateAddTestCases()));

using DynamicUpdateAddUnsupportedHloTest = DynamicSliceReplacementHloTest;
TEST_P(DynamicUpdateAddUnsupportedHloTest, CantReplace) {
  auto instr = FindInstruction(hlo_module_, "update");
  auto dynamic_update = Cast<HloDynamicUpdateSliceInstruction>(instr);

  auto dynamic_update_add = DynamicUpdateAdd(dynamic_update);
  TF_ASSERT_OK_AND_ASSIGN(
      auto multi_update,
      TryReplaceDynamicUpdateAddWithMultiUpdateAdd(dynamic_update_add));
  ASSERT_FALSE(multi_update);
  ASSERT_TRUE(FindInstruction(hlo_module_, "update"));
  ASSERT_TRUE(FindInstruction(hlo_module_, "slice"));
  ASSERT_TRUE(FindInstruction(hlo_module_, "add"));
}

std::vector<HloAndSliceOffsets> UnsupportedDynamicUpdateAddTestCases() {
  // This dynamic_update_add is not replaceable since the add/slice
  // are being used outside of the dynamic_update.
  const char* unreplacable_dynamic_update_add = R"(
HloModule test

ENTRY test {
  input_tensor = s32[$0] parameter(0)
  offsets = s32[2] parameter(1)

  slice.5 = s32[1] slice(offsets), slice={[0:1]}, metadata={op_type="Slice" op_name="Slice"}
  xOffset = s32[] reshape(slice.5), metadata={op_type="Slice" op_name="Slice"}

  slice.7 = s32[1] slice(offsets), slice={[1:2]}, metadata={op_type="Slice" op_name="Slice"}
  yOffset = s32[] reshape(slice.7), metadata={op_type="Slice" op_name="Slice"}

  ones_base = s32[] constant(1)
  ones = s32[$1] broadcast(ones_base), dimensions={}

  slice = s32[$1] dynamic-slice(input_tensor, xOffset, yOffset), dynamic_slice_sizes={$1}, metadata={op_type="Slice" op_name="Slice"}
  add = s32[$1] add(slice, ones)
  update = s32[$0] dynamic-update-slice(input_tensor, add, xOffset, yOffset)
  ROOT output = (s32[$0], s32[$1]) tuple(update, $2)
}
)";
  const auto using_add_outside_dynamic_update_add =
      absl::Substitute(unreplacable_dynamic_update_add, "20,10", "1,10", "add");
  const auto using_slice_outside_dynamic_update_add = absl::Substitute(
      unreplacable_dynamic_update_add, "20,10", "1,10", "slice");

  std::vector<HloAndSliceOffsets> test_cases;
  test_cases.emplace_back(using_add_outside_dynamic_update_add,
                          std::vector<int>{0, 0});
  test_cases.emplace_back(using_slice_outside_dynamic_update_add,
                          std::vector<int>{0, 0});

  for (int i = 0; i < 2; ++i) {
    const bool lhs_add = i;
    test_cases.push_back(
        UpdateAdd3DInputTestCase("10, 5, 2", "2, 5, 2", lhs_add));
    test_cases.push_back(UpdateAdd2DInputTestCase("10, 5", "2, 5", lhs_add));
    test_cases.push_back(
        UpdateAdd3DInputTestCase("10, 5, 2", "2, 5, 2", lhs_add, {4, 0, 0}));
    test_cases.push_back(
        UpdateAdd3DInputTestCase("10, 5, 2", "10, 2, 2", lhs_add, {0, 3, 0}));
    test_cases.push_back(
        UpdateAdd2DInputTestCase("5, 5", "5, 2", lhs_add, {0, 2}));
  }

  return test_cases;
}
INSTANTIATE_TEST_SUITE_P(
    DynamicUpdateAddReplacements, DynamicUpdateAddUnsupportedHloTest,
    ::testing::ValuesIn(UnsupportedDynamicUpdateAddTestCases()));

using DynamicSliceReplacementPassHloTest = DynamicSliceReplacementHloTest;
TEST_P(DynamicSliceReplacementPassHloTest, Replaces) {
  TF_ASSERT_OK_AND_ASSIGN(auto expected_output,
                          RunModule(hlo_module_, offsets_));

  TF_ASSERT_OK_AND_ASSIGN(bool replaced,
                          DynamicSliceReplacer().Run(hlo_module_));
  ASSERT_TRUE(replaced);
  TF_ASSERT_OK_AND_ASSIGN(auto replaced_output,
                          RunModule(hlo_module_, offsets_));

  ASSERT_EQ(replaced_output, expected_output);
}

using DynamicSliceSkippedByPassHloTest = DynamicSliceReplacementHloTest;
TEST_P(DynamicSliceSkippedByPassHloTest, Skips) {
  TF_ASSERT_OK_AND_ASSIGN(bool replaced,
                          DynamicSliceReplacer().Run(hlo_module_));
  ASSERT_FALSE(replaced);
}

INSTANTIATE_TEST_SUITE_P(
    DynamicSliceReplacements, DynamicSliceReplacementPassHloTest,
    ::testing::Values(Slice2DInputTestCase("2000, 5", "1,5"),
                      Slice3DInputTestCase("2000, 2, 5", "1,2,5"),
                      UpdateSlice2DInputTestCase("2000, 10", "1,10"),
                      UpdateSlice3DInputTestCase("2000, 2, 5", "1,2,5"),
                      Slice2DInputTestCase("100,2", "1,2"),
                      Slice3DInputTestCase("5,2,16", "1,2,16"),
                      Slice3DInputTestCase("5,2,512", "1,2,512"),
                      Slice1DInputTestCase("10"),
                      UpdateSlice3DInputTestCase("5,2,512", "1,2,512"),
                      UpdateSlice1DInputTestCase("10")));

// We want these to be skipped since they're not a 1d slice.
INSTANTIATE_TEST_SUITE_P(
    DynamicSliceReplacements, DynamicSliceSkippedByPassHloTest,
    ::testing::Values(Slice3DInputTestCase("5,2,16", "2,2,16"),
                      Slice3DInputTestCase("20000,1,16", "1,0,16"),
                      UpdateSlice2DInputTestCase("100,5", "1,2")));

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
