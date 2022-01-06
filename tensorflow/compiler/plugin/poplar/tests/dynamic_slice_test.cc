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

PrimitiveType GetFloatDataType(bool use_half) { return use_half ? F16 : F32; }

PrimitiveType GetIntDataType(bool use_signed) { return use_signed ? S32 : U32; }

string GetTestType(bool is_update) {
  return is_update ? "DynamicUpdateSlice" : "DynamicSlice";
}

Literal CreateScalarLiteral(PrimitiveType type, int32 value) {
  if (type == S32) {
    return LiteralUtil::CreateR0<int32>(value);
  } else {
    return LiteralUtil::CreateR0<uint32>(value);
  }
}

struct DynamicSliceTestSpec {
  DynamicSliceTestSpec(const std::vector<int64>& to_slice_shape,
                       const std::vector<int64>& slice_shape,
                       const std::vector<int64>& slice_start_indices,
                       const std::vector<int64>& constant_slice_dims,
                       bool use_half, bool use_signed)
      : to_slice_shape(to_slice_shape),
        slice_shape(slice_shape),
        slice_start_indices(slice_start_indices),
        constant_slice_dims(constant_slice_dims),
        use_half(use_half),
        use_signed(use_signed) {}

  const std::vector<int64> to_slice_shape;
  const std::vector<int64> slice_shape;
  const std::vector<int64> slice_start_indices;
  const std::vector<int64> constant_slice_dims;
  bool use_half;
  bool use_signed;

  friend ::std::ostream& operator<<(::std::ostream& os,
                                    const DynamicSliceTestSpec& spec) {
    string str = absl::StrCat(
        "to_slice_shape_", absl::StrJoin(spec.to_slice_shape, ","),
        "_slice_shape_", absl::StrJoin(spec.slice_shape, ","),
        "_slice_start_indices_", absl::StrJoin(spec.slice_start_indices, ","),
        "_constant_slice_dims_", absl::StrJoin(spec.constant_slice_dims, ","),
        "_use_half_", spec.use_half, "_use_signed_", spec.use_signed);
    os << str;
    return os;
  }
};

static std::vector<DynamicSliceTestSpec> GetTestCases() {
  std::vector<DynamicSliceTestSpec> config_set;

  for (auto use_half : {true, false}) {
    for (auto use_signed : {true, false}) {
      // Case 1: slice is shape of input tensor and all slice dims are
      // constants.
      config_set.push_back(
          {{5, 5, 5}, {5, 5, 5}, {}, {}, use_half, use_signed});
      // Case 2: constant slice.
      config_set.push_back(
          {{5, 5, 5}, {1, 5, 5}, {3}, {0}, use_half, use_signed});
      // Case 3: slice is shape of input tensor and non of the slice dims are
      // constant.
      config_set.push_back(
          {{5, 5, 5}, {5, 5, 5}, {}, {}, use_half, use_signed});
      // Case 4: dynamic slice.
      config_set.push_back(
          {{5, 5, 5}, {5, 1, 5}, {2}, {1}, use_half, use_signed});
      // Case 5: Each dimension is sliced, some const some dynamic.
      config_set.push_back(
          {{5, 5, 5}, {2, 2, 2}, {1, 2, 3}, {1}, use_half, use_signed});
      // Case 6: Slice in dynamic and constant dimensions.
      config_set.push_back({{5, 5, 5, 5, 5, 5},
                            {1, 5, 1, 1, 1, 5},
                            {2, 1, 2, 3},
                            {2, 3},
                            use_half,
                            use_signed});
    }
  }

  return config_set;
}

// Helper struct for storing literal info.
struct TestCase {
  std::unique_ptr<HloComputation> computation;
  Literal input;
  Literal update;
  std::vector<Literal> slice_start_literals;
};

TestCase BuildTestCase(const DynamicSliceTestSpec& spec, bool is_update) {
  auto data_type = GetFloatDataType(spec.use_half);
  auto indices_type = GetIntDataType(spec.use_signed);

  TestCase test_case;

  auto builder = HloComputation::Builder(GetTestType(is_update));
  size_t next_param_index = 0;

  Shape input_shape = ShapeUtil::MakeShape(data_type, spec.to_slice_shape);
  Shape slice_shape = ShapeUtil::MakeShape(data_type, spec.slice_shape);
  // Create the input.
  HloInstruction* input_inst =
      builder.AddInstruction(HloInstruction::CreateParameter(
          next_param_index++, input_shape, "input"));
  test_case.input = DataInitializer::GetDataInitializer("normal")
                        ->GetData(input_shape)
                        .ValueOrDie();

  // Create the slices.
  size_t num_dims = spec.slice_shape.size();
  std::vector<HloInstruction*> slice_start_insts(num_dims);
  size_t num_sliced_dims = 0;
  for (size_t dim = 0; dim != num_dims; ++dim) {
    HloInstruction* slice_start;
    if (spec.to_slice_shape[dim] != spec.slice_shape[dim]) {
      size_t start_index = spec.slice_start_indices[num_sliced_dims++];
      Literal start_index_literal =
          CreateScalarLiteral(indices_type, start_index);
      // Slicing.
      auto itr = absl::c_find(spec.constant_slice_dims, dim);
      if (itr != spec.constant_slice_dims.end()) {
        // The slice is with a constant start_index.
        slice_start = builder.AddInstruction(
            HloInstruction::CreateConstant(std::move(start_index_literal)));
      } else {
        // The slice is with a dynamic start_index.
        slice_start = builder.AddInstruction(HloInstruction::CreateParameter(
            next_param_index++, ShapeUtil::MakeShape(indices_type, {}),
            "start_index"));
        test_case.slice_start_literals.push_back(
            std::move(start_index_literal));
      }
    } else {
      // Not slicing, just create zero.
      slice_start = builder.AddInstruction(
          HloInstruction::CreateConstant(CreateScalarLiteral(indices_type, 0)));
    }
    slice_start_insts[dim] = slice_start;
  }

  // Create the actual dynamic (update) slice instruction.
  absl::optional<Literal> update;
  if (is_update) {
    HloInstruction* update_inst =
        builder.AddInstruction(HloInstruction::CreateParameter(
            next_param_index++, slice_shape, "update"));
    test_case.update = DataInitializer::GetDataInitializer("normal")
                           ->GetData(slice_shape)
                           .ValueOrDie();
    builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
        input_shape, input_inst, update_inst, slice_start_insts));
  } else {
    builder.AddInstruction(HloInstruction::CreateDynamicSlice(
        slice_shape, input_inst, slice_start_insts, slice_shape.dimensions()));
  }

  test_case.computation = builder.Build();
  return test_case;
}

class DynamicUpdateSliceTest
    : public HloTestBase,
      public ::testing::WithParamInterface<DynamicSliceTestSpec> {};

INSTANTIATE_TEST_CASE_P(DynamicUpdateSliceTest_Instantiation,
                        DynamicUpdateSliceTest,
                        ::testing::ValuesIn(GetTestCases()));

POPLAR_TEST_P(DynamicUpdateSliceTest, DoIt) {
  VLOG(1) << "Test case " << GetParam();
  const DynamicSliceTestSpec& spec = GetParam();
  TestCase test_case = BuildTestCase(spec, true);
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(std::move(test_case.computation));

  std::vector<Literal*> inputs(2 + test_case.slice_start_literals.size());
  inputs[0] = &test_case.input;
  absl::c_transform(
      test_case.slice_start_literals, std::next(inputs.begin()),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });
  inputs.back() = &test_case.update;
  Literal result = Execute(std::move(module), inputs).ValueOrDie();
  Literal expected = std::move(test_case.input);
  size_t num_dims = spec.slice_shape.size();
  std::vector<int64> zeros(num_dims, 0);
  std::vector<int64> slice_starts(num_dims, 0);
  std::vector<int64> slice_sizes = spec.slice_shape;
  size_t num_sliced_dims = 0;
  for (size_t dim = 0; dim != num_dims; ++dim) {
    if (spec.to_slice_shape[dim] != spec.slice_shape[dim]) {
      slice_starts[dim] = spec.slice_start_indices[num_sliced_dims++];
    }
  }
  TF_ASSERT_OK(expected.CopySliceFrom(test_case.update, zeros, slice_starts,
                                      slice_sizes));
  EXPECT_TRUE(
      LiteralTestUtil::NearOrEqual(expected, result, ErrorSpec{1e-3, 1e-3}));
}

class DynamicSliceTest
    : public HloTestBase,
      public ::testing::WithParamInterface<DynamicSliceTestSpec> {};

INSTANTIATE_TEST_CASE_P(DynamicSliceTest_Instantiation, DynamicSliceTest,
                        ::testing::ValuesIn(GetTestCases()));

POPLAR_TEST_P(DynamicSliceTest, DoIt) {
  VLOG(1) << "Test case " << GetParam();
  const DynamicSliceTestSpec& spec = GetParam();
  TestCase test_case = BuildTestCase(spec, false);
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(std::move(test_case.computation));

  std::vector<Literal*> inputs(1 + test_case.slice_start_literals.size());
  inputs[0] = &test_case.input;
  absl::c_transform(
      test_case.slice_start_literals, std::next(inputs.begin()),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });
  Literal result = Execute(std::move(module), inputs).ValueOrDie();
  size_t num_dims = spec.slice_shape.size();
  std::vector<int64> slice_starts(num_dims, 0);
  std::vector<int64> slice_ends = spec.to_slice_shape;
  size_t num_sliced_dims = 0;
  for (size_t dim = 0; dim != num_dims; ++dim) {
    if (spec.to_slice_shape[dim] != spec.slice_shape[dim]) {
      slice_starts[dim] = spec.slice_start_indices[num_sliced_dims++];
      slice_ends[dim] = slice_starts[dim] + spec.slice_shape[dim];
    }
  }
  Literal expected = test_case.input.Slice(slice_starts, slice_ends);
  EXPECT_TRUE(
      LiteralTestUtil::NearOrEqual(expected, result, ErrorSpec{1e-3, 1e-3}));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
