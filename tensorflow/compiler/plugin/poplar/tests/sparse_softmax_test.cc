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
#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace poplarplugin {
namespace {

string GetFloatDataType(bool use_half) { return use_half ? "f16" : "f32"; }

string GetIntDataType(bool use_signed) { return use_signed ? "s32" : "u32"; }

string GetTestType(bool is_select) {
  return is_select ? "SelectScalarFromRows" : "UpdateScalarInRows";
}

struct SparseTestSpec {
  SparseTestSpec(int64 batch_size, int64 num_classes, bool allow_out_of_range,
                 bool use_half, bool use_signed)
      : batch_size(batch_size),
        num_classes(num_classes),
        allow_out_of_range(allow_out_of_range),
        use_half(use_half),
        use_signed(use_signed) {}

  int64 batch_size;
  int64 num_classes;
  bool allow_out_of_range;
  bool use_half;
  bool use_signed;

  friend ::std::ostream& operator<<(::std::ostream& os,
                                    const SparseTestSpec& spec) {
    string str = absl::StrCat("batch_size_", spec.batch_size, "_num_classes_",
                              spec.num_classes, "_allow_out_of_range_",
                              spec.allow_out_of_range, "_use_half_",
                              spec.use_half, "_use_signed_", spec.use_signed);
    os << str;
    return os;
  }
};

static std::vector<SparseTestSpec> GetTestCases() {
  std::vector<SparseTestSpec> config_set;
  std::vector<std::vector<int64>> config_options = {
      {1, 1}, {1000, 1}, {1, 1000}, {1000, 1000}};

  for (auto allow_out_of_range : {true, false}) {
    for (auto use_half : {true, false}) {
      for (auto use_signed : {true, false}) {
        for (auto options : config_options) {
          config_set.push_back({options[0], options[1], allow_out_of_range,
                                use_half, use_signed});
        }
      }
    }
  }

  return config_set;
}

string BuildHloText(const SparseTestSpec& spec, bool is_select) {
  const string params_data_type = GetFloatDataType(spec.use_half);
  const string indices_data_type = GetIntDataType(spec.use_signed);
  const string params_shape =
      absl::StrCat(spec.batch_size, ",", spec.num_classes);
  const string indices_shape = std::to_string(spec.batch_size);
  const string output_shape = is_select ? indices_shape : params_shape;
  return absl::StrFormat(
      R"(
    HloModule Test

    ENTRY main {
      params = %s[%s] parameter(0)
      indices = %s[%s] parameter(1)
      ROOT out = %s[%s] custom-call(params, indices), custom_call_target="%s"
    }
    )",
      params_data_type, params_shape, indices_data_type, indices_shape,
      params_data_type, output_shape, GetTestType(is_select));
}

template <typename T>
bool SelectValueOk(Literal& result, Literal& params, Literal& indices,
                   uint32 batch, uint32 num_classes) {
  uint32 index = indices.Get<uint32>({batch});
  float out = static_cast<float>(result.Get<T>({batch}));
  if (index < num_classes) {
    return out == static_cast<float>(params.Get<T>({batch, index}));
  } else {
    return std::isnan(out);
  }
}

class SelectScalarFromRowsTest
    : public HloTestBase,
      public ::testing::WithParamInterface<SparseTestSpec> {};

INSTANTIATE_TEST_CASE_P(SelectScalarFromRowsTest_Instantiation,
                        SelectScalarFromRowsTest,
                        ::testing::ValuesIn(GetTestCases()));

POPLAR_TEST_P(SelectScalarFromRowsTest, DoIt) {
  VLOG(1) << "Test case " << GetParam();
  const SparseTestSpec& spec = GetParam();

  const string hlo_text = BuildHloText(spec, true);
  auto module_or_status =
      HloRunner::CreateModuleFromString(hlo_text, GetDebugOptionsForTest());
  EXPECT_TRUE(module_or_status.ok());
  auto module = std::move(module_or_status.ValueOrDie());

  auto params_shape = ShapeUtil::MakeShape(spec.use_half ? F16 : F32,
                                           {spec.batch_size, spec.num_classes});
  Literal params = DataInitializer::GetDataInitializer("normal")
                       ->GetData(params_shape)
                       .ValueOrDie();

  std::vector<uint32> indices_vals(spec.batch_size);
  std::iota(indices_vals.begin(), indices_vals.end(),
            spec.allow_out_of_range ? 5 : 0);
  Literal indices = LiteralUtil::CreateR1<uint32>(indices_vals);
  Literal result = Execute(std::move(module), {&params, &indices}).ValueOrDie();

  for (int64 batch = 0; batch < spec.batch_size; ++batch) {
    if (spec.use_half) {
      EXPECT_TRUE(SelectValueOk<Eigen::half>(result, params, indices, batch,
                                             spec.num_classes));
    } else {
      EXPECT_TRUE(SelectValueOk<float>(result, params, indices, batch,
                                       spec.num_classes));
    }
  }
}

template <typename T>
bool UpdateValueOk(const float error, Literal& result, Literal& params,
                   Literal& indices, uint32 batch, uint32 num_classes) {
  uint32 index = indices.Get<uint32>({batch});
  if (index < num_classes) {
    return (static_cast<float>(result.Get<T>({batch, index})) -
            static_cast<float>(params.Get<T>({batch, index})) + 1.f) < error;
  } else {
    // Check the whole row at batch is NaN.
    std::vector<uint32> indexes(num_classes);
    std::iota(indexes.begin(), indexes.end(), 0);
    return absl::c_all_of(indexes, [&](const uint32 index) {
      return std::isnan(static_cast<float>(result.Get<T>({batch, index})));
    });
  }
}

class UpdateScalarInRowsTest
    : public HloTestBase,
      public ::testing::WithParamInterface<SparseTestSpec> {};

INSTANTIATE_TEST_CASE_P(UpdateScalarInRowsTest_Instantiation,
                        UpdateScalarInRowsTest,
                        ::testing::ValuesIn(GetTestCases()));

POPLAR_TEST_P(UpdateScalarInRowsTest, DoIt) {
  VLOG(1) << "Test case " << GetParam();
  const SparseTestSpec& spec = GetParam();

  const string hlo_text = BuildHloText(spec, false);
  auto module_or_status =
      HloRunner::CreateModuleFromString(hlo_text, GetDebugOptionsForTest());
  EXPECT_TRUE(module_or_status.ok());
  auto module = std::move(module_or_status.ValueOrDie());

  auto params_shape = ShapeUtil::MakeShape(spec.use_half ? F16 : F32,
                                           {spec.batch_size, spec.num_classes});
  Literal params = DataInitializer::GetDataInitializer("normal")
                       ->GetData(params_shape)
                       .ValueOrDie();

  std::vector<uint32> indices_vals(spec.batch_size);
  std::iota(indices_vals.begin(), indices_vals.end(),
            spec.allow_out_of_range ? 5 : 0);
  Literal indices = LiteralUtil::CreateR1<uint32>(indices_vals);
  Literal result = Execute(std::move(module), {&params, &indices}).ValueOrDie();

  for (int64 batch = 0; batch < spec.batch_size; ++batch) {
    if (spec.use_half) {
      EXPECT_TRUE(UpdateValueOk<Eigen::half>(0.1f, result, params, indices,
                                             batch, spec.num_classes));
    } else {
      EXPECT_TRUE(UpdateValueOk<float>(0.01f, result, params, indices, batch,
                                       spec.num_classes));
    }
  }
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
