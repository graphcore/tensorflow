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

#include "absl/types/optional.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/execution_counter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
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

struct CreateBufferTestSpec {
  explicit CreateBufferTestSpec(bool is_remote) : is_remote(is_remote) {}

  const bool is_remote;

  friend ::std::ostream& operator<<(::std::ostream& os,
                                    const CreateBufferTestSpec& spec) {
    string str = absl::StrCat("is_remote_", spec.is_remote);
    os << str;
    return os;
  }
};

static std::vector<CreateBufferTestSpec> GetTestCases() {
  // TODO(T10880) enable remote buffer test when running on HW.
  return {CreateBufferTestSpec(false)};
}

string GetTestType(bool is_remote) { return is_remote ? "Remote" : "InMemory"; }

class CreateBufferTest
    : public HloTestBase,
      public ::testing::WithParamInterface<CreateBufferTestSpec> {};

INSTANTIATE_TEST_CASE_P(CreateBufferTest_Instantiation, CreateBufferTest,
                        ::testing::ValuesIn(GetTestCases()));

POPLAR_TEST_P(CreateBufferTest, DoIt) {
  VLOG(1) << "Test case " << GetParam();
  const CreateBufferTestSpec& spec = GetParam();

  auto hlo_module = CreateNewVerifiedModule();
  const int64_t num_iterations = 50;

  Shape counter_shape = ShapeUtil::MakeShape(S32, {});
  Shape buffer_shape = ShapeUtil::MakeShape(F32, {num_iterations, 128});
  Shape slice_shape = ShapeUtil::MakeShape(F32, {1, 128});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({counter_shape, buffer_shape});

  // While cond.
  HloComputation* comp_cond;
  {
    auto builder_cond = HloComputation::Builder(TestName());
    auto tuple = builder_cond.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "cond_tuple"));
    auto limit = builder_cond.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<int32>(num_iterations)));
    auto c = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 0));
    builder_cond.AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::MakeShape(PRED, {}), c, limit, ComparisonDirection::kLt));

    comp_cond = hlo_module->AddEmbeddedComputation(builder_cond.Build());
  }

  // While body.
  HloComputation* comp_body;
  {
    auto builder_body = HloComputation::Builder(TestName());
    auto tuple_body = builder_body.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "body_tuple"));

    // Increase counter by 1.
    auto c_body = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(counter_shape, tuple_body, 0));
    auto zero = builder_body.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
    auto one = builder_body.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
    auto new_c_body = builder_body.AddInstruction(HloInstruction::CreateBinary(
        c_body->shape(), HloOpcode::kAdd, c_body, one));

    auto in_body = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(buffer_shape, tuple_body, 1));

    auto counter = builder_body.AddInstruction(CreateExecutionCounter());
    IPUCustomKernelsUtil::AttributeMap attribute_map;
    attribute_map.AddAttribute("is_remote", spec.is_remote);
    auto buffer = builder_body.AddInstruction(HloInstruction::CreateCustomCall(
        buffer_shape, {}, PoplarOp_Name(PoplarOp::CreateBuffer),
        attribute_map.Serialise()));

    // Get a slice, copy it into the buffer and then copy it back.
    auto* slice =
        builder_body.AddInstruction(HloInstruction::CreateDynamicSlice(
            slice_shape, in_body, {counter, zero}, {1, 128}));
    auto* buffer_updated =
        builder_body.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            buffer_shape, buffer, slice, {counter, zero}));

    // Overwrite the values in "in".
    auto zero_fp32 = builder_body.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));
    auto* broadcast = builder_body.AddInstruction(
        HloInstruction::CreateBroadcast(slice_shape, zero_fp32, {}));
    in_body =
        builder_body.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            buffer_shape, in_body, broadcast, {counter, zero}));

    auto* buffer_slice =
        builder_body.AddInstruction(HloInstruction::CreateDynamicSlice(
            ShapeUtil::MakeShape(F32, {1, 128}), buffer_updated,
            {counter, zero}, {1, 128}));
    in_body =
        builder_body.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            buffer_shape, in_body, buffer_slice, {counter, zero}));

    builder_body.AddInstruction(
        HloInstruction::CreateTuple({new_c_body, in_body}));

    comp_body = hlo_module->AddEmbeddedComputation(builder_body.Build());
  }

  /* Create main computation */
  auto builder_main = HloComputation::Builder(GetTestType(spec.is_remote));
  auto c = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  auto in = builder_main.AddInstruction(
      HloInstruction::CreateParameter(0, buffer_shape, "in"));

  auto init = builder_main.AddInstruction(HloInstruction::CreateTuple({c, in}));

  auto while_loop = builder_main.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));
  builder_main.AddInstruction(
      HloInstruction::CreateGetTupleElement(buffer_shape, while_loop, 1));

  hlo_module->AddEntryComputation(builder_main.Build());

  Literal input = DataInitializer::GetDataInitializer("normal")
                      ->GetData(buffer_shape)
                      .ValueOrDie();

  Literal result = Execute(std::move(hlo_module), {&input}).ValueOrDie();
  EXPECT_TRUE(LiteralTestUtil::Equal(input, result));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
