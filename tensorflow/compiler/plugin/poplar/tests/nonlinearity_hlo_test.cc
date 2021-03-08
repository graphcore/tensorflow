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

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

namespace xla {
namespace poplarplugin {
namespace {

struct TestParam {
  PoplarOp op;
  std::string hlo;

  friend std::ostream& operator<<(std::ostream& ostream,
                                  const TestParam& param) {
    ostream << "Op=" << PoplarOp_Name(param.op) << ", HLO=" << param.hlo;
    return ostream;
  }
};

class NonLinearityHloTest : public HloTestBase,
                            public ::testing::WithParamInterface<TestParam> {
 public:
  static std::string ParamName(
      const ::testing::TestParamInfo<NonLinearityHloTest::ParamType>& info) {
    return PoplarOp_Name(info.param.op);
  }

  void SetUp() override {
    auto moduleResult = ParseAndReturnVerifiedModule(GetParam().hlo);
    ASSERT_TRUE(moduleResult.ok());

    module_ = moduleResult.ConsumeValueOrDie();
  }

  std::unique_ptr<VerifiedHloModule> module_;
};

TEST_P(NonLinearityHloTest, PoplarOpIsRegistered) {
  ASSERT_TRUE(PoplarOpManager::HasOp(GetParam().op));
}

TEST_P(NonLinearityHloTest, CustomCallIsSupported) {
  auto root = module_->entry_computation()->root_instruction();
  auto call = DynCast<HloCustomCallInstruction>(root);

  ASSERT_TRUE(call);
  ASSERT_TRUE(HloPoplarInstructionFactory::IsCreatable(call));
}

TEST_P(NonLinearityHloTest, CustomCallIsReplacedWithPoplarOpsCall) {
  CustomOpReplacer replacer;
  const auto status = replacer.Run(module_.get());
  ASSERT_TRUE(status.ValueOrDie());

  const auto rootInstruction = module_->entry_computation()->root_instruction();
  ASSERT_TRUE(IsPoplibsHloCustomOp(rootInstruction));
  ASSERT_TRUE(IsPoplarInstruction(GetParam().op, rootInstruction));
}

static const TestParam relu = {PoplarOp::Relu, R"(
HloModule top

ENTRY test {
arg0.1 = f16[10,20] parameter(0), metadata={op_name="XLA_Args/_arg_Placeholder_0_0/_q1"}
ROOT relu = f16[10,20] custom-call(arg0.1), custom_call_target="Relu"
}
)"};

static const TestParam reluGrad = {PoplarOp::ReluGrad, R"(
HloModule top

ENTRY test {
arg0.1 = f32[5,10] parameter(0), metadata={op_name="XLA_Args/_arg_Placeholder_54_0_0/_1"}
arg1.2 = f32[10,6] parameter(1), metadata={op_name="XLA_Args/_arg_Placeholder_55_0_1/_3"}
dot.5 = f32[5,6] dot(arg0.1, arg1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
relu = f32[5,6] custom-call(dot.5), custom_call_target="Relu"
ROOT grad = f32[5,6] custom-call(dot.5, relu), custom_call_target="ReluGrad"
}
)"};

static const TestParam gelu = {PoplarOp::Gelu, R"(
HloModule top

ENTRY test {
arg0.1 = f16[10,20] parameter(0), metadata={op_name="XLA_Args/_arg_Placeholder_0_0/_q1"}
ROOT gelu = f16[10,20] custom-call(arg0.1), custom_call_target="Gelu"
}
)"};

static const TestParam geluGrad = {PoplarOp::GeluGrad, R"(
HloModule top

ENTRY test {
arg0.1 = f32[5,10] parameter(0), metadata={op_name="XLA_Args/_arg_Placeholder_54_0_0/_1"}
arg1.2 = f32[10,6] parameter(1), metadata={op_name="XLA_Args/_arg_Placeholder_55_0_1/_3"}
dot.5 = f32[5,6] dot(arg0.1, arg1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
gelu = f32[5,6] custom-call(dot.5), custom_call_target="Gelu"
ROOT grad = f32[5,6] custom-call(dot.5, gelu), custom_call_target="GeluGrad"
}
)"};

static const TestParam sigmoid = {PoplarOp::Sigmoid, R"(
HloModule top

ENTRY test {
arg0.1 = f16[10,20] parameter(0), metadata={op_name="XLA_Args/_arg_Placeholder_0_0/_q1"}
ROOT sigmoid = f16[10,20] custom-call(arg0.1), custom_call_target="Sigmoid"
}
)"};

static const TestParam sigmoidGrad = {PoplarOp::SigmoidGrad, R"(
HloModule top

ENTRY test {
arg0.1 = f32[5,10] parameter(0), metadata={op_name="XLA_Args/_arg_Placeholder_54_0_0/_1"}
arg1.2 = f32[10,6] parameter(1), metadata={op_name="XLA_Args/_arg_Placeholder_55_0_1/_3"}
dot.5 = f32[5,6] dot(arg0.1, arg1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
sigmoid = f32[5,6] custom-call(dot.5), custom_call_target="Sigmoid"
ROOT grad = f32[5,6] custom-call(dot.5, sigmoid), custom_call_target="SigmoidGrad"
}
)"};

static const TestParam hardSigmoid = {PoplarOp::HardSigmoid, R"(
HloModule top

ENTRY test {
arg0.1 = f16[10,20] parameter(0), metadata={op_name="XLA_Args/_arg_Placeholder_0_0/_q1"}
ROOT hardSigmoid = f16[10,20] custom-call(arg0.1), custom_call_target="HardSigmoid"
}
)"};

static const TestParam hardSigmoidGrad = {PoplarOp::HardSigmoidGrad, R"(
HloModule top

ENTRY test {
arg0.1 = f32[5,10] parameter(0), metadata={op_name="XLA_Args/_arg_Placeholder_54_0_0/_1"}
arg1.2 = f32[10,6] parameter(1), metadata={op_name="XLA_Args/_arg_Placeholder_55_0_1/_3"}
dot.5 = f32[5,6] dot(arg0.1, arg1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
hardSigmoid = f32[5,6] custom-call(dot.5), custom_call_target="HardSigmoid"
ROOT grad = f32[5,6] custom-call(dot.5, hardSigmoid), custom_call_target="HardSigmoidGrad"
}
)"};

static const TestParam swish = {PoplarOp::Swish, R"(
HloModule top

ENTRY test {
arg0.1 = f16[10,20] parameter(0), metadata={op_name="XLA_Args/_arg_Placeholder_0_0/_q1"}
ROOT swish = f16[10,20] custom-call(arg0.1), custom_call_target="Swish"
}
)"};

static const TestParam swishGrad = {PoplarOp::SwishGrad, R"(
HloModule top

ENTRY test {
arg0.1 = f32[5,10] parameter(0), metadata={op_name="XLA_Args/_arg_Placeholder_54_0_0/_1"}
arg1.2 = f32[10,6] parameter(1), metadata={op_name="XLA_Args/_arg_Placeholder_55_0_1/_3"}
dot.5 = f32[5,6] dot(arg0.1, arg1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
swish = f32[5,6] custom-call(dot.5), custom_call_target="Swish"
ROOT grad = f32[5,6] custom-call(dot.5, swish), custom_call_target="SwishGrad"
}
)"};

INSTANTIATE_TEST_SUITE_P(
    NonLinearityTest, NonLinearityHloTest,
    ::testing::Values(relu, gelu, sigmoid, hardSigmoid, swish, reluGrad,
                      geluGrad, sigmoidGrad, hardSigmoidGrad, swishGrad),
    std::bind(NonLinearityHloTest::ParamName, std::placeholders::_1));

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
