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
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_late.h"

namespace xla {
namespace poplarplugin {
namespace {

class ScaledArithmeticTest : public HloTestBase,
                             public ::testing::WithParamInterface<const char*> {
 public:
  void SetUp() override {
    auto module_result = ParseAndReturnVerifiedModule(GetParam());
    ASSERT_TRUE(module_result.ok());

    module_ = module_result.ConsumeValueOrDie();
    annotations_.reset(new CompilerAnnotations(module_.get()));
  }

  ::testing::AssertionResult EntryContainsNoCasts(const HloComputation* entry) {
    for (auto* instruction : entry->instructions()) {
      if (instruction->opcode() == HloOpcode::kConvert) {
        return ::testing::AssertionFailure();
      }
    }
    return ::testing::AssertionSuccess();
  }

  ::testing::AssertionResult FusedCallContainsNoCasts(
      const HloComputation* entry) {
    auto fused_call = entry->root_instruction();
    for (auto operand : fused_call->operands()) {
      if (operand->opcode() == HloOpcode::kConvert) {
        return ::testing::AssertionFailure();
      }
    }
    return ::testing::AssertionSuccess();
  }

  std::unique_ptr<VerifiedHloModule> module_;
  std::unique_ptr<CompilerAnnotations> annotations_;
};

class AXBYTest : public ScaledArithmeticTest {
 public:
  static std::string TestName(
      const ::testing::TestParamInfo<AXBYTest::ParamType>& info) {
    if (info.param == axPlusByWithBroadcastConvert) {
      return "axPlusByWithBroadcastConvert";
    }

    if (info.param == axPlusByWithBroadcastReshapeConvert) {
      return "axPlusByWithBroadcastReshapeConvert";
    }

    if (info.param == axMinusByWithBroadcastReshapeConvert) {
      return "axMinusByWithBroadcastReshapeConvert";
    }

    if (info.param == axMinusByWithBroadcastConvert) {
      return "axMinusByWithBroadcastConvert";
    }

    return "Unknown";
  }

  static constexpr const char* axPlusByWithBroadcastConvert = R"(
HloModule top

ENTRY test {
  %arg_0 = f16[3]{0} parameter(0)
  %arg_2 = f32[] parameter(2)
  %convert.9.clone = f16[] convert(f32[] %arg_2), metadata={op_type="Cast" op_name="Cast"}
  %broadcast.10.clone = f16[3]{0} broadcast(f16[] %convert.9.clone), dimensions={}, metadata={op_type="Mul" op_name="mul"}
  %multiply.11.clone = f16[3]{0} multiply(f16[3]{0} %arg_0, f16[3]{0} %broadcast.10.clone), metadata={op_type="Mul" op_name="mul"}
  %arg_1 = f16[3]{0} parameter(1)
  %arg_3 = f32[] parameter(3)
  %convert.12.clone = f16[] convert(f32[] %arg_3), metadata={op_type="Cast" op_name="Cast_1"}
  %broadcast.13.clone = f16[3]{0} broadcast(f16[] %convert.12.clone), dimensions={}, metadata={op_type="Mul" op_name="mul_1"}
  %multiply.14.clone = f16[3]{0} multiply(f16[3]{0} %arg_1, f16[3]{0} %broadcast.13.clone), metadata={op_type="Mul" op_name="mul_1"}
  ROOT %add.15.clone = f16[3]{0} add(f16[3]{0} %multiply.11.clone, f16[3]{0} %multiply.14.clone), metadata={op_type="AddV2" op_name="add"}
}
)";

  static constexpr const char* axPlusByWithBroadcastReshapeConvert = R"(
    HloModule top

ENTRY test {
  %arg_0 = f16[3]{0} parameter(0)
  %arg_2 = f32[1]{0} parameter(2)
  %convert.9.clone = f16[1]{0} convert(f32[1]{0} %arg_2), metadata={op_type="Cast" op_name="Cast"}
  %reshape.10.clone = f16[] reshape(f16[1]{0} %convert.9.clone), metadata={op_type="Mul" op_name="mul"}
  %broadcast.11.clone = f16[3]{0} broadcast(f16[] %reshape.10.clone), dimensions={}, metadata={op_type="Mul" op_name="mul"}
  %multiply.12.clone = f16[3]{0} multiply(f16[3]{0} %arg_0, f16[3]{0} %broadcast.11.clone), metadata={op_type="Mul" op_name="mul"}
  %arg_1 = f16[3]{0} parameter(1)
  %arg_3 = f32[1]{0} parameter(3)
  %convert.13.clone = f16[1]{0} convert(f32[1]{0} %arg_3), metadata={op_type="Cast" op_name="Cast_1"}
  %reshape.14.clone = f16[] reshape(f16[1]{0} %convert.13.clone), metadata={op_type="Mul" op_name="mul_1"}
  %broadcast.15.clone = f16[3]{0} broadcast(f16[] %reshape.14.clone), dimensions={}, metadata={op_type="Mul" op_name="mul_1"}
  %multiply.16.clone = f16[3]{0} multiply(f16[3]{0} %arg_1, f16[3]{0} %broadcast.15.clone), metadata={op_type="Mul" op_name="mul_1"}
  ROOT %add.17.clone = f16[3]{0} add(f16[3]{0} %multiply.12.clone, f16[3]{0} %multiply.16.clone), metadata={op_type="AddV2" op_name="add"}
}
)";

  static constexpr const char* axMinusByWithBroadcastReshapeConvert = R"(
    HloModule top

ENTRY test {
  %arg_0 = f16[3]{0} parameter(0)
  %arg_2 = f32[1]{0} parameter(2)
  %convert.9.clone = f16[1]{0} convert(f32[1]{0} %arg_2), metadata={op_type="Cast" op_name="Cast"}
  %reshape.10.clone = f16[] reshape(f16[1]{0} %convert.9.clone), metadata={op_type="Mul" op_name="mul"}
  %broadcast.11.clone = f16[3]{0} broadcast(f16[] %reshape.10.clone), dimensions={}, metadata={op_type="Mul" op_name="mul"}
  %multiply.12.clone = f16[3]{0} multiply(f16[3]{0} %arg_0, f16[3]{0} %broadcast.11.clone), metadata={op_type="Mul" op_name="mul"}
  %arg_1 = f16[3]{0} parameter(1)
  %arg_3 = f32[1]{0} parameter(3)
  %convert.13.clone = f16[1]{0} convert(f32[1]{0} %arg_3), metadata={op_type="Cast" op_name="Cast_1"}
  %reshape.14.clone = f16[] reshape(f16[1]{0} %convert.13.clone), metadata={op_type="Mul" op_name="mul_1"}
  %broadcast.15.clone = f16[3]{0} broadcast(f16[] %reshape.14.clone), dimensions={}, metadata={op_type="Mul" op_name="mul_1"}
  %multiply.16.clone = f16[3]{0} multiply(f16[3]{0} %arg_1, f16[3]{0} %broadcast.15.clone), metadata={op_type="Mul" op_name="mul_1"}
  ROOT %subtract.17.clone = f16[3]{0} subtract(f16[3]{0} %multiply.12.clone, f16[3]{0} %multiply.16.clone), metadata={op_type="Sub" op_name="sub"}
}
)";

  static constexpr const char* axMinusByWithBroadcastConvert = R"(
HloModule top

ENTRY test {
  %arg_0 = f16[3]{0} parameter(0)
  %arg_2 = f32[] parameter(2)
  %convert.9.clone = f16[] convert(f32[] %arg_2), metadata={op_type="Cast" op_name="Cast"}
  %broadcast.10.clone = f16[3]{0} broadcast(f16[] %convert.9.clone), dimensions={}, metadata={op_type="Mul" op_name="mul"}
  %multiply.11.clone = f16[3]{0} multiply(f16[3]{0} %arg_0, f16[3]{0} %broadcast.10.clone), metadata={op_type="Mul" op_name="mul"}
  %arg_1 = f16[3]{0} parameter(1)
  %arg_3 = f32[] parameter(3)
  %convert.12.clone = f16[] convert(f32[] %arg_3), metadata={op_type="Cast" op_name="Cast_1"}
  %broadcast.13.clone = f16[3]{0} broadcast(f16[] %convert.12.clone), dimensions={}, metadata={op_type="Mul" op_name="mul_1"}
  %multiply.14.clone = f16[3]{0} multiply(f16[3]{0} %arg_1, f16[3]{0} %broadcast.13.clone), metadata={op_type="Mul" op_name="mul_1"}
  ROOT %subtract.15.clone = f16[3]{0} subtract(f16[3]{0} %multiply.11.clone, f16[3]{0} %multiply.14.clone), metadata={op_type="Sub" op_name="sub"}
}
)";
};

TEST_P(AXBYTest, OperationGetsFused) {
  FuseOpsLate fuse_ops_late(*annotations_);
  ASSERT_TRUE(fuse_ops_late.Run(module_.get()).ValueOrDie());

  auto root = module_->entry_computation()->root_instruction();

  ASSERT_EQ(root->fused_instructions_computation()->name(),
            "_pop_op_scaled_inplace_axby");
  ASSERT_EQ(root->operand_count(), 4)
      << "Expected fused call to have 4 operands, one for each of a, x, b, y.";
}

TEST_P(AXBYTest, FusedOperationUsesNativeTypes) {
  FuseOpsLate fuse_ops_late(*annotations_);
  ASSERT_TRUE(fuse_ops_late.Run(module_.get()).ValueOrDie());

  auto entry = module_->entry_computation();
  ASSERT_TRUE(EntryContainsNoCasts(entry));
  ASSERT_TRUE(FusedCallContainsNoCasts(entry));
}

class XBYTest : public ScaledArithmeticTest {
 public:
  static std::string TestName(
      const ::testing::TestParamInfo<AXBYTest::ParamType>& info) {
    if (info.param == XPlusbYWithBroadcastConvert) {
      return "XPlusbYWithBroadcastConvert";
    }

    if (info.param == XPlusbYWithBroadcastReshapeConvert) {
      return "XPlusbYWithBroadcastReshapeConvert";
    }

    if (info.param == XMinusbYWithBroadcastConvert) {
      return "XMinusbYWithBroadcastConvert";
    }

    if (info.param == XMinusbYWithBroadcastReshapeConvert) {
      return "XMinusbYWithBroadcastReshapeConvert";
    }

    return "Unknown";
  }
  static constexpr const char* XPlusbYWithBroadcastConvert = R"(
HloModule top

ENTRY test {
  %arg_0 = f16[3]{0} parameter(0)
  %arg_1 = f16[3]{0} parameter(1)
  %arg_2 = f32[] parameter(2)
  %convert.7.clone = f16[] convert(f32[] %arg_2), metadata={op_type="Cast" op_name="Cast"}
  %broadcast.8.clone = f16[3]{0} broadcast(f16[] %convert.7.clone), dimensions={}, metadata={op_type="Mul" op_name="mul"}
  %multiply.9.clone = f16[3]{0} multiply(f16[3]{0} %arg_1, f16[3]{0} %broadcast.8.clone), metadata={op_type="Mul" op_name="mul"}
  ROOT %add.10.clone = f16[3]{0} add(f16[3]{0} %arg_0, f16[3]{0} %multiply.9.clone), metadata={op_type="AddV2" op_name="add"}
}
)";

  static constexpr const char* XPlusbYWithBroadcastReshapeConvert = R"(
HloModule top

ENTRY test {
  %arg_0 = f16[3]{0} parameter(0)
  %arg_1 = f16[3]{0} parameter(1)
  %arg_2 = f32[1]{0} parameter(2)
  %convert.7.clone = f16[1]{0} convert(f32[1]{0} %arg_2), metadata={op_type="Cast" op_name="Cast"}
  %reshape.8.clone = f16[] reshape(f16[1]{0} %convert.7.clone), metadata={op_type="Mul" op_name="mul"}
  %broadcast.9.clone = f16[3]{0} broadcast(f16[] %reshape.8.clone), dimensions={}, metadata={op_type="Mul" op_name="mul"}
  %multiply.10.clone = f16[3]{0} multiply(f16[3]{0} %arg_1, f16[3]{0} %broadcast.9.clone), metadata={op_type="Mul" op_name="mul"}
  ROOT %add.11.clone = f16[3]{0} add(f16[3]{0} %arg_0, f16[3]{0} %multiply.10.clone), metadata={op_type="AddV2" op_name="add"}
}
)";

  static constexpr const char* XMinusbYWithBroadcastConvert = R"(
HloModule top

ENTRY test {
  %arg_0 = f16[3]{0} parameter(0)
  %arg_1 = f16[3]{0} parameter(1)
  %arg_2 = f32[] parameter(2)
  %convert.7.clone = f16[] convert(f32[] %arg_2), metadata={op_type="Cast" op_name="Cast"}
  %broadcast.8.clone = f16[3]{0} broadcast(f16[] %convert.7.clone), dimensions={}, metadata={op_type="Mul" op_name="mul"}
  %multiply.9.clone = f16[3]{0} multiply(f16[3]{0} %arg_1, f16[3]{0} %broadcast.8.clone), metadata={op_type="Mul" op_name="mul"}
  ROOT %sub.10.clone = f16[3]{0} subtract(f16[3]{0} %arg_0, f16[3]{0} %multiply.9.clone), metadata={op_type="Sub" op_name="sub"}
}
)";

  static constexpr const char* XMinusbYWithBroadcastReshapeConvert = R"(
HloModule top

ENTRY test {
  %arg_0 = f16[3]{0} parameter(0)
  %arg_1 = f16[3]{0} parameter(1)
  %arg_2 = f32[1]{0} parameter(2)
  %convert.7.clone = f16[1]{0} convert(f32[1]{0} %arg_2), metadata={op_type="Cast" op_name="Cast"}
  %reshape.8.clone = f16[] reshape(f16[1]{0} %convert.7.clone), metadata={op_type="Mul" op_name="mul"}
  %broadcast.9.clone = f16[3]{0} broadcast(f16[] %reshape.8.clone), dimensions={}, metadata={op_type="Mul" op_name="mul"}
  %multiply.10.clone = f16[3]{0} multiply(f16[3]{0} %arg_1, f16[3]{0} %broadcast.9.clone), metadata={op_type="Mul" op_name="mul"}
  ROOT %sub.11.clone = f16[3]{0} subtract(f16[3]{0} %arg_0, f16[3]{0} %multiply.10.clone), metadata={op_type="Sub" op_name="sub"}
}
)";
};

TEST_P(XBYTest, OperationGetsFused) {
  FuseOpsLate fuse_ops_late(*annotations_);
  ASSERT_TRUE(fuse_ops_late.Run(module_.get()).ValueOrDie());

  auto entry = module_->entry_computation();
  auto root = entry->root_instruction();
  ASSERT_EQ(root->fused_instructions_computation()->name(),
            "_pop_op_scaled_inplace_xby");
  ASSERT_EQ(root->operand_count(), 3)
      << "Expected fused call to have 3 operands, one for each of x, b, y.";
}

TEST_P(XBYTest, FusedOperationUsesNativeTypes) {
  FuseOpsLate fuse_ops_late(*annotations_);
  ASSERT_TRUE(fuse_ops_late.Run(module_.get()).ValueOrDie());

  auto entry = module_->entry_computation();
  ASSERT_TRUE(EntryContainsNoCasts(entry));
  ASSERT_TRUE(FusedCallContainsNoCasts(entry));
}

class AXYTest : public ScaledArithmeticTest {
 public:
  static std::string TestName(
      const ::testing::TestParamInfo<AXBYTest::ParamType>& info) {
    if (info.param == aXPlusYWithBroadcastConvert) {
      return "aXPlusYWithBroadcastConvert";
    }

    if (info.param == aXPlusYWithBroadcastReshapeConvert) {
      return "aXPlusYWithBroadcastReshapeConvert";
    }

    if (info.param == aXMinusYWithBroadcastConvert) {
      return "aXMinusYWithBroadcastConvert";
    }

    if (info.param == aXMinusYWithBroadcastReshapeConvert) {
      return "aXMinusYWithBroadcastReshapeConvert";
    }

    return "Unknown";
  }

  static constexpr const char* aXPlusYWithBroadcastConvert = R"(
HloModule top

ENTRY test {
  %arg_0 = f16[3]{0} parameter(0)
  %arg_2 = f32[] parameter(2)
  %convert.7.clone = f16[] convert(f32[] %arg_2), metadata={op_type="Cast" op_name="Cast"}
  %broadcast.8.clone = f16[3]{0} broadcast(f16[] %convert.7.clone), dimensions={}, metadata={op_type="Mul" op_name="mul"}
  %multiply.9.clone = f16[3]{0} multiply(f16[3]{0} %arg_0, f16[3]{0} %broadcast.8.clone), metadata={op_type="Mul" op_name="mul"}
  %arg_1 = f16[3]{0} parameter(1)
  ROOT %add.12.clone = f16[3]{0} add(f16[3]{0} %multiply.9.clone, f16[3]{0} %arg_1), metadata={op_type="AddV2" op_name="add"}
}
)";

  static constexpr const char* aXPlusYWithBroadcastReshapeConvert = R"(
HloModule top

ENTRY test {
%arg_0 = f16[3]{0} parameter(0)
%arg_2 = f32[1]{0} parameter(2)
%convert.7.clone = f16[1]{0} convert(f32[1]{0} %arg_2), metadata={op_type="Cast" op_name="Cast"}
%reshape.8.clone = f16[] reshape(f16[1]{0} %convert.7.clone), metadata={op_type="Mul" op_name="mul"}
%broadcast.9.clone = f16[3]{0} broadcast(f16[] %reshape.8.clone), dimensions={}, metadata={op_type="Mul" op_name="mul"}
%multiply.10.clone = f16[3]{0} multiply(f16[3]{0} %arg_0, f16[3]{0} %broadcast.9.clone), metadata={op_type="Mul" op_name="mul"}
%arg_1 = f16[3]{0} parameter(1)
ROOT %add.14.clone = f16[3]{0} add(f16[3]{0} %multiply.10.clone, f16[3]{0} %arg_1), metadata={op_type="AddV2" op_name="add"}
}
)";

  static constexpr const char* aXMinusYWithBroadcastConvert = R"(
HloModule top

ENTRY test {
  %arg_0 = f16[3]{0} parameter(0)
  %arg_2 = f32[] parameter(2)
  %convert.7.clone = f16[] convert(f32[] %arg_2), metadata={op_type="Cast" op_name="Cast"}
  %broadcast.8.clone = f16[3]{0} broadcast(f16[] %convert.7.clone), dimensions={}, metadata={op_type="Mul" op_name="mul"}
  %multiply.9.clone = f16[3]{0} multiply(f16[3]{0} %arg_0, f16[3]{0} %broadcast.8.clone), metadata={op_type="Mul" op_name="mul"}
  %arg_1 = f16[3]{0} parameter(1)
  ROOT %sub.12.clone = f16[3]{0} subtract(f16[3]{0} %multiply.9.clone, f16[3]{0} %arg_1), metadata={op_type="Sub" op_name="sub"}
}
)";

  static constexpr const char* aXMinusYWithBroadcastReshapeConvert = R"(
HloModule top

ENTRY test {
  %arg_0 = f16[3]{0} parameter(0)
  %arg_2 = f32[1]{0} parameter(2)
  %convert.7.clone = f16[1]{0} convert(f32[1]{0} %arg_2), metadata={op_type="Cast" op_name="Cast"}
  %reshape.8.clone = f16[] reshape(f16[1]{0} %convert.7.clone), metadata={op_type="Mul" op_name="mul"}
  %broadcast.9.clone = f16[3]{0} broadcast(f16[] %reshape.8.clone), dimensions={}, metadata={op_type="Mul" op_name="mul"}
  %multiply.10.clone = f16[3]{0} multiply(f16[3]{0} %arg_0, f16[3]{0} %broadcast.9.clone), metadata={op_type="Mul" op_name="mul"}
  %arg_1 = f16[3]{0} parameter(1)
  ROOT %sub.14.clone = f16[3]{0} subtract(f16[3]{0} %multiply.10.clone, f16[3]{0} %arg_1), metadata={op_type="Sub" op_name="sub"}
  }
  )";
};

TEST_P(AXYTest, OperationGetsFused) {
  FuseOpsLate fuse_ops_late(*annotations_);
  ASSERT_TRUE(fuse_ops_late.Run(module_.get()).ValueOrDie());

  auto entry = module_->entry_computation();
  auto root = entry->root_instruction();
  ASSERT_EQ(root->fused_instructions_computation()->name(),
            "_pop_op_scaled_inplace_axy");
  ASSERT_EQ(root->operand_count(), 3)
      << "Expected fused call to have 3 operands, one for each of a, x, y.";
}

TEST_P(AXYTest, FusedOperationUsesNativeTypes) {
  FuseOpsLate fuse_ops_late(*annotations_);
  ASSERT_TRUE(fuse_ops_late.Run(module_.get()).ValueOrDie());

  auto entry = module_->entry_computation();
  ASSERT_TRUE(EntryContainsNoCasts(entry));
  ASSERT_TRUE(FusedCallContainsNoCasts(entry));
}

INSTANTIATE_TEST_SUITE_P(
    ScaledArithmetic, AXBYTest,
    ::testing::Values(AXBYTest::axPlusByWithBroadcastConvert,
                      AXBYTest::axPlusByWithBroadcastReshapeConvert,
                      AXBYTest::axMinusByWithBroadcastConvert,
                      AXBYTest::axMinusByWithBroadcastReshapeConvert),
    std::bind(&AXBYTest::TestName, std::placeholders::_1));

INSTANTIATE_TEST_SUITE_P(
    ScaledArithmetic, XBYTest,
    ::testing::Values(XBYTest::XPlusbYWithBroadcastConvert,
                      XBYTest::XPlusbYWithBroadcastReshapeConvert,
                      XBYTest::XMinusbYWithBroadcastConvert,
                      XBYTest::XMinusbYWithBroadcastReshapeConvert),
    std::bind(&XBYTest::TestName, std::placeholders::_1));

INSTANTIATE_TEST_SUITE_P(
    ScaledArithmetic, AXYTest,
    ::testing::Values(AXYTest::aXPlusYWithBroadcastConvert,
                      AXYTest::aXPlusYWithBroadcastReshapeConvert,
                      AXYTest::aXMinusYWithBroadcastConvert,
                      AXYTest::aXMinusYWithBroadcastReshapeConvert),
    std::bind(&AXYTest::TestName, std::placeholders::_1));

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
