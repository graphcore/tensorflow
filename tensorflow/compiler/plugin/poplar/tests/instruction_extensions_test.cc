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
#include <functional>
#include <map>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_instruction_extensions.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_noop.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using ExtensionRegistryTest = HloTestBase;

TEST_F(ExtensionRegistryTest, UsesDefaultWhenNothingIsRegistered) {
  struct TestExtension1 {
    std::function<std::string()> impl = []() { return "default1"; };
  };

  struct TestExtension2 {
    std::function<int(bool)> impl = [](bool) { return 5; };
  };

  ExtensionRegistry<TestExtension1> registry1;
  ASSERT_EQ(registry1[HloOpcode::kAdd](), "default1");

  ExtensionRegistry<TestExtension2> registry2;
  ASSERT_EQ(registry2[HloOpcode::kAdd](true), 5);
}

TEST_F(ExtensionRegistryTest, CanSpecifyFunctionPerOpcode) {
  struct TestExtension {
    std::function<std::string()> impl = []() { return "default1"; };
  };

  ExtensionRegistry<TestExtension> registry;

  registry[HloOpcode::kRng] = []() { return "value1"; };

  ASSERT_EQ(registry[HloOpcode::kRng](), "value1");
  ASSERT_EQ(registry[HloOpcode::kAdd](), "default1");
}

TEST_F(ExtensionRegistryTest, UnimplementedDefaultThrows) {
  struct ThrowingExtension {
    std::function<int()> impl;
  };

  ExtensionRegistry<ThrowingExtension> registry;
  ASSERT_THROW(registry[HloOpcode::kAdd](), std::bad_function_call);
}

struct InstructionExtensionTest : HloTestBase {
  std::unique_ptr<HloInstruction> hlo_instruction_ =
      HloInstruction::CreateConstant({});
  HloOpcode op_code_ = hlo_instruction_->opcode();
};

TEST_F(InstructionExtensionTest, CanRegisterAllocatingIndicesExtension) {
  const auto expected = absl::flat_hash_set<int64>{1, 2};

  HloInstructionExtensions extensions;
  extensions.Register<AllocatingIndicesExtension>(
      op_code_, [expected](HloInstruction*) { return expected; });
  const auto result =
      extensions.Call<AllocatingIndicesExtension>(hlo_instruction_.get());

  ASSERT_EQ(result, expected);
}

TEST_F(InstructionExtensionTest, CanRegisterAllocatingOutputExtension) {
  const auto expected = true;

  HloInstructionExtensions extensions;
  extensions.Register<AllocatingOutputExtension>(
      op_code_, [expected](HloInstruction*) { return expected; });
  const auto result =
      extensions.Call<AllocatingOutputExtension>(hlo_instruction_.get());

  ASSERT_EQ(result, expected);
}

TEST_F(InstructionExtensionTest, CanRegisterLayoutDependenciesExtension) {
  const auto expected = absl::flat_hash_map<int64, int64>{{1, 1}, {3, 3}};

  HloInstructionExtensions extensions;
  extensions.Register<LayoutDependenciesExtension>(
      op_code_, [expected](HloInstruction*) { return expected; });
  const auto result =
      extensions.Call<LayoutDependenciesExtension>(hlo_instruction_.get());

  ASSERT_EQ(result, expected);
}

struct PoplarInstructionExtensionTest : HloTestBase {
  // Using HloStatefulNoop as it supports default construction.
  using BaseInstruction = HloStatefulNoop;

  HloInstructionExtensions extensions_ = GetHloInstructionExtensions();
};

TEST_F(PoplarInstructionExtensionTest, AllocatingIndicesExtCallsPoplarImpl) {
  class TestPoplarInstruction : public BaseInstruction {
   public:
    absl::flat_hash_set<int64> AllocatingIndices() const override {
      return {1, 2, 3};
    }
  };

  TestPoplarInstruction poplar_instruction;
  const auto result =
      extensions_.Call<AllocatingIndicesExtension>(&poplar_instruction);
  ASSERT_EQ(result, poplar_instruction.AllocatingIndices());
}

TEST_F(PoplarInstructionExtensionTest, AllocatingOutputExtCallsPoplarImpl) {
  class TestPoplarInstruction : public BaseInstruction {
   public:
    bool AllocatingOutput() const override { return true; }
  };

  TestPoplarInstruction poplar_instruction;
  const auto result =
      extensions_.Call<AllocatingOutputExtension>(&poplar_instruction);
  ASSERT_EQ(result, poplar_instruction.AllocatingOutput());
}

TEST_F(PoplarInstructionExtensionTest, LayoutDependenciesExtCallPoplarImpl) {
  class TestPoplarInstruction : public BaseInstruction {
   public:
    absl::flat_hash_map<int64, int64> LayoutDependencies() const override {
      return {{1, 1}, {3, 3}};
    }
  };

  TestPoplarInstruction poplar_instruction;
  const auto result =
      extensions_.Call<LayoutDependenciesExtension>(&poplar_instruction);
  ASSERT_EQ(result, poplar_instruction.LayoutDependencies());
}

struct HloInstructionRegistrationTest : HloTestBase {
  class TestOp : public PoplarOpDef {
    StatusOr<poplar::program::Program> Creator(
        poplar::Graph&, CompilerResources&, const HloInstruction*,
        const xla::Shape&, TensorMap&, const poplar::DebugContext&) override {
      return FailedPrecondition("Shouldn't be called for these tests.");
    }
  };

  static void RegisterExtensions(HloOpcode op_code) {
    RegisterHloInstructionExtension<AllocatingOutputExtension>(
        op_code, TestAllocatingOutput);
  }

  static bool TestAllocatingOutput(HloInstruction*) { return true; }

  std::unique_ptr<HloInstruction> hlo_instruction_ =
      HloInstruction::CreateConstant({});
  static const HloOpcode op_code_ = HloOpcode::kConstant;
};

TEST_F(HloInstructionRegistrationTest, ExtensionsRegisteredWithOp) {
  // Note this registration is global and persists in other tests.
  HloOpRegistrar op_registrar(
      op_code_, std::unique_ptr<PoplarOpDef>(new TestOp), RegisterExtensions);

  const auto result = CallHloInstructionExtension<AllocatingOutputExtension>(
      hlo_instruction_.get());
  ASSERT_EQ(result, TestAllocatingOutput(hlo_instruction_.get()));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
