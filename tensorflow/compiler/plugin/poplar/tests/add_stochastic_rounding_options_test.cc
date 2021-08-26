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

#include <gtest/gtest.h>

#include "tensorflow/compiler/plugin/poplar/driver/passes/add_stochastic_rounding_options.h"

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"
#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
namespace {

MATCHER_P(HasStochasticRounding, state, "") {
  auto* instr = arg;
  auto statusor_backend = instr->template backend_config<PoplarBackendConfig>();
  if (statusor_backend.ok()) {
    auto backend_cfg = statusor_backend.ValueOrDie();
    return backend_cfg.stochastic_rounding() == state;
  }

  *result_listener << "Failed to get PoplarBackendConfig for instruction '"
                   << instr->name() << "'";
  return false;
}
MATCHER_P(HasDeterministicWorker, state, "") {
  auto* instr = arg;
  auto statusor_backend = instr->template backend_config<PoplarBackendConfig>();
  if (statusor_backend.ok()) {
    auto backend_cfg = statusor_backend.ValueOrDie();
    return backend_cfg.deterministic_workers() == state;
  }

  *result_listener << "Failed to get PoplarBackendConfig for instruction '"
                   << instr->name() << "'";
  return false;
}

const char* simple_hlo = R"(
HloModule test

ENTRY test {
  identical0 = f32[] parameter(0), backend_config="{\"is_replica_identical\":\"1\"}"
  identical1 = f32[] constant(3), backend_config="{\"is_replica_identical\":\"1\"}"

  constant = f32[] constant(2), backend_config="{\"is_replica_identical\":\"1\"}"
  differing0 = f32[] rng(constant, constant), distribution=rng_uniform, backend_config="{\"is_replica_identical\":\"0\"}"
  differing1 = f32[] rng(constant, constant), distribution=rng_uniform, backend_config="{\"is_replica_identical\":\"0\"}"

  ROOT differing_root = (f32[], f32[]) tuple(identical0, differing0), backend_config="{\"is_replica_identical\":\"0\"}"
}
)";
struct AddStochasticRoundingOptionsTest : HloTestFixture {
  void SetUp() override { ASSERT_TRUE(SetUpHloModule(simple_hlo)); }

  absl::flat_hash_set<HloInstruction*> ReplicaIdenticalModuleInstructions() {
    absl::flat_hash_set<HloInstruction*> instructions;

    for (auto* instruction : AllModuleInstructions()) {
      if (IsInstructionReplicaIdentical(instruction)) {
        instructions.insert(instruction);
      }
    }

    return instructions;
  }

  absl::flat_hash_set<HloInstruction*> ReplicaDifferingModuleInstructions() {
    absl::flat_hash_set<HloInstruction*> instructions;

    for (auto* instruction : AllModuleInstructions()) {
      if (!IsInstructionReplicaIdentical(instruction)) {
        instructions.insert(instruction);
      }
    }

    return instructions;
  }

  absl::flat_hash_set<HloInstruction*> AllModuleInstructions() {
    absl::flat_hash_set<HloInstruction*> instructions;

    for (auto* comp : hlo_module_->computations()) {
      for (auto* instr : comp->instructions()) {
        instructions.insert(instr);
      }
    }
    return instructions;
  }

  StochasticRoundingBehaviour default_stochastic_rounding_ =
      StochasticRounding_Off;
};

const char* stochastic_rounding_attrs = R"(
HloModule test
ENTRY test {
  stochastic_rounding_on = f32[] parameter(0), frontend_attributes={STOCHASTIC_ROUNDING="THREESTATE_ON"}
  ROOT stochastic_rounding_default = f32[] parameter(1)
}
)";
TEST_F(AddStochasticRoundingOptionsTest, SettingStochasticRoundingDefault) {
  ASSERT_TRUE(SetUpHloModule(stochastic_rounding_attrs));

  AddStochasticRoundingOptions add_stochastic_rounding_options(
      StochasticRounding_Off);

  TF_ASSERT_OK_AND_ASSIGN(bool modified,
                          add_stochastic_rounding_options.Run(hlo_module_));
  ASSERT_TRUE(modified);

  // Check that the default is only applied to instructions w/o a stochastic
  // rounding frontend attribute.
  auto* stochastic_rounding_on =
      FindInstruction(hlo_module_, "stochastic_rounding_on");
  ASSERT_THAT(stochastic_rounding_on, HasStochasticRounding(THREESTATE_ON));

  auto* stochastic_rounding_default =
      FindInstruction(hlo_module_, "stochastic_rounding_default");
  ASSERT_THAT(stochastic_rounding_default,
              HasStochasticRounding(THREESTATE_OFF));
}

TEST_F(AddStochasticRoundingOptionsTest,
       SettingStochasticRoundingForReplicaIdentical) {
  using ::testing::Each;

  AddStochasticRoundingOptions add_stochastic_rounding_options(
      StochasticRounding_ReplicaIdenticalOnly);

  TF_ASSERT_OK_AND_ASSIGN(bool modified,
                          add_stochastic_rounding_options.Run(hlo_module_));
  ASSERT_TRUE(modified);

  ASSERT_THAT(ReplicaIdenticalModuleInstructions(),
              Each(HasStochasticRounding(THREESTATE_ON)));
  ASSERT_THAT(ReplicaDifferingModuleInstructions(),
              Each(HasStochasticRounding(THREESTATE_OFF)));
}

TEST_F(AddStochasticRoundingOptionsTest, SettingStochasticRoundingOn) {
  using ::testing::Each;

  AddStochasticRoundingOptions add_stochastic_rounding_options(
      StochasticRounding_On);

  TF_ASSERT_OK_AND_ASSIGN(bool modified,
                          add_stochastic_rounding_options.Run(hlo_module_));
  ASSERT_TRUE(modified);

  ASSERT_THAT(AllModuleInstructions(),
              Each(HasStochasticRounding(THREESTATE_ON)));
}

TEST_F(AddStochasticRoundingOptionsTest, SettingStochasticRoundingOff) {
  using ::testing::Each;

  AddStochasticRoundingOptions add_stochastic_rounding_options(
      StochasticRounding_Off);

  TF_ASSERT_OK_AND_ASSIGN(bool modified,
                          add_stochastic_rounding_options.Run(hlo_module_));
  ASSERT_TRUE(modified);

  ASSERT_THAT(AllModuleInstructions(),
              Each(HasStochasticRounding(THREESTATE_OFF)));
}

TEST_F(AddStochasticRoundingOptionsTest, SettingDeterministicWorkers) {
  using ::testing::Each;

  AddStochasticRoundingOptions add_stochastic_rounding_options(
      default_stochastic_rounding_);

  TF_ASSERT_OK_AND_ASSIGN(bool modified,
                          add_stochastic_rounding_options.Run(hlo_module_));
  ASSERT_TRUE(modified);

  // Deterministic workers should only be enabled for replica
  // identical instructions, and left as undefined otherwise.
  ASSERT_THAT(ReplicaIdenticalModuleInstructions(),
              Each(HasDeterministicWorker(THREESTATE_ON)));
  ASSERT_THAT(ReplicaDifferingModuleInstructions(),
              Each(HasDeterministicWorker(THREESTATE_UNDEFINED)));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
