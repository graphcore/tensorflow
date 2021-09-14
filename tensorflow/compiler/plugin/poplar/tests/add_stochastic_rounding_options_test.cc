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
    auto stochastic_rounding = backend_cfg.stochastic_rounding();

    *result_listener << "instruction '" << instr->name()
                     << "' has stochastic rounding set to "
                     << ThreeState_Name(stochastic_rounding) << " expected "
                     << ThreeState_Name(state);

    return stochastic_rounding == state;
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
    auto deterministic_workers = backend_cfg.deterministic_workers();

    *result_listener << "instruction '" << instr->name()
                     << "' has deterministic workers set to "
                     << ThreeState_Name(deterministic_workers) << " expected "
                     << ThreeState_Name(state);

    return deterministic_workers == state;
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
  stochastic_rounding_on = f32[] parameter(0)
  ROOT stochastic_rounding_default = f32[] parameter(1)
}
)";
TEST_F(AddStochasticRoundingOptionsTest, SettingStochasticRoundingDefault) {
  ASSERT_TRUE(SetUpHloModule(stochastic_rounding_attrs));

  auto* stochastic_rounding_on =
      FindInstruction(hlo_module_, "stochastic_rounding_on");

  // Enable stochastic rounding for the stochastic_rounding_on instruction.
  // We cant set it in the Hlo due to a bug in TF1.15.
  FrontendAttributes frontend_attributes;
  (*frontend_attributes.mutable_map())["STOCHASTIC_ROUNDING"] = "THREESTATE_ON";
  stochastic_rounding_on->set_frontend_attributes(frontend_attributes);

  AddStochasticRoundingOptions add_stochastic_rounding_options(
      StochasticRounding_Off);

  TF_ASSERT_OK_AND_ASSIGN(bool modified,
                          add_stochastic_rounding_options.Run(hlo_module_));
  ASSERT_TRUE(modified);

  // Check that the default is only applied to instructions w/o a stochastic
  // rounding frontend attribute.
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

const char* poplar_fusion_hlo = R"(
HloModule test

_pop_op_arithmetic_expression {
  param0 = s32[4] parameter(0), backend_config="{\"stochastic_rounding\":\"THREESTATE_UNDEFINED\", \"deterministic_workers\":\"THREESTATE_ON\"}"
  add0 = s32[4] add(param0, param0), backend_config="{\"stochastic_rounding\":\"THREESTATE_UNDEFINED\", \"deterministic_workers\":\"THREESTATE_ON\"}"
  param1 = s32[4] parameter(1), backend_config="{\"stochastic_rounding\":\"THREESTATE_UNDEFINED\", \"deterministic_workers\":\"THREESTATE_ON\"}"
  ROOT add1 = s32[4] add(add0, param1), backend_config="{\"stochastic_rounding\":\"THREESTATE_UNDEFINED\", \"deterministic_workers\":\"THREESTATE_ON\"}"
}

ENTRY test {
 param0 = s32[4] parameter(0)
 param1 = s32[4] parameter(1)
 ROOT fusion = s32[4]{0} fusion(param0, param1), kind=kCustom, calls=_pop_op_arithmetic_expression
}
)";
TEST_F(AddStochasticRoundingOptionsTest, SettingOptionsForPoplarFusions) {
  using ::testing::Each;

  ASSERT_TRUE(SetUpHloModule(poplar_fusion_hlo));

  AddStochasticRoundingOptions add_stochastic_rounding_options(
      StochasticRounding_On);

  TF_ASSERT_OK_AND_ASSIGN(bool modified,
                          add_stochastic_rounding_options.Run(hlo_module_));
  ASSERT_TRUE(modified);

  // Make sure that we don't skip instructions in poplar fusions. If we did then
  // neither option would be applied and we'd use the existing value
  // (which defaults to THREESTATE_OFF).
  ASSERT_THAT(AllModuleInstructions(),
              Each(HasStochasticRounding(THREESTATE_ON)));
  // Undefined since none of the instructions are replica identical.
  ASSERT_THAT(AllModuleInstructions(),
              Each(HasDeterministicWorker(THREESTATE_UNDEFINED)));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
