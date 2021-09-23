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

bool GetBackendConfig(PoplarBackendConfig& backend_cfg,
                      ::testing::MatchResultListener* listener,
                      HloInstruction* inst) {
  auto statusor_backend = inst->template backend_config<PoplarBackendConfig>();
  if (!statusor_backend.ok()) {
    *listener << "Failed to get PoplarBackendConfig for instruction '"
              << inst->name() << "'";
    return false;
  }
  backend_cfg = statusor_backend.ValueOrDie();
  return true;
}
MATCHER_P(HasStochasticRounding, state, "") {
  auto* inst = arg;

  PoplarBackendConfig backend_cfg;
  if (GetBackendConfig(backend_cfg, result_listener, inst)) {
    auto stochastic_rounding = backend_cfg.stochastic_rounding();

    *result_listener << "instruction '" << inst->name()
                     << "' has stochastic rounding set to "
                     << ThreeState_Name(stochastic_rounding) << " expected "
                     << ThreeState_Name(state);

    return stochastic_rounding == state;
  }

  return false;
}
MATCHER_P(HasStochasticRoundingMethod, state, "") {
  auto* inst = arg;

  PoplarBackendConfig backend_cfg;
  if (GetBackendConfig(backend_cfg, result_listener, inst)) {
    auto stochastic_rounding_method = backend_cfg.stochastic_rounding_method();

    *result_listener << "instruction '" << inst->name()
                     << "' has stochastic rounding method set to "
                     << StochasticRoundingMethod_Name(
                            stochastic_rounding_method)
                     << " expected " << StochasticRoundingMethod_Name(state);

    return stochastic_rounding_method == state;
  }

  return false;
}
MATCHER_P(HasDeterministicWorker, state, "") {
  auto* inst = arg;

  PoplarBackendConfig backend_cfg;
  if (GetBackendConfig(backend_cfg, result_listener, inst)) {
    auto deterministic_workers = backend_cfg.deterministic_workers();

    *result_listener << "instruction '" << inst->name()
                     << "' has deterministic workers set to "
                     << ThreeState_Name(deterministic_workers) << " expected "
                     << ThreeState_Name(state);

    return deterministic_workers == state;
  }

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

  ASSERT_TRUE(SetUpHloModule(simple_hlo));

  const bool enable_experimental_prng_stability = true;
  AddStochasticRoundingOptions add_stochastic_rounding_options(
      StochasticRounding_ReplicaIdenticalOnly,
      enable_experimental_prng_stability);

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

  ASSERT_TRUE(SetUpHloModule(simple_hlo));

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

  ASSERT_TRUE(SetUpHloModule(simple_hlo));

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

  ASSERT_TRUE(SetUpHloModule(simple_hlo));

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

using AnySeedTest =
    ParameterizedHloTestFixture<AddStochasticRoundingOptionsTest>;
// It doesn't matter what seed this uses since it doesn't do any compute.
static const HloTestCase simple_no_compute = {"simple_no_compute", R"(
HloModule test

func {
 param = f16[] parameter(0)
 ROOT root = f16[1,2] broadcast(param), dimensions={}
}

ENTRY test {
 param = f16[] parameter(0)
 constant = f16[] constant(2)
 tuple = (f16[], f16[]) tuple(param, constant)
 value = f16[] get-tuple-element(tuple), index=1
 ROOT root = f16[1,2] call(value), to_apply=func
}
)"};
// It doesn't matter what seed this uses since it's not using f16 types.
static const HloTestCase simple_compute_non_f16 = {"simple_compute_non_f16", R"(
HloModule test

ENTRY test {
  constant = f32[] constant(2)
  rand = f32[4] rng(constant, constant), distribution=rng_uniform
  after-all = token[] after-all()
  infeed = (f32[4], token[]) infeed(token[] after-all)
  value = f32[4] get-tuple-element((f32[4], token[]) infeed), index=0
  ROOT result = f32[4] add(rand, value)
}
)"};
TEST_P(AnySeedTest, StochasticRoundingMethod) {
  using ::testing::Each;

  AddStochasticRoundingOptions add_stochastic_rounding_options(
      StochasticRounding_On);

  TF_ASSERT_OK_AND_ASSIGN(bool modified,
                          add_stochastic_rounding_options.Run(hlo_module_));
  ASSERT_TRUE(modified);

  ASSERT_THAT(AllModuleInstructions(),
              Each(HasStochasticRoundingMethod(StochasticRoundingMethod_Any)));
}

INSTANTIATE_TEST_SUITE_P(AddStochasticRoundingOptionsHLO, AnySeedTest,
                         ::testing::Values(simple_no_compute,
                                           simple_compute_non_f16),
                         HloTestCaseName);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
