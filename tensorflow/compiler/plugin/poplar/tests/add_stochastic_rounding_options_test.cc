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
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
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

const char* simple_hlo = R"(
HloModule test

ENTRY test {
  identical0 = f16[] parameter(0), backend_config="{\"is_replica_identical\":\"1\"}"
  identical1 = f16[] constant(3), backend_config="{\"is_replica_identical\":\"1\"}"
  identical2 = f16[] add(identical0, identical1), backend_config="{\"is_replica_identical\":\"1\"}"

  constant = f16[] constant(2), backend_config="{\"is_replica_identical\":\"1\"}"
  differing0 = f16[] rng(constant, constant), distribution=rng_uniform, backend_config="{\"is_replica_identical\":\"0\"}"
  differing1 = f16[] rng(constant, constant), distribution=rng_uniform, backend_config="{\"is_replica_identical\":\"0\"}"

  ROOT differing_root = (f16[], f16[]) tuple(identical2, differing0), backend_config="{\"is_replica_identical\":\"0\"}"
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
  using ::testing::AnyOf;
  using ::testing::Each;

  ASSERT_TRUE(SetUpHloModule(simple_hlo));

  const bool enable_experimental_prng_stability = true;
  AddStochasticRoundingOptions add_stochastic_rounding_options(
      StochasticRounding_ReplicaIdenticalOnly,
      enable_experimental_prng_stability);

  TF_ASSERT_OK_AND_ASSIGN(bool modified,
                          add_stochastic_rounding_options.Run(hlo_module_));
  ASSERT_TRUE(modified);

  ASSERT_THAT(
      ReplicaDifferingModuleInstructions(),
      Each(AnyOf(HasStochasticRoundingMethod(StochasticRoundingMethod_None),
                 HasStochasticRoundingMethod(StochasticRoundingMethod_Any))));

  ASSERT_THAT(
      ReplicaIdenticalModuleInstructions(),
      Each(AnyOf(
          HasStochasticRoundingMethod(StochasticRoundingMethod_IdenticalSeeds),
          HasStochasticRoundingMethod(StochasticRoundingMethod_Any))));
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

const char* poplar_fusion_hlo = R"(
HloModule test

_pop_op_arithmetic_expression {
  param0 = s32[4] parameter(0), backend_config="{\"stochastic_rounding\":\"THREESTATE_UNDEFINED\"}"
  add0 = s32[4] add(param0, param0), backend_config="{\"stochastic_rounding\":\"THREESTATE_UNDEFINED\"}"
  param1 = s32[4] parameter(1), backend_config="{\"stochastic_rounding\":\"THREESTATE_UNDEFINED\"}"
  ROOT add1 = s32[4] add(add0, param1), backend_config="{\"stochastic_rounding\":\"THREESTATE_UNDEFINED\"}"
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
}

const char* f16_to_f32_hlo = R"(
HloModule test

add_reduce {
  y = f32[] parameter(1)
  x = f32[] parameter(0)
  ROOT add = f32[] add(x, y)
}

_pop_op_reduction_fp16_input {
  param0 = f16[32,1]{1,0} parameter(0)
  convert = f32[32,1]{1,0} convert(param0)
  param1 = f32[] parameter(1)
  ROOT reduce = f32[] reduce(convert, param1), dimensions={0,1}, to_apply=add_reduce
}

ENTRY test {
 param0 = f16[32, 1] parameter(0)
 f32convert = f32[32, 1] convert(param0)

 constant = f32[] constant(1)
 f32reduce = f32[] fusion(param0, constant), kind=kCustom, calls=_pop_op_reduction_fp16_input

 ROOT result = (f32[32, 1], f32[]) tuple(f32convert, f32reduce)
}
)";

TEST_F(AddStochasticRoundingOptionsTest, StochasticRoundingMethodF16ToF32) {
  using ::testing::Not;

  ASSERT_TRUE(SetUpHloModule(f16_to_f32_hlo));

  AddStochasticRoundingOptions add_stochastic_rounding_options(
      StochasticRounding_On);

  TF_ASSERT_OK_AND_ASSIGN(bool modified,
                          add_stochastic_rounding_options.Run(hlo_module_));
  ASSERT_TRUE(modified);

  auto* f32convert = FindInstruction(hlo_module_, "f32convert");
  auto* f32reduce = FindInstruction(hlo_module_, "f32reduce");
  ASSERT_TRUE(f32convert);
  ASSERT_TRUE(f32reduce);

  // Both the convert and reduce use SR and so should use a specific seed.
  ASSERT_THAT(f32convert,
              Not(HasStochasticRoundingMethod(StochasticRoundingMethod_Any)));
  ASSERT_THAT(f32reduce,
              Not(HasStochasticRoundingMethod(StochasticRoundingMethod_Any)));
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
// Similarly none of these poplar instructions do any compute so it shouldn't
// matter what seed they use.
static const HloTestCase simple_no_compute_poplar = {"simple_no_compute_poplar",
                                                     R"(
HloModule test

ENTRY test {
 stateful-noop = () custom-call(), custom_call_target="StatefulNoop", custom_call_has_side_effect=true
 param = f16[] parameter(0)
 constant = f16[] constant(2)
 fifo = f16[] custom-call(param), custom_call_target="Fifo", backend_config="{\"offload\":0, \"depth\":2}"
 tuple = (f16[], f16[]) tuple(fifo, constant)
 value = f16[] get-tuple-element(tuple), index=1
 ROOT copy = f16[] custom-call(value), custom_call_target="InterIpuCopy"
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

  CustomOpReplacer custom_op_replacer;
  custom_op_replacer.Run(hlo_module_);

  AddStochasticRoundingOptions add_stochastic_rounding_options(
      StochasticRounding_On);

  TF_ASSERT_OK_AND_ASSIGN(bool modified,
                          add_stochastic_rounding_options.Run(hlo_module_));
  ASSERT_TRUE(modified);

  ASSERT_THAT(AllModuleInstructions(),
              Each(HasStochasticRoundingMethod(StochasticRoundingMethod_Any)));
}

const char* all_reduce_hlo = R"(
HloModule test
add {
  x = f16[] parameter(0)
  y = f16[] parameter(1)
  add = f16[] add(x, y)
}

ENTRY test {
  after-all = token[] after-all()
  infeed = (f16[4], token[]) infeed(token[] after-all)
  value = f16[4] get-tuple-element((f16[4], token[]) infeed), index=0
  ROOT all_reduce = f16[4] all-reduce(value), to_apply=add, replica_groups={}, backend_config="{\"is_replica_identical\":\"1\"}"
}
)";
TEST_F(AddStochasticRoundingOptionsTest, SettingStochasticRoundingAllReduce) {
  ASSERT_TRUE(SetUpHloModule(all_reduce_hlo));

  auto* all_reduce = FindInstruction(hlo_module_, "all_reduce");
  ASSERT_TRUE(all_reduce);

  // Test that all-reduce instructions do not use stochastic rounding.
  AddStochasticRoundingOptions
      add_stochastic_rounding_options_replica_identical(StochasticRounding_On);
  TF_ASSERT_OK_AND_ASSIGN(
      bool modified,
      add_stochastic_rounding_options_replica_identical.Run(hlo_module_));
  ASSERT_TRUE(modified);
  ASSERT_THAT(all_reduce,
              HasStochasticRoundingMethod(StochasticRoundingMethod_None));
}

INSTANTIATE_TEST_SUITE_P(AddStochasticRoundingOptionsHLO, AnySeedTest,
                         ::testing::Values(simple_no_compute,
                                           simple_no_compute_poplar,
                                           simple_compute_non_f16),
                         HloTestCaseName);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
