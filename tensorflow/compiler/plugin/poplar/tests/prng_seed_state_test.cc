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

#include "tensorflow/compiler/plugin/poplar/driver/prng_seed_state.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_base.h"

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/input_output_aliasing_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/entry_visitor.h"
#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"

#include <poplar/CSRFunctions.hpp>
#include <poplar/RandomSeed.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>

namespace xla {
namespace poplarplugin {
namespace {

struct PrngSeedTest : HloPoplarTestBase {
  bool SetUpOrSkip() {
    // GTEST_SKIP can only be called from a void function, hence
    // this overload.
    bool skip = false;
    SetUpOrSkip(skip);
    return !skip;
  }

  void SetUpOrSkip(bool& skip) {
    TF_ASSERT_OK_AND_ASSIGN(auto ipu_count, GetMaxIpuCount());
    const bool enough_hw = ipu_count >= replication_factor_;
    if (!enough_hw) {
      skip = true;
      GTEST_SKIP() << "Skipping PrngSeedStateTests. They need to run with "
                   << replication_factor_ << " IPUs but only " << ipu_count
                   << "available.";
    } else {
      TF_ASSERT_OK_AND_ASSIGN(device_,
                              CreateIpuDevice(device_count_, tile_count_));

      graph_ = absl::make_unique<poplar::Graph>(
          device_, poplar::replication_factor(replication_factor_));
      poplin::addCodelets(*graph_);
      popnn::addCodelets(*graph_);
      popops::addCodelets(*graph_);
      poprand::addCodelets(*graph_);

      differing_seed_ = graph_->addVariable(
          poplar::UNSIGNED_INT, {2}, poplar::VariableMappingMethod::LINEAR);
      graph_->createHostWrite("differing_seed", differing_seed_);

      identical_seed_ = graph_->addVariable(
          poplar::UNSIGNED_INT, {2}, poplar::VariableMappingMethod::LINEAR);
      graph_->setInitialValue<unsigned>(identical_seed_, {10, 11});
    }
    skip = false;
  }

  int64 replication_factor_ = 2;
  int32 tile_count_ = 4;
  int32 device_count_ = 2;
  poplar::Device device_;

  std::unique_ptr<poplar::Graph> graph_;

  poplar::Tensor differing_seed_;
  poplar::Tensor identical_seed_;
};

struct PrngSeedStateTest : PrngSeedTest {
  void SetUp() override {
    if (SetUpOrSkip()) {
      input_ = graph_->addVariable(poplar::FLOAT, {5},
                                   poplar::VariableMappingMethod::LINEAR);
      graph_->setInitialValue<float>(input_, {0.3f, 0.4f, 0.7f, 0.8f, 0.11f});

      output_ = graph_->addVariable(poplar::FLOAT, {5},
                                    poplar::VariableMappingMethod::LINEAR);
      graph_->createHostRead("outstream", output_);

      poplar::setStochasticRounding(*graph_, seq_, true);
    }
  }

  std::pair<std::vector<float>, std::vector<float>> RunFloatToHalfCast() {
    return RunFloatToHalfCast(seq_);
  }

  std::pair<std::vector<float>, std::vector<float>> RunFloatToHalfCast(
      poplar::program::Sequence& seq) {
    poplar::Tensor half = popops::cast(*graph_, input_, poplar::HALF, seq);
    poplar::Tensor full = popops::cast(*graph_, half, poplar::FLOAT, seq);
    seq.add(poplar::program::Copy(full, output_));

    poplar::Engine engine(*graph_, seq);
    engine.load(device_);

    const uint32_t differing_seed_vals[] = {1, 2, 3, 4};
    engine.writeTensor("differing_seed", &differing_seed_vals[0],
                       &differing_seed_vals[4]);

    engine.run(0);

    std::vector<float> out(10);
    engine.readTensor("outstream", &out[0], &out[out.size()]);

    std::vector<float> replica1(out.begin(), out.begin() + 5);
    std::vector<float> replica2(out.begin() + 5, out.end());
    return {replica1, replica2};
  }

  poplar::program::Sequence seq_;

  poplar::Tensor input_;
  poplar::Tensor output_;
};

TEST_F(PrngSeedStateTest, SetupIdenticalSeed) {
  auto prng_state = PrngSeedState::SetupSeeds(*graph_, identical_seed_,
                                              differing_seed_, seq_);
  prng_state.ChangeStochasticRoundingMethod(
      StochasticRoundingMethod_IdenticalSeeds, seq_);

  auto replicas = RunFloatToHalfCast();

  // Since we're using a replica identical seed we expect
  // the F32->F16 casts to produce the same results
  // on both replicas.
  auto& replica1 = replicas.first;
  auto& replica2 = replicas.second;
  ASSERT_EQ(replica1, replica2);
}

TEST_F(PrngSeedStateTest, SwitchingIdenticalSeed) {
  auto prng_state = PrngSeedState::SetupSeeds(*graph_, identical_seed_,
                                              differing_seed_, seq_);
  prng_state.ChangeStochasticRoundingMethod(
      StochasticRoundingMethod_IdenticalSeeds, seq_);

  auto replicas = RunFloatToHalfCast();

  // We want to check that after switching seeds we still get the behaviour of
  // using a identical seed but where the seed values have changed (since the
  // first cast modifies them).
  prng_state.ChangeStochasticRoundingMethod(
      StochasticRoundingMethod_DifferingSeeds, seq_);
  prng_state.ChangeStochasticRoundingMethod(
      StochasticRoundingMethod_IdenticalSeeds, seq_);

  auto next_replicas = RunFloatToHalfCast();
  ASSERT_EQ(next_replicas.first, next_replicas.second);

  ASSERT_NE(next_replicas.first, replicas.first);
  ASSERT_NE(next_replicas.second, replicas.second);
}

TEST_F(PrngSeedStateTest, SetupDifferingSeed) {
  auto prng_state = PrngSeedState::SetupSeeds(*graph_, identical_seed_,
                                              differing_seed_, seq_);
  prng_state.ChangeStochasticRoundingMethod(
      StochasticRoundingMethod_DifferingSeeds, seq_);

  auto replicas = RunFloatToHalfCast();

  // Since we're using a replica differing seed we expect
  // the F32->F16 casts to produce different results.
  auto& replica1 = replicas.first;
  auto& replica2 = replicas.second;
  ASSERT_NE(replica1, replica2);
}

TEST_F(PrngSeedStateTest, SwichingDifferingSeed) {
  auto prng_state = PrngSeedState::SetupSeeds(*graph_, identical_seed_,
                                              differing_seed_, seq_);
  prng_state.ChangeStochasticRoundingMethod(
      StochasticRoundingMethod_DifferingSeeds, seq_);

  auto replicas = RunFloatToHalfCast();
  ASSERT_NE(replicas.first, replicas.second);

  // We want to check that after switching seeds we still get the behaviour of
  // using a differing seed but where the seed values have changed (since the
  // first cast modifies them).
  prng_state.ChangeStochasticRoundingMethod(
      StochasticRoundingMethod_IdenticalSeeds, seq_);
  prng_state.ChangeStochasticRoundingMethod(
      StochasticRoundingMethod_DifferingSeeds, seq_);

  auto next_replicas = RunFloatToHalfCast();
  ASSERT_NE(next_replicas.first, next_replicas.second);

  ASSERT_NE(next_replicas.first, replicas.first);
  ASSERT_NE(next_replicas.second, replicas.second);
}

TEST_F(PrngSeedStateTest, VanishingSeed) {
  auto prng_state = PrngSeedState::SetupSeeds(*graph_, identical_seed_,
                                              differing_seed_, seq_);

  // We want to check that calling ChangeStochasticRoundingMethod without
  // executing it's poplar::program doesn't modify the seed we're changing from
  // (differing in this case), which can get set to all 0s.
  poplar::program::Sequence unexecuted_seq;
  prng_state.ChangeStochasticRoundingMethod(
      StochasticRoundingMethod_IdenticalSeeds, unexecuted_seq);
  prng_state.ChangeStochasticRoundingMethod(
      StochasticRoundingMethod_DifferingSeeds, seq_);

  auto replicas = RunFloatToHalfCast();
  ASSERT_NE(replicas.first, replicas.second);
}

TEST_F(PrngSeedStateTest, AssertSeed) {
  auto expected_sr_method = StochasticRoundingMethod_DifferingSeeds;
  auto actual_sr_method = StochasticRoundingMethod_IdenticalSeeds;

  poplar::program::Sequence setup_seq;
  auto prng_state = PrngSeedState::SetupSeeds(*graph_, identical_seed_,
                                              differing_seed_, setup_seq);
  prng_state.ChangeStochasticRoundingMethod(actual_sr_method, setup_seq);
  auto hw_seed = poplar::getHwSeeds(*graph_, setup_seq);

  // Set the expected_sr_method and change the seed so it doesn't
  // reflect the semantics of the SR method we've set.
  prng_state.ChangeStochasticRoundingMethod(expected_sr_method, setup_seq);
  poplar::setHwSeeds(*graph_, hw_seed, setup_seq);

  // We want to check that the generated assert program causes an error if the
  // seed and SR method don't match.
  poplar::program::Sequence throw_seq(setup_seq);
  AssertStochasticRoundingMethod(*graph_, expected_sr_method, throw_seq);
  ASSERT_ANY_THROW(RunFloatToHalfCast(throw_seq));

  poplar::program::Sequence no_throw_seq(setup_seq);
  AssertStochasticRoundingMethod(*graph_, actual_sr_method, no_throw_seq);
  ASSERT_NO_THROW(RunFloatToHalfCast(no_throw_seq));
}

struct PrngSeedStateShardedTest : PrngSeedTest {
  void SetUp() override {
    // tile_count_ of 0 means use the native number of tiles.
    tile_count_ = 0;
    device_count_ = 4;
    replication_factor_ = 2;

    if (SetUpOrSkip()) {
      const poplar::Target& target = graph_->getTarget();
      ASSERT_EQ(target.getNumIPUs(), 2);
      const auto tiles_per_ipu = target.getTilesPerIPU();

      left_graph_ = graph_->createVirtualGraph(0, tiles_per_ipu);
      left_input_ =
          left_graph_.addConstant(poplar::FLOAT, {tensor_size_}, 0.1f);
      left_output_ = left_graph_.addVariable(
          poplar::FLOAT, {tensor_size_}, poplar::VariableMappingMethod::LINEAR);
      left_graph_.setTileMapping(left_input_, 0);
      left_graph_.createHostRead("left_outstream", left_output_);

      right_graph_ =
          graph_->createVirtualGraph(tiles_per_ipu, tiles_per_ipu * 2);
      right_input_ =
          right_graph_.addConstant(poplar::FLOAT, {tensor_size_}, 0.1f);
      right_output_ = right_graph_.addVariable(
          poplar::FLOAT, {tensor_size_}, poplar::VariableMappingMethod::LINEAR);
      right_graph_.setTileMapping(right_input_, 0);
      right_graph_.createHostRead("right_outstream", right_output_);
    }
  }

  void AddFloatToHalfCast(poplar::Graph& graph, const poplar::Tensor& input,
                          poplar::Tensor& output,
                          poplar::program::Sequence& seq,
                          const std::string& debug_name) {
    poplar::Tensor half =
        popops::cast(graph, input, poplar::HALF, seq, debug_name + "Half");
    poplar::Tensor full =
        popops::cast(graph, half, poplar::FLOAT, seq, debug_name + "Full");
    seq.add(poplar::program::Copy(full, output));
  }

  std::pair<std::vector<float>, std::vector<float>> RunSequence(
      poplar::program::Sequence& seq) {
    poplar::Engine engine(*graph_, seq);
    engine.load(device_);

    const uint32_t differing_seed_vals[] = {1, 2, 3, 4};
    engine.writeTensor("differing_seed", &differing_seed_vals[0],
                       &differing_seed_vals[4]);
    engine.run(0);

    std::vector<float> left_result(replication_factor_ * tensor_size_);
    engine.readTensor("left_outstream", &left_result.front(),
                      &left_result.back() + 1);

    auto right_result = left_result;
    engine.readTensor("right_outstream", &right_result.front(),
                      &right_result.back() + 1);

    return {left_result, right_result};
  }

  poplar::Graph left_graph_;
  poplar::Tensor left_input_;
  poplar::Tensor left_output_;

  poplar::Graph right_graph_;
  poplar::Tensor right_input_;
  poplar::Tensor right_output_;

  std::size_t tensor_size_ = 1000;
};

TEST_F(PrngSeedStateShardedTest, TaskParallelism) {
  // This test checks that using either stochastic rounding mode produces
  // the correct results when running a replicated graph with task parallelism,
  // where the seed changes can happen at the same time on different IPUs.
  poplar::program::Sequence seq;

  poplar::setStochasticRounding(*graph_, seq, true);
  auto prng_state =
      PrngSeedState::SetupSeeds(*graph_, identical_seed_, differing_seed_, seq);

  // An arbitrary operation so that both graphs don't start with the same
  // programs.
  auto right_square = popops::map(
      right_graph_, popops::expr::UnaryOpType::SQUARE, right_input_, seq);

  prng_state.ChangeStochasticRoundingMethod(
      StochasticRoundingMethod_IdenticalSeeds, seq);
  AddFloatToHalfCast(left_graph_, left_input_, left_output_, seq, "left");

  prng_state.ChangeStochasticRoundingMethod(
      StochasticRoundingMethod_DifferingSeeds, seq);
  AddFloatToHalfCast(right_graph_, right_input_, right_output_, seq, "right");

  // Since the operations in left/right graph have no data dependencies between
  // each other they're able to be run in parallel.
  const auto results = RunSequence(seq);
  const auto& left_result = results.first;
  const auto& right_result = results.second;

  auto left_replica1_start = left_result.begin();
  auto left_replica1_end = left_result.begin() + tensor_size_;
  auto left_replica2_start = left_replica1_end;
  ASSERT_TRUE(
      std::equal(left_replica1_start, left_replica1_end, left_replica2_start))
      << "The left_graph casts are done using identical seed so should produce "
         "the same result across replicas.";

  auto right_replica1_start = right_result.begin();
  auto right_replica1_end = right_result.begin() + tensor_size_;
  auto right_replica2_start = right_replica1_end;
  ASSERT_FALSE(std::equal(right_replica1_start, right_replica1_end,
                          right_replica2_start))
      << "The right_graph casts are done using differing seeds so should "
         "produce different results across replicas.";
}

// Fixture for tests that need to execute lowered hlo on an IPU device.
struct PrngSeedConsistencyTest : PrngSeedTest,
                                 ::testing::WithParamInterface<HloTestCase> {
  void SetUp() override {
    // Force enable synthetic data so that we don't have to setup any
    // infeed/outfeeds. Seeds are uneffected since we're setting those up
    // ourselves as part of the test.
    // This must be called first since the flags are global and set during
    // device setup.
    ForceEnableSyntheticData();

    if (SetUpOrSkip()) {
      auto config = GetModuleConfigForTest();
      config.set_replica_count(replication_factor_);

      const auto& hlo_string = GetParam().hlo;
      TF_ASSERT_OK_AND_ASSIGN(module_,
                              ParseAndReturnVerifiedModule(hlo_string, config));

      CustomOpReplacer custom_op_replacer;
      custom_op_replacer.Run(module_.get());

      HloTrivialScheduler scheduler;
      scheduler.Run(module_.get());

      IpuOptions::FloatingPointBehaviour floating_point_behaviour;
      floating_point_behaviour.set_esr(StochasticRounding_On);

      resources_ = CompilerResources::CreateTestDefault(
          module_.get(), /*enable_prng_seed_consistency_checks*/ true,
          floating_point_behaviour);
      resources_->annotations.input_output_aliasing_map =
          InputOutputAliasingMap(module_.get());
      resources_->module_call_graph = CallGraph::Build(module_.get());
      resources_->partition_replication_factor = replication_factor_;
      resources_->replication_factor = replication_factor_;
      resources_->main_graph = std::move(graph_);

      graph_ = nullptr;
    }
  }

  void TearDown() override {
    setenv(poplar_flag_name, original_poplar_flags_.c_str(),
           true /*overwrite*/);
  }

  void ForceEnableSyntheticData() {
    if (const char* flag_buffer = std::getenv(poplar_flag_name)) {
      original_poplar_flags_ = flag_buffer;
    }

    const auto poplar_flags =
        original_poplar_flags_ + " --use_synthetic_data=true";
    setenv(poplar_flag_name, poplar_flags.c_str(), true /*overwrite*/);
    PoplarXlaFlags::ReloadFlagsForTesting();
  }

  void RunSequence(poplar::Graph& graph, poplar::program::Sequence& seq) {
    poplar::Engine engine(graph, seq);
    engine.load(device_);

    const uint32_t differing_seed_vals[] = {1, 2, 3, 4};
    engine.writeTensor("differing_seed", &differing_seed_vals[0],
                       &differing_seed_vals[4]);

    engine.run(0);
  }

  std::unique_ptr<CompilerResources> resources_;
  std::unique_ptr<HloModule> module_;

  const char* poplar_flag_name = "TF_POPLAR_FLAGS";
  std::string original_poplar_flags_ = "";
};

// The specific instruction types within the various functions aren't that
// important, the stochastic_rounding_method values are the main thing as those
// are what determine the different state changes, which is what we're trying to
// test.
static const HloTestCase simple_repeat = {"simple_repeat", R"(
HloModule test
repeat {
  x = f32[] parameter(0), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  increment = f32[] constant(1), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_DifferingSeeds\"}"
  ROOT count = f32[] add(x, increment), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_IdenticalSeeds\"}"
}

ENTRY test {
  param0 = f32[] parameter(0), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  loop_count = f32[] call(param0), to_apply=repeat, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"3\"}}, \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\", \"stochastic_rounding\":\"THREESTATE_ON\"}"
  ROOT end = f32[] add(loop_count, loop_count), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_IdenticalSeeds\"}"
}
)"};
static const HloTestCase repeat_with_resource_update = {
    "repeat_with_resource_update", R"(
HloModule test
reduce_add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

resource_update {
  ru_arg0 = f32[] parameter(0), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  ru_arg1 = f32[] parameter(1), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  add0 = f32[] add(ru_arg0, ru_arg1), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_DifferingSeeds\"}"
  ROOT t = (f32[],f32[]) tuple(add0, ru_arg1), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  counter = s32[] constant(4), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  gac = () custom-call(s32[] counter), custom_call_target="GradientAccumulationCount", backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
}

loop {
  param0 = f32[] parameter(0), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  param1 = f32[] parameter(1), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_DifferingSeeds\"}"
  call_ru = (f32[],f32[]) call(param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}, \"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  gte0 = f32[] get-tuple-element(call_ru), index=0, backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  gte1 = f32[] get-tuple-element(call_ru), index=1, backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  add = f32[] add(gte1, gte0), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_DifferingSeeds\"}"
  reduce = f32[] all-reduce(add), to_apply=reduce_add, replica_groups={}, backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_IdenticalSeeds\"}"
  ROOT root = (f32[], f32[]) tuple(reduce, gte0), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
}

ENTRY e {
  weights0 = f32[] parameter(0), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  weights1 = f32[] parameter(1), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  loop_call = (f32[], f32[]) call(weights0, weights1), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"8\"}}, \"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_IdenticalSeeds\"}"
  loop_count = f32[] get-tuple-element(loop_call), index=0, backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  ROOT end = f32[] add(loop_count, loop_count), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_DifferingSeeds\"}"
}
)"};
static const HloTestCase while_loop = {"while_loop", R"(
HloModule test
body {
  p_body = (s32[],s32[]) parameter(0), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  p_body.0 = s32[] get-tuple-element(p_body), index=0, backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  const = s32[] constant(1), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  add = s32[] add(p_body.0, const), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_DifferingSeeds\"}"
  p_body.1 = s32[] get-tuple-element(p_body), index=1, backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  ROOT root = (s32[],s32[]) tuple(add, p_body.1), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
}

condition {
  p_cond = (s32[],s32[]) parameter(0), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  p_cond.0 = s32[] get-tuple-element(p_cond), index=0, backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  const = s32[] constant(10), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  ROOT result = pred[] compare(p_cond.0, const), direction=LT, backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_IdenticalSeeds\"}"
}

ENTRY entry {
  const_0 = s32[] constant(0), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  const_1 = s32[] constant(10), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  repeat_init = (s32[],s32[]) tuple(const_0, const_1), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_DifferingSeeds\"}"
  while = (s32[],s32[]) while(repeat_init), condition=condition, body=body, backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
  while_0 = s32[] get-tuple-element(while), index=0, backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_Any\"}"
	ROOT result = s32[] add(while_0, const_1), backend_config="{\"stochastic_rounding\":\"THREESTATE_ON\", \"stochastic_rounding_method\":\"StochasticRoundingMethod_DifferingSeeds\"}"
}
)"};
TEST_P(PrngSeedConsistencyTest, Check) {
  auto& graph = *(resources_->main_graph);

  // This test runs the given module with seed consistency checks (enabled at
  // CompilerResource construction) to make sure that seed value matches the
  // semantics of the StochasticRoundingMethod of the corresponding Hlo
  // instruction. So if the instruction is setup with
  // StochasticRoundingMethod_IdenticalSeeds but the seeds are differing across
  // replicas then the test will fail.
  poplar::program::Sequence seq;

  resources_->enable_experimental_prng_stability = true;
  resources_->prng_seed_state =
      PrngSeedState::SetupSeeds(graph, identical_seed_, differing_seed_, seq);

  auto entry = module_->entry_computation();
  auto order = module_->schedule().sequence(entry).instructions();
  EntryVisitor visitor(*resources_, entry);
  entry->AcceptOrdered(&visitor, order);

  seq.add(resources_->preamble_sequence);
  seq.add(visitor.GetSequenceAndInitializeCounters());

  // Failed consistency checks will cause poplar execution to stop and an
  // exception to be thrown.
  ASSERT_NO_THROW(RunSequence(graph, seq));
}

INSTANTIATE_TEST_SUITE_P(PrngSeedHlo, PrngSeedConsistencyTest,
                         ::testing::Values(simple_repeat,
                                           repeat_with_resource_update,
                                           while_loop),
                         HloTestCaseName);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
