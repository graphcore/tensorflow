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

#include "tensorflow/compiler/xla/literal_util.h"

#include <poplar/CSRFunctions.hpp>
#include <popops/Cast.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>

namespace xla {
namespace poplarplugin {
namespace {

struct PrngSeedStateTest : HloPoplarTestBase {
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(auto ipu_count, GetMaxIpuCount());
    const bool enough_hw = ipu_count >= replication_factor_;
    if (!enough_hw) {
      GTEST_SKIP() << "Skipping PrngSeedStateTests. They need to run with "
                   << replication_factor_ << " IPUs but only " << ipu_count
                   << "available.";
    } else {
      TF_ASSERT_OK_AND_ASSIGN(device_, CreateIpuDevice(replication_factor_, 4));

      graph_ = absl::make_unique<poplar::Graph>(
          device_, poplar::replication_factor(replication_factor_));
      popops::addCodelets(*graph_);
      poprand::addCodelets(*graph_);

      differing_seed_ = graph_->addVariable(
          poplar::UNSIGNED_INT, {2}, poplar::VariableMappingMethod::LINEAR);
      graph_->createHostWrite("differing_seed", differing_seed_);

      identical_seed_ = graph_->addVariable(
          poplar::UNSIGNED_INT, {2}, poplar::VariableMappingMethod::LINEAR);
      graph_->setInitialValue<unsigned>(identical_seed_, {10, 11});

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
    poplar::Tensor half = popops::cast(*graph_, input_, poplar::HALF, seq_);
    poplar::Tensor full = popops::cast(*graph_, half, poplar::FLOAT, seq_);
    seq_.add(poplar::program::Copy(full, output_));

    poplar::Engine engine(*graph_, seq_);
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

  int64 replication_factor_ = 2;
  poplar::Device device_;

  std::unique_ptr<poplar::Graph> graph_;

  poplar::Tensor differing_seed_;
  poplar::Tensor identical_seed_;
  poplar::Tensor input_;
  poplar::Tensor output_;

  poplar::program::Sequence seq_;
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

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
