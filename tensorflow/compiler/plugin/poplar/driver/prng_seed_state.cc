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

#include "tensorflow/compiler/plugin/poplar/driver/prng_seed_state.h"

#include <vector>

#include "absl/memory/memory.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/core/platform/logging.h"

#include <poplar/RandomSeed.hpp>
#include <poprand/RandomGen.hpp>

namespace xla {
namespace poplarplugin {
namespace {

std::unique_ptr<poputil::graphfn::TensorFunction> CreateChangeHwSeedsFn(
    poplar::Graph& graph, poplar::Tensor& seed_template) {
  poplar::DebugNameAndId debug_name_and_id{"ChangeHwSeeds"};
  return absl::make_unique<poputil::graphfn::TensorFunction>(
      graph,
      poputil::graphfn::Signature{poputil::graphfn::input(seed_template)},
      [&graph, &debug_name_and_id](std::vector<poplar::Tensor>& args,
                                   poplar::program::Sequence& seq) {
        auto old_seeds = poplar::getHwSeeds(graph, seq, {debug_name_and_id});
        poplar::setHwSeeds(graph, args[0], seq, {debug_name_and_id});
        return old_seeds;
      },
      /*inlined*/ false, debug_name_and_id);
}
}  // namespace

/*static*/ PrngSeedState PrngSeedState::SetupSeeds(
    poplar::Graph& graph, poplar::Tensor& identical_seed,
    poplar::Tensor& differing_seed, poplar::program::Sequence& seq) {
  const poplar::DebugContext& debug_context = {"__seed"};
  PoplarOpDefDebugInfo debug_info(debug_context, "InitializeSeed");

  poprand::setSeed(graph, identical_seed, 0, seq, {debug_info, "setIdentical"});
  auto identical_hw_seed =
      poplar::getHwSeeds(graph, seq, {debug_info, "getIdenticalHw"});

  poprand::setSeed(graph, differing_seed, 0, seq, {debug_info, "setDistinct"});
  auto differing_hw_seed =
      poplar::getHwSeeds(graph, seq, {debug_info, "getIdenticalHw"});

  // Speciifying DifferingSeeds since the last seed set was the differing one.
  return PrngSeedState(graph, StochasticRoundingMethod_DifferingSeeds,
                       identical_hw_seed, differing_hw_seed);
}

PrngSeedState::PrngSeedState(poplar::Graph& graph,
                             const StochasticRoundingMethod& initial_method,
                             poplar::Tensor& identical_hw_seed,
                             poplar::Tensor& differing_hw_seed)
    : change_hw_seeds_(CreateChangeHwSeedsFn(graph, identical_hw_seed)),
      identical_hw_seed_(identical_hw_seed),
      differing_hw_seed_(differing_hw_seed),
      stochastic_rounding_method_(initial_method) {}

StochasticRoundingMethod PrngSeedState::GetStochasticRoundingMethod() const {
  return stochastic_rounding_method_;
}

bool PrngSeedState::ChangeStochasticRoundingMethod(
    const StochasticRoundingMethod& new_method, poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id) {
  CHECK(change_hw_seeds_);
  CHECK_NE(new_method, StochasticRoundingMethod_Undefined);

  const bool change_seed = new_method != StochasticRoundingMethod_Any &&
                           new_method != stochastic_rounding_method_;
  if (change_seed) {
    if (new_method == StochasticRoundingMethod_IdenticalSeeds) {
      std::vector<poplar::Tensor> args{identical_hw_seed_};
      differing_hw_seed_ = (*change_hw_seeds_)(args, seq, debug_name_and_id);
    } else {
      CHECK_EQ(new_method, StochasticRoundingMethod_DifferingSeeds);

      std::vector<poplar::Tensor> args{differing_hw_seed_};
      identical_hw_seed_ = (*change_hw_seeds_)(args, seq, debug_name_and_id);
    }

    stochastic_rounding_method_ = new_method;
    return true;
  }

  return false;
}

}  // namespace poplarplugin
}  // namespace xla
