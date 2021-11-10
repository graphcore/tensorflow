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

#include <gcl/Collectives.hpp>
#include <poplar/CSRFunctions.hpp>
#include <poplar/RandomSeed.hpp>
#include <popops/AllTrue.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
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
        auto old_seeds =
            poplar::getHwSeeds(graph, seq, {debug_name_and_id, "GetHwSeed"});
        poplar::setHwSeeds(graph, args[0], seq,
                           {debug_name_and_id, "SetHwSeed"});
        return old_seeds;
      },
      /*inlined*/ false, debug_name_and_id);
}

std::unique_ptr<poputil::graphfn::VoidFunction> CreateSetSRModeFn(
    poplar::Graph& graph, bool mode) {
  const std::string debug_name = mode ? "EnableSR" : "DisableSR";
  poplar::DebugNameAndId debug_name_and_id{debug_name};
  return absl::make_unique<poputil::graphfn::VoidFunction>(
      graph, poputil::graphfn::Signature{},
      [&graph, &mode, &debug_name_and_id](std::vector<poplar::Tensor>& args,
                                          poplar::program::Sequence& seq) {
        poplar::setStochasticRounding(graph, seq, mode, debug_name_and_id);
      },
      /*inlined*/ false, debug_name_and_id);
}
}  // namespace

/*static*/ PrngSeedState PrngSeedState::SetupSeed(
    poplar::Graph& graph, poplar::Tensor& seed,
    poplar::program::Sequence& seq) {
  const poplar::DebugContext& debug_context = {"__seed"};
  PoplarOpDefDebugInfo debug_info(debug_context, "InitializeSeed");

  poprand::setSeed(graph, seed, 0, seq, {debug_info, "set"});
  auto differing_hw_seed = poplar::getHwSeeds(graph, seq, {debug_info, "get"});

  // We want the behaviour to be consistent whether we're running
  // with single or multiple seeds, so even when there's no replication
  // we pretend that there's a separate identical seed and do everything
  // else as normal.
  return PrngSeedState(graph, StochasticRoundingMethod_DifferingSeeds,
                       differing_hw_seed, differing_hw_seed);
}

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
      poplar::getHwSeeds(graph, seq, {debug_info, "getDistinctHw"});

  // Speciifying DifferingSeeds since the last seed set was the differing one.
  return PrngSeedState(graph, StochasticRoundingMethod_DifferingSeeds,
                       identical_hw_seed, differing_hw_seed);
}

PrngSeedState::PrngSeedState(poplar::Graph& graph,
                             const StochasticRoundingMethod& initial_method,
                             poplar::Tensor& identical_hw_seed,
                             poplar::Tensor& differing_hw_seed)
    : change_hw_seeds_(CreateChangeHwSeedsFn(graph, identical_hw_seed)),
      set_sr_off_(CreateSetSRModeFn(graph, false)),
      set_sr_on_(CreateSetSRModeFn(graph, true)),
      identical_hw_seed_(identical_hw_seed),
      differing_hw_seed_(differing_hw_seed),
      stochastic_rounding_method_(initial_method) {}

StochasticRoundingMethod PrngSeedState::GetStochasticRoundingMethod() const {
  return stochastic_rounding_method_;
}

void PrngSeedState::SetStochasticRoundingMethod(
    const StochasticRoundingMethod& method) {
  CHECK_NE(method, StochasticRoundingMethod_Undefined);

  if (method != StochasticRoundingMethod_Any) {
    stochastic_rounding_method_ = method;
  }
}

bool PrngSeedState::ChangeStochasticRoundingMethod(
    const StochasticRoundingMethod& new_method, poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id) {
  CHECK(change_hw_seeds_);
  CHECK_NE(new_method, StochasticRoundingMethod_Undefined);

  const auto old_method = stochastic_rounding_method_;
  const bool change_method =
      new_method != StochasticRoundingMethod_Any && new_method != old_method;
  if (change_method) {
    // We do a poplar::program::Copy from the old seeds to avoid invalidating
    // differing_hw_seed_/identical_hw_seed_ if the given seq is not
    // executed, otherwise they can point at an uninitialized tensor.
    if (new_method == StochasticRoundingMethod_IdenticalSeeds) {
      std::vector<poplar::Tensor> args{identical_hw_seed_};
      auto old_seed = (*change_hw_seeds_)(args, seq, debug_name_and_id);
      seq.add(poplar::program::Copy(old_seed, differing_hw_seed_));
    } else if (old_method == StochasticRoundingMethod_IdenticalSeeds) {
      CHECK(new_method == StochasticRoundingMethod_DifferingSeeds ||
            new_method == StochasticRoundingMethod_None);
      // Always use differing_hw_seed_ since we want identical_hw_seed_ to
      // remain identical even when stochastic rounding is off.
      std::vector<poplar::Tensor> args{differing_hw_seed_};
      auto old_seed = (*change_hw_seeds_)(args, seq, debug_name_and_id);
      seq.add(poplar::program::Copy(old_seed, identical_hw_seed_));
    }

    std::vector<poplar::Tensor> empty_args;
    if (new_method == StochasticRoundingMethod_None) {
      (*set_sr_off_)(empty_args, seq, debug_name_and_id);
    } else if (old_method == StochasticRoundingMethod_None) {
      (*set_sr_on_)(empty_args, seq, debug_name_and_id);
    }

    stochastic_rounding_method_ = new_method;
    return true;
  }

  return false;
}

bool AssertStochasticRoundingMethod(poplar::Graph& graph,
                                    const StochasticRoundingMethod& method,
                                    poplar::program::Sequence& seq,
                                    const std::string& inst_name) {
  if (method != StochasticRoundingMethod_Any &&
      method != StochasticRoundingMethod_None) {
    // Verbose logging so it's clear when we're asserting and harder to
    // accidentially submit code with it enabled.
    LOG(INFO) << "AssertStochasticRoundingMethod "
              << StochasticRoundingMethod_Name(method) << " for " << inst_name;

    auto seeds = poplar::getHwSeeds(graph, seq, {});

    auto all_seeds = gcl::allGatherCrossReplica(graph, seeds, seq);
    const auto replication_factor = all_seeds.dim(0);

    // Compare the seed in different replicas. Aborting if they're
    // not identical/differing respetively.
    for (auto i = 1u; i < replication_factor; ++i) {
      auto equal = popops::map(graph, popops::expr::BinaryOpType::EQUAL,
                               all_seeds[0], all_seeds[i], seq);
      auto all_equal = popops::allTrue(graph, equal, seq);

      if (method == StochasticRoundingMethod_IdenticalSeeds) {
        auto not_all_equal = popops::map(
            graph, popops::expr::UnaryOpType::LOGICAL_NOT, all_equal, seq);
        const std::string message =
            "SR method is set to identical but seeds are differing on replicas "
            "0 and " +
            std::to_string(i) + " for " + inst_name;
        seq.add(poplar::program::AbortOnCondition(not_all_equal, message));
      } else {
        CHECK_EQ(method, StochasticRoundingMethod_DifferingSeeds);

        const std::string message =
            "SR method is set to differing but seeds are identical on replicas "
            "0 and " +
            std::to_string(i) + " for " + inst_name;
        seq.add(poplar::program::AbortOnCondition(all_equal, message));
      }
    }

    return true;
  }

  return false;
}

void AssertStochasticRoundingEnabled(poplar::Graph& graph, bool enabled,
                                     poplar::program::Sequence& seq,
                                     const std::string& inst_name) {
  // Verbose logging so it's clear when we're asserting and harder to
  // accidentially submit code with it enabled.
  LOG(INFO) << "AssertStochasticRoundingEnabled " << enabled << " for "
            << inst_name;

  // Temp options to pass into getAndModifyFloatingPointBehaviour, their values
  // dont really matter since they'll be reverted.
  const poplar::FloatingPointBehaviour no_clear(false, false, false, false,
                                                false);
  const poplar::FloatingPointBehaviour temp_set(false, false, false, false,
                                                false);
  const auto old_fp = poplar::getAndModifyFloatingPointBehaviour(
      graph, seq, no_clear, temp_set);

  // We say SR is enabled if every tile has this bit mask.
  const auto sr_enabled_value =
      graph.getTarget().makeFpIctlValue(false, false, false, true, false);

  auto tiles_sr_value =
      popops::bitwiseAnd(graph, old_fp, sr_enabled_value, seq);
  auto tiles_sr_enabled =
      popops::eq(graph, tiles_sr_value, sr_enabled_value, seq);
  auto all_tiles_sr_enabled = popops::reduce(
      graph, tiles_sr_enabled, {0}, popops::Operation::LOGICAL_AND, seq);

  if (enabled) {
    auto all_tiles_sr_disabled =
        popops::map(graph, popops::expr::UnaryOpType::LOGICAL_NOT,
                    all_tiles_sr_enabled, seq);
    seq.add(poplar::program::AbortOnCondition(
        all_tiles_sr_disabled, "Expected SR to be enabled for " + inst_name));
  } else {
    seq.add(poplar::program::AbortOnCondition(
        all_tiles_sr_enabled, "Expected SR to be disabled for " + inst_name));
  }

  poplar::setFloatingPointBehaviour(graph, seq, old_fp);
}

}  // namespace poplarplugin
}  // namespace xla
