/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_range_sampler.h"

#include <cmath>
#include <poplar/RandomSeed.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Fill.hpp>
#include <poprand/RandomGen.hpp>
#include <utility>

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {

Status RangeSampler::Sample(poplar::Graph& graph, poplar::Tensor& samples,
                            poplar::program::Sequence& seq) {
  const poplar::DebugNameAndId debug_name_and_id = {GetDebugNameAndId(),
                                                    "Sample"};
  poplar::Tensor out;
  switch (distribution_) {
    case DistributionType::UNIFORM:
      out = poprand::uniform(graph, &seed_, 0, samples, poplar::INT, 0.0,
                             range_max_ - 1, seq, {debug_name_and_id});
      break;
    case DistributionType::LOG_UNIFORM:
      out = poprand::logUniform(graph, &seed_, 0, samples, poplar::INT, 1.0,
                                range_max_, seq, M_E, {debug_name_and_id});
      popops::mapInPlace(graph, pe::Sub(pe::_1, pe::Const(1)), {out}, seq,
                         {debug_name_and_id, "Minus1"});
      break;
    default:
      return xla::Unimplemented("Distribution not supported.");
  }
  seq.add(poplar::program::Copy(out, samples, false, {debug_name_and_id}));
  return Status::OK();
}

StatusOr<poplar::Tensor> RangeSampler::Expectation(
    poplar::Graph& graph, const poplar::Tensor& samples, const uint64 k,
    poplar::program::Sequence& seq) {
  // E[X] = k*P[X=x]
  TF_ASSIGN_OR_RETURN(poplar::Tensor probabilities,
                      Probabilities(graph, samples, seq));
  pe::Any expect_fn = pe::Mul(pe::_1, pe::Const(static_cast<float>(k)));
  popops::mapInPlace(graph, expect_fn, {probabilities}, seq,
                     {dnai_, "Expectation"});
  return probabilities;
}

StatusOr<poplar::Tensor> RangeSampler::Probabilities(
    poplar::Graph& graph, const poplar::Tensor& samples,
    poplar::program::Sequence& seq) {
  const poplar::DebugNameAndId debug_name_and_id = {GetDebugNameAndId(),
                                                    "Probabilities"};
  switch (distribution_) {
    case DistributionType::UNIFORM: {
      // X~U[0, N-1], P[X=x] = 1/N
      poplar::Tensor probabilities =
          graph.clone(poplar::FLOAT, samples, {debug_name_and_id});
      float prob = 1.0f / static_cast<float>(range_max_);
      popops::fill(graph, probabilities, seq, prob, {debug_name_and_id});
      return probabilities;
    }
    case DistributionType::LOG_UNIFORM: {
      // X~LogUniform[0, N-1], P[X=x] = ln((x+2)/(x+1)) / ln(N+1)
      pe::Any log_uniform_pdf = pe::Divide(
          pe::Log(pe::Divide(
              pe::Add(pe::Cast(pe::_1, poplar::FLOAT), pe::Const(2.0f)),
              pe::Add(pe::Cast(pe::_1, poplar::FLOAT), pe::Const(1.0f)))),
          pe::Const(static_cast<float>(std::log1p(range_max_))));
      return popops::map(graph, log_uniform_pdf, {samples}, seq,
                         {debug_name_and_id});
    }
    default:
      return xla::Unimplemented("Distribution not supported.");
  }
}

Status UniqueRangeSampler::Sample(poplar::Graph& graph, poplar::Tensor& samples,
                                  poplar::program::Sequence& seq) {
  const poplar::DebugNameAndId debug_name_and_id = {GetDebugNameAndId(),
                                                    "Sample"};

  auto cs = graph.addComputeSet({debug_name_and_id, "SequentialSample"});
  // Create a bitmask with at least RangeMax bits for O(1) lookup time, made up
  // of several 32bit integers
  const uint64 num_bitmasks = (RangeMax() >> 5) + 1;
  auto bitmasks = graph.addVariable(poplar::UNSIGNED_INT, {num_bitmasks},
                                    {debug_name_and_id, "bitmasks"});
  graph.setTileMapping(bitmasks, Tile());
  popops::fill(graph, bitmasks, seq, 0u, {debug_name_and_id});

  // Create a num_tries tensor to calculate expectation later
  num_tries_ =
      graph.addVariable(poplar::INT, {}, {debug_name_and_id, "num_tries"});
  graph.setTileMapping(num_tries_, Tile());

  // Create the right vertex for the distribution and set it up
  poplar::VertexRef v;
  uint64 prng_cycles;
  switch (Distribution()) {
    case DistributionType::UNIFORM:
      v = graph.addVertex(cs, "UniformUniqueSeqSample");
      graph.setInitialValue(v["scale"], RangeMax() - 1);
      prng_cycles = 19;
      break;
    case DistributionType::LOG_UNIFORM:
      v = graph.addVertex(cs, "LogUniformUniqueSeqSample");
      graph.setInitialValue(v["scale"], std::log(RangeMax()));
      prng_cycles = 21;
      break;
    default:
      return xla::Unimplemented("Distribution not supported.");
  }
  graph.connect(v["num_tries"], num_tries_);
  graph.connect(v["samples"], samples);
  graph.connect(v["bitmasks"], bitmasks);
  graph.setTileMapping(v, Tile());
  // Note: The best cycle estimate would be a very complicated function of k and
  // N which boils down to the biased urn problem. Instead, we optimistically
  // assume every sample taken is unique a.k.a. the minimum number of cycles the
  // vertex could possibly take.
  graph.setPerfEstimate(v, 9 + prng_cycles * samples.numElements());

  // Execute the compute set with a temporary seed
  auto oldSeed = poplar::getHwSeeds(graph, seq, {debug_name_and_id});
  poprand::setSeed(graph, Seed(), 0u, seq, {debug_name_and_id});
  seq.add(poplar::program::Execute(cs, {debug_name_and_id}));
  poplar::setHwSeeds(graph, oldSeed, seq, {debug_name_and_id});
  return Status::OK();
}

StatusOr<poplar::Tensor> UniqueRangeSampler::Expectation(
    poplar::Graph& graph, const poplar::Tensor& samples, const uint64 k,
    poplar::program::Sequence& seq) {
  const poplar::DebugNameAndId debug_name_and_id = {GetDebugNameAndId(),
                                                    "Expectation"};
  // In unique case, estimate expectation from num_tries calculated from last
  // usage of Sample()
  // E[X] = -expm1(num_tries * log1p(-P[X=x]))
  TF_ASSIGN_OR_RETURN(poplar::Tensor probabilities,
                      Probabilities(graph, samples, seq));
  pe::Any expect_fn = pe::Neg(pe::Expm1(
      pe::Mul(pe::Log1p(pe::Neg(pe::_1)), pe::Cast(pe::_2, poplar::FLOAT))));
  popops::mapInPlace(graph, expect_fn, {probabilities, num_tries_}, seq,
                     {debug_name_and_id});
  return probabilities;
}

StatusOr<DistributionType> DistributionStringToEnum(
    const std::string distribution) {
  if (distribution == "uniform") {
    return DistributionType::UNIFORM;
  }
  if (distribution == "log_uniform") {
    return DistributionType::LOG_UNIFORM;
  }
  return UnimplementedStrCat("Distribution ", distribution, " not supported.");
}

StatusOr<std::unique_ptr<RangeSampler>> RangeSamplerFactory(
    const std::string distribution,
    const poplar::DebugNameAndId& debug_name_and_id, const uint64 range_max,
    const uint64 tile, const poplar::Tensor& seed, bool unique) {
  TF_ASSIGN_OR_RETURN(const DistributionType dist_type,
                      DistributionStringToEnum(distribution));
  std::unique_ptr<RangeSampler> sampler;
  if (unique) {
    sampler = absl::make_unique<UniqueRangeSampler>(
        dist_type, debug_name_and_id, range_max, tile, seed);
  } else {
    sampler = absl::make_unique<RangeSampler>(dist_type, debug_name_and_id,
                                              range_max, tile, seed);
  }
  return std::move(sampler);
}

}  // namespace poplarplugin
}  // namespace xla
