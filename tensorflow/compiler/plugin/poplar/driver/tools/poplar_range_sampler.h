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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_POPLAR_RANGE_SAMPLER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_POPLAR_RANGE_SAMPLER_H_

#include <memory>
#include <string>

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace poplarplugin {

enum DistributionType { UNIFORM, LOG_UNIFORM };

class RangeSampler {
 public:
  RangeSampler(const DistributionType distribution,
               const poplar::DebugNameAndId& debug_name_and_id,
               const uint64 range_max, const uint64 tile,
               const poplar::Tensor& seed)
      : distribution_(distribution),
        dnai_(debug_name_and_id),
        range_max_(range_max),
        tile_(tile),
        seed_(seed) {}

  virtual ~RangeSampler() = default;

  virtual Status Sample(poplar::Graph& graph, poplar::Tensor& samples,
                        poplar::program::Sequence& seq);

  virtual StatusOr<poplar::Tensor> Expectation(poplar::Graph& graph,
                                               const poplar::Tensor& samples,
                                               const uint64 k,
                                               poplar::program::Sequence& seq);

  const poplar::DebugNameAndId& GetDebugNameAndId() const { return dnai_; }
  const DistributionType Distribution() const { return distribution_; }
  const uint64 RangeMax() const { return range_max_; }
  const uint64 Tile() const { return tile_; }
  const poplar::Tensor Seed() const { return seed_; }

 protected:
  StatusOr<poplar::Tensor> Probabilities(poplar::Graph& graph,
                                         const poplar::Tensor& samples,
                                         poplar::program::Sequence& seq);

 private:
  const DistributionType distribution_;
  const poplar::DebugNameAndId dnai_;
  const uint64 range_max_;
  const uint64 tile_;
  const poplar::Tensor seed_;
};

class UniqueRangeSampler : public RangeSampler {
 public:
  UniqueRangeSampler(const DistributionType distribution,
                     const poplar::DebugNameAndId& debug_name_and_id,
                     const uint64 range_max, const uint64 tile,
                     const poplar::Tensor& seed)
      : RangeSampler(distribution, debug_name_and_id, range_max, tile, seed) {}

  Status Sample(poplar::Graph& graph, poplar::Tensor& samples,
                poplar::program::Sequence& seq) override;

  StatusOr<poplar::Tensor> Expectation(poplar::Graph& graph,
                                       const poplar::Tensor& samples,
                                       const uint64 k,
                                       poplar::program::Sequence& seq) override;

 private:
  poplar::Tensor num_tries_;
};

StatusOr<DistributionType> DistributionStringToEnum(
    const std::string distribution);

// Function to instantiate a RangeSampler with a string
StatusOr<std::unique_ptr<RangeSampler>> RangeSamplerFactory(
    const std::string distribution,
    const poplar::DebugNameAndId& debug_name_and_id, const uint64 range_max,
    const uint64 tile, const poplar::Tensor& seed, bool unique);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_POPLAR_RANGE_SAMPLER_H_
