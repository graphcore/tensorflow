/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SEED_GENERATOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SEED_GENERATOR_H_

#include <random>

#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace poplarplugin {

/**
 * A seed generator used to make sure drawing seeds for multiIPU devices from
 * the random number generator is deterministic in a multithreaded environment.
 */

class SeedGenerator {
 public:
  virtual ~SeedGenerator() = default;

  // Prepare `replication_factor` seeds to draw from.
  virtual void PrepareSeedsForReplicas(int64_t replication_factor) = 0;

  // Get the seed value for the `replica_idx` IPU replica device.
  virtual uint64 Get(int64_t replica_idx) const = 0;
};

class DistinctReplicaSeedGenerator : public SeedGenerator {
 public:
  explicit DistinctReplicaSeedGenerator(unsigned seed);

  void PrepareSeedsForReplicas(int64_t replication_factor) override;

  uint64 Get(int64_t replica_idx) const override;

 private:
  // The next value for each replica.
  std::vector<uint64> buffer_;

  // The generator from which the values are drawn from.
  std::mt19937_64 seed_generator_;
};

class IdenticalReplicaSeedGenerator : public SeedGenerator {
 public:
  explicit IdenticalReplicaSeedGenerator(unsigned seed);

  void PrepareSeedsForReplicas(int64_t replication_factor) override;

  uint64 Get(int64_t replica_idx) const override;

 private:
  // The next value for all replicas.
  uint64 value_;

  // The generator from which the values are drawn from.
  std::mt19937_64 seed_generator_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
