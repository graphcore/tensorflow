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

#include "tensorflow/compiler/xla/types.h"

#include <mutex>

namespace xla {
namespace poplarplugin {

/**
 * A seed generator used to make sure drawing seeds for multiIPU devices from
 * the random number generator is deterministic in a multithreaded environment.
 */

class SeedGenerator {
 public:
  // Seed the underlying random number generator.
  void Seed(unsigned seed);

  // Prepare `replication_factor` seeds to draw from.
  void PrepareSeedsForReplicas(int64 replication_factor);

  // Get the seed value for the `replica_idx` IPU replica device.
  uint64 Get(int64 replica_idx) const;

 private:
  // Used for storage of the values.
  std::vector<uint64> buffer_;

  // The generator from which the values are drawn from.
  std::mt19937_64 seed_generator_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
