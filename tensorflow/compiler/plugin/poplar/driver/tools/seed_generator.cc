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

#include "tensorflow/compiler/plugin/poplar/driver/tools/seed_generator.h"

#include "tensorflow/core/platform/default/logging.h"

namespace xla {
namespace poplarplugin {

void SeedGenerator::Seed(unsigned seed) { seed_generator_.seed(seed); }

void SeedGenerator::PrepareSeedsForReplicas(int64 replication_factor) {
  buffer_.resize(replication_factor);
  for (int64 i = 0; i != replication_factor; ++i) {
    buffer_[i] = seed_generator_();
  }
}

uint64 SeedGenerator::Get(int64 replica_idx) const {
  CHECK(static_cast<size_t>(replica_idx) < buffer_.size());
  return buffer_.at(replica_idx);
}
}  // namespace poplarplugin
}  // namespace xla
