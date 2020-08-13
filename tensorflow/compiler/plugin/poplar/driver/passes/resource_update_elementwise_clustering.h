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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_RESOURCE_UPDATE_ELEMENTWISE_CLUSTERING_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_RESOURCE_UPDATE_ELEMENTWISE_CLUSTERING_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

class HloModule;

namespace poplarplugin {

// Find and replace clusters of elementwise instructions, sharding resource
// update computation across replicas. For each cluster, remove
// all-gather(remote-parameter-load) and store result in remote buffer shard.
class ResourceUpdateElementwiseClustering : public HloModulePass {
  uint32 replication_factor_;

 public:
  explicit ResourceUpdateElementwiseClustering(uint32 replication_factor)
      : replication_factor_(replication_factor) {}

  absl::string_view name() const override {
    return "resource-update-elementwise-clustering";
  }

  StatusOr<bool> Run(HloModule* module);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_RESOURCE_UPDATE_ELEMENTWISE_CLUSTERING_H_
