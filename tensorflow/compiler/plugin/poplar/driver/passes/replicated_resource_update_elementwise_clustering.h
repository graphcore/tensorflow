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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_REPLICATED_RESOURCE_UPDATE_ELEMENTWISE_CLUSTERING_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_REPLICATED_RESOURCE_UPDATE_ELEMENTWISE_CLUSTERING_H_

#include <memory>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/passes/resource_update_elementwise_clustering.h"

namespace xla {
namespace poplarplugin {

class CompilerAnnotations;

class ReplicatedResourceUpdateElementwiseClustering final
    : public ResourceUpdateElementwiseClustering {
  using Base = ResourceUpdateElementwiseClustering;

 public:
  ReplicatedResourceUpdateElementwiseClustering(
      CompilerAnnotations& annotations, uint32 partition_replication_factor,
      uint32 global_replication_factor)
      : annotations_(annotations),
        partition_replication_factor_(partition_replication_factor),
        global_replication_factor_(global_replication_factor) {}

  explicit ReplicatedResourceUpdateElementwiseClustering(
      CompilerAnnotations& annotations, uint32 replication_factor)
      : ReplicatedResourceUpdateElementwiseClustering(
            annotations, replication_factor, replication_factor) {}

  absl::string_view name() const override {
    return "replicated-resource-update-elementwise-clustering";
  }

 private:
  std::unique_ptr<ElementwiseClusterValidator> CreateValidator(
      const HloInstruction* call,
      const HloInstruction* resource_update) const override;

  static StatusOr<HloInstruction*> PadInput(const ElementwiseCluster& cluster,
                                            HloInstruction* input,
                                            HloComputation::Builder* builder);

  StatusOr<HloInstruction*> AddClusterInput(
      int64 param_idx, const ElementwiseCluster& cluster,
      HloInstruction* cluster_input, HloComputation::Builder* builder,
      HloCloneContext* context) const override;

  StatusOr<HloInstruction*> AddClusterOutput(
      const ElementwiseCluster& cluster, HloInstruction* cluster_output,
      std::vector<UserPositions>& inst_users, HloComputation::Builder* builder,
      HloCloneContext* context) const override;

  Status AddClusterInstruction(const ElementwiseCluster& cluster,
                               HloInstruction* inst,
                               HloComputation::Builder* builder,
                               HloCloneContext* context) const override;

  ClusterOutlinePolicy GetClusterOutlinePolicy(
      const ElementwiseCluster& cluster) const override;

  Status UpdateClusterBackendConfig(
      const ElementwiseCluster& cluster,
      PoplarBackendConfig& backend_config) const override;

  Status ValidateResourceUpdateAndClusters(
      const HloInstruction* ru,
      std::vector<ElementwiseCluster> clusters) const override;

 private:
  CompilerAnnotations& annotations_;
  uint32 partition_replication_factor_;
  uint32 global_replication_factor_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_REPLICATED_RESOURCE_UPDATE_ELEMENTWISE_CLUSTERING_H_
