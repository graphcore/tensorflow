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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/elementwise_cluster.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

class HloModule;

namespace poplarplugin {

enum struct ClusterOutlinePolicy { Ignore, Outline, OutlineNonUnique };

// Find and replace clusters of elementwise instructions, sharding resource
// update computation across replicas. For each cluster, remove
// all-gather(remote-parameter-load) and store result in remote buffer shard.
class ResourceUpdateElementwiseClustering : public HloModulePass {
 public:
  ResourceUpdateElementwiseClustering() {}

  absl::string_view name() const override {
    return "resource-update-elementwise-clustering";
  }

  StatusOr<bool> Run(HloModule* module) override;

  // Exposed for tests only.
  virtual Status RunDataflowAnalysis(const HloModule* module);

  // Get clusters inside of the call, where the call has to be a repeat loop or
  // a pipeline.
  StatusOr<std::vector<ElementwiseCluster>> GetClustersIn(
      HloInstruction* const call) const;

  // Outline the provided cluster - returns the call instruction to the cluster.
  StatusOr<HloInstruction*> OutlineCluster(ElementwiseCluster& cluster) const;

 protected:
  // Clone instruction using operands from HloCloneContext
  static Status CloneInstruction(const Shape& shape, const HloInstruction* inst,
                                 HloComputation::Builder* builder,
                                 HloCloneContext* context);

  // Creates validator specific to the concrete pass implementation.
  virtual std::unique_ptr<ElementwiseClusterValidator> CreateValidator(
      const HloComputation* resource_update_comp) const;

  std::unique_ptr<ElementwiseClusterValidator> CreateValidator(
      const ElementwiseClusterValidator::Inputs& valid_inputs) const;

  virtual StatusOr<HloInstruction*> AddClusterInput(
      int64_t param_idx, const ElementwiseCluster& cluster,
      HloInstruction* cluster_input, HloComputation::Builder* builder,
      HloCloneContext* context) const;

  virtual StatusOr<HloInstruction*> AddClusterOutput(
      const ElementwiseCluster& cluster, HloInstruction* cluster_output,
      std::vector<UserPositions>& inst_users, HloComputation::Builder* builder,
      HloCloneContext* context) const;

  virtual Status AddClusterInstruction(const ElementwiseCluster& cluster,
                                       HloInstruction* inst,
                                       HloComputation::Builder* builder,
                                       HloCloneContext* context) const;

  virtual ClusterOutlinePolicy GetClusterOutlinePolicy(
      const ElementwiseCluster& cluster) const;

  virtual Status UpdateClusterBackendConfig(
      const ElementwiseCluster& cluster,
      PoplarBackendConfig& backend_config) const;

  // Assess a resource update and the clusters generated from it, printing
  // helpful warnings if something doesn't look right.
  virtual Status ValidateResourceUpdateAndClusters(
      const HloInstruction* ru, std::vector<ElementwiseCluster> clusters) const;

 private:
  StatusOr<bool> RewriteCall(HloModule* module, HloInstruction* call) const;

  StatusOr<HloInstruction*> AddClusterInputToOutlinedComputation(
      int64_t param_idx, const ElementwiseCluster& cluster,
      HloInstruction* cluster_input, HloComputation::Builder* builder,
      HloCloneContext* context) const;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_RESOURCE_UPDATE_ELEMENTWISE_CLUSTERING_H_
