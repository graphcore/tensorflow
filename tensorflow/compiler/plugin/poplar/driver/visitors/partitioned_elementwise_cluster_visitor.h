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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_PARTITIONED_ELEMENTWISE_CLUSTER_VISITOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_PARTITIONED_ELEMENTWISE_CLUSTER_VISITOR_H_

#include <map>
#include <memory>
#include <vector>

#include <gcl/CollectiveBalancedReorder.hpp>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/deferred_visitor.h"

namespace xla {
class HloDataflowAnalysis;

namespace poplarplugin {

struct CompilerResources;

/*
 * This visitor handles elementwise clusters and creates
 * CollectiveBalancedReorder objects.
 */
class PartitionedElementwiseClusterVisitor : public DeferredVisitor {
 public:
  struct HostRearrangementInfo {
    std::unique_ptr<gcl::CollectiveBalancedReorder> host_rearrangement;
    int64_t host_rearrangement_id;
  };

  PartitionedElementwiseClusterVisitor(
      int64_t next_rearrangement_id, CompilerResources& res,
      const DeferredArgRBVectors& callsite_inputs,
      const poplar::DebugNameAndId& debug_name_and_id,
      bool allocate_all_input_tensors = true,
      const std::vector<const DeferredVisitor*>& dependent_computations = {},
      bool reallocate_inputs = true);

  PartitionedElementwiseClusterVisitor(
      int64_t next_rearrangement_id, CompilerResources& res,
      const DeferredArgRBVectors& callsite_inputs,
      const poplar::DebugNameAndId& debug_name_and_id,
      bool allocate_all_input_tensors,
      const std::vector<const DeferredVisitor*>& dependent_computations,
      const ReallocateInputsInfo& reallocate_inputs_info);
  ~PartitionedElementwiseClusterVisitor();

  Status Preprocess(HloInstruction* inst) override;
  Status ValidateShape(HloInstruction* inst, std::size_t tuple_index,
                       const Shape& shape,
                       const TensorOrRemoteBuffer& out) override;
  DeferredAllocateFunction MakeParameterAllocationFunction(
      TensorLocation allocation_location, const Shape& shape,
      absl::optional<TensorOrRemoteBuffer> tensor_like,
      const poplar::DebugNameAndId& debug_name_and_id) override;

  StatusOr<bool> UpdateRemoteBufferInformation(
      int64_t entry_param_idx, const HloInstruction* entry_param);
  Status UpdateRemoteBuffersInformation();
  Status SetRemoteBufferHostRearrangementId(DriverGraph& graph,
                                            const HloComputation* entry_comp,
                                            int64_t entry_param_idx,
                                            int64_t host_rearrangement_id,
                                            int64_t elements_per_replica);

  Status FinishDeferedAllocationVisit(HloInstruction* inst) override;

  StatusOr<const HostRearrangementInfo*> GetCollectiveBalancedReorder(
      const HloInstruction* inst);
  Status SetCollectiveBalancedReorder(
      const HloInstruction* inst,
      std::unique_ptr<gcl::CollectiveBalancedReorder> cbr);

  int64_t GetNextRearrangementId() const { return next_rearrangement_id_; }

 private:
  using HloShardingPtr = std::shared_ptr<const HloSharding>;
  struct HloShardingPtrEqual {
    bool operator()(const HloShardingPtr& a, const HloShardingPtr& b) const {
      return a && b ? (*a == *b) : (!a && !b);
    }
  };

  struct HloShardingPtrHash {
    std::size_t operator()(const HloShardingPtr& sharding) const {
      return sharding ? sharding->Hash() : 0;
    }
  };

  int64_t next_rearrangement_id_;
  std::unique_ptr<HloDataflowAnalysis> dfa_;
  absl::flat_hash_map<HloShardingPtr, HostRearrangementInfo, HloShardingPtrHash,
                      HloShardingPtrEqual>
      cbr_;
  std::map<int64_t, const HloInstruction*> entry_params_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_PARTITIONED_ELEMENTWISE_CLUSTER_VISITOR_H_
