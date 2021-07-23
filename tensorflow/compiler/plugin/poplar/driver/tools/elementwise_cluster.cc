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

#include "tensorflow/compiler/plugin/poplar/driver/tools/elementwise_cluster.h"

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/all_gather.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/reduce_scatter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/replication_factor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/replication_index.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {
namespace {

bool IsBroadcast(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kBroadcast;
}

bool IsParameter(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kParameter;
}

bool IsAllReduce(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kAllReduce;
}

bool IsRemoteParameterLoad(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(inst);
}

bool IsNonReplicatedParameterLoad(
    const HloInstruction* inst, const ElementwiseClusterValidator& validator) {
  if (!IsRemoteParameterLoad(inst)) {
    return false;
  }
  if (!validator.IsValidInput(inst->operand(0))) {
    return false;
  }
  auto remote_load = Cast<HloRemoteParameterLoad>(inst);
  CHECK_EQ(remote_load->GetReplicationFactorCount(), 1);
  return remote_load->GetReplicationFactor(0) == 1;
}

bool IsRemoteParameterStore(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::RemoteParameterStore)(inst);
}

bool IsAllGather(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::AllGather)(inst);
}

// TODO(T42325): It's possible to remove this check and edge case all together
// if all broadcast will be sunk through elementwise ops.
bool IsImplicitOpWithAllScalarArguments(const HloInstruction* inst) {
  return IsPopOpsFusion(inst, "implicit_") &&
         !ShapeUtil::IsScalar(inst->shape()) &&
         absl::c_all_of(inst->operands(), [](const HloInstruction* op) {
           return ShapeUtil::IsScalar(op->shape());
         });
}

bool ValidClusterInput(const HloInstruction* inst,
                       const ElementwiseClusterValidator& validator) {
  return validator.IsValidInput(inst) || IsWideConstant(inst) ||
         IsAllReduce(inst) || IsReplicatedParameterLoad(inst) ||
         IsNonReplicatedParameterLoad(inst, validator);
}

}  // namespace

std::string UserPositions::ToString() const {
  return absl::StrCat("UserPositions: ", instruction->name(), ":",
                      absl::StrJoin(indices, ","));
}

ElementwiseClusterValidator::Inputs ElementwiseClusterValidator::GetValidInputs(
    const std::function<bool(int64)>& parameter_filter,
    const HloComputation* comp) {
  Inputs valid_inputs;
  for (const HloInstruction* inst : comp->MakeInstructionPostOrder()) {
    bool valid_input = false;
    if (IsParameter(inst) && parameter_filter(inst->parameter_number())) {
      valid_input = true;
    } else if (IsScalar(inst) && inst->IsElementwise()) {
      // Note that this also captures constants.
      valid_input = absl::c_all_of(
          inst->operands(), [&valid_inputs](const HloInstruction* operand) {
            return valid_inputs.contains(operand);
          });
    } else if (IsBroadcast(inst)) {
      valid_input =
          IsScalar(inst->operand(0)) && valid_inputs.contains(inst->operand(0));
    }

    if (valid_input) {
      valid_inputs.insert(inst);
    }
  }
  return valid_inputs;
}

ElementwiseCluster::ElementwiseCluster(HloInstruction* top) noexcept
    : top_(top), cluster_shape_(top->shape()) {
  Add(top);
}

Shape ElementwiseCluster::GetClusterShape(PrimitiveType type) const {
  return ShapeUtil::MakeShape(type, GetClusterDimensions());
}

bool ElementwiseCluster::In(HloInstruction* inst) const {
  return ContainsKey(insts_, inst);
}

bool ElementwiseCluster::AnyUserIn(HloInstruction* inst) const {
  return absl::c_any_of(inst->users(),
                        [this](HloInstruction* user) { return In(user); });
}

bool ElementwiseCluster::AllUsersIn(HloInstruction* inst) const {
  return absl::c_all_of(inst->users(),
                        [this](HloInstruction* user) { return In(user); });
}

void ElementwiseCluster::Add(HloInstruction* inst) {
  inputs_.erase(inst);
  insts_.insert(inst);
  for (auto op : inst->operands()) {
    if (!ContainsKey(insts_, op)) {
      inputs_.insert(op);
    }
  }
}

bool ElementwiseCluster::MaybeAdd(HloInstruction* inst) {
  if (!AnyUserIn(inst)) {
    return false;
  }
  Add(inst);
  return true;
}

bool ElementwiseCluster::CanMerge(const ElementwiseCluster& other) {
  // Allow to merge clusters if we use any of other cluster instruction
  bool can_merge = false;
  for (auto inst : insts_) {
    for (auto user : inst->users()) {
      if (other.In(user)) {
        return true;
      }
    }
  }
  return false;
}

void ElementwiseCluster::Merge(const ElementwiseCluster& other) {
  for (auto inst : other.insts_) {
    Add(inst);
  }
}
const HloInstruction* ElementwiseCluster::GetTop() const { return top_; }

HloComputation* ElementwiseCluster::GetComputation() const {
  return top_->parent();
}

bool ElementwiseCluster::CanCluster(
    const HloInstruction* inst, bool allow_inputs,
    const absl::flat_hash_set<const HloComputation*>& elementwise_comps,
    const ElementwiseClusterValidator& validator) {
  if (allow_inputs && ValidClusterInput(inst, validator)) {
    return true;
  }

  if (inst->HasSideEffect()) {
    return false;
  }

  if (IsScalar(inst)) {
    return false;
  }

  // This is explicit because constants are reported as elementwise.
  // Constant scalars are allowed as inputs though.
  if (inst->IsConstant()) {
    return false;
  }

  if (IsImplicitOpWithAllScalarArguments(inst)) {
    return false;
  }

  switch (inst->opcode()) {
    case HloOpcode::kCustomCall:
      return IsPopOpsElementwise(inst);
    case HloOpcode::kFusion:
      return !IsWideConstant(inst) &&
             elementwise_comps.contains(inst->fused_instructions_computation());
    default:
      return inst->IsElementwise();
  }
}

StatusOr<std::vector<ElementwiseCluster>> ElementwiseCluster::GetClustersIn(
    HloInstruction* const resource_update,
    const absl::flat_hash_set<const HloComputation*>& elementwise_comps,
    ElementwiseClusterValidator& validator) {
  HloComputation* resource_update_comp = resource_update->to_apply();
  auto offload_variables =
      GetResourceUpdatePartitionOffloadedVariables(resource_update);

  std::vector<ElementwiseCluster> clusters;

  // Going back post-order growing a tree of elementwise instruction.
  // For each new elementwise instruction, check if any of its users are already
  // in the cluster. If it's true, add it to the cluster.
  //
  // Example:
  //
  // ROOT r = tuple(fusion.1, fusion.2)
  // Ignoring, not elementwise
  //
  // fusion.1 = fusion(add.1, const.1)
  // Check that fused computation is elementwise, creating cluster #1 with top
  // at fusion.1. Add inputs [add.1, const.1]
  //
  // add.1 = add(arg0, arg1)
  // add.1 is among cluster #1 inputs, adding to the cluster, removing input of
  // add.1, inputs are [arg0, arg1], result inputs: [arg0, arg1, const.1]
  //
  // fusion.2 = fusion(add.2, const.1)
  // Not a use of existing cluster, create cluster #2, adding inputs [add.2,
  // const.1]
  //
  // add.2 = add(arg0, broadcast.1)
  // Used in cluster #2, adding to cluster, removing add.2 from inputs, inputs
  // are [arg0, const.1, broadcast.1]
  //
  // broadcast.1 = broadcast(const.2)
  // Used in cluster #2, removing from inputs, inputs are [arg0, const.1]

  auto comp_insts = resource_update_comp->MakeInstructionPostOrder();
  absl::c_reverse(comp_insts);
  for (auto inst : comp_insts) {
    bool can_cluster =
        CanCluster(inst, /*allow_inputs=*/false, elementwise_comps, validator);
    if (can_cluster) {
      VLOG(2) << "Found elementwise instruction: " << inst->ToString();
      bool added = false;
      for (auto& cluster : clusters) {
        if (cluster.MaybeAdd(inst)) {
          VLOG(2) << "Added to cluster with top "
                  << cluster.GetTop()->ToString();
          added = true;
          break;
        }
      }
      if (!added) {
        VLOG(2) << "Creating cluster with top " << inst->ToString();
        clusters.emplace_back(inst);
      }
    }
  }

  bool clusters_merged;
  do {
    VLOG(2) << "Merging clusters...";
    clusters_merged = false;
    for (auto i = clusters.begin(); !clusters_merged && i != clusters.end();
         ++i) {
      ElementwiseCluster& a = *i;
      for (auto j = std::next(i); j != clusters.end(); ++j) {
        ElementwiseCluster& b = *j;

        if (a.CanMerge(b)) {
          VLOG(2) << "Cluster " << b.GetTop()->name()
                  << " could be merged in cluster " << a.GetTop()->name();
          a.Merge(b);
          clusters.erase(j);
          clusters_merged = true;
          break;
        } else if (b.CanMerge(a)) {
          VLOG(2) << "Cluster " << a.GetTop()->name()
                  << " could be merged in cluster " << b.GetTop()->name();
          b.Merge(a);
          clusters.erase(i);
          clusters_merged = true;
          break;
        }
      }
    }
  } while (clusters_merged);

  absl::flat_hash_set<HloInstruction*> seen_insts;
  for (auto it = clusters.begin(); it != clusters.end();) {
    auto& cluster = *it;
    bool valid = cluster.Finalize(validator, offload_variables);

    if (valid) {
      // Make sure that non of the outputs overlap with previously seen
      // instructions.
      valid &= absl::c_all_of(cluster.GetOutputs(),
                              [&seen_insts](const HloInstruction* inst) {
                                return !seen_insts.contains(inst);
                              });
    }

    if (valid) {
      absl::c_copy(cluster.GetPostOrder(),
                   std::inserter(seen_insts, seen_insts.begin()));

      VLOG(2) << "Found cluster suitable for replication (all inputs valid):";
      XLA_VLOG_LINES(2, cluster.ToString());
      ++it;
    } else {
      VLOG(2) << "Invalid cluster:";
      XLA_VLOG_LINES(2, cluster.Dump());
      it = clusters.erase(it);
    }
  }
  return clusters;
}

ElementwiseClusterClass ElementwiseCluster::Classify(
    const ElementwiseClusterValidator& validator) const {
  if (IsScalar(top_)) {
    return ElementwiseClusterClass::Invalid;
  }

  // Check all inputs are valid.
  for (auto input : inputs_) {
    if (!ValidClusterInput(input, validator)) {
      VLOG(2) << "Invalid cluster input: " << input->ToString();
      return ElementwiseClusterClass::Invalid;
    }
  }

  const int64 num_replicated_parameter_load =
      absl::c_count_if(inputs_, IsReplicatedParameterLoad);
  const int64 num_non_replicated_parameter_load =
      absl::c_count_if(inputs_, [&validator](const HloInstruction* input) {
        return IsNonReplicatedParameterLoad(input, validator);
      });
  VLOG(2) << "Number of replicated parameter load inputs: "
          << num_replicated_parameter_load;
  VLOG(2) << "Number of non replicated parameter load inputs: "
          << num_non_replicated_parameter_load;

  if (num_replicated_parameter_load && num_non_replicated_parameter_load) {
    VLOG(2) << "Found a cluster with both replicated and non replicated "
               "parameter loads which is currently unsupported.";
    return ElementwiseClusterClass::Invalid;
  } else if (num_replicated_parameter_load &&
             !num_non_replicated_parameter_load) {
    return ElementwiseClusterClass::Partitioned;
  } else if (!num_replicated_parameter_load &&
             num_non_replicated_parameter_load) {
    return ElementwiseClusterClass::NonPartitioned;
  } else {
    VLOG(2) << "No parameter load inputs found.";
    return ElementwiseClusterClass::Invalid;
  }
}

bool ElementwiseCluster::Finalize(const ElementwiseClusterValidator& validator,
                                  ThreeState partition_offload_variables) {
  CHECK(!finalized_);

  auto cluster_class = Classify(validator);
  if (cluster_class == ElementwiseClusterClass::Invalid) {
    return false;
  }

  if (partition_offload_variables == THREESTATE_OFF &&
      cluster_class == ElementwiseClusterClass::Partitioned) {
    VLOG(2) << "Resource update partition offload is turned off, cannot "
               "offload cluster.";
    return false;
  }

  // Populate all the remaining fields and create a post order traversal.
  inputs_vec_.reserve(inputs_.size());
  absl::c_copy(inputs_, std::back_inserter(inputs_vec_));

  std::vector<HloInstruction*> to_visit;
  absl::flat_hash_set<HloInstruction*> visited;

  auto was_visited = [&visited](const HloInstruction* inst) -> bool {
    return visited.contains(inst);
  };

  auto add_users_for_processing = [&to_visit, &was_visited,
                                   this](HloInstruction* inst) {
    // Find any users in the cluster ready for processing.
    for (auto user : inst->users()) {
      if (!ContainsKey(insts_, user)) {
        continue;
      }
      // Instruction is ready to process when all operands have been
      // visited.
      const bool ready_to_process =
          absl::c_all_of(user->operands(), was_visited);
      if (ready_to_process) {
        to_visit.push_back(user);
      }
    }
  };

  auto add_outputs = [this](HloInstruction* inst) {
    for (auto user : inst->users()) {
      if (!ContainsKey(insts_, user)) {
        auto indices = user->OperandIndices(inst);
        outputs_to_users_[inst].push_back(
            UserPositions{user, {indices.begin(), indices.end()}});
      }
    }
  };

  for (HloInstruction* input : inputs_vec_) {
    visited.insert(input);
    add_users_for_processing(input);
  }

  while (!to_visit.empty()) {
    HloInstruction* inst = to_visit.back();
    to_visit.pop_back();
    if (was_visited(inst)) {
      continue;
    }
    post_order_.push_back(inst);
    visited.insert(inst);
    add_users_for_processing(inst);
    add_outputs(inst);
  }
  CHECK(absl::c_all_of(insts_, was_visited))
      << "Invalid elementwise cluster produced.";
  CHECK_EQ(insts_.size(), post_order_.size());

  outputs_.reserve(outputs_to_users_.size());
  for (auto& pair : outputs_to_users_) {
    outputs_.push_back(pair.first);
  }

  cluster_size_ = ShapeUtil::ElementsIn(cluster_shape_);
  cluster_dimensions_ = {cluster_shape_.dimensions().begin(),
                         cluster_shape_.dimensions().end()};

  // Only perform replica partitioning if there is a replicated parameter load.
  is_replica_partitioned_ =
      cluster_class == ElementwiseClusterClass::Partitioned;
  if (is_replica_partitioned_) {
    // Get all parameter loads.
    std::vector<HloInstruction*> parameter_loads;
    absl::c_copy_if(inputs_vec_, std::back_inserter(parameter_loads),
                    IsReplicatedParameterLoad);

    // Get dimensions for each load and make sure there is only one unique set.
    absl::flat_hash_set<std::vector<int64>> all_shard_dimensions;
    absl::c_transform(
        parameter_loads,
        std::inserter(all_shard_dimensions, all_shard_dimensions.begin()),
        [](const HloInstruction* inst) -> std::vector<int64> {
          auto dimensions = inst->operand(0)->shape().dimensions();
          return {dimensions.begin(), dimensions.end()};
        });

    if (all_shard_dimensions.size() != 1) {
      VLOG(2) << "Multiple shard sizes detected.";
      return false;
    }
    shard_dimensions_ = *std::begin(all_shard_dimensions);
    CHECK_EQ(shard_dimensions_.size(), 1);
    shard_size_ = shard_dimensions_[0];

    // Get sizes for the all gathers inside of the parameter load fusions.
    absl::flat_hash_set<int64> all_gather_sizes;
    absl::c_transform(
        parameter_loads,
        std::inserter(all_gather_sizes, all_gather_sizes.begin()),
        [](const HloInstruction* inst) {
          return ShapeUtil::ElementsIn(
              GetReplicatedParameterLoadFusionAllGatherShape(inst));
        });
    if (all_gather_sizes.size() == 1) {
      aligned_cluster_size_ = *std::begin(all_gather_sizes);
    } else {
      VLOG(2) << "Multiple aligned cluster sizes found.";
      return false;
    }

    if (shard_size_ * (aligned_cluster_size_ / shard_size_) !=
        aligned_cluster_size_) {
      VLOG(2) << "Cluster shape and replica shape don't match " << shard_size_
              << " vs " << cluster_size_ << "(" << aligned_cluster_size_ << ")";
      return false;
    }

  } else {
    // This is a non replica partitioned cluster.
    shard_dimensions_ = {cluster_size_};
    shard_size_ = cluster_size_;
    aligned_cluster_size_ = cluster_size_;
  }

  finalized_ = true;
  return true;
}

const std::vector<HloInstruction*>& ElementwiseCluster::GetInputs() const {
  CHECK(finalized_);
  return inputs_vec_;
}

const std::vector<HloInstruction*>& ElementwiseCluster::GetPostOrder() const {
  CHECK(finalized_);
  return post_order_;
}

const std::vector<HloInstruction*>& ElementwiseCluster::GetOutputs() const {
  CHECK(finalized_);
  return outputs_;
}

const std::vector<UserPositions>& ElementwiseCluster::GetUsersForOutput(
    HloInstruction* inst) const {
  CHECK(finalized_);
  return outputs_to_users_.at(inst);
}

const std::vector<int64>& ElementwiseCluster::GetClusterDimensions() const {
  CHECK(finalized_);
  return cluster_dimensions_;
}

const std::vector<int64>& ElementwiseCluster::GetShardDimensions() const {
  CHECK(finalized_);
  return shard_dimensions_;
}

int64 ElementwiseCluster::GetClusterSize() const {
  CHECK(finalized_);
  return cluster_size_;
}

int64 ElementwiseCluster::GetAlignedClusterSize() const {
  CHECK(finalized_);
  return aligned_cluster_size_;
}

int64 ElementwiseCluster::GetShardSize() const {
  CHECK(finalized_);
  return shard_size_;
}

bool ElementwiseCluster::IsReplicaPartitioned() const {
  CHECK(finalized_);
  return is_replica_partitioned_;
}

std::string ElementwiseCluster::Dump() const {
  std::stringstream ss;
  ss << "Cluster dump:\n";
  ss << "top: " << top_->ToString() << ", " << inputs_.size() << " input(s).\n";
  for (auto inst : inputs_) {
    ss << "Input: " << inst->ToString() << "\n";
  }
  ss << "\n";
  for (auto inst : insts_) {
    ss << "Instruction: " << inst->ToString() << "\n";
  }
  return ss.str();
}

std::string ElementwiseCluster::ToString() const {
  CHECK(finalized_);
  std::stringstream ss;
  ss << "Cluster:\n";
  ss << "Cluster shape: " << absl::StrJoin(GetClusterDimensions(), ",")
     << ", total elements " << GetClusterSize()
     << ", aligned cluster size: " << GetAlignedClusterSize() << "\n";
  ss << "Shard shape: (" << absl::StrJoin(GetShardDimensions(), ",")
     << "), total elements " << GetShardSize() << "\n";
  ss << "Is replica partitioned: " << IsReplicaPartitioned() << "\n";
  ss << "Inputs:\n";
  for (auto inst : inputs_vec_) {
    ss << "* " << inst->ToString() << "\n";
  }
  ss << "\n";
  ss << "Post order:\n";
  for (auto inst : post_order_) {
    ss << "* " << inst->ToString() << "\n";
  }
  ss << "\n";
  ss << "Outputs:\n";
  for (auto inst : outputs_) {
    ss << "* " << inst->ToString() << " used by:\n";
    for (auto user : outputs_to_users_.at(inst)) {
      ss << "  * " << user.ToString() << "\n";
    }
  }
  return ss.str();
}

}  // namespace poplarplugin
}  // namespace xla
