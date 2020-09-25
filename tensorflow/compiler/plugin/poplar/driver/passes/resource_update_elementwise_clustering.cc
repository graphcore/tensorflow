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

#include "tensorflow/compiler/plugin/poplar/driver/passes/resource_update_elementwise_clustering.h"

#include <list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/all_gather.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/replication_factor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/replication_index.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {
namespace {

bool IsParameter(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kParameter;
}

bool IsAllReduce(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kAllReduce;
}

bool IsReplicationIndex(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::ReplicationIndex)(inst);
}

bool IsReplicationNormalise(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::ReplicationNormalise)(inst);
}

bool IsRemoteParameterLoad(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(inst);
}

bool IsNonReplicatedParameterLoad(const HloInstruction* inst) {
  if (!IsRemoteParameterLoad(inst)) {
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

HloInstruction* GetReshapeAllGatherLoad(HloInstruction* inst,
                                        int64* aligned_size = nullptr) {
  if (inst->opcode() != HloOpcode::kReshape) {
    return nullptr;
  }

  auto all_gather = inst->mutable_operand(0);
  // Check for reshape(slice(all-gather) or slice(all-gather)
  HloInstruction* reshape = nullptr;
  if (all_gather->opcode() == HloOpcode::kSlice) {
    reshape = all_gather->mutable_operand(0);
    if (reshape->opcode() == HloOpcode::kReshape) {
      all_gather = reshape->mutable_operand(0);
    }
  }

  if (!IsAllGather(all_gather)) {
    return nullptr;
  }

  HloInstruction* remote_load;
  auto gte = all_gather->mutable_operand(0);
  if (gte->opcode() == HloOpcode::kGetTupleElement) {
    remote_load = gte->mutable_operand(0);
  } else {
    remote_load = gte;
  }

  if (IsRemoteParameterLoad(remote_load)) {
    if (aligned_size && reshape) {
      *aligned_size = ShapeUtil::ElementsIn(reshape->shape());
    }
    return remote_load;
  } else {
    return nullptr;
  }
}

// Extract remote-parameter-store(reshape(dynamic-slice(reshape(inst),
// replication-index))) pattern. Return a pair of (remote-parameter-store,
// reshape) instructions
std::pair<HloInstruction*, HloInstruction*> GetRemoteStoreForUser(
    HloInstruction* user) {
  if (user->opcode() != HloOpcode::kReshape || user->user_count() != 1) {
    return std::make_pair(nullptr, nullptr);
  }
  HloInstruction* dynamic_slice = user->users()[0];
  // Check if we have padding
  if (dynamic_slice->opcode() == HloOpcode::kPad &&
      dynamic_slice->user_count() == 1 &&
      IsConstantZero(dynamic_slice->operand(1))) {
    // Remove padding
    HloInstruction* padded_reshape = dynamic_slice->users()[0];
    if (padded_reshape->opcode() == HloOpcode::kReshape &&
        padded_reshape->user_count() == 1) {
      dynamic_slice = padded_reshape->users()[0];
    }
  }
  if (dynamic_slice->opcode() != HloOpcode::kDynamicSlice ||
      dynamic_slice->user_count() != 1) {
    return std::make_pair(nullptr, nullptr);
  }
  if (!IsReplicationIndex(dynamic_slice->operand(1))) {
    return std::make_pair(nullptr, nullptr);
  }
  HloInstruction* reshape = dynamic_slice->users()[0];
  if (reshape->opcode() != HloOpcode::kReshape || reshape->user_count() != 1) {
    return std::make_pair(nullptr, nullptr);
  }
  HloInstruction* store = reshape->users()[0];
  if (IsRemoteParameterStore(store)) {
    return std::make_pair(store, reshape);
  } else {
    return std::make_pair(nullptr, nullptr);
  }
}

StatusOr<bool> ReplaceRemoteStoreArgumentWith(HloInstruction* cluster_user,
                                              HloInstruction* inst,
                                              const Shape& shard_shape) {
  HloInstruction *store, *reshape;
  std::tie(store, reshape) = GetRemoteStoreForUser(cluster_user);
  if (!store) {
    return false;
  }
  if (reshape->shape() != shard_shape) {
    VLOG(2) << "RemoteParameterStore does not match shard shape, ignoring";
    return false;
  }

  TF_RETURN_IF_ERROR(reshape->ReplaceUseWithDifferentShape(store, inst));
  VLOG(2) << "RemoteParameterStore instruction: " << store->ToString();
  return true;
}

bool IsReshapeAllGatherLoad(HloInstruction* inst) {
  return GetReshapeAllGatherLoad(inst);
}

bool ValidClusterInput(HloInstruction* inst) {
  return inst->IsConstant() || IsScalar(inst) || IsParameter(inst) ||
         IsAllReduce(inst) || IsReshapeAllGatherLoad(inst) ||
         IsReplicationNormalise(inst) || IsNonReplicatedParameterLoad(inst);
}

bool CanCluster(HloInstruction* inst, bool allow_inputs) {
  if (allow_inputs && ValidClusterInput(inst)) {
    return true;
  }

  // This is explicit because scalars are reported as elementwise.
  // Scalars are allowed as inputs though.
  if (IsScalar(inst)) {
    return false;
  }

  switch (inst->opcode()) {
    case HloOpcode::kBroadcast:
      return ShapeUtil::IsScalar(inst->operand(0)->shape());
    case HloOpcode::kCustomCall:
      return IsPopOpsElementwise(inst);
    default:
      return inst->IsElementwise() && !inst->HasSideEffect();
  }
}

bool CanCluster(HloInstruction* inst, bool allow_inputs,
                const absl::flat_hash_set<HloComputation*>& elementwise_comps) {
  if (CanCluster(inst, allow_inputs)) {
    return true;
  }
  if (inst->opcode() == HloOpcode::kFusion) {
    return elementwise_comps.contains(inst->fused_instructions_computation());
  } else {
    return false;
  }
}

Status ChangeInstructionShape(HloInstruction* inst, const Shape& old_shape,
                              const Shape& new_shape) {
  if (IsScalar(inst) || inst->shape() == new_shape) {
    VLOG(2) << "Do not set shape for " << inst->ToString();
    return Status::OK();
  } else if (inst->shape() == old_shape) {
    VLOG(2) << "Reshaped instruction: " << inst->ToString();
    *inst->mutable_shape() = new_shape;
    return Status::OK();
  } else {
    return InternalErrorStrCat(
        "Unexpected shape (",
        inst->shape().ToString() + ") in instruction: ", inst->ToString(),
        ", expected ", old_shape.ToString(), " or ", new_shape.ToString());
  }
}

Status ChangeFusionShape(HloInstruction* fusion, const Shape& old_shape,
                         const Shape& new_shape) {
  VLOG(2) << "Changing fusion shape: " << fusion->ToString();
  CHECK(fusion->opcode() == HloOpcode::kFusion);
  HloComputation* fusion_comp = fusion->fused_instructions_computation();
  for (auto inst : fusion_comp->MakeInstructionPostOrder()) {
    TF_RETURN_IF_ERROR(ChangeInstructionShape(inst, old_shape, new_shape));
  }
  VLOG(2) << "new fusion computation: "
          << fusion->fused_instructions_computation()->ToString();

  *fusion->mutable_shape() = new_shape;
  VLOG(2) << "new fusion: " << fusion->ToString();
  return Status::OK();
}

Status RewriteClusterInput(
    const ElementwiseCluster& cluster, int64 aligned_cluster_size,
    const Shape& shard_shape, int64 replication_factor,
    HloInstruction* cluster_input,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>& input_slices) {
  HloComputation* cluster_comp = cluster.GetComputation();
  int64 cluster_size = ShapeUtil::ElementsIn(cluster.GetShape());
  int64 shard_size = ShapeUtil::ElementsIn(shard_shape);
  // If it's reshape(all-gather(reshape(remote-parameter-laod))), remove
  // all-gather.
  HloInstruction* remote_load = GetReshapeAllGatherLoad(cluster_input);
  if (remote_load && remote_load->shape() == shard_shape) {
    VLOG(2) << "Rewriting remote cluster input " << remote_load->ToString();
    const std::vector<HloInstruction*> insts = cluster.GetPostOrder();
    for (auto user : cluster_input->users()) {
      if (absl::c_find(insts, user) != insts.end()) {
        VLOG(2) << "Removing all-gather, use remote-parameter-load directly.";
        TF_RETURN_IF_ERROR(
            cluster_input->ReplaceUseWithDifferentShape(user, remote_load));
      }
    }
    return Status::OK();
  }

  // All other inputs have to be sliced with dynamic-slice(input,
  // replication-index()) Reuse dynamic-slice for input in case of multiple
  // users
  VLOG(2) << "Rewriting cluster input " << cluster_input->ToString();

  auto cluster_input_shape = cluster_input->shape();
  HloInstruction*& cluster_input_slice = input_slices[cluster_input];
  for (auto inst : cluster.GetPostOrder()) {
    if (inst->IsUserOf(cluster_input)) {
      if (!cluster_input_slice) {
        Shape all_shards_shape =
            ShapeUtil::MakeShape(cluster_input_shape.element_type(),
                                 {replication_factor, shard_size});

        HloInstruction* reshaped;
        if (cluster_size != aligned_cluster_size) {
          HloInstruction* zero_f =
              cluster_comp->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::Zero(cluster_input_shape.element_type())));

          // Flatten the incoming tensor
          Shape flat_shape = ShapeUtil::MakeShape(
              cluster_input_shape.element_type(),
              {ShapeUtil::ElementsIn(cluster_input_shape)});
          HloInstruction* flat = cluster_comp->AddInstruction(
              HloInstruction::CreateReshape(flat_shape, cluster_input));
          VLOG(2) << "reshape: " << flat->ToString();

          // Pad the tensor to be a multiple of `replication_factor`.
          Shape pad_shape = ShapeUtil::MakeShape(
              cluster_input_shape.element_type(), {aligned_cluster_size});

          PaddingConfig padding_config;
          std::size_t difference = ShapeUtil::ElementsIn(pad_shape) -
                                   ShapeUtil::ElementsIn(flat->shape());
          auto padding_config_dim = padding_config.add_dimensions();
          padding_config_dim->set_edge_padding_high(difference);
          padding_config_dim->set_edge_padding_low(0);
          padding_config_dim->set_interior_padding(0);

          HloInstruction* pad =
              cluster_comp->AddInstruction(HloInstruction::CreatePad(
                  pad_shape, flat, zero_f, padding_config));
          VLOG(2) << "pad: " << pad->ToString();
          reshaped = cluster_comp->AddInstruction(
              HloInstruction::CreateReshape(all_shards_shape, pad));
        } else {
          reshaped = cluster_comp->AddInstruction(
              HloInstruction::CreateReshape(all_shards_shape, cluster_input));
        }

        HloInstruction* replica_id =
            cluster_comp->AddInstruction(CreateReplicationIndex());

        HloInstruction* zero_i =
            cluster_comp->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::Zero(replica_id->shape().element_type())));

        // Slice off this replica's storage elements.
        Shape slice_shape = ShapeUtil::MakeShape(
            cluster_input_shape.element_type(), {1, shard_size});
        HloInstruction* slice =
            cluster_comp->AddInstruction(HloInstruction::CreateDynamicSlice(
                slice_shape, reshaped, {replica_id, zero_i}, {1, shard_size}));

        // Squeeze off the outermost dimension.
        Shape squeeze_shape = ShapeUtil::MakeShape(
            cluster_input_shape.element_type(), {shard_size});
        cluster_input_slice = cluster_comp->AddInstruction(
            HloInstruction::CreateReshape(squeeze_shape, slice));
        VLOG(2) << "Input slice: " << cluster_input_slice->ToString();
      } else {
        VLOG(2) << "Reusing input slice: " << cluster_input_slice->ToString();
      }
      TF_RETURN_IF_ERROR(cluster_input->ReplaceUseWithDifferentShape(
          inst, cluster_input_slice));
    }
  }
  return Status::OK();
}

Status RewriteClusterOutput(const ElementwiseCluster& cluster,
                            int64 aligned_cluster_size,
                            const Shape& shard_shape, int64 replication_factor,
                            HloInstruction* inst,
                            const std::vector<HloInstruction*>& outputs) {
  HloComputation* cluster_comp = cluster.GetComputation();
  int64 cluster_size = ShapeUtil::ElementsIn(cluster.GetShape());
  int64 shard_size = ShapeUtil::ElementsIn(shard_shape);

  HloInstruction* all_gather_reshaped = nullptr;
  // Replace all users outside the cluster with the result of all-gather
  for (auto user : outputs) {
    VLOG(2) << "Replace use of cluster instruction " << inst->ToString()
            << " in " << user->ToString();
    TF_ASSIGN_OR_RETURN(
        auto replaced_in_remote_store,
        ReplaceRemoteStoreArgumentWith(user, inst, shard_shape));
    if (replaced_in_remote_store) {
      VLOG(2) << "Replaced argument to RemoteParameterStore";
    } else if (user->shape() != shard_shape) {
      if (!all_gather_reshaped) {
        // Create all gather and replace usage
        auto all_gather_shape =
            ShapeUtil::MakeShape(cluster.GetShape().element_type(),
                                 {replication_factor, shard_size});
        auto all_gather = cluster_comp->AddInstruction(
            CreateAllGather({inst}, all_gather_shape));
        if (all_gather_shape != cluster.GetShape()) {
          if (cluster_size != aligned_cluster_size) {
            Shape flat_cluster_shape = ShapeUtil::MakeShape(
                cluster.GetShape().element_type(), {cluster_size});
            Shape aligned_cluster_shape = ShapeUtil::MakeShape(
                cluster.GetShape().element_type(), {aligned_cluster_size});
            HloInstruction* flat_all_gather =
                cluster_comp->AddInstruction(HloInstruction::CreateReshape(
                    aligned_cluster_shape, all_gather));
            HloInstruction* slice = cluster_comp->AddInstruction(
                HloInstruction::CreateSlice(flat_cluster_shape, flat_all_gather,
                                            {0}, {cluster_size}, {1}));
            VLOG(2) << "Slicing padding, slice: " << slice->ToString();
            all_gather_reshaped = cluster_comp->AddInstruction(
                HloInstruction::CreateReshape(cluster.GetShape(), slice));
          } else {
            all_gather_reshaped = cluster_comp->AddInstruction(
                HloInstruction::CreateReshape(cluster.GetShape(), all_gather));
          }
        } else {
          all_gather_reshaped = all_gather;
        }
        VLOG(2) << "All gather instuction: " << all_gather->name()
                << ", reshaped: " << all_gather_reshaped->name();
      }
      TF_RETURN_IF_ERROR(
          inst->ReplaceUseWithDifferentShape(user, all_gather_reshaped));
      VLOG(2) << "Replacement result " << user->ToString();
    }
  }
  return Status::OK();
}

StatusOr<bool> RewriteResourceUpdate(
    HloModule* module, HloInstruction* resource_update_inst,
    const absl::flat_hash_set<HloComputation*>& elementwise_comps,
    int64 replication_factor) {
  HloComputation* resource_update = resource_update_inst->to_apply();
  auto offload_variables =
      GetResourceUpdatePartitionOffloadedVariables(resource_update_inst);
  if (offload_variables == THREESTATE_OFF) {
    VLOG(2) << "Resource update partition offload is turned off, exiting.";
    return false;
  }

  std::list<ElementwiseCluster> clusters;

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

  auto resource_update_insts = resource_update->MakeInstructionPostOrder();
  absl::c_reverse(resource_update_insts);
  for (auto inst : resource_update_insts) {
    bool can_cluster =
        CanCluster(inst, /*allow_inputs=*/false, elementwise_comps);
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

  for (auto it = clusters.begin(); it != clusters.end();) {
    auto& cluster = *it;
    if (cluster.Finalize()) {
      VLOG(2) << "Found cluster suitable for replication (all inputs valid):";
      XLA_VLOG_LINES(2, cluster.ToString());
      ++it;
    } else {
      VLOG(2) << "Invalid cluster:";
      XLA_VLOG_LINES(2, cluster.Dump());
      it = clusters.erase(it);
    }
  }

  if (clusters.empty()) {
    VLOG(2) << "No clusters found.";
    return false;
  }

  absl::flat_hash_map<HloInstruction*, HloInstruction*> input_slices;
  bool changed;

  VLOG(2) << "Clustering with factor " << replication_factor;
  for (auto& cluster : clusters) {
    VLOG(2) << "Rewriting cluster with top in " << cluster.GetTop()->ToString()
            << ", " << cluster.GetPostOrder().size() << " instructions...";
    HloComputation* cluster_comp = cluster.GetComputation();
    int64 cluster_size = ShapeUtil::ElementsIn(cluster.GetShape());
    int64 aligned_cluster_size = cluster_size;

    // Going through remote inputs and determine shard size from
    // remote-parameter-load instruction. Also check that all shapes match.
    absl::optional<Shape> shard_shape_opt;
    for (auto cluster_input : cluster.GetInputs()) {
      auto remote_load =
          GetReshapeAllGatherLoad(cluster_input, &aligned_cluster_size);
      if (!remote_load) {
        continue;
      }
      auto shard_size = ShapeUtil::ElementsIn(remote_load->shape());
      if (!shard_shape_opt) {
        shard_shape_opt = remote_load->shape();
      } else if (remote_load->shape() != *shard_shape_opt) {
        return InternalErrorStrCat(
            "Cluster input shapes mismatch: ", remote_load->shape().ToString(),
            " vs ", shard_shape_opt->ToString());
      }
    }

    if (!shard_shape_opt) {
      VLOG(2) << "No remote buffers in cluster.";
      continue;
    }

    const Shape& shard_shape = *shard_shape_opt;
    auto shard_size = ShapeUtil::ElementsIn(shard_shape);

    VLOG(2) << "Cluster shard shape is " << shard_shape.ToString()
            << ", cluster shape is " << cluster.GetShape().ToString();

    if (shard_size * replication_factor != aligned_cluster_size) {
      VLOG(2) << "Cluster shape and replica shape don't match " << shard_size
              << " vs " << cluster_size << "(" << aligned_cluster_size << ")";
      continue;
    }

    // Change the shape of the cluster.
    for (auto inst : cluster.GetPostOrder()) {
      if (inst->opcode() == HloOpcode::kFusion) {
        TF_RETURN_IF_ERROR(
            ChangeFusionShape(inst, cluster.GetShape(), shard_shape));
      } else {
        TF_RETURN_IF_ERROR(
            ChangeInstructionShape(inst, cluster.GetShape(), shard_shape));
      }
    }

    // For each input:
    // Replace all-gather(remote-parameter-load)) with remote-parameter-load()
    // Replace other inputs with dynamic-slice(input, replication-index)
    for (auto cluster_input : cluster.GetInputs()) {
      if (IsScalar(cluster_input)) {
        VLOG(2) << "Ignoring scalar: " << cluster_input->ToString();
        continue;
      }

      TF_RETURN_IF_ERROR(RewriteClusterInput(cluster, aligned_cluster_size,
                                             shard_shape, replication_factor,
                                             cluster_input, input_slices));
    }

    // Replacing outputs:
    // For each instruction of the cluster, check its users.
    // If its user is outside of cluster, do all-gather/reshape
    // If its user is store(shape(slice(shape(cluster)))), remove reshape
    // and slice.
    for (auto cluster_output : cluster.GetOutputs()) {
      TF_RETURN_IF_ERROR(RewriteClusterOutput(
          cluster, aligned_cluster_size, shard_shape, replication_factor,
          cluster_output, cluster.GetUsersForOutput(cluster_output)));
    }
    changed = true;
  }
  return changed;
}
}  // namespace

ElementwiseCluster::ElementwiseCluster(HloInstruction* top) noexcept
    : top_(top), shape_(top->shape()) {
  Add(top);
}

bool ElementwiseCluster::In(HloInstruction* inst) const {
  return ContainsKey(insts_, inst);
}

bool ElementwiseCluster::AnyUserIn(HloInstruction* inst) const {
  for (auto user : inst->users()) {
    if (ContainsKey(insts_, user)) {
      return true;
    }
  }
  return false;
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

const Shape& ElementwiseCluster::GetShape() const { return shape_; }

bool ElementwiseCluster::Finalize() {
  CHECK(!finalized_);

  if (IsScalar(top_)) {
    return false;
  }

  // Check all inputs are valid.
  if (!absl::c_all_of(inputs_, ValidClusterInput)) {
    return false;
  }

  // Check at least one input is remote.
  if (!absl::c_any_of(inputs_, IsReshapeAllGatherLoad)) {
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
        outputs_to_users_[inst].push_back(user);
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

const std::vector<HloInstruction*>& ElementwiseCluster::GetUsersForOutput(
    HloInstruction* inst) const {
  CHECK(finalized_);
  return outputs_to_users_.at(inst);
}

std::string ElementwiseCluster::Dump() const {
  std::stringstream ss;
  ss << "Cluster dump:\n";
  ss << "top: " << top_->ToString() << ", " << inputs_.size() << " input(s).";
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
  ss << "Inputs:\n";
  for (auto inst : inputs_vec_) {
    ss << "* " << inst->ToString() << "\n";
  }
  ss << "\n";
  ss << "Post order:\n";
  for (auto inst : post_order_) {
    ss << "* " << inst->ToString() << "\n";
  }
  ss << "Outputs:\n";
  for (auto inst : outputs_) {
    ss << "* " << inst->ToString() << " used by:\n";
    for (auto user : outputs_to_users_.at(inst)) {
      ss << "  * " << user->ToString() << "\n";
    }
  }
  return ss.str();
}

StatusOr<bool> ResourceUpdateElementwiseClustering::Run(HloModule* module) {
  if (replication_factor_ <= 1) {
    VLOG(2) << "Skipping clustering, no replicas.";
    return false;
  }
  VLOG(2) << "Before the ElementwiseClustering:";
  XLA_VLOG_LINES(2, module->ToString());

  // This is primarily for the fusions, but could be useful for other
  // computations as well. Go through all computations and populate the
  // elementwise set. Elementwise computation defined as a set of instructions
  // which are either
  // - valid cluster input (constant, parameter, reduce-all, etc)
  // - elementwise instruction
  // - fusion uses elementwise computation from this set.
  // Also collect resource update computations.

  std::list<HloInstruction*> resource_updates;
  absl::flat_hash_set<HloComputation*> elementwise_comps;

  for (auto comp : module->MakeComputationPostOrder()) {
    bool elementwise = true;
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (IsResourceUpdate(inst)) {
        resource_updates.push_back(inst);
      }
      if (!CanCluster(inst, /*allow_inputs=*/true, elementwise_comps)) {
        VLOG(2) << "Computation contains instruction we can't replicate: "
                << inst->ToString();
        elementwise = false;
      }
    }
    if (elementwise) {
      VLOG(2) << "Found elementwise computation " << comp->name();
      elementwise_comps.insert(comp);
    }
  }

  if (resource_updates.empty()) {
    VLOG(2) << "No resource updates found, exiting.";
    return false;
  }

  bool module_changed = false;
  for (auto resource_update : resource_updates) {
    TF_ASSIGN_OR_RETURN(
        auto changed,
        RewriteResourceUpdate(module, resource_update, elementwise_comps,
                              replication_factor_));
    if (changed) {
      module_changed = true;
    }
  }

  if (module_changed) {
    TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());
    VLOG(2) << "After the ElementwiseClustering:";
    XLA_VLOG_LINES(2, module->ToString());
  }

  return module_changed;
}

}  // namespace poplarplugin
}  // namespace xla
