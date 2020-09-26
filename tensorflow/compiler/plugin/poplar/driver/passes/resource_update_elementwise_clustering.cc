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

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/all_gather.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/replication_factor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/replication_index.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
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

bool IsValidParameterInput(
    const HloInstruction* inst,
    const absl::flat_hash_set<int64>& allowed_parameter_indices) {
  return IsParameter(inst) &&
         allowed_parameter_indices.contains(inst->parameter_number());
}

bool IsAllReduce(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kAllReduce;
}

bool IsReplicationIndex(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::ReplicationIndex)(inst);
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

bool ValidClusterInput(
    const HloInstruction* inst,
    const absl::flat_hash_set<int64>& allowed_parameter_indices) {
  // A valid cluster input is one which is guaranteed to be identical across all
  // replicas.
  return IsScalarConstant(inst) ||
         IsValidParameterInput(inst, allowed_parameter_indices) ||
         IsAllReduce(inst) || IsReplicatedParameterLoad(inst) ||
         IsNonReplicatedParameterLoad(inst);
}

bool CanCluster(
    const HloInstruction* inst, bool allow_inputs,
    const absl::flat_hash_set<const HloComputation*>& elementwise_comps,
    const absl::flat_hash_set<int64>& allowed_parameter_indices) {
  if (allow_inputs && ValidClusterInput(inst, allowed_parameter_indices)) {
    return true;
  }

  if (inst->HasSideEffect()) {
    return false;
  }

  // This is explicit because constants are reported as elementwise.
  // Constant scalars are allowed as inputs though.
  if (inst->IsConstant()) {
    return false;
  }

  switch (inst->opcode()) {
    case HloOpcode::kBroadcast:
      return IsScalar(inst->operand(0));
    case HloOpcode::kCustomCall:
      return IsPopOpsElementwise(inst);
    case HloOpcode::kFusion:
      return elementwise_comps.contains(inst->fused_instructions_computation());
    default:
      return inst->IsElementwise();
  }
}

StatusOr<bool> ReplaceRemoteStoreArgumentWith(const ElementwiseCluster& cluster,
                                              HloInstruction* cluster_user,
                                              HloInstruction* inst) {
  if (!IsReplicatedParameterStore(cluster_user)) {
    return false;
  }
  HloInstruction* store = cluster_user->users()[0];
  HloInstruction* store_input = store->mutable_operand(1);
  if (store_input->shape().dimensions() != cluster.GetShardDimensions()) {
    VLOG(2) << "RemoteParameterStore does not match shard shape, ignoring";
    return false;
  }

  TF_RETURN_IF_ERROR(store_input->ReplaceUseWithDifferentShape(store, inst));
  VLOG(2) << "RemoteParameterStore instruction: " << store->ToString();
  return true;
}

Status ChangeInstructionShape(const ElementwiseCluster& cluster,
                              HloInstruction* inst) {
  if (IsScalar(inst) ||
      inst->shape().dimensions() == cluster.GetShardDimensions()) {
    VLOG(2) << "Do not set shape for " << inst->ToString();
    return Status::OK();
  } else if (inst->shape().dimensions() == cluster.GetClusterDimensions()) {
    VLOG(2) << "Reshaped instruction: " << inst->ToString();
    *inst->mutable_shape() = ShapeUtil::MakeShape(inst->shape().element_type(),
                                                  cluster.GetShardDimensions());
    return Status::OK();
  } else {
    return InternalErrorStrCat(
        "Unexpected shape (",
        inst->shape().ToString() + ") in instruction: ", inst->ToString());
  }
}

Status ChangeFusionShape(const ElementwiseCluster& cluster,
                         HloInstruction* fusion) {
  VLOG(2) << "Changing fusion shape: " << fusion->ToString();
  CHECK(fusion->opcode() == HloOpcode::kFusion);
  HloComputation* fusion_comp = fusion->fused_instructions_computation();
  for (auto inst : fusion_comp->MakeInstructionPostOrder()) {
    TF_RETURN_IF_ERROR(ChangeInstructionShape(cluster, inst));
  }
  VLOG(2) << "new fusion computation: "
          << fusion->fused_instructions_computation()->ToString();

  TF_RETURN_IF_ERROR(ChangeInstructionShape(cluster, fusion));
  VLOG(2) << "new fusion: " << fusion->ToString();
  return Status::OK();
}

Status RewriteClusterInput(const ElementwiseCluster& cluster,
                           int64 replication_factor,
                           HloInstruction* cluster_input) {
  HloComputation* cluster_comp = cluster.GetComputation();
  // If it's reshape(all-gather(reshape(remote-parameter-laod))), remove
  // all-gather.
  if (IsReplicatedParameterLoad(cluster_input)) {
    HloInstruction* remote_load = cluster_input->mutable_operand(0);
    if (remote_load->shape().dimensions() == cluster.GetShardDimensions()) {
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
  }

  // All other inputs have to be sliced with dynamic-slice(input,
  // replication-index()) Reuse dynamic-slice for input in case of multiple
  // users
  VLOG(2) << "Rewriting cluster input " << cluster_input->ToString();

  auto cluster_input_shape = cluster_input->shape();
  HloInstruction* cluster_input_slice = nullptr;
  for (auto inst : cluster.GetPostOrder()) {
    if (inst->IsUserOf(cluster_input)) {
      if (!cluster_input_slice) {
        Shape all_shards_shape =
            ShapeUtil::MakeShape(cluster_input_shape.element_type(),
                                 {replication_factor, cluster.GetShardSize()});

        HloInstruction* reshaped;
        if (cluster.GetClusterSize() != cluster.GetAlignedClusterSize()) {
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
          Shape pad_shape =
              ShapeUtil::MakeShape(cluster_input_shape.element_type(),
                                   {cluster.GetAlignedClusterSize()});

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
            cluster_input_shape.element_type(), {1, cluster.GetShardSize()});
        HloInstruction* slice =
            cluster_comp->AddInstruction(HloInstruction::CreateDynamicSlice(
                slice_shape, reshaped, {replica_id, zero_i},
                {1, cluster.GetShardSize()}));

        // Squeeze off the outermost dimension.
        Shape squeeze_shape = ShapeUtil::MakeShape(
            cluster_input_shape.element_type(), {cluster.GetShardSize()});
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
                            int64 replication_factor, HloInstruction* inst,
                            const std::vector<HloInstruction*>& outputs) {
  HloComputation* cluster_comp = cluster.GetComputation();

  // Replace all users outside the cluster with the result of all-gather
  HloInstruction* all_gather_reshaped = nullptr;
  for (auto user : outputs) {
    VLOG(2) << "Replace use of cluster instruction " << inst->ToString()
            << " in " << user->ToString();
    TF_ASSIGN_OR_RETURN(auto replaced_in_remote_store,
                        ReplaceRemoteStoreArgumentWith(cluster, user, inst));
    if (replaced_in_remote_store) {
      VLOG(2) << "Replaced argument to RemoteParameterStore";
    } else {
      if (!all_gather_reshaped) {
        auto inst_element_type = inst->shape().element_type();
        auto output_shape = ShapeUtil::MakeShape(
            inst_element_type, cluster.GetClusterDimensions());

        // Create all gather and replace usage.
        auto all_gather_shape = ShapeUtil::MakeShape(
            inst_element_type, {replication_factor, cluster.GetShardSize()});
        auto all_gather = cluster_comp->AddInstruction(
            CreateAllGather({inst}, all_gather_shape));
        if (all_gather_shape != output_shape) {
          if (cluster.GetClusterSize() != cluster.GetAlignedClusterSize()) {
            Shape flat_cluster_shape = ShapeUtil::MakeShape(
                inst_element_type, {cluster.GetClusterSize()});
            Shape aligned_cluster_shape = ShapeUtil::MakeShape(
                inst_element_type, {cluster.GetAlignedClusterSize()});
            HloInstruction* flat_all_gather =
                cluster_comp->AddInstruction(HloInstruction::CreateReshape(
                    aligned_cluster_shape, all_gather));
            HloInstruction* slice =
                cluster_comp->AddInstruction(HloInstruction::CreateSlice(
                    flat_cluster_shape, flat_all_gather, {0},
                    {cluster.GetClusterSize()}, {1}));
            VLOG(2) << "Slicing padding, slice: " << slice->ToString();
            all_gather_reshaped = cluster_comp->AddInstruction(
                HloInstruction::CreateReshape(output_shape, slice));
          } else {
            all_gather_reshaped = cluster_comp->AddInstruction(
                HloInstruction::CreateReshape(output_shape, all_gather));
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

StatusOr<bool> RewriteCall(
    HloModule* module, HloInstruction* call,
    const absl::flat_hash_set<const HloComputation*>& elementwise_comps,
    uint32 replication_factor) {
  TF_ASSIGN_OR_RETURN(std::vector<ElementwiseCluster> clusters,
                      ResourceUpdateElementwiseClustering::GetClustersIn(
                          call, elementwise_comps));

  if (clusters.empty()) {
    VLOG(2) << "No clusters found.";
    return false;
  }

  bool changed = false;
  for (auto& cluster : clusters) {
    TF_ASSIGN_OR_RETURN(bool outlined,
                        ResourceUpdateElementwiseClustering::OutlineCluster(
                            cluster, replication_factor));
    changed |= outlined;
  }
  return changed;
}
}  // namespace

ElementwiseCluster::ElementwiseCluster(HloInstruction* top) noexcept
    : top_(top), cluster_shape_(top->shape()) {
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

bool ElementwiseCluster::Finalize(
    const absl::flat_hash_set<int64>& allowed_parameter_indices_inputs) {
  CHECK(!finalized_);

  if (IsScalar(top_)) {
    return false;
  }

  // Check all inputs are valid.
  if (!absl::c_all_of(inputs_, [&allowed_parameter_indices_inputs](
                                   const HloInstruction* inst) {
        return ValidClusterInput(inst, allowed_parameter_indices_inputs);
      })) {
    return false;
  }

  // Check at least one input is remote.
  if (!absl::c_any_of(inputs_, IsReplicatedParameterLoad)) {
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

  cluster_size_ = ShapeUtil::ElementsIn(cluster_shape_);
  cluster_dimensions_ = cluster_shape_.dimensions();

  // Get all parameter loads.
  std::vector<HloInstruction*> parameter_loads;
  absl::c_copy_if(inputs_vec_, std::back_inserter(parameter_loads),
                  IsReplicatedParameterLoad);

  // Get dimensions for each load and make sure there is only one unique set.
  absl::flat_hash_set<std::vector<int64>> all_shard_dimensions;
  absl::c_transform(
      parameter_loads,
      std::inserter(all_shard_dimensions, all_shard_dimensions.begin()),
      [](const HloInstruction* inst) {
        return inst->operand(0)->shape().dimensions();
      });

  if (all_shard_dimensions.size() != 1) {
    VLOG(2) << "Multiple shard sizes detected.";
    return false;
  }
  shard_dimensions_ = *std::begin(all_shard_dimensions);
  shard_size_ =
      absl::c_accumulate(shard_dimensions_, 1LL, std::multiplies<int64>());

  // Get sizes for the all gathers inside of the parameter load fusions.
  absl::flat_hash_set<int64> all_gather_sizes;
  absl::c_transform(parameter_loads,
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
  ss << "Cluster shape: " << absl::StrJoin(GetClusterDimensions(), ",")
     << ", total elements " << GetClusterSize()
     << ", aligned cluster size: " << GetAlignedClusterSize() << "\n";
  ss << "Shard shape: (" << absl::StrJoin(GetShardDimensions(), ",")
     << "), total elements " << GetShardSize() << "\n";
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
      ss << "  * " << user->ToString() << "\n";
    }
  }
  return ss.str();
}

absl::flat_hash_set<const HloComputation*>
ResourceUpdateElementwiseClustering::GetElementwiseClusterableComputations(
    const HloModule* module) {
  // This is primarily for the fusions, but could be useful for other
  // computations as well. Go through all computations and populate the
  // elementwise set. Elementwise computation defined as a set of instructions
  // which are either
  // - valid cluster input (constant, parameter, reduce-all, etc)
  // - elementwise instruction
  // - fusion uses elementwise computation from this set.
  absl::flat_hash_set<const HloComputation*> elementwise_comps;
  for (auto comp : module->computations()) {
    // In fusion computations all parameters are allowed as parameter inputs.
    absl::flat_hash_set<int64> allowed_parameter_indices;
    for (int64 i = 0; i != comp->num_parameters(); ++i) {
      allowed_parameter_indices.insert(i);
    }

    if (absl::c_all_of(comp->instructions(), [&elementwise_comps,
                                              &allowed_parameter_indices](
                                                 const HloInstruction* inst) {
          return CanCluster(inst, /*allow_inputs=*/true, elementwise_comps,
                            allowed_parameter_indices);
        })) {
      VLOG(2) << "Found elementwise computation " << comp->name();
      elementwise_comps.insert(comp);
    }
  }
  return elementwise_comps;
}

StatusOr<std::vector<ElementwiseCluster>>
ResourceUpdateElementwiseClustering::GetClustersIn(
    HloInstruction* const call,
    const absl::flat_hash_set<const HloComputation*>& elementwise_comps) {
  CHECK(IsRepeatLoop(call) || IsPipelineOp(call));
  HloComputation* call_comp = call->to_apply();
  // Make sure that the root of the call op is a tuple instruction.
  TF_RETURN_IF_ERROR(FixRootInstruction(call_comp).status());

  std::vector<ElementwiseCluster> clusters;
  // Find the resource update.
  std::vector<HloInstruction*> resource_updates;
  absl::c_copy_if(call_comp->MakeInstructionPostOrder(),
                  std::back_inserter(resource_updates), IsResourceUpdate);
  if (resource_updates.empty()) {
    return clusters;
  } else if (resource_updates.size() > 1) {
    return FailedPrecondition("Detected multiple resource update.");
  }

  HloInstruction* resource_update = resource_updates[0];
  HloComputation* resource_update_comp = resource_update->to_apply();
  // Make sure that the root of the resource update is a tuple instruction.
  TF_RETURN_IF_ERROR(FixRootInstruction(resource_update_comp).status());

  auto offload_variables =
      GetResourceUpdatePartitionOffloadedVariables(resource_update);
  if (offload_variables == THREESTATE_OFF) {
    VLOG(2) << "Resource update partition offload is turned off, exiting.";
    return clusters;
  }

  // Find all the parameters which can be partitioned - these are the parameters
  // which we can guarantee are identical across replicas - this means that the
  // parameters are only inputs to the pipeline/repeat loop and that they can
  // only be modified by the resource update and their input/output aliasing
  // inside of the pipeline/loop has to match.

  // Do not optimize if this is not a op inside an entry computation.
  if (call->parent() != call->GetModule()->entry_computation()) {
    return clusters;
  }

  HloInstruction* call_root = call_comp->root_instruction();
  if (call_root->user_count() > 0) {
    return clusters;
  }

  absl::flat_hash_set<int64> allowed_resource_update_parameter_indices;
  for (int64 i = 0; i != call->operand_count(); ++i) {
    HloInstruction* call_operand = call->mutable_operand(i);
    if (!IsParameter(call_operand) && !call_operand->IsConstant() &&
        !IsWideConstant(call_operand)) {
      continue;
    }

    // Can only be used by call at a single index.
    if (call_operand->user_count() != 1) {
      continue;
    }
    auto input_indices = call->OperandIndices(call_operand);
    if (input_indices.size() != 1) {
      continue;
    }

    const int64 input_index = input_indices[0];
    HloInstruction* call_parameter =
        call_comp->parameter_instruction(input_index);

    auto resource_update_indices =
        resource_update->OperandIndices(call_parameter);
    const HloInstruction* call_root_operand = call_root->operand(input_index);
    if (call_root_operand == call_parameter) {
      // This value is not modified inside of the loop - any uses within the
      // resource update will be identical across replicas.
    } else {
      // This value is modified within the resource update.
      if (resource_update_indices.size() != 1) {
        continue;
      }
      if (call_root_operand->opcode() != HloOpcode::kGetTupleElement) {
        continue;
      }
      if (call_root_operand->operand(0) != resource_update) {
        continue;
      }
    }
    absl::c_copy(
        resource_update_indices,
        std::inserter(allowed_resource_update_parameter_indices,
                      allowed_resource_update_parameter_indices.begin()));
  }
  VLOG(2) << "Allowed resource update parameters are: "
          << absl::StrJoin(allowed_resource_update_parameter_indices, ", ");

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
        CanCluster(inst, /*allow_inputs=*/false, elementwise_comps,
                   allowed_resource_update_parameter_indices);
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
    if (cluster.Finalize(allowed_resource_update_parameter_indices)) {
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

StatusOr<bool> ResourceUpdateElementwiseClustering::OutlineCluster(
    ElementwiseCluster& cluster, uint32 replication_factor) {
  VLOG(2) << "Rewriting cluster with top in " << cluster.GetTop()->ToString()
          << ", " << cluster.GetPostOrder().size()
          << " instructions and replication factor " << replication_factor;
  if (cluster.GetShardSize() * replication_factor !=
      cluster.GetAlignedClusterSize()) {
    VLOG(2) << "Cluster shape and replica shape don't match "
            << cluster.GetShardSize() << " vs " << cluster.GetClusterSize()
            << "(" << cluster.GetAlignedClusterSize() << ")";
    return false;
  }

  // For each input:
  // Replace all-gather(remote-parameter-load)) with remote-parameter-load()
  // Replace other inputs with dynamic-slice(input, replication-index)
  for (auto cluster_input : cluster.GetInputs()) {
    if (IsScalar(cluster_input)) {
      VLOG(2) << "Scalar input does not need rewriting: "
              << cluster_input->ToString();
      continue;
    }

    TF_RETURN_IF_ERROR(
        RewriteClusterInput(cluster, replication_factor, cluster_input));
  }

  // Change the shape of the cluster.
  for (auto inst : cluster.GetPostOrder()) {
    if (inst->opcode() == HloOpcode::kFusion) {
      TF_RETURN_IF_ERROR(ChangeFusionShape(cluster, inst));
    } else {
      TF_RETURN_IF_ERROR(ChangeInstructionShape(cluster, inst));
    }
  }

  // Replacing outputs:
  // For each instruction of the cluster, check its users.
  // If its user is outside of cluster, do all-gather/reshape
  // If its user is store(shape(slice(shape(cluster)))), remove reshape
  // and slice.
  for (auto cluster_output : cluster.GetOutputs()) {
    if (IsScalar(cluster_output)) {
      VLOG(2) << "Scalar output does not need rewriting: "
              << cluster_output->ToString();
      continue;
    }

    TF_RETURN_IF_ERROR(
        RewriteClusterOutput(cluster, replication_factor, cluster_output,
                             cluster.GetUsersForOutput(cluster_output)));
  }

  return true;
}

StatusOr<bool> ResourceUpdateElementwiseClustering::Run(HloModule* module) {
  if (replication_factor_ <= 1) {
    VLOG(2) << "Skipping clustering, no replicas.";
    return false;
  }
  VLOG(2) << "Before the ResourceUpdateElementwiseClustering:";
  XLA_VLOG_LINES(2, module->ToString());

  std::vector<HloInstruction*> to_optimize;
  for (auto comp : module->MakeComputationPostOrder()) {
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (IsRepeatLoop(inst) || IsPipelineOp(inst)) {
        to_optimize.push_back(inst);
      }
    }
  }

  if (to_optimize.empty()) {
    VLOG(2) << "No resource updates found, exiting.";
    return false;
  }

  const absl::flat_hash_set<const HloComputation*> elementwise_comps =
      GetElementwiseClusterableComputations(module);

  bool module_changed = false;
  for (auto call : to_optimize) {
    TF_ASSIGN_OR_RETURN(
        auto changed,
        RewriteCall(module, call, elementwise_comps, replication_factor_));
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
