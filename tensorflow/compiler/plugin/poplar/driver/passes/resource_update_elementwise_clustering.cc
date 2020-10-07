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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/reduce_scatter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/replication_factor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/replication_index.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {
namespace {

bool IsParameter(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kParameter;
}

bool IsBroadcast(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kBroadcast;
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
    const CrossReplicaValidInputs& cross_replica_valid_inputs) {
  // A valid cluster input is one which is guaranteed to be identical across all
  // replicas.
  return cross_replica_valid_inputs.contains(inst) || IsWideConstant(inst) ||
         IsAllReduce(inst) || IsReplicatedParameterLoad(inst) ||
         IsNonReplicatedParameterLoad(inst);
}

bool CanCluster(
    const HloInstruction* inst, bool allow_inputs,
    const absl::flat_hash_set<const HloComputation*>& elementwise_comps,
    const CrossReplicaValidInputs& cross_replica_valid_inputs) {
  if (allow_inputs && ValidClusterInput(inst, cross_replica_valid_inputs)) {
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

StatusOr<Shape> GetNewShape(const ElementwiseCluster& cluster,
                            HloInstruction* inst) {
  if (IsScalar(inst) ||
      inst->shape().dimensions() == cluster.GetShardDimensions()) {
    VLOG(2) << "Instruction " << inst->ToString() << " preserves shapes.";
    return inst->shape();
  } else if (inst->shape().dimensions() == cluster.GetClusterDimensions()) {
    Shape new_shape = ShapeUtil::MakeShape(inst->shape().element_type(),
                                           cluster.GetShardDimensions());
    VLOG(2) << "Instruction " << inst->ToString() << " will be reshaped to "
            << new_shape;
    return new_shape;
  } else {
    return InternalErrorStrCat(
        "Unexpected shape (",
        inst->shape().ToString() + ") in instruction: ", inst->ToString());
  }
}

StatusOr<HloComputation*> CloneAndReshapeComputation(
    const ElementwiseCluster& cluster, HloComputation* comp) {
  VLOG(2) << "Clone and reshape computation: " << comp->name();
  HloComputation* cloned_comp =
      comp->parent()->AddEmbeddedComputation(comp->Clone());
  for (auto inst : cloned_comp->instructions()) {
    TF_ASSIGN_OR_RETURN(Shape new_shape, GetNewShape(cluster, inst));
    *inst->mutable_shape() = new_shape;
  }
  return cloned_comp;
}

Status AddClusterInstruction(const ElementwiseCluster& cluster,
                             HloInstruction* inst,
                             HloComputation::Builder* builder,
                             HloCloneContext* context) {
  if (inst->opcode() == HloOpcode::kFusion) {
    HloComputation* fusion_comp = inst->fused_instructions_computation();
    TF_ASSIGN_OR_RETURN(HloComputation * reshaped_comp,
                        CloneAndReshapeComputation(cluster, fusion_comp));
    context->MapComputation(fusion_comp, reshaped_comp);
  }

  TF_ASSIGN_OR_RETURN(Shape new_shape, GetNewShape(cluster, inst));
  std::vector<HloInstruction*> new_operands(inst->operand_count());
  absl::c_transform(inst->operands(), new_operands.begin(),
                    [context](HloInstruction* old_operand) {
                      return context->GetInstruction(old_operand);
                    });
  HloInstruction* new_inst = builder->AddInstruction(
      inst->CloneWithNewOperands(new_shape, new_operands, context));
  context->MapInstruction(inst, new_inst);
  return Status::OK();
}

// For each input:
// * pass scalar inputs as normal inputs.
// * reshape(all-gather(reshape(remote-parameter-load))) pass
// remote-parameter-load() as input.
// * for other inputs add dynamic-slice(input, replication-index) inside of the
// computation.
// Returns the instruction which should be the input to the outlined
// computation.
StatusOr<HloInstruction*> AddClusterInput(int64 param_idx,
                                          const ElementwiseCluster& cluster,
                                          int64 replication_factor,
                                          HloInstruction* cluster_input,
                                          HloComputation::Builder* builder,
                                          HloCloneContext* context) {
  HloComputation* input_comp = cluster_input->parent();

  if (IsScalar(cluster_input)) {
    VLOG(2) << "Scalar input does not need rewriting: "
            << cluster_input->ToString();
    HloInstruction* parameter = builder->AddInstruction(
        HloInstruction::CreateParameter(param_idx, cluster_input->shape(),
                                        "parameter-" + cluster_input->name()));
    context->MapInstruction(cluster_input, parameter);
    return cluster_input;
  }

  const Shape& cluster_input_shape = cluster_input->shape();
  auto cluster_input_type = cluster_input_shape.element_type();

  const Shape in_comp_shape =
      ShapeUtil::MakeShape(cluster_input_type, cluster.GetShardDimensions());

  if (IsWideConstant(cluster_input)) {
    // Prevent aliasing from being expanded at the callsite by adding the
    // constant as the input and broadcasting it inside of the computation.
    HloInstruction* fusion_bcast = cluster_input->fused_expression_root();
    HloInstruction* fusion_const = fusion_bcast->mutable_operand(0);
    HloInstruction* new_input =
        input_comp->AddInstruction(fusion_const->Clone());

    HloInstruction* parameter =
        builder->AddInstruction(HloInstruction::CreateParameter(
            param_idx, new_input->shape(), "parameter-" + new_input->name()));

    HloInstruction* bcast = builder->AddInstruction(
        HloInstruction::CreateBroadcast(in_comp_shape, parameter, {}));

    context->MapInstruction(cluster_input, bcast);
    return new_input;
  }

  if (IsBroadcast(cluster_input)) {
    HloInstruction* bcast_input = cluster_input->mutable_operand(0);
    CHECK(IsScalar(bcast_input));

    HloInstruction* parameter = builder->AddInstruction(
        HloInstruction::CreateParameter(param_idx, bcast_input->shape(),
                                        "parameter-" + bcast_input->name()));

    HloInstruction* bcast = builder->AddInstruction(
        HloInstruction::CreateBroadcast(in_comp_shape, parameter, {}));

    context->MapInstruction(cluster_input, bcast);
    return bcast_input;
  }

  // If it's reshape(all-gather(reshape(remote-parameter-laod))), remove
  // all-gather.
  if (IsReplicatedParameterLoad(cluster_input)) {
    HloInstruction* remote_load = cluster_input->mutable_operand(0);
    if (remote_load->shape().dimensions() == cluster.GetShardDimensions()) {
      VLOG(2) << "Adding remote cluster input " << remote_load->ToString();
      HloInstruction* parameter = builder->AddInstruction(
          HloInstruction::CreateParameter(param_idx, remote_load->shape(),
                                          "parameter-" + remote_load->name()));

      HloInstruction* reshape = builder->AddInstruction(
          HloInstruction::CreateReshape(in_comp_shape, parameter));
      context->MapInstruction(cluster_input, reshape);
      return remote_load;
    }
  }

  VLOG(2) << "Adding cluster input " << cluster_input->ToString();
  // Reshape the cluster input into a vector and use that as the parameter -
  // this will allow clusters to be shared between different shapes with the
  // same number of elements.
  CHECK_EQ(cluster.GetClusterSize(),
           ShapeUtil::ElementsIn(cluster_input_shape));

  auto pad_input = [&](HloInstruction* input) -> HloInstruction* {
    if (cluster.GetClusterSize() != cluster.GetAlignedClusterSize()) {
      HloInstruction* zero_f =
          builder->AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::Zero(cluster_input_type)));

      // Pad the tensor to be a multiple of `replication_factor`.
      const Shape pad_shape = ShapeUtil::MakeShape(
          cluster_input_type, {cluster.GetAlignedClusterSize()});

      PaddingConfig padding_config;
      const std::size_t difference =
          cluster.GetAlignedClusterSize() - cluster.GetClusterSize();
      auto padding_config_dim = padding_config.add_dimensions();
      padding_config_dim->set_edge_padding_high(difference);
      padding_config_dim->set_edge_padding_low(0);
      padding_config_dim->set_interior_padding(0);

      input = builder->AddInstruction(
          HloInstruction::CreatePad(pad_shape, input, zero_f, padding_config));
      VLOG(2) << "pad: " << input->ToString();
    }
    return input;
  };

  const Shape flat_shape =
      ShapeUtil::MakeShape(cluster_input_type, {cluster.GetClusterSize()});
  // Convert an all reduce input into an outlined reduce scatter sum.
  if (IsAllReduce(cluster_input)) {
    HloInstruction* input = cluster_input->mutable_operand(0);
    HloInstruction* input_reshaped = input_comp->AddInstruction(
        HloInstruction::CreateReshape(flat_shape, input));

    HloInstruction* parameter =
        builder->AddInstruction(HloInstruction::CreateParameter(
            param_idx, flat_shape, "parameter-" + input->name()));

    // Add any necessary padding before scatter.
    parameter = pad_input(parameter);

    HloInstruction* scatter = builder->AddInstruction(
        CreateReduceScatter({parameter}, in_comp_shape));
    context->MapInstruction(cluster_input, scatter);
    return input_reshaped;
  }

  // All other inputs have to be sliced with dynamic-slice(input,
  // replication-index()).
  HloInstruction* cluster_input_reshaped = input_comp->AddInstruction(
      HloInstruction::CreateReshape(flat_shape, cluster_input));

  const Shape all_shards_shape = ShapeUtil::MakeShape(
      cluster_input_type, {replication_factor, cluster.GetShardSize()});

  HloInstruction* parameter =
      builder->AddInstruction(HloInstruction::CreateParameter(
          param_idx, flat_shape, "parameter-" + cluster_input->name()));

  // Add any necessary padding before slicing.
  parameter = pad_input(parameter);

  // Reshaped the parameter so that it can be sliced.
  HloInstruction* reshaped = builder->AddInstruction(
      HloInstruction::CreateReshape(all_shards_shape, parameter));

  HloInstruction* replica_id =
      builder->AddInstruction(CreateReplicationIndex());

  HloInstruction* zero_i =
      builder->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(replica_id->shape().element_type())));

  // Slice off this replica's storage elements.
  const Shape slice_shape =
      ShapeUtil::MakeShape(cluster_input_type, {1, cluster.GetShardSize()});
  HloInstruction* slice =
      builder->AddInstruction(HloInstruction::CreateDynamicSlice(
          slice_shape, reshaped, {replica_id, zero_i},
          {1, cluster.GetShardSize()}));

  HloInstruction* input_slice = builder->AddInstruction(
      HloInstruction::CreateReshape(in_comp_shape, slice));

  VLOG(2) << "Input slice: " << input_slice->ToString();
  context->MapInstruction(cluster_input, input_slice);
  return cluster_input_reshaped;
}

// For each output of the cluster, check its users.
// If its user is store(shape(slice(shape(cluster)))), remove reshape and slice.
// If its user is outside of cluster, do all-gather and reshape.
// Returns the instruction which should be passed to the output tuple.
StatusOr<HloInstruction*> AddClusterOutput(
    const ElementwiseCluster& cluster, int64 replication_factor,
    HloInstruction* cluster_output, std::vector<UserPositions>& inst_users,
    HloComputation::Builder* builder, HloCloneContext* context) {
  HloInstruction* in_cluster_output = context->GetInstruction(cluster_output);
  if (IsScalar(cluster_output)) {
    VLOG(2) << "Scalar output does not need rewriting: "
            << cluster_output->ToString();
    return in_cluster_output;
  }

  // Check whether the cluster is just being stored.
  if (inst_users.size() == 1 &&
      IsReplicatedParameterStore(inst_users[0].instruction)) {
    HloInstruction* store_input = inst_users[0].instruction;
    HloInstruction* store = store_input->users()[0];
    if (store_input->shape().dimensions() == cluster.GetShardDimensions()) {
      VLOG(2) << "Skipping the extra remote store fusion for "
              << store->ToString();
      // Reshape it so that it can be stored.
      HloInstruction* reshape =
          builder->AddInstruction(HloInstruction::CreateReshape(
              store_input->shape(), in_cluster_output));
      // Override the inst_users to be the parameter store instruction.
      inst_users = {UserPositions{store, {1}, false}};
      return reshape;
    }
  }

  // Create all gather.
  auto inst_element_type = cluster_output->shape().element_type();
  const Shape all_gather_shape = ShapeUtil::MakeShape(
      inst_element_type, {replication_factor, cluster.GetShardSize()});
  HloInstruction* all_gather = builder->AddInstruction(
      CreateAllGather({in_cluster_output}, all_gather_shape));

  const Shape flat_cluster_shape =
      ShapeUtil::MakeShape(inst_element_type, {cluster.GetClusterSize()});
  const Shape aligned_cluster_shape = ShapeUtil::MakeShape(
      inst_element_type, {cluster.GetAlignedClusterSize()});
  HloInstruction* output = builder->AddInstruction(
      HloInstruction::CreateReshape(aligned_cluster_shape, all_gather));

  if (cluster.GetClusterSize() != cluster.GetAlignedClusterSize()) {
    output = builder->AddInstruction(HloInstruction::CreateSlice(
        flat_cluster_shape, output, {0}, {cluster.GetClusterSize()}, {1}));
    VLOG(2) << "Slicing padding, slice: " << output->ToString();
  }
  return output;
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
    const CrossReplicaValidInputs& cross_replica_valid_inputs) {
  CHECK(!finalized_);

  if (IsScalar(top_)) {
    return false;
  }

  // Check all inputs are valid.
  for (auto input : inputs_) {
    if (!ValidClusterInput(input, cross_replica_valid_inputs)) {
      VLOG(2) << "Invalid cluster input: " << input->ToString();
      return false;
    }
  }

  // Check at least one input is remote.
  if (!absl::c_any_of(inputs_, IsReplicatedParameterLoad)) {
    VLOG(2) << "No replicated parameter loads found.";
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
        outputs_to_users_[inst].push_back(
            UserPositions{user, user->OperandIndices(inst), true});
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
  CHECK_EQ(shard_dimensions_.size(), 1);
  shard_size_ = shard_dimensions_[0];

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

namespace {
absl::flat_hash_set<int64> GetPartitionableResourceUpdateInputs(
    const HloInstruction* call, const HloInstruction* resource_update) {
  const HloComputation* call_comp = call->to_apply();
  const HloInstruction* call_root = call_comp->root_instruction();

  absl::flat_hash_set<int64> allowed_resource_update_parameter_indices;
  for (int64 i = 0; i != call->operand_count(); ++i) {
    const HloInstruction* call_operand = call->operand(i);
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
  return allowed_resource_update_parameter_indices;
}

CrossReplicaValidInputs GetCrossReplicaValidInputs(
    const absl::flat_hash_set<int64>& allowed_resource_update_parameter_indices,
    const HloComputation* comp) {
  CrossReplicaValidInputs cross_replica_valid_inputs;
  for (const HloInstruction* inst : comp->MakeInstructionPostOrder()) {
    bool valid_input = false;
    if (IsParameter(inst) && allowed_resource_update_parameter_indices.contains(
                                 inst->parameter_number())) {
      valid_input = true;
    } else if (IsScalar(inst) && inst->IsElementwise()) {
      // Note that this also captures constants.
      valid_input = absl::c_all_of(
          inst->operands(),
          [&cross_replica_valid_inputs](const HloInstruction* operand) {
            return cross_replica_valid_inputs.contains(operand);
          });
    } else if (IsBroadcast(inst)) {
      valid_input = IsScalar(inst->operand(0)) &&
                    cross_replica_valid_inputs.contains(inst->operand(0));
    }

    if (valid_input) {
      cross_replica_valid_inputs.insert(inst);
    }
  }
  return cross_replica_valid_inputs;
}
}  // namespace

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
    CrossReplicaValidInputs cross_replica_valid_inputs =
        GetCrossReplicaValidInputs(allowed_parameter_indices, comp);

    if (absl::c_all_of(comp->instructions(), [&elementwise_comps,
                                              &cross_replica_valid_inputs](
                                                 const HloInstruction* inst) {
          return CanCluster(inst, /*allow_inputs=*/true, elementwise_comps,
                            cross_replica_valid_inputs);
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

  absl::flat_hash_set<int64> allowed_resource_update_parameter_indices =
      GetPartitionableResourceUpdateInputs(call, resource_update);
  VLOG(2) << "Allowed resource update parameters are: "
          << absl::StrJoin(allowed_resource_update_parameter_indices, ", ");
  // Given the valid input indicies, find all the instructions which are valid
  // cluster inputs.
  CrossReplicaValidInputs cross_replica_valid_inputs =
      GetCrossReplicaValidInputs(allowed_resource_update_parameter_indices,
                                 resource_update_comp);

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
                   cross_replica_valid_inputs);
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
    bool valid = cluster.Finalize(cross_replica_valid_inputs);

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

  HloComputation* cluster_comp = cluster.GetComputation();
  HloModule* module = cluster_comp->parent();
  HloCloneContext context(module);
  HloComputation::Builder builder("elementwise_cluster");

  // Add all the inputs to the computation and get the caller inputs.
  std::vector<HloInstruction*> caller_inputs;
  caller_inputs.reserve(cluster.GetInputs().size());
  for (auto cluster_input : cluster.GetInputs()) {
    TF_ASSIGN_OR_RETURN(
        HloInstruction * comp_input,
        AddClusterInput(caller_inputs.size(), cluster, replication_factor,
                        cluster_input, &builder, &context));
    caller_inputs.push_back(comp_input);
  }

  // Add all the instructions to the cluster.
  for (auto inst : cluster.GetPostOrder()) {
    TF_RETURN_IF_ERROR(
        AddClusterInstruction(cluster, inst, &builder, &context));
  }

  // Process all the outputs and get all the operands for the computation tuple
  // output.
  std::vector<HloInstruction*> computation_outputs;

  // When rewriting outputs the computation output users can change - for
  // example all-gathers can be elided.
  HloInstructionMap<std::vector<UserPositions>> computation_output_users;
  computation_outputs.reserve(cluster.GetOutputs().size());
  for (auto cluster_output : cluster.GetOutputs()) {
    std::vector<UserPositions> output_users =
        cluster.GetUsersForOutput(cluster_output);

    TF_ASSIGN_OR_RETURN(
        HloInstruction * in_cluster_output,
        AddClusterOutput(cluster, replication_factor, cluster_output,
                         output_users, &builder, &context));

    computation_outputs.push_back(in_cluster_output);
    computation_output_users[cluster_output] = output_users;
  }

  // Create the root tuple with all the outputs and build the computation.
  HloInstruction* cluster_output =
      builder.AddInstruction(HloInstruction::CreateTuple(computation_outputs));
  HloComputation* outlined_comp =
      module->AddEmbeddedComputation(builder.Build(cluster_output));
  HloInstruction* call =
      cluster_comp->AddInstruction(HloInstruction::CreateCall(
          cluster_output->shape(), caller_inputs, outlined_comp));

  // Set call to be a function.
  auto backend_config =
      call->backend_config<PoplarBackendConfig>().ValueOrDie();
  auto* call_config = backend_config.mutable_call_config();
  call_config->set_type(PoplarBackendConfig::CallConfig::Function);
  auto* function_config = call_config->mutable_function_config();
  // Because inputs will be dynamically sliced, keep the non-sliced layouts at
  // the callsite - this means any rearrangement will only be done once inside
  // of the call rather than at every callsite.
  function_config->set_keep_input_layouts(true);
  // Make sure that all inputs are copied to a single device with the most
  // parameters before the function call - in a resource update this will allow
  // copies of all hyper parameters to be scheduled earlier.
  function_config->set_unique_sharding(true);
  TF_RETURN_IF_ERROR(call->set_backend_config(backend_config));

  // Connect up all the users of the cluster output.
  int64 output_idx = 0;
  for (auto cluster_output : cluster.GetOutputs()) {
    TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                        MakeGetTupleElementHlo(call, output_idx++));

    HloInstruction* reshaped = nullptr;
    for (auto user : computation_output_users.at(cluster_output)) {
      HloInstruction* to_replace_with = gte;
      if (user.reshape) {
        if (!reshaped) {
          reshaped = cluster_comp->AddInstruction(
              HloInstruction::CreateReshape(cluster_output->shape(), gte));
        }
        to_replace_with = reshaped;
      }

      VLOG(2) << "Replacing " << user.ToString();
      for (int64 index : user.indices) {
        TF_RETURN_IF_ERROR(
            user.instruction->ReplaceOperandWith(index, to_replace_with));
      }
    }
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
