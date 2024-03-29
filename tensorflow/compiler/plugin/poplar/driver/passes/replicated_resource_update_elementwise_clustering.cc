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

#include "tensorflow/compiler/plugin/poplar/driver/passes/replicated_resource_update_elementwise_clustering.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/all_gather.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/collective_reorder.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/reduce_scatter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/elementwise_cluster.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/literal_util.h"

namespace xla {
namespace poplarplugin {

namespace {

bool IsParameter(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kParameter;
}

StatusOr<CollectiveOperator> GetAllReduceCollectiveOp(
    const HloInstruction* inst) {
  if (IsAllReduceAdd(inst)) {
    return CollectiveOperator::COLLECTIVE_OP_ADD;
  } else if (IsAllReduceMean(inst)) {
    return CollectiveOperator::COLLECTIVE_OP_MEAN;
  } else {
    return tensorflow::errors::InvalidArgument(
        "Unsupported all reduce instruction, expected computation to match ADD "
        "or MEAN");
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
  HloInstruction* root = cloned_comp->root_instruction();
  if (root->opcode() == HloOpcode::kReduce) {
    TF_RETURN_IF_ERROR(cloned_comp->ReplaceWithNewInstruction(
        root, HloInstruction::CreateReduce(
                  root->shape(), root->mutable_operand(0),
                  root->mutable_operand(1), {0}, root->to_apply())));
  }
  return cloned_comp;
}

std::unique_ptr<HloComputation> CreateSumReduction(const Shape& shape,
                                                   const std::string& name) {
  HloComputation::Builder builder(name);

  auto* lhs =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "lhs"));

  auto* rhs =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "rhs"));

  auto* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, lhs, rhs));

  return builder.Build(add);
}

StatusOr<std::unique_ptr<HloComputation>> CreateMeanReduction(
    const Shape& shape, const std::string& name,
    const float rhs_divisor = 1.0f) {
  HloComputation::Builder builder(name);

  auto* lhs =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "lhs"));

  auto* rhs =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "rhs"));

  HloInstruction* scaled_rhs;
  if (rhs_divisor == 1.0f) {
    scaled_rhs = rhs;
  } else {
    TF_ASSIGN_OR_RETURN(Literal literal,
                        LiteralUtil::CreateR0(rhs_divisor)
                            .Convert(rhs->shape().element_type()));
    auto* divisor = builder.AddInstruction(
        HloInstruction::CreateConstant(std::move(literal)));
    scaled_rhs = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kDivide, rhs, divisor));
  }

  auto* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, lhs, scaled_rhs));

  return builder.Build(add);
}

}  // namespace

std::unique_ptr<ElementwiseClusterValidator>
ReplicatedResourceUpdateElementwiseClustering::CreateValidator(
    const HloComputation* resource_update_comp) const {
  ElementwiseClusterValidator::Inputs valid_inputs;
  for (const HloInstruction* inst :
       resource_update_comp->MakeInstructionPostOrder()) {
    bool valid_input = false;
    if (partition_replication_factor_ < 2) {
      valid_input = true;
    } else {
      // Check that all the leaf shapes are identical.
      valid_input = absl::c_all_of(
          ShapeUtil::GetLeafShapes(inst->shape()),
          [this, inst](const ShapeUtil::IndexedShape& indexed_shape) {
            auto status_or =
                replica_identical_dataflow_analysis_
                    .IsValueIdenticalAcrossReplicas(inst, indexed_shape.index);
            return status_or.ok() ? status_or.ValueOrDie() : false;
          });
    }

    if (valid_input) {
      valid_inputs.insert(inst);
    }
  }

  return ResourceUpdateElementwiseClustering::CreateValidator(valid_inputs);
}

// For each input of partitioned clusters:
// * Reshape replicated parameter load
// * Lower all reduce into cluster if all users are in this cluster too.
// * For other inputs add dynamic-slice(input, replication-index) inside of the
// computation.
// For all other inputs use base implementation.

StatusOr<HloInstruction*>
ReplicatedResourceUpdateElementwiseClustering::AddClusterInput(
    int64_t param_idx, const ElementwiseCluster& cluster,
    HloInstruction* cluster_input, HloComputation::Builder* builder,
    HloCloneContext* context) const {
  CHECK(cluster.IsReplicaPartitioned());

  const Shape& cluster_input_shape = cluster_input->shape();
  auto cluster_input_type = cluster_input_shape.element_type();
  const Shape in_comp_shape =
      ShapeUtil::MakeShape(cluster_input_type, cluster.GetShardDimensions());

  // If it's replicated parameter load, allow it as is without slicing.
  if (IsReplicatedParameterLoad(cluster_input)) {
    HloInstruction* remote_load = cluster_input->mutable_operand(0);
    if (remote_load->shape().dimensions() == cluster.GetShardDimensions()) {
      VLOG(2) << "Adding remote cluster input " << remote_load->ToString();
      HloInstruction* parameter = builder->AddInstruction(
          HloInstruction::CreateParameter(param_idx, remote_load->shape(),
                                          "parameter-" + remote_load->name()));

      if (!ShapeUtil::Compatible(in_comp_shape, parameter->shape())) {
        parameter = builder->AddInstruction(
            HloInstruction::CreateReshape(in_comp_shape, parameter));
      }
      context->MapInstruction(cluster_input, parameter);
      return remote_load;
    }
  }

  VLOG(2) << "Adding cluster input " << cluster_input->ToString();
  // Reshape the cluster input into a vector and use that as the parameter -
  // this will allow clusters to be shared between different shapes with the
  // same number of elements.
  CHECK_EQ(cluster.GetClusterSize(),
           ShapeUtil::ElementsIn(cluster_input_shape));

  const Shape flat_shape =
      ShapeUtil::MakeShape(cluster_input_type, {cluster.GetClusterSize()});

  const auto status_or_collective_op = GetAllReduceCollectiveOp(cluster_input);

  // Lower the all reduce into the cluster if all its users will be in the
  // cluster too.
  const bool lower_all_reduce = IsGlobalAllReduce(cluster_input) &&
                                cluster.ContainsAllUsersOf(cluster_input) &&
                                status_or_collective_op.ok();

  if (lower_all_reduce) {
    HloInstruction* input = cluster_input->mutable_operand(0);
    HloInstruction* parameter = builder->AddInstruction(
        HloInstruction::CreateParameter(param_idx, cluster_input->shape(),
                                        "parameter-reduce-" + input->name()));

    if (!ShapeUtil::Compatible(parameter->shape(), flat_shape)) {
      parameter = builder->AddInstruction(
          HloInstruction::CreateReshape(flat_shape, parameter));
    }

    // Convert an all reduce input into an outlined reduce scatter sum for
    // partitined graphs.
    // Add any necessary padding before scatter.
    TF_ASSIGN_OR_RETURN(parameter, PadInput(cluster, parameter, builder));

    HloInstruction* scatter_input =
        builder->AddInstruction(CreateCollectiveReorderInstruction(parameter));

    HloInstruction* scatter = builder->AddInstruction(CreateReduceScatter(
        in_comp_shape, {scatter_input}, status_or_collective_op.ValueOrDie(),
        PoplarReplicaGroups::Consecutive(partition_replication_factor_)));

    if (partition_replication_factor_ == global_replication_factor_) {
      // If we partition across all the replicas, we already have the correct
      // data per replica from the reduce scatter.
      context->MapInstruction(cluster_input, scatter);
      return input;
    }

    // If we partition across a subset of the replicas, the reduce scatter has
    // only summed over the subset that we partition across. We then also need
    // to sum over the orthogonal replicas that share the same partition of the
    // variable.
    //
    // Reference example partitioning across all the replicas:
    // partition_replication_factor 4 and global_replication_factor 4.
    //            -------                     -------
    // replica 0: 1 0 1 1  reduce-scatter     1
    // replica 1: 0 0 1 1  consecutive    ->    2
    // replica 2: 0 1 0 1  group size 4           3
    // replica 3: 0 1 1 1                           4
    //            -------                     -------
    //
    // The same example, but partitioned across half of the replicas:
    // partition_replication_factor 2 and global_replication_factor 4.
    //            -------                   -------                 -------
    // replica 0: 1 0 1 1                   1 0                     1 2
    // replica 1: 0 0 1 1  reduce-scatter       2 2  all-reduce         3 4
    //            -------  consecutive   -> -------  orthogonal  -> -------
    // replica 2: 0 1 0 1  group size 2     0 2      group size 2   1 2
    // replica 3: 0 1 1 1                       1 2                     3 4
    //            -------                   -------                 -------

    if (ipu_link_domain_replication_factor_ < global_replication_factor_ &&
        partition_replication_factor_ < ipu_link_domain_replication_factor_) {
      // TODO(T45704): GCL does not support this and will report an error, but
      // the message might be hard to understand for the user. So we catch it
      // here to attempt to report a more understandable error message.
      return Unimplemented(
          "Replicated partitioning is not supported when there are multiple "
          "instances per IPU-link domain. The number of local replicas per "
          "partitioning domain (%u) is less than the number of replicas "
          "per IPU-link domain (%u).",
          partition_replication_factor_, ipu_link_domain_replication_factor_);
    }

    CHECK_NE(partition_replication_factor_, 0);
    CHECK_EQ(global_replication_factor_ % partition_replication_factor_, 0);
    const uint64 orthogonal_group_size =
        global_replication_factor_ / partition_replication_factor_;
    const auto orthogonal_groups =
        PoplarReplicaGroups::Orthogonal(orthogonal_group_size);

    std::unique_ptr<HloComputation> reduce_comp;
    if (IsAllReduceAdd(cluster_input)) {
      reduce_comp =
          CreateSumReduction(scatter->shape(), "sum-" + scatter->name());
    } else if (IsAllReduceMean(cluster_input)) {
      TF_ASSIGN_OR_RETURN(
          reduce_comp,
          CreateMeanReduction(
              scatter->shape(), "mean-" + scatter->name(),
              /* Number of partitions */
              static_cast<float>(global_replication_factor_) /
                  static_cast<float>(partition_replication_factor_)));
    } else {
      return tensorflow::errors::InvalidArgument(
          "Unsupported all reduce instruction found for partitions, expected "
          "computation to match ADD or MEAN");
    }

    HloComputation* reduction =
        context->module()->AddEmbeddedComputation(std::move(reduce_comp));

    HloInstruction* all_reduce =
        builder->AddInstruction(HloInstruction::CreateAllReduce(
            scatter->shape(), std::vector<HloInstruction*>{scatter}, reduction,
            orthogonal_groups.ToXlaReplicaGroups(),
            /*constrain_layout=*/false,
            /*channel_id=*/absl::nullopt, /*use_global_device_ids=*/false));

    context->MapInstruction(cluster_input, all_reduce);
    return input;
  }

  // All other inputs have to be sliced with dynamic-slice(input,
  // replication-index()).
  HloInstruction* parameter = builder->AddInstruction(
      HloInstruction::CreateParameter(param_idx, cluster_input->shape(),
                                      "parameter-" + cluster_input->name()));

  // Add any necessary padding before slicing.
  TF_ASSIGN_OR_RETURN(parameter, PadInput(cluster, parameter, builder));

  const Shape all_shards_shape = ShapeUtil::MakeShape(
      cluster_input_type,
      {partition_replication_factor_ * cluster.GetShardSize()});

  // Reshaped the parameter so that it can be sliced.
  if (!ShapeUtil::Compatible(parameter->shape(), all_shards_shape)) {
    parameter = builder->AddInstruction(
        HloInstruction::CreateReshape(all_shards_shape, parameter));
  }

  HloInstruction* scatter_input =
      builder->AddInstruction(CreateCollectiveReorderInstruction(parameter));

  // Slice off this replica's storage elements.
  const Shape slice_shape =
      ShapeUtil::MakeShape(cluster_input_type, {cluster.GetShardSize()});

  // reduce-scatter(op=LOCAL) slices replica-specific shard of the collective
  // input tensor and does nothing with the values. This is more memory/cycles
  // efficient than doing dynamic-slice(replication-index() * shard_size).
  HloInstruction* slice = builder->AddInstruction(CreateReduceScatter(
      slice_shape, {scatter_input}, CollectiveOperator::COLLECTIVE_OP_LOCAL,
      PoplarReplicaGroups::Consecutive(partition_replication_factor_)));

  if (!ShapeUtil::Compatible(in_comp_shape, slice->shape())) {
    slice = builder->AddInstruction(
        HloInstruction::CreateReshape(in_comp_shape, slice));
  }

  VLOG(2) << "Input slice: " << slice->ToString();
  context->MapInstruction(cluster_input, slice);
  return cluster_input;
}

// For each output of the cluster, check its users.
// If its user is store(shape(slice(shape(cluster)))), remove reshape and slice.
// If its user is outside of cluster, do all-gather and reshape.
// Returns the instruction which should be passed to the output tuple.

StatusOr<HloInstruction*>
ReplicatedResourceUpdateElementwiseClustering::AddClusterOutput(
    const ElementwiseCluster& cluster, HloInstruction* cluster_output,
    std::vector<UserPositions>& inst_users, HloComputation::Builder* builder,
    HloCloneContext* context) const {
  CHECK(cluster.IsReplicaPartitioned());
  HloInstruction* in_cluster_output = context->GetInstruction(cluster_output);

  // Check whether the cluster is just being stored.
  if (inst_users.size() == 1 &&
      IsReplicatedParameterStore(inst_users[0].instruction)) {
    HloInstruction* store_input = inst_users[0].instruction;
    HloInstruction* store = store_input->users()[0];
    if (store_input->shape().dimensions() == cluster.GetShardDimensions()) {
      VLOG(2) << "Skipping the extra remote store fusion for "
              << store->ToString();
      // Override the inst_users to be the parameter store instruction.
      inst_users = {UserPositions{store, {1}}};

      if (!ShapeUtil::Compatible(store_input->shape(),
                                 in_cluster_output->shape())) {
        // Reshape it so that it can be stored.
        HloInstruction* reshape =
            builder->AddInstruction(HloInstruction::CreateReshape(
                store_input->shape(), in_cluster_output));
        return reshape;
      } else {
        return in_cluster_output;
      }
    }
  }

  // Create all gather.
  auto inst_element_type = cluster_output->shape().element_type();
  const Shape all_gather_shape = ShapeUtil::MakeShape(
      inst_element_type,
      {partition_replication_factor_, cluster.GetShardSize()});
  HloInstruction* output = builder->AddInstruction(CreatePoplarAllGather(
      {in_cluster_output}, all_gather_shape,
      PoplarReplicaGroups::Consecutive(partition_replication_factor_)));

  output =
      builder->AddInstruction(CreateUndoCollectiveReorderInstruction(output));

  const Shape flat_cluster_shape =
      ShapeUtil::MakeShape(inst_element_type, {cluster.GetClusterSize()});
  const Shape aligned_cluster_shape = ShapeUtil::MakeShape(
      inst_element_type, {cluster.GetAlignedClusterSize()});

  if (cluster.GetClusterSize() != cluster.GetAlignedClusterSize()) {
    output = builder->AddInstruction(
        HloInstruction::CreateReshape(aligned_cluster_shape, output));
    output = builder->AddInstruction(HloInstruction::CreateSlice(
        flat_cluster_shape, output, {0}, {cluster.GetClusterSize()}, {1}));
    VLOG(2) << "Slicing padding, slice: " << output->ToString();
  }

  const Shape cluster_shape = cluster.GetClusterShape(inst_element_type);
  if (!ShapeUtil::Compatible(cluster_shape, output->shape())) {
    output = builder->AddInstruction(
        HloInstruction::CreateReshape(cluster_shape, output));
  }

  return output;
}

Status ReplicatedResourceUpdateElementwiseClustering::AddClusterInstruction(
    const ElementwiseCluster& cluster, HloInstruction* inst,
    HloComputation::Builder* builder, HloCloneContext* context) const {
  CHECK(cluster.IsReplicaPartitioned());

  if (inst->opcode() == HloOpcode::kFusion) {
    HloComputation* fusion_comp = inst->fused_instructions_computation();
    TF_ASSIGN_OR_RETURN(HloComputation * reshaped_comp,
                        CloneAndReshapeComputation(cluster, fusion_comp));
    context->MapComputation(fusion_comp, reshaped_comp);
  }

  TF_ASSIGN_OR_RETURN(Shape new_shape, GetNewShape(cluster, inst));
  TF_RETURN_IF_ERROR(CloneInstruction(new_shape, inst, builder, context));

  if (IsReductionFusion(inst) || inst->opcode() == HloOpcode::kReduce) {
    CHECK(IsScalar(inst));

    HloComputation* fusion_comp =
        context->GetComputation(inst->fused_instructions_computation());
    CHECK_NOTNULL(fusion_comp);

    HloInstruction* cloned_inst = context->GetInstruction(inst);
    CHECK_NOTNULL(cloned_inst);

    // Note that this trick of replacing a local-reduce to a sharded
    // local-reduce then an all-reduce is only valid when the values aren't
    // getting near to the limit of the data type.
    // For example, a whole can be greater than the sum of its parts if each
    // part is rounded down to be representable but the whole is rounded up.
    VLOG(2) << "Adding all-reduce after instruction " << cloned_inst->name()
            << " in the RTS cluster to make it replica-identical";
    HloComputation* sum_reduction = context->module()->AddEmbeddedComputation(
        CreateSumReduction(cloned_inst->shape(), "sum-" + cloned_inst->name()));

    HloInstruction* all_reduce =
        builder->AddInstruction(HloInstruction::CreateAllReduce(
            cloned_inst->shape(), std::vector<HloInstruction*>{cloned_inst},
            sum_reduction, {}, /*constrain_layout=*/false,
            /*channel_id=*/absl::nullopt, /*use_global_device_ids=*/false));

    // Replace all usage of the old local reduction with the new all-reduction
    // by using the clone context map.
    context->MapInstruction(inst, all_reduce);
  }
  return Status::OK();
}

StatusOr<HloInstruction*>
ReplicatedResourceUpdateElementwiseClustering::PadInput(
    const ElementwiseCluster& cluster, HloInstruction* input,
    HloComputation::Builder* builder) {
  if (cluster.GetClusterSize() != cluster.GetAlignedClusterSize()) {
    auto cluster_input_type = input->shape().element_type();
    HloInstruction* zero_f = builder->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(cluster_input_type)));

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

    const Shape flat_shape = ShapeUtil::MakeShape(input->shape().element_type(),
                                                  {cluster.GetClusterSize()});
    if (!ShapeUtil::Compatible(input->shape(), flat_shape)) {
      input = builder->AddInstruction(
          HloInstruction::CreateReshape(flat_shape, input));
    }
    input = builder->AddInstruction(
        HloInstruction::CreatePad(pad_shape, input, zero_f, padding_config));
    VLOG(2) << "pad: " << input->ToString();
  }
  return input;
}

Status ReplicatedResourceUpdateElementwiseClustering::
    ValidateResourceUpdateAndClusters(
        const HloInstruction* ru,
        std::vector<ElementwiseCluster> clusters) const {
  auto partition_variables = GetResourceUpdatePartitionOffloadedVariables(ru);
  auto offload_variables = GetResourceUpdateOffloadVariables(ru);
  if (clusters.empty()) {
    if (partition_variables != THREESTATE_OFF) {
      std::string cause = "";
      if (offload_variables == THREESTATE_OFF) {
        cause =
            "This is because optimizer state is not being offloaded."
            " Offload optimizer state via offload_weight_update_variables";
      } else {
        cause =
            "This can occur when the optimizer contains operations that"
            " don't do the same thing on each replica, such as most non"
            " elementwise operations, dependencies or operations with side"
            " effects.";
      }
      LOG(WARNING)
          << "Failed to shard optimizer state across the replicas:"
          << " replicated_optimizer_state_sharding is on, but optimizer state"
          << " could not be sharded across replicas, as no suitable RTS"
          << " clusters could be found in the computation "
          << ru->to_apply()->name() << ". " << cause;
    }
  }
  return Status::OK();
}

ClusterOutlinePolicy
ReplicatedResourceUpdateElementwiseClustering::GetClusterOutlinePolicy(
    const ElementwiseCluster& cluster) const {
  if (!cluster.IsReplicaPartitioned()) {
    VLOG(2) << "Skipping outlining cluster with top "
            << cluster.GetTop()->name() << " as it is not replica partitioned.";
    return ClusterOutlinePolicy::Ignore;
  }

  return ClusterOutlinePolicy::Outline;
}

Status
ReplicatedResourceUpdateElementwiseClustering::UpdateClusterBackendConfig(
    const ElementwiseCluster& cluster,
    PoplarBackendConfig& backend_config) const {
  TF_RETURN_IF_ERROR(
      ResourceUpdateElementwiseClustering::UpdateClusterBackendConfig(
          cluster, backend_config));
  auto* call_config = backend_config.mutable_call_config();
  auto* function_config = call_config->mutable_function_config();
  // Setting partitioned_elementwise_cluster attribute indicates that we will
  // process those clusters differently:
  // - Remote buffer outlining pass will outline load/stores regardless if it's
  // unique or not.
  // - We may use different visitor for such clusters later.
  function_config->set_partitioned_elementwise_cluster(true);
  return Status::OK();
}

Status ReplicatedResourceUpdateElementwiseClustering::RunDataflowAnalysis(
    const HloModule* module) {
  return replica_identical_dataflow_analysis_.Run(module);
}

}  // namespace poplarplugin
}  // namespace xla
