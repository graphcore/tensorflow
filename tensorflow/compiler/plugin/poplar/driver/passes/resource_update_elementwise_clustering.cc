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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/elementwise_cluster.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {
namespace {

bool IsAllReduce(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kAllReduce;
}

bool IsBroadcast(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kBroadcast;
}

struct ResourceUpdateElementwiseClusterValidator final
    : ElementwiseClusterValidator {
  Inputs valid_inputs;

  ResourceUpdateElementwiseClusterValidator(
      const HloComputation* comp, const std::function<bool(int64)>& filter)
      : valid_inputs(GetValidInputs(filter, comp)) {}

  bool IsValidInput(const HloInstruction* inst) const override {
    return valid_inputs.contains(inst);
  }
};

}  // namespace

Status ResourceUpdateElementwiseClustering::CloneInstruction(
    const Shape& shape, const HloInstruction* inst,
    HloComputation::Builder* builder, HloCloneContext* context) {
  std::vector<HloInstruction*> new_operands(inst->operand_count());
  absl::c_transform(inst->operands(), new_operands.begin(),
                    [context](HloInstruction* old_operand) {
                      return context->GetInstruction(old_operand);
                    });
  HloInstruction* new_inst = builder->AddInstruction(
      inst->CloneWithNewOperands(shape, new_operands, context));
  context->MapInstruction(inst, new_inst);
  return Status::OK();
}

std::unique_ptr<ElementwiseClusterValidator>
ResourceUpdateElementwiseClustering::CreateValidator(
    const HloComputation* comp,
    const std::function<bool(int64)>& allowed_resource_update_parameter) const {
  return absl::make_unique<ResourceUpdateElementwiseClusterValidator>(
      comp, allowed_resource_update_parameter);
}

std::unique_ptr<ElementwiseClusterValidator>
ResourceUpdateElementwiseClustering::CreateValidator(
    const HloInstruction* call, const HloInstruction* resource_update) const {
  return CreateValidator(resource_update->to_apply(),
                         [](int64) { return true; });
}

absl::flat_hash_set<const HloComputation*>
ResourceUpdateElementwiseClustering::GetElementwiseClusterableComputations(
    const HloModule* module) const {
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
    auto validator = CreateValidator(comp, [](int64) { return true; });
    CHECK(validator) << "Internal error: null validator";
    if (absl::c_all_of(comp->instructions(), [&elementwise_comps, &validator](
                                                 const HloInstruction* inst) {
          return ElementwiseCluster::CanCluster(inst, /*allow_inputs=*/true,
                                                elementwise_comps, *validator);
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
    const absl::flat_hash_set<const HloComputation*>& elementwise_comps) const {
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

  auto validator = CreateValidator(call, resource_update);
  return ElementwiseCluster::GetClustersIn(resource_update, elementwise_comps,
                                           *validator);
}

// Returns the instruction which should be the input to the outlined
// computation.
StatusOr<HloInstruction*>
ResourceUpdateElementwiseClustering::AddClusterInputToOutlinedComputation(
    int64 param_idx, const ElementwiseCluster& cluster,
    HloInstruction* cluster_input, HloComputation::Builder* builder,
    HloCloneContext* context) const {
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

  return AddClusterInput(param_idx, cluster, cluster_input, builder, context);
}

StatusOr<HloInstruction*> ResourceUpdateElementwiseClustering::AddClusterInput(
    int64 param_idx, const ElementwiseCluster& cluster,
    HloInstruction* cluster_input, HloComputation::Builder* builder,
    HloCloneContext* context) const {
  HloComputation* input_comp = cluster_input->parent();

  const Shape& cluster_input_shape = cluster_input->shape();

  VLOG(2) << "Adding cluster input " << cluster_input->ToString();
  CHECK_EQ(cluster.GetClusterSize(),
           ShapeUtil::ElementsIn(cluster_input_shape));

  // Lower the all reduce into the cluster if all its users will be in the
  // cluster too.
  const bool lower_all_reduce =
      IsAllReduce(cluster_input) && cluster.AllUsersIn(cluster_input);

  if (lower_all_reduce) {
    HloInstruction* input = cluster_input->mutable_operand(0);
    HloInstruction* parameter = builder->AddInstruction(
        HloInstruction::CreateParameter(param_idx, cluster_input->shape(),
                                        "parameter-reduce-" + input->name()));

    // Lower the allreduce in.
    HloInstruction* all_reduce = builder->AddInstruction(
        cluster_input->CloneWithNewOperands(cluster_input_shape, {parameter}));

    context->MapInstruction(cluster_input, all_reduce);
    return input;
  }
  VLOG(2) << "Adding cluster input " << cluster_input->ToString();

  HloInstruction* parameter = builder->AddInstruction(
      HloInstruction::CreateParameter(param_idx, cluster_input->shape(),
                                      "parameter-" + cluster_input->name()));

  VLOG(2) << "Parameter: " << parameter->ToString();
  context->MapInstruction(cluster_input, parameter);
  return cluster_input;
}

StatusOr<HloInstruction*> ResourceUpdateElementwiseClustering::AddClusterOutput(
    const ElementwiseCluster& cluster, HloInstruction* cluster_output,
    std::vector<UserPositions>& inst_users, HloComputation::Builder* builder,
    HloCloneContext* context) const {
  HloInstruction* in_cluster_output = context->GetInstruction(cluster_output);
  if (IsScalar(cluster_output)) {
    VLOG(2) << "Scalar output does not need rewriting: "
            << cluster_output->ToString();
    return in_cluster_output;
  }

  auto cluster_shape =
      cluster.GetClusterShape(in_cluster_output->shape().element_type());
  if (!ShapeUtil::Compatible(cluster_shape, in_cluster_output->shape())) {
    return builder->AddInstruction(
        HloInstruction::CreateReshape(cluster_shape, in_cluster_output));
  } else {
    return in_cluster_output;
  }
}

Status ResourceUpdateElementwiseClustering::AddClusterInstruction(
    const ElementwiseCluster& cluster, HloInstruction* inst,
    HloComputation::Builder* builder, HloCloneContext* context) const {
  return CloneInstruction(inst->shape(), inst, builder, context);
}

StatusOr<HloInstruction*> ResourceUpdateElementwiseClustering::OutlineCluster(
    ElementwiseCluster& cluster) const {
  VLOG(2) << "Rewriting cluster with top in " << cluster.GetTop()->ToString()
          << ", " << cluster.GetPostOrder().size() << " instructions.";

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
        AddClusterInputToOutlinedComputation(
            caller_inputs.size(), cluster, cluster_input, &builder, &context));
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

    TF_ASSIGN_OR_RETURN(HloInstruction * in_cluster_output,
                        AddClusterOutput(cluster, cluster_output, output_users,
                                         &builder, &context));

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
  TF_RETURN_IF_ERROR(UpdateClusterBackendConfig(cluster, backend_config));
  TF_RETURN_IF_ERROR(call->set_backend_config(backend_config));

  // Connect up all the users of the cluster output.
  int64 output_idx = 0;
  for (auto cluster_output : cluster.GetOutputs()) {
    TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                        MakeGetTupleElementHlo(call, output_idx++));

    for (auto user : computation_output_users.at(cluster_output)) {
      HloInstruction* to_replace_with = gte;
      VLOG(2) << "Replacing " << user.ToString();
      for (int64 index : user.indices) {
        TF_RETURN_IF_ERROR(
            user.instruction->ReplaceOperandWith(index, to_replace_with));
      }
    }
  }

  return call;
}

ClusterOutlinePolicy
ResourceUpdateElementwiseClustering::GetClusterOutlinePolicy(
    const ElementwiseCluster& cluster) const {
  return ClusterOutlinePolicy::OutlineNonUnique;
}

Status ResourceUpdateElementwiseClustering::UpdateClusterBackendConfig(
    const ElementwiseCluster& cluster,
    PoplarBackendConfig& backend_config) const {
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
  return Status::OK();
}

StatusOr<bool> ResourceUpdateElementwiseClustering::RewriteCall(
    HloModule* module, HloInstruction* call,
    const absl::flat_hash_set<const HloComputation*>& elementwise_comps) const {
  TF_ASSIGN_OR_RETURN(std::vector<ElementwiseCluster> clusters,
                      GetClustersIn(call, elementwise_comps));

  if (clusters.empty()) {
    VLOG(2) << "No clusters found.";
    return false;
  }

  std::vector<HloInstruction*> outlined_clusters;
  for (auto& cluster : clusters) {
    ClusterOutlinePolicy policy = GetClusterOutlinePolicy(cluster);
    if (policy == ClusterOutlinePolicy::Ignore) {
      VLOG(2) << "Skipping outlining cluster with top "
              << cluster.GetTop()->name();
      continue;
    }

    TF_ASSIGN_OR_RETURN(HloInstruction * call, OutlineCluster(cluster));
    if (policy == ClusterOutlinePolicy::OutlineNonUnique) {
      outlined_clusters.push_back(call);
    }
  }

  // Only outline the clusters which occur multiple times.
  HloInstructionSet non_unique_clusters;
  for (int64 i = 0; i != outlined_clusters.size(); ++i) {
    HloInstruction* i_call = outlined_clusters[i];
    HloComputation* i_comp = i_call->to_apply();
    for (int64 j = 0; j != i; ++j) {
      HloInstruction* j_call = outlined_clusters[j];
      HloComputation* j_comp = j_call->to_apply();
      if (HloComputationEquals()(i_comp, j_comp)) {
        non_unique_clusters.insert(i_call);
        non_unique_clusters.insert(j_call);
      }
    }
  }
  // Inline all the clusters which are unique.
  for (HloInstruction* call : outlined_clusters) {
    if (!ContainsKey(non_unique_clusters, call)) {
      TF_RETURN_IF_ERROR(CallInliner::Inline(call).status());
    }
  }
  return true;
}

StatusOr<bool> ResourceUpdateElementwiseClustering::Run(HloModule* module) {
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
    TF_ASSIGN_OR_RETURN(auto changed,
                        RewriteCall(module, call, elementwise_comps));
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
