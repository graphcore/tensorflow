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

#include "tensorflow/compiler/plugin/poplar/driver/passes/variables_offload_and_partition.h"

#include <algorithm>
#include <set>
#include <string>
#include <utility>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/all_gather.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/reduce_scatter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {

VariablesOffloadAndPartition::VariablesOffloadAndPartition(
    CompilerAnnotations& annotations, bool remote_memory_supported,
    int64 minimum_remote_tensor_size, int64 partition_replication_factor)
    : annotations_(annotations),
      remote_memory_supported_(remote_memory_supported),
      minimum_remote_tensor_size_(minimum_remote_tensor_size),
      partition_replication_factor_(partition_replication_factor) {}

namespace {
/**
 * Insert instruction sequence to load gather all the elements of the remote
 * tensor from the replicas it is partitioned across.
 *
 * We expect the resulting program to insert a fusion which looks like this
 * (assuming a partition_replication_factor of 4):
 *
 * // pretend this is the "true" shape ignoring partitioning
 * remote_buffer = param(0) : f32[3, 5, 7]
 *
 * // Load the replica-local region
 * a = load(remote_buffer) : f32[27]
 *
 * // gather from replicas
 * b = all-gather(a) : f32[4, 27]
 *
 * // slice off the padding, if needed
 * c = flatten(b) : f32[108]
 * d = slice(c), begin=0, end=105, dim=0 : f32[105]
 *
 * // reshape back to the "true" shape
 * e = reshape(d), shape=[3, 5, 7] : f32[3, 5, 7]
 *
 * This fusion will be either elided or inlined by other passes.
 *
 * @param computation The computation to add the instructions to.
 * @param parameter The parameter instruction to be loaded and gathered.
 * @param partition_replication_factor The number of replicas to partition
 * across.
 *
 * @returns The final instruction with the gathered values.
 *
 * @note `parameter` must be an element of `computation`.
 */
StatusOr<HloInstruction*> InsertReplicatedLoadInstructions(
    HloComputation* computation, HloInstruction* parameter,
    int64 partition_replication_factor) {
  HloInstruction* load =
      computation->AddInstruction(CreateHloRemoteParameterLoad(
          {parameter}, {partition_replication_factor}));

  HloInstruction* replicated_load = load;
  if (partition_replication_factor > 1) {
    HloModule* module = computation->parent();
    HloComputation::Builder builder(GetReplicatedParameterLoadFusionName());

    HloInstruction* load_parameter =
        builder.AddInstruction(HloInstruction::CreateParameter(
            0, load->shape(), "parameter_" + load->name()));

    PrimitiveType element_type = load->shape().element_type();
    // All-gather from the replicas.
    Shape all_gather_shape = ShapeUtil::MakeShape(
        element_type,
        {partition_replication_factor, ShapeUtil::ElementsIn(load->shape())});
    HloInstruction* all_gather = builder.AddInstruction(CreatePoplarAllGather(
        {load_parameter}, all_gather_shape,
        PoplarReplicaGroups::Consecutive(partition_replication_factor)));

    HloInstruction* output;
    // If there's no padding, just reshape the tensor.
    if (ShapeUtil::ElementsIn(all_gather_shape) ==
        ShapeUtil::ElementsIn(load->operand(0)->shape())) {
      // Reshape back to the user shape.
      output = builder.AddInstruction(
          HloInstruction::CreateReshape(load->operand(0)->shape(), all_gather));
    } else {
      // Flatten the tensor.
      Shape flat_shape = ShapeUtil::MakeShape(
          element_type, {ShapeUtil::ElementsIn(all_gather->shape())});
      HloInstruction* reshape = builder.AddInstruction(
          HloInstruction::CreateReshape(flat_shape, all_gather));

      // Slice off the padding.
      Shape slice_shape = ShapeUtil::MakeShape(
          element_type, {ShapeUtil::ElementsIn(load->operand(0)->shape())});
      HloInstruction* slice =
          builder.AddInstruction(HloInstruction::CreateSlice(
              slice_shape, reshape, {0},
              {ShapeUtil::ElementsIn(load->operand(0)->shape())}, {1}));
      // Reshape back to the user shape.
      output = builder.AddInstruction(
          HloInstruction::CreateReshape(load->operand(0)->shape(), slice));
    }

    // Build the fusion.
    HloComputation* fusion_comp =
        module->AddEmbeddedComputation(builder.Build(output));
    replicated_load = computation->AddInstruction(HloInstruction::CreateFusion(
        output->shape(), HloInstruction::FusionKind::kCustom, {load},
        fusion_comp));
  }

  // Replace all users of the input, except load instructions.
  auto users = parameter->users();
  users.erase(absl::c_find(users, load));

  for (auto user : users) {
    TF_RETURN_IF_ERROR(parameter->ReplaceUseWith(user, replicated_load));
  }

  return replicated_load;
}

/**
 * Insert instruction sequence to select the elements for storing in the
 * replica-local remote buffer.
 *
 * @param computation The computation to add the instructions to.
 * @param remote_buffer The remote buffer instruction that we would like to
 * store a value in.
 * @param to_store The to_store instruction that we would like to store.
 * @param partition_replication_factor The number of replicas to partition
 * across.
 *
 * @returns The final instruction with the values to be stored.
 *
 * @note `to_store` must be an element of `computation`.
 */

StatusOr<HloInstruction*> InsertReplicatedStoreInstructions(
    HloComputation* computation, HloInstruction* remote_buffer,
    HloInstruction* to_store, int64 partition_replication_factor) {
  if (partition_replication_factor > 1) {
    HloModule* module = computation->parent();
    HloComputation::Builder builder(GetReplicatedParameterStoreFusionName());

    HloInstruction* to_store_parameter =
        builder.AddInstruction(HloInstruction::CreateParameter(
            0, to_store->shape(), "parameter_" + to_store->name()));

    const PrimitiveType element_type = to_store->shape().element_type();

    const int64 element_count = PartitionedElementCountPerReplica(
        ShapeUtil::ElementsIn(to_store->shape()), partition_replication_factor);

    HloInstruction* output = to_store_parameter;
    // Add padding, if it is needed
    if ((ShapeUtil::ElementsIn(to_store->shape()) %
         partition_replication_factor) != 0) {
      HloInstruction* zero_f = builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(element_type)));

      // Flatten the incoming tensor
      Shape flat_shape = ShapeUtil::MakeShape(
          element_type, {ShapeUtil::ElementsIn(to_store->shape())});
      HloInstruction* flat = builder.AddInstruction(
          HloInstruction::CreateReshape(flat_shape, to_store_parameter));

      // Pad the tensor to be a multiple of `partition_replication_factor`.
      Shape pad_shape = ShapeUtil::MakeShape(
          element_type, {element_count * partition_replication_factor});

      PaddingConfig padding_config;
      std::size_t difference = ShapeUtil::ElementsIn(pad_shape) -
                               ShapeUtil::ElementsIn(flat->shape());
      auto padding_config_dim = padding_config.add_dimensions();
      padding_config_dim->set_edge_padding_high(difference);
      padding_config_dim->set_edge_padding_low(0);
      padding_config_dim->set_interior_padding(0);

      output = builder.AddInstruction(
          HloInstruction::CreatePad(pad_shape, flat, zero_f, padding_config));
    }

    Shape store_shape = ShapeUtil::MakeShape(
        element_type, {partition_replication_factor * element_count});
    if (!ShapeUtil::Compatible(output->shape(), store_shape)) {
      output = builder.AddInstruction(
          HloInstruction::CreateReshape(store_shape, output));
    }
    // Slice off this replica's storage elements.

    // reduce-scatter(op=LOCAL) slices replica-specific shard of the collective
    // input tensor and does nothing with the values. This is more memory/cycles
    // efficient than doing dynamic-slice(replication-index() * shard_size).
    Shape slice_shape = ShapeUtil::MakeShape(element_type, {element_count});
    output = builder.AddInstruction(CreateReduceScatter(
        slice_shape, {output}, CollectiveOperator::COLLECTIVE_OP_LOCAL,
        PoplarReplicaGroups::Consecutive(partition_replication_factor)));

    // Build the fusion.
    HloComputation* fusion_comp =
        module->AddEmbeddedComputation(builder.Build(output));
    to_store = computation->AddInstruction(HloInstruction::CreateFusion(
        output->shape(), HloInstruction::FusionKind::kCustom, {to_store},
        fusion_comp));
  }

  HloInstruction* remote_store =
      computation->AddInstruction(CreateHloRemoteParameterStore(
          {remote_buffer, to_store}, {partition_replication_factor}));
  return remote_store;
}

// Stores the information about a use of a resource variable within a pipeline
// stage/resource update.
struct OffloadedResourceUse {
  enum class Type { ReadOnly, ReadWrite };
  Type type;
  // The actual use of a resource.
  HloInstruction* user;
  // Parameter instruction inside of the computation of the user which
  // represents the read of the user.
  HloInstruction* user_parameter;
  // Only populated for ResourceUpdateReadWrite. Output instruction
  // which represents the use updating the value.
  HloInstruction* user_output;
};

// Stores information about a resource input and its uses.
struct OffloadedResourceInfo {
  enum class Type { NotModified, Modified };
  Type type;
  // Index of this resource in the entry computation.
  int64 entry_param_number;
  // Operand index of the parameter in call.
  int64 call_operand_idx;
  // Input to the call in the entry computation.
  HloInstruction* input_to_call;
  // Input inside of the call computation.
  HloInstruction* input_in_call;
  // Users inside of the call of the resource variable.
  std::vector<OffloadedResourceUse> users;
  // The stream name for this input.
  std::string stream_name;

  // Only populated for Modified type. Output from the call in entry
  // computation.
  HloInstruction* output_from_call;
  // Only populated for Modified type. The user which modifies the resource
  // input.
  OffloadedResourceUse modifying_user;
  // Whether to offload this resource.
  ThreeState offload;
  // Whether to replica partition this resource.
  bool replica_partition;
};
}  // namespace

StatusOr<ThreeState> VariablesOffloadAndPartition::ShouldOffloadInPipeline(
    HloInstruction* const pipeline_op) {
  switch (GetPipelineOffloadVariables(pipeline_op)) {
    case THREESTATE_OFF: {
      return THREESTATE_OFF;
    }
    case THREESTATE_ON: {
      return THREESTATE_ON;
    }
    case THREESTATE_UNDEFINED: {
      // Only offload by default when using batch serialization with sequential
      // schedule.
      const int64 batch_serialization_iterations =
          GetPipelineBatchSerializationIterations(pipeline_op);
      TF_ASSIGN_OR_RETURN(const auto schedule,
                          GetPipelineSchedule(pipeline_op));
      if (batch_serialization_iterations > 1 &&
          schedule ==
              PoplarBackendConfig::CallConfig::PipelineConfig::Sequential) {
        return THREESTATE_UNDEFINED;
      }
      return THREESTATE_OFF;
    }
    default: { return FailedPrecondition("Unknown state."); }
  }
}

StatusOr<bool> VariablesOffloadAndPartition::ShouldPartitionInPipeline(
    HloInstruction* const pipeline_op) {
  switch (GetPipelinePartitionVariables(pipeline_op)) {
    case THREESTATE_OFF: {
      return false;
    }
    case THREESTATE_ON: {
      return true;
    }
    case THREESTATE_UNDEFINED: {
      // Don't try to partition if there is no remote memory.
      if (!remote_memory_supported_) {
        return false;
      }

      const int64 batch_serialization_iterations =
          GetPipelineBatchSerializationIterations(pipeline_op);
      TF_ASSIGN_OR_RETURN(const auto schedule,
                          GetPipelineSchedule(pipeline_op));
      if (batch_serialization_iterations > 1 &&
          schedule ==
              PoplarBackendConfig::CallConfig::PipelineConfig::Sequential) {
        return true;
      }
      return false;
    }
    default: { return FailedPrecondition("Unknown state."); }
  }
}

StatusOr<bool> VariablesOffloadAndPartition::Optimize(HloInstruction* call_op) {
  // Find how many resource updates there are - for pipeliening there can be at
  // most one, for loops there has to be one.
  const int64 num_resource_updates =
      absl::c_count_if(call_op->to_apply()->instructions(), IsResourceUpdate);

  // For repeat loops there always needs to be a resource update to do
  // offloading.
  if (num_resource_updates == 0 && IsRepeatLoop(call_op)) {
    return false;
  } else if (num_resource_updates > 1) {
    return FailedPrecondition(
        "Detected multiple resource update instructions.");
  }
  bool changed = false;

  // Do not optimize if this is not a op inside an entry computation.
  if (call_op->parent() != call_op->GetModule()->entry_computation()) {
    return changed;
  }

  HloComputation* entry_comp = call_op->GetModule()->entry_computation();
  HloComputation* call_comp = call_op->to_apply();

  // Make sure that the root of the call op is a tuple instruction.
  {
    TF_ASSIGN_OR_RETURN(bool changed_call, FixRootInstruction(call_comp));
    changed |= changed_call;
  }

  if (call_op == entry_comp->root_instruction()) {
    // Make sure this is not the root instruction.
    TF_RETURN_IF_ERROR(FixRootInstruction(entry_comp).status());
    changed = true;
  }
  HloInstruction* entry_root = entry_comp->root_instruction();

  // Make sure all call op users are unique GTEs.
  absl::flat_hash_map<int64, HloInstruction*> call_gte_users;
  for (HloInstruction* output : call_op->users()) {
    if (output->opcode() != HloOpcode::kGetTupleElement) {
      return changed;
    }
    // Make sure the GTE is unique.
    const int64 tuple_index = output->tuple_index();
    if (call_gte_users.contains(tuple_index)) {
      return changed;
    }
    call_gte_users[tuple_index] = output;
  }

  // We cannot optimize if the root of the entry computation is not a tuple.
  if (entry_root->opcode() != HloOpcode::kTuple) {
    return changed;
  }

  // We cannot optimize if the output tuple has users - this implies that the
  // parameter has users other than the call.
  if (entry_root->user_count()) {
    return changed;
  }

  HloInstruction* call_root = call_comp->root_instruction();
  CHECK_EQ(call_root->opcode(), HloOpcode::kTuple);

  std::vector<OffloadedResourceInfo> offload_infos;
  for (int64 call_operand_idx = 0; call_operand_idx != call_op->operand_count();
       ++call_operand_idx) {
    HloInstruction* call_input = call_op->mutable_operand(call_operand_idx);
    HloInstruction* call_parameter =
        call_comp->parameter_instruction(call_operand_idx);
    if (call_parameter->shape().IsTuple()) {
      continue;
    }

    // Check whether the input to the call operations at this index is
    // a parameter - i.e. this is a parameter in the entry computation.
    if (call_input->opcode() != HloOpcode::kParameter) {
      continue;
    }
    const int64 entry_param_number = call_input->parameter_number();

    // Also expect the call input to be unique.
    if (call_input->user_count() != 1 ||
        call_op->OperandIndices(call_input).size() != 1) {
      continue;
    }

    // Keep track of all the users.
    std::vector<HloInstruction*> users = call_parameter->users();
    // Do not proceed if this input is not unique in any of its users.
    if (absl::c_any_of(users, [call_parameter](const HloInstruction* user) {
          return user->OperandIndices(call_parameter).size() != 1;
        })) {
      continue;
    }

    const auto& input_info =
        annotations_.input_output_aliasing_map.GetEntryInputInfos().at(
            entry_param_number);
    OffloadedResourceInfo offload_info;
    offload_info.entry_param_number = entry_param_number;
    offload_info.call_operand_idx = call_operand_idx;
    offload_info.input_to_call = call_input;
    offload_info.input_in_call = call_parameter;
    offload_info.stream_name = input_info.Handles().at(0);

    if (input_info.IsResourceNotModified()) {
      // Needs to be used by the root tuple at the same index as the operand
      // index (i.e. unmodified).
      // Find and remove root tuple from users being tracked.
      auto it = absl::c_find(users, call_root);
      if (it == users.end()) {
        continue;
      }
      users.erase(it);
      if (call_root->mutable_operand(call_operand_idx) != call_parameter) {
        continue;
      }
      // The output from the call cannot have any users.
      if (call_gte_users.contains(call_operand_idx)) {
        continue;
      }
      if (users.empty()) {
        continue;
      }
      offload_info.type = OffloadedResourceInfo::Type::NotModified;
    } else if (input_info.IsResource() && !input_info.IsResourceNotModified()) {
      // Needs to be used by the resource update with its value returned and the
      // call output at index `call_operand_idx` is an output from a resource
      // update via a GTE (i.e. the value was updated inside the resource
      // update).

      // Find a resource update amongst the users.
      std::vector<HloInstruction*> resource_updates;
      absl::c_copy_if(users, std::back_inserter(resource_updates),
                      IsResourceUpdate);
      if (resource_updates.size() != 1) {
        continue;
      }
      HloInstruction* resource_update = resource_updates[0];

      // Make sure that the root of the resource update is a tuple instruction.
      {
        TF_ASSIGN_OR_RETURN(bool changed_ru,
                            FixRootInstruction(resource_update->to_apply()));
        changed |= changed_ru;
      }

      // Find and remove resource update from users being tracked.
      auto it = absl::c_find(users, resource_update);
      if (it == users.end()) {
        continue;
      }
      users.erase(it);

      const int64 resource_update_input_index =
          resource_update->operand_index(call_parameter);
      HloInstruction* resource_update_output =
          call_root->mutable_operand(call_operand_idx);
      if (resource_update_output->opcode() != HloOpcode::kGetTupleElement) {
        continue;
      }
      if (resource_update_output->operand(0) != resource_update) {
        continue;
      }

      // Check that the output from the call is only used by the root
      // instruction.
      if (!call_gte_users.contains(call_operand_idx)) {
        continue;
      }
      HloInstruction* output_from_call = call_gte_users.at(call_operand_idx);
      if (output_from_call->user_count() != 1 ||
          output_from_call->users()[0] != entry_root) {
        continue;
      }
      // It is only used once.
      auto output_indices = entry_root->OperandIndices(output_from_call);
      if (output_indices.size() != 1) {
        continue;
      }

      // Only offload if the input is aliasing the output.
      const int64 entry_output_idx = output_indices[0];
      const auto& output_info =
          annotations_.input_output_aliasing_map.GetEntryOutputInfos().at(
              entry_output_idx);
      if (output_info.GetInputIndex() != entry_param_number) {
        continue;
      }
      HloComputation* resource_update_comp = resource_update->to_apply();
      HloInstruction* resource_update_root =
          resource_update_comp->root_instruction();
      CHECK_EQ(resource_update_root->opcode(), HloOpcode::kTuple);
      HloInstruction* output_in_resource_update =
          resource_update_root->mutable_operand(
              resource_update_output->tuple_index());

      OffloadedResourceUse resource_use;
      resource_use.type = OffloadedResourceUse::Type::ReadWrite;
      resource_use.user = resource_update;
      resource_use.user_parameter = resource_update_comp->parameter_instruction(
          resource_update_input_index);
      resource_use.user_output = output_in_resource_update;

      offload_info.type = OffloadedResourceInfo::Type::Modified;
      offload_info.output_from_call = output_from_call;
      offload_info.modifying_user = resource_use;
      offload_info.offload = GetResourceUpdateOffloadVariables(resource_update);
      offload_info.replica_partition =
          GetResourceUpdatePartitionOffloadedVariables(resource_update) !=
          THREESTATE_OFF;
    } else {
      continue;
    }

    // Sort all the users so that (any) resource update is first - this is to
    // make sure offloading and replication partioning information is applied in
    // deterministic order to the offload info.
    absl::c_sort(users, [](const HloInstruction* a, const HloInstruction* b) {
      const bool a_resource_update = IsResourceUpdate(a);
      const bool b_resource_update = IsResourceUpdate(b);
      if (a_resource_update == b_resource_update) {
        return HloPtrComparator()(a, b);
      }
      return a_resource_update;
    });

    // Go through remaining users - check whether they are all supported for
    // offloading. They all need to be read only.
    bool all_users_supported = true;
    for (HloInstruction* user : users) {
      if (IsResourceUpdate(user)) {
        OffloadedResourceUse resource_use;
        resource_use.type = OffloadedResourceUse::Type::ReadOnly;
        resource_use.user = user;
        resource_use.user_parameter = user->to_apply()->parameter_instruction(
            user->operand_index(call_parameter));
        offload_info.users.push_back(resource_use);
        offload_info.offload = GetResourceUpdateOffloadVariables(user);
        offload_info.replica_partition =
            GetResourceUpdatePartitionOffloadedVariables(user) !=
            THREESTATE_OFF;

      } else if (IsAnyPipelineStageOp(user)) {
        CHECK(IsPipelineOp(call_op));
        OffloadedResourceUse resource_use;
        resource_use.type = OffloadedResourceUse::Type::ReadOnly;
        resource_use.user = user;
        resource_use.user_parameter = user->to_apply()->parameter_instruction(
            user->operand_index(call_parameter));
        offload_info.users.push_back(resource_use);
        TF_ASSIGN_OR_RETURN(offload_info.offload,
                            ShouldOffloadInPipeline(call_op));
        TF_ASSIGN_OR_RETURN(offload_info.replica_partition,
                            ShouldPartitionInPipeline(call_op));
      } else if (IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(
                     user)) {
        // Nothing to do - gradient accumulator create might try and use the
        // layout of a variable for allocation.
      } else {
        all_users_supported = false;
      }
    }

    if (!all_users_supported) {
      continue;
    }

    offload_infos.push_back(offload_info);
  }

  VLOG(1) << "Minimum remote tensor size: " << minimum_remote_tensor_size_
          << ", replication factor: " << partition_replication_factor_;
  // For each parameter in offload_info, insert any required load and store
  // operations.
  for (auto& offload_info : offload_infos) {
    if (offload_info.replica_partition &&
        offload_info.offload == THREESTATE_OFF) {
      return UnimplementedStrCat("Requested replicated weight sharding for ",
                                 call_op->ToString(),
                                 " without offloading. This is currently not "
                                 "supported.");
    }

    if (offload_info.offload == THREESTATE_OFF) {
      continue;
    }

    if (!remote_memory_supported_) {
      const std::string message =
          "Current configuration of the IPU devices does not support remote "
          "memory and therefore weight update only variables cannot be "
          "offloaded to remote memory."
          "Consider configuring the IPU system with "
          "'IPUConfig.device_connection.enable_remote_buffers' set to True "
          "if remote memory is supported.";
      switch (offload_info.offload) {
        case THREESTATE_OFF: {
          break;
        }
        case THREESTATE_ON: {
          return FailedPrecondition("%s", message.c_str());
        }
        case THREESTATE_UNDEFINED: {
          VLOG(1) << message;
          break;
        }
        default: { return FailedPrecondition("Unknown case"); }
      }
      continue;
    }

    const std::size_t partition_replication_factor =
        offload_info.replica_partition ? partition_replication_factor_ : 1;

    const std::size_t byte_size =
        ShapeUtil::ByteSizeOf(offload_info.input_to_call->shape());
    if (minimum_remote_tensor_size_ * partition_replication_factor >
        byte_size) {
      VLOG(1) << "Variable " << offload_info.entry_param_number << ": "
              << offload_info.input_to_call->ToString()
              << " is smaller than the minimum remote tensor size ("
              << minimum_remote_tensor_size_
              << "B) and is therefore not offloaded.";
      continue;
    }

    VLOG(1) << "Offloading variable " << offload_info.entry_param_number << ": "
            << offload_info.input_to_call->ToString()
            << ", byte size: " << byte_size;

    if (offload_info.type == OffloadedResourceInfo::Type::Modified) {
      auto& modifying_user = offload_info.modifying_user;
      HloComputation* resource_update_comp = modifying_user.user->to_apply();
      CHECK(modifying_user.type == OffloadedResourceUse::Type::ReadWrite);
      // Insert the load and stores inside of the resource update.
      TF_ASSIGN_OR_RETURN(
          HloInstruction * remote_load,
          InsertReplicatedLoadInstructions(resource_update_comp,
                                           modifying_user.user_parameter,
                                           partition_replication_factor));

      // If the input and output are the same instruction, then we use the
      // remote resource but don't modify it. In this case we simply reconnect
      // the input remote buffer to the original output position.
      if (modifying_user.user_parameter == modifying_user.user_output) {
        TF_RETURN_IF_ERROR(remote_load->ReplaceUseWith(
            resource_update_comp->root_instruction(),
            modifying_user.user_parameter));
      } else {
        // Add a remote store for the updated value inside the resource update.
        TF_ASSIGN_OR_RETURN(
            HloInstruction * remote_store,
            InsertReplicatedStoreInstructions(
                resource_update_comp, modifying_user.user_parameter,
                modifying_user.user_output, partition_replication_factor));

        TF_RETURN_IF_ERROR(modifying_user.user_output->ReplaceUseWith(
            resource_update_comp->root_instruction(), remote_store));
      }
    }
    for (OffloadedResourceUse& user : offload_info.users) {
      // All the other users are read-only.
      CHECK(user.type == OffloadedResourceUse::Type::ReadOnly);
      TF_RETURN_IF_ERROR(InsertReplicatedLoadInstructions(
                             user.user->to_apply(), user.user_parameter,
                             partition_replication_factor)
                             .status());
    }
    // Mark this input as being stored in a remote buffer.
    annotations_.remote_parameter_infos.insert(RemoteParameterInfo{
        offload_info.entry_param_number, partition_replication_factor > 1,
        offload_info.stream_name, /*buffer_offset=*/0, /*num_merged=*/1});
    changed = true;
  }

  return changed;
}

StatusOr<bool> VariablesOffloadAndPartition::Run(HloModule* module) {
  VLOG(2) << "Before VariablesOffloadAndPartition:";
  XLA_VLOG_LINES(2, module->ToString());
  bool changed = false;
  std::vector<HloInstruction*> to_optimize;
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (IsRepeatLoop(inst) || IsPipelineOp(inst)) {
        to_optimize.push_back(inst);
      }
    }
  }

  for (HloInstruction* inst : to_optimize) {
    TF_ASSIGN_OR_RETURN(bool optimized, Optimize(inst));
    changed |= optimized;
  }

  if (changed) {
    VLOG(2) << "After VariablesOffloadAndPartition:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
