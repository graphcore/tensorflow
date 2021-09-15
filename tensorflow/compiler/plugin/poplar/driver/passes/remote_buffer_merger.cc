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

#include "tensorflow/compiler/plugin/poplar/driver/passes/remote_buffer_merger.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {

bool IsRemoteBufferLoad(const HloInstruction* inst) {
  return IsPoplarInstruction(RemoteParameterLoad, inst) ||
         IsPoplarInstruction(BufferLoadSlice, inst);
}

bool IsRemoteBufferStore(const HloInstruction* inst) {
  return IsPoplarInstruction(RemoteParameterStore, inst) ||
         IsPoplarInstruction(BufferStoreSlice, inst);
}

bool IsRemoteBufferUser(const HloInstruction* inst) {
  return IsRemoteBufferLoad(inst) || IsRemoteBufferStore(inst);
}

bool IsRemoteGradientAccumulatorCreate(const HloInstruction* inst) {
  return IsPoplarInstruction(GradientAccumulatorCreate, inst) &&
         Cast<HloGradientAccumulatorCreate>(inst)->IsRemote();
}

bool IsRemoteCreateBuffer(const HloInstruction* inst) {
  return IsPoplarInstruction(CreateBuffer, inst) &&
         Cast<HloCreateBuffer>(inst)->IsRemoteBuffer();
}

bool IsRemoteParameter(const HloInstruction* inst,
                       const CompilerAnnotations& annotations) {
  return IsInstructionInEntryComputation(inst) &&
         inst->opcode() == HloOpcode::kParameter &&
         ContainsKey(annotations.remote_parameter_infos,
                     RemoteParameterInfo(inst->parameter_number()));
}

bool IsReplicaPartitionedRemoteParameter(
    const HloInstruction* inst, const CompilerAnnotations& annotations) {
  if (!IsRemoteParameter(inst, annotations)) {
    return false;
  }

  auto found = annotations.remote_parameter_infos.find(
      RemoteParameterInfo(inst->parameter_number()));
  CHECK(found != annotations.remote_parameter_infos.end());
  return found->is_replica_partitioned;
}

bool IsRemoteBufferCreator(const HloInstruction* inst,
                           const CompilerAnnotations& annotations) {
  return IsRemoteParameter(inst, annotations) ||
         IsRemoteGradientAccumulatorCreate(inst) || IsRemoteCreateBuffer(inst);
}

bool IsGradientAccumulatorSink(const HloInstruction* inst) {
  return IsPoplarInstruction(GradientAccumulatorSink, inst);
}

bool IsReshape(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kReshape;
}

bool IsRemoteBufferPassthrough(const HloInstruction* inst) {
  return IsRemoteBufferStore(inst) || IsGradientAccumulatorSink(inst) ||
         IsReshape(inst);
}

using CreatorToUsers =
    HloInstructionMap<std::vector<HloAbstractRemoteLoadStore*>>;
using UserToCreator = std::map<HloAbstractRemoteLoadStore*, HloInstruction*>;

StatusOr<UserToCreator> FindRemoteBufferSources(
    const HloModule* module, const CompilerAnnotations& annotations) {
  UserToCreator result;

  TF_ASSIGN_OR_RETURN(auto dataflow_analysis,
                      HloDataflowAnalysis::Run(*module));

  for (auto* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (auto* inst : comp->MakeInstructionPostOrder()) {
      if (IsRemoteBufferUser(inst)) {
        auto* load_store_inst = Cast<HloAbstractRemoteLoadStore>(inst);

        // Find the source (creator) of the remote buffer used by this
        // instruction, traversing through instructions that merely pass
        // through the remote buffer.
        // TODO(T10387): Replace with our own alias analysis when we have it.
        HloInstruction* source = inst;
        do {
          CHECK_GT(source->operand_count(), 0) << source->ToString();
          const auto* buffer_operand = source->operand(0);
          const auto value_set =
              dataflow_analysis->GetFlattenedValueSet(buffer_operand);

          if (value_set.values().size() != 1) {
            return FailedPrecondition(
                "Failed to determine remote buffer source. Is complex control "
                "flow used to pass remote buffers? Found %d sources for: %s.",
                value_set.values().size(), source->ToString());
          }

          auto* next = value_set.values()[0]->instruction();
          CHECK_NE(next, source);
          source = next;
        } while (IsRemoteBufferPassthrough(source));

        if (!IsRemoteBufferCreator(source, annotations)) {
          return FailedPrecondition(
              "The source for %s does not create a remote buffer: %s",
              inst->ToString(), source->ToString());
        }

        if (load_store_inst->GetReplicationFactor(0) > 1 &&
            !IsReplicaPartitionedRemoteParameter(source, annotations)) {
          return FailedPrecondition(
              "The source for the replicated load/store %s is not a replica "
              "partitioned remote parameter: %s",
              load_store_inst->ToString(), source->ToString());
        }

        VLOG(2) << "Found remote buffer source for " << inst->ToString() << ": "
                << source->ToString();

        result[load_store_inst] = source;
      }
    }
  }

  return result;
}

HloInstruction* CreateSlicedVersion(HloAbstractRemoteLoadStore* inst,
                                    HloInstruction* offset_inst) {
  HloComputation* comp = inst->parent();
  CHECK_EQ(inst->RemoteBuffers().size(), 1) << inst->ToString();
  CHECK_EQ(inst->GetReplicationFactorCount(), 1) << inst->ToString();

  if (IsPoplarInstruction(RemoteParameterLoad)(inst)) {
    return comp->AddInstruction(
        CreateBufferLoadSlice(inst->shape(), inst->mutable_operand(0),
                              offset_inst, inst->GetReplicationFactor(0)));
  }

  CHECK(IsPoplarInstruction(RemoteParameterStore)(inst));
  return comp->AddInstruction(
      CreateBufferStoreSlice(inst->mutable_operand(0), inst->mutable_operand(1),
                             offset_inst, inst->GetReplicationFactor(0)));
}

Status HoistOffsets(HloInstruction* call,
                    const std::vector<HloInstruction*>& offsets) {
  HloComputation* comp = call->to_apply();
  HloComputation* call_parent = call->parent();
  HloCloneContext clone_context(comp->parent());

  auto builder = HloComputation::Builder(comp->name());

  const int64 num_parameters = comp->num_parameters();
  std::vector<HloInstruction*> hoisted_arguments(offsets.size());
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    auto* offset = offsets[i];

    // Add cloned offset argument in existing caller computation.
    auto* argument =
        call_parent->AddInstruction(offset->Clone("offset", &clone_context));
    hoisted_arguments[i] = argument;

    // Add receiving parameter in new called computation.
    auto* parameter = builder.AddInstruction(
        HloInstruction::CreateParameter(num_parameters + i, argument->shape(),
                                        "parameter." + argument->name()));
    argument->SetupDerivedInstruction(parameter);
    clone_context.MapInstruction(offset, parameter);
  }

  // Clone all the instructions except the hoisted offsets.
  for (auto* old_inst : comp->MakeInstructionPostOrder()) {
    if (absl::c_linear_search(offsets, old_inst)) {
      continue;
    }

    // Given the topological traversal order here, we can safely look up
    // operands and control predecessors in the clone context as they must
    // already have been traversed.

    // Find the new operands in the clone context.
    std::vector<HloInstruction*> new_operands(old_inst->operand_count());
    absl::c_transform(old_inst->operands(), new_operands.begin(),
                      [&clone_context](HloInstruction* old_operand) {
                        return clone_context.GetInstruction(old_operand);
                      });

    auto* new_inst = builder.AddInstruction(
        old_inst->CloneWithNewOperands(old_inst->shape(), new_operands));
    old_inst->SetupDerivedInstruction(new_inst);

    // Find the new control predecessors in the clone context.
    for (auto* old_control_pred : old_inst->control_predecessors()) {
      auto* new_control_pred = clone_context.GetInstruction(old_control_pred);
      new_control_pred->AddControlDependencyTo(new_inst);
    }

    clone_context.MapInstruction(old_inst, new_inst);
  }

  auto* new_root = clone_context.GetInstruction(comp->root_instruction());
  auto* new_comp =
      clone_context.module()->AddEmbeddedComputation(builder.Build(new_root));

  // Replace the call with the new call with new operands.
  std::vector<HloInstruction*> call_operands{call->operands().begin(),
                                             call->operands().end()};
  call_operands.insert(call_operands.end(), hoisted_arguments.begin(),
                       hoisted_arguments.end());
  auto* new_call = call_parent->AddInstruction(
      call->CloneWithNewOperands(new_root->shape(), call_operands));
  new_call->set_to_apply(new_comp);
  call->SetupDerivedInstruction(new_call);
  TF_RETURN_IF_ERROR(call->ReplaceAllUsesWithDifferentShape(new_call));
  TF_RETURN_IF_ERROR(new_call->CopyAllControlDepsFrom(call));
  TF_RETURN_IF_ERROR(call->DropAllControlDeps());
  TF_RETURN_IF_ERROR(call_parent->RemoveInstruction(call));

  return Status::OK();
}

StatusOr<bool> AddLoadStoreOffsets(
    CallGraph& call_graph, HloComputation* comp,
    const UserToCreator& user_to_creator,
    const ConstHloInstructionMap<int64>& creator_to_offset,
    bool is_inside_function = false) {
  bool changed = false;

  std::vector<HloInstruction*> offset_instructions;

  for (auto* inst : comp->MakeInstructionPostOrder()) {
    if (IsRemoteBufferUser(inst)) {
      auto* load_store_inst = Cast<HloAbstractRemoteLoadStore>(inst);
      auto* creator = user_to_creator.at(load_store_inst);
      auto found_offset = creator_to_offset.find(creator);
      if (found_offset == creator_to_offset.end()) {
        // No offset found; this is an unmerged buffer.
        continue;
      }

      const int64 offset = found_offset->second;

      if (IsPoplarInstruction(RemoteParameterLoad, inst) ||
          IsPoplarInstruction(RemoteParameterStore, inst)) {
        HloInstruction* offset_inst = MakeR0ConstantHlo<int32>(comp, offset);
        inst->SetupDerivedInstruction(offset_inst);
        if (inst->has_sharding()) {
          offset_inst->set_sharding(inst->sharding());
        }
        offset_instructions.push_back(offset_inst);

        auto* sliced_inst = CreateSlicedVersion(load_store_inst, offset_inst);
        TF_RETURN_IF_ERROR(sliced_inst->CopyAllControlDepsFrom(inst));
        TF_RETURN_IF_ERROR(inst->DropAllControlDeps());
        sliced_inst->set_raw_backend_config_string(
            inst->raw_backend_config_string());
        TF_RETURN_IF_ERROR(comp->ReplaceInstruction(inst, sliced_inst));
      } else {
        CHECK(IsPoplarInstruction(BufferLoadSlice, inst) ||
              IsPoplarInstruction(BufferStoreSlice, inst));
        // Already a sliced load/store. Adjust the existing offset to an
        // offset into the merged buffer in a "merge-major" order.
        const auto* buffer = inst->operand(0);
        CHECK(buffer->shape().IsArray());
        const int64 num_repeats = ShapeUtil::GetDimension(buffer->shape(), 0);
        const int64 merged_offset = num_repeats * offset;

        const int64 offset_operand_index =
            IsPoplarInstruction(BufferLoadSlice, inst) ? 1 : 2;

        auto* existing_offset = inst->mutable_operand(offset_operand_index);

        auto* offset_inst = MakeR0ConstantHlo<int32>(comp, merged_offset);
        existing_offset->SetupDerivedInstruction(offset_inst);
        offset_instructions.push_back(offset_inst);

        CHECK_EQ(existing_offset->shape(), offset_inst->shape());
        auto* adjusted_offset =
            comp->AddInstruction(HloInstruction::CreateBinary(
                existing_offset->shape(), HloOpcode::kAdd, existing_offset,
                offset_inst));
        existing_offset->SetupDerivedInstruction(adjusted_offset);

        TF_RETURN_IF_ERROR(
            inst->ReplaceOperandWith(offset_operand_index, adjusted_offset));
      }

      changed = true;
    }
  }

  if (is_inside_function && !offset_instructions.empty()) {
    const auto& callsites = call_graph.GetNode(comp).caller_callsites();
    if (callsites.size() != 1) {
      return FailedPrecondition(
          "Expected a unique caller, but found %d callers of: %s.",
          callsites.size(), comp->name());
    }

    auto* call = callsites[0].instruction();

    VLOG(2) << "Hoisting " << offset_instructions.size()
            << " offset instructions from " << comp->name() << " into "
            << call->parent()->name() << " through call: " << call->ToString();

    TF_RETURN_IF_ERROR(HoistOffsets(call, offset_instructions));
  }

  const auto& node = call_graph.GetNode(comp);

  for (const auto& callsite : node.callsites()) {
    auto* call = callsite.instruction();
    if (call->opcode() != HloOpcode::kCall) {
      // Skip fusions etc.
      continue;
    }

    const bool is_function_call = IsFunction(call);

    for (auto* called_comp : callsite.called_computations()) {
      TF_ASSIGN_OR_RETURN(
          bool called_comp_changed,
          AddLoadStoreOffsets(call_graph, called_comp, user_to_creator,
                              creator_to_offset, is_function_call));
      changed |= called_comp_changed;
    }
  }

  return changed;
}

struct RemoteBufferInfo {
  std::vector<int64> dimensions;
  PrimitiveType type;
  int64 num_repeats;
  int64 sharding_device;
  bool is_replica_partitioned;
};

struct RemoteBufferInfoCmp {
  bool operator()(const RemoteBufferInfo& a, const RemoteBufferInfo& b) const {
    return std::tie(a.dimensions, a.type, a.num_repeats, a.sharding_device,
                    a.is_replica_partitioned) <
           std::tie(b.dimensions, b.type, b.num_repeats, b.sharding_device,
                    b.is_replica_partitioned);
  }
};

using RemoteBufferCreators =
    std::map<RemoteBufferInfo, std::vector<HloInstruction*>,
             RemoteBufferInfoCmp>;

RemoteBufferCreators FindRemoteBufferCreators(
    const HloModule* module, const CompilerAnnotations& annotations) {
  RemoteBufferCreators remote_buffers;

  // Gather information from all instructions which create remote buffers.
  for (auto* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (auto* inst : comp->MakeInstructionPostOrder()) {
      if (IsRemoteBufferCreator(inst, annotations)) {
        const auto shape = inst->shape();
        CHECK(shape.IsArray());

        const int64 num_repeats =
            IsRemoteCreateBuffer(inst) ? ShapeUtil::GetDimension(shape, 0) : 1;

        const auto single_shape = IsRemoteCreateBuffer(inst)
                                      ? ShapeUtil::DeleteDimension(0, shape)
                                      : shape;
        const auto& dimensions = single_shape.dimensions();

        const int64 sharding_device = GetSingleShardingDeviceId(inst);

        const bool is_replica_partitioned =
            IsReplicaPartitionedRemoteParameter(inst, annotations);

        const auto key =
            RemoteBufferInfo{{dimensions.begin(), dimensions.end()},
                             single_shape.element_type(),
                             num_repeats,
                             sharding_device,
                             is_replica_partitioned};

        remote_buffers[key].push_back(inst);
      }
    }
  }

  return remote_buffers;
}

StatusOr<bool> IsInsideFunction(const HloInstruction* inst,
                                const CallGraph& call_graph) {
  auto* comp = inst->parent();
  const auto& callsites = call_graph.GetNode(comp).caller_callsites();
  if (callsites.empty()) {
    return false;
  }

  if (callsites.size() > 1) {
    return FailedPrecondition(
        "Expected flat call graph, but found %d callers of: %s.",
        callsites.size(), inst->parent()->name());
  }

  const auto* call = callsites[0].instruction();
  return IsFunction(call);
}

StatusOr<bool> IsInsideElementwiseCluster(const HloInstruction* inst,
                                          const CallGraph& call_graph) {
  auto* comp = inst->parent();
  const auto& callsites = call_graph.GetNode(comp).caller_callsites();
  if (callsites.empty()) {
    return false;
  }

  if (callsites.size() > 1) {
    return FailedPrecondition(
        "Expected flat call graph, but found %d callers of: %s.",
        callsites.size(), inst->parent()->name());
  }

  const auto* call = callsites[0].instruction();
  return GetFunctionPartitionedElementwiseCluster(call);
}

void InstructionPrinter(std::string* out, HloInstruction* inst) {
  out->append(inst->ToString());
}

using GroupedInstructions = std::vector<std::vector<HloInstruction*>>;

StatusOr<GroupedInstructions> ChooseCreatorsToMerge(
    const CompilerAnnotations& annotations,
    const RemoteBufferCreators& buffer_creators,
    const CreatorToUsers& creator_to_users, const CallGraph& call_graph,
    bool merge_all) {
  GroupedInstructions result;

  // The default heuristic (unless merge_all is true) is that a remote buffer
  // that is used inside a function would benefit from merging, as this could
  // enable re-use of the same compiled function (if the only thing that differs
  // between the calls is the remote buffer).
  auto would_benefit_from_merging = [&](HloInstruction* inst) {
    const auto is_inside_function = IsInsideFunction(inst, call_graph);
    TF_CHECK_OK(is_inside_function.status());
    return merge_all || is_inside_function.ValueOrDie();
  };

  for (auto& group : buffer_creators) {
    std::map<int64, std::vector<HloInstruction*>> to_merge;
    absl::flat_hash_map<const HloComputation*, int64, HloComputationHash,
                        HloComputationEquals>
        cluster_ids;

    // We attempt to merge remote buffers that have at least one user that
    // would benefit from merging.
    for (HloInstruction* creator : group.second) {
      auto found_users = creator_to_users.find(creator);
      if (found_users != creator_to_users.end() &&
          absl::c_any_of(found_users->second, would_benefit_from_merging)) {
        int64 cluster_id = -1;
        if (IsRemoteParameter(creator, annotations)) {
          for (HloInstruction* user : found_users->second) {
            auto* comp = user->parent();
            // Checking if remote parameter user is inside of elementwise
            // cluster rearranged with CBR. Number of the elements in the buffer
            // will not be available until CBR instance created. If remote
            // buffer merge candidate is rearranged, allow only buffers from the
            // identical clusters, so they will have the same number of the
            // elements. Provide partitioned cluster visitor with indicies of
            // the merged parameters, so it can adjust number of the elements
            // and host_rearrangement_id for all RemoteParameterInfo structures
            // affected.
            TF_ASSIGN_OR_RETURN(bool is_rearranged,
                                IsInsideElementwiseCluster(user, call_graph));
            if (is_rearranged) {
              // Expect that identical clusters will be merged later in
              // subcomputation graph caching code.
              if (!comp->HasSideEffect()) {
                auto it = cluster_ids.find(comp);
                if (it != cluster_ids.end()) {
                  CHECK(cluster_id == -1 || cluster_id == it->second)
                      << "The same cluster id is used across different "
                         "clusters.";
                  cluster_id = it->second;
                }
              }
              if (cluster_id == -1) {
                cluster_id = cluster_ids.size();
                cluster_ids.emplace(comp, cluster_id);
              }
            }
          }
        }
        to_merge[cluster_id].push_back(creator);
      }
    }

    for (auto& to_merge_group : to_merge) {
      // Do the actual merging if we got at least two of them.
      auto& insts = to_merge_group.second;
      if (insts.size() > 1) {
        VLOG(2) << "Merging remote buffers created by: "
                << absl::StrJoin(insts, ", ", InstructionPrinter);
        result.push_back(std::move(insts));
      }
    }
  }

  return result;
}

Status AddMergedInfo(const GroupedInstructions& creators_to_merge,
                     CompilerAnnotations& annotations) {
  for (const auto& creators : creators_to_merge) {
    CHECK_GT(creators.size(), 1);
    const auto buffer_name = GetDebugName(creators[0]) + "/merged";
    const auto num_merged = creators.size();

    VLOG(2) << "Merged " << num_merged << " remote buffers into one with name '"
            << buffer_name << "'";

    std::vector<int64> merged_params;
    for (const HloInstruction* inst : creators) {
      if (IsRemoteParameter(inst, annotations)) {
        merged_params.push_back(inst->parameter_number());
      }
    }

    for (std::size_t i = 0; i < creators.size(); ++i) {
      auto* inst = creators[i];
      const auto buffer_offset = i;

      if (IsPoplarInstruction(GradientAccumulatorCreate, inst) ||
          IsPoplarInstruction(CreateBuffer, inst)) {
        const auto merged_info =
            HloRemoteBufferInfo{buffer_name, num_merged, buffer_offset};

        auto unique_clone =
            IsPoplarInstruction(GradientAccumulatorCreate, inst)
                ? Cast<HloGradientAccumulatorCreate>(inst)
                      ->CloneWithRemoteBufferInfo(merged_info)
                : Cast<HloCreateBuffer>(inst)->CloneWithRemoteBufferInfo(
                      merged_info);

        auto* comp = inst->parent();
        auto* clone = comp->AddInstruction(std::move(unique_clone));
        TF_RETURN_IF_ERROR(clone->CopyAllControlDepsFrom(inst));
        TF_RETURN_IF_ERROR(inst->DropAllControlDeps());
        TF_RETURN_IF_ERROR(comp->ReplaceInstruction(inst, clone));

        VLOG(2) << "Creator " << i << " for " << buffer_name << ": "
                << clone->ToString();
      } else {
        CHECK(IsRemoteParameter(inst, annotations)) << inst->ToString();

        auto found_info = annotations.remote_parameter_infos.find(
            RemoteParameterInfo(inst->parameter_number()));
        CHECK(found_info != annotations.remote_parameter_infos.end());
        CHECK_EQ(found_info->host_rearrangement_id, 0);

        const auto merged_info = RemoteParameterInfo(
            found_info->parameter_number, found_info->is_replica_partitioned,
            buffer_name, buffer_offset, num_merged, merged_params);

        // Replace the existing info with the merged info.
        annotations.remote_parameter_infos.erase(found_info);
        CHECK(annotations.remote_parameter_infos.insert(merged_info).second);

        VLOG(2) << "Creator " << i << " for " << buffer_name << ": "
                << inst->ToString();
      }
    }
  }

  return Status::OK();
}

}  // namespace

StatusOr<bool> RemoteBufferMerger::Run(HloModule* module) {
  VLOG(3) << "RemoteBufferMerger mode: " << ThreeState_Name(mode_);
  if (mode_ == THREESTATE_OFF) {
    return false;
  }

  VLOG(3) << "Before RemoteBufferMerger:";
  XLA_VLOG_LINES(3, module->ToString());

  // Collect the remote buffer creators in groups compatible for merging.
  const RemoteBufferCreators grouped_creators =
      FindRemoteBufferCreators(module, annotations_);

  // Find the remote buffer source (the instruction creating the remote
  // buffer) for every remote buffer user.
  TF_ASSIGN_OR_RETURN(const auto user_to_creator,
                      FindRemoteBufferSources(module, annotations_));

  // Invert the above map.
  CreatorToUsers creator_to_users;
  for (auto& inst_creator : user_to_creator) {
    creator_to_users[inst_creator.second].push_back(inst_creator.first);
  }

  const auto call_graph = CallGraph::Build(module);

  // Choose the candidates that we are actually going to merge.
  const bool merge_all = mode_ == THREESTATE_ON;
  TF_ASSIGN_OR_RETURN(
      const GroupedInstructions creators_to_merge,
      ChooseCreatorsToMerge(annotations_, grouped_creators, creator_to_users,
                            *call_graph, merge_all));

  ConstHloInstructionMap<int64> creator_to_offset;
  for (const auto& creators : creators_to_merge) {
    for (std::size_t i = 0; i < creators.size(); ++i) {
      creator_to_offset[creators[i]] = i;
    }
  }

  // Make sure that the loads/stores are offset correctly into the merged remote
  // buffer. And try to add the offset tensors in the caller computation to
  // enable function re-use.
  TF_ASSIGN_OR_RETURN(
      bool changed,
      AddLoadStoreOffsets(*call_graph, module->entry_computation(),
                          user_to_creator, creator_to_offset));

  // Add merged remote buffer info to all the merged creators.
  TF_RETURN_IF_ERROR(AddMergedInfo(creators_to_merge, annotations_));

  if (changed) {
    TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());
    VLOG(3) << "After RemoteBufferMerger:";
    XLA_VLOG_LINES(3, module->ToString());
  } else {
    VLOG(2) << "No changes were made.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
