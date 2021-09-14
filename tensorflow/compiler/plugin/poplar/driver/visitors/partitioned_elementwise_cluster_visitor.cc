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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/partitioned_elementwise_cluster_visitor.h"

#include <queue>
#include <set>
#include <utility>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/collective_reorder.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"

namespace xla {
namespace poplarplugin {

using ::tensorflow::str_util::Join;

PartitionedElementwiseClusterVisitor::PartitionedElementwiseClusterVisitor(
    int64 next_rearrangement_id, CompilerResources& res,
    const DeferredArgRBVectors& callsite_inputs,
    const poplar::DebugNameAndId& debug_name_and_id,
    bool allocate_all_input_tensors,
    const std::vector<const DeferredVisitor*>& dependent_computations,
    bool reallocate_inputs)
    : DeferredVisitor(res, callsite_inputs, debug_name_and_id,
                      allocate_all_input_tensors, dependent_computations,
                      reallocate_inputs),
      next_rearrangement_id_(next_rearrangement_id) {
  CHECK_GT(next_rearrangement_id_, 0);
}

PartitionedElementwiseClusterVisitor::PartitionedElementwiseClusterVisitor(
    int64 next_rearrangement_id, CompilerResources& res,
    const DeferredArgRBVectors& callsite_inputs,
    const poplar::DebugNameAndId& debug_name_and_id,
    bool allocate_all_input_tensors,
    const std::vector<const DeferredVisitor*>& dependent_computations,
    const ReallocateInputsInfo& reallocate_inputs_info)
    : DeferredVisitor(res, callsite_inputs, debug_name_and_id,
                      allocate_all_input_tensors, dependent_computations,
                      reallocate_inputs_info),
      next_rearrangement_id_(next_rearrangement_id) {
  CHECK_GT(next_rearrangement_id_, 0);
}

PartitionedElementwiseClusterVisitor::~PartitionedElementwiseClusterVisitor() {}

Status PartitionedElementwiseClusterVisitor::Preprocess(HloInstruction* inst) {
  TF_RETURN_IF_ERROR(DeferredVisitor::Preprocess(inst));
  VLOG(2) << "Preprocess " << inst->ToString();
  if (inst->opcode() == HloOpcode::kParameter) {
    // For each parameter instruction, try to map it to the entry computation
    // parameter. If it's really the entry computation parameter, put it into
    // entry_params_ map. Update remote buffer rearrangement and element numbers
    // for all new params if CBR instance is already created.
    if (!dfa_) {
      TF_ASSIGN_OR_RETURN(dfa_,
                          HloDataflowAnalysis::Run(*inst->parent()->parent()));
    }
    auto entry_param = GetRemoteBufferEntryParameterNumber(*dfa_, inst);
    if (entry_param.ok()) {
      int64 entry_param_idx = entry_param.ValueOrDie();
      VLOG(1) << "Parameter mapped to entry parameter " << entry_param_idx;
      TF_ASSIGN_OR_RETURN(bool updated,
                          UpdateRemoteBufferInformation(entry_param_idx, inst));
      if (!updated) {
        // CBR shape is not known yet, deferring update.
        entry_params_[entry_param_idx] = inst;
      }
    }
  }
  return Status::OK();
}

Status PartitionedElementwiseClusterVisitor::ValidateShape(
    HloInstruction* inst, std::size_t tuple_index, const Shape& shape,
    const TensorOrRemoteBuffer& out) {
  if (inst->opcode() == HloOpcode::kTuple) {
    return Status::OK();
  }

  VLOG(2) << "Validate shape of " << inst->ToString()
          << ", tensor/remote buffer: "
          << (out.IsTensor()
                  ? out.AsTensor().shapeToString()
                  : out.IsRemoteBuffer()
                        ? std::to_string(
                              out.AsRemoteBufferHolder().GetNumElements())
                        : "{invalid}")
          << ", XLA shape: " << shape;

  TF_ASSIGN_OR_RETURN(auto* cbr_info, GetCollectiveBalancedReorder(inst));
  if (!cbr_info) {
    return DeferredVisitor::ValidateShape(inst, tuple_index, shape, out);
  }
  auto* cbr = cbr_info->host_rearrangement.get();

  VLOG(3) << "Checking poplar types.";
  poplar::Type poplar_type;
  if (out.IsTensor()) {
    poplar_type = out.AsTensor().elementType();
  } else if (out.IsRemoteBuffer()) {
    auto& rbuffer_holder = out.AsRemoteBufferHolder();
    poplar_type = rbuffer_holder.GetElementType();
  } else {
    VLOG(3) << "Not a tensor or buffer, falling back to the DeferredVisitor "
               "validation.";
    return DeferredVisitor::ValidateShape(inst, tuple_index, shape, out);
  }

  TF_ASSIGN_OR_RETURN(poplar::Type expected_type, PoplarDataType(shape));
  if (expected_type != poplar_type) {
    return xla::InternalErrorStrCat(
        "Instruction ", inst->name(), " has mismatched Poplar (",
        poplar_type.toString().cloneAsString(), ") and XLA (",
        expected_type.toString().cloneAsString(), ") type",
        " for output tuple index ", tuple_index, ".");
  }
  VLOG(3) << "Validating against CBR.";

  const gcl::CollectiveBalancedHostRearrangement& host_rearrangement =
      cbr->getHostRearrangement();
  const int64 replicated_element_count =
      host_rearrangement.totalElementsPerReplica;
  const int64 non_replicated_element_count =
      replicated_element_count * host_rearrangement.replicationFactor;
  const int64 xla_element_count = ShapeUtil::ElementsIn(shape);
  VLOG(3) << "CBR slice element count: " << replicated_element_count
          << ", total collectives elements: " << non_replicated_element_count
          << ", XLA element count: " << xla_element_count;

  if (out.IsTensor()) {
    // Check shape.
    int64 element_count = out.AsTensor().numElements();
    VLOG(3) << "Validate tensor with " << element_count << " elements.";
    if (replicated_element_count == element_count ||
        non_replicated_element_count == element_count ||
        xla_element_count == element_count) {
      return Status::OK();
    }
  } else if (out.IsRemoteBuffer()) {
    auto& rbuffer_holder = out.AsRemoteBufferHolder();
    const auto merged_element_count =
        rbuffer_holder.GetNumElements() * rbuffer_holder.GetRepeats();
    CHECK_GT(out.NumMerged(), 0);
    CHECK_EQ(merged_element_count % out.NumMerged(), 0);
    const auto replication_factor = resources_.partition_replication_factor;
    CHECK_GT(replication_factor, 1);
    const auto element_count =
        merged_element_count / out.NumMerged() *
        (out.IsReplicaPartitioned() ? replication_factor : 1);

    VLOG(3) << "Validate remote buffer with " << element_count
            << " elements vs XLA shape with " << xla_element_count
            << " elements.";

    // Check shape of replicated case
    if (element_count >= xla_element_count) {
      return Status::OK();
    }
  }

  return xla::InternalErrorStrCat(
      "Instruction ", inst->name(), " has mismatched Poplar and XLA (",
      Join(shape.dimensions(), ","), ") shapes. ", __FUNCTION__, " ", __LINE__);
}

DeferredAllocateFunction
PartitionedElementwiseClusterVisitor::MakeParameterAllocationFunction(
    TensorLocation allocation_location, const Shape& shape,
    absl::optional<TensorOrRemoteBuffer> tensor_like,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return [this, shape, tensor_like,
          debug_name_and_id](TensorLocation allocation_location) mutable
         -> StatusOr<poplar::Tensor> {
    TF_ASSIGN_OR_RETURN(auto cbr_info, GetCollectiveBalancedReorder(
                                           allocation_location.instruction));
    if (!tensor_like && cbr_info) {
      // If it's parameter tensor without reference tensor provided,
      // Allocate it with CBR-friendly layout.
      auto* cbr = cbr_info->host_rearrangement.get();
      VLOG(3) << "Creating collective tensor with shape of " << shape;
      TF_ASSIGN_OR_RETURN(poplar::Type type, PoplarDataType(shape));
      auto element_count = ShapeUtil::ElementsIn(shape);
      auto& host_rearrangement = cbr->getHostRearrangement();
      if (host_rearrangement.totalElementsPerReplica == element_count) {
        tensor_like = cbr->createReplicaSlice(type);
      }
    }
    TF_ASSIGN_OR_RETURN(auto tensor,
                        AllocateInput(allocation_location, shape, tensor_like,
                                      debug_name_and_id));
    return tensor;
  };
}

Status PartitionedElementwiseClusterVisitor::SetRemoteBufferHostRearrangementId(
    poplar::Graph& graph, const HloComputation* entry_comp,
    int64 entry_param_idx, int64 host_rearrangement_id,
    int64 elements_per_replica) {
  auto& annotations = resources_.annotations;
  auto& remote_parameter_infos = annotations.remote_parameter_infos;
  auto old_info =
      remote_parameter_infos.find(RemoteParameterInfo(entry_param_idx));
  CHECK(old_info != remote_parameter_infos.end());

  if (old_info->host_rearrangement_id) {
    if (old_info->host_rearrangement_id == host_rearrangement_id) {
      return Status::OK();
    }
    return InternalErrorStrCat(
        "Collective rearrangement group conflict for entry parameter ",
        entry_param_idx);
  }
  const HloInstruction* param =
      entry_comp->parameter_instruction(entry_param_idx);

  RemoteParameterInfo info(
      old_info->parameter_number, old_info->is_replica_partitioned,
      old_info->buffer_name, old_info->buffer_offset, old_info->num_merged,
      old_info->merged_params, host_rearrangement_id);

  TF_ASSIGN_OR_RETURN(auto poplar_type, PoplarDataType(param->shape()));
  auto found_buffer = resources_.remote_buffers.find(info.buffer_name);
  if (found_buffer == resources_.remote_buffers.end()) {
    return InternalErrorStrCat("No remote buffer with handle ",
                               info.buffer_name);
  }
  TF_ASSIGN_OR_RETURN(
      auto rbuffer_holder,
      GetOrCreateRemoteBuffer(graph, resources_, info.buffer_name, poplar_type,
                              elements_per_replica,
                              /*num_repeats=*/1, info.num_merged));
  TF_RETURN_IF_ERROR(rbuffer_holder.AsRemoteBufferHolder().SetNumElements(
      elements_per_replica));

  remote_parameter_infos.erase(old_info);
  CHECK_EQ(remote_parameter_infos.insert(std::move(info)).second, true);

  return Status::OK();
}

StatusOr<bool>
PartitionedElementwiseClusterVisitor::UpdateRemoteBufferInformation(
    int64 entry_param_idx, const HloInstruction* entry_param) {
  TF_ASSIGN_OR_RETURN(auto cbr_info, GetCollectiveBalancedReorder(entry_param));
  if (!cbr_info) {
    VLOG(3) << "No CBR created, skipping.";
    return false;
  }
  auto* cbr = cbr_info->host_rearrangement.get();
  auto& graph = GetGraph(resources_, entry_param);
  auto& src = cbr->getHostRearrangement();
  auto& annotations = resources_.annotations;
  auto& remote_parameter_infos = annotations.remote_parameter_infos;
  auto& remote_parameter_host_rearrangements =
      annotations.remote_parameter_host_rearrangements;
  auto old_info =
      remote_parameter_infos.find(RemoteParameterInfo(entry_param_idx));
  if (old_info == remote_parameter_infos.end()) {
    VLOG(3) << "No remote parameter with index " << entry_param_idx << ".";
    return true;
  }
  if (!old_info->is_replica_partitioned) {
    VLOG(3) << "Remote parameter " << entry_param_idx
            << " is not replica parititoned.";
    return true;
  }

  if (remote_parameter_host_rearrangements.find(
          cbr_info->host_rearrangement_id) ==
      remote_parameter_host_rearrangements.end()) {
    RemoteParameterHostRearrangement host_rearrangement;
    host_rearrangement.replication_factor = src.replicationFactor;
    host_rearrangement.total_elements_per_replica = src.totalElementsPerReplica;
    for (auto& interval : src.gatheredToRefSlices) {
      host_rearrangement.gathered_to_ref_slice.emplace_back(interval.begin(),
                                                            interval.end());
    }
    host_rearrangement.element_map = src.elementMap;
    remote_parameter_host_rearrangements[cbr_info->host_rearrangement_id] =
        std::move(host_rearrangement);
  }

  // Propagate the same host rearrangement id for all merged remote buffers.
  // Copy indices, because old_info will be replaced by new one.
  const HloComputation* entry_comp =
      entry_param->parent()->parent()->entry_computation();
  auto merged_params = old_info->merged_params;
  if (merged_params.empty()) {
    // This buffer was not merged, set rearrangement only on entry_param_idx
    merged_params.push_back(entry_param_idx);
  }
  for (auto param_idx : merged_params) {
    TF_RETURN_IF_ERROR(SetRemoteBufferHostRearrangementId(
        graph, entry_comp, param_idx, cbr_info->host_rearrangement_id,
        src.totalElementsPerReplica));
  }

  return true;
}

Status PartitionedElementwiseClusterVisitor::UpdateRemoteBuffersInformation() {
  for (auto it = entry_params_.begin(); it != entry_params_.end();) {
    int64 entry_param_idx = it->first;
    VLOG(1) << "Adding rearrangement info for entry parameter "
            << entry_param_idx;

    const HloInstruction* entry_param = it->second;
    TF_ASSIGN_OR_RETURN(bool updated, UpdateRemoteBufferInformation(
                                          entry_param_idx, entry_param));
    if (updated) {
      it = entry_params_.erase(it);
    } else {
      ++it;
    }
  }
  return Status::OK();
}

Status PartitionedElementwiseClusterVisitor::FinishDeferedAllocationVisit(
    HloInstruction* inst) {
  VLOG(3) << "Finishing visit.";
  TF_RETURN_IF_ERROR(DeferredVisitor::FinishDeferedAllocationVisit(inst));

  dfa_.reset();

  TF_RETURN_IF_ERROR(UpdateRemoteBuffersInformation());

  return Status::OK();
}

StatusOr<const PartitionedElementwiseClusterVisitor::HostRearrangementInfo*>
PartitionedElementwiseClusterVisitor::GetCollectiveBalancedReorder(
    const HloInstruction* inst) {
  auto cbr_it = cbr_.find(inst->sharding_ptr());
  return cbr_it != cbr_.end() ? &cbr_it->second : nullptr;
}

Status PartitionedElementwiseClusterVisitor::SetCollectiveBalancedReorder(
    const HloInstruction* inst,
    std::unique_ptr<gcl::CollectiveBalancedReorder> cbr) {
  if (!cbr) {
    return InvalidArgument("CBR instance can't be null.");
  }
  HostRearrangementInfo info{std::move(cbr), next_rearrangement_id_++};
  cbr_[inst->sharding_ptr()] = std::move(info);
  return UpdateRemoteBuffersInformation();
}

}  // namespace poplarplugin
}  // namespace xla
