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

#include "tensorflow/compiler/plugin/poplar/driver/passes/resource_update_variables_offload.h"

#include <algorithm>
#include <set>
#include <utility>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
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

ResourceUpdateVariablesOffload::ResourceUpdateVariablesOffload(
    CompilerAnnotations& annotations, bool remote_memory_supported,
    int64 minimum_remote_tensor_size)
    : annotations_(annotations),
      remote_memory_supported_(remote_memory_supported),
      minimum_remote_tensor_size_(minimum_remote_tensor_size) {}

StatusOr<bool> ResourceUpdateVariablesOffload::Optimize(
    HloInstruction* call_op, HloInstruction* resource_update) {
  bool changed = false;

  // Do not optimize if this is not a op inside an entry computation.
  if (call_op->parent() != call_op->GetModule()->entry_computation()) {
    return changed;
  }
  HloComputation* entry_comp = call_op->GetModule()->entry_computation();
  HloComputation* call_comp = call_op->to_apply();
  HloComputation* resource_update_comp = resource_update->to_apply();

  // Make sure that the root of resource update and the call op is a tuple
  // instruction.
  {
    TF_ASSIGN_OR_RETURN(bool changed_ru,
                        FixRootInstruction(resource_update->to_apply()));
    TF_ASSIGN_OR_RETURN(bool changed_call, FixRootInstruction(call_comp));
    changed |= changed_ru || changed_call;
  }

  // Do not optimize if there is no resource update or pipeline wu variable
  // offloading is turned off.
  if (!GetResourceUpdateOffloadVariables(resource_update)) {
    return changed;
  }

  if (call_op == entry_comp->root_instruction()) {
    // Make sure this is not the root instruction.
    TF_RETURN_IF_ERROR(FixRootInstruction(entry_comp).status());
    changed = true;
  }

  // We cannot optimize if the root of the entry computation is not a tuple.
  if (entry_comp->root_instruction()->opcode() != HloOpcode::kTuple) {
    return changed;
  }

  // We cannot optimize if the output tuple has users - this implies that the
  // parameter has users other than the call.
  if (entry_comp->root_instruction()->user_count()) {
    return changed;
  }

  // Find any parameter instructions which can be offloaded.
  // Helper struct for storing the information.
  struct OffloadHelper {
    // Instructions inside the call computation.
    HloInstruction* input_to_resource_update;
    HloInstruction* output_from_resource_update;

    // Instructions inside the resource update computation.
    HloInstruction* input_in_resource_update;
    HloInstruction* output_in_resource_update;

    // Instructions in entry computation.
    HloInstruction* input_to_call;
    HloInstruction* output_from_call;

    // Entry computation indicies for the tensors.
    int64 entry_param_number;
    int64 entry_output_idx;
    int64 call_operand_idx;
  };

  std::vector<OffloadHelper> offload_infos;
  for (HloInstruction* operand : resource_update->operands()) {
    if (operand->shape().IsTuple()) {
      continue;
    }

    // Has to be a parameter instruction inside the call computation to be
    // considered.
    if (operand->opcode() != HloOpcode::kParameter) {
      continue;
    }

    // Do not proceed if this operand is used multiple times in the resource
    // update or at multiple operands.
    auto operand_indices = resource_update->OperandIndices(operand);
    if (operand_indices.size() != 1 || operand->user_count() != 1) {
      continue;
    }

    // Check whether the input to the call operations at this index is also
    // a parameter - i.e. this is a parameter in the entry computation.
    const int64 call_param_number = operand->parameter_number();
    HloInstruction* call_input = call_op->mutable_operand(call_param_number);
    if (call_input->opcode() != HloOpcode::kParameter) {
      continue;
    }

    // Also expect the call input to be unique.
    if (call_input->user_count() != 1 ||
        call_op->OperandIndices(call_input).size() != 1) {
      continue;
    }

    // Check that the output tuple of the call operation at index
    // `call_param_number` is an output from a resource update via a
    // GTE (i.e. the value was updated inside the resource update).
    HloInstruction* call_root = call_comp->root_instruction();
    CHECK_EQ(call_root->opcode(), HloOpcode::kTuple);

    HloInstruction* resource_update_output =
        call_root->mutable_operand(call_param_number);
    if (resource_update_output->opcode() != HloOpcode::kGetTupleElement) {
      continue;
    }
    if (resource_update_output->operand(0) != resource_update) {
      continue;
    }

    // TODO(T17040) - extend this for read only resource variables.
    // Check the aliasing map and only proceed if the call input is a
    // resource variable which is also an output of the computation.
    const int64 entry_param_number = call_input->parameter_number();
    const auto& input_info =
        annotations_.input_output_aliasing_map.GetEntryInputInfos().at(
            entry_param_number);
    if (input_info.IsResourceNotModified()) {
      continue;
    }

    // Check that the call output is only used by the root instruction.
    std::vector<HloInstruction*> call_outputs;
    bool all_users_gtes = true;
    // First we need to make sure that all users of the call op are GTEs and
    // that there is only one GTE for the current parameter.
    for (HloInstruction* output : call_op->users()) {
      if (output->opcode() != HloOpcode::kGetTupleElement) {
        all_users_gtes = false;
        break;
      }
      if (output->tuple_index() == call_param_number) {
        call_outputs.push_back(output);
      }
    }
    if (!all_users_gtes || call_outputs.size() != 1) {
      continue;
    }
    // Check that it's only used by the root instruction.
    HloInstruction* output_from_call = call_outputs[0];
    if (output_from_call->user_count() != 1 ||
        output_from_call->users()[0] != entry_comp->root_instruction()) {
      continue;
    }
    // It is only used once.
    auto output_indices =
        entry_comp->root_instruction()->OperandIndices(output_from_call);
    if (output_indices.size() != 1) {
      continue;
    }

    // Only offload if the input is aliasing the output.
    int64 entry_output_idx = output_indices[0];
    const auto& output_info =
        annotations_.input_output_aliasing_map.GetEntryOutputInfos().at(
            entry_output_idx);
    if (output_info.GetInputIndex() != entry_param_number) {
      continue;
    }

    // Get the input/output inside the resource update.
    HloInstruction* input_in_resource_update =
        resource_update_comp->parameter_instruction(operand_indices[0]);
    HloInstruction* resource_update_root =
        resource_update_comp->root_instruction();
    CHECK_EQ(resource_update_root->opcode(), HloOpcode::kTuple);
    HloInstruction* output_in_resource_update =
        resource_update_root->mutable_operand(
            resource_update_output->tuple_index());

    OffloadHelper offload_info;
    offload_info.input_to_resource_update = operand;
    offload_info.output_from_resource_update = resource_update_output;
    offload_info.input_in_resource_update = input_in_resource_update;
    offload_info.output_in_resource_update = output_in_resource_update;
    offload_info.input_to_call = call_input;
    offload_info.output_from_call = output_from_call;
    offload_info.entry_param_number = entry_param_number;
    offload_info.entry_output_idx = entry_output_idx;
    offload_info.call_operand_idx = call_param_number;
    offload_infos.emplace_back(offload_info);
  }

  if (offload_infos.empty()) {
    return changed;
  }

  if (!remote_memory_supported_) {
    LOG(INFO) << absl::StrCat(
        "Current configuration of the IPU devices does not support remote "
        "buffers and therefore weight update only variables cannot be "
        "offloaded to remote memory. Set the `offload_weight_update_variables` "
        "argument of ",
        IsPipelineOp(call_op) ? "`pipelining_ops.pipeline`"
                              : "`GradientAccumulationOptimizerV2`",
        "to `False` to stop seeing this message.");
    return changed;
  }

  std::set<int64> call_params_to_remove;
  // For each parameter in offload_info, insert a load and store op.
  for (auto& offload_info : offload_infos) {
    if (minimum_remote_tensor_size_ >
        ShapeUtil::ByteSizeOf(offload_info.input_to_call->shape())) {
      VLOG(1) << "Variable " << offload_info.entry_param_number << ": "
              << offload_info.input_to_call->ToString()
              << " is smaller than the minimum remote tensor size ("
              << minimum_remote_tensor_size_
              << "B) and is therefore not offloaded.";
      continue;
    }

    VLOG(1) << "Offloading variable " << offload_info.entry_param_number << ": "
            << offload_info.input_to_call->ToString();

    // Create a load instruction inside the resource update and use that instead
    // of the parameter.
    HloInstruction* remote_load = resource_update_comp->AddInstruction(
        CreateHloRemoteParameterLoad(offload_info.input_in_resource_update));
    TF_RETURN_IF_ERROR(
        offload_info.input_in_resource_update->ReplaceAllUsesWith(remote_load));

    // If the input and output are the same instruction, then we use the remote
    // resource but don't modify it. In this case we simply reconnect the input
    // remote buffer to the original output position.
    if (offload_info.input_in_resource_update ==
        offload_info.output_in_resource_update) {
      TF_RETURN_IF_ERROR(
          remote_load->ReplaceUseWith(resource_update_comp->root_instruction(),
                                      offload_info.input_in_resource_update));
    } else {
      // Add a remote store for the updated value inside the resource update.
      HloInstruction* remote_store =
          resource_update_comp->AddInstruction(CreateHloRemoteParameterStore(
              offload_info.input_in_resource_update,
              offload_info.output_in_resource_update));

      TF_RETURN_IF_ERROR(offload_info.output_in_resource_update->ReplaceUseWith(
          resource_update_comp->root_instruction(), remote_store));
    }

    // Mark this input as being stored in a remote buffer.
    annotations_.remote_parameter_infos.insert(
        RemoteParameterInfo{offload_info.entry_param_number});
  }

  return true;
}

StatusOr<bool> ResourceUpdateVariablesOffload::Run(HloModule* module) {
  VLOG(2) << "Before ResourceUpdateVariablesOffload:";
  XLA_VLOG_LINES(2, module->ToString());
  bool changed = false;
  std::vector<std::pair<HloInstruction*, HloInstruction*>> to_optimize;
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (IsRepeatLoop(inst) || IsPipelineOp(inst)) {
        std::vector<HloInstruction*> resource_updates;
        absl::c_copy_if(inst->to_apply()->MakeInstructionPostOrder(),
                        std::back_inserter(resource_updates),
                        [](HloInstruction* i) { return IsResourceUpdate(i); });
        if (resource_updates.empty()) {
          continue;
        } else if (resource_updates.size() > 1) {
          return FailedPrecondition(
              "Detected multiple resource update instructions.");
        } else {
          to_optimize.push_back({inst, resource_updates[0]});
        }
      }
    }
  }

  for (auto pair : to_optimize) {
    TF_ASSIGN_OR_RETURN(bool changed_pair, Optimize(pair.first, pair.second));
    changed |= changed_pair;
  }

  if (changed) {
    VLOG(2) << "After ResourceUpdateVariablesOffload:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
