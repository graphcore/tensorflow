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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_resource_variables_offload.h"

#include <algorithm>
#include <set>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {

PipelineResourceVariablesOffload::PipelineResourceVariablesOffload(
    CompilerAnnotations& annotations, bool remote_memory_supported)
    : annotations_(annotations),
      remote_memory_supported_(remote_memory_supported) {}

StatusOr<bool> PipelineResourceVariablesOffload::OptimizePipeline(
    HloInstruction* pipeline_op) {
  bool changed = false;

  // Do not optimize if this is not a pipeline inside an entry computation.
  if (pipeline_op->parent() != pipeline_op->GetModule()->entry_computation()) {
    return changed;
  }
  HloComputation* entry_comp = pipeline_op->GetModule()->entry_computation();
  HloComputation* pipeline_comp = pipeline_op->to_apply();

  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  // Do not optimize if there is no resource update or pipeline wu variable
  // offloading is turned off.
  if (!stages.resource_update || !GetPipelineOffloadWUVariables(pipeline_op)) {
    return changed;
  }

  if (pipeline_op == entry_comp->root_instruction()) {
    // Convert the entry root to create GTEs for each output shape and then
    // create a root tuple instruction so that we can mark outputs as remote
    // buffers.
    const int64 num_outputs =
        ShapeUtil::TupleElementCount(pipeline_op->shape());
    std::vector<HloInstruction*> gtes(num_outputs);
    for (int64 tuple_index = 0; tuple_index != num_outputs; ++tuple_index) {
      TF_ASSIGN_OR_RETURN(gtes[tuple_index],
                          MakeGetTupleElementHlo(pipeline_op, tuple_index));
    }
    HloInstruction* new_root =
        entry_comp->AddInstruction(HloInstruction::CreateTuple(gtes));
    entry_comp->set_root_instruction(new_root);
    changed = true;
  }

  // We cannot optimize if the root of the entry computation is not a tuple.
  if (entry_comp->root_instruction()->opcode() != HloOpcode::kTuple) {
    return changed;
  }

  // We cannot optimize if the output tuple has users - this implies that the
  // parameter has users other than the pipeline.
  if (entry_comp->root_instruction()->user_count()) {
    return changed;
  }

  HloInstruction* resource_update = *stages.resource_update;
  HloComputation* resource_update_comp = resource_update->to_apply();

  // Find any parameter instructions which can be offloaded.
  // Helper struct for storing the information.
  struct OffloadHelper {
    // Instructions inside the pipeline computation.
    HloInstruction* input_to_resource_update;
    HloInstruction* output_from_resource_update;

    // Instructions inside the resource update computation.
    HloInstruction* input_in_resource_update;
    HloInstruction* output_in_resource_update;

    // Instructions in entry computation.
    HloInstruction* input_to_pipeline;
    HloInstruction* output_from_pipeline;

    // Entry computation indicies for the tensors.
    int64 entry_param_number;
    int64 entry_output_idx;
    int64 pipeline_operand_idx;
  };

  std::vector<OffloadHelper> offload_infos;
  for (HloInstruction* operand : resource_update->operands()) {
    if (operand->shape().IsTuple()) {
      continue;
    }

    // Has to be a parameter instruction inside the pipeline computation to be
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

    // Check whether the input to the pipeline operations at this index is also
    // a parameter - i.e. this is a parameter in the entry computation.
    const int64 pipeline_param_number = operand->parameter_number();
    HloInstruction* pipeline_input =
        pipeline_op->mutable_operand(pipeline_param_number);
    if (pipeline_input->opcode() != HloOpcode::kParameter) {
      continue;
    }

    // Also expect the pipeline input to be unique.
    if (pipeline_input->user_count() != 1 ||
        pipeline_op->OperandIndices(pipeline_input).size() != 1) {
      continue;
    }

    // Check that the output tuple of the pipeline operation at index
    // `pipeline_param_number` is an output from a resource update via a
    // GTE (i.e. the value was updated inside the resource update).
    HloInstruction* pipeline_root = pipeline_comp->root_instruction();
    CHECK_EQ(pipeline_root->opcode(), HloOpcode::kTuple);

    HloInstruction* resource_update_output =
        pipeline_root->mutable_operand(pipeline_param_number);
    if (resource_update_output->opcode() != HloOpcode::kGetTupleElement) {
      continue;
    }
    if (resource_update_output->operand(0) != resource_update) {
      continue;
    }

    // TODO(T17040) - extend this for read only resource variables.
    // Check the aliasing map and only proceed if the pipeline input is a
    // resource variable which is also an output of the computation.
    const int64 entry_param_number = pipeline_input->parameter_number();
    const auto& input_info =
        annotations_.input_output_aliasing_map.GetEntryInputInfos().at(
            entry_param_number);
    if (input_info.IsResourceNotModified()) {
      continue;
    }

    // Check that the pipeline output is only used by the root instruction.
    std::vector<HloInstruction*> pipeline_outputs;
    bool all_users_gtes = true;
    // First we need to make sure that all users of the pipeline op are GTEs and
    // that there is only one GTE for the current parameter.
    for (HloInstruction* output : pipeline_op->users()) {
      if (output->opcode() != HloOpcode::kGetTupleElement) {
        all_users_gtes = false;
        break;
      }
      if (output->tuple_index() == pipeline_param_number) {
        pipeline_outputs.push_back(output);
      }
    }
    if (!all_users_gtes || pipeline_outputs.size() != 1) {
      continue;
    }
    // Check that it's only used by the root instruction.
    HloInstruction* output_from_pipeline = pipeline_outputs[0];
    if (output_from_pipeline->user_count() != 1 ||
        output_from_pipeline->users()[0] != entry_comp->root_instruction()) {
      continue;
    }
    // It is only used once.
    auto output_indices =
        entry_comp->root_instruction()->OperandIndices(output_from_pipeline);
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
    offload_info.input_to_pipeline = pipeline_input;
    offload_info.output_from_pipeline = output_from_pipeline;
    offload_info.entry_param_number = entry_param_number;
    offload_info.entry_output_idx = entry_output_idx;
    offload_info.pipeline_operand_idx = pipeline_param_number;
    offload_infos.emplace_back(offload_info);
  }

  if (offload_infos.empty()) {
    return changed;
  }

  if (!remote_memory_supported_) {
    LOG(INFO)
        << "Current configuration of the IPU devices does not support graph "
           "streaming and therefore weight update only variables cannot be "
           "offloaded to remote memory. Set the "
           "`offload_weight_update_variables` argument of "
           "`pipelining_ops.pipeline` to `False` to stop seeing this message.";
    return changed;
  }

  std::set<int64> pipeline_params_to_remove;
  // For each parameter in offload_info, insert a load and store op and a dummy
  // output op.
  for (auto& offload_info : offload_infos) {
    VLOG(1) << "Offloading variable " << offload_info.entry_param_number << ": "
            << offload_info.input_to_pipeline->ToString();

    // Create a load instruction inside the resource update and use that instead
    // of the parameter.
    HloInstruction* remote_load =
        resource_update_comp->AddInstruction(CreateHloRemoteParameterLoad(
            offload_info.input_in_resource_update->shape(),
            offload_info.entry_param_number));
    TF_RETURN_IF_ERROR(
        offload_info.input_in_resource_update->ReplaceAllUsesWith(remote_load));

    // Special case for parameters just being passed through.
    if (offload_info.input_in_resource_update ==
        offload_info.output_in_resource_update) {
      offload_info.output_in_resource_update = remote_load;
    }

    // Add a remote store for the updated value inside the resource update.
    resource_update_comp->AddInstruction(CreateHloRemoteParameterStore(
        offload_info.output_in_resource_update, offload_info.entry_output_idx));

    // Create a dummy output of the variable in the entry computation and use
    // that as the output in the root instruction which we already checked is a
    // tuple.
    TF_RETURN_IF_ERROR(entry_comp->ReplaceWithNewInstruction(
        offload_info.output_from_pipeline,
        CreateHloRemoteParameterDummyOutput(
            offload_info.output_from_pipeline->shape(),
            offload_info.entry_output_idx)));

    // Mark the parameter for removal.
    pipeline_params_to_remove.insert(offload_info.pipeline_operand_idx);

    // Mark this input as being stored in a remote buffer.
    annotations_.remote_parameter_infos.insert(
        RemoteParameterInfo{offload_info.entry_param_number});
  }

  // Remove the outputs for offloaded parameters from the pipeline.
  TF_RETURN_IF_ERROR(
      RemoveOutputsFromCall(pipeline_op, pipeline_params_to_remove));
  TF_RETURN_IF_ERROR(
      HloDCE::RunOnComputation(pipeline_op->to_apply()).status());

  // Now remove unused inputs/outputs in the resource update.
  bool removed_insts = false;
  TF_ASSIGN_OR_RETURN(resource_update,
                      PipelineOptimizer::OptimizeCallInstruction(
                          resource_update, &removed_insts));
  CHECK(removed_insts);
  // Remove the inputs for offloaded parameters to the pipeline.
  TF_ASSIGN_OR_RETURN(pipeline_op, RemoveParametersFromCall(
                                       pipeline_op, pipeline_params_to_remove));
  TF_RETURN_IF_ERROR(
      HloDCE::RunOnComputation(pipeline_op->to_apply()).status());

  return true;
}

StatusOr<bool> PipelineResourceVariablesOffload::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> pipeline_ops,
                      GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to offload.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  VLOG(2) << "Before PipelineResourceVariablesOffload:";
  XLA_VLOG_LINES(2, module->ToString());

  TF_ASSIGN_OR_RETURN(bool changed, OptimizePipeline(pipeline_ops[0]));

  if (changed) {
    VLOG(2) << "After PipelineResourceVariablesOffload:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the Pipeline.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
