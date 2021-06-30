/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_copy_inserter.h"

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {
StatusOr<bool> AddCopyIfParameterModifiedInplace(HloInstruction* call,
                                                 int64 parameter_number) {
  HloComputation* comp = call->to_apply();
  HloInstruction* parameter = comp->parameter_instruction(parameter_number);
  if (IsOutputModifiedInplace(parameter)) {
    VLOG(1) << "Inserting a copy for stage " << call->ToString()
            << " parameter number " << parameter_number;
    // Insert a copy from the the parameter.
    HloInstruction* copy = comp->AddInstruction(HloInstruction::CreateUnary(
        parameter->shape(), HloOpcode::kCopy, parameter));
    parameter->SetupDerivedInstruction(copy);
    TF_RETURN_IF_ERROR(parameter->ReplaceAllUsesWith(copy));
    return true;
  }
  return false;
}

StatusOr<bool> InsertStageInputCopies(PipelineStages& pipeline_stages) {
  bool changed = false;
  for (auto stages : {pipeline_stages.forward, pipeline_stages.backward}) {
    for (HloInstruction* stage : stages) {
      // Go through all the operands to the stage, and insert copies if
      // necessary to make sure no parameter/pipeline execution counter is
      // modified inplace. We could alternatively mark the modifying instruction
      // not inplace, but that might result in higher memory usage (for example
      // a tuple going into a loop is now not inplace).
      // TODO(T10387)
      for (int64 op_idx = 0; op_idx != stage->operand_count(); ++op_idx) {
        if (!IsPipelineStageReadOnlyInput(stage->operand(op_idx))) {
          continue;
        }
        TF_ASSIGN_OR_RETURN(bool added,
                            AddCopyIfParameterModifiedInplace(stage, op_idx));
        changed |= added;
      }
    }
  }
  return changed;
}

StatusOr<bool> InsertReadOnlyVariableCopies(HloInstruction* pipeline_op) {
  bool changed = false;
  HloComputation* pipeline_comp = pipeline_op->to_apply();
  HloInstruction* root = pipeline_comp->root_instruction();
  // Secondly, we make sure that if a parameter of the pipeline is an output
  // of the pipeline as well then it is not modified. TF2XLA lowering of
  // pipelines expects these to not be modified.
  if (ShapeUtil::TupleElementCount(root->shape()) !=
      pipeline_comp->num_parameters()) {
    return FailedPrecondition(
        "Expected the Pipeline to have %d outputs, but has %d",
        root->operand_count(), pipeline_comp->num_parameters());
  }
  for (int64 param_idx = 0; param_idx != pipeline_comp->num_parameters();
       ++param_idx) {
    HloInstruction* param_inst =
        pipeline_comp->parameter_instruction(param_idx);
    if (root->operand(param_idx) != param_inst) {
      // Don't need to do anything, the value is modified.
      continue;
    }
    for (HloInstruction* user : param_inst->users()) {
      if (user == root) {
        continue;
      }
      // If the value already has been copied we don't need to add moe copies.
      if (user->opcode() == HloOpcode::kCopy) {
        continue;
      }

      CHECK(IsAnyPipelineStageOpOrResourceUpdate(user));
      // Go through each use of the parameter in the user and insert kCopy
      // instructions if necessary.
      for (int64 index : user->OperandIndices(param_inst)) {
        TF_ASSIGN_OR_RETURN(bool added,
                            AddCopyIfParameterModifiedInplace(user, index));
        changed |= added;
      }
    }
  }
  return changed;
}

StatusOr<bool> InsertIntraIPUCopies(PipelineStages& stages) {
  bool changed = false;
  // Go through all the inputs to stages, if they are GTEs from the previous
  // stage, then insert a copy to make sure stages are not modifying the same
  // tensor.
  for (auto& stages : {stages.forward, stages.backward}) {
    for (HloInstruction* stage : stages) {
      for (int64 op_idx = 0; op_idx != stage->operand_count(); ++op_idx) {
        const HloInstruction* operand = stage->operand(op_idx);
        if (operand->opcode() == HloOpcode::kGetTupleElement) {
          if (IsPipelineStageOrBackwardOp(operand->operand(0))) {
            TF_ASSIGN_OR_RETURN(
                bool added, AddCopyIfParameterModifiedInplace(stage, op_idx));
            changed |= added;
          }
        }
      }
    }
  }
  return changed;
}

/**
 * Get the pipeline stage to device mapping.
 *
 * @param pipeline The outer pipeline instruction.
 *
 * @returns The mapping of the ith stage to a IPU device.
 *
 * @note Assumes the pipeline is correctly constructed.
 */
std::vector<int> GetPipelineStageDeviceMapping(const HloInstruction* pipeline) {
  HloComputation* pipeline_computation = pipeline->to_apply();

  // Cannot reasonably return StatusOr because this is called inside a
  // constructor.
  auto stage = GetPipelineStages(pipeline_computation).ValueOrDie();
  stage.forward.insert(stage.forward.end(), stage.backward.rbegin(),
                       stage.backward.rend());

  std::vector<int> result(stage.forward.size());

  const auto get_stage_shard = [](const HloInstruction* hlo) -> int {
    return *hlo->sharding_unique_device();
  };
  absl::c_transform(stage.forward, result.begin(), get_stage_shard);

  return result;
}

StatusOr<bool> InsertIntraIPUCopiesBetweenStages(HloInstruction* pipeline_op,
                                                 PipelineStages& stages) {
  TF_ASSIGN_OR_RETURN(const auto schedule, GetPipelineSchedule(pipeline_op));

  // Do not need to insert these copies when the schedule guarantees consecutive
  // stages cannot be executed in the same "time-step".
  switch (schedule) {
    case PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved:
    case PoplarBackendConfig::CallConfig::PipelineConfig::Sequential: {
      return false;
    }
    default: { break; }
  }

  // Get the stage-device mapping
  auto device_mapping = GetPipelineStageDeviceMapping(pipeline_op);
  auto stages_ = stages.forward;
  stages_.insert(stages_.end(), stages.backward.begin(), stages.backward.end());

  // Compute the device index difference between adjacent stages.
  std::adjacent_difference(device_mapping.begin(), device_mapping.end(),
                           device_mapping.begin());
  // Remove the head.
  device_mapping.erase(device_mapping.begin());

  bool changed = false;

  // For each zero, which represents a adjacent pair of stages on the same IPU,
  // insert a copy on the consumer's operands.
  for (auto itr = std::find(device_mapping.begin(), device_mapping.end(), 0);
       itr != device_mapping.end();
       itr = std::find(itr + 1, device_mapping.end(), 0)) {
    auto producer_idx = std::distance(device_mapping.begin(), itr);
    auto consumer_idx = 1 + producer_idx;

    VLOG(3) << "Insert copy between stage " << producer_idx << " and "
            << consumer_idx;

    auto consumer = stages_[consumer_idx];

    for (auto operand : consumer->operands()) {
      // Skip opaque elements.
      if (operand->opcode() == HloOpcode::kGetTupleElement &&
          !operand->shape().IsOpaque()) {
        HloInstruction* copy =
            pipeline_op->to_apply()->AddInstruction(HloInstruction::CreateUnary(
                operand->shape(), HloOpcode::kCopy, operand));

        copy->set_sharding(operand->sharding());
        TF_RETURN_IF_ERROR(operand->ReplaceUseWith(consumer, copy));

        changed = true;
      }
    }
  }

  return changed;
}

bool IsOutfeed(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kOutfeed;
}

StatusOr<bool> InsertIntraIPUCopiesBeforeOutfeed(HloInstruction* pipeline_op) {
  // Find outfeed instructions in the pipeline computation.
  auto pipeline_comp = pipeline_op->to_apply();
  auto instructions = pipeline_comp->MakeInstructionPostOrder();
  auto itr = std::stable_partition(instructions.begin(), instructions.end(),
                                   IsOutfeed);

  instructions.erase(itr, instructions.end());

  // Replace the outfeed operand with a copy of that operand.
  bool changed = false;
  for (auto outfeed : instructions) {
    auto operand = outfeed->mutable_operand(0);

    HloInstruction* copy =
        pipeline_comp->AddInstruction(HloInstruction::CreateUnary(
            operand->shape(), HloOpcode::kCopy, operand));

    copy->set_sharding(operand->sharding());
    TF_RETURN_IF_ERROR(operand->ReplaceUseWith(outfeed, copy));

    changed = true;
  }

  return changed;
}
}  // namespace

StatusOr<bool> PipelineCopyInserter::InsertInPipeline(
    HloInstruction* pipeline_op) {
  HloComputation* pipeline_comp = pipeline_op->to_apply();
  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  // Make sure that stages do not modify parameter and execution counter inputs
  // inplace (only the resource update can modify them).
  TF_ASSIGN_OR_RETURN(bool inserted_stage_input_copies,
                      InsertStageInputCopies(stages));
  // Insert copies for any read-only pipeline inputs.
  TF_ASSIGN_OR_RETURN(bool inserted_ro_copies,
                      InsertReadOnlyVariableCopies(pipeline_op));
  // Insert intra IPU copies between stages that are on the same IPU.
  TF_ASSIGN_OR_RETURN(bool inserted_intra_copies_between_stages,
                      InsertIntraIPUCopiesBetweenStages(pipeline_op, stages));
  // Insert intra IPU copies before outfeed instructions.
  TF_ASSIGN_OR_RETURN(bool inserted_intra_copies_before_outfeed,
                      InsertIntraIPUCopiesBeforeOutfeed(pipeline_op));
  // Insert intra IPU copies for any outputs from the previous stage on the same
  // device.
  TF_ASSIGN_OR_RETURN(bool inserted_intra_copies, InsertIntraIPUCopies(stages));
  return inserted_stage_input_copies || inserted_ro_copies ||
         inserted_intra_copies || inserted_intra_copies_between_stages ||
         inserted_intra_copies_before_outfeed;
}

StatusOr<bool> PipelineCopyInserter::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> pipeline_ops,
                      GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to fix.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  VLOG(2) << "Before PipelineCopyInserter:";
  XLA_VLOG_LINES(2, module->ToString(HloPrintOptions::ShortParsable()));

  TF_ASSIGN_OR_RETURN(bool changed, InsertInPipeline(pipeline_ops[0]));

  if (changed) {
    VLOG(2) << "After PipelineCopyInserter:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the Pipeline.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
