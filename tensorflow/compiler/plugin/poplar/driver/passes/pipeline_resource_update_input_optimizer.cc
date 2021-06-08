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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_resource_update_input_optimizer.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/fifo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {

StatusOr<bool> PipelineResourceUpdateInputOptimizer::OptimizePipeline(
    HloInstruction* pipeline_op) {
  bool changed = false;
  HloComputation* pipeline_comp = pipeline_op->to_apply();

  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  TF_RETURN_IF_ERROR(FixRootInstructions(stages));
  if (!stages.resource_update) {
    return false;
  }

  HloInstruction* resource_update = *stages.resource_update;
  HloComputation* resource_update_comp = resource_update->to_apply();

  HloInstruction* first_stage = stages.forward[0];
  // Make sure the value is threaded between consecutive stages.
  for (int64 i = 0; i != first_stage->operand_count(); ++i) {
    HloInstruction* operand = first_stage->mutable_operand(i);

    if (operand->opcode() != HloOpcode::kParameter) {
      continue;
    }

    // The index of the value passed from the previous stage.
    int64 input_idx = i;

    // Stores the information about a possible elementwise modifier.
    struct ElementwiseModifierInfo {
      HloInstruction* modifier;
      int64 modifier_operand_idx;
    };

    absl::optional<ElementwiseModifierInfo> elementwise_modifier;

    bool valid_path = false;
    for (int64 stage_idx = 0; stage_idx != stages.forward.size(); ++stage_idx) {
      HloInstruction* stage = stages.forward[stage_idx];
      HloComputation* stage_comp = stage->to_apply();
      HloInstruction* parameter = stage_comp->parameter_instruction(input_idx);
      HloInstruction* state_root = stage_comp->root_instruction();

      // Check whether it is used in the root of the stage.
      int64 output_idx = -1;
      if (state_root->OperandIndices(parameter).size() != 1) {
        if (elementwise_modifier) {
          // Only single elementwise modifier is allowed.
          break;
        }

        // Check whether there is an elementwise user - if there is then this
        // can be lowered directly into the resource update iff any other inputs
        // are constant.
        for (HloInstruction* user : parameter->users()) {
          if (user->IsElementwise() && user->operand_count() < 3) {
            int64 modifier_operand_idx = user->operand_index(parameter);

            // For binary operations check that the other side is a constant.
            if (user->operand_count() == 2) {
              const HloInstruction* other_side =
                  user->operand(1 - modifier_operand_idx);

              if (!other_side->IsConstant()) {
                break;
              }
            }

            // Found the elementwise modifier.
            if (state_root->OperandIndices(user).size() == 1) {
              output_idx = state_root->operand_index(user);
              elementwise_modifier =
                  ElementwiseModifierInfo{user, modifier_operand_idx};
              break;
            }
          }
        }

        // Did not find an elementwise modifier - cannot continue.
        if (!elementwise_modifier) {
          break;
        }
      } else {
        output_idx = state_root->operand_index(parameter);
      }

      // Try and get the GTE user.
      auto status_or = GetUniqueGTEUser(stage, output_idx);
      if (!status_or.ok()) {
        break;
      }
      HloInstruction* gte = status_or.ValueOrDie();

      // Check whether this is used by the resource update.
      if (resource_update->OperandIndices(gte).size() == 1) {
        // Found a match.
        input_idx = resource_update->operand_index(gte);
        valid_path = true;
        break;
      }

      if (stage_idx + 1 < stages.forward.size()) {
        // Check whether the value is passed to the next stage.
        HloInstruction* next_stage = stages.forward[stage_idx + 1];
        if (next_stage->OperandIndices(gte).size() != 1) {
          break;
        }
        input_idx = next_stage->operand_index(gte);
      }
    }

    if (!valid_path) {
      continue;
    }

    // Pass the operand to the resource update directly.
    VLOG(1) << "Replacing operand " << input_idx << " of resource update with "
            << operand->ToString();
    TF_RETURN_IF_ERROR(resource_update->ReplaceOperandWith(input_idx, operand));

    HloInstruction* parameter =
        resource_update_comp->parameter_instruction(input_idx);

    if (elementwise_modifier) {
      // Ensure the parameter has the right shape (the modifier might have a
      // different data type).
      *parameter->mutable_shape() = operand->shape();

      HloInstruction* modifier = elementwise_modifier->modifier;
      int64 modifier_operand_idx = elementwise_modifier->modifier_operand_idx;

      VLOG(1) << "Adding elementwise modifier " << modifier->ToString();

      // Get the operands for the modifier inside of the resource update.
      std::vector<HloInstruction*> operands(modifier->operand_count());
      operands[modifier_operand_idx] = parameter;

      if (modifier->operand_count() == 2) {
        // Clone the other operand.
        int64 other_side_idx = 1 - modifier_operand_idx;
        HloInstruction* other_side = modifier->mutable_operand(other_side_idx);
        operands[other_side_idx] =
            resource_update_comp->AddInstruction(other_side->Clone());
      } else {
        CHECK_EQ(modifier->operand_count(), 1);
      }

      // Create the modififer inside of the resource update.
      HloInstruction* in_resource_modifier =
          resource_update_comp->AddInstruction(
              modifier->CloneWithNewOperands(modifier->shape(), operands));
      // Replace all the uses with it.
      TF_RETURN_IF_ERROR(parameter->ReplaceAllUsesWith(in_resource_modifier));
    } else {
      CHECK_EQ(parameter->shape(), operand->shape());
    }
    changed = true;
  }

  return changed;
}

StatusOr<bool> PipelineResourceUpdateInputOptimizer::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> pipeline_ops,
                      GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to optimize.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  HloInstruction* pipeline_op = pipeline_ops[0];
  VLOG(2) << "Before PipelineResourceUpdateInputOptimizer:";
  XLA_VLOG_LINES(2, module->ToString());

  TF_ASSIGN_OR_RETURN(bool changed, OptimizePipeline(pipeline_op));

  if (changed) {
    VLOG(2) << "After PipelineResourceUpdateInputOptimizer:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the Pipeline.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
