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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_feed_hoisting.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/fifo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {
bool CanHoistFeedTokenInput(const HloInstruction* token_input) {
  // We can only outline if the token input has no other dependencies.
  return token_input->opcode() == HloOpcode::kAfterAll &&
         token_input->operand_count() == 0;
}

StatusOr<HloInstruction*> HoistInfeed(HloInstruction* stage,
                                      HloInstruction* infeed) {
  HloComputation* pipeline_comp = stage->parent();
  HloComputation* stage_comp = stage->to_apply();
  HloInstruction* root = stage_comp->root_instruction();

  // Clone the token into the pipeline computation.
  HloInstruction* token = infeed->mutable_operand(0);
  HloInstruction* new_token = pipeline_comp->AddInstruction(token->Clone());

  // Clone the infeed into the pipeline computation.
  HloInstruction* new_infeed = pipeline_comp->AddInstruction(
      infeed->CloneWithNewOperands(infeed->shape(), {new_token}));

  // Clone the gte used to access the infeed.
  HloInstruction* infeed_gte = infeed->users()[0];
  HloInstruction* new_infeed_gte = pipeline_comp->AddInstruction(
      infeed_gte->CloneWithNewOperands(infeed_gte->shape(), {new_infeed}));

  // Create a new computation, adding the infeed as a parameter.
  auto builder = HloComputation::Builder(stage_comp->name());
  // A mapping from instructions in the old computation to the new one which is
  // currently being built.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> old_to_new_computation;
  // Go through the computation.
  for (HloInstruction* old_inst : stage_comp->MakeInstructionPostOrder()) {
    if (old_inst == token || old_inst == infeed) {
      // We do not have the token input or the infeed in the new computation.
    } else if (old_inst == infeed_gte) {
      // Replace the infeed_gte with a parameter for the infeed_gte which was
      // hoisted.
      HloInstruction* new_param =
          builder.AddInstruction(HloInstruction::CreateParameter(
              stage_comp->num_parameters(), infeed_gte->shape(),
              absl::StrCat("arg_", infeed_gte->name())));
      infeed_gte->SetupDerivedInstruction(new_param);
      old_to_new_computation[old_inst] = new_param;
    } else {
      // Get the new operands and clone.
      std::vector<HloInstruction*> new_operands(old_inst->operand_count());
      absl::c_transform(old_inst->operands(), new_operands.begin(),
                        [&old_to_new_computation](HloInstruction* old_operand) {
                          return old_to_new_computation.at(old_operand);
                        });
      // Clone new instruction.
      old_to_new_computation[old_inst] = builder.AddInstruction(
          old_inst->CloneWithNewOperands(old_inst->shape(), new_operands));
    }
  }

  // Replace the pipeline stage with a new one with the new computation.
  // Add the new infeed gte as a new operand.
  std::vector<HloInstruction*> new_operands = {stage->operands().begin(),
                                               stage->operands().end()};
  new_operands.push_back(new_infeed_gte);

  TF_ASSIGN_OR_RETURN(
      stage,
      ReplaceCallWith(stage, builder.Build(old_to_new_computation.at(root)),
                      new_operands, false));

  return stage;
}

StatusOr<HloInstruction*> HoistOutfeed(HloInstruction* stage,
                                       HloInstruction* outfeed) {
  HloComputation* pipeline_comp = stage->parent();
  HloComputation* stage_comp = stage->to_apply();
  HloInstruction* outfeed_input = outfeed->mutable_operand(0);
  HloInstruction* root = stage_comp->root_instruction();

  // New root has all the same inputs, plus the outfeed input.
  auto new_outputs = root->operands();
  new_outputs.push_back(outfeed_input);

  // Create the new root.
  HloInstruction* new_root =
      stage_comp->AddInstruction(HloInstruction::CreateTuple(new_outputs));
  root->SetupDerivedInstruction(new_root);

  // Use the new root and change the shape of the output.
  stage_comp->set_root_instruction(new_root, true);
  *stage->mutable_shape() = new_root->shape();
  if (root->user_count() == 0) {
    // Remove the old root.
    TF_RETURN_IF_ERROR(stage_comp->RemoveInstruction(root));
  }

  // Add the GTE for the new output (it's the last element in the
  // tuple output).
  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_outfeed_input,
      MakeGetTupleElementHlo(stage,
                             ShapeUtil::TupleElementCount(stage->shape()) - 1));
  outfeed_input->SetupDerivedInstruction(new_outfeed_input);

  // Clone the token into the pipeline computation.
  HloInstruction* token = outfeed->mutable_operand(1);
  HloInstruction* new_token = pipeline_comp->AddInstruction(token->Clone());

  // Clone the outfeed into the pipeline computation using the
  // new operands.
  HloInstruction* new_outfeed =
      pipeline_comp->AddInstruction(outfeed->CloneWithNewOperands(
          outfeed->shape(), {new_outfeed_input, new_token}));
  (void)new_outfeed;
  // Remove the old outfeed.
  TF_RETURN_IF_ERROR(stage_comp->RemoveInstructionAndUnusedOperands(outfeed));
  return stage;
}
}  // namespace

StatusOr<bool> PipelineFeedHoisting::HoistInPipeline(
    HloInstruction* pipeline_op) {
  bool changed = false;
  HloComputation* pipeline_comp = pipeline_op->to_apply();
  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  // Make sure that the root of each stage is a tuple.
  TF_RETURN_IF_ERROR(FixRootInstructions(stages));
  for (auto& stages : {stages.forward, stages.backward}) {
    for (HloInstruction* stage : stages) {
      bool hoisted;
      do {
        hoisted = false;
        // Note that hoisting can create a new computation, therefore we always
        // start from the begining.
        HloComputation* stage_comp = stage->to_apply();
        for (HloInstruction* inst : stage_comp->MakeInstructionPostOrder()) {
          // We cannot hoist if there are control dependencies.
          if (inst->control_predecessors().size() ||
              inst->control_successors().size()) {
            continue;
          }

          switch (inst->opcode()) {
            case HloOpcode::kInfeed: {
              if (!CanHoistFeedTokenInput(inst->operand(0))) {
                break;
              }
              // We can only hoist if the infeed has one user which is a GTE at
              // tuple index  = 0 (tuple_index 1 is a token.)
              if (inst->user_count() != 1) {
                break;
              }
              HloInstruction* user = inst->users()[0];
              if (user->opcode() != HloOpcode::kGetTupleElement ||
                  user->tuple_index() != 0) {
                break;
              }
              // Hoist the infeed out.
              TF_ASSIGN_OR_RETURN(stage, HoistInfeed(stage, inst));

              hoisted = true;
              break;
            }
            case HloOpcode::kOutfeed: {
              if (!CanHoistFeedTokenInput(inst->operand(1))) {
                break;
              }
              // Hoist the outfeed out.
              TF_ASSIGN_OR_RETURN(stage, HoistOutfeed(stage, inst));
              hoisted = true;
              break;
            }
            default:
              break;
          }
          // Stop as soon as we hoisted something.
          if (hoisted) {
            changed = true;
            break;
          }
        }
      } while (hoisted);
    }
  }
  return changed;
}

StatusOr<bool> PipelineFeedHoisting::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to hoist.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  VLOG(2) << "Before PipelineFeedHoisting:";
  XLA_VLOG_LINES(2, module->ToString(HloPrintOptions::ShortParsable()));

  TF_ASSIGN_OR_RETURN(bool changed, HoistInPipeline(pipeline_ops[0]));

  if (changed) {
    VLOG(2) << "After PipelineFeedHoisting:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the Pipeline.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
