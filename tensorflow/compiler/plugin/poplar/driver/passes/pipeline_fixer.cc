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
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_fixer.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_noop.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
Status PipelineFixer::InsertStatefulNoopsIntoStages() {
  for (auto& stages : {stages_.forward, stages_.backward}) {
    for (HloInstruction* stage : stages) {
      HloComputation* stage_computation = stage->to_apply();
      stage_computation->AddInstruction(CreateStatefulNoop());
    }
  }
  return Status::OK();
}

Status PipelineFixer::UpdateStage(const StageID& stage_id,
                                  HloInstruction* new_stage) {
  if (stage_id.is_forward) {
    stages_.forward[stage_id.id] = new_stage;
  } else {
    stages_.backward[stage_id.id] = new_stage;
  }
  return Status::OK();
}

HloInstruction* PipelineFixer::GetStage(const StageID& stage_id) {
  if (stage_id.is_forward) {
    return stages_.forward[stage_id.id];
  } else {
    return stages_.backward[stage_id.id];
  }
}

std::vector<HloInstruction*> PipelineFixer::GetOrderedStages() {
  std::vector<HloInstruction*> stages(stages_.forward.size() +
                                      stages_.backward.size());
  absl::c_copy(stages_.forward, stages.begin());
  std::copy(stages_.backward.rbegin(), stages_.backward.rend(),
            std::next(stages.begin(), stages_.forward.size()));
  return stages;
}

namespace {
StatusOr<std::vector<HloInstruction*>> FindClusterToLower(
    HloInstruction* stage, const StageID& stage_id,
    HloInstruction* lowering_root, PipelineDataflowAnalysis* analysis) {
  // Build a cluster of instructions which are lowered and order them.
  // Start from the root and build up the cluster by visiting both
  // operands and users of instructions already in the cluster.
  std::vector<HloInstruction*> ordered_lowering;
  std::vector<const HloValueSet*> value_sets;
  absl::flat_hash_set<HloInstruction*> to_visit;
  absl::flat_hash_set<HloInstruction*> visited;
  to_visit.insert(lowering_root);

  while (!to_visit.empty()) {
    HloInstruction* inst = *to_visit.begin();
    to_visit.erase(inst);

    bool ready_to_lower = true;
    for (HloInstruction* operand : inst->operands()) {
      TF_ASSIGN_OR_RETURN(bool operand_needs_lowering,
                          analysis->HasToBeLowered(operand));
      ready_to_lower &= (visited.contains(operand) || !operand_needs_lowering);
    }
    std::vector<HloInstruction*> candidates;
    if (ready_to_lower) {
      // If we are ready to lower an instruction then add its value set
      // and visit its children.
      ordered_lowering.push_back(inst);
      visited.insert(inst);
      value_sets.push_back(&analysis->GetValueSet(inst));
      candidates = inst->users();
    } else {
      // We need to lower operands first.
      candidates = {inst->operands().begin(), inst->operands().end()};
    }

    // Add any instructions which need to be considered for lowering.
    for (HloInstruction* candidate : candidates) {
      if (!visited.contains(candidate)) {
        TF_ASSIGN_OR_RETURN(bool needs_lowering,
                            analysis->HasToBeLowered(candidate));
        if (needs_lowering) {
          to_visit.insert(candidate);
        }
      }
    }
  }
  HloValueSet value_set;
  value_set.AssignUnionOf(value_sets);

  // If the current stage is a forward stage and the value set contains a
  // backward stage then we can't lower this cluster into the fwd stage
  // without violating data flow constraints (i.e. backprop can't be used
  // in the forward pass).
  // We therefore skip and the bwd stage will deal with this.
  auto value_from_bwd = [](const HloValue* value) {
    return IsPipelineStageBackward(value->instruction());
  };

  if (stage_id.is_forward &&
      absl::c_any_of(value_set.values(), value_from_bwd)) {
    return std::vector<HloInstruction*>();
  }

  // Verify we can lower this.
  TF_RETURN_IF_ERROR(analysis->VerifyPipelineStageOperands(stage, value_set));
  return ordered_lowering;
}
}  // namespace

// Lowers any outputs of the stage into the stage.
// Returns true if the stage has changed.
StatusOr<bool> PipelineFixer::LowerPipelineStagesOutputs() {
  bool changed = false;

  TF_ASSIGN_OR_RETURN(auto analysis,
                      PipelineDataflowAnalysis::GetAnalysis(stages_));
  for (HloInstruction* stage : GetOrderedStages()) {
    TF_ASSIGN_OR_RETURN(StageID stage_id, analysis->GetStageID(stage));
    // We can't just iterate over users as we might remove some of them during
    // lowering.
    absl::flat_hash_set<HloInstruction*> stage_users(stage->users().begin(),
                                                     stage->users().end());
    // We try and lower one user at a time, otherwise we might create clusters
    // which cannot be lowered because they are using multiple pipeline stages.
    while (!stage_users.empty()) {
      HloInstruction* stage_gte_user = *stage_users.begin();
      stage_users.erase(stage_gte_user);
      CHECK_EQ(stage_gte_user->opcode(), HloOpcode::kGetTupleElement);
      CHECK_EQ(stage_gte_user->user_count(), 1);

      HloInstruction* stage_user = stage_gte_user->users()[0];

      TF_ASSIGN_OR_RETURN(bool needs_lowering,
                          analysis->HasToBeLowered(stage_user));
      if (!needs_lowering) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(
          std::vector<HloInstruction*> ordered_lowering,
          FindClusterToLower(stage, stage_id, stage_user, analysis.get()));
      // Nothing to lower.
      if (ordered_lowering.empty()) {
        continue;
      }
      VLOG(3) << "Lowering outputs for stage " << stage_id;
      changed = true;

      // Prevent use after free in the outter loop.
      absl::c_for_each(ordered_lowering, [&stage_users](HloInstruction* inst) {
        stage_users.erase(inst);
      });

      // Lower the instructions into the computation.
      TF_ASSIGN_OR_RETURN(
          stage, AddInstructionsToPipelineStage(stage, ordered_lowering));

      TF_RETURN_IF_ERROR(UpdateStage(stage_id, stage));
      // Recompute the analysis.
      TF_ASSIGN_OR_RETURN(analysis,
                          PipelineDataflowAnalysis::GetAnalysis(stages_));
    }

    // The dataflow in pipelining requires all the tensors to be thread through
    // consecutive pipeline stages. Check if there are any outputs from the
    // previous stage which are used by other stages directly and thread them
    // through this stage.
    // Note that we currently assume that forward stages do not require
    // threading as the Python API does not allow for unthreaded inputs.
    if (stage_id.is_forward) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(StageID previous_stage_id,
                        analysis->GetPreviousStageID(stage));
    HloInstruction* previous_stage = GetStage(previous_stage_id);

    // We remember that a pipeline stage only outputs GTEs and that each
    // GTE has been de-duplicated.
    // Store the tuple index from each user gte.
    HloInstruction* pipeline_root = stage->parent()->root_instruction();
    absl::flat_hash_map<int64, absl::flat_hash_set<HloInstruction*>>
        output_users;
    for (HloInstruction* gte : previous_stage->users()) {
      CHECK_EQ(gte->opcode(), HloOpcode::kGetTupleElement);
      CHECK_EQ(gte->user_count(), 1);
      int64 tuple_index = gte->tuple_index();
      HloInstruction* gte_user = gte->users()[0];
      // Do not track usage in the current stage - this has already been
      // lowered or if the usage is the root instruction.
      if (gte_user != stage && gte_user != pipeline_root) {
        output_users[tuple_index].insert(gte);
      }
    }
    if (output_users.empty()) {
      continue;
    }
    VLOG(3) << "Threading values through stage " << stage_id;
    changed = true;

    // For each GTE tuple index, get one of GTEs X, replace all the other GTEs
    // uses with it, and add X as a forced parameter. The lowering will then
    // thread the value through and create GTEs out of this stage.
    absl::flat_hash_set<HloInstruction*> forced_parameters;
    for (auto& tuple_index_users_pair : output_users) {
      absl::flat_hash_set<HloInstruction*> users =
          tuple_index_users_pair.second;
      HloInstruction* forced_parameter = *users.begin();
      users.erase(forced_parameter);
      forced_parameters.insert(forced_parameter);
      for (HloInstruction* gte : users) {
        TF_RETURN_IF_ERROR(
            gte->parent()->ReplaceInstruction(gte, forced_parameter));
      }
    }

    // Verify we can lower this.
    TF_RETURN_IF_ERROR(analysis->VerifyPipelineStageOperands(
        stage, analysis->GetValueSet(previous_stage)));
    // Lower the instructions into the computation.
    TF_ASSIGN_OR_RETURN(stage, AddInstructionsToPipelineStage(
                                   stage, {}, {}, forced_parameters));

    TF_RETURN_IF_ERROR(UpdateStage(stage_id, stage));
    // Recompute the analysis.
    TF_ASSIGN_OR_RETURN(analysis,
                        PipelineDataflowAnalysis::GetAnalysis(stages_));
  }
  return changed;
}

// This function lowers inputs to the pipeline stage which are arguments
// to the stage, but can be lowered (for examples constants etc.).
// Returns true if the stage has changed.
StatusOr<bool> PipelineFixer::LowerPipelineStagesInputs() {
  bool changed = false;

  TF_ASSIGN_OR_RETURN(auto analysis,
                      PipelineDataflowAnalysis::GetAnalysis(stages_));
  for (HloInstruction* stage : GetOrderedStages()) {
    TF_ASSIGN_OR_RETURN(StageID stage_id, analysis->GetStageID(stage));

    // Store paramter number and instruction which needs lowering.
    // Stored in order by parameter number.
    std::map<int64, HloInstruction*> parameters_to_replace;

    // Check if any operands need lowering.
    for (int64 operand_idx = 0; operand_idx != stage->operand_count();
         ++operand_idx) {
      HloInstruction* operand = stage->mutable_operand(operand_idx);
      TF_ASSIGN_OR_RETURN(bool operand_needs_lowering,
                          analysis->HasToBeLowered(operand));
      if (operand_needs_lowering) {
        parameters_to_replace[operand_idx] = operand;
      }
    }

    // Skip if nothing needs lowering.
    if (parameters_to_replace.empty()) {
      continue;
    }
    VLOG(3) << "Lowering inputs for stage " << stage_id;
    changed = true;
    // Build a cluster of instructions which are lowered and order them.
    // Start from the operands and build the cluster by going through operands
    // only.
    std::vector<HloInstruction*> ordered_lowering;
    std::vector<const HloValueSet*> value_sets;
    absl::flat_hash_set<HloInstruction*> to_visit;
    absl::flat_hash_set<HloInstruction*> visited;
    for (auto& pair : parameters_to_replace) {
      to_visit.insert(pair.second);
    }

    while (!to_visit.empty()) {
      HloInstruction* inst = *to_visit.begin();
      ordered_lowering.push_back(inst);
      to_visit.erase(inst);
      value_sets.push_back(&analysis->GetValueSet(inst));

      // Add any operand which has to be lowered.
      for (HloInstruction* operand : inst->operands()) {
        if (!visited.contains(operand)) {
          TF_ASSIGN_OR_RETURN(bool needs_lowering,
                              analysis->HasToBeLowered(operand));
          if (needs_lowering) {
            to_visit.insert(operand);
          }
        }
      }
    }
    HloValueSet value_set;
    value_set.AssignUnionOf(value_sets);

    // Verify we can lower this.
    TF_RETURN_IF_ERROR(analysis->VerifyPipelineStageOperands(stage, value_set));

    // Make sure the lowering is in post order.
    absl::c_reverse(ordered_lowering);
    // Lower the instructions into the computation.
    TF_ASSIGN_OR_RETURN(stage,
                        AddInstructionsToPipelineStage(stage, ordered_lowering,
                                                       parameters_to_replace));
    // Check that after lowering the parameters are now unused.
    TF_ASSIGN_OR_RETURN(std::set<int64> unused_parameters,
                        GetUnusedParametersInPipelineStage(stage));
    bool lowered_all_params =
        parameters_to_replace.size() <= unused_parameters.size() &&
        absl::c_all_of(
            parameters_to_replace,
            [&unused_parameters](std::pair<int64, HloInstruction*> replaced) {
              return unused_parameters.find(replaced.first) !=
                     unused_parameters.end();
            });
    if (!lowered_all_params) {
      return InternalErrorStrCat("Failed to lower inputs for PipelineStage ",
                                 stage->ToString());
    }
    TF_ASSIGN_OR_RETURN(stage,
                        RemoveParametersFromStage(stage, unused_parameters));
    TF_RETURN_IF_ERROR(UpdateStage(stage_id, stage));
    // Recompute the analysis.
    TF_ASSIGN_OR_RETURN(analysis,
                        PipelineDataflowAnalysis::GetAnalysis(stages_));
  }
  return changed;
}

// Lowers any usages of paramaters into stages.
// Returns true if the stage has changed.
StatusOr<bool> PipelineFixer::LowerParameterUsagesIntoStages() {
  bool changed = false;

  TF_ASSIGN_OR_RETURN(auto analysis,
                      PipelineDataflowAnalysis::GetAnalysis(stages_));
  std::vector<HloInstruction*> ordered_stages = GetOrderedStages();
  // Iterate in reverse order so that we lower any changes affecting parameters
  // into the backward stages first if possible.
  for (auto itr = ordered_stages.rbegin(); itr != ordered_stages.rend();
       ++itr) {
    HloInstruction* stage = *itr;
    TF_ASSIGN_OR_RETURN(StageID stage_id, analysis->GetStageID(stage));
    // Get the corresponding forward stage.
    HloInstruction* fwd_stage = stages_.forward[stage_id.id];
    // Go through the operands to the forward stage.
    absl::flat_hash_set<HloInstruction*> params_users;
    for (HloInstruction* operand : fwd_stage->operands()) {
      if (operand->opcode() == HloOpcode::kParameter) {
        // For parameters, we add all their non-pipeline stage users as
        // potential things we need to lower into the backward stage.
        absl::c_copy_if(operand->users(),
                        std::inserter(params_users, std::begin(params_users)),
                        [](const HloInstruction* inst) {
                          return !IsPiplineStageOrBackwardOp(inst);
                        });
      }
    }
    while (!params_users.empty()) {
      HloInstruction* param_user = *params_users.begin();
      params_users.erase(param_user);
      TF_ASSIGN_OR_RETURN(bool needs_lowering,
                          analysis->HasToBeLowered(param_user));
      if (!needs_lowering) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(
          std::vector<HloInstruction*> ordered_lowering,
          FindClusterToLower(stage, stage_id, param_user, analysis.get()));
      // Nothing to lower.
      if (ordered_lowering.empty()) {
        continue;
      }
      VLOG(3) << "Lowering parameters for stage " << stage_id;
      changed = true;
      // Prevent use after free in the outter loop.
      absl::c_for_each(ordered_lowering, [&params_users](HloInstruction* inst) {
        params_users.erase(inst);
      });

      // Lower the instructions into the computation.
      TF_ASSIGN_OR_RETURN(
          stage, AddInstructionsToPipelineStage(stage, ordered_lowering));

      TF_RETURN_IF_ERROR(UpdateStage(stage_id, stage));
      // Recompute the analysis.
      TF_ASSIGN_OR_RETURN(analysis,
                          PipelineDataflowAnalysis::GetAnalysis(stages_));
    }
  }
  return changed;
}

StatusOr<bool> PipelineFixer::LowerOpsIntoPipelineStages() {
  // Lower any outputs from stages into stages if possible.
  TF_ASSIGN_OR_RETURN(bool lowered_outputs, LowerPipelineStagesOutputs());
  // Lower any inputs into stages if possible.
  TF_ASSIGN_OR_RETURN(bool lowered_inputs, LowerPipelineStagesInputs());
  // Lower any usages of parameters into stages if possible.
  TF_ASSIGN_OR_RETURN(bool lowered_params_uses,
                      LowerParameterUsagesIntoStages());

  return lowered_inputs || lowered_outputs || lowered_params_uses;
}

StatusOr<bool> PipelineFixer::FixPipeline(HloInstruction* pipeline_op) {
  HloComputation* pipeline_comp = pipeline_op->to_apply();

  TF_ASSIGN_OR_RETURN(stages_, GetPipelineStages(pipeline_comp));
  // Go through the stages and insert stateful no-ops to make sure DCE does not
  // remove stages. This is usually caused by the constant propagation in TF2XLA
  // layer.
  // We do not want to remove the stages because later stages of this pass will
  // lower ops/thread ops through stages.
  TF_RETURN_IF_ERROR(InsertStatefulNoopsIntoStages());
  // Clean up the pipelines.
  TupleSimplifier ts(true);
  TF_ASSIGN_OR_RETURN(bool ts_change, ts.Run(pipeline_op->GetModule()));
  HloDCE dce;
  TF_ASSIGN_OR_RETURN(bool dce_change_run_one,
                      dce.Run(pipeline_op->GetModule()));

  // Duplicate edges - this makes analysis easier.
  TF_ASSIGN_OR_RETURN(bool added_edges, DuplicateGTEEdges(stages_));
  // Uniquify computations called by stages.
  TF_ASSIGN_OR_RETURN(bool added_comp, UniquifyPipelineStageCallsites(stages_));
  // Verify we can actually try and lower this Pipeline.
  TF_RETURN_IF_ERROR(VerifyPipelineStagesBeforeLowering(stages_));
  // Run the lowering.
  TF_ASSIGN_OR_RETURN(bool pipeline_modified, LowerOpsIntoPipelineStages());
  // Tidy again.
  TF_ASSIGN_OR_RETURN(bool dce_change_run_two,
                      dce.Run(pipeline_op->GetModule()));
  // Verify the pipeline is now ok to be lowered.
  TF_RETURN_IF_ERROR(VerifyPipelineStagesAfterLowering(pipeline_op));

  return ts_change || dce_change_run_one || added_edges || added_comp ||
         pipeline_modified || dce_change_run_two;
}

StatusOr<bool> PipelineFixer::Run(HloModule* module) {
  std::vector<HloInstruction*> pipeline_ops;
  for (HloComputation* comp : module->MakeNonfusionComputations()) {
    for (HloInstruction* inst : comp->instructions()) {
      if (IsPipelineOp(inst)) {
        pipeline_ops.push_back(inst);
      }
    }
  }

  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to fix.
    return false;
  } else if (pipeline_ops.size() > 1) {
    return FailedPrecondition(
        "Only a single ipu.pipeline() is allowed in a compiled program - if "
        "multiple pipelines are required the program needs to be split into "
        "multiple compilations.");
  }
  VLOG(2) << "Before fixing the Pipeline stages.";
  XLA_VLOG_LINES(2, module->ToString());

  TF_ASSIGN_OR_RETURN(bool changed, FixPipeline(pipeline_ops[0]));

  if (changed) {
    VLOG(2) << "After fixing the Pipeline stages.";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the Pipeline.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
