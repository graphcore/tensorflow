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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_recomputation.h"

#include <map>
#include <memory>
#include <queue>
#include <stack>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/recompute.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {
bool IsRecomputationCheckpoint(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::RecomputationCheckpoint, inst);
}

// Helper struct for storing information about which forward stage outputs are
// used as backward stage inputs and the corresponding output instructions.
struct OutputToInputInfo {
  std::vector<int64_t> bwd_input_idices;
  std::vector<int64_t> fwd_output_idices;
  std::vector<HloInstruction*> fwd_outputs;
};

// Go through all the inputs to the backward stage, and find all the ones which
// are outputs of the forward stage.
StatusOr<OutputToInputInfo> GetForwardOutputsUsed(
    const HloInstruction* fwd_stage, const HloInstruction* bwd_stage) {
  OutputToInputInfo info;

  HloComputation* fwd_comp = fwd_stage->to_apply();
  HloInstruction* fwd_root = fwd_comp->root_instruction();
  CHECK_EQ(fwd_root->opcode(), HloOpcode::kTuple);

  for (int64_t op_idx = 0; op_idx != bwd_stage->operand_count(); ++op_idx) {
    const HloInstruction* operand = bwd_stage->operand(op_idx);
    switch (operand->opcode()) {
      case HloOpcode::kGetTupleElement: {
        const HloInstruction* source = operand->operand(0);
        if (source == fwd_stage) {
          const int64_t output_idx = operand->tuple_index();
          info.bwd_input_idices.push_back(op_idx);
          info.fwd_output_idices.push_back(output_idx);
          info.fwd_outputs.push_back(fwd_root->mutable_operand(output_idx));
        }
        break;
      }
      case HloOpcode::kParameter: {
        // We don't need to do anything for parameters.
        break;
      }
      case HloOpcode::kCustomCall: {
        if (IsPoplarInstruction(PoplarOp::ExecutionCounter)(operand) ||
            IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(operand)) {
          // We don't need to do anything for these creators.
          break;
        }
        TF_FALLTHROUGH_INTENDED;
      }
      default: {
        return InternalErrorStrCat("Invalid input ", operand->ToString(),
                                   " to pipeline stage ", bwd_stage->ToString(),
                                   ".");
      }
    }
  }
  return info;
}

// Get the final checkpoints in the stage. These are the last checkpoints that
// can reach the root. If a checkpoint can reach another checkpoint, then it
// is not final.
std::vector<const HloInstruction*> GetFinalCheckpoints(
    const HloComputation* stage, HloReachabilityMap* reachability_map) {
  // Find all checkpoints in the stage
  std::vector<const HloInstruction*> checkpoints;
  for (const HloInstruction* inst : stage->instructions()) {
    if (IsRecomputationCheckpoint(inst)) {
      checkpoints.push_back(inst);
    }
  }
  // Exclude any checkpoints that can reach another checkpoint
  std::vector<const HloInstruction*> final_checkpoints;
  for (const HloInstruction* ckpt : checkpoints) {
    bool is_final = true;
    for (const HloInstruction* other_ckpt : checkpoints) {
      if (ckpt != other_ckpt) {
        if (reachability_map->IsReachable(ckpt, other_ckpt)) {
          is_final = false;
          break;
        }
      }
    }
    if (is_final) {
      final_checkpoints.push_back(ckpt);
    }
  }
  return final_checkpoints;
}

// Helper struct for storing the information about the cluster for
// recomputation.
struct ClusterInfo {
  // Stores the inputs to the cluster.
  std::vector<HloInstruction*> inputs;
  // Stores the recomputation checkpoint instructions which become inputs.
  std::vector<HloInstruction*> recomputation_checkpoints;
  // All the instructions which can be recomputed - stored in post order.
  std::vector<HloInstruction*> instructions;
};

StatusOr<ClusterInfo> GetRecomputationCluster(HloInstruction* stage,
                                              const OutputToInputInfo& oi_info,
                                              bool last_stage) {
  HloComputation* comp = stage->to_apply();
  HloInstruction* root = comp->root_instruction();
  CHECK_EQ(root->opcode(), HloOpcode::kTuple);

  // Given the forward stage outputs which are used in the backward stage,
  // build a cluster of instructions which can be recomputed inside of the
  // backward stage.
  std::vector<HloInstruction*> inputs;
  std::vector<HloInstruction*> recomputation_checkpoints;
  std::vector<HloInstruction*> instructions;

  // In the last stage, we don't want to recompute anything that's going to be
  // immediately used by the backward pass (a.k.a. is after the final
  // checkpoints in the stage).
  std::vector<const HloInstruction*> final_checkpoints;
  std::unique_ptr<HloReachabilityMap> reachability_map =
      HloReachabilityMap::Build(comp);
  if (last_stage) {
    final_checkpoints = GetFinalCheckpoints(comp, reachability_map.get());
  }

  // For non-final stages, everything visited is added to the cluster. In the
  // final stage, we only add an inst to the cluster if it's after all the last
  // checkpoints in the stage.
  auto should_be_recomputed = [&final_checkpoints, &reachability_map,
                               last_stage](const HloInstruction* inst) -> bool {
    if (!last_stage) {
      return true;
    }
    // If the inst can reach any of the terminal checkpoints, it's behind one
    // and should be recomputed.
    for (const HloInstruction* ckpt : final_checkpoints) {
      if (reachability_map->IsReachable(inst, ckpt)) {
        return true;
      }
    }
    return false;
  };

  // Walk through the cluster to create a post order.
  std::stack<HloInstruction*> worklist;
  absl::c_for_each(oi_info.fwd_outputs,
                   [&worklist](HloInstruction* inst) { worklist.push(inst); });

  enum VisitState { kVisiting, kVisited };
  absl::flat_hash_map<HloInstruction*, VisitState> visited;

  auto is_cluster_input = [](const HloInstruction* a) -> bool {
    if (a->HasSideEffect()) {
      return true;
    }
    if (a->opcode() == HloOpcode::kParameter) {
      return true;
    }
    if (a->opcode() == HloOpcode::kGetTupleElement &&
        a->operand(0)->HasSideEffect()) {
      return true;
    }
    return false;
  };

  // Traverse the graph to create a post order - visit the inputs before
  // visiting the current node.
  while (!worklist.empty()) {
    HloInstruction* inst = worklist.top();

    // Visit inputs straight away.
    if (is_cluster_input(inst)) {
      visited.insert({inst, kVisiting});
    }

    auto itr = visited.find(inst);
    if (itr != visited.end()) {
      worklist.pop();
      if (itr->second == kVisiting) {
        if (should_be_recomputed(inst)) {
          if (is_cluster_input(inst)) {
            inputs.push_back(inst);
          } else {
            instructions.push_back(inst);
          }

          // Store which instructions are checkpoints.
          if (IsRecomputationCheckpoint(inst)) {
            const bool only_root_user =
                inst->users().size() == 1 && inst->users()[0] == root;
            if (only_root_user) {
              VLOG(2) << "Skipping checkpoint " << inst->ToString()
                      << " as it is only a stage output.";
            } else if (inst->operand(0)->opcode() == HloOpcode::kParameter) {
              VLOG(2) << "Skipping checkpoint " << inst->ToString()
                      << " as it is a stage input.";
            } else {
              recomputation_checkpoints.push_back(inst);
            }
          }
        }

        itr->second = kVisited;
      } else {
        CHECK_EQ(itr->second, kVisited);
      }
      continue;
    }

    visited.insert({inst, kVisiting});

    for (HloInstruction* operand : inst->operands()) {
      worklist.push(operand);
    }
  }

  // Make sure each input shape is an array(tensor) - i.e. prevent token shapes.
  const bool all_shapes_allowed =
      absl::c_all_of(inputs, [](const HloInstruction* inst) {
        return absl::c_all_of(ShapeUtil::GetLeafShapes(inst->shape()),
                              [](const ShapeUtil::IndexedShape& is) {
                                return is.shape.IsArray();
                              });
      });

  if (!all_shapes_allowed) {
    return ClusterInfo{};
  }

  VLOG(2) << "Cluster inputs are:";
  for (const HloInstruction* inst : inputs) {
    VLOG(2) << "* " << inst->ToString();
  }
  VLOG(2) << "Cluster recomputation checkpoints are:";
  for (const HloInstruction* inst : recomputation_checkpoints) {
    VLOG(2) << "* " << inst->ToString();
  }
  VLOG(2) << "Cluster is:";
  for (const HloInstruction* inst : instructions) {
    VLOG(2) << "* " << inst->ToString();
  }

  return ClusterInfo{inputs, recomputation_checkpoints, instructions};
}

Status AddNewOutputsToStage(HloInstruction* const stage,
                            const ClusterInfo& cluster_info) {
  HloComputation* comp = stage->to_apply();
  HloInstruction* old_root = comp->root_instruction();
  CHECK_EQ(old_root->opcode(), HloOpcode::kTuple);

  HloInstruction::InstructionVector new_outputs = old_root->operands();
  new_outputs.insert(new_outputs.end(), cluster_info.inputs.begin(),
                     cluster_info.inputs.end());
  new_outputs.insert(new_outputs.end(),
                     cluster_info.recomputation_checkpoints.begin(),
                     cluster_info.recomputation_checkpoints.end());

  HloInstruction* new_root =
      comp->AddInstruction(HloInstruction::CreateTuple(new_outputs));

  // Replace the root with the new shape with a different shape.
  comp->set_root_instruction(new_root, true);
  TF_RETURN_IF_ERROR(comp->RemoveInstruction(old_root));

  // Update the stage shape.
  *stage->mutable_shape() = new_root->shape();

  return Status::OK();
}

Status AddClusterToBackwardStage(HloInstruction* const fwd_stage,
                                 HloInstruction* bwd_stage,
                                 const ClusterInfo& cluster_info,
                                 const OutputToInputInfo& oi_info) {
  const int64_t num_inputs = cluster_info.inputs.size();
  const int64_t num_recomputed_inputs =
      cluster_info.recomputation_checkpoints.size();

  HloCloneContext context(fwd_stage->GetModule());

  // The recomputation cluster is added to the pipeline computation and then
  // lowered into the backward pipeline stage.
  HloComputation* pipeline_comp = fwd_stage->parent();

  // Keep track of inputs to the cluster.
  absl::flat_hash_set<HloInstruction*> cluster_inputs;

  // Get the cluster inputs which were previously added as outputs of the
  // forward stage.
  {
    const int64_t start_index =
        ShapeUtil::TupleElementCount(fwd_stage->shape()) - num_inputs -
        num_recomputed_inputs;
    for (int64_t i = 0; i != num_inputs; ++i) {
      HloInstruction* input = cluster_info.inputs[i];
      TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                          MakeGetTupleElementHlo(fwd_stage, start_index + i));
      context.MapInstruction(input, gte);
      cluster_inputs.insert(input);
    }
  }

  // Get the checkpointed inputs.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> recomputed_inputs;
  {
    const int64_t start_index =
        ShapeUtil::TupleElementCount(fwd_stage->shape()) -
        num_recomputed_inputs;
    for (int64_t i = 0; i != num_recomputed_inputs; ++i) {
      HloInstruction* input = cluster_info.recomputation_checkpoints[i];
      TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                          MakeGetTupleElementHlo(fwd_stage, start_index + i));
      context.MapInstruction(input, gte);
      recomputed_inputs[input] = gte;
    }
  }

  std::vector<HloInstruction*> to_lower;
  to_lower.reserve(cluster_info.instructions.size());
  for (HloInstruction* old_inst : cluster_info.instructions) {
    HloInstruction* new_inst;
    if (recomputed_inputs.contains(old_inst)) {
      // Convert valid recomputation checkpoint instruction into recomputation
      // input instruction.
      CHECK(IsRecomputationCheckpoint(old_inst));
      HloInstruction* checkpointed_input = recomputed_inputs.at(old_inst);
      HloInstruction* old_input =
          context.GetInstruction(old_inst->mutable_operand(0));
      new_inst = pipeline_comp->AddInstruction(
          CreateRecomputationInput(checkpointed_input, old_input));
    } else {
      std::vector<HloInstruction*> new_operands(old_inst->operand_count());
      absl::c_transform(old_inst->operands(), new_operands.begin(),
                        [&context](HloInstruction* old_operand) {
                          return context.GetInstruction(old_operand);
                        });
      new_inst = pipeline_comp->AddInstruction(
          old_inst->CloneWithNewOperands(old_inst->shape(), new_operands));
      OpMetadata metadata = new_inst->metadata();
      metadata.set_op_name(metadata.op_name() + "/Recomputed");
      new_inst->set_metadata(metadata);
    }

    context.MapInstruction(old_inst, new_inst);
    to_lower.push_back(new_inst);
  }

  // Lower the instructions into the backward pipeline stage, replacing all the
  // uses of the fwd stage with the outputs of the recomputation cluster.
  std::map<int64_t, HloInstruction*> replacements;
  for (int64_t i = 0; i != oi_info.fwd_outputs.size(); ++i) {
    HloInstruction* output = oi_info.fwd_outputs.at(i);
    // Do not replace cluster inputs.
    if (cluster_inputs.contains(output)) {
      continue;
    }

    // Only replace if the output is actually part of the cluster (the output
    // might not be part of the cluster if we're in the final fwd stage).
    auto it = absl::c_find(cluster_info.instructions, output);
    if (it == cluster_info.instructions.end()) {
      continue;
    }

    HloInstruction* new_output = context.GetInstruction(output);
    replacements.emplace(oi_info.bwd_input_idices.at(i), new_output);
  }

  TF_ASSIGN_OR_RETURN(bwd_stage, AddInstructionsToPipelineStage(
                                     bwd_stage, to_lower, replacements));
  return Status::OK();
}

// Helper struct for storing the information about the pipeline stage for
// recomputation.
struct PipelineStageRecomputationInfo {
  HloInstruction* fwd_stage;
  HloInstruction* bwd_stage;
  int64_t stage_id;
  bool is_last_stage;
  OutputToInputInfo oi_info;
  ClusterInfo cluster_info;
};

// Function which finds all the pipeline stages and their clusters to recompute.
StatusOr<std::vector<PipelineStageRecomputationInfo>> GetRecomputationInfos(
    HloInstruction* pipeline_op) {
  HloComputation* pipeline_comp = pipeline_op->to_apply();
  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  TF_RETURN_IF_ERROR(FixRootInstructions(stages));

  std::vector<PipelineStageRecomputationInfo> stage_recomputation_infos;
  // Do not perform recomputation if there are no backward stages.
  if (stages.backward.empty()) {
    return stage_recomputation_infos;
  }

  // Go through all the forward stages.
  const int64_t num_stages = static_cast<int64_t>(stages.forward.size());
  for (int64_t stage_id = 0; stage_id != num_stages; ++stage_id) {
    HloInstruction* fwd_stage = stages.forward[stage_id];
    HloInstruction* bwd_stage = stages.backward[stage_id];
    const bool is_last_stage = stage_id == num_stages - 1;

    // Find all the forward outputs used by the backward pass.
    TF_ASSIGN_OR_RETURN(OutputToInputInfo oi_info,
                        GetForwardOutputsUsed(fwd_stage, bwd_stage));

    // Only recompute if the bwd stage needs something from the fwd stage.
    if (oi_info.fwd_output_idices.empty()) {
      continue;
    }

    // Find a cluster which can be recomputed.
    // In the last stage, make sure we don't recompute anything past the last
    // checkpoint.
    TF_ASSIGN_OR_RETURN(
        ClusterInfo cluster_info,
        GetRecomputationCluster(fwd_stage, oi_info, is_last_stage));

    stage_recomputation_infos.push_back(PipelineStageRecomputationInfo{
        fwd_stage, bwd_stage, stage_id, is_last_stage, oi_info, cluster_info});
  }

  return stage_recomputation_infos;
}
}  // namespace

PipelineRecomputation::PipelineRecomputation(bool allow_recomputation)
    : allow_recomputation_(allow_recomputation) {}

StatusOr<bool> PipelineRecomputation::RecomputePipeline(
    HloInstruction* pipeline_op) {
  TF_ASSIGN_OR_RETURN(auto stage_recomputation_infos,
                      GetRecomputationInfos(pipeline_op));

  bool changed = false;
  for (PipelineStageRecomputationInfo& stage_recomputation_info :
       stage_recomputation_infos) {
    HloInstruction* fwd_stage = stage_recomputation_info.fwd_stage;
    HloInstruction* bwd_stage = stage_recomputation_info.bwd_stage;
    const int64_t stage_id = stage_recomputation_info.stage_id;
    const bool is_last_stage = stage_recomputation_info.is_last_stage;
    const OutputToInputInfo& oi_info = stage_recomputation_info.oi_info;
    const ClusterInfo& cluster_info = stage_recomputation_info.cluster_info;

    if (cluster_info.instructions.empty()) {
      if (cluster_info.recomputation_checkpoints.size()) {
        LOG(INFO) << "Found checkpoint instructions in pipeline stage "
                  << stage_id
                  << ", however could not find any operations which can be "
                     "recomputed.";
      } else if (!is_last_stage) {
        LOG(INFO) << "Cannot recompute pipeline stage " << stage_id << " ("
                  << fwd_stage->ToString() << ")";
      }
      continue;
    }

    // Make all the cluster inputs/recomputation checkpoints as outputs of the
    // forward pipeline stage.
    TF_RETURN_IF_ERROR(AddNewOutputsToStage(fwd_stage, cluster_info));

    // Lower the recomputation cluster into the backward stage.
    TF_RETURN_IF_ERROR(
        AddClusterToBackwardStage(fwd_stage, bwd_stage, cluster_info, oi_info));
    changed = true;
  }

  return changed;
}

StatusOr<std::vector<HloInstruction*>>
PipelineRecomputation::GetInstructionsToRecompute(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
  std::vector<HloInstruction*> instructions_to_recompute;

  for (HloInstruction* pipeline_op : pipeline_ops) {
    TF_ASSIGN_OR_RETURN(const auto recomputation_mode,
                        GetPipelineRecomputationMode(pipeline_op));
    if (recomputation_mode != PoplarBackendConfig::CallConfig::PipelineConfig::
                                  Recompute_and_backpropagate_interleaved) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(auto stage_recomputation_infos,
                        GetRecomputationInfos(pipeline_op));
    for (PipelineStageRecomputationInfo& stage_recomputation_info :
         stage_recomputation_infos) {
      const ClusterInfo& cluster_info = stage_recomputation_info.cluster_info;
      instructions_to_recompute.insert(instructions_to_recompute.end(),
                                       cluster_info.instructions.begin(),
                                       cluster_info.instructions.end());
    }
  }

  return instructions_to_recompute;
}

StatusOr<bool> PipelineRecomputation::Run(HloModule* module) {
  if (!allow_recomputation_) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to fix.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);

  // Check whether this is the requested method of recomputation.
  TF_ASSIGN_OR_RETURN(const auto recomputation_mode,
                      GetPipelineRecomputationMode(pipeline_ops[0]));
  if (recomputation_mode != PoplarBackendConfig::CallConfig::PipelineConfig::
                                Recompute_and_backpropagate_interleaved) {
    return false;
  }

  VLOG(2) << "Before PipelineRecomputation:";
  XLA_VLOG_LINES(2, module->ToString());

  TF_ASSIGN_OR_RETURN(bool changed, RecomputePipeline(pipeline_ops[0]));

  if (changed) {
    VLOG(2) << "After PipelineRecomputation:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the Pipeline.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
