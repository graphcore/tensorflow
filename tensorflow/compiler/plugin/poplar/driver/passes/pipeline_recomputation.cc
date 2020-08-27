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
#include <stack>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/fifo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_noop.h"
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
// Helper struct for storing information about which forward stage outputs are
// used as backward stage inputs and the corresponding output instructions.
struct OutputToInputInfo {
  std::vector<int64> bwd_input_idices;
  std::vector<int64> fwd_output_idices;
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

  for (int64 op_idx = 0; op_idx != bwd_stage->operand_count(); ++op_idx) {
    const HloInstruction* operand = bwd_stage->operand(op_idx);
    switch (operand->opcode()) {
      case HloOpcode::kGetTupleElement: {
        const HloInstruction* source = operand->operand(0);
        if (source == fwd_stage) {
          const int64 output_idx = operand->tuple_index();
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

// Helper struct for storing the information about the cluster for
// recomputation.
struct ClusterInfo {
  // Stores the inputs to the cluster.
  std::vector<HloInstruction*> inputs;
  // All the instructions which can be recomputed - stored in post order.
  std::vector<HloInstruction*> instructions;
};

StatusOr<ClusterInfo> GetRecomputationCluster(
    HloInstruction* stage, const OutputToInputInfo& oi_info) {
  HloComputation* comp = stage->to_apply();
  HloInstruction* root = comp->root_instruction();
  CHECK_EQ(root->opcode(), HloOpcode::kTuple);

  // Given the forward stage outputs which are used in the backward stage,
  // build a cluster of instructions which can be recomputed inside of the
  // backward stage.
  std::vector<HloInstruction*> inputs;
  std::vector<HloInstruction*> instructions;

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
        if (is_cluster_input(inst)) {
          inputs.push_back(inst);
        } else {
          instructions.push_back(inst);
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

  // Make sure each input shape is an array(tensor).
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
  VLOG(2) << "Cluster is:";
  for (const HloInstruction* inst : instructions) {
    VLOG(2) << "* " << inst->ToString();
  }

  return ClusterInfo{inputs, instructions};
}

Status AddNewOutputsToStage(
    HloInstruction* const stage,
    const std::vector<HloInstruction*>& outputs_to_add) {
  HloComputation* comp = stage->to_apply();
  HloInstruction* old_root = comp->root_instruction();
  CHECK_EQ(old_root->opcode(), HloOpcode::kTuple);

  HloInstruction::InstructionVector new_outputs = old_root->operands();
  new_outputs.insert(new_outputs.end(), outputs_to_add.begin(),
                     outputs_to_add.end());

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
  HloCloneContext context(fwd_stage->GetModule());

  // The recomputation cluster is added to the pipeline computation and then
  // lowered into the backward pipeline stage.
  HloComputation* pipeline_comp = fwd_stage->parent();

  // Get the cluster inputs which were previously added as outputs of the
  // forward stage.
  const int64 start_index = ShapeUtil::TupleElementCount(fwd_stage->shape()) -
                            cluster_info.inputs.size();
  for (int64 i = 0; i != cluster_info.inputs.size(); ++i) {
    TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                        MakeGetTupleElementHlo(fwd_stage, start_index + i));
    context.MapInstruction(cluster_info.inputs[i], gte);
  }

  std::vector<HloInstruction*> to_lower;
  to_lower.reserve(cluster_info.instructions.size());
  for (HloInstruction* old_inst : cluster_info.instructions) {
    std::vector<HloInstruction*> new_operands(old_inst->operand_count());
    absl::c_transform(old_inst->operands(), new_operands.begin(),
                      [&context](HloInstruction* old_operand) {
                        return context.GetInstruction(old_operand);
                      });
    HloInstruction* new_inst = pipeline_comp->AddInstruction(
        old_inst->CloneWithNewOperands(old_inst->shape(), new_operands));

    context.MapInstruction(old_inst, new_inst);
    to_lower.push_back(new_inst);
  }

  // Lower the instructions into the backward pipeline stage, replacing all the
  // uses of the fwd stage with the outputs of the recomputation cluster.
  std::map<int64, HloInstruction*> replacements;
  for (int64 i = 0; i != oi_info.fwd_outputs.size(); ++i) {
    HloInstruction* new_output =
        context.GetInstruction(oi_info.fwd_outputs.at(i));
    replacements.emplace(oi_info.bwd_input_idices.at(i), new_output);
  }

  TF_ASSIGN_OR_RETURN(bwd_stage, AddInstructionsToPipelineStage(
                                     bwd_stage, to_lower, replacements));
  return Status::OK();
}
}  // namespace

PipelineRecomputation::PipelineRecomputation(bool allow_recomputation)
    : allow_recomputation_(allow_recomputation) {}

StatusOr<bool> PipelineRecomputation::RecomputePipeline(
    HloInstruction* pipeline_op) {
  HloComputation* pipeline_comp = pipeline_op->to_apply();
  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  TF_RETURN_IF_ERROR(FixRootInstructions(stages));

  // Do not perform recomputation if there are no backward stages.
  if (stages.backward.empty()) {
    return false;
  }

  bool changed = false;
  // Go through all the forward stages (apart from the last one which does not
  // need recomputation).
  for (int64 stage_id = 0;
       stage_id != static_cast<int64>(stages.forward.size()) - 1; ++stage_id) {
    HloInstruction* fwd_stage = stages.forward[stage_id];
    HloInstruction* bwd_stage = stages.backward[stage_id];

    // Find all the forward outputs used by the backward pass.
    TF_ASSIGN_OR_RETURN(OutputToInputInfo oi_info,
                        GetForwardOutputsUsed(fwd_stage, bwd_stage));

    if (oi_info.fwd_output_idices.empty()) {
      continue;
    }

    // Find a cluster which can be recomputed.
    TF_ASSIGN_OR_RETURN(ClusterInfo cluster_info,
                        GetRecomputationCluster(fwd_stage, oi_info));

    if (cluster_info.instructions.empty()) {
      LOG(INFO) << "Cannot recompute pipline stage " << fwd_stage->ToString();
      continue;
    }

    // Make all the cluster inputs as outputs of the forward pipeline stage.
    TF_RETURN_IF_ERROR(AddNewOutputsToStage(fwd_stage, cluster_info.inputs));

    // Lower the recomputation cluster into the backward stage.
    TF_RETURN_IF_ERROR(
        AddClusterToBackwardStage(fwd_stage, bwd_stage, cluster_info, oi_info));
    changed = true;
  }
  return changed;
}

StatusOr<bool> PipelineRecomputation::Run(HloModule* module) {
  if (!allow_recomputation_) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> pipeline_ops,
                      GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to fix.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);

  TF_ASSIGN_OR_RETURN(const auto schedule,
                      GetPipelineSchedule(pipeline_ops[0]));
  if (schedule != PoplarBackendConfig::CallConfig::PipelineConfig::Sequential) {
    VLOG(2) << "Non sequential schedules use "
               "PipelineRecomputationStageInserter for recomputation.";
    return false;
  }

  VLOG(2) << "Before PipelineRecomputation:";
  XLA_VLOG_LINES(2, module->ToString(HloPrintOptions::ShortParsable()));

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
