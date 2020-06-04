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
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"

#include <algorithm>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/fifo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/ipu_inter_copy.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {
namespace {
// Returns whether the instruction is a gradient accumulation creator.
bool IsGradientAccumulatorCreate(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(inst);
}
}  // namespace

std::string StageID::ToString() const {
  std::stringstream ss;
  ss << "PipelineStage";
  switch (stage_type) {
    case StageType::kForward: {
      break;
    }
    case StageType::kBackward: {
      ss << "Backward";
      break;
    }
    default:
    case StageType::kRecomputation: {
      ss << "Recomputation";
      break;
    }
  }
  ss << " with ID " << id;
  return ss.str();
}

std::ostream& operator<<(std::ostream& stream, const StageID& stage_id) {
  stream << stage_id.ToString();
  return stream;
}

bool IsPipelineStageOrBackwardOp(const HloInstruction* inst) {
  return IsPipelineStage(inst) || IsPipelineStageBackward(inst);
}

bool IsAnyPipelineStageOp(const HloInstruction* inst) {
  return IsPipelineStageOrBackwardOp(inst) ||
         IsPipelineStageRecomputation(inst);
}

bool IsAnyPipelineStageOpOrResourceUpdate(const HloInstruction* inst) {
  return IsAnyPipelineStageOp(inst) || IsResourceUpdate(inst);
}

bool IsProducerOp(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kCall:
      return IsAnyPipelineStageOp(inst);
    case HloOpcode::kCustomCall:
      return IsGradientAccumulatorCreate(inst);
    case HloOpcode::kParameter:
      return true;
    case HloOpcode::kInfeed:
      return true;
    default:
      return false;
  }
}

StatusOr<std::vector<HloInstruction*>> GetPipelines(const HloModule* module) {
  std::vector<HloInstruction*> pipeline_ops;
  for (auto comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (IsPipelineOp(inst)) {
        pipeline_ops.push_back(inst);
      }
    }
  }

  if (pipeline_ops.size() > 1) {
    return FailedPrecondition(
        "Only a single ipu.pipeline() is allowed in a compiled program - if "
        "multiple pipelines are required the program needs to be split into "
        "multiple compilations.");
  }
  return pipeline_ops;
}

StatusOr<PipelineStages> GetPipelineStages(HloComputation* pipeline_computation,
                                           bool validate_stages) {
  PipelineStages pipeline_stages;
  // Find all the stages - note that they might not be in order as some stages
  // might have no inputs/outputs.
  for (HloInstruction* inst : pipeline_computation->instructions()) {
    if (IsPipelineStage(inst)) {
      pipeline_stages.forward.push_back(inst);
    } else if (IsPipelineStageBackward(inst)) {
      pipeline_stages.backward.push_back(inst);
    } else if (IsPipelineStageRecomputation(inst)) {
      pipeline_stages.recomputation[GetPipelineStageID(inst)] = inst;
    } else if (IsResourceUpdate(inst)) {
      pipeline_stages.resource_update = inst;
    }
  }
  // Sort the stages and make sure the stages are continuos and starting at 0.
  auto sort_stages = [](std::vector<HloInstruction*>& stages) {
    absl::c_sort(stages,
                 [](const HloInstruction* lhs, const HloInstruction* rhs) {
                   return GetPipelineStageID(lhs) < GetPipelineStageID(rhs);
                 });
  };

  auto check_stages = [](std::vector<HloInstruction*>& stages) {
    for (int64 i = 0; i != static_cast<int64>(stages.size()); ++i) {
      const int64 stage_id = GetPipelineStageID(stages[i]);
      if (stage_id != i) {
        return FailedPrecondition(
            "Detected Pipeline Stage with id %d but expected id %d.", stage_id,
            i);
      }
    }
    return Status::OK();
  };
  sort_stages(pipeline_stages.forward);
  sort_stages(pipeline_stages.backward);
  if (validate_stages) {
    TF_RETURN_IF_ERROR(check_stages(pipeline_stages.forward));
    TF_RETURN_IF_ERROR(check_stages(pipeline_stages.backward));

    if (pipeline_stages.forward.empty()) {
      return FailedPrecondition(
          "Expected the pipeline to have at least one PipelineStage.");
    }

    // If we have any bwd pipeline stages then we expect them to match the
    // number of fwd stages (i.e. backprop has a stage for each forward prop).
    if (pipeline_stages.backward.size() &&
        pipeline_stages.forward.size() != pipeline_stages.backward.size()) {
      return FailedPrecondition(
          "Expected the number of PipelineStages (%d) and "
          "PipelineStageBackwards "
          "(%d) to match.",
          pipeline_stages.forward.size(), pipeline_stages.backward.size());
    }

    // We expect the number of recomputation stages to be less than or equal to
    // the number of forward stages.
    if (pipeline_stages.forward.size() < pipeline_stages.recomputation.size()) {
      return FailedPrecondition(
          "Expected the number of PipelineStageRecomputations (%d) to be at "
          "most %d.",
          pipeline_stages.forward.size(), pipeline_stages.recomputation.size());
    }
  }

  if (pipeline_stages.backward.size() && !pipeline_stages.resource_update) {
    return FailedPrecondition(
        "Expected the XLA graph to contain a resource update function in the "
        "Pipelining graph.");
  }

  return pipeline_stages;
}

StatusOr<absl::flat_hash_set<HloComputation*>> GetAllComputationsCalledBy(
    HloInstruction* pipeline_stage, CallGraph* call_graph) {
  CHECK(IsAnyPipelineStageOpOrResourceUpdate(pipeline_stage));
  absl::flat_hash_set<HloComputation*> computations_in_pipeline;
  absl::flat_hash_set<HloComputation*> to_visit;
  to_visit.insert(pipeline_stage->to_apply());
  // We keep separate visited as some computations might be called but we do not
  // want to return them.
  absl::flat_hash_set<HloComputation*> visited;
  while (!to_visit.empty()) {
    HloComputation* comp = *to_visit.begin();
    to_visit.erase(comp);
    // Skip if already visited.
    if (visited.contains(comp)) {
      continue;
    }
    visited.insert(comp);
    // Get the context.
    CallGraphNode& node = call_graph->GetNode(comp);
    // We do not consider sharding in parallel context or fusions.
    if (node.context() == CallContext::kParallel ||
        comp->IsFusionComputation()) {
      continue;
    }
    // Both context is not allowed.
    if (node.context() == CallContext::kBoth) {
      return InternalErrorStrCat(
          "Detected a computation ", comp->name(),
          " with CallContext::kBoth inside the PipelineStage ",
          pipeline_stage->ToString());
    }
    computations_in_pipeline.insert(comp);

    for (HloInstruction* inst : comp->instructions()) {
      // Visit any called computations.
      absl::c_copy(inst->called_computations(),
                   std::inserter(to_visit, to_visit.end()));
    }
  }
  return computations_in_pipeline;
}

Status VerifyPipelineStagesBeforeFixing(const PipelineStages& pipeline_stages) {
  auto is_stage_ok = [](const HloInstruction* stage) {
    HloOpcode root_opcode = stage->to_apply()->root_instruction()->opcode();
    if (root_opcode != HloOpcode::kTuple) {
      return UnimplementedStrCat("Expected the PipelineStage(Backward) ",
                                 stage->ToString(),
                                 " to have a Tuple root instruction but got ",
                                 HloOpcodeString(root_opcode), " instead.");
    }
    if (!absl::c_all_of(stage->users(), [](HloInstruction* user) {
          return user->opcode() == HloOpcode::kGetTupleElement;
        })) {
      return UnimplementedStrCat(
          "Expected all the users of the PipelineStage(Backward) ",
          stage->ToString(), " to be GetTupleElement instructions.");
    }
    if (stage->parent()->root_instruction() == stage) {
      return UnimplementedStrCat(
          "Pipeline stage cannot be the root instruction of the Pipeline.");
    }
    return Status::OK();
  };
  for (HloInstruction* forward_stage : pipeline_stages.forward) {
    TF_RETURN_IF_ERROR(is_stage_ok(forward_stage));
    // We expect forward stages to have supported sharding on them.
    if (!forward_stage->has_sharding()) {
      return FailedPrecondition(
          "Expected the pipeline stage %s to have sharding.",
          forward_stage->ToShortString().c_str());
    }
    if (!IsSupportedSharding(forward_stage->sharding())) {
      return FailedPrecondition("Unsupported sharding for pipeline stage %s.",
                                forward_stage->ToShortString().c_str());
    }
  }
  for (HloInstruction* backward_stage : pipeline_stages.backward) {
    TF_RETURN_IF_ERROR(is_stage_ok(backward_stage));
    // We expect the backward stages to not have any sharding.
    if (backward_stage->has_sharding()) {
      return FailedPrecondition(
          "Expected the pipeline stage %s to not have sharding.",
          backward_stage->ToShortString().c_str());
    }
  }
  return Status::OK();
}

Status FixRootInstructions(const PipelineStages& pipeline_stages) {
  const uint64 num_stages = pipeline_stages.forward.size() +
                            pipeline_stages.backward.size() +
                            (pipeline_stages.resource_update ? 1 : 0);
  std::vector<const HloInstruction*> stages(num_stages);
  absl::c_copy(pipeline_stages.forward, stages.begin());
  absl::c_copy(pipeline_stages.backward,
               std::next(stages.begin(), pipeline_stages.forward.size()));
  if (pipeline_stages.resource_update) {
    stages.back() = *pipeline_stages.resource_update;
  }
  for (const HloInstruction* stage : stages) {
    HloComputation* comp = stage->to_apply();
    HloInstruction* root = comp->root_instruction();
    if (root->opcode() == HloOpcode::kTuple) {
      continue;
    }
    if (!root->shape().IsTuple()) {
      return UnimplementedStrCat("Expected the PipelineStage(Backward) ",
                                 stage->ToString(),
                                 " to have a tuple output shape.");
    }
    // Create a GTE from each subshape.
    const int64 num_elements = ShapeUtil::TupleElementCount(root->shape());
    std::vector<HloInstruction*> gtes(num_elements);
    for (int64 tuple_index = 0; tuple_index != num_elements; ++tuple_index) {
      TF_ASSIGN_OR_RETURN(gtes[tuple_index],
                          MakeGetTupleElementHlo(root, tuple_index));
      if (root->has_sharding()) {
        // If there is any sharding, then forward it.
        gtes[tuple_index]->set_sharding(root->sharding().GetSubSharding(
            root->shape(), ShapeIndex{tuple_index}));
      }
    }
    // Create the root tuple.
    HloInstruction* new_root =
        comp->AddInstruction(HloInstruction::CreateTuple(gtes));
    root->SetupDerivedInstruction(new_root);
    comp->set_root_instruction(new_root);
  }
  return Status::OK();
}

Status VerifyPipelineAfterFixing(HloInstruction* pipeline_op) {
  HloComputation* pipeline_computation = pipeline_op->to_apply();
  TF_ASSIGN_OR_RETURN(PipelineStages pipeline_stages,
                      GetPipelineStages(pipeline_computation));
  // Get the analysis.
  TF_ASSIGN_OR_RETURN(auto analysis,
                      PipelineDataflowAnalysis::GetAnalysis(pipeline_stages));
  // Make sure all the instructions in the pipeline_computation do not require
  // lowering.
  for (HloInstruction* inst : pipeline_computation->instructions()) {
    TF_ASSIGN_OR_RETURN(bool needs_lowering, analysis->HasToBeLowered(inst));
    if (needs_lowering) {
      return UnimplementedStrCat(
          "Detected instruction ", inst->ToString(),
          " which should have been lowered into a PipelineStage.");
    }
  }
  return Status::OK();
}

StatusOr<bool> InsertGTEEdges(PipelineStages& pipeline_stages) {
  bool added_edges = false;
  for (auto& stages : {pipeline_stages.forward, pipeline_stages.backward}) {
    for (HloInstruction* stage : stages) {
      HloComputation* comp = stage->parent();
      std::vector<HloInstruction*> users = stage->users();
      for (HloInstruction* user : users) {
        // User is a GTE, we don't need to do anything.
        if (user->opcode() == HloOpcode::kGetTupleElement) {
          continue;
        }
        // User is not a GTE, insert a GTE for each tuple element.
        int64 num_elements = ShapeUtil::TupleElementCount(stage->shape());
        std::vector<HloInstruction*> gtes(num_elements);
        for (int64 tuple_index = 0; tuple_index != num_elements;
             ++tuple_index) {
          TF_ASSIGN_OR_RETURN(gtes[tuple_index],
                              MakeGetTupleElementHlo(stage, tuple_index));
        }
        // Create a tuple.
        HloInstruction* tuple =
            comp->AddInstruction(HloInstruction::CreateTuple(gtes));

        // Replace the usage of stage with the new tuple.
        TF_RETURN_IF_ERROR(stage->ReplaceUseWith(user, tuple));
        added_edges = true;
      }
    }
  }
  return added_edges;
}

StatusOr<bool> DuplicateGTEEdges(PipelineStages& pipeline_stages) {
  bool added_edges = false;
  for (auto& stages : {pipeline_stages.forward, pipeline_stages.backward}) {
    for (HloInstruction* stage : stages) {
      std::vector<HloInstruction*> gtes = stage->users();
      for (HloInstruction* gte : gtes) {
        // We expect GTEs because we did the TF2XLA lowering.
        if (gte->opcode() != HloOpcode::kGetTupleElement) {
          return FailedPrecondition(
              "Expected user of a PipelineStage(Backward) to be a GTE, but got "
              "%s instead.",
              gte->ToString());
        }
        if (gte->user_count() == 1) {
          continue;
        }
        HloComputation* comp = gte->parent();

        std::vector<HloInstruction*> gte_users = gte->users();
        for (HloInstruction* gte_user : gte_users) {
          VLOG(2) << "Adding a new edge from " << stage->ToString() << " to "
                  << gte_user->ToString() << ".";
          HloInstruction* gte_clone = comp->AddInstruction(gte->Clone());
          gte->SetupDerivedInstruction(gte_clone);
          TF_RETURN_IF_ERROR(gte->ReplaceUseWith(gte_user, gte_clone));
        }
        added_edges = true;
        TF_RETURN_IF_ERROR(comp->ForceRemoveInstruction(gte));
      }
    }
  }
  return added_edges;
}

StatusOr<bool> UniquifyPipelineStageCallsites(PipelineStages& pipeline_stages) {
  absl::flat_hash_set<HloComputation*> already_used;
  bool added_computations = false;
  for (auto& stages : {pipeline_stages.forward, pipeline_stages.backward}) {
    for (HloInstruction* stage : stages) {
      HloComputation* comp = stage->to_apply();
      if (already_used.contains(comp)) {
        VLOG(2) << "Duplicating the computation for stage " << stage->ToString()
                << ".";
        HloModule* module = comp->parent();
        comp = module->AddEmbeddedComputation(comp->Clone());
        stage->set_to_apply(comp);
        added_computations = true;
      }
      already_used.insert(comp);
    }
  }
  return added_computations;
}

namespace {
Status SetTupleUniqueDeviceSharding(const HloInstruction* source,
                                    HloInstruction* dest) {
  auto optional_sharding = source->sharding().ExtractSingleSharding();
  if (!optional_sharding) {
    return FailedPrecondition("Could not extract single sharding.");
  }
  dest->set_sharding(
      HloSharding::SingleTuple(dest->shape(), *optional_sharding));
  return Status::OK();
}
}  // namespace

// Replace pipeline call with a new one with a new computation.
StatusOr<HloInstruction*> ReplaceCallWith(
    HloInstruction* call, std::unique_ptr<HloComputation> new_computation,
    const std::vector<HloInstruction*> new_operands,
    bool remove_unused_operands) {
  HloComputation* parent_computation = call->parent();
  HloComputation* call_computation = call->to_apply();
  HloModule* module = call->GetModule();

  HloComputation* new_call_computation =
      module->AddEmbeddedComputation(std::move(new_computation));

  HloInstruction* new_call =
      parent_computation->AddInstruction(HloInstruction::CreateCall(
          new_call_computation->root_instruction()->shape(), new_operands,
          new_call_computation));
  call->SetupDerivedInstruction(new_call);
  new_call->set_raw_backend_config_string(call->raw_backend_config_string());
  if (call->has_sharding()) {
    TF_RETURN_IF_ERROR(SetTupleUniqueDeviceSharding(call, new_call));
  }

  VLOG(3) << "Replacing " << call->ToString() << " and computation:";
  XLA_VLOG_LINES(3, call_computation->ToString());
  VLOG(3) << "With " << new_call->ToString() << " and computation:";
  XLA_VLOG_LINES(3, new_call_computation->ToString());

  TF_RETURN_IF_ERROR(call->ReplaceAllUsesWithDifferentShape(new_call));
  if (remove_unused_operands) {
    TF_RETURN_IF_ERROR(
        parent_computation->RemoveInstructionAndUnusedOperands(call));
  } else {
    TF_RETURN_IF_ERROR(parent_computation->RemoveInstruction(call));
  }
  TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(call_computation));

  return new_call;
}

StatusOr<HloInstruction*> AddInstructionsToPipelineStage(
    HloInstruction* stage, const std::vector<HloInstruction*>& ordered_lowering,
    std::map<int64, HloInstruction*> replace_parameter_with_lowered_instruction,
    HloInstructionSet forced_parameters, bool replace_resource_update_uses) {
  CHECK(IsPipelineStageOrBackwardOp(stage));

  HloComputation* pipeline_computation = stage->parent();

  VLOG(3) << "Lowering the following into the computation "
          << pipeline_computation->ToString();
  for (HloInstruction* inst : ordered_lowering) {
    VLOG(3) << "\t* " << inst->ToShortString();
  }

  // Create a set of instructions to lower for faster lookup.
  const HloInstructionSet ordered_lowering_set(ordered_lowering.begin(),
                                               ordered_lowering.end());

  // Mapping of the lowered instructions.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> lowered_insts;
  // Mapping of operands to parameters inside the new computation.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> operand_to_parameter;
  // Mapping from parameter number to instruction.
  absl::flat_hash_map<int64, HloInstruction*> parameter_instructions;

  // Note that current Hlo API does not allow us to modify the instruction or
  // computation so the algorithm builds new ones.
  HloComputation* stage_computation = stage->to_apply();
  auto builder = HloComputation::Builder(stage_computation->name());
  // A mapping from instructions in the old computation to the new one which is
  // currently being built.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> old_to_new_computation;
  // Duplicate the computation.
  for (HloInstruction* old_inst :
       stage_computation->MakeInstructionPostOrder()) {
    // Get the operands for the instruction we are about to lower.
    std::vector<HloInstruction*> new_operands(old_inst->operand_count());
    absl::c_transform(old_inst->operands(), new_operands.begin(),
                      [&old_to_new_computation](HloInstruction* old_operand) {
                        return old_to_new_computation.at(old_operand);
                      });
    // Clone new instruction.
    HloInstruction* new_inst = builder.AddInstruction(
        old_inst->CloneWithNewOperands(old_inst->shape(), new_operands));
    if (new_inst->opcode() == HloOpcode::kParameter) {
      // Make sure to mark inputs to the computation.
      HloInstruction* input_inst =
          stage->mutable_operand(new_inst->parameter_number());
      operand_to_parameter[input_inst] = new_inst;
      parameter_instructions[new_inst->parameter_number()] = new_inst;
    }

    old_inst->SetupDerivedInstruction(new_inst);
    old_to_new_computation[old_inst] = new_inst;
  }

  // The new pipeline stage arguments.
  std::vector<HloInstruction*> new_stage_operands = {stage->operands().begin(),
                                                     stage->operands().end()};

  auto create_parameter_for = [&](HloInstruction* operand) {
    HloInstruction* parameter =
        builder.AddInstruction(HloInstruction::CreateParameter(
            new_stage_operands.size(), operand->shape(), operand->name()));
    new_stage_operands.push_back(operand);
    return parameter;
  };

  // Clone the instructions into the new computation.
  for (HloInstruction* inst : ordered_lowering) {
    // Get all the new operands - check if we need to lower any into the
    // pipeline stage.
    std::vector<HloInstruction*> new_operands(inst->operand_count());
    for (int64 operand_idx = 0; operand_idx != inst->operand_count();
         ++operand_idx) {
      HloInstruction* operand = inst->mutable_operand(operand_idx);
      // Check if the operand for the instruction being lowered is also being
      // lowered.
      if (!ContainsKey(ordered_lowering_set, operand)) {
        // The operand is not being lowered - we need to make sure it can be
        // accessed inside the computation.
        auto itr = operand_to_parameter.find(operand);
        if (itr == operand_to_parameter.end()) {
          // We have not tried to lower it yet.
          HloInstruction* lowered_operand;
          // If this is a GTE on current pipeline stage then just use the input
          // to the tuple as the operand which is already in the new
          // computation.
          if (operand->opcode() == HloOpcode::kGetTupleElement &&
              operand->operand(0) == stage) {
            CHECK_EQ(operand->user_count(), 1);
            // In TF2XLA we expect the root to be a tuple, hence we can
            // get the relevant instruction (guaranteed by
            // VerifyPipelineStagesBeforeFixing).
            HloInstruction* root = stage_computation->root_instruction();
            CHECK_EQ(root->opcode(), HloOpcode::kTuple);
            root = old_to_new_computation.at(root);
            lowered_operand = root->mutable_operand(operand->tuple_index());
          } else {
            // Check if the operand is already an operand of the pipeline
            // stage.
            std::vector<int64> indices = stage->OperandIndices(operand);
            if (indices.size()) {
              // If it is, then get the parameter instruction from the new
              // computation and use it as the operand.
              lowered_operand =
                  stage_computation->parameter_instruction(indices[0]);
              lowered_operand = old_to_new_computation.at(lowered_operand);
            } else {
              // Otherwise create a new parameter and add a new operand to the
              // computation.
              lowered_operand = create_parameter_for(operand);
            }
          }
          // Add the new operand to the mapping so that we don't lower the same
          // operand twice.
          itr = operand_to_parameter.emplace(operand, lowered_operand).first;
        }
        new_operands[operand_idx] = itr->second;
      } else {
        // Because we are iterating in post order we must have lowered the
        // operand already.
        new_operands[operand_idx] = lowered_insts.at(operand);
      }
    }
    // Clone the instruction inside the new computation with new operands.
    HloInstruction* lowered = builder.AddInstruction(
        inst->CloneWithNewOperands(inst->shape(), new_operands));
    inst->SetupDerivedInstruction(lowered);
    lowered_insts[inst] = lowered;
  }

  // Add any forced operands.
  for (HloInstruction* forced_operand : forced_parameters) {
    HloInstruction* parameter = create_parameter_for(forced_operand);
    operand_to_parameter[forced_operand] = parameter;
    lowered_insts[forced_operand] = parameter;
  }

  // Sanity check we lowered everything.
  if (lowered_insts.size() !=
      (ordered_lowering.size() + forced_parameters.size())) {
    return InternalError("Failed to fully lower a cluster into a computation.");
  }

  auto replace_external_use =
      [&replace_resource_update_uses](const HloInstruction* inst) -> bool {
    return replace_resource_update_uses ? true : !IsResourceUpdate(inst);
  };

  // Keep track of instructions which are being lowered and have users outside
  // of this lowering. Using a map so that we iterate in the same order.
  HloInstructionMap<HloInstructionSet> external_uses;
  for (HloInstruction* inst : ordered_lowering) {
    // Go through all the users.
    for (HloInstruction* user : inst->users()) {
      if (user != stage && !ContainsKey(ordered_lowering_set, user) &&
          replace_external_use(user)) {
        // Find any outside uses of the inst - we need to create GTEs for
        // those.
        VLOG(3) << "Lowered instruction " << inst->ToString()
                << " has an external user: " << user->ToString();
        external_uses[inst].insert(user);
      }
    }
  }
  // Also track uses of forced parameters.
  for (HloInstruction* forced_operand : forced_parameters) {
    // Go through all the users.
    for (HloInstruction* user : forced_operand->users()) {
      if (user != stage && replace_external_use(user)) {
        // Find any outside uses of the inst - we need to create GTEs for
        // those.
        VLOG(3) << "Forced parameter instruction " << forced_operand->ToString()
                << " has an external user: " << user->ToString();
        external_uses[forced_operand].insert(user);
      }
    }
  }

  // Replace uses of parameters with lowered instruction. Note that this does
  // not remove parameters.
  for (auto& pair : replace_parameter_with_lowered_instruction) {
    int64 param_numer = pair.first;
    HloInstruction* parameter = parameter_instructions.at(param_numer);
    HloInstruction* inst_to_lower = pair.second;
    HloInstruction* lowered = lowered_insts.at(inst_to_lower);
    VLOG(3) << "Replacing the use of parameter " << parameter->ToString()
            << " with lowered " << lowered->ToString();
    TF_RETURN_IF_ERROR(parameter->ReplaceAllUsesWith(lowered));
  }

  // Finalize the computation extend the root by outputing any
  // instructions from the cluster which are used outside of the cluster.
  HloInstruction* builder_root;
  {
    HloInstruction* root =
        old_to_new_computation.at(stage_computation->root_instruction());
    CHECK_EQ(root->opcode(), HloOpcode::kTuple);
    // We are creating a new root tuple - first get all the existing
    // operands.
    std::vector<HloInstruction*> tuple_elements(root->operand_count() +
                                                external_uses.size());
    absl::c_copy(root->operands(), tuple_elements.begin());
    // Add extra outputs.
    auto tuple_elements_itr =
        std::next(tuple_elements.begin(), root->operand_count());
    for (auto users_pair : external_uses) {
      *tuple_elements_itr = lowered_insts.at(users_pair.first);
      tuple_elements_itr = std::next(tuple_elements_itr);
    }
    // Add the new root instruction.
    builder_root =
        builder.AddInstruction(HloInstruction::CreateTuple(tuple_elements));
    root->SetupDerivedInstruction(builder_root);
  }

  const int64 old_num_outputs = ShapeUtil::TupleElementCount(stage->shape());
  // Build the new computation and the new pipeline stage with new operands.
  std::unique_ptr<HloComputation> new_computation = builder.Build(builder_root);
  TF_ASSIGN_OR_RETURN(HloInstruction * new_stage,
                      ReplaceCallWith(stage, std::move(new_computation),
                                      new_stage_operands, false));

  // Add GTEs for any new outputs.
  {
    auto itr = external_uses.begin();
    for (size_t idx = 0; idx != external_uses.size(); ++idx, ++itr) {
      int64 tuple_index = old_num_outputs + idx;
      HloInstruction* inst = itr->first;
      for (HloInstruction* user : itr->second) {
        // Create a separate GTE for each user to preserve the duplicate GTE
        // condition.
        std::vector<int64> op_indices = user->OperandIndices(inst);
        CHECK(op_indices.size());
        for (int64 op_idx : op_indices) {
          TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                              MakeGetTupleElementHlo(new_stage, tuple_index));
          TF_RETURN_IF_ERROR(user->ReplaceOperandWith(op_idx, gte));
        }
      }
    }
  }
  // Remove any dead instructions.
  for (auto itr = ordered_lowering.rbegin(); itr != ordered_lowering.rend();
       ++itr) {
    if ((*itr)->user_count() == 0) {
      TF_RETURN_IF_ERROR(pipeline_computation->ForceRemoveInstruction(*itr));
    }
  }

  return new_stage;
}

StatusOr<std::set<int64>> GetUnusedCallOutputIndices(
    const HloInstruction* call) {
  std::set<int64> unused_outputs;
  if (call->parent()->root_instruction() != call) {
    for (int64 i = 0; i != ShapeUtil::TupleElementCount(call->shape()); ++i) {
      unused_outputs.insert(i);
    }
    for (HloInstruction* user : call->users()) {
      CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
      unused_outputs.erase(user->tuple_index());
    }
  }
  return unused_outputs;
}

StatusOr<std::set<int64>> GetUnusedParametersInCall(
    const HloInstruction* stage) {
  const HloComputation* stage_computation = stage->to_apply();
  std::set<int64> unused_params;
  for (int64 param_number = 0;
       param_number != stage_computation->num_parameters(); ++param_number) {
    const HloInstruction* parameter =
        stage_computation->parameter_instruction(param_number);
    if (parameter->user_count() == 0) {
      unused_params.insert(param_number);
    }
  }
  return unused_params;
}

namespace {
StatusOr<std::map<int64, std::set<int64>>> GetDuplicateOperands(
    const HloInstruction* inst) {
  absl::flat_hash_map<const HloInstruction*, int64> first_occurrence;
  std::map<int64, std::set<int64>> duplicate_operands;
  // Go through all the operands in order. First time we see it, add to
  // first_occurrence when we first saw it, next time we see it add it to the
  // duplicate operands.
  for (int64 op_idx = 0; op_idx != inst->operand_count(); ++op_idx) {
    const HloInstruction* operand = inst->operand(op_idx);
    auto itr = first_occurrence.find(operand);
    if (itr == first_occurrence.end()) {
      first_occurrence[operand] = op_idx;
    } else {
      duplicate_operands[itr->second].insert(op_idx);
    }
  }
  return duplicate_operands;
}
}  // namespace

StatusOr<std::map<int64, std::set<int64>>> GetDuplicateCallOutputs(
    const HloInstruction* call) {
  return GetDuplicateOperands(call->to_apply()->root_instruction());
}

StatusOr<std::map<int64, std::set<int64>>> GetDuplicateCallInputs(
    const HloInstruction* call) {
  return GetDuplicateOperands(call);
}

StatusOr<HloInstruction*> RemoveParametersFromCall(
    HloInstruction* call, const std::set<int64>& parameters_to_remove) {
  // Nothing to remove.
  if (parameters_to_remove.empty()) {
    return call;
  }

  HloComputation* call_computation = call->to_apply();

  VLOG(3) << "Removing the following parameters from " << call->ToString();
  for (int64 param_number : parameters_to_remove) {
    VLOG(3)
        << "\t* " << param_number << " "
        << call_computation->parameter_instruction(param_number)->ToString();
  }
  // A mapping from instructions in the old computation to the new one which is
  // currently being built.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> old_to_new_computation;
  auto builder = HloComputation::Builder(call_computation->name());

  // Lower/remove the parameters first.
  const int64 old_num_parameters = call_computation->num_parameters();
  std::vector<HloInstruction*> new_call_operands(old_num_parameters -
                                                 parameters_to_remove.size());
  int64 next_parameter_number = 0;
  auto next_to_remove_itr = parameters_to_remove.begin();
  for (int64 param_number = 0; param_number != old_num_parameters;
       ++param_number) {
    HloInstruction* old_parameter =
        call_computation->parameter_instruction(param_number);
    // Skip the parameter if we are removing it.
    if (next_to_remove_itr != parameters_to_remove.end() &&
        *next_to_remove_itr == param_number) {
      CHECK_EQ(old_parameter->user_count(), 0);
      next_to_remove_itr++;
    } else {
      // Otherwise lower it with the right index.
      HloInstruction* new_parameter =
          builder.AddInstruction(HloInstruction::CreateParameter(
              next_parameter_number, old_parameter->shape(),
              old_parameter->name()));
      new_call_operands[next_parameter_number++] =
          call->mutable_operand(param_number);
      old_parameter->SetupDerivedInstruction(new_parameter);
      old_to_new_computation[old_parameter] = new_parameter;
    }
  }
  CHECK_EQ(next_parameter_number, new_call_operands.size());

  // Lower all the other instructions.
  for (HloInstruction* old_inst :
       call_computation->MakeInstructionPostOrder()) {
    if (old_inst->opcode() == HloOpcode::kParameter) {
      continue;
    }

    // Get the operands for the instruction we are about to lower.
    std::vector<HloInstruction*> new_operands(old_inst->operand_count());
    absl::c_transform(old_inst->operands(), new_operands.begin(),
                      [&old_to_new_computation](HloInstruction* old_operand) {
                        return old_to_new_computation.at(old_operand);
                      });
    // Clone new instruction.
    HloInstruction* new_inst = builder.AddInstruction(
        old_inst->CloneWithNewOperands(old_inst->shape(), new_operands));
    old_inst->SetupDerivedInstruction(new_inst);
    old_to_new_computation[old_inst] = new_inst;
  }
  // Build the new computation and the new call with new operands.
  HloInstruction* new_root =
      old_to_new_computation.at(call_computation->root_instruction());
  std::unique_ptr<HloComputation> new_computation = builder.Build(new_root);
  TF_ASSIGN_OR_RETURN(HloInstruction * new_call,
                      ReplaceCallWith(call, std::move(new_computation),
                                      new_call_operands, true));
  return new_call;
}

StatusOr<HloInstruction*> InlineComputation(HloInstruction* caller,
                                            HloComputation* comp_to_inline,
                                            bool copy_sharding) {
  HloComputation* comp = caller->parent();
  // Hoist the computation out.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> hoisting_map;
  for (HloInstruction* inst : comp_to_inline->MakeInstructionPostOrder()) {
    HloInstruction* hoisted;
    if (inst->opcode() == HloOpcode::kParameter) {
      hoisted = caller->mutable_operand(inst->parameter_number());
    } else {
      std::vector<HloInstruction*> new_operands(inst->operand_count());
      absl::c_transform(inst->operands(), new_operands.begin(),
                        [&hoisting_map](HloInstruction* operand) {
                          return hoisting_map.at(operand);
                        });
      // Clone new instruction inside the computation.
      hoisted = comp->AddInstruction(
          inst->CloneWithNewOperands(inst->shape(), new_operands));

      if (copy_sharding) {
        CopyShardingIfPresent(caller, hoisted);
      }
    }
    hoisting_map[inst] = hoisted;
  }
  HloInstruction* new_root =
      hoisting_map.at(comp_to_inline->root_instruction());
  // Replace all uses.
  TF_RETURN_IF_ERROR(comp->ReplaceInstruction(caller, new_root));
  return new_root;
}

StatusOr<PoplarBackendConfig::CallConfig::PipelineConfig::Schedule>
GetPipelineSchedule(const HloInstruction* pipeline_op) {
  TF_ASSIGN_OR_RETURN(auto backend_config,
                      pipeline_op->backend_config<PoplarBackendConfig>());

  return backend_config.call_config().pipeline_config().schedule();
}

StatusOr<int> GetFifoDepthMultiplier(const HloInstruction* pipeline_op) {
  TF_ASSIGN_OR_RETURN(const auto schedule, GetPipelineSchedule(pipeline_op));

  switch (schedule) {
    case PoplarBackendConfig::CallConfig::PipelineConfig::Grouped:
      return 2;
    case PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved:
      return 1;
    case PoplarBackendConfig::CallConfig::PipelineConfig::Sequential:
      return 0;
    default:
      return FailedPrecondition("Unknown pipeline schedule.");
  }
}

Status RemoveOutputsFromCall(HloInstruction* call,
                             const std::set<int64>& outputs_to_remove) {
  // Nothing to remove.
  if (outputs_to_remove.empty()) {
    return Status::OK();
  }

  const int64 num_outputs_old = ShapeUtil::TupleElementCount(call->shape());
  HloComputation* call_computation = call->to_apply();
  HloInstruction* root = call_computation->root_instruction();

  VLOG(3) << "Removing outputs " << absl::StrJoin(outputs_to_remove, ", ")
          << " from " << call->ToString();

  // Get all the GTEs.
  std::map<int64, absl::flat_hash_set<HloInstruction*>> tuple_index_to_gte;
  for (HloInstruction* user : call->users()) {
    CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
    tuple_index_to_gte[user->tuple_index()].insert(user);
  }

  // Get the new outputs, preserving the relative order.
  std::vector<HloInstruction*> new_outputs(num_outputs_old -
                                           outputs_to_remove.size());
  auto next_to_remove_itr = outputs_to_remove.begin();
  for (int64 output_idx = 0, new_output_idx = 0; output_idx != num_outputs_old;
       ++output_idx) {
    if (next_to_remove_itr != outputs_to_remove.end() &&
        *next_to_remove_itr == output_idx) {
      next_to_remove_itr++;
      CHECK(tuple_index_to_gte[output_idx].empty());
    } else {
      // Change the gte tuple index.
      for (HloInstruction* gte : tuple_index_to_gte[output_idx]) {
        gte->set_tuple_index(new_output_idx);
      }
      new_outputs[new_output_idx++] = root->mutable_operand(output_idx);
    }
  }

  // Create a new root and change the shapes.
  HloInstruction* new_root = call_computation->AddInstruction(
      HloInstruction::CreateTuple(new_outputs));
  std::vector<Shape>* mutable_call_tuple_shapes =
      call->mutable_shape()->mutable_tuple_shapes();
  *mutable_call_tuple_shapes = new_root->shape().tuple_shapes();
  if (call->has_sharding()) {
    TF_RETURN_IF_ERROR(SetTupleUniqueDeviceSharding(call, call));
    TF_RETURN_IF_ERROR(SetTupleUniqueDeviceSharding(call, new_root));
  }
  call_computation->set_root_instruction(new_root, true);

  if (root->user_count() == 0) {
    TF_RETURN_IF_ERROR(
        call_computation->RemoveInstructionAndUnusedOperands(root));
  }

  return Status::OK();
}

PipelineDataflowAnalysis::PipelineDataflowAnalysis(
    const PipelineStages& pipeline_stages, bool allow_duplicate_gte_edges,
    bool allow_communication_ops, bool allow_feeds, bool allow_recomputation,
    bool allow_fifo_optimizations)
    : pipeline_stages_(pipeline_stages),
      allow_duplicate_gte_edges_(allow_duplicate_gte_edges),
      allow_communication_ops_(allow_communication_ops),
      allow_feeds_(allow_feeds),
      allow_recomputation_(allow_recomputation),
      allow_fifo_optimizations_(allow_fifo_optimizations) {
  // Put stages into lookup tables so that we can quickly get the stage id from
  // an instruction.
  for (size_t id = 0; id != pipeline_stages_.forward.size(); ++id) {
    fwd_stages_lookup_[pipeline_stages_.forward[id]] = id;
  }

  for (size_t id = 0; id != pipeline_stages_.backward.size(); ++id) {
    bwd_stages_lookup_[pipeline_stages_.backward[id]] = id;
  }

  for (auto pair : pipeline_stages.recomputation) {
    recomputation_stages_lookup_[pair.second] = pair.first;
  }
};

HloValueSet* PipelineDataflowAnalysis::GetMutableValueSet(
    HloInstruction* inst) {
  return inst_to_value_set_.at(inst).mutable_element(ShapeIndex());
}

const HloValueSet& PipelineDataflowAnalysis::GetValueSet(
    const HloInstruction* inst) const {
  return inst_to_value_set_.at(inst).element(ShapeIndex());
}

HloValueSet* PipelineDataflowAnalysis::CreateValueSet(HloInstruction* inst) {
  return inst_to_value_set_.emplace(inst, InstructionValueSet(inst->shape()))
      .first->second.mutable_element(ShapeIndex());
}

HloValueSet PipelineDataflowAnalysis::GetOperandsValueSet(
    const HloInstruction* inst) const {
  std::vector<const HloValueSet*> operand_sets(inst->operand_count());
  absl::c_transform(
      inst->operands(), operand_sets.begin(),
      [&](HloInstruction* operand) { return &GetValueSet(operand); });
  HloValueSet operands_set;
  operands_set.AssignUnionOf(operand_sets);
  return operands_set;
}

HloValue* PipelineDataflowAnalysis::CreateValue(HloInstruction* inst) {
  auto id = next_value_id_++;
  // Create a value and add it to the internal storage.
  return &values_.emplace(id, HloValue(id, inst, ShapeIndex())).first->second;
}

StatusOr<StageID> PipelineDataflowAnalysis::GetStageID(
    const HloInstruction* inst) const {
  if (IsPipelineStage(inst)) {
    return StageID(StageType::kForward, fwd_stages_lookup_.at(inst));
  } else if (IsPipelineStageRecomputation(inst)) {
    return StageID(StageType::kRecomputation,
                   recomputation_stages_lookup_.at(inst));
  } else if (IsPipelineStageBackward(inst)) {
    return StageID(StageType::kBackward, bwd_stages_lookup_.at(inst));
  } else {
    return InternalErrorStrCat("Trying to get StageID for ", inst->ToString(),
                               " which is not a PipelineStage.");
  }
}

StatusOr<StageID> PipelineDataflowAnalysis::GetPreviousStageID(
    const HloInstruction* inst) const {
  TF_ASSIGN_OR_RETURN(StageID stage_id, GetStageID(inst));

  switch (stage_id.stage_type) {
    case StageType::kForward:
    case StageType::kRecomputation: {
      if (stage_id.id == 0) {
        return InternalError(
            "Trying to call GetPreviousStageID on PipelineStage ID 0 which is "
            "the first pipeline stage.");
      }
      // The previous stage for a recomputation stage is a forward stage too.
      return StageID(StageType::kForward, stage_id.id - 1);
    }
    case StageType::kBackward: {
      if (stage_id.id ==
          static_cast<int64>(pipeline_stages_.forward.size()) - 1) {
        return StageID(StageType::kForward, stage_id.id);
      } else {
        return StageID(StageType::kBackward, stage_id.id + 1);
      }
    }
    default: { return FailedPrecondition("Invalid enum type."); }
  }
}

StatusOr<int64> PipelineDataflowAnalysis::GetShardForStage(
    const StageID& stage_id) const {
  switch (stage_id.stage_type) {
    case StageType::kForward:
    case StageType::kBackward:
    case StageType::kRecomputation: {
      // Get the shard for the pipeline stage from the forward stage as we can
      // guarantee that it has sharding information attached.
      return *pipeline_stages_.forward[stage_id.id]->sharding_unique_device();
    }
    default: {
      return FailedPrecondition("Invalid enum type for GetShardForStage.");
    }
  }
}

Status PipelineDataflowAnalysis::VerifyPipelineUsage(
    const HloInstruction* pipeline_stage,
    const HloInstruction* pipeline_stage_user) const {
  if (pipeline_stage == pipeline_stage_user) {
    return Status::OK();
  }

  TF_ASSIGN_OR_RETURN(StageID pipeline_stage_id, GetStageID(pipeline_stage));
  TF_ASSIGN_OR_RETURN(StageID pipeline_stage_user_id,
                      GetStageID(pipeline_stage_user));

  switch (pipeline_stage_id.stage_type) {
    case StageType::kForward: {
      // A forward stage can be used by the next forward stage.
      if (pipeline_stage_user_id.stage_type == StageType::kForward &&
          (pipeline_stage_id.id + 1) == pipeline_stage_user_id.id) {
        return Status::OK();
      }

      // A forward stage can be used by the corresponding backward stage.
      if (pipeline_stage_user_id.stage_type == StageType::kBackward &&
          pipeline_stage_id.id == pipeline_stage_user_id.id) {
        return Status::OK();
      }

      // If we allow recomputation, then it can also be used by the next
      // recomputation stage.
      if (allow_recomputation_ &&
          pipeline_stage_user_id.stage_type == StageType::kRecomputation &&
          (pipeline_stage_id.id + 1) == pipeline_stage_user_id.id) {
        return Status::OK();
      }

      // If we allow recomputation, then it can also be used by the
      // corresponding recomputation stage if it's a stateful stage.
      if (allow_recomputation_ &&
          pipeline_stage_user_id.stage_type == StageType::kRecomputation &&
          pipeline_stage_id.id == pipeline_stage_user_id.id) {
        return Status::OK();
      }

      // If we allow fifo optimizations, then it can also be used by any other
      // forward which occurs later.
      if (allow_fifo_optimizations_ &&
          pipeline_stage_user_id.stage_type == StageType::kForward &&
          pipeline_stage_id.id < pipeline_stage_user_id.id) {
        return Status::OK();
      }

      // If we allow fifo optimizations and recomputation, then it can also be
      // used by any other recomputation stage which occurs later.
      if (allow_fifo_optimizations_ && allow_recomputation_ &&
          pipeline_stage_user_id.stage_type == StageType::kRecomputation &&
          pipeline_stage_id.id < pipeline_stage_user_id.id) {
        return Status::OK();
      }
      break;
    }
    case StageType::kBackward: {
      // A backward stage can be used by the next backward stage.
      if (pipeline_stage_user_id.stage_type == StageType::kBackward &&
          pipeline_stage_id.id == (pipeline_stage_user_id.id + 1)) {
        return Status::OK();
      }

      // If we allow fifo optimizations, then it can also be used by any other
      // forward which occurs later.
      if (allow_fifo_optimizations_ &&
          pipeline_stage_user_id.stage_type == StageType::kBackward &&
          pipeline_stage_id.id > pipeline_stage_user_id.id) {
        return Status::OK();
      }
      break;
    }
    case StageType::kRecomputation: {
      // A recomputation stage can only be used by the corresponding backward
      // stage.
      if (pipeline_stage_user_id.stage_type == StageType::kBackward &&
          pipeline_stage_id.id == pipeline_stage_user_id.id) {
        return Status::OK();
      }
      break;
    }
    default: { return FailedPrecondition("Invalid enum type."); }
  }

  // Everything else is an error.
  return InternalErrorStrCat(
      "Trying to use an output of a ", pipeline_stage_id.ToString(),
      " as an input to a ", pipeline_stage_user_id.ToString(),
      " which violates the data flow constraints for Pipelines.");
}

Status PipelineDataflowAnalysis::VerifyParameterUsage(
    const HloInstruction* parameter,
    const HloInstruction* pipeline_stage_user) {
  CHECK_EQ(parameter->opcode(), HloOpcode::kParameter);
  TF_ASSIGN_OR_RETURN(StageID stage_id, GetStageID(pipeline_stage_user));
  // Get the shard for the pipeline stage.
  TF_ASSIGN_OR_RETURN(const int64 shard, GetShardForStage(stage_id));
  // Get the parameter value and check where it is used.
  const HloValue& parameter_value = GetValueSet(parameter).GetUniqueValue();
  for (const HloInstruction* user_stage :
       used_by_stages_.at(&parameter_value)) {
    TF_ASSIGN_OR_RETURN(StageID user_stage_id, GetStageID(user_stage));
    // Get the shard for the other user stage.
    TF_ASSIGN_OR_RETURN(const int64 user_shard,
                        GetShardForStage(user_stage_id));
    if (stage_id.id != user_stage_id.id && shard != user_shard) {
      return UnimplementedStrCat(
          "The ", stage_id.ToString(), " is trying to use an input (",
          parameter->name(), ") which is already used by the ",
          user_stage_id.ToString(),
          ". This violates the dataflow constraints because an input can only "
          "be used by pipeline stages on the same IPU.\n"
          "If the model requires for multiple pipeline stages to access the "
          "same `tf.Variable` then these pipeline stages need to placed onto "
          "the same IPU using the `device_mapping` argument to "
          "`tf.python.ipu.pipelining_ops.pipeline`.");
    }
  }
  return Status::OK();
};

Status PipelineDataflowAnalysis::VerifyGradientAccumulatorCreateUsage(
    const HloInstruction* gradient_accumulator_create,
    const HloInstruction* pipeline_stage_user) {
  CHECK(IsGradientAccumulatorCreate(gradient_accumulator_create));
  TF_ASSIGN_OR_RETURN(StageID stage_id, GetStageID(pipeline_stage_user));
  // Get the shard for the pipeline stage.
  TF_ASSIGN_OR_RETURN(const int64 shard, GetShardForStage(stage_id));
  // Get the gradient_accumulator_create value and check where it is used.
  const HloValue& gradient_accumulator_create_value =
      GetValueSet(gradient_accumulator_create).GetUniqueValue();

  // The gradient buffer is not used by any stage and it will be removed by DCE.
  if (!used_by_stages_.contains(&gradient_accumulator_create_value)) {
    return Status::OK();
  }

  for (const HloInstruction* user_stage :
       used_by_stages_.at(&gradient_accumulator_create_value)) {
    TF_ASSIGN_OR_RETURN(StageID user_stage_id, GetStageID(user_stage));
    // Get the shard for the other user stage.
    TF_ASSIGN_OR_RETURN(const int64 user_shard,
                        GetShardForStage(user_stage_id));
    if (stage_id.id != user_stage_id.id && shard != user_shard) {
      return UnimplementedStrCat(
          "The ", stage_id.ToString(),
          " is trying to use a gradient accumulation buffer (",
          gradient_accumulator_create->name(),
          ") which is already used by the ", user_stage_id.ToString(),
          ". This violates the dataflow constraints because gradient for the "
          "same parameter can only be accumulated by pipeline stages on the "
          "same IPU.\n"
          "If the model requires for multiple pipeline stages to access the "
          "same `tf.Variable` then these pipeline stages need to placed onto "
          "the same IPU using the `device_mapping` argument to "
          "`tf.python.ipu.pipelining_ops.pipeline`.");
    }
  }
  return Status::OK();
}

Status PipelineDataflowAnalysis::VerifyInfeedUsage(
    const HloInstruction* infeed, const HloInstruction* pipeline_stage_user) {
  CHECK_EQ(infeed->opcode(), HloOpcode::kInfeed);
  TF_ASSIGN_OR_RETURN(StageID stage_id, GetStageID(pipeline_stage_user));
  // Get the infeed value and check where it is used.
  const HloValue& infeed_value = GetValueSet(infeed).GetUniqueValue();
  for (const HloInstruction* user_stage : used_by_stages_.at(&infeed_value)) {
    TF_ASSIGN_OR_RETURN(StageID user_stage_id, GetStageID(user_stage));
    if (stage_id.id != user_stage_id.id) {
      return UnimplementedStrCat(
          "The ", stage_id.ToString(),
          " is trying to use an infeed which is already used by the ",
          user_stage_id.ToString(),
          ". This violates the dataflow  constraints because an infeed can "
          "only be used by a single PipelineStage.");
    }
  }
  return Status::OK();
};

Status PipelineDataflowAnalysis::VerifyPipelineStageOperands(
    const HloInstruction* pipeline_stage, const HloValueSet& new_inputs) {
  CHECK(IsAnyPipelineStageOp(pipeline_stage));
  TF_RETURN_IF_ERROR(GetStageID(pipeline_stage).status());
  // Get all the values used by the operands of inst.
  HloValueSet operands_set = GetOperandsValueSet(pipeline_stage);
  operands_set.AssignUnionOf({&operands_set, &new_inputs});

  for (const HloValue* value : operands_set.values()) {
    HloInstruction* producer = value->instruction();
    switch (producer->opcode()) {
      case HloOpcode::kParameter: {
        TF_RETURN_IF_ERROR(VerifyParameterUsage(producer, pipeline_stage));
        break;
      }
      case HloOpcode::kCustomCall: {
        CHECK(IsGradientAccumulatorCreate(producer));
        TF_RETURN_IF_ERROR(
            VerifyGradientAccumulatorCreateUsage(producer, pipeline_stage));
        break;
      }
      case HloOpcode::kCall: {
        if (IsAnyPipelineStageOp(producer)) {
          TF_RETURN_IF_ERROR(VerifyPipelineUsage(producer, pipeline_stage));
          break;
        }
      }
      case HloOpcode::kInfeed: {
        TF_RETURN_IF_ERROR(VerifyInfeedUsage(producer, pipeline_stage));
        break;
      }
      default: { return InternalError("Invalid producer in the pipelines."); }
    }
  }
  return Status::OK();
}

StatusOr<bool> PipelineDataflowAnalysis::HasToBeLoweredIntoStage(
    const HloInstruction* stage, const HloInstruction* inst) const {
  TF_ASSIGN_OR_RETURN(bool lower, HasToBeLowered(inst));
  if (lower && IsPipelineStageBackward(stage)) {
    TF_ASSIGN_OR_RETURN(StageID stage_id, GetStageID(stage));
    // inst has to be lowered, however it might have to be lowered into a
    // different stage - example is when a resource has multiple gradients
    // coming from different stages - we need to lower it into the last stage it
    // is used in.

    // We only do this for the bwd stages because the fwd stages were threaded
    // correctly anyway.
    HloValueSet value_set = GetOperandsValueSet(inst);
    for (const HloValue* value : value_set.values()) {
      HloInstruction* producer = value->instruction();
      if (IsPipelineStageBackward(producer)) {
        TF_ASSIGN_OR_RETURN(StageID producer_stage_id, GetStageID(producer));
        if (producer_stage_id.id < stage_id.id) {
          // Let a later stage deal with this.
          return false;
        }
      }
    }
    return true;
  } else {
    return lower;
  }
}

StatusOr<bool> PipelineDataflowAnalysis::HasToBeLowered(
    const HloInstruction* inst) const {
  const HloInstruction* root_instruction = inst->parent()->root_instruction();
  // A root never needs lowering.
  if (root_instruction == inst) {
    return false;
  }

  switch (inst->opcode()) {
    case HloOpcode::kCall:
      return !IsAnyPipelineStageOpOrResourceUpdate(inst);
    case HloOpcode::kCopy:
      return !allow_communication_ops_;
    case HloOpcode::kParameter:
      return false;
    case HloOpcode::kGetTupleElement: {
      // A GTE on the output from a PipelineStage needs to be lowered unless:
      // (1) It is used by the next pipeline stage.
      // (2) It is used by the corresponding backward pipeline stage.
      // It is an error if a GTE on the output from a PipelineStage is
      // used by any other pipeline stage.
      //
      // We do not need to lower GTEs on infeeds if we are allowing feeds.
      //
      // Any other GTE needs to be lowered.
      const HloInstruction* gte_input = inst->operand(0);
      if (IsAnyPipelineStageOp(gte_input)) {
        // DuplicateGTEEdges should make sure each GTE only has one user.
        if (!allow_duplicate_gte_edges_ && inst->user_count() != 1) {
          return InternalErrorStrCat("Expected instruction ", inst->ToString(),
                                     " to have exactly one user.");
        }
        for (const HloInstruction* gte_user : inst->users()) {
          // Verify that the pipeline usage is legal. If the user is not a
          // PipelineStage then the user will be lowered later.
          if (IsAnyPipelineStageOp(gte_user)) {
            TF_RETURN_IF_ERROR(VerifyPipelineUsage(gte_input, gte_user));
          }
        }
        return false;
      } else if (allow_feeds_ && gte_input->opcode() == HloOpcode::kInfeed) {
        return false;
      } else if (IsResourceUpdate(gte_input)) {
        for (const HloInstruction* gte_user : inst->users()) {
          // Expect that all users of the resource update are the root
          // instruction.
          if (gte_user->parent()->root_instruction() != gte_user) {
            return InternalErrorStrCat(
                "Expected the ResourceUpdate outputs to be used by the "
                "root instruction only.");
          }
        }
        return false;
      } else {
        // Any other GTE has to be lowered.
        return true;
      }
    }
    case HloOpcode::kCustomCall: {
      if (IsGradientAccumulatorCreate(inst)) {
        return false;
      } else if (IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(inst)) {
        // The sink op combines the same gradient accumulation buffer being
        // the output from different pipeline stages residing on the same IPU.
        // We expect it to only have a single user - the pipeline resource
        // update.
        if (inst->user_count() != 1) {
          return FailedPrecondition(
              "Expected the gradient accumulator %s to have one user.",
              inst->ToString());
        }

        if (!IsResourceUpdate(inst->users()[0])) {
          return FailedPrecondition(
              "Expected the gradient accumulator to only be used by the "
              "pipeline resource update, but is used by %s instead.",
              inst->ToString());
        }
        return false;
      }

      if (!allow_communication_ops_) {
        return true;
      }
      if (IsPoplarInstruction(PoplarOp::IpuInterCopy)(inst)) {
        // For an inter IPU copy we expect that the input is a chain like:
        // Stage -> GTE -> InterIPUCopy -> NextStage
        const HloInstruction* gte = inst->operand(0);
        const HloInstruction* gte_input = gte->operand(0);
        if (gte->opcode() != HloOpcode::kGetTupleElement) {
          return FailedPrecondition(
              "Expected the input of an inter IPU copy to be a GTE "
              "instruction, but is %s instead.",
              gte->ToString());
        }
        TF_ASSIGN_OR_RETURN(StageID copy_input_stage_id, GetStageID(gte_input));
        if (allow_recomputation_) {
          switch (copy_input_stage_id.stage_type) {
            case StageType::kForward: {
              // A forward stage can be used by the next forward stage and a
              // recomputation stage.
              if (inst->user_count() > 2) {
                return FailedPrecondition(
                    "Expected the output of an inter IPU copy to have at most "
                    "two users.");
              }
              break;
            }
            // Backward stage can only have a copy to another backward stage.
            case StageType::kBackward: {
              if (inst->user_count() != 1) {
                return FailedPrecondition(
                    "Expected the output of an inter IPU copy to be used "
                    "exactly once.");
              }
              break;
            }
            default: {
              return FailedPrecondition("Invalid use of an inter IPU copy.");
            }
          }
        } else {
          // When not recomputing, we expect the copy to come from a stage and
          // for it to be used by the subsequent stage.
          if (inst->user_count() != 1) {
            return FailedPrecondition(
                "Expected the output of an inter IPU copy to be used exactly "
                "once.");
          }
        }
        // Verify the stage IDs match up. We expect the stages are continuous.
        for (const HloInstruction* user : inst->users()) {
          // Look through FIFOs when doing recomputation.
          const bool look_through =
              allow_recomputation_ && IsPoplarInstruction(PoplarOp::Fifo)(user);
          if (look_through && user->user_count() != 1) {
            return FailedPrecondition(
                "Expected the FIFO to have a single user.");
          }
          const HloInstruction* output = look_through ? user->users()[0] : user;
          TF_ASSIGN_OR_RETURN(StageID copy_output_stage_id, GetStageID(output));
          TF_ASSIGN_OR_RETURN(StageID copy_output_previous_stage_id,
                              GetPreviousStageID(output));
          if (copy_input_stage_id != copy_output_previous_stage_id) {
            return UnimplementedStrCat(
                "Trying to copy data between ", copy_input_stage_id.ToString(),
                " and ", copy_output_stage_id.ToString(),
                ". This violates the dataflow constraints because an output of "
                "one PipelineStage can only be used by the next "
                "PipelineStage.");
          }
        }
        return false;
      } else if (IsPoplarInstruction(PoplarOp::Fifo)(inst)) {
        // We always expect FIFO input to be a GTE.
        const HloInstruction* fifo_input = inst->operand(0);
        // We always expect the FIFO to only have at least a single user.
        // However if we allow fifo instructions between stages on the same IPU
        // which are not intermediate *and* recomputation, then the fifo can
        // have two users.
        const uint64 max_user_count =
            (allow_fifo_optimizations_ && allow_recomputation_) ? 2 : 1;

        if (inst->user_count() > max_user_count) {
          return FailedPrecondition(
              "Expected the FIFO operation to have at most %d users, but has "
              "%d users.",
              max_user_count, inst->user_count());
        }
        const bool fifo_input_is_fifo =
            IsPoplarInstruction(PoplarOp::Fifo)(fifo_input);
        for (const HloInstruction* user : inst->users()) {
          const bool input_to_recomputation_stage =
              allow_recomputation_ && IsPipelineStageRecomputation(user);
          if (IsPoplarInstruction(PoplarOp::Fifo)(user)) {
            // This is a fifo feeding into another fifo, which is allowed.
            continue;
          } else if (fifo_input_is_fifo) {
            if (!IsAnyPipelineStageOpOrResourceUpdate(user)) {
              return UnimplementedStrCat(
                  "Trying to create a FIFO which is consumed by ",
                  inst->ToString(), ", which is not supported.");
            }
          } else if (input_to_recomputation_stage) {
            // When we are recomputing a stage, for a FIFO we expect:
            // Infeed/ForwardStage -> GTE -> (InterIPUCopy)-> FIFO ->
            // RecomputationStage.
            const HloInstruction* gte_input =
                fifo_input->operand(0)->LatestNonGteAncestor();
            if (gte_input->opcode() != HloOpcode::kInfeed) {
              TF_ASSIGN_OR_RETURN(StageID fifo_input_stage_id,
                                  GetStageID(gte_input));
              TF_ASSIGN_OR_RETURN(StageID fifo_output_stage_id,
                                  GetStageID(user));
              // Expect for a FIFO to either be between:
              // - A forward stage and the next stage's recomputation stage
              // (Input).
              // - A forward stage and the corresponding recomputatoin stage
              // (State).
              if (fifo_input_stage_id.stage_type != StageType::kForward ||
                  fifo_output_stage_id.stage_type !=
                      StageType::kRecomputation ||
                  ((fifo_input_stage_id.id != fifo_output_stage_id.id) &&
                   ((fifo_input_stage_id.id + 1) != fifo_output_stage_id.id))) {
                return UnimplementedStrCat(
                    "Trying to create a FIFO between ",
                    fifo_input_stage_id.ToString(), " and ",
                    fifo_output_stage_id.ToString(),
                    ". This violates the dataflow constraints because a FIFO "
                    "operation can only be placed between a forward "
                    "PipelineStage and its corresponding or the next "
                    "Recomputation PipelineStage.");
              }
            }
          } else {
            if (fifo_input->opcode() != HloOpcode::kGetTupleElement) {
              return FailedPrecondition(
                  "Expected the input of a FIFO operation to be a GTE  "
                  "instruction, but is %s instead.",
                  fifo_input->ToString());
            }
            // When we are not recomputing, for a FIFO we expect that the input
            // is a chain: ForwardStage -> GTE -> FIFO -> BackwardStage.
            TF_ASSIGN_OR_RETURN(StageID fifo_input_stage_id,
                                GetStageID(fifo_input->operand(0)));
            TF_ASSIGN_OR_RETURN(StageID fifo_output_stage_id, GetStageID(user));
            // Expect the input to FIFO to be a forward stage and the output of
            // FIFO to be a backward stage. Expect their IDs to match.
            bool allowed_fifo =
                fifo_input_stage_id.stage_type == StageType::kForward &&
                fifo_output_stage_id.stage_type == StageType::kBackward &&
                fifo_input_stage_id.id == fifo_output_stage_id.id;

            if (allow_fifo_optimizations_) {
              // Allow FIFOs between stages of the same type.
              allowed_fifo |= fifo_input_stage_id.stage_type ==
                              fifo_output_stage_id.stage_type;
            }
            if (!allowed_fifo) {
              return UnimplementedStrCat(
                  "Trying to create a FIFO between ",
                  fifo_input_stage_id.ToString(), " and ",
                  fifo_output_stage_id.ToString(),
                  ". This violates the dataflow constraints because a FIFO "
                  "operation can only be placed between a forward "
                  "PipelineStage and a backward PipelineStage with the same "
                  "stage ID.");
            }
          }
        }
        return false;
      } else {
        return true;
      }
    }
    case HloOpcode::kAfterAll: {
      if (allow_feeds_) {
        // Need to lower an after all if it doesn't have a single infeed/outfeed
        // user or it has operands.
        auto user_is_feed = [](const HloInstruction* inst) {
          return inst->opcode() == HloOpcode::kInfeed ||
                 inst->opcode() == HloOpcode::kOutfeed;
        };
        return inst->operand_count() != 0 || inst->user_count() != 1 ||
               !user_is_feed(inst->users()[0]);
      } else {
        return true;
      }
    }
    case HloOpcode::kInfeed: {
      if (allow_feeds_) {
        // Make sure the token does not need to be lowered.
        TF_ASSIGN_OR_RETURN(bool token_needs_lowering,
                            HasToBeLowered(inst->operand(0)));
        if (token_needs_lowering) {
          return true;
        }
        // We need to lower the infeed if:
        // * it does not have a single user
        if (inst->user_count() != 1) {
          return true;
        }
        // * or that single user is not a GTE on tuple index = 0
        const HloInstruction* user = inst->users()[0];
        if (user->opcode() != HloOpcode::kGetTupleElement ||
            user->tuple_index() != 0) {
          return true;
        }

        // * or, when not recomputing, the GTE doesn't have a single user,
        if (!allow_recomputation_ && user->user_count() != 1) {
          return true;
        }

        // * or, when recomputing, the GTE has more than two users,
        if (allow_recomputation_ && user->user_count() > 2) {
          return true;
        }

        // * or, all the users of the GTE are not pipeline stages or FIFOs when
        // recomputing.
        return !absl::c_all_of(user->users(), [&](const HloInstruction* u) {
          return IsAnyPipelineStageOp(u) ||
                 (allow_recomputation_ &&
                  IsPoplarInstruction(PoplarOp::Fifo)(u));
        });
      } else {
        return true;
      }
    }
    case HloOpcode::kOutfeed: {
      if (allow_feeds_) {
        // We need to lower an outfeed if either of its operands needs lowering.
        TF_ASSIGN_OR_RETURN(bool input_needs_lowering,
                            HasToBeLowered(inst->operand(0)));
        TF_ASSIGN_OR_RETURN(bool token_needs_lowering,
                            HasToBeLowered(inst->operand(1)));
        return input_needs_lowering || token_needs_lowering;
      } else {
        return true;
      }
    }
    default:
      return true;
  }
}  // namespace poplarplugin

Status PipelineDataflowAnalysis::UpdateThroughInstruction(
    HloInstruction* inst) {
  HloValueSet operands_value_set = GetOperandsValueSet(inst);
  if (IsProducerOp(inst)) {
    // Producers create their sets.
    if (IsAnyPipelineStageOp(inst)) {
      // Mark values as used by the stage.
      for (const HloValue* value : operands_value_set.values()) {
        used_by_stages_[value].insert(inst);
      }
      // Make sure the operands of the stage do not violate the data flow
      // constraints.
      TF_RETURN_IF_ERROR(VerifyPipelineStageOperands(inst));
    } else if (IsGradientAccumulatorCreate(inst)) {
      // Make sure the input to the creator is just a parameter.
      CHECK_EQ(operands_value_set.values().size(), 1);
      CHECK_EQ(operands_value_set.values()[0]->instruction()->opcode(),
               HloOpcode::kParameter);
    } else {
      CHECK(operands_value_set.values().empty());
    }
    HloValueSet* value_set = CreateValueSet(inst);
    value_set->AddValue(CreateValue(inst));
  } else {
    // Forward all the values from operands.
    HloValueSet* value_set = CreateValueSet(inst);
    value_set->AssignUnionOf({&operands_value_set});
  }
  return Status::OK();
}

StatusOr<std::unique_ptr<PipelineDataflowAnalysis>>
PipelineDataflowAnalysis::GetAnalysis(const PipelineStages& pipeline_stages,
                                      bool allow_duplicate_gte_edges,
                                      bool allow_communication_ops,
                                      bool allow_feeds,
                                      bool allow_recomputation,
                                      bool allow_fifo_optimizations) {
  auto analysis = absl::make_unique<PipelineDataflowAnalysis>(
      pipeline_stages, allow_duplicate_gte_edges, allow_communication_ops,
      allow_feeds, allow_recomputation, allow_fifo_optimizations);
  if (!allow_recomputation && analysis->pipeline_stages_.recomputation.size()) {
    return FailedPrecondition(
        "Detected PipelineStageRecomputation which are not allowed");
  }
  if (analysis->pipeline_stages_.forward.size()) {
    HloComputation* pipeline_computation =
        analysis->pipeline_stages_.forward[0]->parent();
    for (HloInstruction* inst :
         pipeline_computation->MakeInstructionPostOrder()) {
      TF_RETURN_IF_ERROR(analysis->UpdateThroughInstruction(inst));
    }
  }

  return std::move(analysis);
}

PipelinePath::PipelinePath(HloInstruction* new_consumer, uint64 stage_idx,
                           uint64 input_idx, uint64 output_idx)
    : visited_stages_({stage_idx}),
      inputs_path_({input_idx}),
      outputs_path_({output_idx}),
      new_consumer_(new_consumer) {}

bool PipelinePath::FinishPath(PipelineStages& stages) {
  finished_ = true;
  // Only add a path if we visited more than two stages otherwise we
  // would be inserting FIFOs of size zero between consecutive stages.
  if (visited_stages_.size() < 3) {
    return false;
  }

  // We can only have a path between stages that are either:
  // (1) all in forward stages,
  // (2) all in backward stages,
  // (3) between a forward and a backward stage with the same id.
  const uint64 start_stage_idx = visited_stages_[0];
  const uint64 end_stage_idx = visited_stages_.back();

  const uint64 num_backward_stages = stages.backward.size();
  const uint64 num_forward_stages = stages.forward.size();
  // It is worth remembering that the path can only be a chain between
  // consecutive stages.
  const bool start_is_backward_stage = start_stage_idx < num_backward_stages;
  const bool end_is_backward_stage = end_stage_idx < num_backward_stages;

  // The old consumer stage is the input to the second to last stage which was
  // visited.
  uint64 old_consumer_id = visited_stages_.at(visited_stages_.size() - 2);
  if (old_consumer_id < num_backward_stages) {
    old_consumer_ = stages.backward.at(old_consumer_id);
  } else {
    old_consumer_id =
        (num_forward_stages + num_backward_stages - old_consumer_id - 1);
    old_consumer_ = stages.forward.at(old_consumer_id);
  }

  if (start_is_backward_stage == end_is_backward_stage) {
    // Both start and end are of same type - handle cases (1) and (2).
    fifo_depth_ = end_stage_idx - start_stage_idx - 1;
    type_ = end_is_backward_stage ? Type::kBackward : Type::kForward;
    return true;
  } else if (start_is_backward_stage && !end_is_backward_stage) {
    // Handle case (3).
    fifo_depth_ = end_stage_idx - num_backward_stages;
    type_ = Type::kForwardToBackward;
    return start_stage_idx ==
           (num_forward_stages + num_backward_stages - end_stage_idx - 1);
  } else {
    return false;
  }
}

std::vector<uint64>& PipelinePath::GetVisitedStages() {
  return visited_stages_;
}

std::vector<uint64>& PipelinePath::GetInputsPath() { return inputs_path_; }

std::vector<uint64>& PipelinePath::GetOutputsPath() { return outputs_path_; }

StatusOr<int64> PipelinePath::GetFifoDepth(const HloInstruction* pipeline_op) {
  if (finished_) {
    TF_ASSIGN_OR_RETURN(const auto schedule, GetPipelineSchedule(pipeline_op));
    // We need to take schedule into account.
    uint64 multiplier = 1;
    switch (schedule) {
      case PoplarBackendConfig::CallConfig::PipelineConfig::Grouped:
        multiplier = type_ == Type::kForwardToBackward ? 2 : 1;
        break;
      case PoplarBackendConfig::CallConfig::PipelineConfig::Sequential:
        multiplier = 0;
        break;
      default:
        return FailedPrecondition("Unsupported pipeline schedule.");
    }

    return fifo_depth_ * multiplier;
  } else {
    return InternalErrorStrCat("Expected the path to have been validated.");
  }
}

HloInstruction* PipelinePath::GetNewConsumerStage() const {
  return new_consumer_;
}

HloInstruction* PipelinePath::GetOldConsumerStage() const {
  CHECK(finished_);
  return old_consumer_;
}

PipelinePath::Type PipelinePath::GetType() const { return type_; }

StatusOr<std::vector<PipelinePath>> FindPassthroughPipelinePaths(
    PipelineStages& stages) {
  const uint64 num_stages = stages.forward.size() + stages.backward.size();
  // Set up stages in last to first order.
  std::vector<HloInstruction*> stages_last_to_first(num_stages);
  absl::c_copy(stages.backward, stages_last_to_first.begin());
  std::copy(stages.forward.rbegin(), stages.forward.rend(),
            std::next(stages_last_to_first.begin(), stages.backward.size()));

  // Set up the sharding in last to first order.
  std::vector<int64> sharding_last_to_first(stages_last_to_first.size());
  {
    // Get the sharding from the forward stages - they are always expected to
    // have it.
    std::vector<int64> forward_sharding_devices(stages.forward.size());
    for (uint64 i = 0; i != stages.forward.size(); ++i) {
      HloInstruction* forward_stage = stages.forward[i];
      auto optional_device = forward_stage->sharding_unique_device();
      if (!optional_device) {
        return InternalErrorStrCat("Expected stage ", forward_stage->ToString(),
                                   " to have sharding information.");
      }
      forward_sharding_devices[i] = *optional_device;
    }

    if (stages.backward.size()) {
      absl::c_copy(forward_sharding_devices, sharding_last_to_first.begin());
    }
    std::copy(
        forward_sharding_devices.rbegin(), forward_sharding_devices.rend(),
        std::next(sharding_last_to_first.begin(), stages.backward.size()));
  }

  // For each stage we store a map which represents tensors being passed-through
  // the stage. For example given a stage:
  // {
  //   p0 = parameter(0)
  //   p1 = parameter(1)
  //   ...
  //   ROOT t = tuple(a, b, p0, p1)
  // }
  // the inputs p0 and p1 are passed-through the stage, with p0 having an output
  // index 2 (its location in the root tuple) and input index 0 (its parameter
  // number). Similarly p1 has output index 3 and input index 1. Other outputs
  // (a and b) are not inputs to the stage therefore they are not in the map.
  std::vector<absl::flat_hash_map<uint64, uint64>>
      intra_stage_output_to_input_map(num_stages);

  // For each stage we store a map which represents connections between stage
  // inputs and the outputs of the previous stage. For example given stages:
  // s0 = (fp32[], fp32[], fp32[]) pipeline_stage(x, y, z) stage_id = 0
  // s0.0 = gte(s0), index=0
  // s0.1 = gte(s0), index=1
  // s0.2 = gte(s0), index=2
  // s1 = (fp32[]) pipeline_stage(s0.0, s0.1) stage_id = 1
  // s1.0 = gte(s1), index=0
  // s2 = (fp32[], fp32[], fp32[]) pipeline_stage(s0.2, s1.0) stage_id = 2
  //
  // * s1 at input index 0 uses output of s0 at output index 0
  // * s1 at input index 1 uses output of s0 at output index 1
  // * s2 at input index 1 uses output of s1 at output index 0
  // Note that s2 also uses an output from s0, however the previous stage of s2
  // is s1 so it is not included in the map.
  std::vector<absl::flat_hash_map<uint64, uint64>>
      inter_stage_input_to_previous_output_map(num_stages);

  // Build the intra and inter maps.
  for (uint64 stage_idx = 0; stage_idx != num_stages; ++stage_idx) {
    HloInstruction* stage = stages_last_to_first[stage_idx];

    HloComputation* stage_computation = stage->to_apply();
    // Building the intra map.
    // To build up the intra map, we go through the tuple elements of the root
    // instruction of the pipeline stage computation which is expected to be a
    // tuple.
    HloInstruction* root = stage_computation->root_instruction();
    CHECK_EQ(root->opcode(), HloOpcode::kTuple);
    for (uint64 output_idx = 0; output_idx != root->operand_count();
         ++output_idx) {
      const HloInstruction* operand = root->operand(output_idx);
      if (operand->opcode() == HloOpcode::kParameter) {
        const uint64 input_idx = operand->parameter_number();
        if (intra_stage_output_to_input_map[stage_idx].contains(output_idx)) {
          return InternalErrorStrCat("Expected pipeline stage ",
                                     stage->ToString(),
                                     " to not have duplicate outputs.");
        }
        intra_stage_output_to_input_map[stage_idx][output_idx] = input_idx;
      }
    }
    // Building the inter map.
    // Note that the last stage does not a previous stage so we do not build a
    // map for it.
    if (stage_idx == (num_stages - 1)) {
      continue;
    }
    // Go through all the operands to the stage and if they are GTEs on the
    // previous stage, then add them to the map.
    for (uint64 input_idx = 0; input_idx != stage->operand_count();
         ++input_idx) {
      const HloInstruction* operand = stage->operand(input_idx);
      if (operand->opcode() == HloOpcode::kGetTupleElement &&
          operand->operand(0) == stages_last_to_first[stage_idx + 1]) {
        if (inter_stage_input_to_previous_output_map[stage_idx].contains(
                input_idx)) {
          return InternalErrorStrCat("Expected pipeline stage ",
                                     stage->ToString(),
                                     " to not have duplicate inputs.");
        }
        inter_stage_input_to_previous_output_map[stage_idx][input_idx] =
            operand->tuple_index();
      }
    }
  }

  std::vector<PipelinePath> paths;
  // Build the paths.
  // We build the path by traversing a path, note that for each starting point
  // there can only be at most one path.
  for (uint64 stage_idx = 0; stage_idx != num_stages; ++stage_idx) {
    HloInstruction* stage = stages_last_to_first[stage_idx];
    const int64 shard = sharding_last_to_first[stage_idx];
    for (uint64 input_idx = 0; input_idx != stage->operand_count();
         ++input_idx) {
      if (!inter_stage_input_to_previous_output_map[stage_idx].contains(
              input_idx)) {
        // No path to build - skip.
        continue;
      }
      // Create the start of a path.
      PipelinePath path{
          stage, stage_idx, input_idx,
          inter_stage_input_to_previous_output_map[stage_idx].at(input_idx)};

      for (uint64 next_stage_idx = stage_idx + 1; next_stage_idx != num_stages;
           ++next_stage_idx) {
        HloInstruction* next_stage = stages_last_to_first[next_stage_idx];
        const int64 next_shard = sharding_last_to_first[next_stage_idx];
        const bool shard_matches = next_shard == shard;
        path.GetVisitedStages().push_back(next_stage_idx);

        if (shard_matches) {
          // Stop if we reached a stage on the same shard.
          // We stop as soon as possible to avoid creating parallel pipelines.
          if (path.FinishPath(stages)) {
            paths.push_back(path);
          }
          break;
        }

        // Try to extend the path.
        // Use the intra map to see if this output has been threaded through.
        const uint64 output_idx = path.GetOutputsPath().back();
        auto intra_itr =
            intra_stage_output_to_input_map[next_stage_idx].find(output_idx);
        if (intra_itr ==
            intra_stage_output_to_input_map[next_stage_idx].end()) {
          // The output at this index has not been threaded through - therefore
          // there is no path.
          break;
        }
        const uint64 input_idx = intra_itr->second;
        // Now see if the input at that index is the `next` stage by looking at
        // the inter map.
        auto inter_itr =
            inter_stage_input_to_previous_output_map[next_stage_idx].find(
                input_idx);
        if (inter_itr ==
            inter_stage_input_to_previous_output_map[next_stage_idx].end()) {
          // The output at this index has not been threaded through - therefore
          // there is no path.
          break;
        }
        const uint64 next_output_idx = inter_itr->second;
        path.GetInputsPath().push_back(input_idx);
        path.GetOutputsPath().push_back(next_output_idx);
      }
    }
  }
  return paths;
}

}  // namespace poplarplugin
}  // namespace xla
