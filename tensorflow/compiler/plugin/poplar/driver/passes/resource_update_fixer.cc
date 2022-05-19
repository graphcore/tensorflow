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

#include "tensorflow/compiler/plugin/poplar/driver/passes/resource_update_fixer.h"

#include <map>
#include <memory>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {
namespace {

// De-duplicate outputs from the resource update to make sure that each
// GTE tuple index occurs only once and that each GTE is only used once.
StatusOr<bool> FixResourceUpdate(HloInstruction* resource_update,
                                 HloInstruction* caller) {
  HloComputation* comp = resource_update->parent();
  HloComputation* resource_update_comp = resource_update->to_apply();

  // Make sure the resource update and its outer computations have tuples as
  // root instructions.
  TF_ASSIGN_OR_RETURN(bool changed_comp, FixRootInstruction(comp));
  TF_ASSIGN_OR_RETURN(bool changed_resource_update_comp,
                      FixRootInstruction(resource_update_comp));
  bool changed = changed_comp || changed_resource_update_comp;

  HloInstruction* comp_root = comp->root_instruction();
  HloInstruction* resource_update_root =
      resource_update_comp->root_instruction();

  CHECK_EQ(comp_root->opcode(), HloOpcode::kTuple);
  CHECK_EQ(resource_update_root->opcode(), HloOpcode::kTuple);
  const int64_t num_resource_update_outputs =
      ShapeUtil::TupleElementCount(resource_update->shape());

  // Find all the gtes for the resource update.
  std::map<int64_t, std::vector<HloInstruction*>> all_gtes;
  for (HloInstruction* user : resource_update->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement ||
        absl::c_any_of(user->users(),
                       [&comp_root](const HloInstruction* user_of) {
                         return user_of != comp_root;
                       })) {
      return InternalErrorStrCat(
          "Expected the ResourceUpdate outputs to be used by the "
          "root instruction only.");
    }
    all_gtes[user->tuple_index()].push_back(user);
  }

  struct DuplicateOutput {
    int64_t tuple_index;
    int64_t root_operand_index;
  };

  std::vector<DuplicateOutput> duplicate_outputs;
  for (auto pair : all_gtes) {
    const int64_t tuple_index = pair.first;
    const std::vector<HloInstruction*>& gtes = pair.second;
    // Zeroth GTE is allowed a single user, all other uses are duplicates.
    for (int64_t i = 0; i != gtes.size(); ++i) {
      const int64_t offset = i == 0;
      const auto& root_indices = comp_root->OperandIndices(gtes[i]);
      for (int64_t j = offset; j != root_indices.size(); ++j) {
        duplicate_outputs.push_back({tuple_index, root_indices[j]});
      }
    }
  }

  if (duplicate_outputs.empty()) {
    return changed;
  }

  // Create a new root tuple for the resource update computation.
  std::vector<HloInstruction*> new_outputs = {
      resource_update_root->operands().begin(),
      resource_update_root->operands().end()};

  CHECK_EQ(num_resource_update_outputs, new_outputs.size());

  // Copy duplicate outputs.
  for (const auto& duplicate_output : duplicate_outputs) {
    HloInstruction* output = new_outputs[duplicate_output.tuple_index];
    HloInstruction* copied_output = resource_update_comp->AddInstruction(
        HloInstruction::CreateUnary(output->shape(), HloOpcode::kCopy, output));
    output->SetupDerivedInstruction(copied_output);
    new_outputs.push_back(copied_output);
  }

  // Create the new root tuple.
  HloInstruction* new_resource_update_root =
      resource_update_comp->AddInstruction(
          HloInstruction::CreateTuple(new_outputs));

  resource_update_root->SetupDerivedInstruction(new_resource_update_root);
  resource_update_comp->set_root_instruction(new_resource_update_root, true);
  *resource_update->mutable_shape() = new_resource_update_root->shape();

  if (resource_update_root->user_count()) {
    // Wire outputs of the new root as inputs to the old tuple so that the users
    // don't have to be modified.
    for (int64_t i = 0; i != num_resource_update_outputs; ++i) {
      TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                          MakeGetTupleElementHlo(new_resource_update_root, i));
      TF_RETURN_IF_ERROR(resource_update_root->ReplaceOperandWith(i, gte));
    }
  } else {
    TF_RETURN_IF_ERROR(
        resource_update_comp->RemoveInstruction(resource_update_root));
  }

  // Add new GTEs from the updated resource update and use them in the root
  // instruction.
  for (int64_t i = 0; i != duplicate_outputs.size(); ++i) {
    const int64_t root_operand_index = duplicate_outputs[i].root_operand_index;
    TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                        MakeGetTupleElementHlo(
                            resource_update, num_resource_update_outputs + i));
    HloInstruction* current_operand =
        comp_root->mutable_operand(root_operand_index);
    TF_RETURN_IF_ERROR(comp_root->ReplaceOperandWith(root_operand_index, gte));
    if (current_operand->user_count() == 0) {
      TF_RETURN_IF_ERROR(comp->RemoveInstruction(current_operand));
    }
  }

  return true;
}
}  // namespace

StatusOr<bool> ResourceUpdateFixer::Run(HloModule* module) {
  std::vector<HloInstruction*> resource_updates;
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (IsResourceUpdate(inst)) {
        resource_updates.push_back(inst);
      }
    }
  }

  if (resource_updates.empty()) {
    return false;
  }

  VLOG(2) << "Before ResourceUpdateFixer:";
  XLA_VLOG_LINES(2, module->ToString());
  bool changed = false;

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  for (HloInstruction* resource_update : resource_updates) {
    auto& call_graph_node = call_graph->GetNode(resource_update->parent());
    auto callsites = call_graph_node.caller_callsites();
    CHECK_EQ(callsites.size(), 1);
    HloInstruction* caller = callsites[0].instruction();
    CHECK(IsRepeatLoop(caller) || IsPipelineOp(caller));
    TF_ASSIGN_OR_RETURN(bool changed_resource_update,
                        FixResourceUpdate(resource_update, caller));
    changed |= changed_resource_update;
  }

  if (changed) {
    VLOG(2) << "After ResourceUpdateFixer:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the Pipeline.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
