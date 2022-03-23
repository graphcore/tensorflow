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

#include "tensorflow/compiler/plugin/poplar/driver/passes/apply_recompute_suggestion.h"

#include <unordered_set>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/recompute.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

namespace {
bool IsRecomputeInstruction(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::SuggestRecompute)(inst);
}

bool IsBlockRecomputeInstruction(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::BlockRecompute)(inst);
}
}  // namespace

bool UsesRecomputationSuggestions(const HloModule* module) {
  // Cant use IsRecomputeInstruction since we're expecting this function
  // to be used on Hlo before its had CustomOpReplacer called on it, which
  // would replace the custom call with a recompute instruction.
  const auto is_recomputation_instr = [](const HloInstruction* inst) {
    return (inst->opcode() == HloOpcode::kCustomCall &&
            inst->custom_call_target() == "SuggestRecompute");
  };

  for (auto comp : module->computations()) {
    if (absl::c_any_of(comp->instructions(), is_recomputation_instr)) {
      return true;
    }
  }

  return false;
}

StatusOr<bool> ApplyRecomputeSuggestion::Run(HloModule* module) {
  std::vector<HloCustomCallInstruction*> custom_calls;

  for (auto comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (auto inst : comp->MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kCustomCall) {
        auto custom_call = Cast<HloCustomCallInstruction>(inst);

        // Have we found a recomputation suggestion with more than one user?
        if (IsRecomputeInstruction(custom_call)) {
          auto operand = custom_call->mutable_operand(0);

          // If the operand is also a recompute suggestion, remove this one.
          // Otherwise, check we haven't reached a blocker.
          if (IsRecomputeInstruction(operand)) {
            TF_RETURN_IF_ERROR(custom_call->ReplaceAllUsesWith(operand));
            TF_RETURN_IF_ERROR(comp->RemoveInstruction(custom_call));

            return true;
          } else if (!IsBlockRecomputeInstruction(operand)) {
            // For each operand, propogate the recomputation suggestion upward.
            for (int i = 0; i < operand->operand_count(); ++i) {
              auto input = operand->mutable_operand(i);
              if (!IsRecomputeInstruction(input)) {
                auto recomp_input =
                    comp->AddInstruction(CreateSuggestRecompute(input));
                input->SetupDerivedInstruction(recomp_input);
                TF_RETURN_IF_ERROR(input->ReplaceAllUsesWith(recomp_input));
              }
            }

            std::unordered_set<const HloInstruction*> unique_users(
                custom_call->users().begin(), custom_call->users().end());

            VLOG(3) << "Recomputing " << operand->name();

            // Deterministicly ordered set of unique users.
            auto inst_post_order = comp->MakeInstructionPostOrder();
            auto p = absl::c_stable_partition(
                inst_post_order, [&unique_users](const HloInstruction* inst) {
                  return unique_users.count(inst) > 0;
                });
            inst_post_order.erase(p, inst_post_order.end());

            // We can keep one of the users using the original instruction.
            TF_RETURN_IF_ERROR(
                custom_call->ReplaceUseWith(inst_post_order.front(), operand));
            inst_post_order.erase(inst_post_order.begin());

            // For each unique user, replace this instruction with a fresh clone
            // of the instruction.
            for (auto& user : inst_post_order) {
              auto replacement = comp->AddInstruction(operand->Clone());

              // Add control dependencies to ensure the clone is executed last
              for (auto predecessor : user->operands()) {
                if (!IsRecomputeInstruction(predecessor) &&
                    !IsBlockRecomputeInstruction(predecessor) &&
                    predecessor->opcode() != HloOpcode::kConstant &&
                    predecessor->opcode() != HloOpcode::kParameter) {
                  predecessor->AddControlDependencyTo(replacement);
                }
              }

              // Copy any control predecessors from the user to the clone.
              // This means that control predecessors will propogate up the
              // recompute graph.
              for (auto predecessor : user->control_predecessors()) {
                if (!IsRecomputeInstruction(predecessor) &&
                    !IsBlockRecomputeInstruction(predecessor) &&
                    predecessor->opcode() != HloOpcode::kConstant &&
                    predecessor->opcode() != HloOpcode::kParameter) {
                  predecessor->AddControlDependencyTo(replacement);
                }
              }

              operand->SetupDerivedInstruction(replacement);
              TF_RETURN_IF_ERROR(
                  custom_call->ReplaceUseWith(user, replacement));
            }

            // Remove the recomputation suggestion from the graph.
            TF_RETURN_IF_ERROR(comp->RemoveInstruction(custom_call));

            return true;
          }
        }
      }
    }
  }

  return false;
}

}  // namespace poplarplugin
}  // namespace xla
