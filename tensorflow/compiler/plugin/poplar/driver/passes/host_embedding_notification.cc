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

#include <set>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/passes/host_embedding_notification.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/host_embedding.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

namespace xla {
namespace poplarplugin {

namespace {
// Add all the embedding lookup/update instructions in the computation to the
// map.
void PopulateEmbeddingMap(
    absl::flat_hash_map<std::string, std::vector<HloInstruction*>>&
        embedding_instructions,
    HloComputation* computation) {
  for (auto instruction : computation->MakeInstructionPostOrder()) {
    if (IsPoplarInstruction(PoplarOp::HostEmbeddingLookup, instruction)) {
      HloHostEmbeddingLookupInstruction* lookup =
          Cast<HloHostEmbeddingLookupInstruction>(instruction);

      embedding_instructions[lookup->EmbeddingId()].push_back(instruction);
    }
    if (IsPoplarInstruction(PoplarOp::HostEmbeddingUpdate, instruction)) {
      HloHostEmbeddingUpdateInstruction* update =
          Cast<HloHostEmbeddingUpdateInstruction>(instruction);

      embedding_instructions[update->EmbeddingId()].push_back(instruction);
    }
  }
}
}  // namespace

StatusOr<bool> HostEmbeddingNotification::Run(HloModule* module) {
  std::vector<HloInstruction*> notication_insts;
  std::vector<HloInstruction*> resource_updates;

  absl::flat_hash_map<std::string, std::vector<HloInstruction*>>
      embedding_instructions;

  // Find all the notification, resource updates, and lookup/updates.
  for (auto computation : module->MakeComputationPostOrder()) {
    absl::c_copy_if(computation->MakeInstructionPostOrder(),
                    std::back_inserter(notication_insts),
                    IsPoplarInstruction(PoplarOp::HostEmbeddingNotify));

    absl::c_copy_if(computation->MakeInstructionPostOrder(),
                    std::back_inserter(resource_updates), IsResourceUpdate);

    PopulateEmbeddingMap(embedding_instructions, computation);
  }

  // Nothing to do, if there aren't any notifications.
  if (notication_insts.empty()) {
    return false;
  }

  // More than one resource update?
  if (resource_updates.size() > 1) {
    return FailedPrecondition(
        "More than once resource update computation found.");
  }

  // When not using gradient accumulation, maybe this should also error.
  // Force the notifications to be after the lookup/update instructions.
  if (resource_updates.empty()) {
    bool changed = false;

    for (auto notication_inst : notication_insts) {
      HloHostEmbeddingNotifyInstruction* notify =
          Cast<HloHostEmbeddingNotifyInstruction>(notication_inst);

      for (auto target : embedding_instructions[notify->EmbeddingId()]) {
        TF_RETURN_IF_ERROR(target->AddControlDependencyTo(notication_inst));
        changed |= true;
      }
    }

    return changed;
  }

  // resource_updates.size() == 1
  absl::flat_hash_set<std::string> seen_notification;
  for (auto notication_inst : notication_insts) {
    HloHostEmbeddingNotifyInstruction* notify =
        Cast<HloHostEmbeddingNotifyInstruction>(notication_inst);

    // Check whether we've already added a notifcation for the embedding id to
    // the resource update.
    if (!seen_notification.contains(notify->EmbeddingId())) {
      resource_updates.back()->to_apply()->AddInstruction(
          CreateHloHostEmbeddingNotify(notify->EmbeddingId()));

      seen_notification.insert(notify->EmbeddingId());
    }

    // Remove the original instruction.
    TF_RETURN_IF_ERROR(notify->parent()->RemoveInstruction(notify));
  }

  // Ordered set of embedding ids.
  std::set<std::string> embedding_ids(seen_notification.begin(),
                                      seen_notification.end());

  // Make sure the embedding updates are complete before the resource update.
  for (auto embedding_id : embedding_ids) {
    for (auto instruction : embedding_instructions[embedding_id]) {
      auto root = instruction->parent()->root_instruction();

      TF_RETURN_IF_ERROR(instruction->AddControlDependencyTo(root));

      if (instruction->parent() == resource_updates.back()->parent()) {
        TF_RETURN_IF_ERROR(
            instruction->AddControlDependencyTo(resource_updates.back()));
      }
    }
  }

  return !notication_insts.empty();
}

}  // namespace poplarplugin
}  // namespace xla
