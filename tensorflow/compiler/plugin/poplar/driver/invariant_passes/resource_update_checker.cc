/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/invariant_passes/resource_update_checker.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

StatusOr<bool> ResourceUpdateChecker::Run(HloModule* module) {
  for (const HloComputation* comp : module->computations()) {
    for (const HloInstruction* inst : comp->instructions()) {
      if (IsResourceUpdate(inst)) {
        absl::flat_hash_set<int64> seen_indices;
        for (const HloInstruction* user : inst->users()) {
          if (user->opcode() != HloOpcode::kGetTupleElement) {
            return InternalError(
                "Expected all users of a resource update to be GTE "
                "instructions.");
          }

          if (seen_indices.contains(user->tuple_index())) {
            return InternalError(
                "Expected all users of a resource update to be unique GTE "
                "instructions.");
          }

          if (user->user_count() != 1) {
            return InternalError(
                "Expected all users of a resource update to be unique GTE "
                "instructions with a single user.");
          }

          const HloInstruction* user_of = user->users()[0];
          if (user_of->OperandIndices(user).size() != 1) {
            return InternalError(
                "Expected all users of a resource update to be unique GTE "
                "instructions with single usage.");
          }

          seen_indices.insert(user->tuple_index());
        }
      }
    }
  }
  return false;
}

}  // namespace poplarplugin
}  // namespace xla
