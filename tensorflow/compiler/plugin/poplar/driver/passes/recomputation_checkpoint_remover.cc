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

#include "tensorflow/compiler/plugin/poplar/driver/passes/recomputation_checkpoint_remover.h"

#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
namespace poplarplugin {
StatusOr<bool> RecomputationCheckpointRemover::Run(HloModule* module) {
  std::vector<HloInstruction*> to_remove;
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    absl::c_copy_if(comp->MakeInstructionPostOrder(),
                    std::back_inserter(to_remove),
                    IsPoplarInstruction(PoplarOp::RecomputationCheckpoint));
  }

  for (HloInstruction* inst : to_remove) {
    VLOG(2) << "Removing " << inst->ToString();
    TF_RETURN_IF_ERROR(
        inst->parent()->ReplaceInstruction(inst, inst->mutable_operand(0)));
  }

  return to_remove.size();
}
}  // namespace poplarplugin
}  // namespace xla
