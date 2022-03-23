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

#include "tensorflow/compiler/plugin/poplar/driver/passes/dead_control_dependencies_elimination.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace poplarplugin {

StatusOr<bool> DeadControlDependenciesElimination::Run(HloModule* module) {
  bool changed = false;

  VLOG(2) << "Before DeadControlDependenciesElimination:";
  XLA_VLOG_LINES(2, module->ToString());

  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (inst->user_count() == 0 && comp->root_instruction() != inst &&
          !inst->HasSideEffect()) {
        VLOG(2) << "Dropping control dependencies for " << inst->ToString();
        // To preserve topological constraints, make sure that control
        // predecessors are copied to control successors.
        for (HloInstruction* successor : inst->control_successors()) {
          for (HloInstruction* predecessor : inst->control_predecessors()) {
            TF_RETURN_IF_ERROR(predecessor->AddControlDependencyTo(successor));
          }
        }
        changed = inst->control_successors().size() ||
                  inst->control_predecessors().size();

        TF_RETURN_IF_ERROR(inst->DropAllControlDeps());
        if (inst->opcode() != HloOpcode::kParameter) {
          CHECK(comp->IsSafelyRemovable(inst));
        }
      }
    }
  }

  if (changed) {
    VLOG(2) << "After DeadControlDependenciesElimination:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes.";
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
