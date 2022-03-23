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

#include "tensorflow/compiler/plugin/poplar/driver/passes/repeat_loop_copy_inserter.h"

#include <memory>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {

/**
 * Insert a regular copy to avoid clobbering the input to the
 * inter-tileset-copy. This can happen when the repeat loop is overlapping
 * computation and IO, and there is any inplacing/aliasing in the computation.
 */
StatusOr<bool> InsertCopyBeforeInterTilesetCopy(HloComputation* computation) {
  auto instructions = computation->MakeInstructionPostOrder();
  auto itr = absl::c_stable_partition(
      instructions, IsPoplarInstruction(PoplarOp::InterTilesetCopy));

  instructions.erase(itr, instructions.end());

  // Replace the inter-tileset-copy operand with a copy of that operand.
  bool changed = false;
  for (HloInstruction* inst : instructions) {
    HloInstruction* operand = inst->mutable_operand(0);

    if (operand->user_count() > 1) {
      HloInstruction* copy =
          computation->AddInstruction(HloInstruction::CreateUnary(
              operand->shape(), HloOpcode::kCopy, operand));

      if (operand->has_sharding()) {
        copy->set_sharding(operand->sharding());
      }
      TF_RETURN_IF_ERROR(operand->ReplaceUseWith(inst, copy));

      changed = true;
    }
  }

  return changed;
}
}  // namespace

StatusOr<bool> RepeatLoopCopyInserter::Run(HloModule* module) {
  bool changed = false;
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);

  VLOG(2) << "Before RepeatLoopCopyInserter:";
  XLA_VLOG_LINES(2, module->ToString(HloPrintOptions::ShortParsable()));

  for (auto computation : module->MakeComputationPostOrder()) {
    auto callers = call_graph->GetComputationCallers(computation);

    // Only change repeat loops with a single callsite.
    if (callers.size() == 1 && IsRepeatLoop(callers.front())) {
      TF_ASSIGN_OR_RETURN(bool changed_comp,
                          InsertCopyBeforeInterTilesetCopy(computation));
      changed |= changed_comp;
    }
  }

  if (changed) {
    VLOG(2) << "After RepeatLoopCopyInserter:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the Pipeline.";
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
