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

#include "tensorflow/compiler/plugin/poplar/driver/passes/io_tiles_placer.h"

#include <algorithm>
#include <map>
#include <queue>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

Status SetIoTileset(HloInstruction* inst) {
  TF_ASSIGN_OR_RETURN(auto backend_config,
                      inst->backend_config<PoplarBackendConfig>());
  backend_config.set_tileset(TILESET_IO_TILES);
  return inst->set_backend_config(backend_config);
}

Status AssignToIoTilesAndPropagateToGteUsers(HloInstruction* inst) {
  VLOG(3) << "Placing on IO tiles: " << inst->ToShortString();
  TF_RETURN_IF_ERROR(SetIoTileset(inst));
  for (auto* gte : inst->users()) {
    if (gte->opcode() == HloOpcode::kGetTupleElement) {
      TF_RETURN_IF_ERROR(AssignToIoTilesAndPropagateToGteUsers(gte));
    }
  }
  return Status::OK();
}

bool IsInPipeline(const HloInstruction* inst, const CallGraph& call_graph) {
  const auto callers = call_graph.GetNode(inst->parent()).caller_callsites();
  return callers.size() == 1 && IsPipelineOp(callers[0].instruction());
}

bool ShouldBeOnIoTiles(const HloInstruction* inst,
                       const CallGraph& call_graph) {
  switch (inst->opcode()) {
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
      // Currently incompatible with pipeline lowering.
      return !IsInPipeline(inst, call_graph);
    case HloOpcode::kCustomCall:
      return IsPoplarInstruction(RemoteParameterLoad)(inst) ||
             IsPoplarInstruction(RemoteParameterStore)(inst) ||
             IsPoplarInstruction(BufferLoadSlice)(inst) ||
             IsPoplarInstruction(BufferStoreSlice)(inst) ||
             IsPoplarInstruction(RecvFromHost)(inst) ||
             IsPoplarInstruction(SendToHost)(inst);
    default:
      return false;
  }
}

StatusOr<bool> IoTilesPlacer::RunOnComputation(HloComputation* comp,
                                               const CallGraph& call_graph) {
  bool changed = false;

  for (auto* inst : comp->MakeInstructionPostOrder()) {
    if (ShouldBeOnIoTiles(inst, call_graph)) {
      TF_RETURN_IF_ERROR(AssignToIoTilesAndPropagateToGteUsers(inst));
      changed = true;
    }
  }

  return changed;
}

StatusOr<bool> IoTilesPlacer::Run(HloModule* module) {
  if (!enabled_) {
    return false;
  }

  VLOG(2) << "Before IoTilesPlacer:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;

  const auto call_graph = CallGraph::Build(module);

  for (auto* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(const bool computation_changed,
                        RunOnComputation(comp, *call_graph));
    changed |= computation_changed;
  }

  if (changed) {
    VLOG(2) << "After IoTilesPlacer:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
