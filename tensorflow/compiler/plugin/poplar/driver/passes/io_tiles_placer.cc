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

bool IsInSerialPipeline(const HloInstruction* inst,
                        const CallGraph& call_graph) {
  const auto callers = call_graph.GetNode(inst->parent()).caller_callsites();
  return callers.size() == 1 && IsPipelineOp(callers[0].instruction()) &&
         IsBatchSerializedPipelineOp(callers[0].instruction());
}

static int64 GetMaxAvailableIoBytes(
    const int64 num_io_tiles, const int64 bytes_per_io_tile,
    const double available_io_tile_memory_proportion) {
  return static_cast<int64>(num_io_tiles * bytes_per_io_tile *
                            available_io_tile_memory_proportion);
}

static int64 GetInstructionBufferSize(const HloInstruction* inst) {
  const auto& shape = inst->shape();
  // The host exchange instructions are either 1 to 1 or 1 to token so
  // only need to look at either result of operand
  if (shape.IsToken()) {
    // if token sum up size of all operands
    return absl::c_accumulate(inst->operands(), static_cast<int64>(0),
                              [](int64 sum, const HloInstruction* i) {
                                return sum + GetInstructionBufferSize(i);
                              });
  }
  return GetByteSizeOfTotalShape(shape);
}

int64 GetMaxLiveBytes(const HloInstructionSequence& potential_io_tile_insts) {
  // Looks like the Heap simulator doesn't really work for this purpose
  // as none of these instructions allocate. Use just accumulation of size
  // until poplar specific liveness simulator is implemented
  int64 ans = absl::c_accumulate(potential_io_tile_insts.instructions(),
                                 static_cast<int64>(0),
                                 [](int64 sum, const HloInstruction* inst) {
                                   return sum + GetInstructionBufferSize(inst);
                                 });

  return ans;
}

bool ShouldBeOnIoTiles(const HloInstruction* inst,
                       const CallGraph& call_graph) {
  switch (inst->opcode()) {
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
      // Currently incompatible with batch serial pipeline lowering.
      return !IsInSerialPipeline(inst, call_graph);
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
  HloInstructionSequence potential_io_tile_insts;

  for (auto* inst : comp->MakeInstructionPostOrder()) {
    if (ShouldBeOnIoTiles(inst, call_graph)) {
      potential_io_tile_insts.push_back(inst);
    }
  }

  const int64 max_live_bytes = GetMaxLiveBytes(potential_io_tile_insts);
  const int64 target_io_bytes = GetMaxAvailableIoBytes(
      num_io_tiles, bytes_per_io_tile, AvailableMemoryProportion());

  const bool insts_fit_on_io_tiles = max_live_bytes < target_io_bytes;
  const bool change =
      insts_fit_on_io_tiles && !potential_io_tile_insts.instructions().empty();
  if (change) {
    for (auto* inst : potential_io_tile_insts.instructions()) {
      TF_RETURN_IF_ERROR(AssignToIoTilesAndPropagateToGteUsers(inst));
    }
  } else if (!insts_fit_on_io_tiles) {
    LOG(INFO) << absl::StrCat(
        "Computation too large to fit on IO tiles, ", max_live_bytes,
        " >= ", target_io_bytes,
        ". Currently the number of IO tiles is set to ", num_io_tiles,
        " with the available memory"
        " proportion set to ",
        AvailableMemoryProportion(),
        ". To try and fit all the data into IO tiles you either need to"
        " increase the number of IO tiles or the available memory proportion"
        " using the `ipu.config.IPUConfig.io_tiles` category.");
  }
  return change;
}

static bool PoplarInstructionUsesIOTiles(const HloInstruction* inst) {
  // The call to should be on io tiles ensures this GetTileset will succeed
  return GetTileset(inst).ValueOrDie() == TILESET_IO_TILES;
}

static bool UsesIOTiles(const HloInstruction* inst,
                        const CallGraph& call_graph) {
  return ShouldBeOnIoTiles(inst, call_graph) &&
         PoplarInstructionUsesIOTiles(inst);
}

static bool UsesIOTiles(const HloComputation* computation,
                        const CallGraph& call_graph) {
  return absl::c_any_of(computation->instructions(),
                        [&](const HloInstruction* inst) {
                          return UsesIOTiles(inst, call_graph);
                        });
}

static bool UsesIOTiles(const std::vector<HloComputation*>& computations,
                        const CallGraph& call_graph) {
  return absl::c_any_of(computations, [&](const HloComputation* comp) {
    return UsesIOTiles(comp, call_graph);
  });
}

static bool UpdateNumIoTiles(int64& resources_num_io_tiles_,
                             const std::vector<HloComputation*>& computations,
                             const CallGraph& call_graph) {
  const bool any_instruction_on_io_tiles =
      UsesIOTiles(computations, call_graph);
  if (any_instruction_on_io_tiles || resources_num_io_tiles_ == 0) {
    return false;  // Haven't changed resources so return false
  }
  // Remove io tiles from resources/graph as no ops placed on them
  resources_num_io_tiles_ = 0;
  return true;
}

StatusOr<bool> IoTilesPlacer::Run(HloModule* module) {
  if (!enabled_) {
    return false;
  }

  VLOG(2) << "Before IoTilesPlacer:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;

  const auto call_graph = CallGraph::Build(module);

  auto computations = module->MakeComputationPostOrder();

  for (auto* comp : computations) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(const bool computation_changed,
                        RunOnComputation(comp, *call_graph));
    changed |= computation_changed;
  }

  changed |=
      UpdateNumIoTiles(resources_num_io_tiles_, computations, *call_graph);

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
