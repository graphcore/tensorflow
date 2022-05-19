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

#include "tensorflow/compiler/plugin/poplar/driver/passes/inter_tileset_copy_inserter.h"

#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/inter_tileset_copy.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {
namespace {

Status SetTileset(HloInstruction* inst, Tileset tileset) {
  TF_ASSIGN_OR_RETURN(auto backend_config,
                      inst->backend_config<PoplarBackendConfig>());
  backend_config.set_tileset(tileset);
  return inst->set_backend_config(backend_config);
}

Status InsertInterTilesetCopy(
    HloComputation* comp, HloInstruction* source,
    const std::vector<HloInstruction*>& destinations) {
  CHECK(!destinations.empty());

  TF_ASSIGN_OR_RETURN(const auto src_tileset, GetTileset(source));
  TF_ASSIGN_OR_RETURN(const auto dst_tileset, GetTileset(destinations.front()));

  VLOG(3) << "Insert copy of " << source->ToShortString() << " to "
          << destinations.front()->ToShortString() << " : "
          << Tileset_Name(dst_tileset);

  HloInstruction* copy = nullptr;
  if (source->shape().IsTuple()) {
    std::vector<HloInstruction*> gtes;
    gtes.reserve(source->shape().tuple_shapes_size());

    for (int64_t i = 0; i != source->shape().tuple_shapes_size(); ++i) {
      // Add a GTE.
      auto gte = HloInstruction::CreateGetTupleElement(
          source->shape().tuple_shapes(i), source, i);

      // Set the GTE to the source tileset.
      TF_RETURN_IF_ERROR(SetTileset(gte.get(), src_tileset));
      if (source->has_sharding()) {
        gte->set_sharding(source->sharding());
      }

      // Add GTE to the computation.
      gtes.push_back(comp->AddInstruction(std::move(gte)));
    }

    // Create a tuple to recreated the source original tuple.
    copy = comp->AddInstruction(HloInstruction::CreateTuple(gtes));

    // Assign the created tuple to the destintion tileset
    TF_RETURN_IF_ERROR(SetTileset(copy, dst_tileset));

    for (auto gte : gtes) {
      // Recursively add inter-tileset-copies between the gte and tuple.
      TF_RETURN_IF_ERROR(InsertInterTilesetCopy(comp, gte, {copy}));
    }
  } else {
    // No tuple, just add a inter-tileset-copy between the source and
    // destination.
    CHECK(source->shape().IsArray());
    copy = comp->AddInstruction(CreateInterTilesetCopy(source));
  }

  TF_RETURN_IF_ERROR(SetTileset(copy, dst_tileset));
  if (source->has_sharding()) {
    copy->set_sharding(source->sharding());
  }

  for (auto* dst : destinations) {
    TF_ASSIGN_OR_RETURN(const auto tileset, GetTileset(dst));
    CHECK_EQ(tileset, dst_tileset);

    // Handle the case where an instruction uses the source multiple times.
    const auto indices = dst->OperandIndices(source);
    CHECK(!indices.empty());
    for (auto index : indices) {
      TF_RETURN_IF_ERROR(dst->ReplaceOperandWith(index, copy));
    }
  }

  return Status::OK();
}

// Returns true if this instruction can output tensors that would need to be
// copied across the tilesets (e.g. across disjoint Poplar virtual graph
// boundaries). Only entities that are mapped to a specific set of tiles
// (like poplar::Tensor, but not poplar::RemoteBuffer), need to be copied.
bool IsTensorProducer(const HloInstruction* inst) {
  if (inst->shape().IsToken()) {
    // Tokens are not tensors and cannot be copied.
    return false;
  }

  // The outputs of these remote buffer stores represent the remote buffer, and
  // not a tensor that need to be copied.
  if (IsPoplarInstruction(RemoteParameterStore)(inst)) {
    return false;
  }
  if (IsPoplarInstruction(BufferStoreSlice)(inst)) {
    return false;
  }

  return true;
}

// Returns true if an edge between the two instructions can transfer Poplar
// tensors that would need to be copied copied across the tilesets.
bool IsTensorEdge(const HloInstruction* src, const HloInstruction* dst) {
  if (IsPoplarInstruction(RemoteParameterLoad)(dst) ||
      IsPoplarInstruction(RemoteParameterStore)(dst) ||
      IsPoplarInstruction(BufferLoadSlice)(dst) ||
      IsPoplarInstruction(BufferStoreSlice)(dst)) {
    const bool is_remote_buffer = dst->operand_index(src) == 0;
    return !is_remote_buffer;
  }

  return true;
}

using Edges = HloInstructionMap<std::vector<HloInstruction*>>;

StatusOr<Edges> FindInterTilesetEdges(HloComputation* comp) {
  Edges edges;

  for (auto* src : comp->MakeInstructionPostOrder()) {
    if (!IsTensorProducer(src)) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(const auto src_tileset, GetTileset(src));

    for (auto* dst : src->users()) {
      if (!IsTensorEdge(src, dst)) {
        continue;
      }

      TF_ASSIGN_OR_RETURN(const auto dst_tileset, GetTileset(dst));
      if (src_tileset != dst_tileset) {
        edges[src].push_back(dst);
      }
    }
  }

  return edges;
}

}  // namespace

StatusOr<bool> InterTilesetCopyInserter::Run(HloModule* module) {
  bool changed = false;

  VLOG(2) << "Before InterTilesetCopyInserter:";
  XLA_VLOG_LINES(2, module->ToString());

  for (auto* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    // Find edges between instructions on different tilesets.
    TF_ASSIGN_OR_RETURN(const auto edges, FindInterTilesetEdges(comp));

    // Add copies on the edges.
    for (const auto& source : edges) {
      TF_RETURN_IF_ERROR(
          InsertInterTilesetCopy(comp, source.first, source.second));
      changed = true;
    }
  }

  if (changed) {
    VLOG(2) << "After InterTilesetCopyInserter:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made.";
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
