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

#include "tensorflow/compiler/plugin/poplar/driver/passes/dropout_with_reference_finder.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/dropout_hlo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"

namespace xla {
namespace poplarplugin {
namespace {
bool IsDropout(const HloInstruction* inst) {
  return IsPoplarInstruction(Dropout, inst);
}

bool IsDropoutWhichCanCreateReferenceTensor(const HloInstruction* inst) {
  return IsDropout(inst) && Cast<HloDropout>(inst)->CanCreateReferenceTensor();
}

// Run Hlo passes to remove all the redundant tuples and GTEs.
// Note that only the passes which will not change the mapping of custom calls
// from the flattened to the real model are safe to run.
Status SimplifyModule(HloModule* module) {
  HloPassPipeline pipeline("Flattened module simplifier");
  auto& pass = pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification");
  pass.AddPass<PoplarAlgebraicSimplifier>();
  pass.AddPass<HloDCE>();
  pass.AddPass<TupleSimplifier>();
  pass.AddPass<HloDCE>();
  return pipeline.Run(module).status();
}

std::vector<HloInstruction*> GetDropoutInstructions(HloComputation* comp) {
  std::vector<HloInstruction*> dropouts;
  for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
    if (IsDropoutWhichCanCreateReferenceTensor(inst)) {
      dropouts.push_back(inst);
    }
  }
  return dropouts;
}

Status ReplaceDropout(HloInstruction* inst, const std::string& reference_key) {
  HloComputation* comp = inst->parent();
  HloDropout* dropout = Cast<HloDropout>(inst);
  return comp->ReplaceWithNewInstruction(
      dropout, CreateDropoutWithReference(
                   dropout->mutable_operand(0), dropout->mutable_operand(1),
                   dropout->Rate(), dropout->Scale(), dropout->NoiseShape(),
                   reference_key));
}

Status ReplaceDropoutPair(HloInstruction* fwd, HloInstruction* bwd) {
  VLOG(2) << "Found " << fwd->ToString() << " linked to " << bwd->ToString();
  // Use the unique name of the instruction as a key for the references.
  const std::string reference_key = fwd->name();

  TF_RETURN_IF_ERROR(ReplaceDropout(fwd, reference_key));
  TF_RETURN_IF_ERROR(ReplaceDropout(bwd, reference_key));
  return Status::OK();
}
}  // namespace

DropoutWithReferenceFinder::DropoutWithReferenceFinder(
    CompilerAnnotations& annotations)
    : annotations_(annotations) {}

StatusOr<bool> DropoutWithReferenceFinder::Run(HloModule* module) {
  // Check that there is a flattened module.
  if (!annotations_.flattened_module) {
    return FailedPrecondition("No flattened module created.");
  }

  VLOG(2) << "Before DropoutWithReferenceFinder:";
  XLA_VLOG_LINES(2, module->ToString(HloPrintOptions::ShortParsable()));

  HloModule* flat_module = annotations_.flattened_module.get();
  HloComputation* entry_flat = flat_module->entry_computation();

  if (GetDropoutInstructions(entry_flat).empty()) {
    return false;
  }

  // Simplify the flattened module.
  TF_RETURN_IF_ERROR(SimplifyModule(flat_module));

  // Match up all the dropouts.
  auto dropouts = GetDropoutInstructions(entry_flat);

  absl::flat_hash_set<HloInstruction*> matched;
  HloInstructionMap<HloInstruction*> fwd_to_bwd_dropouts;

  std::unique_ptr<HloReachabilityMap> reachability_map =
      HloReachabilityMap::Build(entry_flat);

  const HloInstruction* entry_flat_root = entry_flat->root_instruction();

  // Try to match from the forward dropout to the backward one.
  for (HloInstruction* fwd_dropout : dropouts) {
    if (matched.contains(fwd_dropout)) {
      continue;
    }

    // Make sure that the seed input is independent of all other dropouts.
    auto dependent_seed = [fwd_dropout,
                           &reachability_map](const HloInstruction* other) {
      if (fwd_dropout == other) {
        return false;
      }
      return reachability_map->IsReachable(other, fwd_dropout->operand(1));
    };

    if (absl::c_any_of(dropouts, dependent_seed)) {
      continue;
    }

    // Make sure that there is only a single user of the dropout seed otherwise
    // reference tensor cannot be used.
    auto statusor_gte = GetUniqueGTEUser(fwd_dropout, /*tuple_index=*/1);
    if (!statusor_gte.ok()) {
      continue;
    }

    HloInstruction* gte = statusor_gte.ValueOrDie();
    if (gte->user_count() != 1) {
      continue;
    }

    HloInstruction* bwd_dropout = gte->users()[0];
    if (!IsDropout(bwd_dropout) || matched.contains(bwd_dropout)) {
      continue;
    }

    // The user dropout cannot create a reference tensor too.
    if (IsDropoutWhichCanCreateReferenceTensor(bwd_dropout)) {
      continue;
    }

    // Make sure that the bwd dropout has no users for the seed.
    statusor_gte = GetUniqueGTEUser(bwd_dropout, /*tuple_index=*/1);
    if (statusor_gte.ok()) {
      continue;
    }

    // Neither fwd or bwd dropout can be a root instruction.
    if (fwd_dropout == entry_flat_root || bwd_dropout == entry_flat_root) {
      continue;
    }

    // Found a match.
    fwd_to_bwd_dropouts[fwd_dropout] = bwd_dropout;
    matched.insert(fwd_dropout);
    matched.insert(bwd_dropout);
  }

  for (auto& pair : fwd_to_bwd_dropouts) {
    HloInstruction* fwd = annotations_.flattened_inst_map_bwd.at(pair.first);
    HloInstruction* bwd = annotations_.flattened_inst_map_bwd.at(pair.second);
    TF_RETURN_IF_ERROR(ReplaceDropoutPair(fwd, bwd));
  }

  const bool changed = fwd_to_bwd_dropouts.size();

  if (changed) {
    VLOG(2) << "After DropoutWithReferenceFinder:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
