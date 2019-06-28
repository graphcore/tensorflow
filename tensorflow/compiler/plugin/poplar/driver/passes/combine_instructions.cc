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

#include "tensorflow/compiler/plugin/poplar/driver/passes/combine_instructions.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/instruction_colocator_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/core/lib/core/errors.h"

#include "absl/types/optional.h"

#include <algorithm>

namespace xla {
namespace poplarplugin {

namespace {

// Partition the ops into regions where they are independent, can colocate,
// have the same type and same inplaceness.
template <typename Iter>
std::vector<absl::flat_hash_set<HloInstruction*>> Partition(
    Iter begin, Iter end, HloReachabilityMap* reachability_map) {
  std::vector<absl::flat_hash_set<HloInstruction*>> result;
  while (begin != end) {
    auto first = *begin;
    auto pred = [&](const HloInstruction* inst) {
      return CanColocate(first, inst) &&
             (first->shape().element_type() == inst->shape().element_type()) &&
             (IsUsedInplace(first) == IsUsedInplace(inst));
    };

    auto itr = std::stable_partition(begin, end, pred);
    // The vector of instructions in [begin, itr) might not be independent.
    // We now greedily cluster it into independent clusters.
    std::vector<absl::flat_hash_set<HloInstruction*>> clusters;
    while (begin != itr) {
      bool found_cluster = false;
      // Go through all existing clusters and check if the instruction is
      // independent of all the instructions in the cluster. If it is, add it to
      // the cluster, otherwise create a new cluster.
      for (auto& cluster : clusters) {
        bool can_insert =
            absl::c_all_of(cluster, [&](const HloInstruction* inst) {
              return !reachability_map->IsConnected(inst, *begin);
            });
        if (can_insert) {
          cluster.insert(*begin);
          found_cluster = true;
          break;
        }
      }
      if (!found_cluster) {
        clusters.push_back({*begin});
      }
      begin = std::next(begin);
    }
    result.insert(std::end(result), std::begin(clusters), std::end(clusters));
  }

  return result;
}

template <typename Iter>
HloInstruction* Combine(HloComputation* comp, Iter begin, Iter end) {
  auto dist = std::distance(begin, end);
  // We only have a single instruction, so nothing to combine
  if (dist == 1) {
    return *begin;
  }

  std::vector<Shape> shapes(dist);
  std::transform(begin, end, shapes.begin(),
                 [](HloInstruction* inst) { return inst->shape(); });

  // The new output shape
  auto shape = ShapeUtil::MakeTupleShape(shapes);

  // The new list of operands
  auto operands = std::accumulate(
      begin, end, std::vector<HloInstruction*>{},
      [](std::vector<HloInstruction*>& accum, HloInstruction* inst) {
        accum.insert(accum.end(), inst->operands().begin(),
                     inst->operands().end());

        return accum;
      });

  // Add the new instruction.
  auto new_inst =
      comp->AddInstruction((*begin)->CloneWithNewOperands(shape, operands));
  // Copy the sharding information if there was any.
  if ((*begin)->has_sharding()) {
    new_inst->set_sharding((*begin)->sharding());
  }
  return new_inst;
}

template <typename Iter>
StatusOr<std::vector<HloInstruction*>> Replace(HloComputation* comp, Iter begin,
                                               Iter end,
                                               HloInstruction* combined) {
  auto dist = std::distance(begin, end);

  if (dist == 1) {
    return std::vector<HloInstruction*>{};
  }

  std::vector<HloInstruction*> result;
  result.reserve(dist);

  for (auto itr = begin; itr != end; ++itr) {
    HloInstruction* inst = *itr;
    // Add a get tuple to unpack the combined result
    auto gte = comp->AddInstruction(HloInstruction::CreateGetTupleElement(
        inst->shape(), combined, std::distance(begin, itr)));
    MakeUsedInplace(gte);
    result.push_back(gte);
    TF_RETURN_IF_ERROR(combined->CopyAllControlDepsFrom(inst));
    TF_RETURN_IF_ERROR(inst->DropAllControlDeps());

    // Replace the op
    TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(gte));
    TF_RETURN_IF_ERROR(comp->ForceRemoveInstruction(inst));
  }

  return result;
}
}  // namespace

StatusOr<absl::optional<HloInstructionSequence>>
CombineInstructions::CombineInstructionsInComputation(
    HloComputation* comp, const HloInstructionSequence& sequence) {
  auto reachability_map = HloReachabilityMap::Build(comp);
  bool changed = false;
  auto instructions = sequence.instructions();

  std::vector<const HloInstruction*> result;
  result.reserve(instructions.size());

  // Find the first region of consecutive instructions with colocators to merge
  // together. First find the first instruction with a colocator.
  const auto has_colocator = [](HloInstruction* inst) {
    return GetInstructionColocatorHelper(inst).has_value();
  };
  //       v beg
  // [a,b,c|r,r,r,r,r|d,e,f,g,r,r,r]
  auto region_begin =
      std::find_if(instructions.begin(), instructions.end(), has_colocator);
  // Then find the next instruction which can't be colocated with the begining.
  const auto can_not_colocate = [&](HloInstruction* inst) {
    return !CanColocate(*region_begin, inst);
  };
  //       v beg     v end
  // [a,b,c|r,r,r,r,r|d,e,f,g,r,r,r]
  auto region_end =
      std::find_if(region_begin, instructions.end(), can_not_colocate);

  // While we have a region to process
  while (region_begin != instructions.end()) {
    // Partition the instructions into combinable groups
    auto subregions =
        Partition(region_begin, region_end, reachability_map.get());

    std::vector<HloInstruction*> replacements;

    // Create the combined instructions
    for (auto& subregion : subregions) {
      replacements.push_back(
          Combine(comp, std::begin(subregion), std::end(subregion)));

      TF_ASSIGN_OR_RETURN(
          auto ops, Replace(comp, std::begin(subregion), std::end(subregion),
                            replacements.back()));
      changed |= ops.size();
      replacements.insert(replacements.end(), ops.begin(), ops.end());
    }

    // Replace the previous instruction in the schedule
    //       v beg     v end
    // [a,b,c|r,r,r,r,r|d,e,f,g,r,r,r]
    // becomes
    //       v itr
    // [a,b,c|d,e,f,g,r,r,r]
    auto insert_itr = instructions.erase(region_begin, region_end);
    //       v itr   v end
    // [a,b,c|r,t,t,t|d,e,f,g,r,r,r]
    region_end = instructions.insert(insert_itr, replacements.begin(),
                                     replacements.end()) +
                 replacements.size();

    // Find the next region of consecutive colocated instructions
    //               v end   v beg
    // [a,b,c,r,t,t,t|d,e,f,g|r,r,r]
    region_begin = std::find_if(region_end, instructions.end(), has_colocator);
    //                       v beg v end
    // [a,b,c,r,t,t,t,d,e,f,g|r,r,r|]
    region_end =
        std::find_if(region_begin, instructions.end(), can_not_colocate);
  }

  // Returns a new sequence if any instructions were combined.
  return changed ? absl::optional<HloInstructionSequence>(
                       HloInstructionSequence(instructions))
                 : absl::nullopt;
}

StatusOr<bool> CombineInstructions::Run(HloModule* module) {
  if (!module->has_schedule()) {
    return tensorflow::errors::FailedPrecondition(
        "CombineInstructions: module doesn't have a schedule");
  }
  bool changed = false;
  const auto& schedule = module->schedule();
  const auto& sequences = schedule.sequences();

  absl::flat_hash_map<int64, HloComputation*> computations;
  for (int i = 0; i < module->computation_count(); ++i) {
    auto comp = module->mutable_computation(i);
    computations[comp->unique_id()] = comp;
  }

  HloSchedule new_schedule(module);
  for (auto& pair : sequences) {
    auto comp = computations[pair.first];
    TF_ASSIGN_OR_RETURN(auto new_seq,
                        CombineInstructionsInComputation(comp, pair.second));
    if (new_seq) {
      new_schedule.set_sequence(comp, *new_seq);
      changed = true;
    } else {
      new_schedule.set_sequence(comp, pair.second);
    }
  }

  TF_RETURN_IF_ERROR(new_schedule.Verify());
  module->set_schedule(new_schedule);

  VLOG(1) << "Combined all reduce schedule " << new_schedule.ToString();

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
