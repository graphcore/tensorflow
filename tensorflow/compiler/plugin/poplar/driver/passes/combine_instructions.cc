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
std::vector<std::vector<HloInstruction*>> Partition(
    Iter begin, Iter end, HloReachabilityMap* reachability_map) {
  std::vector<std::vector<HloInstruction*>> result;
  while (begin != end) {
    auto first = *begin;
    auto pred = [&](const HloInstruction* inst) {
      return CanColocate(first, inst) &&
             (first->shape().element_type() == inst->shape().element_type()) &&
             (IsLoweredInplace(first) == IsLoweredInplace(inst));
    };

    auto itr = std::stable_partition(begin, end, pred);
    // The vector of instructions in [begin, itr) might not be independent.
    // We now greedily cluster it into independent clusters.
    std::vector<std::vector<HloInstruction*>> clusters;
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
          cluster.push_back(*begin);
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
  // Then find the next instruction which can't be colocated with the beginning.
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
      const auto num_insts =
          std::distance(std::begin(subregion), std::end(subregion));
      if (num_insts > 1) {
        auto colocator = GetInstructionColocatorHelper(*std::begin(subregion));
        CHECK(colocator);
        TF_ASSIGN_OR_RETURN(
            auto ops, (*colocator)
                          ->CombineAndReplaceColocatedInstructions(
                              {std::begin(subregion), std::end(subregion)}));
        replacements.insert(replacements.end(), ops.begin(), ops.end());
        changed = true;
      } else {
        replacements.push_back(*std::begin(subregion));
      }
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

    // If a region of one instruction is identified, that instruction has a
    // colocator, but cannot colocate or cannot colocate with itself, then
    // there will be an infinite loop, as neither the beginning or end of
    // the region will move, because no replacements have been made.
    // Therefore, extend the next search region to move the region along.
    if (replacements.empty()) {
      region_end++;
    }

    CHECK(region_begin != region_end);

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

  VLOG(1) << "Combined schedule " << new_schedule.ToString();

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
