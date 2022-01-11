/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_dataflow_analysis.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {
enum class InplacePriority {
  kHigh = 0,
  kMedium,
  kLow,
};

using InplaceCandidates =
    std::map<InplacePriority, std::vector<HloInstruction*>>;
}  // namespace

static bool IsInplaceCandidate(HloInstruction* instruction,
                               const HloPoplarDataflowAnalysis& dataflow) {
  // if any of the outputs could be inplaced consider instruction as a candidate
  const auto& instruction_set = dataflow.GetInstructionBufferSet(instruction);
  const auto& buffer_sets = instruction_set.GetBufferSets();
  bool result = false;
  buffer_sets.ForEachElement(
      [&](const ShapeIndex& index, const HloPoplarBufferSet& data) {
        result |= (data.GetUseKind() > BufferUseKind::USE_NO_ALIAS);
      });
  return result;
}

static std::vector<HloInstruction*> FindAllCandidates(
    HloComputation* computation, const HloPoplarDataflowAnalysis& dataflow) {
  std::vector<HloInstruction*> result;
  result.reserve(computation->instruction_count());
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    if (IsInplaceCandidate(instruction, dataflow)) {
      result.emplace_back(instruction);
    }
  }
  return result;
}

static std::map<HloInstructionType, InplaceCandidates> FindInplaceCandidates(
    HloComputation* comp, const HloPoplarDataflowAnalysis& dataflow) {
  std::vector<HloInstruction*> candidates = FindAllCandidates(comp, dataflow);

  std::map<HloInstructionType, InplaceCandidates> inplace_candidates;
  absl::flat_hash_map<const HloInstruction*, InplacePriority> current_priority;

  auto UpdateQueue = [&](HloInstruction* inst, InplaceCandidates& candidates,
                         InplacePriority new_priority) {
    auto it = current_priority.find(inst);
    if (it != current_priority.end()) {
      InplacePriority& old_priority = it->second;
      if (old_priority == new_priority) {
        return;
      }
      auto& queue = candidates[old_priority];
      queue.erase(std::remove(queue.begin(), queue.end(), inst), queue.end());
      old_priority = new_priority;
    } else {
      current_priority.emplace(inst, new_priority);
    }
    candidates[new_priority].push_back(inst);
  };

  auto AddToQueue = [&](HloInstruction* inst, InplacePriority priority) {
    auto inst_description = GetInplaceDescription(inst);
    UpdateQueue(inst, inplace_candidates[inst_description.GetType()], priority);
  };

  // For each route in map mark inplace ops as high priority inplace
  // candidates.
  for (auto& inst : candidates) {
    switch (inst->opcode()) {
      case HloOpcode::kAdd:
      case HloOpcode::kFusion:
      case HloOpcode::kDynamicUpdateSlice:
      case HloOpcode::kMultiply:
      case HloOpcode::kSubtract:
      case HloOpcode::kAddDependency:
      case HloOpcode::kGetTupleElement: {
        AddToQueue(inst, InplacePriority::kHigh);
        break;
      }
      default:
        break;
    }
  }

  // Get all possible remaining inplace instructions.
  // Give medium priority to outlined poplibs calls.
  for (auto* inst : comp->MakeInstructionPostOrder()) {
    if (inst->parent()->root_instruction() == inst) {
      AddToQueue(inst, InplacePriority::kHigh);
    } else {
      switch (inst->opcode()) {
        case HloOpcode::kCustomCall:
        case HloOpcode::kFusion: {
          AddToQueue(inst, InplacePriority::kMedium);
          break;
        }
        default: {
          AddToQueue(inst, InplacePriority::kLow);
          break;
        }
      }
    }
  }
  return inplace_candidates;
}

StatusOr<bool> InplaceFinder::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto dataflow,
                      HloPoplarDataflowAnalysis::Run(module, annotations));
  bool changed = false;
  for (auto* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    InplaceWorkList worklist;
    // The reachability map is used for adding and finding control dependencies
    // in order to allow for inplace ops to be executed after other instructions
    // which are using the inplace input.
    std::unique_ptr<HloReachabilityMap> reachability_map =
        HloReachabilityMap::Build(comp);
    // Because we are using a map, we first inplace GTEs, then Read/Write and
    // then Read-Only.
    const auto inplace_candidates = FindInplaceCandidates(comp, *dataflow);
    for (auto type_candidates_pair : inplace_candidates) {
      auto& inplace_candidates_queues = type_candidates_pair.second;
      // Because we are using a map, all the candidate queues are sorted from
      // High to Low priority.
      for (auto inplace_priority_candidates_pair : inplace_candidates_queues) {
        auto& inplace_instruction_candidates =
            inplace_priority_candidates_pair.second;
        for (auto* inst : inplace_instruction_candidates) {
          if (HloPoplarInplaceDescription::ConvertToInplace(
                  inst, reachability_map.get(), worklist)) {
            changed = true;
            VLOG(1) << "Inplacing " << inst->ToString();
          }
        }
      }
    }
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
