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
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/shortest_path_scheduler.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_dataflow_analysis.h"

namespace xla {
namespace poplarplugin {
namespace {

class SequenceCosts {
 public:
  // Compute shortest distance of instructions from first_inst.
  void FindCosts(HloInstruction* first_inst) {
    std::queue<HloInstruction*> q;
    absl::flat_hash_set<HloInstruction*> visited;

    visited.emplace(first_inst);
    q.push(first_inst);

    costs_[first_inst] = 0ULL;

    while (!q.empty()) {
      HloInstruction* inst = q.front();
      q.pop();

      auto process_instruction = [this, &visited, &q](
                                     HloInstruction* current_inst,
                                     HloInstruction* input_inst) -> void {
        if (visited.count(input_inst) == 0) {
          visited.insert(input_inst);
          q.push(input_inst);

          const uint64 cost = costs_[current_inst] + 1ULL;

          if (costs_.count(input_inst) != 0) {
            costs_[input_inst] = std::min(costs_[input_inst], cost);
          } else {
            costs_[input_inst] = cost;
          }
        }
      };

      for (auto* operand : inst->operands()) {
        process_instruction(inst, operand);
      }

      for (auto* pred : inst->control_predecessors()) {
        process_instruction(inst, pred);
      }
    }
  }

  // Populate costs_ which gives the shortest distance from one of computation
  // outputs to each instruction in the computation.
  void FindCosts(HloComputation* comp) {
    // Find the distance for each instruction to one of the computation outputs
    // (output has no users).
    FindCosts(comp->root_instruction());
    for (auto inst : comp->MakeInstructionPostOrder()) {
      if (inst->user_count() == 0 && inst != comp->root_instruction()) {
        FindCosts(inst);
      }
    }

    // Reverse the cost of parameters to make sure the ones closest to one of
    // the become live as late as possible.
    // Note that the map is sorted in descending orderer.
    std::map<uint64, std::vector<HloInstruction*>, std::greater<uint64>>
        parameters;
    for (HloInstruction* parameter : comp->parameter_instructions()) {
      parameters[costs_.at(parameter)].push_back(parameter);
    }

    uint64 next_cost = comp->instruction_count();
    // Iterate over the map in descending cost order.
    for (auto pair : parameters) {
      for (HloInstruction* parameter : pair.second) {
        costs_[parameter] = next_cost++;
      }
    }

    CHECK_EQ(costs_.size(), comp->instruction_count());
  }

  // Populate the final output.
  void InsertToSequence(HloInstruction* inst) {
    CHECK(!scheduled_.contains(inst));
    sequence_.push_back(inst);
    scheduled_.insert(inst);
  }

  void InsertIntoReady(HloInstruction* inst) {
    const uint64 cost = costs_.at(inst);
    ready_[cost].insert(inst);
  }

  void EraseFromReady(HloInstruction* inst) {
    const uint64 cost = costs_.at(inst);
    ready_[cost].erase(inst);
    if (ready_[cost].empty()) {
      ready_.erase(cost);
    }
  }

  // Instruction can be scheduled if it has not been scheduled,
  // if all its operands were scheduled and if its all predecessor
  // were scheduled.
  bool CanBeScheduled(const HloInstruction* inst) {
    auto is_scheduled = [this](const HloInstruction* a) {
      return scheduled_.contains(a);
    };

    if (is_scheduled(inst)) {
      return false;
    }

    // Can only schedule the instruction once all the operands and control
    // dependencies have executed.
    return absl::c_all_of(inst->operands(), is_scheduled) &&
           absl::c_all_of(inst->control_predecessors(), is_scheduled);
  }

  // Before running the main loop in the ComputeSchedule function, initialise
  // ready container with instructions which have no dependencies.
  void InitReadyQueue() {
    for (auto inst_cost_pair : costs_) {
      HloInstruction* inst = inst_cost_pair.first;
      if (inst->operand_count() == 0 &&
          inst->control_predecessors().size() == 0) {
        InsertIntoReady(inst);
      }
    }
  }

  // When we schedule instruction we use this function to check
  // if a user/successor is ready to be scheduled.
  void CheckIfNewReady(HloInstruction* inst) {
    for (auto user : inst->users()) {
      if (CanBeScheduled(user)) {
        InsertIntoReady(user);
      }
    }

    for (auto successor : inst->control_successors()) {
      if (CanBeScheduled(successor)) {
        InsertIntoReady(successor);
      }
    }
  }

  void ScheduleInstruction() {
    // Get the set with the lowest cost.
    auto& set = ready_.begin()->second;
    CHECK(!set.empty());

    // Get the first instruction in that set.
    HloInstruction* inst = *(set.begin());
    InsertToSequence(inst);
    EraseFromReady(inst);
    CheckIfNewReady(inst);
  }

  // Compute the schedule by starting from instructions which have no
  // predecessors and processing the instructions with lowest cost first.
  void ComputeSchedule() {
    InitReadyQueue();

    while (ready_.size()) {
      ScheduleInstruction();
    }

    CHECK_EQ(costs_.size(), sequence_.size());
  }

  HloInstructionSequence GetScheduleSequence() { return sequence_; }

 private:
  // Distance of instructions from root/outfeeds.
  HloInstructionMap<uint64> costs_;
  // In order schedule of instructions.
  HloInstructionSequence sequence_;
  // Instructions which have been scheduled.
  absl::flat_hash_set<const HloInstruction*> scheduled_;
  // Map of sets of instructions which are ready to be scheduled, priortising
  // the instructions with lowest cost.
  std::map<uint64, HloInstructionSet> ready_;
};

StatusOr<HloInstructionSequence> ScheduleInstructions(
    HloComputation* comp, const HloPoplarDataflowAnalysis& dataflow_analysis,
    const absl::flat_hash_map<const HloComputation*, int64_t>&
        memory_by_computation) {
  SequenceCosts seq_costs;
  // The algorithm "only" supports computations with 2^63 - 1 instructions as
  // half the range is reserved for parameters.
  CHECK_LT(comp->instruction_count(), (1LL << 63) - 1LL);
  seq_costs.FindCosts(comp);
  seq_costs.ComputeSchedule();
  return seq_costs.GetScheduleSequence();
}

StatusOr<HloInstructionSequence> ShortestPathScheduler(
    HloComputation* computation,
    const HloPoplarDataflowAnalysis& dataflow_analysis,
    const absl::flat_hash_map<const HloComputation*, int64_t>&
        memory_by_computation) {
  TF_ASSIGN_OR_RETURN(auto sched,
                      ScheduleInstructions(computation, dataflow_analysis,
                                           memory_by_computation));

  return sched;
}

}  // namespace

// Create a functor which performs the shortest path scheduling.
IpuSchedulerAlgorithm CreateShortestPathScheduler(
    const CompilerInformation& information) {
  return [=](HloComputation* computation,
             const HloPoplarDataflowAnalysis& dataflow_analysis,
             const absl::flat_hash_map<const HloComputation*, int64_t>&
                 memory_by_computation) {
    return ShortestPathScheduler(computation, dataflow_analysis,
                                 memory_by_computation);
  };
}

}  // namespace poplarplugin
}  // namespace xla
