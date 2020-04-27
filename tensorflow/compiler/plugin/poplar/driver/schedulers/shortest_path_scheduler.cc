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

#include <functional>
#include <map>
#include <queue>
#include <set>
#include <utility>
#include <vector>

namespace xla {
namespace poplarplugin {
namespace {

class SequenceCosts {
 public:
  SequenceCosts() {}

  // Compute shortest distance of instructions from root.
  void LabelInstructionsBFS(HloInstruction* first_inst) {
    std::queue<HloInstruction*> q;
    absl::flat_hash_set<HloInstruction*> visited;

    visited.emplace(first_inst);
    q.push(first_inst);
    int64 cost = 0;
    costs_[first_inst] = cost;

    while (!q.empty()) {
      HloInstruction* inst = q.front();
      q.pop();

      for (auto o : inst->operands()) {
        if (visited.count(o) == 0) {
          visited.emplace(o);
          q.push(o);
          if (costs_.count(o) != 0) {
            costs_[o] = std::min(costs_[o], costs_[inst] + 1);
          } else {
            costs_[o] = costs_[inst] + 1;
          }
        }
      }
    }
  }

  // Costs from the outfeed.
  void FindCosts(HloInstruction* root_inst) { LabelInstructionsBFS(root_inst); }

  // Populate costs_ which gives cost, the shortest distance from outfeeds, for
  // each instruction.
  void FindCosts(HloComputation* comp) {
    FindCosts(comp->root_instruction());
    for (auto inst : comp->instructions()) {
      if (inst->user_count() == 0 && inst != comp->root_instruction()) {
        FindCosts(inst);
      }
    }
  }

  // Populate the final output.
  void InsertToSequence(const HloInstruction* inst) {
    HloInstruction* inst_nonconst = const_cast<HloInstruction*>(inst);
    sequence_.push_back(inst_nonconst);
  }

  // Insert instruction into ready and not ready containers.
  template <typename T>
  void InsertInto(const HloInstruction* inst, T& my_map) {
    int64 cost = costs_.at(inst);
    if (my_map.count(cost) > 0) {
      my_map.at(cost).insert(inst);
    } else {
      HloInstructionSet un_set;
      un_set.insert(inst);
      my_map.insert(std::make_pair(cost, std::move(un_set)));
    }
  }

  void InsertIntoReady(const HloInstruction* inst) { InsertInto(inst, ready_); }
  void InsertIntoNotReady(const HloInstruction* inst) {
    InsertInto(inst, not_ready_);
  }

  // Erase instruction from ready and not ready containers.
  template <typename T>
  void EraseFrom(const HloInstruction* inst, T& my_map) {
    int64 cost = costs_.at(inst);
    if (my_map.count(cost) > 0) {
      if (my_map.at(cost).count(inst) > 0) {
        my_map.at(cost).erase(inst);
        if (my_map.at(cost).size() == 0) {
          my_map.erase(cost);
        }
      }
    }
  }

  void EraseFromReady(const HloInstruction* inst) { EraseFrom(inst, ready_); }
  void EraseFromNotReady(const HloInstruction* inst) {
    EraseFrom(inst, not_ready_);
  }

  // Move instruction from not ready to ready container.
  void FromNotReadyToReady(const HloInstruction* inst) {
    InsertIntoReady(inst);
    EraseFromNotReady(inst);
  }

  // Instruction can be scheduled if it has not been scheduled,
  // if all its operands were scheduled and if its all predecessor
  // were scheduled.
  bool CanBeScheduled(const HloInstruction* inst) {
    if (scheduled_.contains(inst)) {
      return false;
    }

    for (auto operand : inst->operands()) {
      if (!scheduled_.contains(operand)) {
        return false;
      }
    }

    for (auto predecessor : inst->control_predecessors()) {
      if (!scheduled_.contains(predecessor)) {
        return false;
      }
    }

    return true;
  }

  // Before running the main loop in the main compute function we initialise
  // ready and not ready containers.
  // If instruction has no operands and is not parameter it will be on ready
  // list. if it has operands it will be on not ready list. Parameters are not
  // on those lists. It is sufficient to truck them with scheduled_.
  void InitReadyNotReady() {
    for (auto inst_cost_pair : costs_) {
      const HloInstruction* inst = inst_cost_pair.first;
      if (inst->opcode() != HloOpcode::kParameter) {
        if (inst->operand_count() == 0 &&
            inst->control_predecessors().size() == 0) {
          InsertIntoReady(inst);
        } else {
          InsertIntoNotReady(inst);
        }
      } else if (inst->user_count() == 0 &&
                 inst->control_successors().empty()) {
        // unreachable/unused kParameter
        unused_parameters_.push_back(inst);
      }
    }
  }

  // When we schedule instruction we use this function to check
  // if a user/successor become ready to be schedule.
  void CheckIfNewReady(const HloInstruction* inst) {
    for (auto user : inst->users()) {
      if (CanBeScheduled(user)) {
        FromNotReadyToReady(user);
      }
    }

    for (auto successor : inst->control_successors()) {
      if (CanBeScheduled(successor)) {
        FromNotReadyToReady(successor);
      }
    }
  }

  // This function schedules the first (best) instruction from ready container.
  void ScheduleReady() {
    auto un_set = ready_.begin()->second;
    CHECK(!un_set.empty());

    const HloInstruction* inst = *(un_set.begin());
    scheduled_.insert(inst);
    InsertToSequence(inst);
    EraseFromReady(inst);
    CheckIfNewReady(inst);
  }

  // Check if all parameters of instruction were scheduled.
  bool AreAllParamScheduled(const HloInstruction* inst) {
    for (auto operand : inst->operands()) {
      if (operand->opcode() == HloOpcode::kParameter &&
          !scheduled_.contains(operand)) {
        return false;
      }
    }

    return true;
  }

  // Will schedule all parameters, which has not been scheduled yet, of the
  // instruction. And it populates list list_scheduled_param with them.
  bool ScheduleParam(const HloInstruction* inst,
                     std::vector<const HloInstruction*>& list_scheduled_param) {
    for (auto operand : inst->operands()) {
      if (operand->opcode() == HloOpcode::kParameter &&
          !scheduled_.contains(operand)) {
        bool ok = absl::c_all_of(
            operand->control_predecessors(),
            [&](HloInstruction* dep) { return scheduled_.contains(dep); });

        if (ok) {
          scheduled_.insert(operand);
          InsertToSequence(operand);
          list_scheduled_param.push_back(operand);
          return true;
        }
      }
    }
    return false;
  }

  // When we schedule parameters of first (best) instruction from not ready list
  // we check if users of parameters become ready to be scheduled.
  void CheckIfNewReadyAfterParamOfHighestNotReady(
      const std::vector<const HloInstruction*>& list_scheduled_param) {
    for (auto param : list_scheduled_param) {
      CheckIfNewReady(param);
    }
  }

  // Schedule parameters of first (best) instruction from not ready list,
  // which has unscheduled parameters.
  // This function could be improved by hashing. Instead of looping. (possible
  // redundancy: it iterates till find action)
  void ScheduleParamOfHighestNotReady() {
    std::vector<const HloInstruction*> list_scheduled_param;
    bool new_param_scheduled = false;
    for (auto pair_not_ready : not_ready_) {
      if (new_param_scheduled) {
        break;
      }

      for (auto inst : pair_not_ready.second) {
        if (!AreAllParamScheduled(inst)) {
          if (ScheduleParam(inst, list_scheduled_param)) {
            new_param_scheduled = true;
            break;
          }
        }
      }
    }

    if (new_param_scheduled) {
      CheckIfNewReadyAfterParamOfHighestNotReady(list_scheduled_param);
    }
  }

  // Main function to provide scheduled sequence.
  // We follow 2 main rules.
  // A: Schedule instruction from ready, list of instruction which can be
  // scheduled, with the lowest cost (distance from root/outfeeds). B: If
  // nothing in A do B. Schedule all parameters of instruction from not ready
  // list, instruction which can not be schedule yet, with highest cost.
  void ComputeSchedule() {
    InitReadyNotReady();

    while (true) {
      if (ready_.empty() && not_ready_.empty()) {
        break;
      }

      if (!ready_.empty()) {
        ScheduleReady();
      } else {
        ScheduleParamOfHighestNotReady();
      }
    }
    for (auto inst : unused_parameters_) {
      InsertToSequence(inst);
    }
  }

  HloInstructionSequence GetScheduleSequence() { return sequence_; }

 private:
  // costs_ - distance of instructions from root/outfeeds.
  // sequence_, scheduled_ - result scheduled sequence. Set is for performance.
  // ready_ - list of instructions which can be scheduled.
  // not_ready_- list of instructions which can not be schedule yet.
  // Parameters are not on ready_, not_ready_. Having scheduled_ is sufficient
  // for them.
  using HloInstructionSet = std::set<const HloInstruction*, HloPtrComparator>;
  absl::flat_hash_map<const HloInstruction*, int64> costs_;
  HloInstructionSequence sequence_;
  absl::flat_hash_set<const HloInstruction*> scheduled_;
  std::map<int64, HloInstructionSet> ready_;
  std::map<int64, HloInstructionSet, std::greater<int64> > not_ready_;
  std::vector<const HloInstruction*> unused_parameters_;
};

StatusOr<HloInstructionSequence> ScheduleInstructions(
    HloComputation* comp, const TuplePointsToAnalysis& points_to_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const absl::flat_hash_map<const HloComputation*, int64>&
        memory_by_computation) {
  SequenceCosts seq_costs;
  seq_costs.FindCosts(comp);
  seq_costs.ComputeSchedule();
  HloInstructionSequence sequence = seq_costs.GetScheduleSequence();

  if (sequence.size() == 0) {
    // Computations of only parameters will not be scheduled
    sequence = HloInstructionSequence(comp->MakeInstructionPostOrder());
  }

  return sequence;
}

StatusOr<HloInstructionSequence> ShortestPathScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const absl::flat_hash_map<const HloComputation*, int64>&
        memory_by_computation) {
  TF_ASSIGN_OR_RETURN(
      auto sched, ScheduleInstructions(computation, points_to_analysis,
                                       size_function, memory_by_computation));

  return sched;
}

}  // namespace

// Create a functor which performs the shortest path scheduling.
IpuSchedulerAlgorithm CreateShortestPathScheduler(
    const CompilerInformation& information) {
  return [=](HloComputation* computation,
             const TuplePointsToAnalysis& points_to_analysis,
             const LogicalBuffer::SizeFunction& size_function,
             const absl::flat_hash_map<const HloComputation*, int64>&
                 memory_by_computation) {
    return ShortestPathScheduler(computation, points_to_analysis, size_function,
                                 memory_by_computation);
  };
}

}  // namespace poplarplugin
}  // namespace xla
