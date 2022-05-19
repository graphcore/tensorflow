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

#include "tensorflow/compiler/plugin/poplar/driver/schedulers/sync_list_scheduler.h"

#include <map>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/heap_simulator.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace poplarplugin {
namespace {

class SyncListScheduler {
 public:
  // Construct and return a memory-minimizing sequence of HLO instructions
  // containing the given HLO computation.
  static StatusOr<HloInstructionSequence> Run(
      HloComputation* computation,
      const HloPoplarDataflowAnalysis& dataflow_analysis,
      const absl::flat_hash_map<const HloComputation*, int64_t>&
          memory_by_computation,
      int64_t max_syncs) {
    SyncListScheduler scheduler(computation, dataflow_analysis,
                                memory_by_computation, max_syncs);
    return scheduler.CreateSchedule();
  }

  // Returns whether the memory used by the given HLO should be ignored by the
  // scheduling heuristic.
  static bool IgnoreInstruction(const HloInstruction& instruction) {
    return instruction.opcode() == HloOpcode::kParameter ||
           instruction.opcode() == HloOpcode::kConstant;
  }

 private:
  // The scheduling priority of an instruction is first the number of bytes
  // waiting to sync, second the number of bytes freed by scheduling the
  // instruction, and finally (tie-breaker) by the number of users. This is
  // represented as a std::tuple containing these three values (first element is
  // the bytes waiting to sync). std::tuple provides the necessary comparison
  // operators.
  using Priority = std::tuple<int64_t, int64_t, int64_t>;

  SyncListScheduler(HloComputation* computation,
                    const HloPoplarDataflowAnalysis& dataflow_analysis,
                    const absl::flat_hash_map<const HloComputation*, int64_t>&
                        memory_by_computation,
                    int64_t max_syncs)
      : computation_(computation),
        dataflow_analysis_(dataflow_analysis),
        memory_by_computation_(memory_by_computation),
        max_syncs_(max_syncs) {
    // Create a map containing the buffer uses for each HLO
    // instruction. An HLO instruction "uses" a buffer if the
    // buffer is in an operand of the instruction as indicated by
    // the dataflow analysis.
    for (auto* instruction : computation->instructions()) {
      absl::flat_hash_set<const HloPoplarBuffer*> instr_uses;
      for (auto* operand : instruction->operands()) {
        dataflow_analysis.GetInstructionBufferSet(operand).ForEachElement(
            [&](const ShapeIndex& /*index*/,
                const HloPoplarBufferSet& buffer_set) {
              const auto& buffers = buffer_set.buffers();
              instr_uses.insert(buffers.begin(), buffers.end());
            });
      }
      buffer_uses_[instruction] = std::vector<const HloPoplarBuffer*>(
          instr_uses.begin(), instr_uses.end());
    }

    // Create map containing the number of unscheduled uses (hlo instructions)
    // of each logical buffer.
    for (auto* instruction : computation->instructions()) {
      dataflow_analysis.GetInstructionBufferSet(instruction)
          .ForEachElement([&](const ShapeIndex& /*index*/,
                              const HloPoplarBufferSet& buffer_set) {
            for (auto* buffer : buffer_set.buffers()) {
              if (buffer->DefinedBy(instruction)) {
                unscheduled_use_count_[buffer] = 0;
              }
            }
          });
    }

    for (auto* instruction : computation->instructions()) {
      for (auto* buffer : buffer_uses_.at(instruction)) {
        ++unscheduled_use_count_[buffer];
      }
    }

    // Buffers live out of the computation have an implicit use at the end of
    // the computation.
    const auto* root = computation->root_instruction();
    dataflow_analysis.GetInstructionBufferSet(root).ForEachElement(
        [&](const ShapeIndex& /*index*/, const HloPoplarBufferSet& buffer_set) {
          for (auto* live_out_buffer : buffer_set.buffers()) {
            ++unscheduled_use_count_[live_out_buffer];
          }
        });
  }

  // Returns whether the memory used by the given buffer should be ignored by
  // the scheduling heuristic.
  static bool IgnoreBuffer(const HloPoplarBuffer& buffer) {
    return IgnoreInstruction(*buffer.instruction());
  }

  // An entry in the worklist used by CreateSchedule.  Corresponds to one
  // HloInstruction, plus some cached metadata, saved for the purposes of making
  // BytesFreedIfScheduled fast.
  struct ReadyListEntry {
    HloInstruction* instruction;

    // The total size of all buffers defined by this instruction.
    int64_t bytes_defined;

    // For each buffer B used by this instruction, we keep a pair (B, U), where
    // U is the number of uses of B that have not yet been scheduled. This pair
    // is a pointer into the unscheduled_use_count_ map, so it gets updated for
    // free when we update counts in the map.
    std::vector<const std::pair<const HloPoplarBuffer* const, int64_t>*>
        used_buffer_unscheduled_use_counts;
  };

  // Creates a ReadyListEntry for the given instruction.
  ReadyListEntry MakeReadyListEntry(HloInstruction* instruction) {
    ReadyListEntry entry;
    entry.instruction = instruction;

    entry.bytes_defined = 0;
    dataflow_analysis_.GetInstructionBufferSet(instruction)
        .ForEachElement([&](const ShapeIndex& /*index*/,
                            const HloPoplarBufferSet& buffer_set) {
          for (auto* buffer : buffer_set.buffers()) {
            if (buffer->DefinedBy(instruction) && !IgnoreBuffer(*buffer)) {
              entry.bytes_defined += buffer->SizeInBytes();
            }
          }
        });

    for (auto* buffer : buffer_uses_.at(instruction)) {
      if (IgnoreBuffer(*buffer)) {
        continue;
      }
      auto unscheduled_use_count_it = unscheduled_use_count_.find(buffer);
      CHECK(unscheduled_use_count_it != unscheduled_use_count_.end());
      entry.used_buffer_unscheduled_use_counts.push_back(
          &*unscheduled_use_count_it);
    }
    return entry;
  }

  // Returns the number of bytes freed *after* the HLO instruction finishes.
  // The current List algorithm only considers two states for an instruction:
  // right before it runs, and after it finishes. We don't represent memory
  // usage during the execution of an instruction. But if the instruction calls
  // subcomputations, they are only live during the instruction's execution.
  // We end up counting the memory used by subcomputations as memory "defined"
  // by the instruction. This is not entirely accurate, but it is more accurate
  // than not taking subcomputations into account at all. In the future, we may
  // improve accounting for subcomputation memory (b/65409243).
  int64_t BytesFreedIfScheduled(const ReadyListEntry& entry) {
    auto instruction = entry.instruction;
    auto opcode = instruction->opcode();

    int64_t freed_bytes = 0;
    for (const auto& kv : entry.used_buffer_unscheduled_use_counts) {
      auto buffer = kv->first;
      auto use_count = kv->second;
      if (use_count == 1) {
        freed_bytes += buffer->SizeInBytes();
      }
    }
    // We only count the memory usage of the largest subcomputation, instead of
    // adding them all, because subcomputations won't execute in parallel.
    int64_t max_subcomputation_bytes = 0;
    for (const auto* c : instruction->called_computations()) {
      auto it = memory_by_computation_.find(c);
      if (it != memory_by_computation_.end()) {
        int64_t subcomputation_bytes = it->second;
        if (subcomputation_bytes > max_subcomputation_bytes) {
          max_subcomputation_bytes = subcomputation_bytes;
        }
      }
    }
    int64_t bytes_defined;
    if (max_subcomputation_bytes > 0 &&
        (opcode == HloOpcode::kWhile || opcode == HloOpcode::kCall ||
         opcode == HloOpcode::kConditional)) {
      // The output buffer of while/call/conditional is always aliased with the
      // output buffer of the root instruction in the body. Don't double count.
      bytes_defined = max_subcomputation_bytes;
    } else {
      bytes_defined = entry.bytes_defined + max_subcomputation_bytes;
    }
    return freed_bytes - bytes_defined;
  }

  int64_t BytesSyncedIfScheduled(const ReadyListEntry& entry) {
    auto instruction = entry.instruction;

    // We only consider all-reduce at the moment
    if (instruction->opcode() != HloOpcode::kAllReduce) {
      return 0;
    }

    return ShapeUtil::ByteSizeOf(instruction->shape());
  }

  // Constructs the scheduling priority of the given instruction.
  Priority GetPriority(const ReadyListEntry& entry) {
    return {-BytesSyncedIfScheduled(entry), BytesFreedIfScheduled(entry),
            entry.instruction->user_count()};
  }

  HloInstructionSequence CreateSchedule() {
    HloInstructionSequence schedule;

    // Populate the ready list with instructions which have no operands or
    // control predecessors.
    absl::flat_hash_map<const HloInstruction*, int64_t> unscheduled_pred_count;
    for (auto* instruction : computation_->instructions()) {
      // TODO(b/34466113): Replace this and above with successors() or
      // predecessors() when these methods are added to HloInstruction.
      for (HloInstruction* user : instruction->users()) {
        unscheduled_pred_count[user]++;
      }
      for (HloInstruction* succ : instruction->control_successors()) {
        unscheduled_pred_count[succ]++;
      }
    }

    // Use a multimap to sort ReadyListEntry according to their priority.
    std::multimap<Priority, ReadyListEntry> ready_queue;

    // Map of ready instructions to their iterators in ready_queue.
    absl::flat_hash_map<const HloInstruction*,
                        std::multimap<Priority, ReadyListEntry>::iterator>
        ready_instructions;

    int64_t waiting_syncs = 0;

    auto add_to_ready_queue = [&](HloInstruction* inst) {
      auto entry = MakeReadyListEntry(inst);
      auto priority = GetPriority(entry);
      waiting_syncs -= std::get<0>(priority);
      auto it = ready_queue.emplace(priority, std::move(entry));
      ready_instructions[inst] = it;
    };

    for (auto* instruction : computation_->instructions()) {
      if (instruction->operands().empty() &&
          instruction->control_predecessors().empty()) {
        add_to_ready_queue(instruction);
      }
    }

    while (!ready_queue.empty()) {
      // Remove the selected instruction from the ready list and add it to the
      // schedule.
      auto best_it = ready_queue.end();
      --best_it;
      HloInstruction* best = best_it->second.instruction;
      VLOG(2) << "Schedule instruction: " << best->ToShortString()
              << " Sync bytes: " << std::get<0>(best_it->first)
              << " Bytes freed: " << std::get<1>(best_it->first);
      ready_queue.erase(best_it);
      ready_instructions.erase(best);
      schedule.push_back(best);
      scheduled_instructions_.insert(best);

      bool adjust_ready_queue = false;
      // Update the unscheduled uses of the buffers.
      for (auto* buffer : buffer_uses_.at(best)) {
        int64_t& count = unscheduled_use_count_[buffer];
        CHECK_GT(count, 0);
        --count;
        if (count == 1) {
          adjust_ready_queue = true;
        }
      }

      // Add new instructions to ready list.
      auto update_pred_count = [&](HloInstruction* inst) {
        int64_t pred_count = --unscheduled_pred_count.at(inst);
        CHECK_GE(pred_count, 0);
        if (pred_count == 0) {
          add_to_ready_queue(inst);
        }
      };
      // TODO(b/34466113): Replace this and above with successors() or
      // predecessors() when these methods are added to HloInstruction.
      for (HloInstruction* user : best->users()) {
        update_pred_count(user);
      }
      for (HloInstruction* succ : best->control_successors()) {
        update_pred_count(succ);
      }
      // The unscheduled use count for a buffer has changed to 1, so the
      // priorities of some ready instructions may go up. We update them in the
      // ready queue, so that they can appear earlier.
      if (adjust_ready_queue) {
        for (HloInstruction* operand : best->operands()) {
          for (HloInstruction* operand_user : operand->users()) {
            auto ready_instructions_it = ready_instructions.find(operand_user);
            if (ready_instructions_it == ready_instructions.end()) {
              continue;
            }
            auto ready_queue_it = ready_instructions_it->second;
            auto& entry = ready_queue_it->second;
            Priority new_priority = GetPriority(entry);
            if (new_priority == ready_queue_it->first) {
              continue;
            }
            // Create a new entry in ready_queue, then update
            // ready_instructions[operand_user] to refer to the new entry.
            ready_instructions_it->second =
                ready_queue.emplace(new_priority, std::move(entry));
            // Remove the old entry in ready_queue.
            ready_queue.erase(ready_queue_it);
          }
        }
      }

      // If we are scheduling an all-reduce op, or we are over the sync limit,
      // we will schedule all of the all-reduce ops
      if (best->opcode() == HloOpcode::kAllReduce ||
          waiting_syncs > max_syncs_) {
        auto itr = ready_queue.begin();
        while (itr != ready_queue.end()) {
          auto instr = itr->second.instruction;

          // If it's an all-reduce op
          if (instr->opcode() == HloOpcode::kAllReduce) {
            // Add to the schedule
            schedule.push_back(instr);
            // Add to the set of scheduled instructions
            scheduled_instructions_.insert(instr);

            // Update priority of users
            for (HloInstruction* user : instr->users()) {
              update_pred_count(user);
            }

            // Update priority of control successors
            for (HloInstruction* succ : instr->control_successors()) {
              update_pred_count(succ);
            }

            // Remove the op from the read queue and set
            itr = ready_queue.erase(itr);
            ready_instructions.erase(instr);
          } else {
            ++itr;
          }
        }

        // We have scheduled all syncs
        waiting_syncs = 0;
      }
    }

    CHECK_EQ(schedule.size(), computation_->instruction_count());
    CHECK_EQ(scheduled_instructions_.size(), computation_->instruction_count());

    return schedule;
  }

  HloComputation* computation_;
  const HloPoplarDataflowAnalysis& dataflow_analysis_;
  // Computations are analyzed in post-order. When scheduling an instruction
  // that includes subcomputations, such as a while loop, we use this map to
  // look up the memory needed by subcomputations.
  const absl::flat_hash_map<const HloComputation*, int64_t>&
      memory_by_computation_;

  // A map containing the buffers that each instruction uses.
  absl::flat_hash_map<const HloInstruction*,
                      std::vector<const HloPoplarBuffer*>>
      buffer_uses_;

  // A map containing the count of unscheduled HLOs which using a particular
  // buffer.
  absl::flat_hash_map<const HloPoplarBuffer*, int64_t> unscheduled_use_count_;

  // Set of instructions which have been scheduled.
  absl::flat_hash_set<const HloInstruction*> scheduled_instructions_;

  int64_t max_syncs_;
};

// List scheduler
StatusOr<HloInstructionSequence> SyncListMemoryScheduler(
    HloComputation* computation,
    const HloPoplarDataflowAnalysis& dataflow_analysis,
    const absl::flat_hash_map<const HloComputation*, int64_t>&
        memory_by_computation,
    int64_t max_syncs) {
  return SyncListScheduler::Run(computation, dataflow_analysis,
                                memory_by_computation, max_syncs);
}
}  // namespace

IpuSchedulerAlgorithm CreateSyncListMemoryScheduler(int64_t max_syncs) {
  return [=](HloComputation* computation,
             const HloPoplarDataflowAnalysis& dataflow_analysis,
             const absl::flat_hash_map<const HloComputation*, int64_t>&
                 memory_by_computation) {
    return SyncListMemoryScheduler(computation, dataflow_analysis,
                                   memory_by_computation, max_syncs);
  };
}

}  // namespace poplarplugin
}  // namespace xla
