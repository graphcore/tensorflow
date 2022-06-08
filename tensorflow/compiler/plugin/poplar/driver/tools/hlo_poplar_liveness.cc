/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_liveness.h"

#include <algorithm>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
namespace poplarplugin {
namespace {
void CombineBufferSets(const HloPoplarBufferSet& source,
                       HloPoplarBufferSet& destination) {
  for (auto* buffer : source.buffers()) {
    destination.AddBuffer(buffer);
  }
}

void CombineBufferSets(const InstructionPoplarBufferSet& instruction_buffer_set,
                       HloPoplarBufferSet& buffers) {
  for (auto& item : instruction_buffer_set.GetBufferSets().leaves()) {
    auto& buffer_set = item.second;
    CombineBufferSets(buffer_set, buffers);
  }
}

HloPoplarBufferSet GetInputBufferSets(
    const HloInstruction* inst,
    const InstructionBufferSets& instruction_buffer_sets) {
  HloPoplarBufferSet buffers;

  for (auto* operand : inst->operands()) {
    auto it = instruction_buffer_sets.find(operand);
    if (it != instruction_buffer_sets.end()) {
      CombineBufferSets(it->second, buffers);
    }
  }

  return buffers;
}

HloPoplarBufferSet GetOutputBufferSets(
    const HloInstruction* inst,
    const InstructionBufferSets& instruction_buffer_sets) {
  HloPoplarBufferSet buffers;

  auto it = instruction_buffer_sets.find(inst);
  if (it != instruction_buffer_sets.end()) {
    CombineBufferSets(it->second, buffers);
  }

  return buffers;
}

HloPoplarBufferSet FindBuffersUsedBy(
    const HloInstruction* inst,
    const InstructionBufferSets& instruction_buffer_sets) {
  const auto input_buffers = GetInputBufferSets(inst, instruction_buffer_sets);
  auto output_buffers = GetOutputBufferSets(inst, instruction_buffer_sets);
  CombineBufferSets(input_buffers, output_buffers);

  return output_buffers;
}
}  // namespace

HloInstructionMap<HloPoplarBufferSet> FindUsedBuffers(
    const HloModule* module,
    const HloPoplarDataflowAnalysis& dataflow_analysis) {
  return FindUsedBuffers(module, dataflow_analysis.GetInstructionBufferSets());
}

HloInstructionMap<HloPoplarBufferSet> FindUsedBuffers(
    const HloModule* module,
    const InstructionBufferSets& instruction_buffer_sets) {
  HloInstructionMap<HloPoplarBufferSet> used_buffers;

  for (auto* comp : module->computations()) {
    for (auto* inst : comp->instructions()) {
      used_buffers[inst] = FindBuffersUsedBy(inst, instruction_buffer_sets);
    }
  }

  return used_buffers;
}

namespace {
void MarkBufferAsAlive(HloPoplarBuffer::Id buffer,
                       HloPoplarBufferIdSet& live_set) {
  live_set.insert(buffer);
}

void MarkBufferAsDead(HloPoplarBuffer::Id buffer,
                      HloPoplarBufferIdSet& live_set) {
  live_set.erase(buffer);
}

bool IsFinalResult(const HloInstruction* inst) {
  auto* module = inst->GetModule();
  auto* entry = module->entry_computation();
  return inst == entry->root_instruction();
}

void UpdateLivenessBackwards(
    const HloInstruction* inst, const HloPoplarBuffer* buffer,
    const absl::flat_hash_map<const HloPoplarBuffer*, const HloInstruction*>
        buffer_creators,
    HloPoplarBufferIdSet& live_set) {
  // Since we're going backwards we kill a buffer when it's defined. The only
  // exception being if it's the final result of the module - it's assumed
  // it'll be read after.
  if (buffer_creators.at(buffer) == inst && !IsFinalResult(inst)) {
    MarkBufferAsDead(buffer->id(), live_set);
  } else {
    MarkBufferAsAlive(buffer->id(), live_set);
  }
}

const HloPoplarBufferSet& GetBufferUsage(
    const HloInstructionMap<HloPoplarBufferSet>& buffer_usages,
    HloInstruction* key) {
  static HloPoplarBufferSet empty_set;
  return FindOrDefault(buffer_usages, key, empty_set);
}

// Return the instructions that define the buffers used in the given
// schedule. We treat buffers returned from a subcomputation as being
// defined by the caller. Otherwise we wouldn't be able to determine
// when those buffers were born, so they'd be seen as being always
// live for the duration of the schedule.
absl::flat_hash_map<const HloPoplarBuffer*, const HloInstruction*>
FindBufferCreators(const std::vector<HloInstruction*>& flat_schedule,
                   const HloInstructionMap<HloPoplarBufferSet>& buffer_usages) {
  absl::flat_hash_map<const HloPoplarBuffer*, const HloInstruction*>
      buffer_creators;

  HloPoplarBufferIdSet visited;

  for (auto* inst : flat_schedule) {
    const auto& buffer_set = GetBufferUsage(buffer_usages, inst);
    for (auto& buffer : buffer_set.buffers()) {
      if (!visited.contains(buffer->id())) {
        if (buffer->DefinedBy(inst)) {
          buffer_creators[buffer] = inst;
        } else if (inst->opcode() == HloOpcode::kParameter) {
          buffer_creators[buffer] = buffer->instruction();
        } else {
          // A buffer which isn't from a parameter and hasn't already been
          // defined must come from a subcomputation..
          CHECK(!inst->called_computations().empty());
          buffer_creators[buffer] = inst;
        }
      }

      visited.insert(buffer->id());
    }
  }

  return buffer_creators;
}

HloPoplarBufferIdSet FindInplaceParameters(
    const std::vector<HloInstruction*>& flat_schedule,
    const HloInstructionMap<HloPoplarBufferSet>& buffer_usages) {
  HloPoplarBufferIdSet reused_buffers;

  for (auto* inst : flat_schedule) {
    if (inst->opcode() == HloOpcode::kParameter) {
      const auto& buffer_set = GetBufferUsage(buffer_usages, inst);
      for (auto* buffer : buffer_set.buffers()) {
        // A parameter is inplace if it hasn't created its own buffer.
        auto inplace = !buffer->DefinedBy(inst);
        if (inplace) {
          reused_buffers.insert(buffer->id());
        }
      }
    }
  }

  return reused_buffers;
}
}  // namespace

HloInstructionMap<HloPoplarBufferIdSet> GenerateProgramLiveness(
    const std::vector<HloInstruction*>& flat_schedule,
    const HloInstructionMap<HloPoplarBufferSet>& buffer_usages) {
  HloInstructionMap<HloPoplarBufferIdSet> program_liveness;

  const auto buffer_creators = FindBufferCreators(flat_schedule, buffer_usages);

  HloPoplarBufferIdSet live_set =
      FindInplaceParameters(flat_schedule, buffer_usages);

  for (auto it = flat_schedule.rbegin(); it != flat_schedule.rend(); ++it) {
    auto* inst = *it;

    const auto& buffer_set = GetBufferUsage(buffer_usages, inst);
    for (auto& buffer : buffer_set.buffers()) {
      UpdateLivenessBackwards(inst, buffer, buffer_creators, live_set);
    }
    program_liveness[inst] = live_set;
  }

  return program_liveness;
}

namespace {
int64_t MemoryUsageOfBufferSet(
    const HloPoplarBufferIdSet& buffers,
    const absl::flat_hash_map<HloPoplarBuffer::Id, int64_t>&
        buffer_sizes_in_bytes) {
  const int64_t memory_usage = absl::c_accumulate(
      buffers, 0l,
      [&buffer_sizes_in_bytes](int64_t sum, HloPoplarBuffer::Id buffer_id) {
        return sum + buffer_sizes_in_bytes.at(buffer_id);
      });
  return memory_usage;
}

int64_t MemoryUsageOfComputations(
    const std::vector<HloComputation*> computations,
    const absl::flat_hash_map<const HloComputation*, int64_t>&
        computation_costs_in_bytes) {
  const int64_t memory_usage = absl::c_accumulate(
      computations, 0l,
      [&computation_costs_in_bytes](int64_t sum, HloComputation* comp) {
        return sum + FindOrDefault(computation_costs_in_bytes, comp, 0l);
      });

  return memory_usage;
}
}  // namespace

int64_t EstimateMinimumLiveMemory(
    const HloInstructionMap<HloPoplarBufferIdSet>& program_liveness,
    const absl::flat_hash_map<HloPoplarBuffer::Id, int64_t>&
        buffer_sizes_in_bytes,
    const absl::flat_hash_map<const HloComputation*, int64_t>&
        computation_costs_in_bytes) {
  // Max buffer usage is the minimum live memory since we don't know what other
  // memory the Poplar ops of our program will use.
  int64_t max_memory_usage = 0;

  for (auto& item : program_liveness) {
    const auto* inst = item.first;
    const auto& live_buffers = item.second;

    const auto memory_usage =
        MemoryUsageOfBufferSet(live_buffers, buffer_sizes_in_bytes) +
        MemoryUsageOfComputations(inst->called_computations(),
                                  computation_costs_in_bytes);
    max_memory_usage = std::max(max_memory_usage, memory_usage);
  }

  return max_memory_usage;
}

}  // namespace poplarplugin
}  // namespace xla
