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

void UpdateLivenessBackwards(const HloInstruction* inst,
                             const HloPoplarBuffer* buffer,
                             HloPoplarBufferIdSet& live_set) {
  // Since we're going backwards we kill a buffer when it's defined. The only
  // exception being if it's the final result of the module - it's assumed
  // it'll be read after.
  if (buffer->DefinedBy(inst) && !IsFinalResult(inst)) {
    MarkBufferAsDead(buffer->id(), live_set);
  } else {
    MarkBufferAsAlive(buffer->id(), live_set);
  }
}
}  // namespace

HloInstructionMap<HloPoplarBufferIdSet> GenerateProgramLiveness(
    const std::vector<HloInstruction*>& flat_schedule,
    const HloInstructionMap<HloPoplarBufferSet>& buffer_usages) {
  HloInstructionMap<HloPoplarBufferIdSet> program_liveness;

  HloPoplarBufferIdSet live_set;

  for (auto it = flat_schedule.rbegin(); it != flat_schedule.rend(); ++it) {
    auto* inst = *it;

    auto usage_it = buffer_usages.find(inst);
    if (usage_it != buffer_usages.end()) {
      const auto& buffer_set = usage_it->second;

      for (auto& buffer : buffer_set.buffers()) {
        UpdateLivenessBackwards(inst, buffer, live_set);
      }
    }
    program_liveness[inst] = live_set;
  }

  return program_liveness;
}

namespace {
int64 MemoryUsageOfBufferSet(
    const HloPoplarBufferIdSet& buffers,
    const absl::flat_hash_map<HloPoplarBuffer::Id, int64>&
        buffer_sizes_in_bytes) {
  const int64 memory_usage = absl::c_accumulate(
      buffers, 0l,
      [&buffer_sizes_in_bytes](int64 sum, HloPoplarBuffer::Id buffer_id) {
        return sum + buffer_sizes_in_bytes.at(buffer_id);
      });
  return memory_usage;
}
}  // namespace

int64 EstimateMinimumLiveMemory(
    const HloInstructionMap<HloPoplarBufferIdSet>& program_liveness,
    const absl::flat_hash_map<HloPoplarBuffer::Id, int64>&
        buffer_sizes_in_bytes) {
  // Max buffer usage is the minimum live memory since we don't know what other
  // memory the Poplar ops of our program will use.
  int64 max_buffer_set_memory_usage = 0;

  for (auto& item : program_liveness) {
    auto& live_buffers = item.second;

    max_buffer_set_memory_usage =
        std::max(max_buffer_set_memory_usage,
                 MemoryUsageOfBufferSet(live_buffers, buffer_sizes_in_bytes));
  }

  return max_buffer_set_memory_usage;
}

}  // namespace poplarplugin
}  // namespace xla
