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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_LIVENESS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_LIVENESS_H_

#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_dataflow_analysis.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace poplarplugin {

// Find all the HloPoplarBuffer objects read/written to by each instruction in
// the given module.
HloInstructionMap<HloPoplarBufferSet> FindUsedBuffers(
    const HloModule* module,
    const HloPoplarDataflowAnalysis& dataflow_analysis);
HloInstructionMap<HloPoplarBufferSet> FindUsedBuffers(
    const HloModule* module,
    const InstructionBufferSets& instruction_buffer_sets);

// Generate a liveness map for the instructions in the given schedule. This
// describes which buffers are live at each instruction of the schedule.
HloInstructionMap<HloPoplarBufferIdSet> GenerateProgramLiveness(
    const std::vector<HloInstruction*>& flat_schedule,
    const HloInstructionMap<HloPoplarBufferSet>& buffer_usages);

// Estimate the minimum amount of IPU memory required to run a program with
// the given liveness. The costs for any subcomputations called can be provided
// via the `computation_costs_in_bytes` argument.
int64_t EstimateMinimumLiveMemory(
    const HloInstructionMap<HloPoplarBufferIdSet>& program_liveness,
    const absl::flat_hash_map<HloPoplarBuffer::Id, int64_t>&
        buffer_sizes_in_bytes,
    const absl::flat_hash_map<const HloComputation*, int64_t>&
        computation_costs_in_bytes = {});

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_LIVENESS_H_
