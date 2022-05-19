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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_EXECUTION_COUNTER_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_EXECUTION_COUNTER_UTIL_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/driver_types.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
class HloModule;
class HloComputation;

namespace poplarplugin {
struct CompilerResources;

/**
 * A helper class for keeping track of the execution count in a given context.
 * For example:
 *
 * for (int i = z; i != n; ++i) {
 *   x = fn(x, i)
 * }
 *
 * `i` is the execution counter keeping track of how many times the body of the
 * loop has been executed. It has its initial value set before the loop is
 * executed and its value increases by one with each execution of the loop body.
 *
 * The execution counter can then be used to generate values which vary with
 * each execution of the body, for example random number seed generation, where
 * a new seed is required every execution of the same poplar sequence.
 *
 * When an inner scope is entered, for example:
 *
 * for (int i = z; i != n; ++i) {
 *   for (int j = i; j != n; ++j) {
 *     x = fn(x, j)
 *   }
 * }
 *
 * The inner scope inherits the initial values from the outer scope, but it
 * does not modify the counters of the outer scope.
 *
 * Care needs to be taken when the same sequence is used for multiple iterating
 * scopes (for examples pipeline stages). Independent counters for each use of
 * the same sequence need to be kept and copied in and out:
 *
 * T fn(x, &counter) {
 *   x = fn(x, counter)
 *   return x
 * }
 *
 * for (int i = z; i != n; ++i) {
 *   // fwd
 *   for (int j = i; j != n; ++j) {
 *     x = fn(x, j)
 *   }
 *   ...
 *   // recomp
 *   for (int j = i; j != n; ++j) {
 *     x = fn(x, j)
 *   }
 *   ...
 * }
 *
 * Because the `fn` (the Poplar sequence) is used multiple times, the `counter`
 * can be thought of as being passed by reference, and so separate counters are
 * needed.
 */
class ExecutionCounters {
 public:
  ExecutionCounters() = delete;
  explicit ExecutionCounters(CompilerResources& resources,
                             const poplar::DebugNameAndId& debug_name_and_id);

  // Clone the execution counters.
  ExecutionCounters Clone(const poplar::DebugNameAndId& debug_name_and_id);

  // Get a counter for a particular shard and mark it as live.
  StatusOr<DriverTensor> GetCounter(int64_t shard);

  // Copy counters from `source`. Any counters which are live in `this` are
  // marked as live in `source`.
  StatusOr<DriverProgramSequence> SetInitialValuesFrom(
      DriverGraph& graph, ExecutionCounters* source);

  // Create a sequence which sets the values of live counters to zero.
  DriverProgramSequence SetInitialValuesToZero(DriverGraph& graph);

  // Update counters in `destination` by copying the values of current counters.
  // Any counters which are live in `this` are expected to also be live in
  // `destination`
  StatusOr<DriverProgramSequence> UpdateCounters(
      DriverGraph& graph, ExecutionCounters* destination);

  // Increment all the live counters by one.
  DriverProgramSequence IncrementLiveCounters(DriverGraph& graph) const;

  const std::vector<bool>& GetLiveCounters() const;

  bool Initialized() const;

 private:
  // Tensors for each shard which requires an execution counter.
  std::vector<DriverTensor> counters_;

  // Track which tensors are actually live.
  std::vector<bool> live_counters_;

  CompilerResources& resources_;

  const poplar::DebugNameAndId dnai_;

  bool initialized_ = false;
};

// Get the execution counter tensor given an instruction and its sharding.
// Expects the instruction to either have no sharding or single device sharding.
StatusOr<DriverTensor> GetExecutionCounter(CompilerResources& resources,
                                           const HloInstruction* inst);

// Add copy programs to the sequence which set the counter values from the
// current scope currently at the top of the stack.
Status CopyExecutionCountersFromScope(DriverGraph& graph,
                                      CompilerResources& resources,
                                      ExecutionCounters& counters,
                                      DriverProgramSequence& sequence);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_EXECUTION_COUNTER_UTIL_H_
