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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_ALLOCATION_ANALYSIS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_ALLOCATION_ANALYSIS_H_

#include <string>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

class HloModule;

namespace poplarplugin {

struct IndexedLocation {
  HloInstruction* instruction;
  ShapeIndex index;
  IndexedLocation(HloInstruction* inst, ShapeIndex index)
      : instruction(inst), index(std::move(index)) {}

  std::string ToString() const;
};

struct InputLocation {
  HloInstruction* instruction;
  int64 operand_index;
  InputLocation(HloInstruction* inst, int64 index)
      : instruction(inst), operand_index(index) {}
  std::string ToString() const;
};

template <typename H>
H AbslHashValue(H h, const IndexedLocation& location) {
  return H::combine(std::move(h),
                    reinterpret_cast<uintptr_t>(location.instruction),
                    location.index);
}

template <typename H>
H AbslHashValue(H h, const InputLocation& location) {
  return H::combine(std::move(h),
                    reinterpret_cast<uintptr_t>(location.instruction),
                    location.operand_index);
}

static inline bool operator==(const IndexedLocation& lhs,
                              const IndexedLocation& rhs) {
  return lhs.instruction == rhs.instruction && lhs.index == rhs.index;
}

static inline bool operator!=(const IndexedLocation& lhs,
                              const IndexedLocation& rhs) {
  return !(lhs == rhs);
}

static inline bool operator==(const InputLocation& lhs,
                              const InputLocation& rhs) {
  return lhs.instruction == rhs.instruction &&
         lhs.operand_index == rhs.operand_index;
}

static inline bool operator!=(const InputLocation& lhs,
                              const InputLocation& rhs) {
  return !(lhs == rhs);
}

// An allocation group is meant to represent all instructions, whose tile
// mappings are related/dependant on each other. This usually means identical
// but also includes instructions whose mapping is dependant on their
// input/output mapping
struct AllocationGroup {
  // First instruction in the group that can set desired tile mapping
  IndexedLocation producer;
  // The instructions that have inputs in this group, but allocate
  // so aren't in the group themselves
  absl::flat_hash_set<InputLocation> inputs_only;
  // All instructions in the group that will get a tile mapping determined
  // by the producer. Note not necessarily the same mapping. Eg dynamic-slice
  // can propagate a tile mapping of it users through to a different tile
  // mapping for it's operand
  absl::flat_hash_set<IndexedLocation> group;

  explicit AllocationGroup(IndexedLocation producer) : producer(producer) {
    AddInstructionToGroup(producer);
  }

  std::string ToString() const;

  void AddInstructionToGroup(IndexedLocation inst) { group.insert(inst); }

  void AddGroupEndInstruction(InputLocation inst) { inputs_only.insert(inst); }

  void clear() {
    inputs_only.clear();
    group.clear();
    producer = {nullptr, {}};
  }
};

template <typename T, typename Pred>
void FilterInPlace(T& vec, Pred p) {
  auto not_pred = [&](const typename T::value_type& a) { return !p(a); };
  vec.erase(std::remove_if(vec.begin(), vec.end(), not_pred), vec.end());
}

// All the allocation groups in the module. Every IndexLocation
// that can have a tensor associated with it should be in one of these groups.
// Also provides a verify method for testing
struct AllocationGroups {
  std::vector<AllocationGroup> groups;

  std::string ToString() const;

  // Create the allocation groups for this module. This functions attempts to
  // split allocations groups if it doesn't know what to do. This
  // is so optimisation passes that use this can guarantee that if 2
  // instructions are in the same group then this is actually correct.
  static StatusOr<AllocationGroups> CreateAllocationGroups(HloModule* module);
  void Verify(HloModule* module) const;
  absl::flat_hash_map<IndexedLocation, int64> CreateLocationToGroupMap() const;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_ALLOCATION_ANALYSIS_H_
