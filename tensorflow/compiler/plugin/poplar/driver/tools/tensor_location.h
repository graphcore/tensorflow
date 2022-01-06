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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_TENSOR_LOCATION_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_TENSOR_LOCATION_H_

#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/platform/default/integral_types.h"

using tensorflow::int64;

namespace xla {
namespace poplarplugin {

template <typename Key, typename Value, typename Pair>
class ConstMapIterator {
 public:
  using MapType = std::map<Key, Value>;
  explicit ConstMapIterator(typename MapType::const_iterator iterator)
      : _iterator(iterator) {}
  ConstMapIterator operator++() {
    ++_iterator;
    return *this;
  }
  bool operator!=(const ConstMapIterator& other) const {
    return _iterator != other._iterator;
  }
  Pair operator*() const { return Pair(_iterator->first, _iterator->second); }

 private:
  typename MapType::const_iterator _iterator;
};

template <typename Key, typename Value, typename Pair>
class MapIterator {
 public:
  using MapType = std::map<Key, Value>;
  explicit MapIterator(typename MapType::iterator iterator)
      : _iterator(iterator) {}
  MapIterator operator++() {
    ++_iterator;
    return *this;
  }
  bool operator!=(const MapIterator& other) const {
    return _iterator != other._iterator;
  }
  Pair operator*() const { return Pair(_iterator->first, _iterator->second); }

 private:
  typename MapType::iterator _iterator;
};

// Represent the source location of a tensor as a unique pair of instruction and
// output index.
struct TensorLocation {
  TensorLocation() = default;
  TensorLocation(const HloInstruction* instr, int64 output_index)
      : instruction(instr), flattened_output_tuple_index(output_index) {}

  const HloInstruction* instruction;
  int64 flattened_output_tuple_index;
  bool operator<(const TensorLocation& other) const {
    // Standard pair comparison:
    return HloPtrComparator()(instruction, other.instruction) ||
           (!HloPtrComparator()(other.instruction, instruction) &&
            flattened_output_tuple_index < other.flattened_output_tuple_index);
  }
};

template <typename H>
H AbslHashValue(H h, const TensorLocation& location) {
  return H::combine(std::move(h),
                    reinterpret_cast<uintptr_t>(location.instruction),
                    location.flattened_output_tuple_index);
}

// Return index's value if set or 0 otherwise.
static inline int64 DefaultToFirst(absl::optional<int64> index) {
  return index.value_or(0);
}
// Return index's value if set or int64 max otherwise.
static inline int64 DefaultToLast(absl::optional<int64> index) {
  return index.value_or(std::numeric_limits<int64>::max());
}

static inline bool operator==(const TensorLocation& lhs,
                              const TensorLocation& rhs) {
  return lhs.instruction == rhs.instruction &&
         lhs.flattened_output_tuple_index == rhs.flattened_output_tuple_index;
}

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_TENSOR_LOCATION_H_
