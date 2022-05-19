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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_ISOMORPHIC_FUNCTIONS_MAP_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_ISOMORPHIC_FUNCTIONS_MAP_H_

#include <map>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

using Functions = HloInstructionSet;

// A container for storing isomorphic functions which allows deterministic
// iterating over the functions.
template <bool is_sharding_sensitive, bool sort_by_increasing_size>
class IsomorphicFunctionsMap {
  struct FunctionHash {
    size_t operator()(const HloInstruction* inst) const {
      return HloComputationHash()(inst->to_apply());
    }
  };

  struct FunctionEquals {
    bool operator()(const HloInstruction* a, const HloInstruction* b) const {
      return a->to_apply()->Equal(*b->to_apply(),
                                  /*is_layout_sensitive=*/false,
                                  is_sharding_sensitive);
    }
  };

  using StorageMap = std::unordered_map<HloInstruction*, Functions,
                                        FunctionHash, FunctionEquals>;

  struct FunctionCompare {
    bool operator()(const HloInstruction* a, const HloInstruction* b) const {
      if (sort_by_increasing_size) {
        const int64_t a_size = GetByteSizeOfTotalShape(a->shape());
        const int64_t b_size = GetByteSizeOfTotalShape(b->shape());

        if (a_size != b_size) {
          return a_size < b_size;
        }
      }

      return HloPtrComparator()(a, b);
    }
  };

  using AccessMap = std::map<HloInstruction*, Functions*, FunctionCompare>;

 public:
  void insert(HloInstruction* inst) {
    auto itr = storage_map_.find(inst);
    if (itr == storage_map_.end()) {
      itr = storage_map_.emplace(inst, Functions{inst}).first;
      // Add the reference to the storage map.
      CHECK(!ContainsKey(access_map_, inst));
      access_map_[inst] = &itr->second;
    } else {
      itr->second.insert(inst);
    }
  }

  void erase(HloInstruction* inst) {
    auto itr = storage_map_.find(inst);
    if (itr != storage_map_.end()) {
      auto key = itr->first;
      CHECK(ContainsKey(access_map_, key));
      access_map_.erase(key);
      storage_map_.erase(key);
    }
  }

  const Functions& at(HloInstruction* inst) const {
    return *access_map_.at(inst);
  }

  std::size_t size() const { return access_map_.size(); }

  bool empty() const { return access_map_.empty(); }

  typename AccessMap::const_iterator begin() const {
    return access_map_.begin();
  }

  typename AccessMap::const_iterator end() const { return access_map_.end(); }

 private:
  // Note: access map stores references to storage map. The order matters.
  StorageMap storage_map_;
  AccessMap access_map_;
};

using SingleShardIsomorphicFunctions = IsomorphicFunctionsMap<true, false>;
using CrossShardIsomorphicFunctions = IsomorphicFunctionsMap<false, false>;
// Variants where the iterator is sorted by function output size.
using SingleShardIsomorphicFunctionsOutputSizeSorted =
    IsomorphicFunctionsMap<true, true>;
using CrossShardIsomorphicFunctionsOutputSizeSorted =
    IsomorphicFunctionsMap<false, true>;

}  // namespace poplarplugin
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_ISOMORPHIC_FUNCTIONS_MAP_H_
