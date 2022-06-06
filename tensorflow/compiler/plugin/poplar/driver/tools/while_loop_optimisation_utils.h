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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_WHILE_LOOP_OPTIMISATION_UTILS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_WHILE_LOOP_OPTIMISATION_UTILS_H_

#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "tensorflow/core/platform/default/integral_types.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {

class CallGraph;

namespace poplarplugin {

namespace WhileLoopOptimisationUtils {

struct Uses {
  absl::InlinedVector<HloInstruction*, 1> uses;

  std::string ToString() const;
};

struct SliceAndIndex {
  int64_t input_index;
  HloInstruction* dynamic_update;
  int64_t index_index;

  SliceAndIndex(int64_t input_index, HloInstruction* dynamic_update,
                int64_t index_index);
  std::string ToString() const;
};

struct BroadcastAndSlice {
  SliceAndIndex broadcast;
  Uses uses;

  BroadcastAndSlice(SliceAndIndex broadcast, Uses uses);
  std::string ToString() const;
};

struct FindUsesTemplate {
  absl::InlinedVector<HloInstruction*, 1> result;
  absl::flat_hash_set<HloComputation*> while_bodies;

  void DefaultAction(HloInstruction* user);
  void GTEAction(HloInstruction* user);
  void TupleAction(HloInstruction* user);
  void WhileAction(HloInstruction* user);
};

absl::flat_hash_set<HloInstruction*> FindLoopConstants(
    HloComputation* comp, bool include_constants = true);

std::vector<BroadcastAndSlice> FindAllValidBroadcasts(HloInstruction* inst);

}  // namespace WhileLoopOptimisationUtils
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_WHILE_LOOP_OPTIMISATION_UTILS_H_
