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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_FUNCTION_COMBINER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_FUNCTION_COMBINER_H_

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/isomorphic_functions_map.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
class HloInstruction;
class HloModule;

namespace poplarplugin {

/**
 * This pass tries to combine single shard functions on different shards with
 * remote loads/stores together to allow for parallel load/stores.
 *
 * This pass aims to optimize the performance without adding any extra memory
 * overheads, meaning that only certain functions can be combined.
 *
 * For each resource update this pass does the following:
 * 1. Find sets of isomorphic functions with remote buffer loads/stores.
 * 2. For each set of functions, create a `CrossShardFunctionKey` - this
 * identifies all functions between shards which can be combined.
 * 3. Iterate through the per shard functions, combining the largest functions
 * from each shard first.
 */
using FunctionsToCombine = std::vector<std::vector<Functions>>;

class FunctionCombiner : public HloModulePass {
 public:
  absl::string_view name() const override { return "function-combiner"; }

  StatusOr<bool> Run(HloModule* module) override;
  StatusOr<bool> RunOnComputation(HloComputation* comp);

  static FunctionsToCombine GetFunctionsToCombine(HloComputation* comp);

  // Helper struct for storing permutations when combining functions.
  struct Permutations {
    std::vector<int64_t> old_to_new_inputs_permutation;
    std::vector<int64_t> old_to_new_outputs_permutation;
  };

  // Create permutation of inputs and outputs to make sure all the remote
  // buffers are the first inputs/outputs.
  static Permutations GetInputsOutputsPermutation(
      const std::vector<HloInstruction*>& functions);

  // Takes a vector of sets of isomorphic functions, returns all the combined
  // functions.
  static StatusOr<std::vector<HloInstruction*>> CombineFunctions(
      const std::vector<Functions>& per_shard_functions);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_FUNCTION_COMBINER_H_
