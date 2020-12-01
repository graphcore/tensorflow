/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_COMPUTATION_FLATTENER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_COMPUTATION_FLATTENER_H_

#include <unordered_set>

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
class CallGraphNode;

namespace poplarplugin {

// If this computation has only one caller, and the callsite is a kCall
// operation, then merge with the calling computation.
class ComputationFlattener : public HloModulePass {
 public:
  absl::string_view name() const override { return "computation-flattener"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  Status FlattenNode(const CallGraphNode&);
  Status GenerateFunctionSet(const CallGraphNode&);
  Status FindRecomputableComputations(const HloModule*);

  // This set tracks computations based on being identical despite being
  // different computations.  During the descison to inline, this set is used
  // to determine whether there are multiple identical computations.
  std::unordered_multiset<const HloComputation*, HloComputationHash,
                          HloComputationEquals>
      all_function_comps_;

  // Some computations might be recomputed later and therefore they should not
  // be inlined.
  absl::flat_hash_set<const HloComputation*> recomputable_computations_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
