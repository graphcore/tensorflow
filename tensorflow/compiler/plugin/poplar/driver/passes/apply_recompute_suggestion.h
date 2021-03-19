/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_APPLY_RECOMPUTE_SUGGESTION_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_APPLY_RECOMPUTE_SUGGESTION_H_

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/core/framework/types.h"

namespace xla {

namespace poplarplugin {

// Return whether or not the given un-preprocessed Hlo module contains
// any SuggestsRecompute instruction.
bool UsesRecomputationSuggestions(const HloModule* module);

/**
 * Where possible, apply the suggestions for recomputation by cloning the input
 * of all suggestion ops.
 */
class ApplyRecomputeSuggestion : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "apply-recompute-suggestion";
  }
  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_APPLY_RECOMPUTE_SUGGESTION_H_
