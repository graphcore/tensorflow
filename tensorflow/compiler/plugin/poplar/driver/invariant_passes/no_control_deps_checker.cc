/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/invariant_passes/no_control_deps_checker.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

StatusOr<bool> NoControlDepsChecker::Run(HloModule* module) {
  for (HloComputation* comp : module->computations()) {
    const bool has_control_deps =
        absl::c_any_of(comp->instructions(), [](const HloInstruction* inst) {
          return inst->control_successors().size();
        });
    if (has_control_deps) {
      return InternalErrorStrCat(
          "Expected no control dependencies in a module, however ",
          comp->name(), " has control dependencies.");
    }
  }
  return false;
}

}  // namespace poplarplugin
}  // namespace xla
