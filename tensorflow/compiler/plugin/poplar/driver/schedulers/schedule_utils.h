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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_SCHEDULERS_SCHEDULE_UTILS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_SCHEDULERS_SCHEDULE_UTILS_H_

namespace xla {
namespace poplarplugin {

struct HloInstructionForEachPredecessor {
  template <typename F>
  void operator()(HloInstruction* inst, F f) const {
    for (auto user : inst->unique_operands()) {
      f(user);
    }

    for (auto user : inst->control_predecessors()) {
      f(user);
    }
  }
};

struct HloInstructionForEachSucessor {
  template <typename F>
  void operator()(HloInstruction* inst, F f) const {
    for (auto user : inst->users()) {
      f(user);
    }

    for (auto user : inst->control_successors()) {
      f(user);
    }
  }
};

}  // namespace poplarplugin
}  // namespace xla

#endif
