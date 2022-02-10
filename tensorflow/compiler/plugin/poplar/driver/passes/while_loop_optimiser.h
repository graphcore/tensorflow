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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_WHILE_LOOP_OPTIMISER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_WHILE_LOOP_OPTIMISER_H_

#include <vector>
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

class PoplarWhileLoopOptimiser : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "poplar-while-loop-optimiser";
  }
  StatusOr<bool> Run(HloModule* module) override;

  // Propagate a new shape through the module.
  // This method isn't general yet and should only be used
  // when reshaping conditions are satisfied. These include:
  //  - Can't propogate backwards through operands, only
  //    forward through users
  //  - Only a subset of opcodes are supported
  //  - If changing a computations parameters it must have
  //    only one callsite.
  //  - New instruction shapes must be able to be consistently
  //    propagated through the module
  static Status PropagateNewShapes(
      const std::vector<HloInstruction*>& instructions_with_new_shapes);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_WHILE_LOOP_OPTIMISER_H_
