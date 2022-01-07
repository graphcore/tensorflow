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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_COMBINE_INSTRUCTIONS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_COMBINE_INSTRUCTIONS_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

#include "absl/types/optional.h"

namespace xla {
namespace poplarplugin {

struct CompilerAnnotations;

class CombineInstructions : public HloModulePass {
 public:
  absl::string_view name() const override { return "combine-instructions"; };

  // Run the pass on the given HLO module.  Returns whether it modified the
  // module.
  StatusOr<bool> Run(HloModule* module) override;

 private:
  // Returns a new sequence if any instructions were combined.
  StatusOr<absl::optional<HloInstructionSequence>>
  CombineInstructionsInComputation(HloComputation* comp,
                                   const HloInstructionSequence& sequence);
};

}  // namespace poplarplugin
}  // namespace xla

#endif
