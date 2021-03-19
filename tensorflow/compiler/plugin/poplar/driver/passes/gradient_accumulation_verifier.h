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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_GRADIENT_ACCUMULATION_VERIFIER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_GRADIENT_ACCUMULATION_VERIFIER_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class CallGraph;
class HloModule;

namespace poplarplugin {

/**
 * Pass which verifies that gradient accumulation is being executed correctly.
 */
class GradientAccumulationVerifier : public HloModulePass {
 public:
  explicit GradientAccumulationVerifier(uint32 replication_factor)
      : replication_factor_(replication_factor) {}

  absl::string_view name() const override {
    return "gradient-accumulation-verifier";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  Status VerifyStatefulGradientAccumulation(HloInstruction* const inst,
                                            CallGraph* call_graph);
  Status VerifyGenericGradientAccumulation(HloInstruction* const inst,
                                           CallGraph* call_graph);

  const uint32 replication_factor_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_GRADIENT_ACCUMULATION_VERIFIER_H_
