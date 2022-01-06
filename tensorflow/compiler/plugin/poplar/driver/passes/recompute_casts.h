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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_RECOMPUTE_CASTS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_RECOMPUTE_CASTS_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace poplarplugin {

// Pass for cloning kConvert HLO instructions so each consumer has their own
// copy. This is intended to reduce memory livliness
class RecomputeCasts : public HloModulePass {
 public:
  absl::string_view name() const override { return "recompute-casts"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  Status SetupRecomputation(HloComputation* comp, HloInstruction* inst);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_RECOMPUTE_CASTS_H_
