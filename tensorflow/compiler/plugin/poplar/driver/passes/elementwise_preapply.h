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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_ELEMENTWISE_PREAPPLY_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_ELEMENTWISE_PREAPPLY_H_

#include <utility>

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace poplarplugin {

// A pass which finds uniform elementwise ops on the output of target ops,
// and moves them to be applied to smaller inputs of the target.
// Uniform here means an elementwise op which uniformly affect the target,
// i.e. all inputs except the target are broadcasted scalars.
class ElementwisePreapply : public HloModulePass {
 public:
  absl::string_view name() const override { return "elementwise-preapply"; }

  // Run algebraic simplification on the given computation. Returns whether the
  // computation was changed.
  StatusOr<bool> Run(HloModule* module) override;
};
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_ELEMENTWISE_PREAPPLY_H_
