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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_MODULE_FLATTEN_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_MODULE_FLATTEN_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

struct CompilerAnnotations;

/**
 * Produce a flattened deep copy of the module, and generate a bi-directional
 * map between instructions in the original module and instructions in the
 * flattened module.
 *
 * The flattened module does not contain all of the control dependencies that
 * were present in the original module.
 */
class ModuleFlatten : public HloModulePass {
 public:
  ModuleFlatten(CompilerAnnotations& annotations);

  absl::string_view name() const override { return "module-flatten"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  void RemoveMapEntry(HloInstruction* inst);

  CompilerAnnotations& annotations_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
