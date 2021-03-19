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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_RESOURCE_UPDATE_COPY_INSERTER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_RESOURCE_UPDATE_COPY_INSERTER_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

/**
 * In repeat loops and pipelining, where resource updates can be present, the
 * output tensors have to be copied to the input tensors if they are not the
 * exact same tensor - this can occur when an input is not updated inplace.
 * This can increase the memory usage as there might be a lot of parameters to
 * be copied at once.
 *
 * This pass inserts these aliasing copies explicitly into the graph so that
 * they can be scheduled earlier.
 */
class ResourceUpdateCopyInserter : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "resource-update-copy-inserter";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  StatusOr<bool> InsertCopiesInCall(HloInstruction* const call);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_RESOURCE_UPDATE_COPY_INSERTER_H_
