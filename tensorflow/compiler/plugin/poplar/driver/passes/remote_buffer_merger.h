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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_REMOTE_BUFFER_MERGER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_REMOTE_BUFFER_MERGER_H_

#include "tensorflow/compiler/plugin/poplar/driver/threestate.pb.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
class HloModule;

namespace poplarplugin {
class CompilerAnnotations;

/**
 * This pass tries to merge compatible remote buffers. When a buffer is merged,
 * all its users (the loads and stores) are rewritten to access their part of
 * the merged buffer by using an offset. If the user is inside a function, the
 * offset is hoisted out of the function and made an argument to the call. The
 * hope is that this can allow for re-using the same function for accessing the
 * different parts of the merged buffer, as the offset is just a tensor.
 */
class RemoteBufferMerger : public HloModulePass {
 public:
  explicit RemoteBufferMerger(CompilerAnnotations& annotations,
                              ThreeState mode = THREESTATE_UNDEFINED)
      : annotations_(annotations), mode_(mode) {}

  absl::string_view name() const override { return "remote-buffer-merger"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  CompilerAnnotations& annotations_;
  ThreeState mode_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_REMOTE_BUFFER_MERGER_H_
