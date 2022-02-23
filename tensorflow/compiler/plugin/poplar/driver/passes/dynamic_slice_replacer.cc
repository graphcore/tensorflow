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

#include "tensorflow/compiler/plugin/poplar/driver/passes/dynamic_slice_replacer.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/slice_util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {
namespace poplarplugin {

StatusOr<bool> DynamicSliceReplacer::Run(HloModule* module) {
  bool changed = false;

  for (auto* comp : module->MakeComputationPostOrder()) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      if (auto dynamic_slice = DynCast<HloDynamicIndexInstruction>(inst)) {
        const auto old_instr_str = dynamic_slice->ToString();
        TF_ASSIGN_OR_RETURN(auto replaced,
                            TryReplaceDynamicWithMultiSlice(dynamic_slice));
        if (replaced) {
          changed = true;
          VLOG(3) << "Replaced " << old_instr_str << " with "
                  << replaced->ToString();
        }
      }
    }
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
