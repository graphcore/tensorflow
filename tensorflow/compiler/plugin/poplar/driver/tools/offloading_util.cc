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
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace poplarplugin {
Status GetRemoteLoadStoreUsers(HloInstruction* inst, HloInstruction** load,
                               HloInstruction** store) {
  if (inst->user_count() != 2) {
    return FailedPrecondition(
        "Expected an offloaded instruction %s to have two users.",
        inst->name().c_str());
  }

  const int64 load_user_idx =
      IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(inst->users()[0]) ? 0
                                                                           : 1;

  *load = inst->users()[load_user_idx];
  *store = inst->users()[1 - load_user_idx];

  if (!IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(*load)) {
    return FailedPrecondition("Could not find a load instruction.");
  }
  if (!IsPoplarInstruction(PoplarOp::RemoteParameterStore)(*store)) {
    return FailedPrecondition("Could not find a store instruction.");
  }
  return Status::OK();
}
}  // namespace poplarplugin
}  // namespace xla
