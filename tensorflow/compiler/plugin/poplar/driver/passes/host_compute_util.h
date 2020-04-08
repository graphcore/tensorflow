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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_HOST_COMPUTE_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_HOST_COMPUTE_UTIL_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {

namespace poplarplugin {

struct SendRecvs {
  std::vector<HloInstruction*> sends;
  std::vector<HloInstruction*> recvs;
};

using OpSendRecvs = std::unordered_map<string, SendRecvs>;

OpSendRecvs GroupSendRecvsByHostComputeOp(const HloComputation* comp);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_HOST_COMPUTE_UTIL_H_
