/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_WHILE_LOOP_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_WHILE_LOOP_UTIL_H_

/*
 * These functions are independent of poplar, and are included in the
 * optimizers target within the BUILD file.
 */

#include <vector>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

class WhileLoopUtil {
 public:
  static bool IsGTEFromParamIndex(const HloInstruction* inst,
                                  int64 param_index);
  static bool CanRepresentInstructionAsInt64Constant(
      const HloInstruction* inst);

  // Find instructions which are incremented/decremented by a (-)1 and for
  // which the resulting increment is *only* used in the output tuple of the
  // while body in the same index as the inst.
  // Returns a vector of pairs of instructions and the constant
  // increment/decrement.
  static std::vector<std::pair<HloInstruction*, int64>>
  FindMatchingLoopDeltasInsideBody(const HloInstruction* inst,
                                   const HloComputation* while_body);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(WhileLoopUtil);
};

}  // namespace poplarplugin
}  // namespace xla

#endif
