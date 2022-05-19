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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_FIND_ALL_USERS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_FIND_ALL_USERS_H_

#include <vector>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

namespace poplarplugin {

using InstructionList = std::vector<HloInstruction*>;

/**
 * A class which finds all of the 'real' users of an instruction, looking
 * through kCall, kWhile, kTuple and kGetTupleElement operations.
 */
class FindAllUsers {
 public:
  FindAllUsers() {}

  void Find(HloInstruction* inst);

  std::set<HloInstruction*> Users() const;
  const std::set<InstructionList>& Paths() const;
  const InstructionList& PathFor(HloInstruction* target) const;

 private:
  void FindUsers(HloInstruction* tgt, const InstructionList& stack,
                 int64_t index);

  InstructionList path;
  std::set<InstructionList> paths;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
