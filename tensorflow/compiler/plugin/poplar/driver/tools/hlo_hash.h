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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_HASH_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_HASH_H_

#include <map>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/macros.h"

#include "tensorflow/compiler/xla/service/hlo.pb.h"

namespace xla {

class HloModule;
class HloComputation;
class HloInstruction;

namespace poplarplugin {

class HloHash {
 public:
  HloHash(const HloModule* module) : module_(module){};
  HloHash() = delete;

  uint64 GetHash();
  std::string GetProtoStr();

 private:
  const HloModule* module_;
  uint64 hash_ = 0;
  std::string proto_str_;
  bool performed_hash_ = false;

  void HashModule();
  void SanitizeHloModuleProto(HloModuleProto*, const HloModule*);
  uint64 SanitizeHloComputationProto(HloComputationProto*, uint64);
  uint64 SanitizeHloInstructionProto(HloInstructionProto*, uint64);
  void SanitizeComputeProgramShape(ProgramShapeProto*);

  void PatchInstructionReferences(HloInstructionProto*,
                                  const std::map<uint64, uint64>&);
  void PatchComputationReferences(HloInstructionProto*,
                                  const std::map<uint64, uint64>&);
};

}  // namespace poplarplugin
}  // namespace xla

#endif