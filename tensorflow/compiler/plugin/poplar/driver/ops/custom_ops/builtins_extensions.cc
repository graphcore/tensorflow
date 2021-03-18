/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/service/hlo_opcode.h"

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_instruction_extensions.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

namespace xla {
namespace poplarplugin {
namespace {

// This file contains the extensions for HloInstructions which have no
// corresponding PoplarOpDef.

void RegisterSharedExtensions(HloOpcode opcode) {
  auto allocatingOutput = [](HloInstruction*) { return true; };
  RegisterHloInstructionExtension<AllocatingOutputExtension>(opcode,
                                                             allocatingOutput);
}
REGISTER_HLO_OP_EXTENSIONS(kConstant, RegisterSharedExtensions);
REGISTER_HLO_OP_EXTENSIONS(kInfeed, RegisterSharedExtensions);
REGISTER_HLO_OP_EXTENSIONS(kParameter, RegisterSharedExtensions);
REGISTER_HLO_OP_EXTENSIONS(kReduce, RegisterSharedExtensions);
REGISTER_HLO_OP_EXTENSIONS(kReduceWindow, RegisterSharedExtensions);

void RegisterFuseExtensions(HloOpcode opcode) {
  auto allocatingOutput = [](HloInstruction* inst) {
    return (IsWideConstant(inst) || IsReductionFusion(inst));
  };
  RegisterHloInstructionExtension<AllocatingOutputExtension>(opcode,
                                                             allocatingOutput);
}
REGISTER_HLO_OP_EXTENSIONS(kFusion, RegisterFuseExtensions);

void RegisterDotExtensions(HloOpcode opcode) {
  auto allocatingIndices = [](HloInstruction* inst) {
    absl::flat_hash_set<int64> indices;
    for (auto i = 0u; i < inst->operand_count(); ++i) {
      indices.insert(i);
    }

    return indices;
  };

  RegisterHloInstructionExtension<AllocatingIndicesExtension>(
      opcode, allocatingIndices);
}
REGISTER_HLO_OP_EXTENSIONS(kDot, RegisterDotExtensions);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
