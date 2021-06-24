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

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_instruction_extensions.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

namespace xla {
namespace poplarplugin {
namespace {

void RegisterSharedExtensions(HloOpcode opcode) {
  auto allocating_output = [](const HloInstruction*) { return true; };
  RegisterHloInstructionExtension<AllocatingOutputExtension>(opcode,
                                                             allocating_output);
}
REGISTER_HLO_INST_EXTENSIONS(kConstant, RegisterSharedExtensions);
REGISTER_HLO_INST_EXTENSIONS(kInfeed, RegisterSharedExtensions);
REGISTER_HLO_INST_EXTENSIONS(kParameter, RegisterSharedExtensions);
REGISTER_HLO_INST_EXTENSIONS(kReduce, RegisterSharedExtensions);
REGISTER_HLO_INST_EXTENSIONS(kReduceWindow, RegisterSharedExtensions);
REGISTER_HLO_INST_EXTENSIONS(kRng, RegisterSharedExtensions);

void RegisterFuseExtensions(HloOpcode opcode) {
  auto allocating_output = [](const HloInstruction* inst) {
    return (IsWideConstant(inst) || IsReductionFusion(inst));
  };
  RegisterHloInstructionExtension<AllocatingOutputExtension>(opcode,
                                                             allocating_output);
}
REGISTER_HLO_INST_EXTENSIONS(kFusion, RegisterFuseExtensions);

void RegisterDotExtensions(HloOpcode opcode) {
  auto allocating_indices = [](const HloInstruction* inst) {
    absl::flat_hash_set<int64> indices;
    for (auto i = 0u; i < inst->operand_count(); ++i) {
      indices.insert(i);
    }

    return indices;
  };

  RegisterHloInstructionExtension<AllocatingIndicesExtension>(
      opcode, allocating_indices);
}
REGISTER_HLO_INST_EXTENSIONS(kDot, RegisterDotExtensions);

void RegisterCholeskyExtensions(HloOpcode opcode) {
  auto allocating_indices = [](const HloInstruction*) {
    return absl::flat_hash_set<int64>{0};
  };
  RegisterHloInstructionExtension<AllocatingIndicesExtension>(
      opcode, allocating_indices);

  auto allocating_output = [](const HloInstruction*) { return true; };
  RegisterHloInstructionExtension<AllocatingOutputExtension>(opcode,
                                                             allocating_output);
}
REGISTER_HLO_INST_EXTENSIONS(kCholesky, RegisterCholeskyExtensions);

void RegisterScatterExtensions(HloOpcode opcode) {
  auto allocating_indices = [](const HloInstruction*) {
    return absl::flat_hash_set<int64>{0, 1, 2};
  };
  RegisterHloInstructionExtension<AllocatingIndicesExtension>(
      opcode, allocating_indices);
}
REGISTER_HLO_INST_EXTENSIONS(kScatter, RegisterScatterExtensions);

void RegisterGatherExtensions(HloOpcode opcode) {
  auto allocating_indices = [](const HloInstruction*) {
    return absl::flat_hash_set<int64>{0, 1};
  };
  RegisterHloInstructionExtension<AllocatingIndicesExtension>(
      opcode, allocating_indices);
}
REGISTER_HLO_INST_EXTENSIONS(kGather, RegisterGatherExtensions);

void RegisterTriangularSolveExtensions(HloOpcode opcode) {
  auto allocating_indices = [](const HloInstruction*) {
    return absl::flat_hash_set<int64>{0, 1};
  };
  RegisterHloInstructionExtension<AllocatingIndicesExtension>(
      opcode, allocating_indices);

  auto allocating_output = [](const HloInstruction*) { return true; };
  RegisterHloInstructionExtension<AllocatingOutputExtension>(opcode,
                                                             allocating_output);
}
REGISTER_HLO_INST_EXTENSIONS(kTriangularSolve,
                             RegisterTriangularSolveExtensions);

void RegisterConvolutionExtensions(HloOpcode opcode) {
  auto allocating_indices = [](const HloInstruction* inst) {
    absl::flat_hash_set<int64> indices;
    for (auto i = 0u; i < inst->operand_count(); ++i) {
      indices.insert(i);
    }

    return indices;
  };

  RegisterHloInstructionExtension<AllocatingIndicesExtension>(
      opcode, allocating_indices);
}
REGISTER_HLO_INST_EXTENSIONS(kConvolution, RegisterConvolutionExtensions);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
