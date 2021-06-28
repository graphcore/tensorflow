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

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_instruction_extensions.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_extensions.h"

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

  auto do_find_consumers =
      [](const HloInstruction* user,
         FindConsumersExtensionParams params) -> FindConsumersExtensionResults {
    auto& [src, tgt, index, op_index, permutation] = params;

    HloComputation* comp = user->fused_instructions_computation();
    if (IsPopOpsFusion(user)) {
      if (IsPopOpsFusion(user, "zero_pad")) {
        FindConsumersExtensionResults result{/*do_find_consumers=*/true, user,
                                             index, permutation};
        return result;
      } else if (IsPopOpsFusion(user, "implicit")) {
        // Look through implicit elementwise ops if the shape dimensions
        // match.
        if (user->shape() == user->operand(op_index)->shape()) {
          FindConsumersExtensionResults result{/*do_find_consumers=*/true, user,
                                               index, permutation};
          return result;
        }
      }
    }

    return FindConsumersExtensionResults::DoNotFindConsumers();
  };
  RegisterHloInstructionExtension<FindConsumersExtension>(opcode,
                                                          do_find_consumers);
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

  auto do_find_consumers =
      [](const HloInstruction* user,
         FindConsumersExtensionParams params) -> FindConsumersExtensionResults {
    return FindConsumersExtensionResults::DoNotFindConsumers();
  };
  RegisterHloInstructionExtension<FindConsumersExtension>(opcode,
                                                          do_find_consumers);
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

  auto do_find_consumers =
      [](const HloInstruction* user,
         FindConsumersExtensionParams params) -> FindConsumersExtensionResults {
    return FindConsumersExtensionResults::DoNotFindConsumers();
  };
  RegisterHloInstructionExtension<FindConsumersExtension>(opcode,
                                                          do_find_consumers);
}
REGISTER_HLO_INST_EXTENSIONS(kCholesky, RegisterCholeskyExtensions);

void RegisterScatterExtensions(HloOpcode opcode) {
  auto allocating_indices = [](const HloInstruction*) {
    return absl::flat_hash_set<int64>{0, 1, 2};
  };
  RegisterHloInstructionExtension<AllocatingIndicesExtension>(
      opcode, allocating_indices);

  auto do_find_consumers =
      [](const HloInstruction* user,
         FindConsumersExtensionParams params) -> FindConsumersExtensionResults {
    return FindConsumersExtensionResults::DoNotFindConsumers();
  };
  RegisterHloInstructionExtension<FindConsumersExtension>(opcode,
                                                          do_find_consumers);
}
REGISTER_HLO_INST_EXTENSIONS(kScatter, RegisterScatterExtensions);

void RegisterGatherExtensions(HloOpcode opcode) {
  auto allocating_indices = [](const HloInstruction*) {
    return absl::flat_hash_set<int64>{0, 1};
  };
  RegisterHloInstructionExtension<AllocatingIndicesExtension>(
      opcode, allocating_indices);

  auto do_find_consumers =
      [](const HloInstruction* user,
         FindConsumersExtensionParams params) -> FindConsumersExtensionResults {
    return FindConsumersExtensionResults::DoNotFindConsumers();
  };
  RegisterHloInstructionExtension<FindConsumersExtension>(opcode,
                                                          do_find_consumers);
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

  auto do_find_consumers =
      [](const HloInstruction* user,
         FindConsumersExtensionParams params) -> FindConsumersExtensionResults {
    return FindConsumersExtensionResults::DoNotFindConsumers();
  };
  RegisterHloInstructionExtension<FindConsumersExtension>(opcode,
                                                          do_find_consumers);
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

  auto do_find_consumers =
      [](const HloInstruction* user,
         FindConsumersExtensionParams params) -> FindConsumersExtensionResults {
    return FindConsumersExtensionResults::DoNotFindConsumers();
  };
  RegisterHloInstructionExtension<FindConsumersExtension>(opcode,
                                                          do_find_consumers);
}
REGISTER_HLO_INST_EXTENSIONS(kConvolution, RegisterConvolutionExtensions);

void RegisterCallExtensions(HloOpcode opcode) {
  auto do_find_consumers =
      [](const HloInstruction* user,
         FindConsumersExtensionParams params) -> FindConsumersExtensionResults {
    HloComputation* comp = user->to_apply();
    HloInstruction* param = comp->parameter_instruction(params.op_index);
    FindConsumersExtensionResults result{/*do_find_consumers=*/true, param,
                                         params.index, params.permutation};
    return result;
  };
  RegisterHloInstructionExtension<FindConsumersExtension>(opcode,
                                                          do_find_consumers);
}
REGISTER_HLO_INST_EXTENSIONS(kCall, RegisterCallExtensions);

void RegisterConditionalExtensions(HloOpcode opcode) {
  auto do_find_consumers =
      [](const HloInstruction* user,
         FindConsumersExtensionParams params) -> FindConsumersExtensionResults {
    if (params.op_index != 0) {
      HloComputation* comp = user->branch_computation(params.op_index - 1);
      HloInstruction* param = comp->parameter_instruction(0);
      FindConsumersExtensionResults result{/*do_find_consumers=*/true, param,
                                           params.index, params.permutation};
      return result;
    }
    return FindConsumersExtensionResults::DoNotFindConsumers();
  };
  RegisterHloInstructionExtension<FindConsumersExtension>(opcode,
                                                          do_find_consumers);
}
REGISTER_HLO_INST_EXTENSIONS(kConditional, RegisterConditionalExtensions);

void RegisterWhileExtensions(HloOpcode opcode) {
  auto do_find_consumers =
      [](const HloInstruction* user,
         FindConsumersExtensionParams params) -> FindConsumersExtensionResults {
    HloComputation* comp = user->while_body();
    HloInstruction* param = comp->parameter_instruction(params.op_index);
    FindConsumersExtensionResults result{/*do_find_consumers=*/true, param,
                                         params.index, params.permutation};
    return result;
  };
  RegisterHloInstructionExtension<FindConsumersExtension>(opcode,
                                                          do_find_consumers);
}
REGISTER_HLO_INST_EXTENSIONS(kWhile, RegisterWhileExtensions);

void RegisterTupleExtensions(HloOpcode opcode) {
  auto do_find_consumers =
      [](const HloInstruction* user,
         FindConsumersExtensionParams params) -> FindConsumersExtensionResults {
    int64 new_index =
        InsertIntoTuple(user->shape(), params.op_index, params.index);
    FindConsumersExtensionResults result{/*do_find_consumers=*/true, user,
                                         new_index, params.permutation};
    return result;
  };
  RegisterHloInstructionExtension<FindConsumersExtension>(opcode,
                                                          do_find_consumers);
}
REGISTER_HLO_INST_EXTENSIONS(kTuple, RegisterTupleExtensions);

void RegisterGetTupleElementExtensions(HloOpcode opcode) {
  auto do_find_consumers =
      [](const HloInstruction* user,
         FindConsumersExtensionParams params) -> FindConsumersExtensionResults {
    int64 tuple_index = user->tuple_index();
    int64 new_index =
        ExtractFromTuple(params.tgt->shape(), tuple_index, params.index);
    if (new_index != -1) {
      FindConsumersExtensionResults result{/*do_find_consumers=*/true, user,
                                           new_index, params.permutation};
      return result;
    }
    return FindConsumersExtensionResults::DoNotFindConsumers();
  };
  RegisterHloInstructionExtension<FindConsumersExtension>(opcode,
                                                          do_find_consumers);
}
REGISTER_HLO_INST_EXTENSIONS(kGetTupleElement,
                             RegisterGetTupleElementExtensions);

void RegisterReshapeExtensions(HloOpcode opcode) {
  auto do_find_consumers = [](const HloInstruction* user,
                              FindConsumersExtensionParams params) {
    FindConsumersExtensionResults result{/*do_find_consumers=*/true, user,
                                         params.index, absl::nullopt};
    return result;
  };
  RegisterHloInstructionExtension<FindConsumersExtension>(opcode,
                                                          do_find_consumers);
}
REGISTER_HLO_INST_EXTENSIONS(kReshape, RegisterReshapeExtensions);

void RegisterTransposeExtensions(HloOpcode opcode) {
  auto do_find_consumers = [](const HloInstruction* user,
                              FindConsumersExtensionParams params) {
    absl::optional<std::vector<int64>> new_permutation;
    auto& permutation = params.permutation;
    if (permutation) {
      // Permute the dimensions according to the transpose.
      new_permutation = std::vector<int64>(permutation->size());
      const std::vector<int64> transpose_permutation = user->dimensions();
      for (int64 d = 0; d != permutation->size(); ++d) {
        (*new_permutation)[d] = (*permutation)[transpose_permutation[d]];
      }
    }
    FindConsumersExtensionResults result{/*do_find_consumers=*/true, user,
                                         params.index, new_permutation};
    return result;
  };
  RegisterHloInstructionExtension<FindConsumersExtension>(opcode,
                                                          do_find_consumers);
}
REGISTER_HLO_INST_EXTENSIONS(kTranspose, RegisterTransposeExtensions);

void RegisterConvertExtensions(HloOpcode opcode) {
  auto do_find_consumers = [](const HloInstruction* user,
                              FindConsumersExtensionParams params) {
    FindConsumersExtensionResults result{/*do_find_consumers=*/true, user,
                                         params.index, params.permutation};
    return result;
  };
  RegisterHloInstructionExtension<FindConsumersExtension>(opcode,
                                                          do_find_consumers);
}
REGISTER_HLO_INST_EXTENSIONS(kConvert, RegisterConvertExtensions);

void RegisterConcatenateExtensions(HloOpcode opcode) {
  auto do_find_consumers = [](const HloInstruction* user,
                              FindConsumersExtensionParams params) {
    FindConsumersExtensionResults result{/*do_find_consumers=*/true, user,
                                         params.index, params.permutation};
    return result;
  };
  RegisterHloInstructionExtension<FindConsumersExtension>(opcode,
                                                          do_find_consumers);
}
REGISTER_HLO_INST_EXTENSIONS(kConcatenate, RegisterConcatenateExtensions);

void RegisterSliceExtensions(HloOpcode opcode) {
  auto do_find_consumers =
      [](const HloInstruction* user,
         FindConsumersExtensionParams params) -> FindConsumersExtensionResults {
    if (IsUniformSingleDimSlice(user)) {
      FindConsumersExtensionResults result{/*do_find_consumers=*/true, user,
                                           params.index, params.permutation};
      return result;
    } else {
      return FindConsumersExtensionResults::DoNotFindConsumers();
    }
  };
  RegisterHloInstructionExtension<FindConsumersExtension>(opcode,
                                                          do_find_consumers);
}
REGISTER_HLO_INST_EXTENSIONS(kSlice, RegisterSliceExtensions);

void RegisterPadExtensions(HloOpcode opcode) {
  auto do_find_consumers =
      [](const HloInstruction* user,
         FindConsumersExtensionParams params) -> FindConsumersExtensionResults {
    if (params.op_index == 0) {
      FindConsumersExtensionResults result{/*do_find_consumers=*/true, user,
                                           params.index, params.permutation};
      return result;
    } else {
      return FindConsumersExtensionResults::DoNotFindConsumers();
    }
  };
  RegisterHloInstructionExtension<FindConsumersExtension>(opcode,
                                                          do_find_consumers);
}
REGISTER_HLO_INST_EXTENSIONS(kPad, RegisterPadExtensions);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
