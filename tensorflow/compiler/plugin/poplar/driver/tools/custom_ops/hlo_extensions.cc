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

#include <utility>
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_instruction_extensions.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
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
REGISTER_HLO_INST_EXTENSIONS(kReduceWindow, RegisterSharedExtensions);
REGISTER_HLO_INST_EXTENSIONS(kRng, RegisterSharedExtensions);

void RegisterReduceExtensions(HloOpcode opcode) {
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
REGISTER_HLO_INST_EXTENSIONS(kReduce, RegisterReduceExtensions);

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
    absl::flat_hash_set<int64_t> indices;
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
    return absl::flat_hash_set<int64_t>{0};
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
    return absl::flat_hash_set<int64_t>{0, 1, 2};
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
    return absl::flat_hash_set<int64_t>{0, 1};
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
    return absl::flat_hash_set<int64_t>{0, 1};
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
    absl::flat_hash_set<int64_t> indices;
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
  RegisterHloInstructionExtension<FindConsumersExtension>(
      opcode,
      [](const HloInstruction* user,
         FindConsumersExtensionParams params) -> FindConsumersExtensionResults {
        HloComputation* comp = user->to_apply();
        HloInstruction* param = comp->parameter_instruction(params.op_index);
        FindConsumersExtensionResults result{/*do_find_consumers=*/true, param,
                                             params.index, params.permutation};
        return result;
      });
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
    int64_t new_index =
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
    int64_t tuple_index = user->tuple_index();
    int64_t new_index =
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
    absl::optional<std::vector<int64_t>> new_permutation;
    auto& permutation = params.permutation;
    if (permutation) {
      // Permute the dimensions according to the transpose.
      new_permutation = std::vector<int64_t>(permutation->size());
      const std::vector<int64_t> transpose_permutation = user->dimensions();
      for (int64_t d = 0; d != permutation->size(); ++d) {
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

void RegisterInplaceOperand0Extension(HloOpcode opcode,
                                      HloInstructionType type) {
  RegisterHloInstructionExtension<InplaceExtension>(
      opcode, [type](const HloInstruction* inst) {
        return HloPoplarInplaceDescription(type, /*inplace_operands=*/{0},
                                           /*allow_non_inplace=*/false);
      });
}

void RegisterInplaceRWOperand0Extension(HloOpcode opcode) {
  RegisterInplaceOperand0Extension(opcode,
                                   HloInstructionType::kInplaceReadWrite);
}
void RegisterInplaceROOperand0Extension(HloOpcode opcode) {
  RegisterInplaceOperand0Extension(opcode,
                                   HloInstructionType::kInplaceReadOnly);
}

void RegisterInplaceAllOperandsExtension(HloOpcode opcode,
                                         HloInstructionType type,
                                         bool allow_non_inplace = false) {
  RegisterHloInstructionExtension<InplaceExtension>(
      opcode, [type, allow_non_inplace](const HloInstruction* inst) {
        HloPoplarInplaceDescription::OperandIndices indices(
            inst->operand_count());
        absl::c_iota(indices, 0);
        return HloPoplarInplaceDescription(type, std::move(indices),
                                           allow_non_inplace);
      });
}

void RegisterInplaceRWAllOperandsExtension(HloOpcode opcode) {
  RegisterInplaceAllOperandsExtension(opcode,
                                      HloInstructionType::kInplaceReadWrite);
}
void RegisterInplaceROAllOperandsExtension(HloOpcode opcode) {
  RegisterInplaceAllOperandsExtension(opcode,
                                      HloInstructionType::kInplaceReadOnly);
}

// Inplace on the first operand
REGISTER_HLO_INST_EXTENSIONS(kDynamicUpdateSlice,
                             RegisterInplaceRWOperand0Extension);
REGISTER_HLO_INST_EXTENSIONS(kScatter, RegisterInplaceRWOperand0Extension);

// Inplace on all operands.
REGISTER_HLO_INST_EXTENSIONS(kAllReduce, RegisterInplaceRWAllOperandsExtension);
REGISTER_HLO_INST_EXTENSIONS(kMap, RegisterInplaceRWAllOperandsExtension);
REGISTER_HLO_INST_EXTENSIONS(kSort, RegisterInplaceRWAllOperandsExtension);
REGISTER_HLO_INST_EXTENSIONS(kTuple, [](HloOpcode opcode) {
  return RegisterInplaceAllOperandsExtension(
      opcode, HloInstructionType::kInplaceReadWrite,
      /*allow_non_inplace=*/false);
});

// Inplace read-only ops.
// These ops are implemented as inplace ops on operand 0.
REGISTER_HLO_INST_EXTENSIONS(kAddDependency,
                             RegisterInplaceROOperand0Extension);
REGISTER_HLO_INST_EXTENSIONS(kBitcastConvert,
                             RegisterInplaceROOperand0Extension);
REGISTER_HLO_INST_EXTENSIONS(kBroadcast, RegisterInplaceROOperand0Extension);
REGISTER_HLO_INST_EXTENSIONS(kReshape, RegisterInplaceROOperand0Extension);
REGISTER_HLO_INST_EXTENSIONS(kReverse, RegisterInplaceROOperand0Extension);
REGISTER_HLO_INST_EXTENSIONS(kTranspose, RegisterInplaceROOperand0Extension);

// Inplace on all operands.
REGISTER_HLO_INST_EXTENSIONS(kConcatenate,
                             RegisterInplaceROAllOperandsExtension);

REGISTER_HLO_INST_EXTENSIONS(kWhile, [](HloOpcode opcode) {
  RegisterHloInstructionExtension<InplaceExtension>(
      opcode, [](const HloInstruction* inst) {
        CHECK_EQ(inst->operand_count(), 1);
        return HloPoplarInplaceDescription(
            HloInstructionType::kInplaceReadWrite, /*inplace_operands=*/{0},
            /*allow_non_inplace=*/false);
      });
});

REGISTER_HLO_INST_EXTENSIONS(kSlice, [](HloOpcode opcode) {
  RegisterHloInstructionExtension<InplaceExtension>(
      opcode, [](const HloInstruction* inst) {
        return HloPoplarInplaceDescription(HloInstructionType::kInplaceReadOnly,
                                           /*inplace_operands=*/{0},
                                           /*allow_non_inplace=*/true);
      });
});

// Inplace on the first 2 ops.
REGISTER_HLO_INST_EXTENSIONS(kPad, [](HloOpcode opcode) {
  RegisterHloInstructionExtension<InplaceExtension>(
      opcode, [](const HloInstruction* inst) {
        return HloPoplarInplaceDescription(HloInstructionType::kInplaceReadOnly,
                                           /*inplace_operands=*/{0, 1},
                                           /*allow_non_inplace=*/false);
      });
});

REGISTER_HLO_INST_EXTENSIONS(kGetTupleElement, [](HloOpcode opcode) {
  RegisterHloInstructionExtension<InplaceExtension>(
      opcode, [](const HloInstruction* inst) {
        return HloPoplarInplaceDescription(
            HloInstructionType::kInplaceGetTupleElement,
            /*inplace_operands=*/{0}, /*allow_non_inplace=*/false);
      });
});

REGISTER_HLO_INST_EXTENSIONS(kFusion, [](HloOpcode opcode) {
  RegisterHloInstructionExtension<InplaceExtension>(
      opcode, [](const HloInstruction* inst) {
        if (IsPopOpsFusion(inst)) {
          // Aggregate op inplace type from its buffer use kind.
          // For each inplace description, look on its buffer use.
          // Because we have only per-instruction inplace type,
          // we mark instruction read/write if it has at least one r/w aliased
          // buffer. We mark instruction read/only if it has at least one r/o
          // aliased buffer and has no r/w buffers.
          BufferUseKind use_kind = BufferUseKind::USE_NO_ALIAS;
          HloPoplarInplaceDescription::OperandIndices inplace_operands;
          auto fusion_config = inst->backend_config<PoplarBackendConfig>()
                                   .ValueOrDie()
                                   .fusion_config();
          auto inplace_descriptions = fusion_config.inplace_descriptions();
          for (const auto& inplace_description : inplace_descriptions) {
            auto use_description =
                HloPoplarUseDescription::FromProto(inplace_description);
            if (use_description.kind() > use_kind) {
              use_kind = use_description.kind();
            }
            inplace_operands.push_back(use_description.operand_number());
          }
          absl::c_sort(inplace_operands);
          if (inplace_operands.size()) {
            HloInstructionType inplace_type;
            switch (use_kind) {
              case BufferUseKind::USE_ALIAS_READ_ONLY:
                inplace_type = HloInstructionType::kInplaceReadOnly;
                break;
              case BufferUseKind::USE_ALIAS_READ_WRITE:
                inplace_type = HloInstructionType::kInplaceReadWrite;
                break;
              case BufferUseKind::USE_NO_ALIAS:
                inplace_type = HloInstructionType::kNotInplace;
                break;
              default:
                LOG(FATAL) << "Invalid buffer use kind.";
            }
            bool allow_non_inplace = fusion_config.allow_non_inplace();
            return HloPoplarInplaceDescription(
                inplace_type, std::move(inplace_operands), allow_non_inplace);
          } else {
            return HloPoplarInplaceDescription();
          }
        } else {
          // A non poplibs fusion is inplace on all operands.
          HloPoplarInplaceDescription::OperandIndices indices(
              inst->operand_count());
          absl::c_iota(indices, 0);
          return HloPoplarInplaceDescription(
              HloInstructionType::kInplaceReadWrite, std::move(indices),
              /*allow_non_inplace=*/false);
        }
      });
});

REGISTER_HLO_INST_EXTENSIONS(kCall, [](HloOpcode opcode) {
  RegisterHloInstructionExtension<InplaceExtension>(
      opcode, [](const HloInstruction* inst) {
        if (IsRepeatLoop(inst)) {
          HloPoplarInplaceDescription::OperandIndices indices;
          const int64_t num_operands = inst->operand_count();
          const HloComputation* comp = inst->to_apply();
          const HloInstruction* root = comp->root_instruction();

          // The loop is considered to be inplace on all operands unless all
          // it's users are GTEs
          const bool all_users_gtes = absl::c_all_of(
              inst->users(), [](const HloInstruction* user) -> bool {
                return user->opcode() == HloOpcode::kGetTupleElement;
              });
          // The root instruction needs to be an inplace tuple - this makes sure
          // that an particular input is only used in a single place.
          // The loop also must have been broken up into individual inputs.

          // Check which inputs are actually modified.
          if (GetRepeatLoopAllowFinerAliasAnalysis(inst) &&
              IsLoweredInplace(root) && root->opcode() == HloOpcode::kTuple &&
              num_operands == root->operand_count() && all_users_gtes) {
            // Vector indiciating whether a given input/output index has a gte
            // output.
            std::vector<bool> has_gte(num_operands, false);
            for (const HloInstruction* user : inst->users()) {
              CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
              has_gte[user->tuple_index()] = true;
            }

            for (int64_t idx = 0; idx != num_operands; ++idx) {
              // An operand is not inplace if there is no gte for it and it's
              // used directly in the root instruction at the same index.
              if (has_gte[idx] ||
                  root->operand(idx) != comp->parameter_instruction(idx)) {
                indices.push_back(idx);
              }
            }
          } else {
            // Inplace on all its inputs.
            indices.resize(num_operands);
            absl::c_iota(indices, 0);
          }

          return HloPoplarInplaceDescription(
              HloInstructionType::kInplaceReadWrite, std::move(indices),
              /*allow_non_inplace=*/false);
        } else if (IsPipelineOp(inst) || IsResourceUpdate(inst)) {
          // Pipeline and ResourceUpdate operations are inplace on all
          // their inputs.
          HloPoplarInplaceDescription::OperandIndices indices(
              inst->operand_count());
          absl::c_iota(indices, 0);
          return HloPoplarInplaceDescription(
              HloInstructionType::kInplaceReadWrite, std::move(indices),
              /*allow_non_inplace=*/false);
        } else if (IsAnyPipelineStageOp(inst)) {
          // Pipeline stages are only inplace on operands which are not
          // parameters/execution counters.

          HloPoplarInplaceDescription::OperandIndices indices;
          // Backward pipeline stages don't mark gradient accumulators as
          // inplace inputs.
          const bool is_bwd = IsPipelineStageBackward(inst);

          HloComputation* comp = inst->to_apply();
          for (int64_t op_idx = 0; op_idx != inst->operand_count(); ++op_idx) {
            const HloInstruction* operand = inst->operand(op_idx);
            if (!IsPipelineStageReadOnlyInput(operand) &&
                !(is_bwd &&
                  IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(
                      operand))) {
              // If the stage modifies the input inplace, add it as an inplace
              // operand.
              if (IsOutputModifiedInplace(
                      comp->parameter_instruction(op_idx))) {
                indices.push_back(op_idx);
              }
            }
          }
          return HloPoplarInplaceDescription(
              HloInstructionType::kInplaceReadWrite, std::move(indices),
              /*allow_non_inplace=*/false);
        } else if (IsFunction(inst)) {
          // Functions are inplace on remote buffer inputs.
          // Assume that the first "num_modified_remote_buffers" inputs are
          // remote buffers which are modified and they are also the first
          // "num_modified_remote_buffers" outputs.
          // Assume that the next "num_unmodified_remote_buffers" inputs are
          // remote buffers which are only loaded.
          const int64_t num_modified_remote_buffers =
              GetFunctionNumberModifiedRemoteBufferInputs(inst);
          const int64_t num_unmodified_remote_buffers =
              GetFunctionNumberUnmodifiedRemoteBufferInputs(inst);
          // TODO(T10387): consider unmodified remote buffers as read only.
          if (num_modified_remote_buffers + num_unmodified_remote_buffers) {
            HloPoplarInplaceDescription::OperandIndices indices(
                num_modified_remote_buffers + num_unmodified_remote_buffers);
            absl::c_iota(indices, 0);
            return HloPoplarInplaceDescription(
                HloInstructionType::kInplaceReadWrite, std::move(indices),
                /*allow_non_inplace=*/false);
          } else {
            return HloPoplarInplaceDescription();
          }
        } else {
          // Calls are not inplace.
          return HloPoplarInplaceDescription();
        }
      });
});

REGISTER_HLO_INST_EXTENSIONS(kCustomCall, [](HloOpcode opcode) {
  RegisterHloInstructionExtension<InplaceExtension>(
      opcode, [](const HloInstruction* inst) {
        CHECK(!IsPoplibsHloCustomOp(inst));
        HloPoplarInplaceDescription::OperandIndices indices(
            inst->operand_count());
        absl::c_iota(indices, 0);
        return HloPoplarInplaceDescription(
            HloInstructionType::kInplaceReadWrite, std::move(indices),
            /*allow_non_inplace=*/false);
      });
});

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
