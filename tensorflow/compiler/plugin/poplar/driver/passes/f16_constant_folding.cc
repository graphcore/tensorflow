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

#include "tensorflow/compiler/plugin/poplar/driver/passes/f16_constant_folding.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

namespace xla {
namespace poplarplugin {

namespace {

Status ReplaceF16InstructionByF32(
    HloInstruction* inst, absl::flat_hash_set<HloInstruction*>& constant_32_dag,
    bool& graph_modified) {
  HloInstruction* inst_f32 = ConvertInstruction(inst, PrimitiveType::F32);
  TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(inst_f32));
  constant_32_dag.insert(inst_f32);
  graph_modified = true;
  return Status::OK();
}

// Check if case B
bool CaseBAllOperandsInConst32Set(
    HloInstruction* inst,
    absl::flat_hash_set<HloInstruction*>& constant_32_dag) {
  bool all_operands_in_const32_set =
      absl::c_all_of(inst->operands(), [&](const HloInstruction* inst) {
        return constant_32_dag.contains(inst);
      });

  return (!inst->HasSideEffect() && inst->operand_count() > 0 &&
          inst->shape().element_type() == PrimitiveType::F16 &&
          all_operands_in_const32_set &&
          inst->opcode() != HloOpcode::kBroadcast);
}

}  // namespace
/*
 * Finds subtrees, DAGs, of fp16 ops. Change them all to fp32 ops.
 * We iterate through all instructions in post order computation.
 * Case A: The instruction is fp16 constant. We replace it by fp32 constant and
 * we use set - DAG32 to remember that.
 * Case B: All operands of the instruction are in the DAG32. We replace it by
 * fp32 constant.
 * Case C: At least one operand is in the DAG32. We create fp16
 * convert between the instruction and the operand. At the end, If root is in
 * the DAG32 we add fp16 convert.
 *
 * Broadcasts dramatically increase the size of constants, which is often
 * detrimental to performance and memory capacity, so do not fold broadcasts.
 */

StatusOr<bool> F16ConstantFolding::Run(HloModule* module) {
  bool graph_modified = false;

  for (auto comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    absl::flat_hash_set<HloInstruction*> constant_32_dag;

    // Finds DAG subtrees of fp16 ops.
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      // Do not fold kIota.
      if (inst->opcode() == HloOpcode::kIota) {
        continue;
      }

      // Case A
      if (inst->opcode() == HloOpcode::kConstant &&
          inst->shape().element_type() == PrimitiveType::F16) {
        ReplaceF16InstructionByF32(inst, constant_32_dag, graph_modified);
        continue;
      }

      bool at_least_one_operand_in_const32_set =
          absl::c_any_of(inst->operands(), [&](const HloInstruction* inst) {
            return constant_32_dag.contains(inst);
          });

      // Case B
      if (CaseBAllOperandsInConst32Set(inst, constant_32_dag)) {
        ReplaceF16InstructionByF32(inst, constant_32_dag, graph_modified);
      }
      // Case C
      else if (at_least_one_operand_in_const32_set) {
        for (HloInstruction* operand_inst : inst->operands()) {
          if (constant_32_dag.contains(operand_inst)) {
            Shape f16_shape_operand_inst = ShapeUtil::ChangeElementType(
                operand_inst->shape(), PrimitiveType::F16);

            HloInstruction* operand_inst_conv =
                comp->AddInstruction(HloInstruction::CreateConvert(
                    f16_shape_operand_inst, operand_inst));
            int64_t operand_index = inst->operand_index(operand_inst);
            TF_RETURN_IF_ERROR(
                inst->ReplaceOperandWith(operand_index, operand_inst_conv));
            graph_modified = true;
          }
        }
      }
    }

    // case root in 32DAG
    auto* root = comp->root_instruction();
    if (constant_32_dag.contains(root)) {
      Shape f16_shape =
          ShapeUtil::ChangeElementType(root->shape(), PrimitiveType::F16);
      HloInstruction* root_conv =
          comp->AddInstruction(HloInstruction::CreateConvert(f16_shape, root));
      TF_RETURN_IF_ERROR(root->ReplaceAllUsesWith(root_conv));
      graph_modified = true;
    }
  }

  return graph_modified;
}

}  // namespace poplarplugin
}  // namespace xla
