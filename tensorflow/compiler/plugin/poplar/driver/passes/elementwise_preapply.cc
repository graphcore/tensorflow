/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/elementwise_preapply.h"

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace poplarplugin {
namespace {

using HandlerFunc =
    std::function<StatusOr<bool>(HloInstruction*, HloInstruction*)>;
static absl::flat_hash_map<HloOpcode, HandlerFunc> op_handlers;
static absl::flat_hash_map<PoplarOp, HandlerFunc> poplar_op_handlers;

#define REGISTER_OP_HANDLER(opcode, handler)                               \
  static HandlerFunc _op_handler_##opcode(op_handlers[HloOpcode::opcode] = \
                                              handler);

#define REGISTER_POPLAR_OP_HANDLER(poplar_op, handler) \
  static HandlerFunc _poplar_op_handler_##poplar_op(   \
      poplar_op_handlers[PoplarOp::poplar_op] = handler);

absl::optional<HandlerFunc> GetHandler(const HloInstruction* inst) {
  if (op_handlers.contains(inst->opcode())) {
    return op_handlers[inst->opcode()];
  }
  for (auto pair : poplar_op_handlers) {
    if (IsPoplarInstruction(pair.first)(inst)) {
      return pair.second;
    }
  }
  return absl::nullopt;
}

StatusOr<bool> TryHandle(HloInstruction* inst) {
  if (inst->parent()->IsMarkedAsDead(inst)) {
    // Skip instructions which have been removed from the conputation.
    return false;
  }

  // Find handler.
  absl::optional<HandlerFunc> handler = GetHandler(inst);
  if (!handler.has_value()) {
    return false;
  }

  // Find single uniform elementwise user.
  if (inst->user_count() != 1) {
    return false;
  }
  HloInstruction* user = inst->users()[0];
  if (!user->IsElementwise() || user->opcode() == HloOpcode::kCopy) {
    // This optimisation does not make sense for copies even though they are
    // elementwise.
    return false;
  }
  for (const HloInstruction* operand : user->operands()) {
    // An elementwise op is uniform on inst if all of its operands are either
    // inst, or broadcast scalars.
    if (!(operand == inst || (operand->opcode() == HloOpcode::kBroadcast &&
                              IsScalar(operand->operand(0))))) {
      return false;
    }
  }

  return handler.value()(inst, user);
}

StatusOr<bool> HandleOneHot(HloInstruction* inst, HloInstruction* elementwise) {
  HloComputation* comp = inst->parent();
  HloInstruction* on = inst->mutable_operand(1);
  HloInstruction* off = inst->mutable_operand(2);

  // Preapply elementwise to on and off values of one-hot.
  std::vector<HloInstruction*> on_operands;
  std::vector<HloInstruction*> off_operands;
  for (HloInstruction* op : elementwise->operands()) {
    if (op == inst) {
      on_operands.push_back(on);
      off_operands.push_back(off);
    } else {
      CHECK(op->opcode() == HloOpcode::kBroadcast);
      CHECK(IsScalar(op->operand(0)));
      on_operands.push_back(op->mutable_operand(0));
      off_operands.push_back(op->mutable_operand(0));
    }
  }
  HloInstruction* on_transformed = comp->AddInstruction(
      elementwise->CloneWithNewOperands(on->shape(), on_operands));
  HloInstruction* off_transformed = comp->AddInstruction(
      elementwise->CloneWithNewOperands(off->shape(), off_operands));
  TF_RETURN_IF_ERROR(inst->ReplaceOperandWith(1, on_transformed));
  TF_RETURN_IF_ERROR(inst->ReplaceOperandWith(2, off_transformed));

  // Remove elementwise and use one-hot directly.
  TF_RETURN_IF_ERROR(inst->CopyAllControlDepsFrom(elementwise));
  TF_RETURN_IF_ERROR(elementwise->ReplaceAllUsesWith(inst));
  TF_RETURN_IF_ERROR(comp->RemoveInstructionAndUnusedOperands(elementwise));
  return true;
}
REGISTER_POPLAR_OP_HANDLER(OneHot, HandleOneHot);

}  // anonymous namespace

StatusOr<bool> ElementwisePreapply::Run(HloModule* module) {
  VLOG(2) << "Before the ElementwisePreapply:";
  XLA_VLOG_LINES(2, module->ToString());
  bool changed = false;

  for (auto comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool changed_, TryHandle(inst));
      changed |= changed_;
    }
  }

  if (changed) {
    VLOG(2) << "After the ElementwisePreapply:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "There were no changes.";
  }
  return changed;
}
}  // namespace poplarplugin
}  // namespace xla
