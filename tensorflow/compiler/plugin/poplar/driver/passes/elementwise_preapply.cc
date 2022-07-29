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

bool HasSize1(const HloInstruction* inst) {
  return std::all_of(inst->shape().dimensions().begin(),
                     inst->shape().dimensions().end(),
                     [](auto dim) { return dim == 1; });
}

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
    // An elementwise op is uniform on inst if all of its operands are one of
    // inst, have size 1, or are a broadcast of a variable with size 1.
    if (!(operand == inst || HasSize1(operand) ||
          (operand->opcode() == HloOpcode::kBroadcast &&
           HasSize1(operand->operand(0))))) {
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
      CHECK(HasSize1(op) || (op->opcode() == HloOpcode::kBroadcast &&
                             HasSize1(op->operand(0))));
      HloInstruction* new_elementwise_operand;
      if (op->opcode() == HloOpcode::kBroadcast) {
        new_elementwise_operand = op->mutable_operand(0);
      } else {
        new_elementwise_operand = op;
      }

      CHECK(HasSize1(new_elementwise_operand));
      if (!IsScalar(new_elementwise_operand)) {
        // Reshape operand to be a scalar.
        auto new_shape = ShapeUtil::DropDegenerateDimensions(
            new_elementwise_operand->shape());
        new_elementwise_operand = comp->AddInstruction(
            HloInstruction::CreateReshape(new_shape, new_elementwise_operand));
      }
      on_operands.push_back(new_elementwise_operand);
      off_operands.push_back(new_elementwise_operand);
    }
  }
  // Create replacements for `on`, `off` and `inst`.
  // We need to use the element_type of elementwise in case it is a convert.
  auto new_on_shape = ShapeUtil::ChangeElementType(
      on->shape(), elementwise->shape().element_type());
  HloInstruction* on_transformed = comp->AddInstruction(
      elementwise->CloneWithNewOperands(new_on_shape, on_operands));

  // We need to use the element_type of elementwise in case it is a convert.
  auto new_off_shape = ShapeUtil::ChangeElementType(
      off->shape(), elementwise->shape().element_type());
  HloInstruction* off_transformed = comp->AddInstruction(
      elementwise->CloneWithNewOperands(new_off_shape, off_operands));

  auto new_shape_inst = comp->AddInstruction(inst->CloneWithNewOperands(
      elementwise->shape(),
      {inst->mutable_operand(0), on_transformed, off_transformed}));

  // Remove elementwise and replace it with the new one-hot instruction.
  TF_RETURN_IF_ERROR(elementwise->ReplaceAllUsesWith(new_shape_inst));
  TF_RETURN_IF_ERROR(comp->RemoveInstructionAndUnusedOperands(elementwise));
  return true;
}
REGISTER_POPLAR_OP_HANDLER(OneHot, HandleOneHot);

StatusOr<bool> HandleBroadcast(HloInstruction* inst,
                               HloInstruction* elementwise) {
  HloComputation* comp = inst->parent();
  HloInstruction* target_operand = inst->mutable_operand(0);
  std::vector<HloInstruction*> operands;

  for (HloInstruction* op : elementwise->operands()) {
    if (op == inst) {
      operands.push_back(target_operand);
    } else {
      HloInstruction* new_elementwise_operand;
      if (op->opcode() == HloOpcode::kBroadcast) {
        new_elementwise_operand = op->mutable_operand(0);
      } else {
        new_elementwise_operand = op;
      }
      CHECK(HasSize1(new_elementwise_operand));
      // Check if we need to adjust the shape of new_elementwise_operand
      if (!ShapeUtil::SameDimensions(target_operand->shape(),
                                     new_elementwise_operand->shape())) {
        if (HasSize1(target_operand)) {
          // Simply reshape new_elementwise_operand to have same shape as
          // target_operand, while keeping op's type.
          auto new_shape = ShapeUtil::ChangeElementType(
              target_operand->shape(), op->shape().element_type());
          new_elementwise_operand =
              comp->AddInstruction(HloInstruction::CreateReshape(
                  new_shape, new_elementwise_operand));
        } else {
          // Make sure new_elementwise_operand is a scalar.
          if (!IsScalar(new_elementwise_operand)) {
            auto new_shape = ShapeUtil::DropDegenerateDimensions(
                new_elementwise_operand->shape());
            new_elementwise_operand =
                comp->AddInstruction(HloInstruction::CreateReshape(
                    new_shape, new_elementwise_operand));
          }
          // Broadcast new_elementwise_operand to size of target_operand.
          auto new_shape = ShapeUtil::ChangeElementType(
              target_operand->shape(), op->shape().element_type());
          new_elementwise_operand =
              comp->AddInstruction(HloInstruction::CreateBroadcast(
                  new_shape, new_elementwise_operand, {}));
        }
      }
      operands.push_back(new_elementwise_operand);
    }
  }
  // We need to use the element_type of elementwise in case it is a convert.
  Shape new_target_shape = ShapeUtil::ChangeElementType(
      target_operand->shape(), elementwise->shape().element_type());
  HloInstruction* operand_transformed = comp->AddInstruction(
      elementwise->CloneWithNewOperands(new_target_shape, operands));
  // Check if we're dealing with an unneccessary broadcast.
  if (target_operand->shape() == inst->shape()) {
    // Can remove the broadcast entirely.
    TF_RETURN_IF_ERROR(elementwise->ReplaceAllUsesWith(operand_transformed));
  } else {
    // Need to keep the broadcast.
    auto new_shape_inst = comp->AddInstruction(inst->CloneWithNewOperands(
        elementwise->shape(), {operand_transformed}));
    TF_RETURN_IF_ERROR(elementwise->ReplaceAllUsesWith(new_shape_inst));
  }
  TF_RETURN_IF_ERROR(comp->RemoveInstructionAndUnusedOperands(elementwise));
  return true;
}
REGISTER_OP_HANDLER(kBroadcast, HandleBroadcast);

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
