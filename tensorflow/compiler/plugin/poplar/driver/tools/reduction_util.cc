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
#include "tensorflow/compiler/plugin/poplar/driver/tools/reduction_util.h"

#include <popops/ExprOp.hpp>
#include <popops/Operation.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace poplarplugin {

ReductionInfo::ReductionInfo(const ReductionInfo& info)
    : reduction_dims(info.reduction_dims),
      identity_literal(info.identity_literal.Clone()),
      reduction_op(info.reduction_op) {}

StatusOr<popops::Operation> ToPopopsReductionOp(
    const ReductionOperation& op_type) {
  switch (op_type) {
    case ReductionOperation::ADD:
      return popops::Operation::ADD;
    case ReductionOperation::MUL:
      return popops::Operation::MUL;
    case ReductionOperation::MIN:
      return popops::Operation::MIN;
    case ReductionOperation::MAX:
      return popops::Operation::MAX;
    case ReductionOperation::LOGICAL_AND:
      return popops::Operation::LOGICAL_AND;
    case ReductionOperation::LOGICAL_OR:
      return popops::Operation::LOGICAL_OR;
    case ReductionOperation::SQUARE_ADD:
      return popops::Operation::SQUARE_ADD;
    case ReductionOperation::LOG_ADD:
      return popops::Operation::LOG_ADD;
    default:
      return xla::FailedPrecondition("Invalid operation type: %d", op_type);
  }
}

StatusOr<ReductionOperation> FromPopopsReductionOps(
    const popops::Operation& op_type) {
  switch (op_type) {
    case popops::Operation::ADD:
      return ReductionOperation::ADD;
    case popops::Operation::MUL:
      return ReductionOperation::MUL;
    case popops::Operation::MIN:
      return ReductionOperation::MIN;
    case popops::Operation::MAX:
      return ReductionOperation::MAX;
    case popops::Operation::LOGICAL_AND:
      return ReductionOperation::LOGICAL_AND;
    case popops::Operation::LOGICAL_OR:
      return ReductionOperation::LOGICAL_OR;
    case popops::Operation::SQUARE_ADD:
      return ReductionOperation::SQUARE_ADD;
    case popops::Operation::LOG_ADD:
      return ReductionOperation::LOG_ADD;
    default:
      return xla::FailedPrecondition("Invalid operation type: %d", op_type);
  }
}

StatusOr<popops::expr::BinaryOpType> ToBinaryOpType(
    const ReductionOperation& op_type) {
  switch (op_type) {
    case ReductionOperation::ADD:
      return popops::expr::BinaryOpType::ADD;
    case ReductionOperation::MUL:
      return popops::expr::BinaryOpType::MULTIPLY;
    case ReductionOperation::MIN:
      return popops::expr::BinaryOpType::MINIMUM;
    case ReductionOperation::MAX:
      return popops::expr::BinaryOpType::MAXIMUM;
    case ReductionOperation::LOGICAL_AND:
      return popops::expr::BinaryOpType::LOGICAL_AND;
    case ReductionOperation::LOGICAL_OR:
      return popops::expr::BinaryOpType::LOGICAL_OR;
    case ReductionOperation::SQUARE_ADD:
      return popops::expr::BinaryOpType::ADD;
    case ReductionOperation::LOG_ADD:
      return popops::expr::BinaryOpType::ADD;
    default:
      return xla::FailedPrecondition("Invalid operation type: %d", op_type);
  }
}

StatusOr<ReductionInfo> GetReductionInfo(const HloInstruction* inst) {
  ReductionInfo info;

  // Find reduction operation.
  // GetPoplibsReductionOperation handles reduce fusions properly.
  TF_ASSIGN_OR_RETURN(const auto poplibs_reduction_op,
                      GetPoplibsReductionOperation(inst));
  TF_ASSIGN_OR_RETURN(info.reduction_op,
                      FromPopopsReductionOps(poplibs_reduction_op));
  const HloInstruction* root;
  if (inst->opcode() == HloOpcode::kFusion) {
    const HloInstruction* fusion_inner_reduce_inst =
        inst->fused_instructions_computation()->root_instruction();
    root = fusion_inner_reduce_inst->to_apply()->root_instruction();
    // Find reduction dims.
    info.reduction_dims.assign(fusion_inner_reduce_inst->dimensions().begin(),
                               fusion_inner_reduce_inst->dimensions().end());
  } else {
    root = inst->to_apply()->root_instruction();
    // Find reduction dims.
    info.reduction_dims.assign(inst->dimensions().begin(),
                               inst->dimensions().end());
  }
  // Find identity literal.
  info.identity_literal =
      GetIdentityConstantLiteral(root, inst->shape().element_type());

  return info;
}

StatusOr<popops::Operation> GetPoplibsReductionOperation(
    const HloInstruction* inst) {
  const HloComputation* reduction_comp;

  switch (inst->opcode()) {
    case HloOpcode::kFusion:
      if (IsPopOpsFusion(inst, "reduction_fp16_input") ||
          IsPopOpsFusion(inst, "reduction_scaled")) {
        reduction_comp = inst->fused_instructions_computation()
                             ->root_instruction()
                             ->to_apply();
      } else if (IsPopOpsFusion(inst, "reduction_square_add")) {
        return popops::Operation::SQUARE_ADD;
      } else {
        return xla::FailedPrecondition("Unsupported reduce fusion: %s",
                                       inst->ToString());
      }
      break;

    case HloOpcode::kSelectAndScatter:
      reduction_comp = inst->scatter();
      break;

    case HloOpcode::kReduce:
      reduction_comp = inst->to_apply();
      break;

    default:
      return xla::FailedPrecondition(
          "Instruction has unsupported opcode for reduction: %s",
          inst->opcode());
  }

  switch (reduction_comp->root_instruction()->opcode()) {
    case HloOpcode::kAdd:
      return popops::Operation::ADD;
    case HloOpcode::kMultiply:
      return popops::Operation::MUL;
    case HloOpcode::kMaximum:
      return popops::Operation::MAX;
    case HloOpcode::kMinimum:
      return popops::Operation::MIN;
    case HloOpcode::kAnd:
      return popops::Operation::LOGICAL_AND;
    case HloOpcode::kOr:
      return popops::Operation::LOGICAL_OR;
    default:
      return xla::FailedPrecondition(
          "Unsupported fusion root (%s) from fusion (%s)",
          reduction_comp->root_instruction()->ToString(), inst->ToString());
  }
}

Literal GetIdentityConstantLiteral(const HloInstruction* inst,
                                   const PrimitiveType& dtype) {
  switch (inst->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    default:
      return LiteralUtil::Zero(dtype);
    case HloOpcode::kMultiply:
    case HloOpcode::kOr:
      return LiteralUtil::One(dtype);
    case HloOpcode::kMaximum:
      return LiteralUtil::MinValue(dtype);
    case HloOpcode::kMinimum:
      return LiteralUtil::MaxValue(dtype);
    case HloOpcode::kCompare:
      switch (inst->comparison_direction()) {
        case ComparisonDirection::kGe:
        case ComparisonDirection::kGt:
          return LiteralUtil::MinValue(dtype);
        case ComparisonDirection::kLe:
        case ComparisonDirection::kLt:
          return LiteralUtil::MaxValue(dtype);
        default:
          return LiteralUtil::Zero(dtype);
      }
  }
}

}  // namespace poplarplugin
}  // namespace xla
