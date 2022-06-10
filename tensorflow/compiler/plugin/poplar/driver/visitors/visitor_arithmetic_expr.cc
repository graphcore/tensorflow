/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 */

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_arithmetic_expr.h"

#include <map>
#include <string>
#include <utility>

#include <snap/popops/ElementWise.hpp>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {

ArithmeticExprVisitor::ArithmeticExprVisitor(
    CompilerResources& res, const TensorVectors& inputs,
    const HloInstruction* caller,
    const poplar::DebugNameAndId& debug_name_and_id)
    : BaseVisitor(res, debug_name_and_id),
      inputs_(std::move(inputs)),
      caller_(caller) {
  // We don need to change the seed as we visit instructions, since we will not
  // be executing them individually but as fused unit via a single poplar call.
  // Hence the seed we need to use will be set by the calling
  // instruction/visitor.
  allow_seed_changes_ = false;
}

StatusOr<std::unique_ptr<popops::expr::Expr>>
ArithmeticExprVisitor::FindExpressionInput(const HloInstruction* inst) {
  // When finding an expression, need to differentiate between parameters and
  // actual expressions
  if (inst->opcode() == HloOpcode::kParameter) {
    // Find the input tensor - tuples are not supported
    DriverTensor& in0 = inputs_[inst->parameter_number()][0];
    // Check if an expression exists
    if (tensors_map_.count(&in0) == 0) {
      // If the tensor has not been mapped yet
      // add it to tensor inputs
      ts_.push_back(in0);
      // and create an expression
      tensors_map_[&in0] = std::unique_ptr<popops::expr::PlaceHolder>(
          new popops::expr::PlaceHolder(ts_.size()));
    }
    return tensors_map_[&in0]->clone();
  } else {
    auto it = expressions_map_.find(inst);
    if (it == expressions_map_.end()) {
      return tensorflow::errors::Unknown(
          StrCat("[Poplar] Couldn't find expression for %s", inst->name()));
    }
    return it->second->clone();
  }
}

Status ArithmeticExprVisitor::HandleElementwiseUnary(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  // find the op
  TF_ASSIGN_OR_RETURN(popops::expr::UnaryOpType op, LookupUnaryFn(inst));

  // get the input
  TF_ASSIGN_OR_RETURN(auto in, FindExpressionInput(inst->operand(0)));

  // create new expression
  expressions_map_[inst] = std::unique_ptr<popops::expr::UnaryOp>(
      new popops::expr::UnaryOp(op, *in.get()));
  return Status::OK();
}

Status ArithmeticExprVisitor::HandleCustomCall(HloInstruction* inst) {
  CHECK_EQ(inst->operand_count(), 1);
  return HandleElementwiseUnary(inst);
}

Status ArithmeticExprVisitor::HandleElementwiseBinary(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  // find the op
  TF_ASSIGN_OR_RETURN(popops::expr::BinaryOpType op, LookupBinaryFn(inst));

  // get the inputs
  TF_ASSIGN_OR_RETURN(auto lhs, FindExpressionInput(inst->operand(0)));
  TF_ASSIGN_OR_RETURN(auto rhs, FindExpressionInput(inst->operand(1)));

  // create new expression
  expressions_map_[inst] = std::unique_ptr<popops::expr::BinaryOp>(
      new popops::expr::BinaryOp(op, *lhs.get(), *rhs.get()));
  return Status::OK();
}

Status ArithmeticExprVisitor::HandleCompare(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  // find the op
  TF_ASSIGN_OR_RETURN(popops::expr::BinaryOpType op, LookupComparisonFn(inst));

  // get the inputs
  TF_ASSIGN_OR_RETURN(auto lhs, FindExpressionInput(inst->operand(0)));
  TF_ASSIGN_OR_RETURN(auto rhs, FindExpressionInput(inst->operand(1)));

  // create new expression
  expressions_map_[inst] = std::unique_ptr<popops::expr::BinaryOp>(
      new popops::expr::BinaryOp(op, *lhs.get(), *rhs.get()));
  return Status::OK();
}

Status ArithmeticExprVisitor::HandleConvert(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  TF_ASSIGN_OR_RETURN(poplar::Type poplar_type, PoplarDataType(inst->shape()));

  // Get the input.
  TF_ASSIGN_OR_RETURN(auto in, FindExpressionInput(inst->operand(0)));

  // Create new expression.
  expressions_map_[inst] = std::unique_ptr<popops::expr::Cast>(
      new popops::expr::Cast(*in.get(), poplar_type));
  return Status::OK();
}

Status ArithmeticExprVisitor::HandleSelect(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  // set the op
  const popops::expr::TernaryOpType op = popops::expr::TernaryOpType::SELECT;

  TF_ASSIGN_OR_RETURN(auto pred, FindExpressionInput(inst->operand(0)));
  TF_ASSIGN_OR_RETURN(auto in0, FindExpressionInput(inst->operand(1)));
  TF_ASSIGN_OR_RETURN(auto in1, FindExpressionInput(inst->operand(2)));
  // create new expression
  expressions_map_[inst] = std::unique_ptr<popops::expr::TernaryOp>(
      new popops::expr::TernaryOp(op, *in0.get(), *in1.get(), *pred.get()));
  return Status::OK();
}

Status ArithmeticExprVisitor::HandleClamp(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  // set the op
  const popops::expr::TernaryOpType op = popops::expr::TernaryOpType::CLAMP;

  TF_ASSIGN_OR_RETURN(auto min, FindExpressionInput(inst->operand(0)));
  TF_ASSIGN_OR_RETURN(auto arg, FindExpressionInput(inst->operand(1)));
  TF_ASSIGN_OR_RETURN(auto max, FindExpressionInput(inst->operand(2)));

  // create new expression
  expressions_map_[inst] = std::unique_ptr<popops::expr::TernaryOp>(
      new popops::expr::TernaryOp(op, *arg.get(), *min.get(), *max.get()));
  return Status::OK();
}

Status ArithmeticExprVisitor::HandleParameter(HloInstruction* inst) {
  // TODO ArithmeticExprVisitor does not support tuples
  if (inputs_[inst->parameter_number()].size() > 1)
    return xla::Unimplemented(
        "Support for tuples in outlined arithmetic expressions is not "
        "implemented");
  return Status::OK();
}

Status ArithmeticExprVisitor::FinishScopedVisit(HloInstruction* inst) {
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  auto& graph = GetGraph(resources_, caller_);
  DriverProgramSequence seq(graph, debug_name_and_id);

  // get the expression
  TF_ASSIGN_OR_RETURN(auto expr, FindExpressionInput(inst));
  // map expression with the tensors
  DriverTensor out = snap::popops::map(graph, *expr, GetSnapTensors(ts_), seq,
                                       {debug_name_and_id, "expression"});
  outputs_.push_back(out);

  resources_.tensor_maps.AddTensorMapForComputation(inst->parent()->name(),
                                                    std::move(tensor_map));

  TF_RETURN_IF_ERROR(AddSequenceForInstruction(inst, seq));

  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla
