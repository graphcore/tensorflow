/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/elementwise_simplifier.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/elementwise.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

HloInstruction* PreserveFrontendAttributesIfNeeded(
    HloInstruction* new_inst, const HloInstruction* old_inst) {
  if (new_inst->frontend_attributes().map().empty() &&
      !old_inst->frontend_attributes().map().empty()) {
    new_inst->set_frontend_attributes(old_inst->frontend_attributes());
  }
  return new_inst;
}

bool IsAll(const HloInstruction* op, int8 value) {
  switch (op->opcode()) {
    case HloOpcode::kBroadcast:
      return IsAll(op->operand(0), value);
    case HloOpcode::kConstant:
      return op->literal().IsAll(value);
    default:
      return false;
  }
}

// ElementwiseSimplifierVisitor traverses the HLO computation and reduces
// certain algebraic expressions to custom calls. Note: This only supports
// simplifications that simply look at the operands of an instruction.
class ElementwiseSimplifierVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleDivide(HloInstruction* divide) override;
  Status HandleMultiply(HloInstruction* multiply) override;

  // Runs the visitor on a computation.
  bool Run(HloComputation* computation);

 private:
  // Useful when we want to use the same visitor over multiple computations.
  void ResetState(HloComputation* computation);

  // Current HloComputation instance the ElementwiseSimplifierVisitor is
  // traversing.
  HloComputation* computation_;
};
}  // namespace

void ElementwiseSimplifierVisitor::ResetState(HloComputation* computation) {
  changed_ = false;
  ResetVisitStates();
  computation_ = computation;
}

bool ElementwiseSimplifierVisitor::Run(HloComputation* computation) {
  ResetState(computation);
  TF_CHECK_OK(computation->Accept(this));
  return changed();
}

Status ElementwiseSimplifierVisitor::HandleDivide(HloInstruction* divide) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(divide, m::Divide(m::Op(&lhs), m::Op(&rhs))));
  // 1 / A => Inverse(A)
  if (IsAll(lhs, 1)) {
    HloInstruction* inverse = PreserveFrontendAttributesIfNeeded(
        computation_->AddInstruction(CreateInverse(rhs)), divide);
    TF_RETURN_IF_ERROR(ReplaceInstruction(divide, inverse));
  }
  return Status::OK();
}

Status ElementwiseSimplifierVisitor::HandleMultiply(HloInstruction* multiply) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(multiply, m::Multiply(m::Op(&lhs), m::Op(&rhs))));
  // A * A => Square(A)
  if (lhs == rhs) {
    HloInstruction* square = PreserveFrontendAttributesIfNeeded(
        computation_->AddInstruction(CreateSquare(lhs)), multiply);
    TF_RETURN_IF_ERROR(ReplaceInstruction(multiply, square));
  }
  return Status::OK();
}

StatusOr<bool> ElementwiseSimplifier::Run(HloModule* module) {
  XLA_VLOG_LINES(
      2, "ElementwiseSimplifier::Run(), before:\n" + module->ToString());
  bool changed = false;
  ElementwiseSimplifierVisitor visitor;
  for (auto comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    if (visitor.Run(comp)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(2,
                 "ElementwiseSimplifier::Run(), after:\n" + module->ToString());
  return changed;
}
}  // namespace poplarplugin
}  // namespace xla
