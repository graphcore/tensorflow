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

#include "tensorflow/compiler/plugin/poplar/driver/passes/expression_outliner.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/bcast.h"

namespace xla {
namespace poplarplugin {

using InplaceSet = std::set<const HloInstruction*>;

namespace {

bool IsPopopsElementwise(const HloInstruction* inst) {
  switch (inst->opcode()) {
    // Unary
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kCeil:
    case HloOpcode::kClz:
    case HloOpcode::kConvert:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kTanh:
    // Binary
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
    case HloOpcode::kCompare:
    case HloOpcode::kDivide:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kSubtract:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:

    // Ternary
    case HloOpcode::kSelect:
      return !inst->shape().IsTuple();
    case HloOpcode::kClamp:
      return true;
    // Ops not supported in Expressions
    // Unary
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kImag:
    case HloOpcode::kReal:
    case HloOpcode::kReducePrecision:
    // Binary
    case HloOpcode::kComplex:
      return false;
    default:
      return false;
  }
}

StatusOr<bool> ModuleExpressionOutliner(HloComputation* comp) {
  std::list<HloInstruction*> all_ops;
  for (auto* inst : comp->MakeInstructionPostOrder()) {
    if (IsPopopsElementwise(inst) && inst->user_count() == 1 &&
        !IsLoweredInplace(inst) && inst->control_predecessors().size() == 0 &&
        inst->control_successors().size() == 0) {
      bool add_op = true;
      if (inst->IsElementwiseBinary()) {
        // for BinaryOps check the shapes of inputs match
        const HloInstruction* in0 = inst->operand(0);
        const HloInstruction* in1 = inst->operand(1);
        const bool input_shapes_match =
            ShapeUtil::Equal(in0->shape(), in1->shape());
        if (!input_shapes_match) {
          // if shapes don't match check that they can be broadcasted to the
          // same shape
          auto shape0_optional =
              convert_array<tensorflow::BCast::Vec>(in0->shape().dimensions());
          auto shape1_optional =
              convert_array<tensorflow::BCast::Vec>(in1->shape().dimensions());
          if (!shape0_optional || !shape1_optional) {
            return xla::FailedPrecondition(
                "ExpressionOutliner - cannot cast input shape.");
          }
          tensorflow::BCast::Vec shape0 = *shape0_optional;
          tensorflow::BCast::Vec shape1 = *shape1_optional;

          const bool valid_bcast = tensorflow::BCast(shape0, shape1).IsValid();
          if (!valid_bcast) {
            add_op = false;
          }
        }
      } else if (inst->opcode() == HloOpcode::kClamp) {
        // don't add ClampOps for which inputs don't have the same shape as
        // output
        const bool shapes_match =
            ShapeUtil::Equal(inst->shape(), inst->operand(0)->shape()) &&
            ShapeUtil::Equal(inst->shape(), inst->operand(1)->shape()) &&
            ShapeUtil::Equal(inst->shape(), inst->operand(2)->shape());
        if (!shapes_match) {
          add_op = false;
        }
      } else if (inst->opcode() == HloOpcode::kSelect) {
        const HloInstruction* pred = inst->operand(0);
        const HloInstruction* in0 = inst->operand(1);
        const HloInstruction* in1 = inst->operand(2);
        // for Elementwise Select, predicate has to be scalar
        const bool pred_scalar = ShapeUtil::ElementsIn(pred->shape()) == 1;
        // or match the shape with the inputs
        const bool shapes_match =
            ShapeUtil::Equal(pred->shape(), in0->shape()) &&
            ShapeUtil::Equal(pred->shape(), in1->shape());
        if (!(pred_scalar || shapes_match)) {
          add_op = false;
        }
      }
      if (add_op) {
        all_ops.push_front(inst);
      }
    }
  }

  bool was_outlined = false;

  while (all_ops.size() > 0) {
    HloInstruction* root = all_ops.front();
    all_ops.pop_front();
    std::vector<HloInstruction*> instructions_to_outline;

    std::list<HloInstruction*> potential_list;
    std::set<HloInstruction*> potential_set;

    std::set<HloInstruction*> outlined;

    potential_list.push_back(root);

    while (potential_list.size() > 0) {
      HloInstruction* inst = potential_list.front();
      potential_list.pop_front();
      potential_set.erase(inst);
      auto current = std::find(instructions_to_outline.begin(),
                               instructions_to_outline.end(), inst);
      if (current != instructions_to_outline.end()) {
        instructions_to_outline.erase(current);
      }
      instructions_to_outline.push_back(inst);
      outlined.insert(inst);

      for (auto* op : inst->operands()) {
        bool ok_to_outline =
            (std::find(all_ops.begin(), all_ops.end(), op) != all_ops.end());

        if (IsLoweredInplace(op)) {
          ok_to_outline = false;
        }

        if (inst->has_sharding() && op->has_sharding()) {
          if (inst->sharding() != op->sharding()) {
            ok_to_outline = false;
          }
        }

        bool all_users_ok = true;
        for (auto* user : op->users()) {
          all_users_ok &=
              ((potential_set.count(user) > 0) || (outlined.count(user) > 0));
        }
        if (ok_to_outline && all_users_ok) {
          if (potential_set.count(op) == 0) {
            potential_list.push_back(op);
            potential_set.insert(op);
          }
        }
      }
    }

    for (auto* inst : instructions_to_outline) {
      all_ops.remove(inst);
    }

    if (instructions_to_outline.size() > 1) {
      HloSharding sharding(HloSharding::AssignDevice(0));
      bool has_sharding = false;
      if (instructions_to_outline[0]->has_sharding()) {
        sharding = instructions_to_outline[0]->sharding();
        has_sharding = true;
      }

      std::reverse(instructions_to_outline.begin(),
                   instructions_to_outline.end());

      auto* fusion = OutlineExpressionFromComputationWithFusion(
          instructions_to_outline, "_arithmetic_expression", comp);

      was_outlined = true;

      if (has_sharding) {
        fusion->set_sharding(sharding);
      }
    }
  }

  return was_outlined;
}

}  // namespace

StatusOr<bool> ExpressionOutliner::Run(HloModule* module) {
  bool was_outlined = false;
  for (auto comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(bool was_modified, ModuleExpressionOutliner(comp));
    if (was_modified) {
      was_outlined = true;
    }
  }

  return was_outlined;
}

}  // namespace poplarplugin
}  // namespace xla
