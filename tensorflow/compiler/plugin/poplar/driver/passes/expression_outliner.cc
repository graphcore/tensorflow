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

bool IsSupportedElementwise(const HloInstruction* inst) {
  switch (inst->opcode()) {
    // Unary
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kCbrt:
    case HloOpcode::kCeil:
    case HloOpcode::kClz:
    case HloOpcode::kConvert:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLogistic:
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
    default:
      return false;
  }
}
}  // namespace

ExpressionOutliner::ExpressionOutliner(int64_t maximum_num_elements)
    : maximum_num_elements_(maximum_num_elements) {}

StatusOr<bool> ExpressionOutliner::ModuleExpressionOutliner(
    HloComputation* comp) {
  std::list<HloInstruction*> all_ops;
  for (auto* inst : comp->MakeInstructionPostOrder()) {
    if (!IsSupportedElementwise(inst)) {
      continue;
    }

    // Skip if:
    // * Instruction has more than one user - Poplibs maps only really support
    // trees.
    // * Instruction is lowered inplace.
    // * Instruction has control deps.
    // * Instruction has more elements than the maximum (-1 means no maximum).
    if (inst->user_count() != 1 || IsLoweredInplace(inst) ||
        inst->control_predecessors().size() ||
        inst->control_successors().size() ||
        (maximum_num_elements_ >= 0 &&
         ShapeUtil::ElementsIn(inst->shape()) > maximum_num_elements_)) {
      continue;
    }

    all_ops.push_front(inst);
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
          instructions_to_outline, "_pop_op_arithmetic_expression", comp);

      was_outlined = true;

      if (has_sharding) {
        fusion->set_sharding(sharding);
      }
    }
  }

  return was_outlined;
}

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
