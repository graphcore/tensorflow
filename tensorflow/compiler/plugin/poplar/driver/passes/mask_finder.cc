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
#include "tensorflow/compiler/plugin/poplar/driver/passes/mask_finder.h"

#include <queue>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace poplarplugin {

namespace {

using InstructionsToOutline = std::vector<HloInstruction*>;
using InstructionsToOutlineSet = absl::flat_hash_set<HloInstruction*>;

// Collect all select condition instructions and broadcast operand for
// outlining.
InstructionsToOutlineSet GetInstructionsToOutline(HloInstruction* select) {
  InstructionsToOutlineSet instructions_to_outline;

  instructions_to_outline.insert(select);

  if (IsConstantBroadcast(select->operand(1))) {
    HloInstruction* bcast = select->mutable_operand(1);
    instructions_to_outline.insert(bcast);
    instructions_to_outline.insert(bcast->mutable_operand(0));
  }
  if (IsConstantBroadcast(select->operand(2))) {
    HloInstruction* bcast = select->mutable_operand(2);
    instructions_to_outline.insert(bcast);
    instructions_to_outline.insert(bcast->mutable_operand(0));
  }

  std::queue<HloInstruction*> to_visit;
  to_visit.push(select->mutable_operand(0));

  while (!to_visit.empty()) {
    HloInstruction* inst = to_visit.front();
    to_visit.pop();
    if (instructions_to_outline.contains(inst)) {
      continue;
    }
    instructions_to_outline.insert(inst);
    VLOG(10) << "Visiting " << inst->name();

    bool valid = false;
    switch (inst->opcode()) {
      case HloOpcode::kBroadcast:
      case HloOpcode::kConcatenate:
      case HloOpcode::kDot:
      case HloOpcode::kIota:
      case HloOpcode::kReduce:
      case HloOpcode::kReshape:
      case HloOpcode::kTranspose:
        valid = true;
        break;
      case HloOpcode::kCustomCall:
        valid = false;
        break;
      default:
        valid = inst->IsElementwise() && !inst->HasSideEffect();
        break;
    }
    if (!valid) {
      VLOG(10) << "Found invalid instruction " << inst->name();
      return {};
    } else {
      instructions_to_outline.insert(inst);
      for (HloInstruction* op : inst->operands()) {
        to_visit.push(op);
      }
    }
  }
  return instructions_to_outline;
}

// Now check every outlined instruction for users outside of the outlined
// computation. Clone instructions outside the computation.
Status CloneUsersOutsideOutlinedInstructions(
    HloInstruction* select,
    const InstructionsToOutline& instructions_to_outline,
    const InstructionsToOutlineSet& instructions_to_outline_set) {
  HloComputation* comp = select->parent();
  HloCloneContext clones(comp->parent());
  bool found_outside_user = true;
  // Cloning instruction for outside user requires cloning all operands which
  // are inside the outlined computation, because this instruction becomes
  // outside user for them. Instead of recursive cloning we can keep cloning
  // outside users until every outside user operands are cloned.
  while (found_outside_user) {
    found_outside_user = false;
    for (HloInstruction* inst : instructions_to_outline) {
      if (inst == select) {
        continue;
      }
      VLOG(10) << "Checking instruction " << inst->name()
               << ", users: " << inst->user_count();
      auto users = inst->users();
      for (HloInstruction* user : users) {
        if (!instructions_to_outline_set.contains(user)) {
          found_outside_user = true;
          HloInstruction* clone = clones.FindInstruction(inst);
          if (!clone) {
            clone = comp->AddInstruction(inst->Clone(""));
            VLOG(1) << "Cloned " << inst->name() << " into " << clone->name();
            clones.MapInstruction(inst, clone);
          }
          TF_RETURN_IF_ERROR(inst->ReplaceUseWith(user, clone));
        }
      }
    }
  }
  return Status::OK();
}

StatusOr<HloInstruction*> RewriteMask(HloInstruction* select) {
  VLOG(10) << "Inspecting select " << select->name();
  if (ShapeUtil::IsEffectiveScalar(select->shape())) {
    VLOG(10) << "Ignoring scalar selects.";
    return nullptr;
  }
  const bool op_1_bcast = IsConstantBroadcast(select->operand(1));
  const bool op_2_bcast = IsConstantBroadcast(select->operand(2));
  if ((!op_1_bcast && !op_2_bcast) || (op_1_bcast && op_2_bcast)) {
    VLOG(10) << "Expect one of select operands to be constant broadcast.";
    return nullptr;
  }

  auto instructions_to_outline_set = GetInstructionsToOutline(select);
  if (instructions_to_outline_set.empty()) {
    return nullptr;
  }

  int64_t rank = select->shape().rank();
  VLOG(3) << "Found valid select, rewriting...";

  // Create topologically sorted instruction vector for outlining.
  std::vector<HloInstruction*> instructions_to_outline;
  instructions_to_outline.reserve(instructions_to_outline_set.size());
  HloComputation* comp = select->parent();
  absl::c_copy_if(comp->MakeInstructionPostOrder(),
                  std::back_inserter(instructions_to_outline),
                  [&](const HloInstruction* inst) {
                    return instructions_to_outline_set.contains(inst);
                  });
  CHECK_EQ(instructions_to_outline.size(), instructions_to_outline_set.size());

  TF_RETURN_IF_ERROR(CloneUsersOutsideOutlinedInstructions(
      select, instructions_to_outline, instructions_to_outline_set));

  HloInstruction* fusion = OutlineExpressionFromComputationWithFusion(
      instructions_to_outline, "_pop_op_mask", select->parent(), {},
      /*replace=*/false);

  TF_RETURN_IF_ERROR(SetPoplarUserDescriptions(
      fusion,
      UseDescriptionsSimpleNoTuple0thOperandAliasing(
          fusion, BufferUseKind::USE_ALIAS_READ_WRITE),
      /*allow_non_inplace=*/true));
  return fusion;
}
}  // namespace

StatusOr<bool> MaskFinder::Run(HloModule* module) {
  VLOG(2) << "Before the MaskFinder:";
  XLA_VLOG_LINES(2, module->ToString());

  std::vector<HloInstruction*> selects;
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kSelect) {
        selects.push_back(inst);
      }
    }
  }

  bool changed = false;
  std::vector<std::pair<HloInstruction*, HloInstruction*>> masks;
  for (HloInstruction* select : selects) {
    TF_ASSIGN_OR_RETURN(HloInstruction * fusion, RewriteMask(select));
    if (fusion) {
      changed = true;
      masks.emplace_back(select, fusion);
    }
  }
  if (changed) {
    for (auto& mask : masks) {
      TF_RETURN_IF_ERROR(mask.first->ReplaceAllUsesWith(mask.second));
    }
  }

  if (changed) {
    VLOG(2) << "After the MaskFinder:";
    XLA_VLOG_LINES(2, module->ToString());
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
