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
#include <memory>
#include <set>
#include <tuple>
#include <utility>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_dataflow_analysis.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {
enum class InplacePriority {
  kHigh = 0,
  kMedium = 1,
  kLow = 2,
};

using InplaceCandidates =
    std::map<InplacePriority, std::vector<HloInstruction*>>;

// Only allow loops and any calls including pipelining stages.
// Skip map/reduce computations because they are lowered as poplar expressions
// and they do not need any inplacing/copying.
bool AllowedComputation(CallGraph& call_graph, HloComputation* comp) {
  if (IsPopOpsFusion(comp)) {
    return false;
  }
  if (comp == comp->parent()->entry_computation()) {
    // Allow entry computation.
    return true;
  }

  auto callers = call_graph.GetComputationCallers(comp);
  if (callers.empty()) {
    return false;
  }

  CallContext context = GetInstructionCallContext(callers[0]->opcode());
  // Do not consider map/reduce/fusion computations.
  return context == CallContext::kSequential;
}

StatusOr<bool> ConvertToReallocatingCopy(HloInstruction* copy) {
  bool changed = false;
  TF_ASSIGN_OR_RETURN(auto clone_method, GetCopyCloneMethod(copy));
  for (auto& leaf : clone_method.leaves()) {
    switch (leaf.second) {
      case CloneMethod_PreserveOrderAndAliases: {
        leaf.second = CloneMethod_DeduceNewOrderOrPreserveAliases;
        changed = true;
        break;
      }
      case CloneMethod_PreserveOrderUnlessAliases: {
        leaf.second = CloneMethod_DeduceNewOrderOrExpandAliases;
        changed = true;
        break;
      }
      default: { break; }
    }
  }
  if (changed) {
    TF_RETURN_IF_ERROR(SetCopyCloneMethod(copy, clone_method));
  }
  return changed;
}

StatusOr<bool> ConvertToReallocatingCopies(HloInstruction* inst) {
  bool changed = false;
  if (IsRepeatLoop(inst) || (inst->opcode() == HloOpcode::kWhile &&
                             inst->operand(0)->opcode() == HloOpcode::kTuple)) {
    HloInstruction* inputs =
        IsRepeatLoop(inst) ? inst : inst->mutable_operand(0);
    for (int64_t op_idx = 0; op_idx < inputs->operand_count(); ++op_idx) {
      HloInstruction* copy = inputs->mutable_operand(op_idx);
      if (copy->opcode() == HloOpcode::kCopy) {
        TF_ASSIGN_OR_RETURN(bool copy_changed, ConvertToReallocatingCopy(copy));
        changed |= copy_changed;
      }
    }
  } else if ((inst->opcode() == HloOpcode::kWhile &&
              inst->operand(0)->opcode() == HloOpcode::kCopy)) {
    TF_ASSIGN_OR_RETURN(bool copy_changed,
                        ConvertToReallocatingCopy(inst->mutable_operand(0)));
    changed |= copy_changed;
  }
  return changed;
}

bool IsInplaceCandidate(HloInstruction* instruction,
                        const HloPoplarDataflowAnalysis& dataflow) {
  // if any of the outputs could be inplaced consider instruction as a candidate
  if (instruction->opcode() == HloOpcode::kParameter) {
    return false;  // Don't handle these later down the line
  }
  if (instruction->opcode() == HloOpcode::kCopy) {
    return false;  // Can't inplace a copy
  }
  if (IsFunction(instruction)) {
    if (GetFunctionNumberModifiedRemoteBufferInputs(instruction) +
        GetFunctionNumberUnmodifiedRemoteBufferInputs(instruction)) {
      // In this case we must inline so don't even check buffers
      return true;
    }
  }
  const auto& instruction_set = dataflow.GetInstructionBufferSet(instruction);
  const auto& buffer_sets = instruction_set.GetBufferSets();
  bool result = false;
  buffer_sets.ForEachElement(
      [&](const ShapeIndex& index, const HloPoplarBufferSet& data) {
        result |= (data.GetUseKind() > BufferUseKind::USE_NO_ALIAS);
      });
  return result;
}

std::vector<HloInstruction*> FindAllCandidates(
    HloComputation* computation, const HloPoplarDataflowAnalysis& dataflow) {
  std::vector<HloInstruction*> result;
  result.reserve(computation->instruction_count());
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    if (IsInplaceCandidate(instruction, dataflow)) {
      result.emplace_back(instruction);
    }
  }
  return result;
}

HloInstructionType GetInplaceType(
    const HloInstruction* inst,
    const HloPoplarInplaceDescription& inst_description) {
  auto result = inst_description.GetType();
  CHECK(result == HloInstructionType::kInplaceGetTupleElement ||
        result == HloInstructionType::kInplaceReadWrite ||
        result == HloInstructionType::kInplaceReadOnly)
      << "Not allowed to inplace " << static_cast<int>(result)
      << inst->ToString();
  return result;
}

InplacePriority GetPriority(const HloInstruction* inst) {
  if (inst->parent()->root_instruction() == inst) {
    return InplacePriority::kHigh;
  }
  switch (inst->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kFusion:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kMultiply:
    case HloOpcode::kSubtract:
    case HloOpcode::kAddDependency:
    case HloOpcode::kGetTupleElement:
      return InplacePriority::kHigh;
    case HloOpcode::kCustomCall:
      return InplacePriority::kMedium;
    default:
      return InplacePriority::kLow;
  }
}

int64_t GetBufferOperandsSize(
    const HloInstruction* inst,
    const HloPoplarInplaceDescription& inst_description) {
  int64_t result = 0;
  for (const int64_t index : inst_description.GetInplaceOperandIndices()) {
    const auto* inplace_operand = (inst->operand(index));
    result += GetByteSizeOfTotalShapeSafe(inplace_operand->shape());
  }
  return result;
}

using SortKeyType = std::tuple<HloInstructionType, InplacePriority, int64_t>;
SortKeyType CreateSortKey(const HloInstruction* inst) {
  auto inst_description = GetInplaceDescription(inst);
  // Prioritise instructions by how they'll be inplaced, how important they
  // are to inplace, and then by how large they are
  return std::make_tuple(GetInplaceType(inst, inst_description),
                         GetPriority(inst),
                         -1 * GetBufferOperandsSize(inst, inst_description));
}

absl::flat_hash_map<const HloInstruction*, SortKeyType> CreateKeyMap(
    const std::vector<HloInstruction*>& candidates) {
  absl::flat_hash_map<const HloInstruction*, SortKeyType> result;
  result.reserve(candidates.size());
  for (const auto* inst : candidates) {
    result.emplace(inst, CreateSortKey(inst));
  }
  return result;
}

std::vector<HloInstruction*> OrderInplaceCandidates(
    std::vector<HloInstruction*> candidates, HloComputation* comp) {
  const auto key_map = CreateKeyMap(candidates);
  absl::c_stable_sort(candidates,
                      [&](const HloInstruction* a, const HloInstruction* b) {
                        return key_map.at(a) < key_map.at(b);
                      });
  return candidates;
}

std::vector<HloInstruction*> FindInplaceCandidates(
    HloComputation* comp, const HloPoplarDataflowAnalysis& dataflow) {
  std::vector<HloInstruction*> candidates = FindAllCandidates(comp, dataflow);
  return OrderInplaceCandidates(std::move(candidates), comp);
}

bool ConstantSharedBetweenInplaceLoops(const HloInstruction* inst) {
  // Effectively scalar consts. This will accept scalars with extra
  // singular dims too. Clone wide consts because they are broadcasts of scalar.
  bool supported_const = (inst->opcode() == HloOpcode::kConstant &&
                          ShapeUtil::IsEffectiveScalar(inst->shape())) ||
                         IsWideConstant(inst) ||
                         IsPoplarInstruction(PoplarOp::Uninitialised, inst);
  if (!supported_const) {
    return false;
  }
  // Only consider constants used in while/repeat loops.
  if (!absl::c_any_of(inst->users(), [](const HloInstruction* inst) {
        return IsRepeatLoop(inst) || inst->opcode() == HloOpcode::kWhile ||
               (inst->opcode() == HloOpcode::kTuple &&
                absl::c_any_of(inst->users(), [](const HloInstruction* user) {
                  return user->opcode() == HloOpcode::kWhile;
                }));
      })) {
    return false;
  }
  // Only if there's more than one (potentially) inplace user.
  std::size_t count =
      absl::c_count_if(inst->users(), [](const HloInstruction* user) {
        return GetInplaceDescription(user).IsInplaceType();
      });
  return count > 1;
}

StatusOr<bool> CloneConstantsSharedBetweenInplaceLoops(HloModule* module) {
  bool changed = false;
  for (auto* comp : module->MakeComputationPostOrder()) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      if (!ConstantSharedBetweenInplaceLoops(inst)) {
        continue;
      }
      VLOG(3) << "Found shared constant to clone: " << inst->name();
      auto users = inst->users();
      for (auto* user : users) {
        auto indices = user->OperandIndices(inst);
        for (int64_t idx : indices) {
          TF_RETURN_IF_ERROR(user->ReplaceOperandWith(
              idx, comp->AddInstruction(inst->Clone(""))));
        }
      }
      TF_RETURN_IF_ERROR(comp->RemoveInstruction(inst));
    }
  }
  return changed;
}

// If we fail to inplace an instruction with ConvertToInplace but the
// instruction requires inplacing, we will add copies (see D52050).
void AddCopiesForFailedInplace(HloInstruction* inst, InplacingState& state) {
  // Do not add copies for root tuple in entry computation.
  if (inst == state.comp->root_instruction() &&
      state.comp->parent()->entry_computation() == state.comp) {
    return;
  }
  const auto inplace_description = GetInplaceDescription(inst);
  if (!inplace_description.IsInplaceType()) {
    return;
  }

  CHECK(!IsLoweredInplace(inst));
  const bool allow_non_inplace = inplace_description.AllowNonInplaceLowering();
  VLOG(3) << "Processing " << inst->name()
          << ", allow non-inplace: " << allow_non_inplace;
  if (allow_non_inplace) {
    // No copies required - op will be lowered as non-inplace.
    return;
  }

  // Those instructions are inplace only on remote buffers, and we can't
  // copy them, they should be always inplace.
  CHECK(!IsPoplarInstruction(PoplarOp::RemoteParameterStore, inst) &&
        !IsPoplarInstruction(PoplarOp::BufferStoreSlice, inst));

  if (inplace_description.GetType() ==
      HloInstructionType::kInplaceGetTupleElement) {
    // Instead of copying inputs, copy output.
    state.instructions_to_copy.push_back(inst);
  } else {
    state.operands_to_copy.emplace_back(inst);
    for (auto op_index : inplace_description.GetInplaceOperandIndices()) {
      auto* op = inst->operand(op_index);
      if (op->user_count() == 1 &&
          IsPoplarInstruction(PoplarOp::Uninitialised, op)) {
        VLOG(3) << "Do not copy uninitialised " << op->name();
        continue;
      }
      if (IsWideConstant(op) && op->user_count() == 1) {
        VLOG(3) << "Do not copy wide copy with unique user " << op->name();
        continue;
      }
      state.operands_to_copy.back().operands.push_back(op_index);
    }
    if (state.operands_to_copy.back().operands.empty()) {
      state.operands_to_copy.pop_back();
    }
  }
}

// Insert the copies accumulated in the inplacing state.
StatusOr<bool> InsertCopies(InplacingState& state) {
  bool changed = false;

  for (HloInstruction* inst : state.instructions_to_copy) {
    auto users = inst->users();
    HloInstruction* copy = state.comp->AddInstruction(
        HloInstruction::CreateUnary(inst->shape(), HloOpcode::kCopy, inst));
    VLOG(3) << "Adding a copy for " << inst->name();
    for (auto succ : inst->control_successors()) {
      TF_RETURN_IF_ERROR(copy->AddControlDependencyTo(succ));
    }
    inst->SetupDerivedInstruction(copy);
    for (HloInstruction* user : users) {
      TF_RETURN_IF_ERROR(inst->ReplaceUseWith(user, copy));
    }
    changed = true;
  }

  for (auto& operand_to_copy : state.operands_to_copy) {
    HloInstruction* inst = operand_to_copy.inst;
    auto inplace_description = GetInplaceDescription(inst);
    for (int64_t op_index : operand_to_copy.operands) {
      HloInstruction* op = inst->mutable_operand(op_index);
      HloInstruction* copy;
      if (op->opcode() == HloOpcode::kCopy && op->user_count() == 1) {
        // Reuse copy used only for this input.
        copy = op;
      } else {
        VLOG(3) << "Adding a copy of operand " << op_index << " (" << op->name()
                << ") "
                << " for " << inst->name();
        copy = state.comp->AddInstruction(
            HloInstruction::CreateUnary(op->shape(), HloOpcode::kCopy, op));
      }
      if (inplace_description.GetType() ==
              HloInstructionType::kInplaceReadWrite &&
          inst->opcode() != HloOpcode::kTuple) {
        // As we need to add copy anyway, set clone method to
        // CloneMethod_PreserveOrderUnlessAliases. This will rebalance tensor
        // and expand aliasing, because read/write inplace ops can't have
        // tensors with aliases as their inputs. Expand them here to avoid
        // another copy later.
        // Tuples should always preserve aliasing.
        TF_RETURN_IF_ERROR(SetCopyCloneMethod(
            copy, ShapeTree<CloneMethod>(
                      copy->shape(), CloneMethod_PreserveOrderUnlessAliases)));
      }

      // If we copy result of the instruction, we have to guarantee that
      // result was copied before any control successors (that potentially may
      // modify this buffer inplace).
      for (auto succ : op->control_successors()) {
        TF_RETURN_IF_ERROR(copy->AddControlDependencyTo(succ));
      }
      // If the instruction has control predecessors, we have to guarantee
      // that copies are made after all predecessors.
      for (HloInstruction* pred : inst->control_predecessors()) {
        TF_RETURN_IF_ERROR(pred->AddControlDependencyTo(copy));
      }
      op->SetupDerivedInstruction(copy);
      TF_RETURN_IF_ERROR(inst->ReplaceOperandWith(op_index, copy));
      changed = true;
    }
    TF_ASSIGN_OR_RETURN(bool copies_changed, ConvertToReallocatingCopies(inst));
    changed |= copies_changed;
  }

  return changed;
}

}  // namespace

StatusOr<bool> InplaceFinder::Run(HloModule* module) {
  // Remove situation when sharing constant between inplace ops will prevent
  // inplace finder from inplacing them. The simpliest example would be
  // (pseudocode):
  //   loop = while((counter, limit), condition=(counter < limit))
  //   ROOT tuple = (loop, counter, limit, learning_rate)
  // The loop above can't be lowered inplace, because it's read/write on its
  // arguments. Following the logic in ConvertToInplaceReadWrite,
  // inplace operand users should be scheduled after root tuple, but it's
  // impossible to do so. To resolve this situation, we can clone scalar
  // constants if they have inplace instructions among their users. This
  // optimisation temporarily limited to the loops only, but it could be used
  // for any other inplace instructions too.

  TF_ASSIGN_OR_RETURN(bool changed,
                      CloneConstantsSharedBetweenInplaceLoops(module));

  auto call_graph = CallGraph::Build(module);
  if (!call_graph->IsFlattened()) {
    return FailedPrecondition(
        "InplaceFinder requires the call graph to be flattened.");
  }

  TF_ASSIGN_OR_RETURN(auto dataflow, HloPoplarDataflowAnalysis::Run(
                                         module, annotations, *call_graph));
  for (auto* comp : module->MakeComputationPostOrder()) {
    if (!AllowedComputation(*call_graph, comp)) {
      continue;
    }

    InplacingState state(comp);
    const auto inplace_candidates = FindInplaceCandidates(comp, *dataflow);

    for (auto* inst : inplace_candidates) {
      if (HloPoplarInplaceDescription::ConvertToInplace(inst, state)) {
        changed = true;
        TF_RETURN_IF_ERROR(ConvertToReallocatingCopies(inst).status());
        VLOG(3) << "Inplaced " << inst->ToString();
      } else {
        AddCopiesForFailedInplace(inst, state);
      }
    }

    // Copies are inserted at the end as inserting instructions invalidates the
    // reachability map (part of the InplacingState).
    TF_ASSIGN_OR_RETURN(bool changed_, InsertCopies(state));
    changed |= changed_;
  }

  if (changed) {
    VLOG(2) << "After the InplaceFinder:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "There were no changes.";
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
