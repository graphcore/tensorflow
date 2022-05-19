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

#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_analysis.h"

#include <memory>
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_extensions.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_instruction_extensions.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/print_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"

namespace xla {

namespace poplarplugin {

namespace {
using InstructionToOperandIndices = absl::flat_hash_map<
    HloInstruction*,
    absl::flat_hash_map<HloInstruction*, absl::InlinedVector<int64_t, 1>>>;

struct AllocationLoopState {
  AllocationLoopState() = default;
  AllocationLoopState(const AllocationLoopState&) = delete;
  // holds index into state.result
  std::vector<int64_t> to_visit;
  AllocationGroups result;
  std::unique_ptr<CallGraph> call_graph;
  absl::flat_hash_map<IndexedLocation, int64_t> producer_to_index;
  absl::flat_hash_set<int64_t> visited_groups;
  absl::flat_hash_set<IndexedLocation> visited;
  InstructionToOperandIndices instruction_to_operand_indices;
  AllocationGroup& group(int64_t i) { return result.groups[i]; }
};

const TensorLocation null_location = {nullptr, 0};

bool ContinueSearching(const DoFindConsumers do_find_consumers,
                       HloInstruction* user, HloInstruction* operand) {
  switch (do_find_consumers) {
    case DoFindConsumers::FALSE: {
      return false;
    }
    case DoFindConsumers::TRUE: {
      return true;
    }
    default: {
      // This matches what the allocation finder does in the unspecified case
      return ShapeUtil::Compatible(user->shape(), operand->shape());
    }
  }
}

bool ContinueSearching(HloInstruction* user, HloInstruction* operand,
                       int64_t index) {
  auto result = CallHloInstructionExtension<FindConsumersExtension,
                                            FindConsumersExtensionParams>(
      user, FindConsumersExtensionParams{null_location, operand,
                                         /*index=*/~0U,
                                         /*op_index=*/index,
                                         /*permutation=*/absl::nullopt});
  return ContinueSearching(result.do_find_consumers, user, operand);
}

absl::InlinedVector<int64_t, 2> OperandsAffectingMapping(HloInstruction* inst) {
  // I'd expect this to be the same per opcode so would be nice if
  // didn't need to take whole instruction
  absl::InlinedVector<int64_t, 2> result;
  for (int64_t i = 0; i < inst->operand_count(); ++i) {
    if (ContinueSearching(inst, inst->mutable_operand(i), i)) {
      result.emplace_back(i);
    }
  }
  return result;
}

int64_t AddGroup(AllocationGroup&& group, AllocationLoopState& state) {
  auto insert_result = state.producer_to_index.emplace(
      group.producer, state.result.groups.size());
  if (insert_result.second) {
    // Allocating instructions are visited once per operand, if this
    // is the case don't insert and return the group already created
    // for them
    state.result.groups.emplace_back(std::move(group));
  }
  return insert_result.first->second;
}

void AddGroupToVisit(AllocationGroup&& group, AllocationLoopState& state) {
  state.to_visit.push_back(AddGroup(std::move(group), state));
}

bool IsTensorIndex(const Shape& shape, const ShapeIndex& index) {
  return ShapeUtil::GetSubshape(shape, index).IsArray();
}

// Generate all the IndexedLocations containing this instruction
absl::InlinedVector<IndexedLocation, 1> GenerateAllLocations(
    HloInstruction* inst) {
  absl::InlinedVector<IndexedLocation, 1> result;
  const auto& shape = inst->shape();
  ShapeUtil::ForEachSubshape(
      shape, [&](const Shape& sub_shape, const ShapeIndex& index) {
        if (IsTensorIndex(shape, index)) {
          result.emplace_back(inst, index);
        }
      });
  return result;
}

bool IsHandledSeperately(HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kWhile:
    case HloOpcode::kTuple:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kConditional: {
      return true;
    }
    default: { return false; }
  }
}

StatusOr<bool> InstructionCreatesNewMapping(HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kDot) {
    return true;
  }
  if (inst->opcode() == HloOpcode::kCopy) {
    TF_ASSIGN_OR_RETURN(auto clone_method_tree, GetCopyCloneMethod(inst));
    return absl::c_any_of(
        clone_method_tree.leaves(),
        [](const std::pair<ShapeIndex, CloneMethod>& pair) -> bool {
          auto method = pair.second;
          return method ==
                     CloneMethod::CloneMethod_DeduceNewOrderOrPreserveAliases ||
                 method ==
                     CloneMethod::CloneMethod_DeduceNewOrderOrExpandAliases;
        });
  }
  // The extension only works for custom ops hence the check for dots above
  return CallHloInstructionExtension<AllocatingOutputExtension>(inst);
}

// for call instructions that aren't repeats, we don't know if they
// allocate yet, so assume they do and merge them later if they
// don't
StatusOr<bool> InstructionMaybeCreatesNewMapping(HloInstruction* inst) {
  TF_ASSIGN_OR_RETURN(bool remaps, InstructionCreatesNewMapping(inst));
  return remaps ||
         (inst->opcode() == HloOpcode::kCall && !IsRepeatLoop(inst)) ||
         inst->opcode() == HloOpcode::kConditional;
}

// Instructions that take in no tensors which affect their output tile mapping
StatusOr<bool> IsProducer(HloInstruction* inst) {
  if (inst->operand_count() == 0) {
    return true;
  }
  if (absl::c_all_of(inst->operands(), [](HloInstruction* op) {
        return GenerateAllLocations(op).empty();
      })) {
    return true;
  }
  if (inst->opcode() == HloOpcode::kGetTupleElement) {
    return false;
  }
  TF_ASSIGN_OR_RETURN(bool maybe_remaps,
                      InstructionMaybeCreatesNewMapping(inst));
  if (maybe_remaps) {
    return true;
  }
  // This shouldn't be needed as if no inputs affect mapping then I'd say
  // that the instruction allocates. As code currently stand though,
  // broadcast instructions have no inputs that affect the output and
  // don't allocate. Though I'd say this is because one of those extensions
  // isn't correct for them.
  return OperandsAffectingMapping(inst).empty();
}

Status FindStartPoints(AllocationLoopState& state, HloModule* module) {
  for (auto* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool is_producer, IsProducer(inst));
      if (is_producer) {
        for (auto& location : GenerateAllLocations(inst)) {
          AddGroupToVisit(AllocationGroup(location), state);
        }
      }
    }
  }
  return Status::OK();
}

// Only some inputs affect tile mapping of output, for this function
// only add the user if next.instruction's tile mapping is related to
// it's output
Status AddIfMappingDependsOnOperand(std::vector<IndexedLocation>& to_visit,
                                    HloInstruction* user,
                                    const IndexedLocation& operand,
                                    const AllocationLoopState& state) {
  switch (user->opcode()) {
    case HloOpcode::kCall: {
      if (!IsRepeatLoop(user)) {
        // We will deal with calls in the group merging
        break;
      }
      // fall through for repeats as inputs match outputs
    }
    case HloOpcode::kTuple: {
      const auto& indices =
          state.instruction_to_operand_indices.at(user).at(operand.instruction);
      for (auto i : indices) {
        auto new_index = operand.index;
        new_index.push_front(i);
        to_visit.emplace_back(user, std::move(new_index));
      }
      break;
    }
    case HloOpcode::kGetTupleElement: {
      if (user->tuple_index() == operand.index[0]) {
        to_visit.emplace_back(
            user, ShapeIndex(operand.index.begin() + 1, operand.index.end()));
      }
      break;
    }
    default: {
      for (int64_t i = 0; i < user->operand_count(); ++i) {
        if (user->operand(i) != operand.instruction) {
          continue;
        }
        if (ContinueSearching(user, operand.instruction, i)) {
          to_visit.emplace_back(user, operand.index);
        }
      }
      break;
    }
  }
  return Status::OK();
}

Status IterateThroughUsers(int64_t index, AllocationLoopState& state) {
  std::vector<IndexedLocation> to_visit = {state.group(index).producer};
  while (to_visit.size()) {
    IndexedLocation next = to_visit.back();
    to_visit.pop_back();
    // Hit the end of the group as this is an allocating instruction.
    // If the instruction allocates (or in kCall case maybe allocates),
    // stop iterating and add a new group to the to visit state.
    if (next != state.group(index).producer) {
      TF_ASSIGN_OR_RETURN(bool maybe_remaps,
                          InstructionMaybeCreatesNewMapping(next.instruction));
      if (maybe_remaps) {
        continue;
      }
    }
    // Note this doesn't guard above code as end/start points need to visited
    // multiple times. One to mark the end, and one to start the next group
    auto insert_result = state.visited.insert(next);
    if (!insert_result.second) {
      continue;
    }
    state.group(index).AddInstructionToGroup(next);
    for (auto* user : next.instruction->users()) {
      TF_RETURN_IF_ERROR(
          AddIfMappingDependsOnOperand(to_visit, user, next, state));
    }
  }
  return Status::OK();
}

Status AddUsersInSameComputationsToGroups(AllocationLoopState& state) {
  while (state.to_visit.size()) {
    int64_t index = state.to_visit.back();
    state.to_visit.pop_back();
    auto insert_res = state.visited_groups.insert(index);
    if (insert_res.second) {
      TF_RETURN_IF_ERROR(IterateThroughUsers(index, state));
    }
  }
  return Status::OK();
}

absl::flat_hash_map<IndexedLocation, int64_t> CreateLocationToGroupMap(
    const AllocationLoopState& state) {
  return state.result.CreateLocationToGroupMap();
}

void MergeInstructionGroups(
    HloInstruction* merge_into, HloInstruction* merge_from,
    absl::flat_hash_map<IndexedLocation, int64_t>& loc_to_group,
    AllocationLoopState& state) {
  const auto locations = GenerateAllLocations(merge_into);
  for (const auto& loc : locations) {
    IndexedLocation location_from = {merge_from, loc.index};
    int64_t merge_to_index = loc_to_group.at(loc);
    auto& group_to_merge_to = state.result.groups[merge_to_index];
    int64_t merge_from_index = loc_to_group.at(location_from);
    if (merge_to_index == merge_from_index) {
      // already same group so nothing to do
      continue;
    }
    auto& group_to_merge_from = state.result.groups[merge_from_index];
    // Keep the loc_to_group map correct as we are still iterating
    // groups to merge and still use it
    for (const auto& moved_loc : group_to_merge_from.group) {
      loc_to_group[moved_loc] = merge_to_index;
    }
    group_to_merge_to.group.insert(
        std::make_move_iterator(group_to_merge_from.group.begin()),
        std::make_move_iterator(group_to_merge_from.group.end()));
    group_to_merge_to.inputs_only.insert(
        group_to_merge_from.inputs_only.begin(),
        group_to_merge_from.inputs_only.end());
    group_to_merge_from.clear();
  }
}

void MergeWhileGroups(
    HloInstruction* while_op,
    absl::flat_hash_map<IndexedLocation, int64_t>& loc_to_group,
    AllocationLoopState& state) {
  // by convention we are going to add everything to the groups containing the
  // while body parameter instruction
  auto* called_inst = while_op->while_body()->parameter_instruction(0);
  auto* cond_inst = while_op->while_condition()->parameter_instruction(0);
  MergeInstructionGroups(while_op, called_inst, loc_to_group, state);
  MergeInstructionGroups(while_op, cond_inst, loc_to_group, state);
}

void MergeCallGroups(
    HloInstruction* call_op,
    absl::flat_hash_map<IndexedLocation, int64_t>& loc_to_group,
    AllocationLoopState& state) {
  for (int64_t i = 0; i < call_op->operand_count(); ++i) {
    auto* parameter = call_op->to_apply()->parameter_instruction(i);
    MergeInstructionGroups(call_op->mutable_operand(i), parameter, loc_to_group,
                           state);
  }
  if (!IsRepeatLoop(call_op)) {
    // merge root instruction to outputs
    auto* root = call_op->to_apply()->root_instruction();
    MergeInstructionGroups(root, call_op, loc_to_group, state);
  }
}

// Now merge groups produced by
// parameter instructions with the places that call them. For Calls that aren't
// repeats also merge the groups containing the root instruction with the call
// group
void MergeGroupsFromCallsites(AllocationLoopState& state, HloModule* module) {
  auto loc_to_group = CreateLocationToGroupMap(state);
  auto post_order = module->MakeComputationPostOrder();
  for (auto it = post_order.rbegin(); it < post_order.rend(); ++it) {
    // iterate from entry computation down merging groups into instructions that
    // it calls
    auto* comp = *it;
    if (IsPopOpsFusion(comp)) {
      continue;
    }
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      switch (inst->opcode()) {
        case HloOpcode::kWhile: {
          MergeWhileGroups(inst, loc_to_group, state);
          break;
        }
        case HloOpcode::kCall: {
          MergeCallGroups(inst, loc_to_group, state);
          break;
        }
        case HloOpcode::kConditional: {
          // TODO(samuelh) extend to handle conditionals. Right now this just
          // means the allocation groups aren't merged, which is fine, just not
          // optimal
          break;
        }
        default: {}
      }
    }
  }
}

bool IsHandlableMergePoint(HloInstruction* inst) {
  if (IsHandledSeperately(inst)) {
    return false;
  }
  const auto indices = OperandsAffectingMapping(inst);
  if (indices.size() < 2) {
    return false;
  }
  for (auto i : indices) {
    if (!inst->operand(i)->shape().IsArray()) {
      // don't know how to handle tuple shaped inputs so
      // play safe and don't merge
      return false;
    }
  }
  if (!inst->shape().IsArray()) {
    return false;
  }
  return !absl::c_all_of(indices, [&](const int64_t i) {
    return inst->operand(0) == inst->operand(i);
  });
}

std::vector<IndexedLocation> FindMergePoints(const AllocationLoopState& state) {
  std::vector<IndexedLocation> merge_points;
  for (const auto& group : state.result.groups) {
    for (auto loc : group.group) {
      if (IsHandlableMergePoint(loc.instruction)) {
        merge_points.emplace_back(loc.instruction, ShapeIndex());
      }
    }
  }
  return merge_points;
}

void TryMergeGroups(
    AllocationLoopState& state, std::vector<IndexedLocation> merge_points,
    absl::flat_hash_map<IndexedLocation, int64_t> loc_to_group) {
  for (const auto& merge_point : merge_points) {
    const auto indices = OperandsAffectingMapping(merge_point.instruction);
    // we already asserted no tuple shapes so just assume all indexes are {}
    for (int64_t index : indices) {
      MergeInstructionGroups(merge_point.instruction,
                             merge_point.instruction->mutable_operand(index),
                             loc_to_group, state);
    }
  }
}

// If we have an instruction like
// Add(X, Y). We want X and Y to be in the same
// group as they should have the same layout
void MergeGroupsFromRelatedOperands(AllocationLoopState& state) {
  auto merge_points = FindMergePoints(state);
  auto loc_to_group = CreateLocationToGroupMap(state);
  TryMergeGroups(state, std::move(merge_points), std::move(loc_to_group));
}

void FilterEmpty(AllocationLoopState& state) {
  FilterInPlace(state.result.groups, [](const AllocationGroup& group) {
    return group.producer.instruction != nullptr;
  });
}

bool WorthBuilding(HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kTuple || IsRepeatLoop(inst);
}

InstructionToOperandIndices CreateInstructionOperandsMap(HloModule* module) {
  InstructionToOperandIndices result;
  for (auto* comp : module->computations()) {
    for (auto* inst : comp->instructions()) {
      if (WorthBuilding(inst)) {
        auto& inst_map = result[inst];
        for (int64_t i = 0; i < inst->operand_count(); ++i) {
          inst_map[inst->mutable_operand(i)].emplace_back(i);
        }
      }
    }
  }
  return result;
}

Status AddGroupInputInstructions(AllocationLoopState& state) {
  const auto loc_to_group = CreateLocationToGroupMap(state);
  for (auto& group : state.result.groups) {
    const auto& producer = group.producer;
    TF_ASSIGN_OR_RETURN(bool remaps,
                        InstructionCreatesNewMapping(producer.instruction));
    if (!remaps) {
      continue;
    }
    for (int64_t i = 0; i < producer.instruction->operand_count(); ++i) {
      HloInstruction* operand = producer.instruction->mutable_operand(i);
      if (!operand->shape().IsArray()) {
        continue;
      }
      const auto input_group_index =
          loc_to_group.at(IndexedLocation(operand, {}));
      auto& input_group = state.result.groups[input_group_index];
      input_group.AddGroupEndInstruction(
          InputLocation(producer.instruction, i));
    }
  }
  return Status::OK();
}

}  // namespace

StatusOr<AllocationGroups> AllocationGroups::CreateAllocationGroups(
    HloModule* module) {
  AllocationLoopState state;
  state.instruction_to_operand_indices = CreateInstructionOperandsMap(module);
  state.call_graph = CallGraph::Build(module);
  TF_RETURN_IF_ERROR(FindStartPoints(state, module));
  TF_RETURN_IF_ERROR(AddUsersInSameComputationsToGroups(state));
  MergeGroupsFromRelatedOperands(state);
  FilterEmpty(state);
  MergeGroupsFromCallsites(state, module);
  FilterEmpty(state);
  TF_RETURN_IF_ERROR(AddGroupInputInstructions(state));
  return state.result;
}

// ***********************************************************************
// ToString methods
// ***********************************************************************

std::string IndexedLocation::ToString() const {
  return absl::StrCat("{", instruction->name(), ",", index.ToString(), "}");
}

std::string InputLocation::ToString() const {
  return absl::StrCat("{", instruction->name(), ",", operand_index, "}");
}

std::string AllocationGroup::ToString() const {
  return absl::StrCat(
      "AllocationGroup: {", producer.instruction->name(), " (",
      ShapeUtil::GetSubshape(producer.instruction->shape(), producer.index)
          .ToString(),
      "): {", absl::StrJoin(group, ",", ToStringFormatter()), "}}");
}

std::string AllocationGroups::ToString() const {
  return absl::StrCat("AllocationGroups: {\n  ",
                      absl::StrJoin(groups, "\n  ", ToStringFormatter()),
                      "\n}");
}

// ***********************************************************************
// Verify methods
// ***********************************************************************

void AllocationGroups::Verify(HloModule* module) const {
  VLOG(1) << ToString();
  absl::flat_hash_map<IndexedLocation, AllocationGroups> inst_to_groups;
  for (auto* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      for (auto& location : GenerateAllLocations(inst)) {
        for (const auto& group : groups) {
          if (group.group.contains(location)) {
            inst_to_groups[location].groups.emplace_back(group);
          }
        }
      }
    }
  }

  for (auto* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      for (auto& location : GenerateAllLocations(inst)) {
        VLOG(1) << "Verifying: " << location.ToString();
        const auto& groups = inst_to_groups.at(location);
        CHECK_EQ(groups.groups.size(), 1)
            << location.ToString() << " in :" << groups.ToString();
      }
    }
  }
  absl::flat_hash_set<IndexedLocation> producers;
  for (const auto& group : groups) {
    const auto res = producers.insert(group.producer);
    CHECK(res.second) << "Producer creates multiple groups "
                      << group.producer.ToString();
  }
}

absl::flat_hash_map<IndexedLocation, int64_t>
AllocationGroups::CreateLocationToGroupMap() const {
  absl::flat_hash_map<IndexedLocation, int64_t> result;
  for (int64_t i = 0; i < groups.size(); ++i) {
    for (const auto& loc : groups[i].group) {
      result.emplace(loc, i);
    }
  }
  return result;
}

}  // namespace poplarplugin
}  // namespace xla
