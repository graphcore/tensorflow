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

#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/print_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/slice_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/while_loop_optimisation_utils.h"

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/while_loop_analysis.h"
#include "tensorflow/compiler/xla/service/while_util.h"

namespace xla {

namespace m = match;

namespace poplarplugin {
namespace WhileLoopOptimisationUtils {

struct KnownScalarIntegers {
  // Obeys intersection of negative and positive is nothing
  // but an instruction can be in negative/positive and
  // maybe zero
  absl::flat_hash_set<HloInstruction*> negative_numbers;
  absl::flat_hash_set<HloInstruction*> positive_numbers;
  absl::flat_hash_set<HloComputation*> searched_computation;
  // As we are checking for < (not <=) we store a bool to check
  // we have hit one value that is defiantely less than
  // Can't use the positive/negative set as those include
  // potential zeros
  bool hit_one_definately_less = false;

  bool is_unknown(HloInstruction* inst) const {
    return (!negative_numbers.contains(inst)) &&
           (!positive_numbers.contains(inst));
  }

  bool is_negative(HloInstruction* inst) const {
    return negative_numbers.contains(inst);
  }

  std::string ToString() const {
    return absl::StrCat("{\n  negative {",
                        absl::StrJoin(negative_numbers, ",", NameFormatter()),
                        "}\n  positive {",
                        absl::StrJoin(positive_numbers, ",", NameFormatter()),
                        "}}");
  }
};

std::string Uses::ToString() const {
  return absl::StrCat("Uses {", absl::StrJoin(uses, ", ", NameFormatter()),
                      "}");
}

SliceAndIndex::SliceAndIndex(int64_t input_index,
                             HloInstruction* dynamic_update,
                             int64_t index_index)
    : input_index(input_index),
      dynamic_update(dynamic_update),
      index_index(index_index) {}

std::string SliceAndIndex::ToString() const {
  return absl::StrCat("{", input_index, ", ", dynamic_update->name(), ",",
                      dynamic_update->shape().ToString(), ", ", index_index,
                      "}");
}

BroadcastAndSlice::BroadcastAndSlice(SliceAndIndex broadcast, Uses uses)
    : broadcast(std::move(broadcast)), uses(std::move(uses)) {}

std::string BroadcastAndSlice::ToString() const {
  return absl::StrCat("{", broadcast.ToString(), ", ",
                      absl::StrJoin(uses.uses, ", ", NameFormatter()), "}");
}

void FindUsesTemplate::DefaultAction(HloInstruction* user) {
  result.emplace_back(user);
}

void FindUsesTemplate::GTEAction(HloInstruction* user) {}

void FindUsesTemplate::TupleAction(HloInstruction* user) {}

void FindUsesTemplate::WhileAction(HloInstruction* user) {
  while_bodies.emplace(user->while_body());
}

bool IsTripCounter(HloInstruction* while_loop, HloInstruction* gte) {
  if (gte->opcode() != HloOpcode::kGetTupleElement) {
    return false;
  }
  auto* root = while_loop->while_body()->root_instruction();
  if (gte->operand(0) != while_loop->while_body()->parameter_instruction(0)) {
    return false;
  }
  if (!Match(
          root->mutable_operand(gte->tuple_index()),
          m::AddAnyOrder(m::GetTupleElement(m::Parameter(), gte->tuple_index()),
                         m::ConstantScalar(1)))) {
    return false;
  }
  auto init = GetConstantValue<int>(
      while_loop->operand(0)->operand(gte->tuple_index()));
  if (!init) {
    return false;
  }
  return *init == 0;
}

bool InstructionIsTripCounter(const CallGraph& call_graph,
                              HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kGetTupleElement) {
    return false;
  }
  const auto& callsites = call_graph.GetNode(inst->parent()).caller_callsites();
  if (callsites.size() != 1) {
    return false;
  }
  if (callsites[0].instruction()->opcode() != HloOpcode::kWhile) {
    return false;
  }
  return IsTripCounter(callsites[0].instruction(), inst);
}

// Check if we have got to the end that, it's not a get tuple
// index of wrong element and so wouldn't be a real use.
bool IsActualUse(HloInstruction* user,
                 const std::stack<int64_t>& tuple_indices) {
  return user->opcode() != HloOpcode::kGetTupleElement ||
         user->tuple_index() == tuple_indices.top();
}

template <class ArgsTemplate>
void FindNonTrivialUses(HloInstruction* inst, ArgsTemplate& args,
                        std::stack<int64_t>& tuple_indices) {
  for (auto* user : inst->users()) {
    if (user == user->parent()->root_instruction() &&
        IsActualUse(user, tuple_indices)) {
      // If is a while loop invariant it is not really used anymore
      // so can skip
      if (!IsWhileLoopInvariant(inst, user, args)) {
        // If is root instruction treat as a non trivial use
        // and exit out
        args.DefaultAction(user);
      }
      continue;
    }
    VLOG(10) << "Visiting " << user->name();
    switch (user->opcode()) {
      case HloOpcode::kGetTupleElement: {
        if (tuple_indices.top() == user->tuple_index()) {
          args.GTEAction(user);
          tuple_indices.pop();
          FindNonTrivialUses(user, args, tuple_indices);
          // Restore the tuple indices back to current state as
          // still inside a for loop of users
          tuple_indices.push(user->tuple_index());
        }
        break;
      }
      case HloOpcode::kTuple: {
        for (int64_t i = 0; i < user->operands().size(); ++i) {
          args.TupleAction(user);
          if (user->mutable_operand(i) == inst) {
            tuple_indices.push(i);
            FindNonTrivialUses(user, args, tuple_indices);
            // restore tuple indices back to old state
            tuple_indices.pop();
          }
        }
        break;
      }
      case HloOpcode::kWhile: {
        CHECK_EQ(user->operands().size(), 1U);
        args.WhileAction(user);
        FindNonTrivialUses(user->while_body()->parameter_instruction(0), args,
                           tuple_indices);
        FindNonTrivialUses(user->while_condition()->parameter_instruction(0),
                           args, tuple_indices);
        FindNonTrivialUses(user, args, tuple_indices);

        break;
      }
      default: {
        args.DefaultAction(user);
        break;
      }
    }
  }
}

Uses FindNonTrivialUses(HloInstruction* inst,
                        std::stack<int64_t> starting_gte_stack) {
  FindUsesTemplate args;
  FindNonTrivialUses(inst, args, starting_gte_stack);
  return {std::move(args.result)};
}

void InsertKnownNumbersFromThisComputation(KnownScalarIntegers& known_numbers,
                                           HloComputation* comp,
                                           const CallGraph& call_graph) {
  for (auto* inst : comp->MakeInstructionPostOrder()) {
    if (!ShapeUtil::IsScalar(inst->shape()) ||
        inst->shape().element_type() != S32) {
      // only interested in scalar integers
      continue;
    }
    if (inst->opcode() == HloOpcode::kConstant) {
      int value = *GetConstantValue<int>(inst);
      // if value is zero just pretend it's unknown,
      // zeors will nearly always get optimised away
      // so not going to worry about them
      if (value < 0) {
        known_numbers.negative_numbers.insert(inst);
      } else if (value > 0) {
        known_numbers.positive_numbers.insert(inst);
      }
      continue;
    }
    if (InstructionIsTripCounter(call_graph, inst)) {
      known_numbers.positive_numbers.insert(inst);
      continue;
    }
    if (inst->opcode() == HloOpcode::kMultiply) {
      auto* op_a = inst->mutable_operand(0);
      auto* op_b = inst->mutable_operand(1);
      if (known_numbers.is_unknown(op_a) || known_numbers.is_unknown(op_b)) {
        continue;
      }
      bool a_is_negative = known_numbers.negative_numbers.contains(op_a);
      bool b_is_negative = known_numbers.negative_numbers.contains(op_b);
      if (a_is_negative ^ b_is_negative) {
        known_numbers.negative_numbers.insert(inst);
      } else {
        known_numbers.positive_numbers.insert(inst);
      }

      continue;
    }
  }
}

bool CanWorkOutAndLessThan(HloInstruction* inst,
                           const KnownScalarIntegers& known_numbers) {
  switch (inst->opcode()) {
    // for now only support add as all we need but
    // easy to extend
    case HloOpcode::kAdd: {
      HloInstruction* op_a = inst->mutable_operand(0);
      HloInstruction* op_b = inst->mutable_operand(1);
      return known_numbers.is_negative(op_a) || known_numbers.is_negative(op_b);
    }
    default: { return false; }
  }
}

bool OpIsAlwaysLess(HloInstruction* inst) {
  HloInstruction* constant;
  if (!Match(inst, m::AddAnyOrder(m::Op(), m::ConstantScalar(&constant)))) {
    return false;
  }
  int value = *GetConstantValue<int>(constant);
  return value < 0;
}

bool CanGuarenteeUnknownIndexIsLess(HloInstruction* while_loop,
                                    const SliceAndIndex& broadcast,
                                    HloInstruction* slice) {
  auto call_graph = CallGraph::Build(slice->GetModule());
  KnownScalarIntegers known_numbers;
  std::stack<int64_t> starting_stack;
  starting_stack.push(broadcast.index_index);
  absl::InlinedVector<HloInstruction*, 1> to_visit =
      std::move(FindNonTrivialUses(while_loop, starting_stack).uses);
  // Iterate through all instructions less than or equal to this
  // one until we hit the slice instruction
  while (!to_visit.empty()) {
    auto* inst = to_visit.back();
    to_visit.pop_back();
    if (inst == slice) {
      return known_numbers.hit_one_definately_less;
    }
    auto insert_result =
        known_numbers.searched_computation.emplace(inst->parent());
    if (insert_result.second) {
      InsertKnownNumbersFromThisComputation(known_numbers, inst->parent(),
                                            *call_graph);
    }
    if (CanWorkOutAndLessThan(inst, known_numbers)) {
      auto new_uses = FindNonTrivialUses(inst, std::stack<int64_t>());
      to_visit.insert(to_visit.end(), new_uses.uses.begin(),
                      new_uses.uses.end());
      known_numbers.hit_one_definately_less |= OpIsAlwaysLess(inst);
    }
  }
  return false;
}

template <class ArgsTemplate>
bool IsWhileLoopInvariant(HloInstruction* inst, HloInstruction* user,
                          ArgsTemplate& args) {
  // not a while loop body
  if (!args.while_bodies.contains(user->parent())) {
    return false;
  }
  // not of expected form
  if (inst->opcode() != HloOpcode::kGetTupleElement ||
      user->opcode() != HloOpcode::kTuple ||
      inst->operand(0) != user->parent()->parameter_instruction(0)) {
    return false;
  }
  // Before adding this to the pipeline remove this for loop
  // as I expect it is too expensive
  for (int64_t i = 0; i < user->operands().size(); ++i) {
    if (i == inst->tuple_index()) {
      if (inst == user->operand(i)) {
        continue;
      } else {
        return false;
      }
    }
    if (inst == user->operand(i)) {
      return false;
    }
  }
  return true;
}

void FindLoopConstants(HloComputation* comp,
                       absl::flat_hash_set<HloInstruction*>& result,
                       const bool include_constants) {
  for (auto* inst : comp->MakeInstructionPostOrder()) {
    if (inst->opcode() == HloOpcode::kParameter || inst->HasSideEffect()) {
      continue;
    }
    if (!include_constants && inst->opcode() == HloOpcode::kConstant) {
      continue;
    }
    const bool all_const_inputs =
        absl::c_all_of(inst->operands(), [&](HloInstruction* operand) {
          return result.contains(operand) ||
                 operand->opcode() == HloOpcode::kConstant;
        });
    if (all_const_inputs) {
      result.emplace(inst);
    }
  }
}

absl::flat_hash_set<HloInstruction*> FindLoopConstants(
    HloComputation* comp, const bool include_constants) {
  absl::flat_hash_set<HloInstruction*> result;
  auto gtes = WhileUtil::GetInvariantGTEsForWhileBody(*comp);
  result.insert(gtes.begin(), gtes.end());
  FindLoopConstants(comp, result, include_constants);
  return result;
}

absl::flat_hash_set<HloInstruction*> FindTripCounters(
    HloInstruction* while_op) {
  auto* body = while_op->while_body();
  auto* root = body->root_instruction();
  auto* while_init = while_op->mutable_operand(0);
  absl::flat_hash_set<HloInstruction*> result;
  for (int64_t i = 0; i < root->operand_count(); ++i) {
    if (!while_init->operand(i)->IsConstant() ||
        !while_init->operand(i)->literal().IsAll(0)) {
      continue;
    }
    HloInstruction* gte;
    if (!Match(root->mutable_operand(i),
               m::AddAnyOrder(m::GetTupleElement(&gte, m::Parameter(), i),
                              m::ConstantScalar(1)))) {
      continue;
    }
    result.emplace(gte);
  }
  return result;
}

std::vector<HloInstruction*> FindSingleUseParameters(HloComputation* comp) {
  CHECK_EQ(comp->num_parameters(), 1);
  HloInstruction* input_tuple = comp->parameter_instruction(0);
  if (!input_tuple->shape().IsTuple()) {
    return {};
  }
  std::vector<HloInstruction*> result(
      ShapeUtil::TupleElementCount(input_tuple->shape()));
  std::vector<int64_t> use_count(result.size(), 0);
  for (auto* inst : input_tuple->users()) {
    if (inst->opcode() == HloOpcode::kGetTupleElement) {
      int64_t index = inst->tuple_index();
      result[index] = inst;
      ++use_count[index];
    } else {
      return {};
    }
  }
  for (int64_t i = 0; i < result.size(); ++i) {
    if (use_count[i] != 1) {
      result[i] = nullptr;
    }
  }
  return result;
}

bool BroadcastIndexIsTripCounter(HloInstruction* while_loop,
                                 const SliceAndIndex& broadcast) {
  return IsTripCounter(while_loop,
                       broadcast.dynamic_update->mutable_operand(2));
}

bool SliceIndexAlreadyWrittenTo(HloInstruction* while_loop,
                                const SliceAndIndex& broadcast,
                                HloInstruction* slice) {
  auto slice_index = GetConstantValue<int>(slice->operand(1));
  auto trip_count = ComputeWhileLoopTripCount(while_loop, 0);
  if (slice_index && trip_count) {
    return (*slice_index) < (*trip_count);
  }
  return CanGuarenteeUnknownIndexIsLess(while_loop, broadcast, slice);
}

bool SliceIndexAlreadyWrittenTo(HloInstruction* while_loop,
                                const SliceAndIndex& broadcast,
                                const Uses& slices) {
  if (!BroadcastIndexIsTripCounter(while_loop, broadcast)) {
    VLOG(10) << "Broadcast index is not trip counter";
    return false;
  }
  return absl::c_all_of(slices.uses, [&](HloInstruction* slice) {
    return SliceIndexAlreadyWrittenTo(while_loop, broadcast, slice);
  });
}

bool AllAreSlices(const absl::InlinedVector<HloInstruction*, 1>& vec) {
  if (vec.empty()) {
    // no uses so can just let other passes optimise away
    return false;
  }
  return absl::c_all_of(vec, [](const HloInstruction* inst) {
    return inst->opcode() == HloOpcode::kDynamicSlice;
  });
}

bool SliceDimensionsAreCorrect(
    const absl::InlinedVector<HloInstruction*, 1>& vec) {
  return absl::c_all_of(vec, [](const HloInstruction* slice) {
    return Is1DSliceInFirstDimension(slice);
  });
}

bool UnexpectedWhileLoopForm(HloInstruction* while_op) {
  HloComputation* comp = while_op->while_body();
  return comp->root_instruction()->opcode() != HloOpcode::kTuple ||
         comp->num_parameters() != 1 ||
         while_op->mutable_operand(0)->opcode() != HloOpcode::kTuple;
}

// Find cases where a new value is pushed into a dynamic update buffer inside
// a while loop. eg
// While(i<x):
//   Y = dynamic-update-slice(Y, Z, i)
//   ++i
std::vector<SliceAndIndex> FindTensorListBroadcasts(HloInstruction* while_op) {
  if (UnexpectedWhileLoopForm(while_op)) {
    VLOG(10) << "While loop in unexpected form " << while_op->name();
    return {};  // Can't do any optimisations
  }
  HloComputation* comp = while_op->while_body();
  const auto trip_counters = FindTripCounters(while_op);
  std::vector<SliceAndIndex> result;
  for (auto* candidate_gte : FindSingleUseParameters(comp)) {
    if (candidate_gte == nullptr) {
      continue;
    }
    // We are looking for the pattern
    // Root(
    //      dynamic-update(gte
    //                     loop_invariant,
    //                     loop_counter))

    // TODO(samuelh) would be better to do thus using instruction->Match with a
    // matcher as much more readable

    // Must only be used my a dynamic update and must be the operand updated
    // into (eg operand 0)
    auto candidate_update = candidate_gte->users();
    if (candidate_update.size() != 1U ||
        candidate_update[0]->opcode() != HloOpcode::kDynamicUpdateSlice ||
        candidate_update[0]->mutable_operand(0) != candidate_gte) {
      VLOG(10) << "Instruction used multiple times or not my dynamic update "
               << candidate_update.size();
      continue;
    }
    auto* updated_data = candidate_update[0];
    auto* root = comp->root_instruction();
    CHECK(updated_data->opcode() == HloOpcode::kDynamicUpdateSlice);
    // Check no other instruction inside the while body are using the update
    if (updated_data->users().size() != 1U ||
        updated_data->users()[0] != root) {
      VLOG(10) << "Dynamic update is used else where in the body";
      continue;
    }
    // We have the only user of the gte element as a dynamic update, now
    // check that this is fed to the root tuple at the same index as the gte
    int64_t parameter_index = candidate_gte->tuple_index();
    CHECK(parameter_index < root->operand_count());
    if (root->mutable_operand(parameter_index) != updated_data) {
      VLOG(10) << "Not at correct index to be an inplace update "
               << parameter_index;
      continue;
    }
    auto* input_slice = updated_data->mutable_operand(1);

    // Need to check the dimensions are correct, we are only expecting a slice
    // into the first dimension.
    if (ShapeUtil::DeleteDimension(0, updated_data->shape()) !=
            ShapeUtil::DeleteDimension(0, input_slice->shape()) ||
        ShapeUtil::GetDimension(input_slice->shape(), 0) != 1) {
      VLOG(10) << "Update slice is wrong dimensions are wrong";
      continue;
    }
    // We now know a loop invariant is being updated into the tensor
    // list at every iteration, now need to confirm that the index of
    // update is the loop counter
    auto* index = updated_data->mutable_operand(2);
    if (!trip_counters.contains(index)) {
      VLOG(10) << "Update index is not a loop index counter " << index->name();
      continue;
    }
    VLOG(10) << updated_data->name() << " is a tensor list broadcast";
    result.emplace_back(candidate_gte->tuple_index(), updated_data,
                        index->tuple_index());
  }
  return result;
}

std::vector<BroadcastAndSlice> FindBroadcastsOnlyUsedBySlices(
    HloInstruction* while_loop, std::vector<SliceAndIndex> broadcasts) {
  std::vector<BroadcastAndSlice> result;
  result.reserve(broadcasts.size());
  for (auto& broadcast : broadcasts) {
    std::stack<int64_t> start;
    start.push(0);
    start.push(broadcast.input_index);
    auto users = FindNonTrivialUses(while_loop, start);
    if (!AllAreSlices(users.uses)) {
      VLOG(10) << "Skipping as not all slices";
      continue;
    }
    if (!SliceDimensionsAreCorrect(users.uses)) {
      VLOG(10) << "Skipping as dimensions are wrong";
      continue;
    }
    if (!SliceIndexAlreadyWrittenTo(while_loop, broadcast, users)) {
      VLOG(10) << "Slice index is unknown or too large";
      continue;
    }
    VLOG(10) << "Found candidate";
    result.emplace_back(std::move(broadcast), std::move(users));
  }
  return result;
}

std::vector<BroadcastAndSlice> FindAllValidBroadcasts(HloInstruction* inst) {
  // Detect loop constants being pushed into tensor list
  auto broadcast_params = FindTensorListBroadcasts(inst);

  // Filter these tensor lists down to ones only used
  // by slice instructions
  return FindBroadcastsOnlyUsedBySlices(inst, std::move(broadcast_params));
}

}  // namespace WhileLoopOptimisationUtils
}  // namespace poplarplugin
}  // namespace xla
