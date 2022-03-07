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

#include "tensorflow/compiler/plugin/poplar/driver/passes/while_loop_optimiser.h"

#include <queue>
#include <stack>
#include <string>
#include <utility>
#include <vector>
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/print_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/slice_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/service/while_loop_invariant_code_motion.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/while_loop_analysis.h"
#include "tensorflow/compiler/xla/service/while_util.h"

namespace xla {

namespace m = match;

namespace poplarplugin {

namespace {

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
    HloComputation* comp, const bool include_constants = true) {
  absl::flat_hash_set<HloInstruction*> result;
  auto gtes = WhileUtil::GetInvariantGTEsForWhileBody(*comp);
  result.insert(gtes.begin(), gtes.end());
  FindLoopConstants(comp, result, include_constants);
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
  std::vector<int64> use_count(result.size(), 0);
  for (auto* inst : input_tuple->users()) {
    if (inst->opcode() == HloOpcode::kGetTupleElement) {
      int64 index = inst->tuple_index();
      result[index] = inst;
      ++use_count[index];
    } else {
      return {};
    }
  }
  for (int64 i = 0; i < result.size(); ++i) {
    if (use_count[i] != 1) {
      result[i] = nullptr;
    }
  }
  return result;
}

struct SliceAndIndex {
  int64 input_index;
  HloInstruction* dynamic_update;
  int64 index_index;
  SliceAndIndex(int64 input_index, HloInstruction* dynamic_update,
                int64 index_index)
      : input_index(input_index),
        dynamic_update(dynamic_update),
        index_index(index_index) {}
  std::string ToString() const {
    return absl::StrCat("{", input_index, ", ", dynamic_update->name(), ",",
                        dynamic_update->shape().ToString(), ", ", index_index,
                        "}");
  }
};

struct Uses {
  absl::InlinedVector<HloInstruction*, 1> uses;
  std::string ToString() const {
    return absl::StrCat("Uses {", absl::StrJoin(uses, ", ", NameFormatter()));
  }
};

struct BroadcastAndSlice {
  SliceAndIndex broadcast;
  Uses uses;
  std::string ToString() const {
    return absl::StrCat("{", broadcast.ToString(), ", ",
                        absl::StrJoin(uses.uses, ", ", NameFormatter()), "}");
  }
  BroadcastAndSlice(SliceAndIndex broadcast, Uses uses)
      : broadcast(std::move(broadcast)), uses(std::move(uses)) {}
};

bool InputIsLoopInvariant(
    HloInstruction* input,
    const absl::flat_hash_set<HloInstruction*>& constants) {
  return constants.contains(input);
}

absl::flat_hash_set<HloInstruction*> FindTripCounters(
    HloInstruction* while_op) {
  auto* body = while_op->while_body();
  auto* root = body->root_instruction();
  auto* while_init = while_op->mutable_operand(0);
  absl::flat_hash_set<HloInstruction*> result;
  for (int64 i = 0; i < root->operand_count(); ++i) {
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
  const auto loop_invariants = FindLoopConstants(comp);
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
    int64 parameter_index = candidate_gte->tuple_index();
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
  for (int64 i = 0; i < user->operands().size(); ++i) {
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

// Check if we have got to the end that, it's not a get tuple
// index of wrong element and so wouldn't be a real use.
bool IsActualUse(HloInstruction* user, const std::stack<int64>& tuple_indices) {
  return user->opcode() != HloOpcode::kGetTupleElement ||
         user->tuple_index() == tuple_indices.top();
}

template <class ArgsTemplate>
void FindNonTrivialUses(HloInstruction* inst, ArgsTemplate& args,
                        std::stack<int64>& tuple_indices) {
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
        for (int64 i = 0; i < user->operands().size(); ++i) {
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

struct FindUsesTemplate {
  absl::InlinedVector<HloInstruction*, 1> result;
  absl::flat_hash_set<HloComputation*> while_bodies;

  void DefaultAction(HloInstruction* user) { result.emplace_back(user); }
  void GTEAction(HloInstruction* user) {}
  void TupleAction(HloInstruction* user) {}
  void WhileAction(HloInstruction* user) {
    while_bodies.emplace(user->while_body());
  }
};

Uses FindNonTrivialUses(HloInstruction* inst,
                        std::stack<int64> starting_gte_stack) {
  FindUsesTemplate args;
  FindNonTrivialUses(inst, args, starting_gte_stack);
  return {std::move(args.result)};
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

bool BroadcastIndexIsTripCounter(HloInstruction* while_loop,
                                 const SliceAndIndex& broadcast) {
  return IsTripCounter(while_loop,
                       broadcast.dynamic_update->mutable_operand(2));
}

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
  std::stack<int64> starting_stack;
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
      auto new_uses = FindNonTrivialUses(inst, std::stack<int64>());
      to_visit.insert(to_visit.end(), new_uses.uses.begin(),
                      new_uses.uses.end());
      known_numbers.hit_one_definately_less |= OpIsAlwaysLess(inst);
    }
  }
  return false;
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

std::vector<BroadcastAndSlice> FindBroadcastsOnlyUsedBySlices(
    HloInstruction* while_loop, std::vector<SliceAndIndex> broadcasts) {
  std::vector<BroadcastAndSlice> result;
  result.reserve(broadcasts.size());
  for (auto& broadcast : broadcasts) {
    std::stack<int64> start;
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

template <typename T, typename Pred>
void FilterInPlace(T& vec, Pred p) {
  auto not_pred = [&](const typename T::value_type& a) { return !p(a); };
  vec.erase(std::remove_if(vec.begin(), vec.end(), not_pred), vec.end());
}

void SelectInvariantBroadcasts(HloInstruction* while_loop,
                               std::vector<BroadcastAndSlice>& broadcasts) {
  const auto loop_constants = FindLoopConstants(while_loop->while_body());
  FilterInPlace(broadcasts, [&](const BroadcastAndSlice& A) {
    return loop_constants.contains(
        A.broadcast.dynamic_update->mutable_operand(1));
  });
}

static StatusOr<Shape> DynamicSliceShape(HloInstruction* inst) {
  return ShapeInference::InferDynamicSliceShape(
      inst->operand(0)->shape(),
      Cast<HloDynamicSliceInstruction>(inst)->index_shapes(),
      inst->dynamic_slice_sizes());
}

static Shape DynamicUpdateShape(HloInstruction* inst) {
  return inst->operand(0)->shape();
}

static Shape GetTupleShape(HloInstruction* inst) {
  std::vector<Shape> shapes;
  shapes.reserve(inst->operands().size());
  for (const auto* op : inst->operands()) {
    shapes.emplace_back(op->shape());
  }
  return ShapeUtil::MakeTupleShape(shapes);
}

static StatusOr<Shape> WorkOutShapeFromOperands(HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kGetTupleElement: {
      return ShapeUtil::GetTupleElementShape(inst->operand(0)->shape(),
                                             inst->tuple_index());
    }
    case HloOpcode::kTuple: {
      return GetTupleShape(inst);
    }
    case HloOpcode::kCopy:
    case HloOpcode::kWhile: {
      return inst->operand(0)->shape();
    }
    case HloOpcode::kDynamicSlice: {
      return DynamicSliceShape(inst);
    }
    case HloOpcode::kDynamicUpdateSlice: {
      return DynamicUpdateShape(inst);
    }
    case HloOpcode::kCall: {
      return inst->to_apply()->root_instruction()->shape();
    }
    default: {
      return xla::FailedPrecondition("Unhandled type %s", inst->name());
    }
  }
}

static void ReplaceInstructionWithNewShape(HloInstruction* inst,
                                           const Shape& new_shape) {
  *(inst->mutable_shape()) = new_shape;
}

struct InstructionAndShape {
  HloInstruction* inst;
  Shape shape;
  InstructionAndShape(HloInstruction* inst, Shape shape)
      : inst(inst), shape(std::move(shape)) {}
};

static std::vector<InstructionAndShape> FindMismatchedParmeters(
    HloInstruction* inst) {
  std::vector<InstructionAndShape> result;
  switch (inst->opcode()) {
    case HloOpcode::kCall: {
      for (int64 i = 0; i < inst->operand_count(); ++i) {
        auto* param = inst->to_apply()->parameter_instruction(i);
        const Shape& new_shape = inst->operand(i)->shape();
        if (new_shape != param->shape()) {
          result.emplace_back(InstructionAndShape(param, new_shape));
        }
      }
      return result;
    }
    case HloOpcode::kWhile: {
      for (auto* comp : inst->called_computations()) {
        if (inst->operand(0)->shape() !=
            comp->parameter_instruction(0)->shape()) {
          result.emplace_back(InstructionAndShape(
              comp->parameter_instruction(0), inst->operand(0)->shape()));
        }
      }
      return result;
    }
    default: {
      if (inst->called_computations().size()) {
        LOG(FATAL) << "Op calling computations not handled " << inst->name();
      }
      return {};
    }
  }
}

static Status ReplaceInstructionAndPushUsersToQueue(
    std::queue<HloInstruction*>& to_visit, HloInstruction* inst,
    const Shape& new_shape, const CallGraph& call_graph) {
  ReplaceInstructionWithNewShape(inst, new_shape);
  for (auto* user : inst->users()) {
    to_visit.emplace(user);
  }
  // If change the shape of a root instruction we
  // have changed the shape of it's callers so put them
  // in the list
  if (inst == inst->parent()->root_instruction()) {
    CHECK(!inst->parent()->IsEntryComputation());
    for (const auto& callsite :
         call_graph.GetNode(inst->parent()).caller_callsites()) {
      to_visit.emplace(callsite.instruction());
    }
  }

  return Status::OK();
}

struct HoistedOutput {
  absl::flat_hash_map<HloInstruction*, HloInstruction*> hoisted;
  HloInstruction* new_while_op;
  HoistedOutput(absl::flat_hash_map<HloInstruction*, HloInstruction*> hoisted,
                HloInstruction* new_while_op)
      : hoisted(std::move(hoisted)), new_while_op(new_while_op) {}
};

// Looking to take parameter out of while body with a transform like
// Before:
// init = Tuple(A, B, C)
// (...) while(init), body=body
//
// body {
//   P = parameter(0)
//   invariant-gte = get-tuple-element(P), index=0
//   non-hoisted-op = reshape(invariant-gte)
//   use-specified-in-params = dynamic-update(buffer, non-hoisted-op)
//   ... // rest of while body
// }
//
// And we are going to transform it to
// After:
// new-A = reshape(A)
// init = Tuple(new-A, B, C)
// (...) while(init), body=body
//
// body {
//   P = parameter(0)
//   invariant-gte = get-tuple-element(P), index=0
//   use-specified-in-params = dynamic-update(buffer, invariant-gte)
//   ... // rest of while body
// }
// So that now the direct input to the dynamic-update is in the outer
// computation and can be used by any users of the while
StatusOr<HoistedOutput> HoistBroardcastInputs(
    HloInstruction* while_op, std::vector<BroadcastAndSlice>& params) {
  absl::flat_hash_map<HloInstruction*, HloInstruction*> hoisted;
  // Don't need to be strict about unhoisted being loop invariant as it's
  // not used for control flow in creating the copies. Need to not include
  // constants as upstream doesn't as a compile time optimisation and
  // checks there are none in the hash
  absl::flat_hash_set<HloInstruction*> unhoisted =
      FindLoopConstants(while_op->while_body(), false);

  // instructions_to_replace[i] is hoisted into a loop invariant instruction
  // replacement_instructions[i].
  std::vector<HloInstruction*> instructions_to_replace;
  std::vector<HloInstruction*> replacement_instructions;

  for (auto& param : params) {
    auto* to_hoist = param.broadcast.dynamic_update->mutable_operand(1);
    // This erasing is because upstreams function checks that
    // to hoist isn't in unhoisted, (though if it weren't for
    // that check it would be fine as it doesn't use the
    // unhoisted set)
    if (!hoisted.contains(to_hoist)) {
      unhoisted.erase(to_hoist);
      WhileLoopInvariantCodeMotion::CreateLoopInvariantCopy(
          &hoisted, &unhoisted, while_op, to_hoist);
      instructions_to_replace.push_back(to_hoist);
      replacement_instructions.push_back(FindOrDie(hoisted, to_hoist));
    }
    // From now on when looking at the params this will point to the
    // hoisted input
    param.broadcast.dynamic_update = FindOrDie(hoisted, to_hoist);
  }
  TF_ASSIGN_OR_RETURN(auto live_in, WhileUtil::MakeInstructionsLiveIn(
                                        while_op, replacement_instructions));

  for (int64 i = 0; i < instructions_to_replace.size(); i++) {
    auto* old_inst = FindOrDie(live_in.while_body_instruction_map,
                               instructions_to_replace[i]);
    TF_RETURN_IF_ERROR(
        live_in.new_while_instr->while_body()->ReplaceInstruction(
            old_inst, live_in.while_body_live_in_values[i]));
  }
  return HoistedOutput(std::move(hoisted), live_in.new_while_instr);
}

Status ReplaceAllUsesWithDifferentShape(HloInstruction* old_inst,
                                        absl::Span<HloInstruction* const> users,
                                        HloInstruction* new_producer) {
  for (HloInstruction* user : users) {
    TF_RETURN_IF_ERROR(
        old_inst->ReplaceUseWithDifferentShape(user, new_producer));
  }

  if (old_inst->parent() &&
      old_inst->parent()->root_instruction() == old_inst) {
    old_inst->parent()->set_root_instruction(new_producer,
                                             /*accept_different_shape=*/true);
  }
  return Status::OK();
}

HloInstruction* CreateNewTupleOutput(
    const HoistedOutput& while_op,
    const std::vector<BroadcastAndSlice>& params) {
  auto* while_inst = while_op.new_while_op;
  const auto orig_users = while_inst->users();
  absl::flat_hash_map<int64, HloInstruction*> to_replace_outputs;
  for (const auto& param : params) {
    // The dynamic slice in the broadcast was updated to the hoisted instruction
    // in the hoisting phase
    to_replace_outputs.emplace(param.broadcast.input_index,
                               param.broadcast.dynamic_update);
  }
  auto* comp = while_inst->parent();
  std::vector<HloInstruction*> tuple_operands(
      ShapeUtil::TupleElementCount(while_inst->shape()), nullptr);
  for (int64 i = 0; i < tuple_operands.size(); ++i) {
    auto it = to_replace_outputs.find(i);
    if (it == to_replace_outputs.end()) {
      tuple_operands[i] =
          comp->AddInstruction(HloInstruction::CreateGetTupleElement(
              ShapeUtil::GetTupleElementShape(while_inst->shape(), i),
              while_inst, i));
    } else {
      tuple_operands[i] = it->second;
    }
  }
  auto* output =
      comp->AddInstruction(HloInstruction::CreateTuple(tuple_operands));
  ReplaceAllUsesWithDifferentShape(while_inst, orig_users, output);
  return output;
}

Status RemoveTensorListBroadcasts(HloInstruction* inst,
                                  std::vector<BroadcastAndSlice>& params) {
  VLOG(10) << "Removing " << absl::StrJoin(params, ", ", ToStringFormatter())
           << " from " << inst->name();
  // Hoist the input and remove the broadcast
  TF_ASSIGN_OR_RETURN(const auto output, HoistBroardcastInputs(inst, params));
  // Create a new tuple with the similar shape as the old while that
  // is then used by all subsequent instructions. The indices
  // corresponding to broadcasts are replaced by the hoisted
  // instruction
  auto* tuple_output = CreateNewTupleOutput(output, params);
  TF_RETURN_IF_ERROR(
      PoplarWhileLoopOptimiser::PropagateNewShapes({tuple_output}));
  // We are not replacing the dynamic slice but letting the algebraic
  // simplifier do that for us
  return Status::OK();
}

}  // namespace

Status PoplarWhileLoopOptimiser::PropagateNewShapes(
    const std::vector<HloInstruction*>& instructions_with_new_shapes) {
  std::queue<HloInstruction*> to_visit;
  for (auto* inst : instructions_with_new_shapes) {
    for (auto* user : inst->users()) {
      to_visit.push(user);
    }
  }
  if (to_visit.empty()) {
    return Status::OK();
  }
  auto call_graph = CallGraph::Build(to_visit.front()->GetModule());
  while (to_visit.size()) {
    auto* inst = to_visit.front();
    to_visit.pop();
    TF_ASSIGN_OR_RETURN(auto new_shape, WorkOutShapeFromOperands(inst));
    if (inst->shape() != new_shape) {
      TF_RETURN_IF_ERROR(ReplaceInstructionAndPushUsersToQueue(
          to_visit, inst, new_shape, *call_graph));
    }
    auto mismatched_params = FindMismatchedParmeters(inst);
    for (auto& param : mismatched_params) {
      TF_RETURN_IF_ERROR(ReplaceInstructionAndPushUsersToQueue(
          to_visit, param.inst, param.shape, *call_graph));
    }
  }
  return Status::OK();
}

StatusOr<bool> PoplarWhileLoopOptimiser::Run(HloModule* module) {
  bool changed = false;
  for (auto* comp : module->MakeComputationPostOrder()) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      if (inst->opcode() != HloOpcode::kWhile) {
        continue;
      }
      auto slice_only_broadcast_params = FindAllValidBroadcasts(inst);
      SelectInvariantBroadcasts(inst, slice_only_broadcast_params);
      if (slice_only_broadcast_params.empty()) {
        continue;
      }
      TF_RETURN_IF_ERROR(
          RemoveTensorListBroadcasts(inst, slice_only_broadcast_params));
      changed = true;
    }
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
