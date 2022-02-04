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
#include "tensorflow/compiler/plugin/poplar/driver/tools/slice_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/while_util.h"

namespace xla {

namespace m = match;

namespace poplarplugin {

namespace {

void FindLoopConstants(HloComputation* comp,
                       absl::flat_hash_set<HloInstruction*>& result) {
  // we are not finding the dropout ones because this constant looking
  // is too restrictive. TODO fix this
  for (auto* inst : comp->MakeInstructionPostOrder()) {
    switch (inst->opcode()) {
      // Note this is meant to match NotWorthHoistingIndividually
      // inside WhileLoopInvariantCodeMotion
      case HloOpcode::kConstant:
      case HloOpcode::kBitcast:
      case HloOpcode::kBroadcast:
      case HloOpcode::kIota:
      case HloOpcode::kReshape:
      case HloOpcode::kReverse:
      case HloOpcode::kSlice:
      case HloOpcode::kTranspose:
      case HloOpcode::kTuple: {
        const bool all_const_inputs = absl::c_all_of(
            inst->operands(),
            [&](HloInstruction* operand) { return result.contains(operand); });
        if (all_const_inputs) {
          result.emplace(inst);
        }
        break;
      }
      default: {}
    }
  }
}

absl::flat_hash_set<HloInstruction*> FindLoopConstants(HloComputation* comp) {
  absl::flat_hash_set<HloInstruction*> result;
  auto gtes = WhileUtil::GetInvariantGTEsForWhileBody(*comp);
  result.insert(gtes.begin(), gtes.end());
  FindLoopConstants(comp, result);
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
  SliceAndIndex(int64 input_index, HloInstruction* dynamic_update)
      : input_index(input_index), dynamic_update(dynamic_update) {}
  std::string ToString() const {
    return absl::StrCat("{", input_index, ", ", dynamic_update->name(), ",",
                        dynamic_update->shape().ToString(), "}");
  }
};

// Formatter that converts to string by calling the name attribute,
// works for pointes as well as references
class NameFormatter {
  template <class T>
  void dispatch(std::string* out, const T* const t, std::true_type) const {
    return absl::AlphaNumFormatter()(out, t->name());
  }
  template <class T>
  void dispatch(std::string* out, const T& t, std::false_type) const {
    return absl::AlphaNumFormatter()(out, t.name());
  }

 public:
  template <class T>
  void operator()(std::string* out, const T& t) const {
    return dispatch(out, t, std::is_pointer<T>());
  }
};

// Formatter that converts to string by calling the ToString attribute,
// works for pointes as well as references
class ToStringFormatter {
  template <class T>
  void dispatch(std::string* out, const T* const t, std::true_type) const {
    return absl::AlphaNumFormatter()(out, t->ToString());
  }
  template <class T>
  void dispatch(std::string* out, const T& t, std::false_type) const {
    return absl::AlphaNumFormatter()(out, t.ToString());
  }

 public:
  template <class T>
  void operator()(std::string* out, const T& t) const {
    return dispatch(out, t, std::is_pointer<T>());
  }
};

struct UsesAndIntermediates {
  absl::InlinedVector<HloInstruction*, 1> uses;
  absl::InlinedVector<HloInstruction*, 4> intermediates;
  std::string ToString() const {
    return absl::StrCat("Uses {", absl::StrJoin(uses, ", ", NameFormatter()));
  }
};

struct BroadcastAndSlice {
  SliceAndIndex broadcast;
  UsesAndIntermediates uses;
  std::string ToString() const {
    return absl::StrCat("{", broadcast.ToString(), ", ",
                        absl::StrJoin(uses.uses, ", ", NameFormatter()));
  }
  BroadcastAndSlice(SliceAndIndex broadcast, UsesAndIntermediates uses)
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
    // Check no other instruction inside the while body are using the update
    if (updated_data->users().size() != 1U ||
        updated_data->users()[0] != root) {
      VLOG(10) << "Dynamic update is used else where in the body";
      continue;
    }
    // We have the only user of the gte element as a dynamic update, now
    // check that this is fed to the root tuple at the same index as the gte
    int64 parameter_index = candidate_gte->tuple_index();
    if (root->mutable_operand(parameter_index) != updated_data) {
      VLOG(10) << "Not at correct index to be an inplace update "
               << parameter_index;
      continue;
    }
    // We must now check that the slice is into the outer most dimension,
    // that the input is a loop invariant, and that the index is the loop
    // counter.
    auto* input_slice = updated_data->mutable_operand(1);
    if (!InputIsLoopInvariant(input_slice, loop_invariants)) {
      VLOG(10) << "Input is not a loop invariant " << input_slice->name();
      continue;
    }
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
    result.emplace_back(candidate_gte->tuple_index(), updated_data);
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
  absl::InlinedVector<HloInstruction*, 4> path;
  absl::flat_hash_set<HloComputation*> while_bodies;

  void DefaultAction(HloInstruction* user) {
    result.emplace_back(user);
    path.emplace_back(user);
  }
  void GTEAction(HloInstruction* user) { path.emplace_back(user); }
  void TupleAction(HloInstruction* user) { path.emplace_back(user); }
  void WhileAction(HloInstruction* user) {
    path.emplace_back(user);
    while_bodies.emplace(user->while_body());
  }
};

UsesAndIntermediates FindNonTrivialUses(HloInstruction* inst,
                                        std::stack<int64> starting_gte_stack) {
  FindUsesTemplate args;
  FindNonTrivialUses(inst, args, starting_gte_stack);
  return {std::move(args.result), std::move(args.path)};
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
    VLOG(10) << "Found candidate";
    result.emplace_back(std::move(broadcast), std::move(users));
  }
  return result;
}

void RemoveTensorListBroadcasts(HloInstruction* inst,
                                const std::vector<BroadcastAndSlice>& params) {
  // Does nothing for now, as this diff only adds detection phase
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
      // Detect loop constants being pushed into tensor list
      auto broadcast_params = FindTensorListBroadcasts(inst);
      // Filter these tensor lists down to ones only used
      // by slice instructions
      auto slice_only_broadcast_params =
          FindBroadcastsOnlyUsedBySlices(inst, std::move(broadcast_params));
      if (slice_only_broadcast_params.empty()) {
        continue;
      }
      RemoveTensorListBroadcasts(inst, slice_only_broadcast_params);
      changed = true;
    }
  }
  return changed;
}

int64 PoplarWhileLoopOptimiser::CountOptimisations(HloModule* module) const {
  int64 result = 0;
  for (auto* comp : module->MakeComputationPostOrder()) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      if (inst->opcode() != HloOpcode::kWhile) {
        continue;
      }
      auto broadcast_params = FindTensorListBroadcasts(inst);
      auto slice_only_broadcast_params =
          FindBroadcastsOnlyUsedBySlices(inst, std::move(broadcast_params));
      VLOG(1) << "Can eliminate for " << inst->name() << ": {"
              << absl::StrJoin(slice_only_broadcast_params, ", ",
                               ToStringFormatter())
              << "}";
      result += slice_only_broadcast_params.size();
    }
  }
  return result;
}

}  // namespace poplarplugin
}  // namespace xla
