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
#include <string>
#include <utility>
#include <vector>
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_analysis.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/uninitialised.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/print_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/while_loop_optimisation_utils.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/service/while_loop_invariant_code_motion.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/while_util.h"

using xla::poplarplugin::WhileLoopOptimisationUtils::BroadcastAndSlice;
using xla::poplarplugin::WhileLoopOptimisationUtils::FindAllValidBroadcasts;
using xla::poplarplugin::WhileLoopOptimisationUtils::FindLoopConstants;

namespace xla {

namespace poplarplugin {

namespace {

bool InputIsLoopInvariant(
    HloInstruction* input,
    const absl::flat_hash_set<HloInstruction*>& constants) {
  return constants.contains(input);
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
      for (int64_t i = 0; i < inst->operand_count(); ++i) {
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

  for (int64_t i = 0; i < instructions_to_replace.size(); i++) {
    auto* old_inst = FindOrDie(live_in.while_body_instruction_map,
                               instructions_to_replace[i]);
    TF_RETURN_IF_ERROR(
        live_in.new_while_instr->while_body()->ReplaceInstruction(
            old_inst, live_in.while_body_live_in_values[i]));
  }
  return HoistedOutput(std::move(hoisted), live_in.new_while_instr);
}

HloInstruction* CreateNewTupleOutput(
    const HoistedOutput& while_op,
    const std::vector<BroadcastAndSlice>& params) {
  auto* while_inst = while_op.new_while_op;
  const auto orig_users = while_inst->users();
  absl::flat_hash_map<int64_t, HloInstruction*> to_replace_outputs;
  for (const auto& param : params) {
    // The dynamic slice in the broadcast was updated to the hoisted instruction
    // in the hoisting phase
    to_replace_outputs.emplace(param.broadcast.input_index,
                               param.broadcast.dynamic_update);
  }
  auto* comp = while_inst->parent();
  std::vector<HloInstruction*> tuple_operands(
      ShapeUtil::TupleElementCount(while_inst->shape()), nullptr);
  for (int64_t i = 0; i < tuple_operands.size(); ++i) {
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
  while_inst->ReplaceAllUsesWithDifferentShape(orig_users, output);
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

bool MakeInputsUninitialised(HloInstruction* while_loop,
                             const BroadcastAndSlice& broadcast,
                             int64_t& num_uninitialised) {
  int64_t index = broadcast.broadcast.input_index;
  auto* while_init = while_loop->mutable_operand(0);
  auto* orig = while_init->operand(index);
  if (IsPoplarInstruction(PoplarOp::Uninitialised)(orig)) {
    return false;
  }
  auto* comp = while_init->parent();
  auto* uninit = comp->AddInstruction(
      CreateUninitialisedInstruction(orig->shape(), num_uninitialised));
  while_init->ReplaceOperandWith(index, uninit);
  return true;
}

bool MakeInputsUninitialised(
    HloInstruction* while_loop,
    const std::vector<BroadcastAndSlice>& slice_only_broadcast_params,
    int64_t& num_uninitialised) {
  bool changed = false;
  for (const auto& broadcast : slice_only_broadcast_params) {
    changed |=
        MakeInputsUninitialised(while_loop, broadcast, num_uninitialised);
  }
  return changed;
}

struct IndexAndInstruction {
  HloInstruction* instruction;
  int64_t index;
  HloInstruction* loop;
  IndexAndInstruction(HloInstruction* instruction, int64_t index,
                      HloInstruction* loop)
      : instruction(instruction), index(index), loop(loop) {}
  std::string ToString() const {
    return absl::StrCat(loop->name(), ", ", index, ", ",
                        instruction->ToString());
  }
};

std::vector<IndexAndInstruction> FindWhileInvariants(HloInstruction* loop) {
  auto gtes = WhileUtil::GetInvariantGTEsForWhileBody(*(loop->while_body()));
  std::vector<IndexAndInstruction> result;
  result.reserve(gtes.size());
  absl::c_transform(gtes, std::back_inserter(result), [&](HloInstruction* gte) {
    return IndexAndInstruction(gte, gte->tuple_index(), loop);
  });
  return result;
}

std::vector<IndexAndInstruction> FindRepeatInvariants(HloInstruction* loop) {
  std::vector<IndexAndInstruction> result;
  auto* root = loop->to_apply()->root_instruction();
  for (int64_t i = 0; i < root->operand_count(); ++i) {
    HloInstruction* operand = root->mutable_operand(i);
    if (operand->opcode() == HloOpcode::kParameter &&
        operand->parameter_number() == i) {
      result.emplace_back(operand, i, loop);
    }
  }
  return result;
}

std::vector<IndexAndInstruction> FindLoopInvariants(HloInstruction* loop) {
  return loop->opcode() == HloOpcode::kWhile ? FindWhileInvariants(loop)
                                             : FindRepeatInvariants(loop);
}

bool IsLoop(HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kWhile || IsRepeatLoop(inst);
}

bool InsideWhileLoopAndInvariant(HloInstruction* dot, HloInstruction* operand,
                                 const CallGraph& call_graph) {
  const auto& callsites = call_graph.GetNode(dot->parent()).caller_callsites();
  if (callsites.size() != 1) {
    return false;
  }
  if (!IsLoop(callsites[0].instruction())) {
    return false;
  }
  // This is not very efficient as I'm finding all the loop invariants
  // when I could just pattern match this instruction, but as I already
  // had this function and this isn't a hot path, I think I can get
  // away with it.
  auto loop_invariants = FindLoopInvariants(callsites[0].instruction());
  auto it =
      absl::c_find_if(loop_invariants, [&](const IndexAndInstruction& val) {
        // hack to get around fact we don't hoist reshapes, when we move
        // the the new expensive LICM we can hoist them there and remove
        // this.
        if (operand->opcode() == HloOpcode::kReshape) {
          return val.instruction == operand->operand(0);
        }
        return val.instruction == operand;
      });
  // If it's not invariant it could be written to in the while loop by
  // an instruction that has another mapping.
  return it != loop_invariants.end();
}

google::protobuf::RepeatedField<std::int64_t> GetContractingDimensions(
    const InputLocation& input) {
  auto* inst = input.instruction;
  return input.operand_index == 0
             ? inst->dot_dimension_numbers().lhs_contracting_dimensions()
             : inst->dot_dimension_numbers().rhs_contracting_dimensions();
}

bool IsWorthRemapping(const AllocationGroup& group) {
  absl::flat_hash_map<HloInstruction*, int64_t> dot_to_contracting_dim;
  for (const auto& input : group.inputs_only) {
    if (input.instruction->opcode() != HloOpcode::kDot) {
      continue;
    }
    const auto contracting_dims = GetContractingDimensions(input);
    if (contracting_dims.size() != 1) {
      // Will try handle these cases in a separate diff
      return false;
    }
    dot_to_contracting_dim.emplace(input.instruction, contracting_dims[0]);
  }
  // If they all have the same contracting dim then they are all ok to
  // have the same mapping
  if (absl::c_all_of(dot_to_contracting_dim,
                     [&](const std::pair<HloInstruction*, int64_t>& val) {
                       return val.second ==
                              dot_to_contracting_dim.begin()->second;
                     })) {
    return false;
  }
  // We could make this more generic but for now take the simple
  // and common case and only consider case we have 2 dot products,
  // and I'll early out the optimisation otherwise.
  if (dot_to_contracting_dim.size() != 2) {
    return false;
  }
  auto* dotA = dot_to_contracting_dim.begin()->first;
  auto* dotB = (++dot_to_contracting_dim.begin())->first;
  if (dotA->parent() == dotB->parent()) {
    // We are trying to insert a remap between these instructions
    // but outside the while loop. Not possible if they
    // are in the same computation
    return false;
  }
  return true;
}

Status RemapInput(const InputLocation& location, const CallGraph& call_graph) {
  auto* operand = location.instruction->mutable_operand(location.operand_index);
  auto* input = operand->opcode() == HloOpcode::kReshape
                    ? operand->mutable_operand(0)
                    : operand;
  int64_t index = input->opcode() == HloOpcode::kParameter
                      ? input->parameter_number()
                      : input->tuple_index();
  const auto& callsites =
      call_graph.GetNode(operand->parent()).caller_callsites();
  auto* loop = callsites[0].instruction();
  auto* init =
      loop->opcode() == HloOpcode::kWhile ? loop->mutable_operand(0) : loop;
  auto* to_remap = init->mutable_operand(index);
  auto* remapped =
      to_remap->parent()->AddInstruction(HloInstruction::CreateUnary(
          to_remap->shape(), HloOpcode::kCopy, to_remap));
  TF_RETURN_IF_ERROR(SetCopyCloneMethod(
      remapped,
      ShapeTree<CloneMethod>(remapped->shape(),
                             CloneMethod_PreserveOrderUnlessAliases)));
  init->ReplaceOperandWith(index, remapped);
  return Status::OK();
}

InputLocation ChooseCandidate(std::vector<InputLocation>& candidates) {
  if (candidates.size() == 1) {
    return candidates[0];
  }
  // At this stage it's not so important which candidate we pick, though
  // as the vector is created from a hash_set don't just pick the first.
  // When we start targeting preArrangeMatMulInputRHS we should try aim
  // for the one in the bwd pass (and when I say backwards pass we
  // obviously don't know what that is in HLO so need to change to be
  // looking at contracting dims to make these decisions)
  // Another alternative is we could see if there is a copy in the
  // allocation group that only affects one of the dots and we
  // could try making that allocate
  absl::c_sort(candidates, [&](const InputLocation& A, const InputLocation& B) {
    const auto dims_a = GetContractingDimensions(A);
    const auto dims_b = GetContractingDimensions(B);
    const auto span_a = absl::MakeSpan(dims_a.begin(), dims_a.size());
    const auto span_b = absl::MakeSpan(dims_b.begin(), dims_b.size());
    return span_a < span_b;
  });
  return candidates.back();
}

StatusOr<bool> MaybeRemapOperand(const AllocationGroup& group,
                                 const CallGraph& call_graph) {
  std::vector<InputLocation> remapping_canditate;
  absl::c_copy_if(group.inputs_only, std::back_inserter(remapping_canditate),
                  [&](const InputLocation& input) {
                    return InsideWhileLoopAndInvariant(
                        input.instruction,
                        input.instruction->mutable_operand(input.operand_index),
                        call_graph);
                  });
  if (remapping_canditate.empty()) {
    return false;
  }
  // Assuming there are at most 2 candidates here. This should really
  // be generalised at some point.
  TF_RETURN_IF_ERROR(
      RemapInput(ChooseCandidate(remapping_canditate), call_graph));
  return true;
}

// If only one dot product then there is nothing for it to
// clash with
bool HasEnoughDotProducts(const AllocationGroup& group) {
  return 1 < absl::c_count_if(group.inputs_only, [](const InputLocation& inst) {
           return inst.instruction->opcode() == HloOpcode::kDot;
         });
}

}  // namespace

StatusOr<bool> PoplarWhileLoopRemapper::Run(HloModule* module) {
  bool changed = false;
  TF_ASSIGN_OR_RETURN(auto allocation_groups,
                      AllocationGroups::CreateAllocationGroups(module));
  auto call_graph = CallGraph::Build(module);
  FilterInPlace(allocation_groups.groups, HasEnoughDotProducts);
  FilterInPlace(allocation_groups.groups, IsWorthRemapping);
  for (const auto& group : allocation_groups.groups) {
    TF_ASSIGN_OR_RETURN(bool changed_, MaybeRemapOperand(group, *call_graph));
    changed |= changed_;
  }
  return changed;
}

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
      changed |= MakeInputsUninitialised(inst, slice_only_broadcast_params,
                                         num_uninitialised_);
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
