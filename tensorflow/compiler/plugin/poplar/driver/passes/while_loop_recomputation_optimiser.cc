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

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"

#include "tensorflow/compiler/plugin/poplar/driver/passes/while_loop_recomputation_optimiser.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/print_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/while_loop_optimisation_utils.h"

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

using xla::poplarplugin::WhileLoopOptimisationUtils::BroadcastAndSlice;
using xla::poplarplugin::WhileLoopOptimisationUtils::FindAllValidBroadcasts;

namespace xla {

namespace poplarplugin {

namespace {

struct RecomputationInfo final {
  absl::flat_hash_set<HloInstruction*> start_points;

  // The intermediate computation(s) applied before the dynamic-update-slice
  // to be removed, i.e. user of the recomputable instruction that is an
  // operand to the dynamic-update-slice.
  // Keys are the dynamic-slice instructions.
  std::vector<HloInstruction*> usage_path;

  HloInstruction* dynamic_update_slice = nullptr;

  bool Valid() const {
    return !(start_points.empty() || usage_path.empty() ||
             dynamic_update_slice == nullptr);
  }
};

bool operator==(const RecomputationInfo& lhs, const RecomputationInfo& rhs) {
  return lhs.dynamic_update_slice == rhs.dynamic_update_slice &&
         lhs.start_points == rhs.start_points &&
         lhs.usage_path == rhs.usage_path;
}

bool RecomputationIsAffordable(const RecomputationInfo& ri,
                               unsigned int ds_count) {
  constexpr unsigned int elementwise_cost = 1;
  constexpr unsigned int with_copy_cost = 2 * elementwise_cost;

  // Number of operations with copy is the initial dynamic-update-slice plus
  // the number of dynamic-slice operations that use the result.
  const auto max_cost = (ds_count + 1) * with_copy_cost;

  auto sum_path = [elementwise_cost, with_copy_cost,
                   max_cost](const std::vector<HloInstruction*>& path) {
    unsigned int cost = 0;
    for (const auto* const inst : path) {
      switch (inst->opcode()) {
        // Operations that are "free".
        case HloOpcode::kTuple:
        case HloOpcode::kGetTupleElement:
        case HloOpcode::kSlice:
        case HloOpcode::kReshape: {
          continue;
        }
        // Operations that may incur an additional copy.
        case HloOpcode::kDynamicSlice:
        case HloOpcode::kDynamicUpdateSlice: {
          cost += with_copy_cost;
          continue;
        }
        // Elementwise operations.
        case HloOpcode::kTanh:
        case HloOpcode::kSin: {
          cost += elementwise_cost;
          continue;
        }
        // Operation is not whitelisted.
        default: { return max_cost + 1; }
      }
    }
    return cost;
  };

  const auto cost = ds_count * with_copy_cost + sum_path(ri.usage_path);
  return cost <= max_cost;
}

absl::flat_hash_set<HloInstruction*> GetInputs(
    const std::vector<BroadcastAndSlice>& broadcasts) {
  absl::flat_hash_set<HloInstruction*> result;
  for (const auto& broadcast : broadcasts) {
    auto* input = broadcast.broadcast.dynamic_update->mutable_operand(1);
    if (input->opcode() == HloOpcode::kReshape && input->user_count() == 1) {
      input = input->mutable_operand(0);
    }
    result.emplace(input);
  }
  return result;
}

RecomputationInfo FindRecomputationInfo(
    HloInstruction* const start_point,
    const absl::flat_hash_set<HloInstruction*>& update_inputs,
    const absl::flat_hash_set<HloInstruction*>& constants) {
  RecomputationInfo info;

  // depth, to_visit
  std::vector<std::pair<int64_t, HloInstruction*>> to_visit;
  to_visit.emplace_back(0, start_point);

  absl::flat_hash_set<HloInstruction*> visited;

  while (!to_visit.empty()) {
    const auto state = to_visit.back();
    const auto state_depth = state.first;
    auto* const state_inst = state.second;
    to_visit.pop_back();

    auto res = visited.insert(state_inst);
    if (!res.second) {
      continue;
    }

    // If we have found an instruction in the path that cannot be recomputed.
    constexpr int64_t max_depth = 3;
    if (state_depth >= max_depth || state_inst->HasSideEffect()) {
      return RecomputationInfo();
    }

    // If the next instruction to visit is one of the slice updates,
    // then we have likely hit another TL. If it's from another TL,
    // has no operands or is constant, then add it as a starting point.
    if (update_inputs.contains(state_inst) || constants.contains(state_inst) ||
        state_inst->operand_count() == 0) {
      info.start_points.emplace(state_inst);
      continue;
    }

    // We can only increment the depth if the are several operands
    // as that is when to_visit starts to explode.
    const int64_t increment_depth = state_inst->operand_count() > 1;
    for (auto* const operand : state_inst->operands()) {
      to_visit.emplace_back(state_depth + increment_depth, operand);
    }
  }
  return info;
}

StatusOr<absl::flat_hash_map<HloInstruction*, std::vector<RecomputationInfo>>>
FindRecomputableTensorLists(const std::vector<BroadcastAndSlice>& broadcasts) {
  // Constants should be consistent across broadcasts.
  const auto* const bc_module =
      broadcasts[0].broadcast.dynamic_update->parent();
  absl::flat_hash_set<HloInstruction*> constants;
  for (auto* const inst : bc_module->instructions()) {
    if (inst->opcode() == HloOpcode::kConstant) {
      constants.insert(inst);
    }
  }

  // Get the set of all slice update tensors.
  const auto start_points = GetInputs(broadcasts);

  // For each broadcast instruction, figure out the starting points and path
  // to that instruction.
  absl::flat_hash_map<HloInstruction*, std::vector<RecomputationInfo>>
      recomputation_infos;
  for (const auto& broadcast : broadcasts) {
    std::vector<HloInstruction*> path_to_dynamic_update;
    std::function<HloInstruction*(HloInstruction* const)> find_path;
    find_path = [&find_path, &path_to_dynamic_update,
                 &start_points](HloInstruction* const op) {
      // At present, only unary ops can be in the path to the
      // dynamic-update-slice. We assume here that the last op that we
      // encounter before a non-unary op is our slice update (before any)
      // unary ops are applied. Currently this only applies if the non-unary
      // op is a parameter.
      if (op->operand_count() != 1) {
        if (op->opcode() == HloOpcode::kParameter ||
            (start_points.find(op) != start_points.end() &&
             op->opcode() == HloOpcode::kDynamicUpdateSlice)) {
          return path_to_dynamic_update.back();
        }
        path_to_dynamic_update.clear();
        return static_cast<HloInstruction*>(nullptr);
      }

      path_to_dynamic_update.push_back(op);
      return find_path(op->mutable_operand(0));
    };

    // Get the update slice and build the path of ops between it and
    // the dynamic-update-slice. This will be a nullptr if we have an
    // invalid path (i.e. contains a non-unary operation that is not a
    // while loop parameter).
    auto* update_slice =
        find_path(broadcast.broadcast.dynamic_update->mutable_operand(1));
    if (update_slice == nullptr) {
      continue;
    }

    // For each slice update, determine if it is recomputable. It is
    // recomputable if the starting instructions in the path are from
    // another TL, take no operands or are constants.
    auto info = FindRecomputationInfo(update_slice, start_points, constants);

    // For later cross-referencing when replacing.
    info.dynamic_update_slice = broadcast.broadcast.dynamic_update;

    // Store the path - this constitutes the set of intermediate instructions
    // to be moved outside of the loop body (excluding, of course,
    // the recomputable itself and the dynamic-update-slice).
    if (recomputation_infos.find(update_slice) == recomputation_infos.end()) {
      recomputation_infos[update_slice] = {};
    }

    // Check for side effects.
    if (absl::c_any_of(path_to_dynamic_update, [](const auto* const a) {
          return a->HasSideEffect();
        })) {
      path_to_dynamic_update.clear();
    }

    absl::c_reverse(path_to_dynamic_update);
    info.usage_path = std::move(path_to_dynamic_update);

    if (info.Valid()) {
      recomputation_infos[update_slice].push_back(std::move(info));
    }
  }

  return recomputation_infos;
}

StatusOr<bool> ReplaceDynamicSliceUses(
    const absl::InlinedVector<HloInstruction*, 1>& uses,
    const absl::InlinedVector<HloInstruction*, 1>& src_uses,
    const RecomputationInfo& ri, HloInstruction* const slice_update,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>& replaced) {
  bool changed = false;
  for (auto* const ds : uses) {
    // Find a dynamic-slice in the uses of the source dynamic-update-slice
    // that has compatible indices from which we can derive the new
    // dynamic-slice.
    auto src_dynamic_slice =
        absl::c_find_if(src_uses, [ds](const auto* const u) {
          return ds->operand(1) == u->operand(1) &&  // start indices
                 ds->operand(2) == u->operand(2);    // slice indices
        });

    if (src_dynamic_slice == src_uses.end()) {
      continue;
    }

    // Check we have only unary intermediate instructions.
    if (absl::c_any_of(ri.usage_path, [slice_update](const auto* const a) {
          if (a == slice_update) {
            return false;
          }

          return a->operand_count() > 1;
        })) {
      return false;
    }

    // Do the replacement.
    auto* const comp = (*src_dynamic_slice)->parent();
    HloInstruction* param = *src_dynamic_slice;
    HloInstruction* new_inst = nullptr;
    for (auto* f : ri.usage_path) {
      if (f == slice_update) {
        continue;
      }

      if (new_inst != nullptr) {
        param = new_inst;
      }

      // Before adding a new instruction based on the old, check that it
      // hasn't already been replaced. This can happen if the slice update
      // is used by multipe dynamic-slice-updates. I.e. it exists in multiple
      // usage paths.
      if (replaced.find(f) != replaced.end()) {
        new_inst = replaced[f];
        continue;
      }

      // Clone the instruction from the path and add it to the computation.
      new_inst =
          comp->AddInstruction(f->CloneWithNewOperands(f->shape(), {param}));
      replaced[f] = new_inst;
    }

    // Clone the instruction from the path and add it to the computation.
    // TF_RETURN_IF_ERROR(comp->ReplaceInstruction(ds, new_inst));
    for (auto* const user : ds->users()) {
      const auto op_index = user->operand_index(ds);
      TF_RETURN_IF_ERROR(user->ReplaceOperandWith(op_index, new_inst));
    }
    changed = true;
  }
  return changed;
}

StatusOr<bool> ReplaceInstructions(
    const std::vector<BroadcastAndSlice>& broadcasts,
    const absl::flat_hash_map<HloInstruction*, std::vector<RecomputationInfo>>&
        recomputation_infos,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>& replaced) {
  bool changed = false;

  auto find_bcast_for_dus = [broadcasts](const HloInstruction* const dus) {
    auto bc = absl::c_find_if(broadcasts, [dus](const auto& a) {
      return a.broadcast.dynamic_update == dus;
    });

    return bc != broadcasts.end() ? absl::make_optional(*bc) : absl::nullopt;
  };

  for (const auto& [slice_update, infos] : recomputation_infos) {
    // We need to determine if one of the dynamic-update-slice's is just a
    // straightforward call on the recomputable. I.e. no intermediate
    // instructions in it's path. This is the dynamic-update-slice that we will
    // be using (via the loop body output) as the input to the new, replaced
    // dynamic-slice call(s).
    const auto src_recomp_info =
        absl::c_find_if(infos, [slice_update](const auto& ri) {
          const auto has_reshape = absl::c_any_of(ri.usage_path, [](auto a) {
            return a->opcode() == HloOpcode::kReshape;
          });
          const auto n = ri.usage_path.size();
          return ri.Valid() && slice_update == ri.usage_path.front() &&
                 (n == 1 || (n == 2 && has_reshape));
        });

    if (src_recomp_info == infos.end()) {
      continue;
    }

    for (const auto& ri : infos) {
      if (ri == *src_recomp_info) {
        continue;
      }

      // Replace each dynamic-slice (outside the loop body, except the one
      // derived from our source dynamic-update-slice, from which we derive
      // replacements) with the sequence of intermediate instructions applied
      // to the source in the loop.
      //
      // See the example in the comment above Run; Xp is the source, tanh is
      // the single intermediate instruction and d is to be rewritten in terms
      // of c; d = tanh(c). Yp should be removed via DCE.
      const auto* const src_dus = src_recomp_info->dynamic_update_slice;
      const auto* const dus = ri.dynamic_update_slice;
      if (dus->parent() != src_dus->parent() ||
          dus->operand(2) != src_dus->operand(2) ||
          dus->operand(3) != src_dus->operand(3)) {
        continue;
      }

      // Check both dynamic-update-slice's have broadcasts.
      const auto src_bcast = find_bcast_for_dus(src_dus);
      const auto bcast = find_bcast_for_dus(dus);
      if (!src_bcast.has_value() || !bcast.has_value()) {
        continue;
      }

      // If the broadcast has no uses, skip.
      const auto& src_uses = src_bcast.value().uses.uses;
      const auto& uses = bcast.value().uses.uses;
      if (uses.empty()) {
        continue;
      }

      // Check all the uses are in the same computation.
      const auto* const use_comp = uses.front()->parent();
      if (!absl::c_all_of(uses, [use_comp](const auto* const a) {
            return a->parent() == use_comp;
          })) {
        continue;
      }

      // Final check is the recomputation cost; i.e. do we gain anything by
      // doing the replacement?
      if (!RecomputationIsAffordable(ri, src_uses.size())) {
        continue;
      }

      // Do the replacement.
      TF_ASSIGN_OR_RETURN(
          const auto status,
          ReplaceDynamicSliceUses(uses, src_uses, ri, slice_update, replaced));
      changed |= status;
    }
  }

  return changed;
}

}  // namespace

/* For a recomputable instruction, inst, traverse it's users to identify the
 * following pattern:
 *
 * inst -> some-operation -> dynamic-update-slice -> ROOT -> dynamic-slice
 *
 * such that some-operation is moved outside of the loop body and is taken
 * on the ending dynamic-slice instead (eliminating the dynamic-update-slice).
 *
 * For example:
 * ****************************** BEFORE PASS *********************************
 *   loop_body {
 *     params = parameter(0)
 *     X = gte(params), index=m
 *     Y = gte(params), index=n
 *     Z = gte(params), index=k
 *
 *     Xp = dynamic-update-slice(X, Z...)
 *     A = tanh(Z)
 *
 *     ...
 *
 *     Yp = dynamic-update-slice(Y, A...)
 *
 *     ROOT res = tuple(Xp, Yp, Z)
 *   }
 *
 *   ENTRY e {
 *
 *     ...
 *
 *     loop_init = tuple(...)
 *     loop = while(loop_init), body=loop_body, condition=...
 *
 *     a = gte(loop), index=0
 *     b = gte(loop), index=1
 *
 *     c = dynamic-slice(a...)
 *     d = dynamic-slice(b...)
 *   }
 *
 * ****************************** AFTER PASS *********************************
 *   loop_body {
 *     params = parameter(0)
 *     X = gte(params), index=m
 *     Y = gte(params), index=n
 *     Z = gte(params), index=k
 *
 *     Xp = dynamic-update-slice(X, Z...)
 *     A = tanh(Z)
 *
 *     ...
 *
 *     ROOT res = tuple(Xp, Z)
 *   }
 *
 *   ENTRY e {
 *
 *     ...
 *
 *     loop_init = tuple(...)
 *     loop = while(loop_init), body=loop_body, condition=...
 *
 *     a = gte(loop), index=0
 *
 *     b = dynamic-slice(a...)
 *     c = tanh(b)
 *   }
 *
 * In the above example, if after the pass Z and A are unused, they will be
 * removed via DCE. The working assumption is that they will be used elsewhere
 * in the loop body.
 *
 * The pass is subject to the following constraints:
 * -> The result of the dynamic-update-slice to be removed (Yp) must only
 *    have the root instruction as a user.
 * -> The operation prior to the dynamic-update-slice to be removed (Yp) should
 *    be either constant, or reachable from an instruction that is saved to a
 *    buffer. Saved to a buffer in the above example as Z is saved to Xp, and
 *    Xp is used in the same computation as Yp; a and b in the entry
 *    computation.
 * -> Indices must be consistent between dynamic-update-slice operations
 *    (Xp and Yp), and between dynamic-slice operations (c and d). However,
 *    index consistency is not required between the two sets of operations.
 * -> All dynamic-slice operations need to be in the same computation to be
 *    recomputable.
 */
StatusOr<bool> PoplarWhileLoopRecomputationOptimiser::Run(HloModule* module) {
  VLOG(2) << "Before " << name() << ":";
  XLA_VLOG_LINES(2, module->ToString(HloPrintOptions::ShortParsable()));

  bool changed = false;

  absl::flat_hash_map<HloInstruction*, HloInstruction*> replaced;
  for (auto* const comp : module->MakeComputationPostOrder()) {
    for (auto* const inst : comp->MakeInstructionPostOrder()) {
      if (inst->opcode() != HloOpcode::kWhile) {
        continue;
      }

      /*
       * TensorList Broadcasts that are *only* used by slice operations *and*
       * are of the form:
       *
       * while_body {
       *   while_loop_params = (...) parameter(0)
       *
       *   idx = ...
       *
       *   loop_counter = T[...] get-tuple-element(
       *     while_loop_params), index=<last_index>
       *
       *   val = T[] constant(...)
       *   X = T[...] reshape(val)
       *
       *   gte = T[...] get-tuple-element(while_loop_params), index=idx
       *   ROOT r = T[...] dynamic-update-slice(gte, X, loop_counter)
       * }
       *
       * The following assumptions are made about the parameter at idx to
       * the while loop:
       *   -> It is a broadcast as dynamic-update-slice writes to the next
       *      index on each iteration.
       *   -> It is often from a TensorList (though not always).
       */
      auto broadcasts = FindAllValidBroadcasts(inst);
      if (broadcasts.empty()) {
        continue;
      }

      TF_ASSIGN_OR_RETURN(const auto recomputable,
                          FindRecomputableTensorLists(broadcasts));
      if (recomputable.empty()) {
        continue;
      }

      TF_ASSIGN_OR_RETURN(
          const auto status,
          ReplaceInstructions(broadcasts, recomputable, replaced));
      changed |= status;
    }
  }

  if (changed) {
    VLOG(2) << "After " << name() << ":";
    XLA_VLOG_LINES(2, module->ToString(HloPrintOptions::ShortParsable()));
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
