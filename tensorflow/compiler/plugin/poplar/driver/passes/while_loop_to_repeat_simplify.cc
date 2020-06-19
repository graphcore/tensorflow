/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/while_loop_to_repeat_simplify.h"

#include <map>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/while_loop_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_analysis.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {
namespace {
// Allow user to change the upper bound of the brute force method
int64 GetMaxLoopTripCount() {
  auto max_trip_count =
      PoplarXlaFlags::Get().while_loop_brute_force_max_trip_count;
  return max_trip_count < 0 ? 128 : max_trip_count;
}

StatusOr<int64> ConvertWhileToRepeat(HloInstruction* while_inst) {
  static const char* err_msg = "Unable to convert this while loop";
  HloComputation* while_condition = while_inst->while_condition();
  HloComputation* while_body = while_inst->while_body();
  // Make sure that this is a while loop with a single conditional of form
  // "cond COMP const". There must be only 4 instructions which prevents
  // detached stateful instructions from being excluded from execution.
  if (while_condition->instruction_count() != 4) {
    return xla::FailedPrecondition("%s", err_msg);
  }

  // The root instruction must be the comparison
  HloInstruction* c_inst = while_condition->root_instruction();
  if (c_inst->opcode() != HloOpcode::kCompare) {
    return xla::FailedPrecondition("%s", err_msg);
  }

  // Only some comparisons are ok
  switch (c_inst->comparison_direction()) {
    case ComparisonDirection::kLt:
    case ComparisonDirection::kLe:
    case ComparisonDirection::kGt:
    case ComparisonDirection::kGe:
      break;
    default:
      return xla::FailedPrecondition("%s", err_msg);
  }

  // Make sure that for the comparison instruction:
  // * LHS is an integer GTE from from parameter 0
  // * RHS is a constant which is an integer
  {
    const bool lhs_is_GTE_param_from_param_0 =
        WhileLoopUtil::IsGTEFromParamIndex(c_inst->operand(0), 0);
    if (!lhs_is_GTE_param_from_param_0) {
      return xla::FailedPrecondition("%s", err_msg);
    }

    const bool rhs_is_integer_const =
        WhileLoopUtil::Is32BitsOrLessIntegerConstant(c_inst->operand(1));
    if (!rhs_is_integer_const) {
      return xla::FailedPrecondition("%s", err_msg);
    }
  }
  HloInstruction* comp_GTE = c_inst->mutable_operand(0);
  int64 tuple_index = comp_GTE->tuple_index();

  HloInstruction* input_tuple = while_inst->mutable_operand(0);
  HloInstruction* init_inst = input_tuple->mutable_operand(tuple_index);

  if (init_inst->opcode() != HloOpcode::kConstant) {
    return xla::FailedPrecondition("%s", err_msg);
  }

  const HloInstruction* limit_inst = c_inst->operand(1);

  if (limit_inst->opcode() != HloOpcode::kConstant) {
    return xla::FailedPrecondition("%s", err_msg);
  }

  // Find corresponding GTE in the body
  HloInstruction* body_GTE = nullptr;
  int64 matching_GTEs = 0;
  for (HloInstruction* inst : while_body->MakeInstructionPostOrder()) {
    const bool is_GTE_from_param_0 =
        WhileLoopUtil::IsGTEFromParamIndex(inst, 0);
    if (!is_GTE_from_param_0) continue;
    if (inst->tuple_index() == tuple_index) {
      body_GTE = inst;
      matching_GTEs++;
    }
  }
  // Make sure there is only one
  if (matching_GTEs != 1) {
    return xla::FailedPrecondition("%s", err_msg);
  }

  // Check that the mapped GTE instruction is modified by 1 (or -1 for greater
  // than (or equal)) and that the resulting increment is *only* used in the
  // output tuple of the while body in the same index
  int64 delta;
  switch (c_inst->comparison_direction()) {
    case ComparisonDirection::kLt:
    case ComparisonDirection::kLe:
      delta = 1;
      break;
    case ComparisonDirection::kGe:
    case ComparisonDirection::kGt:
    default:
      delta = -1;
      break;
  }

  auto matching_increments =
      WhileLoopUtil::FindMatchingLoopDeltasInsideBody(body_GTE, while_body);

  if (matching_increments.size() != 1 ||
      matching_increments[0].second != delta) {
    return xla::FailedPrecondition("%s", err_msg);
  }

  TF_ASSIGN_OR_RETURN(int64 initial_value,
                      LiteralScalarToNativeType<int64>(init_inst->literal()));
  TF_ASSIGN_OR_RETURN(int64 compare_value,
                      LiteralScalarToNativeType<int64>(limit_inst->literal()));

  // Calculate the number of iterations and the final counter state
  // * Take caution when the condition was initially true (i.e. no iterations
  //   are executed) - to do that set the number of iterations to 0.
  int64 number_of_iterations = 0;

  switch (c_inst->comparison_direction()) {
    case ComparisonDirection::kLt:
      if (initial_value < compare_value) {
        number_of_iterations = compare_value - initial_value;
      }
      break;
    case ComparisonDirection::kLe:
      if (initial_value <= compare_value) {
        number_of_iterations = compare_value - initial_value + 1;
      }
      break;
    case ComparisonDirection::kGe:
      if (initial_value >= compare_value) {
        number_of_iterations = initial_value - compare_value + 1;
      }
      break;
    case ComparisonDirection::kGt:
    default:
      if (initial_value > compare_value) {
        number_of_iterations = initial_value - compare_value;
      }
      break;
  }

  return number_of_iterations;
}

template <typename NativeT>
HloInstruction* GetFinalValue(HloInstruction* init_value_inst,
                              const int64 number_of_iterations,
                              const int64 delta) {
  NativeT value = LiteralScalarToNativeType<NativeT>(init_value_inst->literal())
                      .ValueOrDie();
  // Adjust the value in the right type.
  value += (NativeT(number_of_iterations) * NativeT(delta));
  return init_value_inst->parent()->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(value)));
}

HloInstruction* ConvertToRepeat(HloInstruction* while_inst,
                                const int64 number_of_iterations) {
  // We represent repeat as kCall and store the number of iterations in the
  // backend config field. We clone the repeat computation and use it as the
  // computation for the call.
  HloComputation* parent_computation = while_inst->parent();
  HloModule* module = parent_computation->parent();
  HloComputation* repeat_body =
      module->AddEmbeddedComputation(while_inst->while_body()->Clone());
  HloInstruction* repeat_body_root = repeat_body->root_instruction();
  HloInstruction* input_tuple = while_inst->mutable_operand(0);

  // If the number of iterations is 0, don't create a loop.
  if (number_of_iterations == 0) {
    return input_tuple;
  }

  // Note that we can also clone the input tuple iff it's a kTuple and then we
  // can hoist out constants to that input tuple.
  bool can_hoist_input_tuple = input_tuple->opcode() == HloOpcode::kTuple;
  input_tuple = can_hoist_input_tuple
                    ? parent_computation->AddInstruction(input_tuple->Clone())
                    : input_tuple;

  // Also hoist out all the constants.
  // A map of scalar values which we know the value of after the loop has
  // completed.
  absl::flat_hash_map<int64, HloInstruction*> gte_to_final_value;
  absl::flat_hash_set<int64> dead_gtes;

  // Go through all the other scalar counters in the while loop and try and
  // determine their values given the number of iterations. A value can be
  // determined if:
  // * The input to the tuple is a constant
  // * The value is accessed from the input tuple at index x, modified by 1 (add
  // or subtract), stored in the output tuple at index x.
  for (int64 tuple_index = 0; tuple_index < input_tuple->operand_count();
       tuple_index++) {
    HloInstruction* operand = input_tuple->mutable_operand(tuple_index);
    // Skip if it is not a integer scalar constant
    if (!WhileLoopUtil::Is32BitsOrLessIntegerConstant(operand)) {
      continue;
    }

    // Find corresponding GTE in the body
    HloInstruction* gte = nullptr;
    int64 matching_GTEs = 0;
    for (HloInstruction* inst :
         repeat_body->parameter_instruction(0)->users()) {
      const bool is_GTE_from_param_0 =
          WhileLoopUtil::IsGTEFromParamIndex(inst, 0);
      if (!is_GTE_from_param_0) continue;
      if (inst->tuple_index() == tuple_index) {
        gte = inst;
        matching_GTEs++;
      }
    }
    // Make sure there is only one
    if (matching_GTEs != 1) {
      continue;
    }

    // Find a matching increment which is then used in a tuple index
    // tuple_index.
    auto matching_increments =
        WhileLoopUtil::FindMatchingLoopDeltasInsideBody(gte, repeat_body);

    // Check that there is only one.
    if (matching_increments.size() != 1) {
      continue;
    }
    HloInstruction* increment = matching_increments[0].first;
    const int64 delta = matching_increments[0].second;

    switch (operand->shape().element_type()) {
#define GET_FINAL_VALUE(XLA_T, NATIVE_T)                               \
  case (XLA_T):                                                        \
    gte_to_final_value[tuple_index] =                                  \
        GetFinalValue<NATIVE_T>(operand, number_of_iterations, delta); \
    break;
      GET_FINAL_VALUE(U8, uint8);
      GET_FINAL_VALUE(U16, uint16);
      GET_FINAL_VALUE(U32, uint32);
      GET_FINAL_VALUE(S8, int8);
      GET_FINAL_VALUE(S16, int16);
      GET_FINAL_VALUE(S32, int32);
#undef GET_FINAL_VALUE
      default:
        continue;
    }

    // If the unique GTE in the body is only incremented and used by the
    // matching index in the return tuple then we can make this tuple index
    // dead.
    if (gte->user_count() == 1 && increment->user_count() == 1 &&
        repeat_body_root->operand(tuple_index) == increment) {
      // Check that it only appears once in the return tuple.
      const auto used_count =
          absl::c_count(repeat_body_root->operands(), increment);

      if (used_count == 1) {
        dead_gtes.insert(tuple_index);
      }
    }
  }

  // Replace all the known values with constants.
  for (auto pair : gte_to_final_value) {
    int64 tuple_index = pair.first;
    HloInstruction* final_value = pair.second;

    // Replace all the GTEs for `tuple_index` from the output tuple use with a
    // constant.
    for (auto user : while_inst->users()) {
      if (user->opcode() == HloOpcode::kGetTupleElement &&
          user->tuple_index() == tuple_index) {
        HloInstruction* constant =
            parent_computation->AddInstruction(final_value->Clone());

        if (user->has_sharding()) {
          constant->set_sharding(user->sharding());
        }

        user->ReplaceAllUsesWith(constant);
      }
    }
  }

  // Tidy up dead tuple elements.
  if (dead_gtes.size()) {
    // For each dead GTE, we replace the input tuple with a constant and also
    // remove the increment in the while body.
    for (int64 tuple_index : dead_gtes) {
      // Replace the root tuple operand.
      for (auto* user : repeat_body->parameter_instruction(0)->users()) {
        if (user->opcode() == HloOpcode::kGetTupleElement &&
            user->tuple_index() == tuple_index) {
          repeat_body_root->ReplaceOperandWith(tuple_index, user);
          break;
        }
      }

      if (can_hoist_input_tuple) {
        // Replace the input tuple operand.
        HloInstruction* old_inst = input_tuple->mutable_operand(tuple_index);
        HloInstruction* constant = parent_computation->AddInstruction(
            gte_to_final_value.at(tuple_index)->Clone());

        if (old_inst->has_sharding()) {
          constant->set_sharding(old_inst->sharding());
        }
        input_tuple->ReplaceOperandWith(tuple_index, constant);
      }
    }
  }

  HloInstruction* repeat_call;
  // Unpack the tuple parameters if possible.
  if (input_tuple->opcode() == HloOpcode::kTuple) {
    // Create a new computation which handles the parameters separately.
    HloComputation::Builder builder(repeat_body->name() + "_repeat");
    auto new_operands = input_tuple->operands();
    absl::flat_hash_map<HloInstruction*, HloInstruction*> clone_map;

    std::vector<HloInstruction*> new_parameters(new_operands.size());
    for (int64 param_idx = 0; param_idx != new_operands.size(); ++param_idx) {
      new_parameters[param_idx] =
          builder.AddInstruction(HloInstruction::CreateParameter(
              param_idx, new_operands[param_idx]->shape(),
              absl::StrCat("arg_", param_idx)));
    }
    for (HloInstruction* inst : repeat_body->MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kParameter) {
        // Use the new parameters for the tuple instruction.
        CHECK_EQ(inst->parameter_number(), 0);
        clone_map[inst] =
            builder.AddInstruction(input_tuple->CloneWithNewOperands(
                input_tuple->shape(), new_parameters));
      } else {
        std::vector<HloInstruction*> clone_operands(inst->operand_count());
        absl::c_transform(inst->operands(), clone_operands.begin(),
                          [&clone_map](HloInstruction* old_operand) {
                            return clone_map.at(old_operand);
                          });
        // Clone new instruction.
        clone_map[inst] = builder.AddInstruction(
            inst->CloneWithNewOperands(inst->shape(), clone_operands));
      }
    }
    HloComputation* new_repeat_body = module->AddEmbeddedComputation(
        builder.Build(clone_map.at(repeat_body_root)));

    repeat_call = parent_computation->AddInstruction(HloInstruction::CreateCall(
        while_inst->shape(), new_operands, new_repeat_body));
  } else {
    repeat_call = parent_computation->AddInstruction(HloInstruction::CreateCall(
        while_inst->shape(), {input_tuple}, repeat_body));
  }

  auto backend_config =
      while_inst->backend_config<PoplarBackendConfig>().ValueOrDie();
  auto* call_config = backend_config.mutable_call_config();
  call_config->set_type(PoplarBackendConfig::CallConfig::RepeatLoop);
  auto* repeat_cfg = call_config->mutable_repeat_config();
  repeat_cfg->set_repeat_count(number_of_iterations);
  repeat_call->set_backend_config(backend_config);

  // Copy sharding info from the while_inst to the repeat.
  if (while_inst->has_sharding()) {
    repeat_call->set_sharding(while_inst->sharding());
  }

  return repeat_call;
}

/**
 * Remove computations from the HloModule that are not reachable from the entry
 * computation.
 */
void PruneComputations(HloModule* module) {
  // Find the reachable computations
  absl::flat_hash_set<HloComputation*> reachable_comps;
  reachable_comps.insert(module->entry_computation());

  const auto sub_comps =
      module->entry_computation()->MakeEmbeddedComputationsList();
  reachable_comps.insert(sub_comps.begin(), sub_comps.end());

  auto is_unreachable_pred = [&reachable_comps](HloComputation* computation) {
    return reachable_comps.count(computation) == 0;
  };

  // Find the unreachable computations
  auto unreachable_comps = module->MakeComputationPostOrder();
  auto itr = absl::c_partition(unreachable_comps, is_unreachable_pred);
  unreachable_comps.erase(itr, unreachable_comps.end());

  // Remove the unreachable computations
  for (auto computation : unreachable_comps) {
    module->RemoveEmbeddedComputation(computation);
  }
}

}  // namespace

StatusOr<bool> WhileLoopToRepeatSimplify::Run(HloModule* module) {
  // For each while instruction
  for (auto* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (auto* inst : comp->MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kWhile) {
        HloInstruction* while_inst = inst;
        // For each while loop, try and simplify the logic to convert the loop
        // into a repeat.
        auto statusor = ConvertWhileToRepeat(while_inst);
        int64 count = 0;
        bool simplified = false;
        if (statusor.ok()) {
          simplified = true;
          count = statusor.ValueOrDie();
        } else {
          statusor.IgnoreError();

          // Ignore the error and try the brute force method
          auto op_count =
              ComputeWhileLoopTripCount(while_inst, GetMaxLoopTripCount());
          if (op_count) {
            simplified = true;
            count = *op_count;
          }
        }

        if (simplified) {
          HloInstruction* repeat_call = ConvertToRepeat(while_inst, count);
          while_inst->ReplaceAllUsesWith(repeat_call);

          VLOG(1) << "Simplified while loop " << while_inst->name()
                  << " with a repeat of count " << count;

          while_inst->parent()->RemoveInstructionAndUnusedOperands(while_inst);
          PruneComputations(module);
          TF_RETURN_IF_ERROR(TupleSimplifier(true).Run(module).status());
          return true;
        }
      }
    }
  }

  return false;
}

}  // namespace poplarplugin
}  // namespace xla
