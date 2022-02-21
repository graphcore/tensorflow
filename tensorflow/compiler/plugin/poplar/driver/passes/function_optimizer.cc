/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/function_optimizer.h"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {

/** Go through the functions outputs, if any of the outputs is a function input,
 * then remove it from the output tuple and make sure any users of that output
 * use the function input instead.
 *
 * For example, replaces:
 * a = ....
 * b = ....
 * f = function(a, b)
 *  _______________________
 * | p0 = parameter(0)     |
 * | p1 = parameter(1)     |
 * | x = dot(p0, p1)       |
 * | ROOT t = tuple(x, p1) |
 * |_______________________|
 * c = gte(f), index = 0
 * d = gte(f), index = 1
 * e = add(c, d)
 *
 * with:
 * a = ....
 * b = ....
 * f = function(a, b)
 *  _______________________
 * | p0 = parameter(0)     |
 * | p1 = parameter(1)     |
 * | x = dot(p0, p1)       |
 * | ROOT t = tuple(x)     |
 * |_______________________|
 * c = gte(f), index = 0
 * e = add(c, b) <-- use the tensor directly
 *
 * This optimization can reduce the number of copies of a particular tensor.
 **/
StatusOr<bool> RemoveOutputInputParameters(HloInstruction* function) {
  HloComputation* comp = function->parent();
  HloComputation* function_comp = function->to_apply();
  HloInstruction* root = function_comp->root_instruction();
  CHECK_EQ(root->opcode(), HloOpcode::kTuple);

  struct InputOutputInfo {
    int64 input_index;
    int64 output_index;
  };

  std::vector<InputOutputInfo> input_output_parameters;
  for (int64 i = 0; i != root->operand_count(); ++i) {
    const HloInstruction* operand = root->operand(i);
    if (operand->opcode() == HloOpcode::kParameter) {
      VLOG(2) << "Function " << function->ToString() << " output " << i
              << " is a parameter.";
      input_output_parameters.push_back({operand->parameter_number(), i});
    }
  }

  if (input_output_parameters.empty()) {
    return false;
  }

  // Make sure all the users of the function call are GTEs.
  TF_RETURN_IF_ERROR(ConvertAllUsersToGTEs(function).status());

  // Find all the gtes.
  absl::flat_hash_map<int64, HloInstructionSet> gtes;
  for (HloInstruction* user : function->users()) {
    CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
    gtes[user->tuple_index()].insert(user);
  }

  // Replace all the GTEs for the function outputs with the corresponding
  // function inputs.
  for (const InputOutputInfo& info : input_output_parameters) {
    const auto& output_gtes = gtes[info.output_index];
    HloInstruction* input = function->mutable_operand(info.input_index);
    for (HloInstruction* gte : output_gtes) {
      TF_RETURN_IF_ERROR(comp->ReplaceInstruction(gte, input));
    }
    gtes.erase(info.output_index);
  }

  if (gtes.size()) {
    // Remove these parameter outputs if the function is still there.
    absl::flat_hash_set<int64> outputs_to_remove;
    for (const InputOutputInfo& info : input_output_parameters) {
      outputs_to_remove.insert(info.output_index);
    }
    TF_RETURN_IF_ERROR(RemoveOutputsFromCall(function, outputs_to_remove));
  }

  return true;
}
}  // namespace

StatusOr<bool> FunctionOptimizer::OptimizeFunction(HloInstruction* function) {
  HloComputation* function_comp = function->to_apply();

  // Make sure the root instruction is a tuple.
  TF_ASSIGN_OR_RETURN(bool changed_root, FixRootInstruction(function_comp));
  TF_ASSIGN_OR_RETURN(bool moved_parameters,
                      RemoveOutputInputParameters(function));

  return changed_root || moved_parameters;
}

StatusOr<bool> FunctionOptimizer::Run(HloModule* module) {
  VLOG(2) << "Before FunctionOptimizer:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  // Find all the function instructions - note that users might be
  // modified/removed.
  std::vector<HloInstruction*> functions;
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (IsFunction(inst)) {
        functions.push_back(inst);
      }
    }
  }

  for (HloInstruction* function : functions) {
    TF_ASSIGN_OR_RETURN(bool function_changed, OptimizeFunction(function));
    changed |= function_changed;
  }

  if (changed) {
    VLOG(2) << "After FunctionOptimizer:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
