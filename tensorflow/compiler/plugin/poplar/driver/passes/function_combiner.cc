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

#include "tensorflow/compiler/plugin/poplar/driver/passes/function_combiner.h"

#include <map>
#include <set>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/passes/outline_remote_buffers.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/isomorphic_functions_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {

namespace {
bool InputsOutputsOnSameShard(const HloInstruction* inst) {
  if (!inst->has_sharding()) {
    return false;
  }

  auto optional_device = inst->sharding().UniqueDevice();
  if (!optional_device) {
    return false;
  }

  return absl::c_all_of(
      inst->operands(), [&optional_device](const HloInstruction* operand) {
        return operand->sharding().UniqueDevice() == optional_device;
      });
}

bool IsFunctionForCombining(const HloInstruction* inst) {
  return IsFunction(inst) && InputsOutputsOnSameShard(inst) &&
         (GetFunctionNumberModifiedRemoteBufferInputs(inst) ||
          GetFunctionNumberUnmodifiedRemoteBufferInputs(inst)) &&
         AllUsersUniqueGTEs(inst);
}

bool AllInstructionIndependent(
    const HloReachabilityMap& reachability_map,
    const std::vector<HloInstruction*>& function_keys,
    const SingleShardIsomorphicFunctions& iso_functions) {
  // Compare all the possible pairs of functions.
  for (HloInstruction* key1 : function_keys) {
    for (HloInstruction* key2 : function_keys) {
      for (HloInstruction* func1 : iso_functions.at(key1)) {
        for (HloInstruction* func2 : iso_functions.at(key2)) {
          if (func1 != func2 && reachability_map.IsConnected(func1, func2)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

// Key for which functions can be combined together.
// In order for functions to be combined they need to occur the same number of
// times to make sure the code is still only generated once.
// Take the remote buffer inputs into into account to make sure to only combine
// functions which will cause the same number of syncronisations.
struct CrossShardFunctionKey {
  int64_t num_occurrences;
  int64_t num_modified_remote_buffer_inputs;
  int64_t num_unmodified_remote_buffer_inputs;

  bool operator<(const CrossShardFunctionKey& other) const {
    return std::make_tuple(num_occurrences, num_modified_remote_buffer_inputs,
                           num_unmodified_remote_buffer_inputs) <
           std::make_tuple(other.num_occurrences,
                           other.num_modified_remote_buffer_inputs,
                           other.num_unmodified_remote_buffer_inputs);
  }
};

// When combining functions, prefer to merge the larger functions first (note
// that the remote buffers will be in the function shape).
struct SizeSortedHloPtrComparator {
  bool operator()(const HloInstruction* a, const HloInstruction* b) const {
    const int64_t a_size = GetByteSizeOfTotalShape(a->shape());
    const int64_t b_size = GetByteSizeOfTotalShape(b->shape());

    if (a_size != b_size) {
      return a_size > b_size;
    }

    return HloPtrComparator()(a, b);
  }
};

// Structure used to store functions on a single shard in decreasing output
// size.
using SizeSortedFunctions =
    std::set<HloInstruction*, SizeSortedHloPtrComparator>;

// Structure to map from a shard to set of functions.
using ShardToFunctions = std::map<int64_t, SizeSortedFunctions>;

// Structure which maps all the functions which can be combined together across
// the shards.
using CrossShardFunctions = std::map<CrossShardFunctionKey, ShardToFunctions>;
}  // namespace

FunctionsToCombine FunctionCombiner::GetFunctionsToCombine(
    HloComputation* comp) {
  FunctionsToCombine outputs;
  // Find identical functions within each shard which have remote buffer
  // inputs/outputs.
  SingleShardIsomorphicFunctions iso_functions;
  for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
    if (IsFunctionForCombining(inst)) {
      iso_functions.insert(inst);
    }
  }

  if (iso_functions.empty()) {
    return outputs;
  }

  // Match up the functions between shards which should be considered to be
  // merged.
  CrossShardFunctions cross_shard_functions;
  for (auto& pair : iso_functions) {
    HloInstruction* functions_key = pair.first;
    const int64_t shard = *functions_key->sharding_unique_device();
    const int64_t num_modified_remote_buffer_inputs =
        GetFunctionNumberModifiedRemoteBufferInputs(functions_key);
    const int64_t num_unmodified_remote_buffer_inputs =
        GetFunctionNumberUnmodifiedRemoteBufferInputs(functions_key);

    CrossShardFunctionKey cross_shard_key{pair.second->size(),
                                          num_modified_remote_buffer_inputs,
                                          num_unmodified_remote_buffer_inputs};

    cross_shard_functions[cross_shard_key][shard].insert(functions_key);
  }

  if (cross_shard_functions.empty()) {
    return outputs;
  }

  auto reachability_map = HloReachabilityMap::Build(comp);

  VLOG(2) << cross_shard_functions.size();

  std::vector<std::vector<HloInstruction*>> per_combination_functions;
  for (auto& pair : cross_shard_functions) {
    auto& shard_to_functions = pair.second;
    // Keep picking functions from each shard to merge until there are no more
    // candidates.
    while (shard_to_functions.size() > 1) {
      std::vector<HloInstruction*> functions_to_merge;
      std::vector<int64_t> keys_to_erase;
      // Get the largest function from each shard.
      for (auto& func_pairs : shard_to_functions) {
        auto func_itr = func_pairs.second.begin();
        functions_to_merge.push_back(*func_itr);

        // Each function key can only be merged once.
        func_pairs.second.erase(func_itr);
        // Mark that no more functions from this shard can be merged.
        if (func_pairs.second.empty()) {
          keys_to_erase.push_back(func_pairs.first);
        }
      }
      // Make sure all the functions are independent of each other to allow for
      // them to be combined.
      const bool all_independent = AllInstructionIndependent(
          *reachability_map.get(), functions_to_merge, iso_functions);

      if (all_independent) {
        per_combination_functions.push_back(functions_to_merge);
      }

      for (int64_t shard : keys_to_erase) {
        shard_to_functions.erase(shard);
      }
    }
  }

  outputs.reserve(per_combination_functions.size());
  for (auto& keys : per_combination_functions) {
    VLOG(2) << "Can combine the functions:";
    std::vector<Functions> output;
    for (HloInstruction* key : keys) {
      auto& functions = iso_functions.at(key);
      VLOG(2) << "* " << key->ToString() << " which appears "
              << functions.size() << " times.";
      output.push_back(functions);
    }
    outputs.push_back(output);
  }

  return outputs;
}

FunctionCombiner::Permutations FunctionCombiner::GetInputsOutputsPermutation(
    const std::vector<HloInstruction*>& functions) {
  CHECK_GE(functions.size(), 1);
  const int64_t num_functions = functions.size();
  const HloInstruction* function = functions[0];

  // All the functions are expected to have the same number of remote buffers
  // (see CrossShardFunctionKey).
  const int64_t num_modified_remote_buffer_inputs =
      GetFunctionNumberModifiedRemoteBufferInputs(function);
  const int64_t num_unmodified_remote_buffer_inputs =
      GetFunctionNumberUnmodifiedRemoteBufferInputs(function);

  const int64_t num_remote_buffer_inputs =
      num_modified_remote_buffer_inputs + num_unmodified_remote_buffer_inputs;

  std::vector<int64_t> old_to_new_inputs_permutation;
  std::vector<int64_t> old_to_new_outputs_permutation;

  {
    int64_t next_modified_remote_buffer_input = 0;
    int64_t next_unmodified_remote_buffer_input =
        num_modified_remote_buffer_inputs * num_functions;
    int64_t next_input = num_remote_buffer_inputs * num_functions;

    for (const HloInstruction* func : functions) {
      for (int64_t operand_idx = 0; operand_idx != func->operand_count();
           ++operand_idx) {
        if (operand_idx < num_modified_remote_buffer_inputs) {
          old_to_new_inputs_permutation.push_back(
              next_modified_remote_buffer_input++);
        } else if (operand_idx < num_remote_buffer_inputs) {
          old_to_new_inputs_permutation.push_back(
              next_unmodified_remote_buffer_input++);
        } else {
          old_to_new_inputs_permutation.push_back(next_input++);
        }
      }
    }

    VLOG(2) << "Inputs permutation "
            << absl::StrJoin(old_to_new_inputs_permutation, ",");
  }

  {
    int64_t next_modified_remote_buffer_output = 0;
    int64_t next_output = num_modified_remote_buffer_inputs * num_functions;

    for (const HloInstruction* func : functions) {
      const int64_t num_outputs = ShapeUtil::TupleElementCount(func->shape());
      for (int64_t output_idx = 0; output_idx != num_outputs; ++output_idx) {
        if (output_idx < num_modified_remote_buffer_inputs) {
          old_to_new_outputs_permutation.push_back(
              next_modified_remote_buffer_output++);
        } else {
          old_to_new_outputs_permutation.push_back(next_output++);
        }
      }
    }

    VLOG(2) << "Outputs permutation "
            << absl::StrJoin(old_to_new_outputs_permutation, ",");
  }
  return {old_to_new_inputs_permutation, old_to_new_outputs_permutation};
}

StatusOr<std::vector<HloInstruction*>> FunctionCombiner::CombineFunctions(
    const std::vector<Functions>& per_shard_functions) {
  const int64_t num_functions = per_shard_functions.size();
  // Get a function from each shard which is being combined.
  std::vector<HloInstruction*> functions(num_functions);
  absl::c_transform(per_shard_functions, functions.begin(),
                    [](const Functions& shard_functions) {
                      return *std::begin(shard_functions);
                    });

  HloInstruction* func = functions[0];
  HloComputation* parent = func->parent();
  HloModule* module = func->GetModule();

  // Make sure all functions output a tuple.
  for (HloInstruction* function : functions) {
    TF_RETURN_IF_ERROR(FixRootInstruction(function->to_apply()).status());
  }

  // Work out how to permute the inputs/outputs.
  auto permutation = GetInputsOutputsPermutation(functions);

  // All the functions are expected to have the same number of remote buffers
  // (see CrossShardFunctionKey).
  const int64_t num_modified_remote_buffer_inputs =
      GetFunctionNumberModifiedRemoteBufferInputs(func);
  const int64_t num_unmodified_remote_buffer_inputs =
      GetFunctionNumberUnmodifiedRemoteBufferInputs(func);

  // Clone the functions into a single computation.
  HloCloneContext context(module);
  auto builder = HloComputation::Builder(func->to_apply()->name());

  auto get_operands =
      [&context](HloInstruction* old_inst) -> std::vector<HloInstruction*> {
    std::vector<HloInstruction*> new_operands(old_inst->operand_count());
    absl::c_transform(old_inst->operands(), new_operands.begin(),
                      [&context](HloInstruction* old_operand) {
                        return context.GetInstruction(old_operand);
                      });
    return new_operands;
  };

  auto copy_control_dependencies = [&context](HloInstruction* old_inst,
                                              HloInstruction* new_inst) {
    // Find the new control predecessors in the clone context.
    for (auto* old_control_pred : old_inst->control_predecessors()) {
      auto* new_control_pred = context.GetInstruction(old_control_pred);
      new_control_pred->AddControlDependencyTo(new_inst);
    }
  };

  auto post_process = [&context, &copy_control_dependencies](
                          HloInstruction* old_inst, HloInstruction* new_inst) {
    old_inst->SetupDerivedInstruction(new_inst);
    context.MapInstruction(old_inst, new_inst);
    copy_control_dependencies(old_inst, new_inst);
  };

  std::vector<HloInstruction*> old_roots;
  int64_t function_start_parameter_idx = 0;
  for (int64_t func_idx = 0; func_idx != num_functions; ++func_idx) {
    HloInstruction* function = functions[func_idx];
    HloComputation* comp = function->to_apply();

    for (HloInstruction* old_inst : comp->MakeInstructionPostOrder()) {
      if (old_inst->opcode() == HloOpcode::kParameter) {
        const int64_t parameter_number =
            function_start_parameter_idx + old_inst->parameter_number();

        const int64_t new_parameter_number =
            permutation.old_to_new_inputs_permutation.at(parameter_number);

        // Create a parameter.
        HloInstruction* new_inst =
            builder.AddInstruction(HloInstruction::CreateParameter(
                new_parameter_number, old_inst->shape(), old_inst->name()));

        post_process(old_inst, new_inst);
      } else {
        HloInstruction* new_inst =
            builder.AddInstruction(old_inst->CloneWithNewOperands(
                old_inst->shape(), get_operands(old_inst)));

        post_process(old_inst, new_inst);

        if (old_inst == comp->root_instruction()) {
          CHECK_EQ(old_inst->opcode(), HloOpcode::kTuple);
          old_roots.push_back(old_inst);
        }
      }
    }
    function_start_parameter_idx += function->operand_count();
  }
  CHECK_EQ(old_roots.size(), num_functions);
  // Create GTEs from each root and then create a single root tuple instruction
  // with all the outputs correctly permuted.
  std::vector<HloInstruction*> all_outputs;
  for (int64_t func_idx = 0; func_idx != num_functions; ++func_idx) {
    HloInstruction* function = functions[func_idx];
    HloInstruction* old_root = function->to_apply()->root_instruction();
    HloInstruction* new_root = context.GetInstruction(old_root);
    const int64_t shard = *new_root->sharding().UniqueDevice();
    const int64_t num_outputs = ShapeUtil::TupleElementCount(new_root->shape());
    for (int64_t output_idx = 0; output_idx != num_outputs; ++output_idx) {
      const HloInstruction* operand = new_root->operand(output_idx);
      HloInstruction* gte =
          builder.AddInstruction(HloInstruction::CreateGetTupleElement(
              operand->shape(), new_root, output_idx));
      operand->SetupDerivedInstruction(gte);
      all_outputs.push_back(gte);
    }
  }

  // Permute the outputs to the right order.
  all_outputs =
      Permute(all_outputs, permutation.old_to_new_outputs_permutation);

  // Create the root instruction.
  HloInstruction* root =
      builder.AddInstruction(HloInstruction::CreateTuple(all_outputs));

  // Get the new sharding for the output.
  std::vector<HloSharding> outputs_sharding;
  for (HloInstruction* output : all_outputs) {
    const HloSharding& sharding = output->sharding();
    if (sharding.IsTuple()) {
      auto& tuple_sharding = sharding.tuple_elements();
      outputs_sharding.insert(outputs_sharding.end(), tuple_sharding.begin(),
                              tuple_sharding.end());
    } else {
      outputs_sharding.push_back(sharding);
    }
  }
  HloSharding output_sharding =
      HloSharding::Tuple(root->shape(), outputs_sharding);
  root->set_sharding(output_sharding);

  // Create a copy of the computation to outline.
  HloComputation* outlined_computation =
      module->AddEmbeddedComputation(builder.Build(root));

  // For each set of functions across the shards, replace them with a combined
  // function.
  std::vector<Functions::iterator> per_shard_iterators;
  absl::c_transform(per_shard_functions,
                    std::back_inserter(per_shard_iterators),
                    [](const Functions& shard_functions) {
                      return std::begin(shard_functions);
                    });

  std::vector<HloInstruction*> combined_functions;
  for (int64_t i = 0; i != per_shard_functions[0].size(); ++i) {
    // Get a function from each shard, their operands and output GTEs.
    std::vector<HloInstruction*> operands;
    std::vector<HloInstruction*> gtes;
    VLOG(2) << "Combining functions: ";
    for (int64_t func_num = 0; func_num != num_functions; ++func_num) {
      HloInstruction* function = *per_shard_iterators[func_num];
      VLOG(2) << "* " << function->ToString();
      functions[func_num] = function;
      // Get the operands.
      operands.insert(operands.end(), function->operands().begin(),
                      function->operands().end());

      // Get the output GTEs.
      const int64_t num_outputs =
          ShapeUtil::TupleElementCount(function->shape());
      for (int64_t output_idx = 0; output_idx != num_outputs; ++output_idx) {
        TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                            GetUniqueGTEUser(function, output_idx));
        gtes.push_back(gte);
      }
      // Move the iterator for the next function.
      ++per_shard_iterators[func_num];
    }

    // Permute the operands.
    operands = Permute(operands, permutation.old_to_new_inputs_permutation);
    // Create a new call with the new computations and the new sharding.
    HloInstruction* new_function = parent->AddInstruction(
        functions[0]->CloneWithNewOperands(root->shape(), operands));
    new_function->SetAndSanitizeName(
        absl::StrCat(functions[0]->name(), "_combined"));

    HloComputation* new_computation =
        module->AddEmbeddedComputation(outlined_computation->Clone());

    new_function->set_to_apply(new_computation);
    new_function->set_sharding(output_sharding);
    functions[0]->SetupDerivedInstruction(new_function);

    // Set the information about remote buffers.
    TF_ASSIGN_OR_RETURN(PoplarBackendConfig config,
                        new_function->backend_config<PoplarBackendConfig>());
    auto* function_config =
        config.mutable_call_config()->mutable_function_config();
    function_config->set_num_modified_remote_buffer_inputs(
        num_modified_remote_buffer_inputs * num_functions);
    function_config->set_num_unmodified_remote_buffer_inputs(
        num_unmodified_remote_buffer_inputs * num_functions);
    function_config->set_unique_sharding(false);
    TF_RETURN_IF_ERROR(new_function->set_backend_config(config));

    // Permute the GTEs to the right order.
    gtes = Permute(gtes, permutation.old_to_new_outputs_permutation);
    // Rewire all the GTEs to use the new function call at the right tuple
    // index.
    for (int64_t output_idx = 0; output_idx != gtes.size(); ++output_idx) {
      HloInstruction* gte = gtes.at(output_idx);
      TF_RETURN_IF_ERROR(
          gte->ReplaceOperandWithDifferentShape(0, new_function));
      gte->set_tuple_index(output_idx);
    }

    // Copy over any control dependencies and remove the old functions.
    for (HloInstruction* function : functions) {
      TF_RETURN_IF_ERROR(new_function->CopyAllControlDepsFrom(function));
      TF_RETURN_IF_ERROR(function->DropAllControlDeps());
      TF_RETURN_IF_ERROR(parent->RemoveInstruction(function));
    }
    VLOG(2) << "New function is " << new_function->ToString();
    combined_functions.push_back(new_function);
  }
  return combined_functions;
}

StatusOr<bool> FunctionCombiner::RunOnComputation(HloComputation* comp) {
  auto functions_to_combine = GetFunctionsToCombine(comp);

  std::vector<std::vector<HloInstruction*>> combined_functions;
  for (auto& per_shard_functions : functions_to_combine) {
    TF_ASSIGN_OR_RETURN(auto combined, CombineFunctions(per_shard_functions));
    combined_functions.push_back(combined);
  }

  return functions_to_combine.size();
}

StatusOr<bool> FunctionCombiner::Run(HloModule* module) {
  VLOG(2) << "Before FunctionCombiner:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;

  // Run it for all resource updates.
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(const bool computation_changed, RunOnComputation(comp));
    changed |= computation_changed;
  }

  if (changed) {
    TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());
    VLOG(3) << "After FunctionCombiner:";
    XLA_VLOG_LINES(3, module->ToString());
  } else {
    VLOG(2) << "No changes were made.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
