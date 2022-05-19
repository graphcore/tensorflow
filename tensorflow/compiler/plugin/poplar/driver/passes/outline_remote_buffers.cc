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

#include "tensorflow/compiler/plugin/poplar/driver/passes/outline_remote_buffers.h"

#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {
bool IsRemoteBufferLoad(const HloInstruction* inst) {
  return IsPoplarInstruction(RemoteParameterLoad, inst);
}

bool IsRemoteBufferStore(const HloInstruction* inst) {
  return IsPoplarInstruction(RemoteParameterStore, inst);
}
}  // namespace

RemoteBufferInputsOutputsInfos::RemoteBufferInputsOutputsInfos(
    HloInstruction* inst) {
  absl::flat_hash_set<HloInstruction*> gtes;
  for (HloInstruction* user : inst->users()) {
    CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
    gtes.insert(user);
  }

  std::set<int64_t> outlined_modified_load_indices;
  std::set<int64_t> outlined_unmodified_load_indices;
  std::map<int64_t, int64_t> load_to_store_index;

  for (int64_t i = 0; i != inst->operand_count(); ++i) {
    HloInstruction* remote_load = inst->mutable_operand(i);
    if (!IsRemoteBufferLoad(remote_load)) {
      continue;
    }

    // Only consider outlining remote buffers which are used inside the
    // function only.
    if (remote_load->user_count() != 1) {
      continue;
    }

    const uint64 replication_factor =
        Cast<HloRemoteParameterLoad>(remote_load)->GetReplicationFactor(0);

    HloInstruction* remote_input = remote_load->mutable_operand(0);
    if (remote_input->user_count() == 1) {
      // Read only load.
      outlined_unmodified_load_indices.insert(i);

    } else if (remote_input->user_count() == 2) {
      // Try and get the store instruction.
      HloInstruction *store, *load;
      if (!GetRemoteLoadStoreUsers(remote_input, &load, &store).ok()) {
        continue;
      }
      CHECK_EQ(load, remote_load);
      HloInstruction* store_input = store->mutable_operand(1);

      // Skip if the input to the store is not a GTE from a function, or it
      // has other users
      if (!gtes.contains(store_input) || store_input->user_count() != 1) {
        continue;
      }

      // Make sure replication factor matches.
      if (Cast<HloRemoteParameterStore>(store)->GetReplicationFactor(0) !=
          replication_factor) {
        continue;
      }

      const int64_t output_idx = store_input->tuple_index();

      VLOG(2) << "Cluster input " << remote_load->ToString()
              << " is a modified parameter load index " << i;
      VLOG(2) << "Cluster output " << store->ToString()
              << " is a modified parameter store index " << output_idx;

      outlined_modified_load_indices.insert(i);
      load_to_store_index[i] = output_idx;
    } else {
      continue;
    }
    input_to_replication_factor_[i] = replication_factor;
  }
  CHECK_EQ(input_to_replication_factor_.size(),
           outlined_modified_load_indices.size() +
               outlined_unmodified_load_indices.size());
  CHECK_EQ(outlined_modified_load_indices.size(), load_to_store_index.size());

  num_modified_load_stores_ = outlined_modified_load_indices.size();
  num_unmodified_loads_ = outlined_unmodified_load_indices.size();

  const int64_t num_inputs = inst->operand_count();
  const int64_t num_outputs = ShapeUtil::TupleElementCount(inst->shape());
  // Create the input and output permutations - this is used such that:
  // * Fist x inputs/outputs correspond to pairs of parameter load/stores which
  // can be outlined.
  // * Next y inputs correspond to parameter loads which are not modified.
  // * Rest of the inputs/outputs.

  // Store both "ways" of the permutations to make the lookup easier.
  inputs_old_to_new_permutation_.resize(num_inputs);
  inputs_new_to_old_permutation_.resize(num_inputs);
  outputs_old_to_new_permutation_.resize(num_outputs);
  outputs_new_to_old_permutation_.resize(num_outputs);

  {
    int64_t next_idx = 0;
    absl::flat_hash_set<int64_t> visited;
    for (int64_t i : outlined_modified_load_indices) {
      inputs_old_to_new_permutation_[i] = next_idx;
      inputs_new_to_old_permutation_[next_idx++] = i;
      visited.insert(i);
    }
    for (int64_t i : outlined_unmodified_load_indices) {
      inputs_old_to_new_permutation_[i] = next_idx;
      inputs_new_to_old_permutation_[next_idx++] = i;
      visited.insert(i);
    }
    for (int64_t i = 0; i != num_inputs; ++i) {
      if (visited.contains(i)) {
        continue;
      }
      inputs_old_to_new_permutation_[i] = next_idx;
      inputs_new_to_old_permutation_[next_idx++] = i;
    }
    VLOG(2) << "Inputs permutations: old-to-new "
            << absl::StrJoin(inputs_old_to_new_permutation_, ", ")
            << ", new-to-old "
            << absl::StrJoin(inputs_new_to_old_permutation_, ", ");
  }

  {
    int64_t next_idx = 0;
    absl::flat_hash_set<int64_t> visited;
    for (auto pair : load_to_store_index) {
      const int64_t output_idx = pair.second;
      outputs_new_to_old_permutation_[next_idx] = output_idx;
      outputs_old_to_new_permutation_[output_idx] = next_idx++;
      visited.insert(output_idx);
    }

    for (int64_t i = 0; i != num_outputs; ++i) {
      if (visited.contains(i)) {
        continue;
      }
      outputs_new_to_old_permutation_[next_idx] = i;
      outputs_old_to_new_permutation_[i] = next_idx++;
    }
    VLOG(2) << "Outputs permutations: old-to-new "
            << absl::StrJoin(outputs_old_to_new_permutation_, ", ")
            << ", new-to-old "
            << absl::StrJoin(outputs_new_to_old_permutation_, ", ");
  }
}

uint64 RemoteBufferInputsOutputsInfos::GetLoadReplicationFactor(
    int64_t index) const {
  return input_to_replication_factor_.at(index);
}

int64_t RemoteBufferInputsOutputsInfos::GetNumModifiedLoadStores() const {
  return num_modified_load_stores_;
}

int64_t RemoteBufferInputsOutputsInfos::GetNumUnmodifiedLoads() const {
  return num_unmodified_loads_;
}

int64_t RemoteBufferInputsOutputsInfos::GetNumLoadInputs() const {
  return GetNumModifiedLoadStores() + GetNumUnmodifiedLoads();
}

const absl::flat_hash_map<int64_t, uint64>&
RemoteBufferInputsOutputsInfos::GetReplicationFactors() const {
  return input_to_replication_factor_;
}

const std::vector<int64_t>&
RemoteBufferInputsOutputsInfos::GetInputsOldToNewPermutation() const {
  return inputs_old_to_new_permutation_;
}

const std::vector<int64_t>&
RemoteBufferInputsOutputsInfos::GetInputsNewToOldPermutation() const {
  return inputs_new_to_old_permutation_;
}

const std::vector<int64_t>&
RemoteBufferInputsOutputsInfos::GetOutputsOldToNewPermutation() const {
  return outputs_old_to_new_permutation_;
}

const std::vector<int64_t>&
RemoteBufferInputsOutputsInfos::GetOutputsNewToOldPermutation() const {
  return outputs_new_to_old_permutation_;
}

bool RemoteBufferInputsOutputsInfos::operator==(
    const RemoteBufferInputsOutputsInfos& other) const {
  return std::make_tuple(inputs_old_to_new_permutation_,
                         outputs_old_to_new_permutation_,
                         input_to_replication_factor_,
                         num_modified_load_stores_, num_unmodified_loads_) ==
         std::make_tuple(other.inputs_old_to_new_permutation_,
                         other.outputs_old_to_new_permutation_,
                         other.input_to_replication_factor_,
                         other.num_modified_load_stores_,
                         other.num_unmodified_loads_);
}

bool RemoteBufferInputsOutputsInfos::operator!=(
    const RemoteBufferInputsOutputsInfos& other) const {
  return !operator==(other);
}

namespace {
StatusOr<bool> OutlineIntoFunctions(const Functions& functions) {
  // Get parameter load indices for one function.
  HloInstruction* func = *std::begin(functions);
  HloModule* module = func->GetModule();
  const RemoteBufferInputsOutputsInfos rbioi(func);

  HloComputation* comp = func->to_apply();
  // Make sure the root is a tuple instruction.
  TF_RETURN_IF_ERROR(FixRootInstruction(comp).status());
  const HloInstruction* old_root = comp->root_instruction();
  CHECK_EQ(old_root->opcode(), HloOpcode::kTuple);

  // Create a new computation, with the parameters and outputs permuted and the
  // load and stores added in.
  HloCloneContext context(module);
  auto builder = HloComputation::Builder(comp->name());

  auto get_operands =
      [&context](HloInstruction* old_inst) -> std::vector<HloInstruction*> {
    std::vector<HloInstruction*> new_operands(old_inst->operand_count());
    absl::c_transform(old_inst->operands(), new_operands.begin(),
                      [&context](HloInstruction* old_operand) {
                        return context.GetInstruction(old_operand);
                      });
    return new_operands;
  };

  // Helper struct for connecting a Load and Store instruction.
  struct RemoteBufferParameter {
    HloInstruction* inst;
    uint64 replication_factor;
  };
  absl::flat_hash_map<int64_t, RemoteBufferParameter> remote_buffer_parameters;

  // Clone the computation and:
  // * Permute the parameters and outline the RemoteParameterLoads.
  // * Permute the outputs and outline the RemoteParameterStores.
  for (HloInstruction* old_inst : comp->MakeInstructionPostOrder()) {
    HloInstruction* new_inst;
    if (old_inst->opcode() == HloOpcode::kParameter) {
      const int64_t parameter_number = old_inst->parameter_number();
      const int64_t new_parameter_number =
          rbioi.GetInputsOldToNewPermutation().at(parameter_number);

      // Add a parameter load instruction if required.
      if (new_parameter_number < rbioi.GetNumLoadInputs()) {
        // The shape of this parameter has to be the shape of the remote buffer,
        // not RemoteBufferLoad instruction itself. Consider the following
        // example for the replication_factor of 2.
        // Before:
        //   f32[8] shard = remote-buffer-load(f32[16] buffer)
        //   call(f32[8] shard)
        // After:
        //   call(f32[16] buffer):
        //     f32[8] shard = remote-buffer-load(f32[16] buffer)
        const HloInstruction* old_func_operand =
            func->operand(parameter_number);
        VLOG(2) << "RemoteBufferLoad instruction: "
                << old_func_operand->ToString();
        CHECK(IsRemoteBufferLoad(old_func_operand));
        const HloInstruction* remote_buffer = old_func_operand->operand(0);
        new_inst = builder.AddInstruction(HloInstruction::CreateParameter(
            new_parameter_number, remote_buffer->shape(),
            remote_buffer->name()));

        const uint64 replication_factor =
            rbioi.GetLoadReplicationFactor(parameter_number);

        remote_buffer_parameters[new_parameter_number] = {new_inst,
                                                          replication_factor};

        new_inst = builder.AddInstruction(
            CreateHloRemoteParameterLoad({new_inst}, {replication_factor}));
      } else {
        // Create a parameter.
        new_inst = builder.AddInstruction(HloInstruction::CreateParameter(
            new_parameter_number, old_inst->shape(), old_inst->name()));
      }

    } else if (old_inst == old_root) {
      // Create a root tuple with correctly permutated outputs.
      auto operands = Permute(get_operands(old_inst),
                              rbioi.GetOutputsOldToNewPermutation());

      CHECK_LE(rbioi.GetNumModifiedLoadStores(), operands.size());
      // Add store instructions for the remote buffer outputs.
      for (int64_t i = 0; i != rbioi.GetNumModifiedLoadStores(); ++i) {
        RemoteBufferParameter& remote_buffer = remote_buffer_parameters.at(i);

        // Store the value into the remote buffer.
        operands[i] = builder.AddInstruction(
            CreateHloRemoteParameterStore({remote_buffer.inst, operands[i]},
                                          {remote_buffer.replication_factor}));
      }
      new_inst = builder.AddInstruction(HloInstruction::CreateTuple(operands));

    } else {
      new_inst = builder.AddInstruction(old_inst->CloneWithNewOperands(
          old_inst->shape(), get_operands(old_inst)));
    }
    old_inst->SetupDerivedInstruction(new_inst);
    context.MapInstruction(old_inst, new_inst);
  }
  HloInstruction* new_root = context.GetInstruction(old_root);

  HloComputation* outlined_computation =
      module->AddEmbeddedComputation(builder.Build(new_root));
  const int64_t num_outputs = ShapeUtil::TupleElementCount(func->shape());

  // Go through all the function calls, and use the new computation, permute
  // inputs/outputs and replace all the users.
  for (HloInstruction* old_function : functions) {
    VLOG(2) << "Replacing " << old_function->ToString();
    HloComputation* parent = old_function->parent();
    auto new_operands =
        Permute(old_function->operands(), rbioi.GetInputsOldToNewPermutation());
    // Look through the parameter loads for operands.
    for (int64_t i = 0; i != rbioi.GetNumLoadInputs(); ++i) {
      CHECK(IsRemoteBufferLoad(new_operands[i]));
      new_operands[i] = new_operands[i]->mutable_operand(0);
    }

    HloInstruction* new_function = parent->AddInstruction(
        old_function->CloneWithNewOperands(new_root->shape(), new_operands));
    new_function->SetAndSanitizeName(
        absl::StrCat(old_function->name(), "_outlined"));

    HloComputation* new_computation =
        module->AddEmbeddedComputation(outlined_computation->Clone());

    new_function->set_to_apply(new_computation);
    old_function->SetupDerivedInstruction(new_function);

    // Set the information about remote buffers.
    TF_ASSIGN_OR_RETURN(PoplarBackendConfig config,
                        new_function->backend_config<PoplarBackendConfig>());
    auto function_config =
        config.mutable_call_config()->mutable_function_config();
    function_config->set_num_modified_remote_buffer_inputs(
        rbioi.GetNumModifiedLoadStores());
    function_config->set_num_unmodified_remote_buffer_inputs(
        rbioi.GetNumUnmodifiedLoads());
    TF_RETURN_IF_ERROR(new_function->set_backend_config(config));

    // Get all the old gtes and create the new ones.
    std::vector<HloInstruction*> new_gtes(num_outputs);
    std::vector<HloInstruction*> old_gtes(num_outputs);
    for (int64_t i = 0; i != num_outputs; ++i) {
      TF_ASSIGN_OR_RETURN(old_gtes[i], GetUniqueGTEUser(old_function, i));
      TF_ASSIGN_OR_RETURN(new_gtes[i], MakeGetTupleElementHlo(new_function, i));
    }
    // Replace the old uses of GTEs with the new ones, looking through the
    // parameter stores.
    old_gtes = Permute(old_gtes, rbioi.GetOutputsOldToNewPermutation());
    for (int64_t i = 0; i != num_outputs; ++i) {
      HloInstruction* old_gte = old_gtes.at(i);
      HloInstruction* new_gte = new_gtes.at(i);
      old_gte->SetupDerivedInstruction(new_gte);

      HloInstruction* to_replace = old_gte;
      if (i < rbioi.GetNumModifiedLoadStores()) {
        CHECK_EQ(old_gte->user_count(), 1);
        to_replace = old_gte->users()[0];
        CHECK(IsRemoteBufferStore(to_replace));
      }
      TF_RETURN_IF_ERROR(to_replace->ReplaceAllUsesWith(new_gte));
    }
  }
  return true;
}
}  // namespace

namespace {
bool ShouldOutlineFunctions(const Functions& functions) {
  // Do not outline if the function only appears once.
  if (functions.size() == 1 &&
      !GetFunctionPartitionedElementwiseCluster(*functions.begin())) {
    return false;
  }

  // Get parameter load indices for one function.
  HloInstruction* func = *std::begin(functions);
  HloModule* module = func->GetModule();
  const RemoteBufferInputsOutputsInfos rbioi(func);

  // Can't outline loads/stores without them being inputs.
  if (rbioi.GetNumLoadInputs() == 0) {
    return false;
  }

  // Only outline if all the functions have the same loads/stores indicies.
  for (HloInstruction* function : functions) {
    if (rbioi != RemoteBufferInputsOutputsInfos(function)) {
      return false;
    }
  }
  return true;
}
}  // namespace

SingleShardIsomorphicFunctions OutlineRemoteBuffers::GetFunctionsForOutlining(
    HloModule* module) {
  SingleShardIsomorphicFunctions isomorphic_functions;
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      auto can_outline_into = [](const HloInstruction* inst) {
        if (!IsFunction(inst)) {
          return false;
        }
        if (!AllUsersUniqueGTEs(inst)) {
          return false;
        }

        return GetFunctionNumberModifiedRemoteBufferInputs(inst) == 0 &&
               GetFunctionNumberUnmodifiedRemoteBufferInputs(inst) == 0;
      };

      if (can_outline_into(inst)) {
        CHECK(inst->shape().IsTuple());
        isomorphic_functions.insert(inst);
      }
    }
  }

  std::vector<HloInstruction*> keys_to_erase;
  // For each set of functions, check whether they should be outlined.
  for (auto& pair : isomorphic_functions) {
    if (!ShouldOutlineFunctions(*pair.second)) {
      keys_to_erase.push_back(pair.first);
    }
  }

  for (HloInstruction* key : keys_to_erase) {
    isomorphic_functions.erase(key);
  }

  return isomorphic_functions;
}

StatusOr<bool> OutlineRemoteBuffers::Run(HloModule* module) {
  VLOG(2) << "Before OutlineRemoteBuffers:";
  XLA_VLOG_LINES(2, module->ToString(HloPrintOptions::ShortParsable()));
  bool changed = false;

  SingleShardIsomorphicFunctions functions = GetFunctionsForOutlining(module);
  for (auto& pair : functions) {
    TF_ASSIGN_OR_RETURN(bool changed_funcs, OutlineIntoFunctions(*pair.second));
    changed |= changed_funcs;
  }

  if (changed) {
    TF_RETURN_IF_ERROR(HloDCE().Run(module).status());
    VLOG(2) << "After OutlineRemoteBuffers:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
