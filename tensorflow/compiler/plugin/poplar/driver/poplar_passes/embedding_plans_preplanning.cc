/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/poplar_passes/embedding_plans_preplanning.h"

#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

#include <poplar/Target.hpp>
#include <popops/DynamicSlice.hpp>

namespace xla {
namespace poplarplugin {
namespace {
// Helper struct for storing the slice plan and the associated lookups.
struct SlicePlanHelper {
  const popops::SlicePlan* slice_plan;
  std::vector<std::size_t> lookups;
};

using InputToSliceUsersMap =
    absl::flat_hash_map<const HloInstruction*,
                        std::vector<const HloInstruction*>>;

using SlicePlanMap =
    absl::flat_hash_map<const HloInstruction*, SlicePlanHelper>;

// Given the slice instructions, get all the slice sizes.
const std::vector<std::size_t> GetLookUps(
    const std::vector<const HloInstruction*>& slices) {
  std::vector<std::size_t> lookups(slices.size());
  absl::c_transform(slices, lookups.begin(), [](const HloInstruction* inst) {
    // All the slice instructions have the indices as operand 1.
    int64 num_elements = ShapeUtil::ElementsIn(inst->operand(1)->shape());
    if (IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(inst) ||
        IsPoplarInstruction(PoplarOp::MultiUpdate)(inst)) {
      auto* cast_inst = Cast<HloMultiUpdateInstruction>(inst);
      num_elements = num_elements / cast_inst->GetSerializationFactor();
    }
    return num_elements;
  });
  return lookups;
}

// Given the lookups, get the total number of slices.
std::size_t GetNumLookUps(const std::vector<std::size_t>& lookups) {
  return absl::c_accumulate(lookups, std::size_t(0), std::plus<std::size_t>());
}

// Get slice plans.
StatusOr<SlicePlanMap> GetSlicePlans(const InputToSliceUsersMap& user_map,
                                     CompilerResources& res,
                                     const SlicePlanMap& existing_plans = {}) {
  SlicePlanMap result;
  for (auto& pair : user_map) {
    const HloInstruction* operand = pair.first;
    const std::vector<const HloInstruction*> slices = pair.second;
    const std::vector<std::size_t> lookups = GetLookUps(slices);

    // Use an existing plan iff there is a plan for the current operand and the
    // total number of lookups matches.
    const bool use_existing_plan =
        existing_plans.contains(operand) &&
        GetNumLookUps(existing_plans.at(operand).lookups) ==
            GetNumLookUps(lookups);
    if (use_existing_plan) {
      // Use an existing plan.
      result[operand] = existing_plans.at(operand);
    } else {
      // Create a new plan.
      const Shape operand_shape = operand->shape();

      TF_ASSIGN_OR_RETURN(poplar::Type data_type,
                          PoplarDataType(operand_shape));
      const int64 input_size = operand_shape.dimensions(0);
      const int64 output_size = operand_shape.dimensions(1);

      // Get all the shards for operand/slices.
      absl::flat_hash_set<int64> shards;
      if (operand->has_sharding()) {
        // If there is sharding, we expect it to be unique.
        auto get_shard = [](const HloInstruction* inst) {
          return inst->sharding().GetUniqueDevice();
        };
        shards.insert(get_shard(operand));
        absl::c_transform(slices, std::inserter(shards, shards.begin()),
                          [&get_shard](const HloInstruction* inst) {
                            return get_shard(inst);
                          });
      }

      // Plan in the master graph if this is used across multiple graph.
      poplar::Graph& graph =
          shards.size() > 1 ? GetMasterGraph(res) : GetGraph(res, operand);

      TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                          GetSliceOptionsForInst(operand, res));

      res.slice_plans.push_back(popops::embedding::plan(
          graph, data_type, input_size, output_size, lookups, opts));
      result[operand] = {&res.slice_plans.back(), lookups};
    }
  }
  return result;
}

Status PopulateWithPlans(const InputToSliceUsersMap& user_map,
                         const SlicePlanMap& plans, CompilerResources& res) {
  for (auto& pair : user_map) {
    const HloInstruction* operand = pair.first;
    const std::vector<const HloInstruction*> slices = pair.second;
    auto itr = plans.find(operand);
    if (itr == plans.end()) {
      return InternalErrorStrCat("Failed to create a slice plan for ",
                                 operand->ToString(), ".");
    }
    const popops::SlicePlan* slice_plan = itr->second.slice_plan;
    for (const HloInstruction* slice : slices) {
      // Map the plan to the instruction in the original module.
      const HloInstruction* original_slice =
          res.annotations.flattened_inst_map_bwd.at(slice);
      res.slice_plan_mappings[original_slice] = slice_plan;
    }
  }
  return Status::OK();
}
}  // namespace

/*
 * All tensors which are related because they are either sliceable / a slice /
 * updateable need to be created using the same SlicePlan object. In order to be
 * able to create a SlicePlan using embedding::plan we need:
 * - The total number of elements in the input.
 * - The total number of elements in the output.
 * - The data type of the input / output.
 * - A list of the number of offsets used by all the slice / updates calls made
 * on a given input.
 *
 * This visitor works in two stages:
 * (1) Walk through the flattened_module to find all the calls to multiSlice /
 * multiUpdate / multiUpdateAdd. For each of these calls iterate through its
 * operands and collect the data needed to create its slice plan. Identify and
 * keep track of which HloInstructions should share a SlicePlan. (2) Create all
 * the SlicePlans and populate the slice_plans and slice_plan_mappings in
 * CompilerResources in order to be later able to retrieve the SlicePlan
 * associated to a given HloInstruction during lowering.
 */
StatusOr<bool> EmbeddingPlansPreplanning::Run(HloModule* module) {
  VLOG(2) << "Preplanning embedding operations.";

  absl::flat_hash_map<const HloInstruction*, popops::SlicePlan> slice_plans;
  HloComputation* entry_computation =
      resources_.annotations.flattened_module->entry_computation();

  InputToSliceUsersMap multi_slices;
  InputToSliceUsersMap multi_update_adds;
  InputToSliceUsersMap multi_updates;

  for (const HloInstruction* inst :
       entry_computation->MakeInstructionPostOrder()) {
    if (IsPoplarInstruction(PoplarOp::MultiSlice)(inst)) {
      multi_slices[inst->operand(0)].push_back(inst);
    } else if (IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(inst)) {
      multi_update_adds[inst->operand(0)].push_back(inst);
    } else if (IsPoplarInstruction(PoplarOp::MultiUpdate)(inst)) {
      // Note that HloMultiUpdateAddInstruction inherits from
      // HloMultiUpdateInstruction, so we need to handle it first.
      multi_updates[inst->operand(0)].push_back(inst);
    }
  }

  // Populate the multi slices.
  TF_ASSIGN_OR_RETURN(SlicePlanMap multi_slice_plans,
                      GetSlicePlans(multi_slices, resources_));

  // Populate multi update add slice plans and try to reuse the
  // multi_slice_plans.
  TF_ASSIGN_OR_RETURN(
      SlicePlanMap multi_update_add_plans,
      GetSlicePlans(multi_update_adds, resources_, multi_slice_plans));

  // Same as above, but for multi-update.
  TF_ASSIGN_OR_RETURN(
      SlicePlanMap multi_update_plans,
      GetSlicePlans(multi_updates, resources_, multi_slice_plans));

  // Populate the slice plans in the CompilerResources.
  TF_RETURN_IF_ERROR(
      PopulateWithPlans(multi_slices, multi_slice_plans, resources_));
  TF_RETURN_IF_ERROR(
      PopulateWithPlans(multi_update_adds, multi_update_add_plans, resources_));
  TF_RETURN_IF_ERROR(
      PopulateWithPlans(multi_updates, multi_update_plans, resources_));

  return false;
}

}  // namespace poplarplugin
}  // namespace xla
