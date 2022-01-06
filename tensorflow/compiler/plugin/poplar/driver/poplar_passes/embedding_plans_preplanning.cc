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

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

#include <poplar/OptionFlags.hpp>
#include <poplar/Target.hpp>
#include <popops/DynamicSlice.hpp>

namespace xla {
namespace poplarplugin {
namespace {

// Describes the type of slice plan that should be used.
enum class PlanType {
  // Slice plan only used when all operations are multi slices.
  kSliceOnly,
  // Slice plan only used when all operations are multi updates.
  kUpdateOnly,
  // Slice plan only used when all operations are multi update adds.
  kUpdateAddOnly,
  // Slice plan only used when all operations are either multi slice or multi
  // update.
  kSliceAndUpdate,
  // Slice plan only used when all operations are either multi slice or multi
  // update adds.
  kSliceAndUpdateAdd,
};

using InputToSliceUsersMap =
    ConstHloInstructionMap<std::vector<const HloInstruction*>>;

bool IsMultiSlice(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::MultiSlice)(inst);
}

bool IsMultiUpdate(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::MultiUpdate)(inst);
}

bool IsMultiUpdateAdd(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(inst);
}

bool IsSliceOnlyPlan(const std::vector<const HloInstruction*> insts) {
  return absl::c_all_of(insts, IsMultiSlice);
}

bool IsUpdateOnlyPlan(const std::vector<const HloInstruction*> insts) {
  return absl::c_all_of(insts, IsMultiUpdate);
}

bool IsUpdateAddOnlyPlan(const std::vector<const HloInstruction*> insts) {
  return absl::c_all_of(insts, IsMultiUpdateAdd);
}

bool IsSliceAndUpdatePlan(const std::vector<const HloInstruction*> insts) {
  return absl::c_all_of(insts, [](const HloInstruction* inst) -> bool {
    return IsMultiSlice(inst) || IsMultiUpdate(inst);
  });
}

bool IsSliceAndUpdateAddPlan(const std::vector<const HloInstruction*> insts) {
  return absl::c_all_of(insts, [](const HloInstruction* inst) -> bool {
    return IsMultiSlice(inst) || IsMultiUpdateAdd(inst);
  });
}

StatusOr<bool> AllOptionsMatch(const std::vector<const HloInstruction*> insts,
                               CompilerResources& res) {
  CHECK(insts.size());
  TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                      GetSliceOptionsForInst(insts[0], res));
  for (auto inst : insts) {
    TF_ASSIGN_OR_RETURN(poplar::OptionFlags inst_opts,
                        GetSliceOptionsForInst(inst, res));
    if (!(opts == inst_opts)) {
      return false;
    }
  }
  return true;
}

// Given the slice instructions, get all the slice sizes.
const std::vector<std::size_t> GetLookUps(
    const std::vector<const HloInstruction*>& slices) {
  std::vector<std::size_t> lookups(slices.size());
  absl::c_transform(slices, lookups.begin(), [](const HloInstruction* inst) {
    // All the slice instructions have the indices as operand 1.
    return ShapeUtil::ElementsIn(inst->operand(1)->shape());
  });
  return lookups;
}

// Given the lookups, get the total number of slices.
std::size_t GetNumLookUps(const std::vector<std::size_t>& lookups) {
  return absl::c_accumulate(lookups, std::size_t(0), std::plus<std::size_t>());
}

poplar::OptionFlags PopulateWithOptionsForPlan(
    const PlanType& plan_type, const poplar::OptionFlags& opts) {
  poplar::OptionFlags new_opts = opts;
  switch (plan_type) {
    case PlanType::kSliceOnly: {
      new_opts.set("usedForSlice", "true");
      new_opts.set("usedForUpdate", "false");
      break;
    }
    case PlanType::kUpdateOnly: {
      new_opts.set("usedForSlice", "false");
      new_opts.set("usedForUpdate", "true");
      new_opts.set("operationForUpdate", "none");
      break;
    }
    case PlanType::kUpdateAddOnly: {
      new_opts.set("usedForSlice", "false");
      new_opts.set("usedForUpdate", "true");
      new_opts.set("operationForUpdate", "add");
      break;
    }
    case PlanType::kSliceAndUpdate: {
      new_opts.set("usedForSlice", "true");
      new_opts.set("usedForUpdate", "true");
      new_opts.set("operationForUpdate", "none");
      break;
    }
    case PlanType::kSliceAndUpdateAdd: {
      new_opts.set("usedForSlice", "true");
      new_opts.set("usedForUpdate", "true");
      new_opts.set("operationForUpdate", "add");
      break;
    }
    default: { LOG(FATAL) << "Unknown PlanType"; }
  }

  return new_opts;
}

Status AddPlan(const PlanType& plan_type,
               const std::vector<std::size_t>& lookups,
               const HloInstruction* operand,
               const std::vector<const HloInstruction*>& insts,
               CompilerResources& res) {
  // Get the type information for the plan.
  const Shape operand_shape = operand->shape();
  TF_ASSIGN_OR_RETURN(poplar::Type data_type, PoplarDataType(operand_shape));
  const int64 input_size = operand_shape.dimensions(0);
  const int64 output_size = operand_shape.dimensions(1);

  // Get the options.
  TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                      GetSliceOptionsForInst(insts[0], res));
  opts = PopulateWithOptionsForPlan(plan_type, opts);

  // Get all the shards for operand/insts.
  absl::flat_hash_set<int64> shards;
  if (operand->has_sharding()) {
    // If there is sharding, we expect it to be unique.
    auto get_shard = [](const HloInstruction* inst) {
      return inst->sharding().GetUniqueDevice();
    };
    shards.insert(get_shard(operand));
    absl::c_transform(
        insts, std::inserter(shards, shards.begin()),
        [&get_shard](const HloInstruction* inst) { return get_shard(inst); });
  }

  // Plan in the master graph if this is used across multiple graph.
  poplar::Graph& graph =
      shards.size() > 1 ? GetMasterGraph(res) : GetGraph(res, operand);

  // Add the plan.
  res.slice_plans.push_back(popops::embedding::plan(
      graph, data_type, input_size, output_size, lookups, opts));

  for (const HloInstruction* inst : insts) {
    // Map the plan to the instruction in the original module.
    const HloInstruction* original_inst =
        res.annotations.flattened_inst_map_bwd.at(inst);
    res.slice_plan_mappings[original_inst] = &res.slice_plans.back();
  }

  return Status::OK();
}

Status PopulatePlans(const InputToSliceUsersMap& user_map,
                     CompilerResources& res) {
  for (auto& pair : user_map) {
    const HloInstruction* operand = pair.first;
    const std::vector<const HloInstruction*> slice_ops = pair.second;
    const std::vector<std::size_t> lookups = GetLookUps(slice_ops);
    // Only share a plan if the options match.
    TF_ASSIGN_OR_RETURN(const bool options_match,
                        AllOptionsMatch(slice_ops, res));

    if (IsSliceOnlyPlan(slice_ops) && options_match) {
      TF_RETURN_IF_ERROR(AddPlan(/*plan_type=*/PlanType::kSliceOnly, lookups,
                                 operand, slice_ops, res));
    } else if (IsUpdateOnlyPlan(slice_ops) && options_match) {
      TF_RETURN_IF_ERROR(AddPlan(/*plan_type=*/PlanType::kUpdateOnly, lookups,
                                 operand, slice_ops, res));
    } else if (IsUpdateAddOnlyPlan(slice_ops) && options_match) {
      TF_RETURN_IF_ERROR(AddPlan(/*plan_type=*/PlanType::kUpdateAddOnly,
                                 lookups, operand, slice_ops, res));
    } else if (IsSliceAndUpdatePlan(slice_ops) && options_match) {
      TF_RETURN_IF_ERROR(AddPlan(/*plan_type=*/PlanType::kSliceAndUpdate,
                                 lookups, operand, slice_ops, res));
    } else if (IsSliceAndUpdateAddPlan(slice_ops) && options_match) {
      TF_RETURN_IF_ERROR(AddPlan(/*plan_type=*/PlanType::kSliceAndUpdateAdd,
                                 lookups, operand, slice_ops, res));
    } else {
      // Unsupported mix - make a plan for each instruction.
      for (int64 i = 0; i != slice_ops.size(); ++i) {
        const HloInstruction* inst = slice_ops[i];
        const std::size_t lookup = lookups[i];

        PlanType plan_type;
        if (IsMultiSlice(inst)) {
          plan_type = PlanType::kSliceOnly;
        } else if (IsMultiUpdate(inst)) {
          plan_type = PlanType::kUpdateOnly;
        } else {
          CHECK(IsMultiUpdateAdd(inst));
          plan_type = PlanType::kUpdateAddOnly;
        }
        TF_RETURN_IF_ERROR(AddPlan(plan_type, /*lookups=*/{lookup}, operand,
                                   /*insts=*/{inst}, res));
      }
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
 * This pass works in the following way:
 * 1. In the flattened_module find all calls to multiSlice /
 * multiUpdate / multiUpdateAdd which share the lookup operand (operand idx 0).
 * 2. For each group of instructions find whether they can be categorized into
 * PlanType as a group (see the enum description), if not, create a slice plan
 * for each instruction.
 * 3. Translate the instructions back into the module to make sure each
 * instruction has a plan.
 */
StatusOr<bool> EmbeddingPlansPreplanning::Run(HloModule* module) {
  VLOG(2) << "Preplanning embedding operations.";

  absl::flat_hash_map<const HloInstruction*, popops::SlicePlan> slice_plans;
  HloComputation* entry_computation =
      resources_.annotations.flattened_module->entry_computation();

  InputToSliceUsersMap slice_ops_map;
  std::vector<const HloInstruction*> slice_ops;

  for (const HloInstruction* inst :
       entry_computation->MakeInstructionPostOrder()) {
    if (IsMultiSliceOrUpdate(inst)) {
      slice_ops.push_back(inst);
      slice_ops_map[inst->operand(0)].push_back(inst);
    }
  }

  // Count slice ops in the current HloModule.
  size_t non_flattened_slice_ops = 0;
  for (auto* comp : module->computations()) {
    for (const HloInstruction* inst : comp->instructions()) {
      if (IsMultiSliceOrUpdate(inst)) {
        const HloInstruction* flattened_slice_op =
            resources_.annotations.flattened_inst_map_fwd.at(inst);
        ++non_flattened_slice_ops;
        // Check precise mappings if VLOG is enabled.
        if (VLOG_IS_ON(1)) {
          if (absl::c_find(slice_ops, flattened_slice_op) == slice_ops.end()) {
            VLOG(1) << "Slice operation " << inst->ToString()
                    << " does not exist in the flattened graph";
          }
        }
      }
    }
  }
  if (slice_ops.size() != non_flattened_slice_ops) {
    return FailedPrecondition(
        "%s flattened module with %d slice ops does not match HloModule with "
        "%d slice ops",
        name(), slice_ops.size(), non_flattened_slice_ops);
  }

  TF_RETURN_IF_ERROR(PopulatePlans(slice_ops_map, resources_));
  return false;
}

}  // namespace poplarplugin
}  // namespace xla
