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
#include "tensorflow/compiler/plugin/poplar/driver/tools/embedding_plans_preplanning.h"

#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

#include <poplar/Target.hpp>

namespace xla {
namespace poplarplugin {

namespace {
class EmbeddingSlicePlanInfo {
 public:
  Status AddOperand(const HloInstruction* operand, int index,
                    bool is_multi_update) {
    switch (index) {
      case 0:
        return HandleInputOperand(operand);
      case 1:
        return HandleIndicesOperand(operand);
      case 2:
        if (is_multi_update) {
          // It's the updates tensor: nothing to do
          return Status::OK();
        }
      default:
        return xla::InvalidArgument("Invalid index");
    };
  }

  Status Merge(EmbeddingSlicePlanInfo& other) {
    if (other.input_size != input_size && other.input_size > 0 &&
        input_size > 0) {
      return xla::InvalidArgument(
          "Input sizes mismatch: can't share a SlicePlan");
    }
    if (other.output_size != output_size && output_size > 0 &&
        other.output_size > 0) {
      return xla::InvalidArgument(
          "Output sizes mismatch: can't share a SlicePlan");
    }

    absl::c_copy(other.indices, std::back_inserter(indices));
    if (input_size == 0) {
      input_size = other.input_size;
    }
    if (output_size == 0) {
      output_size = other.output_size;
    }
    other.merged = true;
    return Status::OK();
  }

  StatusOr<popops::SlicePlan*> GetOrCreatePlan(const poplar::Graph& graph,
                                               CompilerResources& resources) {
    if (plan) {
      return plan;
    }
    if (merged) {
      return xla::InvalidArgument("This plan was merged with another plan.");
    }
    if (input_size == 0) {
      return xla::InvalidArgument("Unknown input size");
    }
    if (output_size == 0) {
      return xla::InvalidArgument("Unknown output size");
    }
    if (indices.size() == 0) {
      return xla::InvalidArgument("No known slice/update");
    }
    // Allocate a new SlicePlan.
    resources.slice_plans.push_back(popops::embedding::plan(
        graph, data_type, input_size, output_size, indices, {}));
    plan = &resources.slice_plans.back();
    return plan;
  }

 private:
  Status HandleInputOperand(const HloInstruction* operand) {
    const Shape& shape = operand->shape();
    if (shape.dimensions_size() != 2) {
      return xla::InvalidArgument("Embeddings must be 2D");
    }
    if (input_size > 0 &&
        input_size != shape.dimensions(0) * shape.dimensions(1)) {
      return xla::InvalidArgument(
          "Input sizes mismatch: can't share a SlicePlan");
    }
    TF_ASSIGN_OR_RETURN(data_type, PoplarDataType(shape));
    input_size = shape.dimensions(0) * shape.dimensions(1);
    output_size = shape.dimensions(1);
    return Status::OK();
  }

  Status HandleIndicesOperand(const HloInstruction* operand) {
    const Shape& shape = operand->shape();
    size_t num_indices =
        std::accumulate(shape.dimensions().begin(), shape.dimensions().end(), 1,
                        std::multiplies<std::size_t>());
    indices.push_back(num_indices);
    return Status::OK();
  }

  popops::SlicePlan* plan{nullptr};
  poplar::Type data_type;
  size_t input_size{0};
  size_t output_size{0};
  std::vector<size_t> indices;
  bool merged{false};
};

class EmbeddingPlansMap {
 public:
  using PlanId = int;
  using OpIndex = int;

  Status AddInstruction(const HloInstruction* inst, bool is_multi_slice) {
    // Retrieve an existing plan or create a new one if needed.
    TF_ASSIGN_OR_RETURN(PlanId plan, _AddInstructionToMappings(inst));

    // Collect the data needed to create the SlicePlan.
    EmbeddingSlicePlanInfo& info = plans[plan];
    for (int i = 0; i < inst->operand_count(); i++) {
      auto* op = inst->operand(i);
      info.AddOperand(op, i, is_multi_slice);
    }
    return Status::OK();
  }

  Status PopulateCompilerResources(CompilerResources& resources) {
    // Associate each instruction from mappings to a SlicePlan.
    for (auto mapping : mappings) {
      poplar::Graph& graph = GetGraph(resources, mapping.first);
      TF_ASSIGN_OR_RETURN(
          popops::SlicePlan * plan,
          plans[mapping.second].GetOrCreatePlan(graph, resources));
      resources.slice_plan_mappings[resources.annotations.flattened_inst_map_bwd
                                        .at(mapping.first)] = plan;
    }
    return Status::OK();
  }

 private:
  StatusOr<PlanId> _AddInstructionToMappings(const HloInstruction* inst) {
    std::vector<const HloInstruction*> new_ops;
    PlanId plan = -1;

    for (int i = 0; i < inst->operand_count(); i++) {
      const HloInstruction* op = inst->operand(i);
      auto mapping = mappings.find(op);
      if (mapping == mappings.end()) {
        if (plan < 0) {
          // Defer creating a new plan in case one of the other operands is
          // associated to an existing plan
          new_ops.push_back(op);
        } else {
          mappings[op] = plan;
        }
      } else {
        if (plan < 0) {
          plan = mapping->second;
        } else if (plan != mapping->second) {
          TF_ASSIGN_OR_RETURN(plan, _FusePlans(plan, mapping->second));
        }
      }
    }
    if (plan < 0) {
      plan = plans.size();
      plans.emplace_back();
    }
    for (auto new_op : new_ops) {
      mappings[new_op] = plan;
    }
    return plan;
  }

  StatusOr<PlanId> _FusePlans(PlanId plan, PlanId to_merge) {
    TF_RETURN_IF_ERROR(plans[plan].Merge(plans[to_merge]));
    for (auto mapping : mappings) {
      if (mapping.second == to_merge) {
        mapping.second = plan;
      }
    }

    return plan;
  }

  std::map<const HloInstruction*, PlanId> mappings;
  std::vector<EmbeddingSlicePlanInfo> plans;
};
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
Status EmbeddingPlansPreplanning::Plan(const HloModule* module,
                                       CompilerResources& resources) {
  EmbeddingPlansMap embedding_plans;
  for (auto* comp : resources.annotations.flattened_module->computations()) {
    for (HloInstruction* inst : comp->instructions()) {
      // Identify all the multiSlice / multiUpdate / multiUpdateAdd operations
      // and add them to the list of plans.
      bool is_multi_slice = DynCast<HloMultiSliceInstruction>(inst) != nullptr;
      bool is_multi_update =
          DynCast<HloMultiUpdateInstruction>(inst) != nullptr;
      if (IsPopOpsFusion(inst)) {
        HloComputation* comp = inst->fused_instructions_computation();
        auto end = comp->name().find('.');
        std::string name = comp->name().substr(8, end - 8);
        if (name == "fused_multi_update_add") {
          is_multi_update = true;
        }
      }

      if (is_multi_slice || is_multi_update) {
        TF_RETURN_IF_ERROR(
            embedding_plans.AddInstruction(inst, is_multi_slice));
      }
    }
  }

  // Allocate all the SlicePlans and populate slice_plans and
  // slice_plan_mappings in CompilerResources.
  TF_RETURN_IF_ERROR(embedding_plans.PopulateCompilerResources(resources));

  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla
