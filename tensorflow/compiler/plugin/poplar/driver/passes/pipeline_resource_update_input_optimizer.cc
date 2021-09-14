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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_resource_update_input_optimizer.h"

#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/fifo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {

namespace {

using ParamSet = HloInstructionSet;

struct ComputationOutput {
  const HloComputation* computation;
  int64 output_index;

  bool operator==(const ComputationOutput& other) const {
    return computation == other.computation &&
           output_index == other.output_index;
  }
  struct Hash {
    std::size_t operator()(const ComputationOutput& output) const {
      auto h1 = std::hash<const HloComputation*>{}(output.computation);
      auto h2 = std::hash<int64>{}(output.output_index);
      return tensorflow::Hash64Combine(h1, h2);
    }
  };
};

// This class provides simple dependency map for elementwise expression within
// HloComputation.
// Consider the following example:
// stage_0 {
//  arg0 = parameter(0)
//  arg1 = parameter(1)
//  add = add(arg0, arg1)
//  ROOT output = tuple(arg0, add)
// }
// GetParametersForOutput(stage_0, 0) will return { arg0 }
// GetParametersForOutput(stage_0, 1) will return { arg0, arg1 }
class ElementwiseExpressionDependencyMap {
 public:
  // For the given computation output, GetParameterIndices returns
  // corrensponding parameter instructions for the elementwise expression. In
  // case there's no valid elementwise expression, GetParameterIndices return
  // false.
  bool GetParametersForOutput(const HloComputation* comp, int64 output_index,
                              ParamSet& result) {
    auto key = ComputationOutput{comp, output_index};
    auto it = dependencies_.find(key);
    if (it != dependencies_.end()) {
      result = it->second.inputs;
      return it->second.valid;
    }

    std::queue<HloInstruction*> to_check;
    to_check.push(comp->root_instruction()->mutable_operand(output_index));
    ParamSet inputs;

    while (!to_check.empty()) {
      HloInstruction* next = to_check.front();
      to_check.pop();
      if (next->opcode() == HloOpcode::kParameter) {
        VLOG(3) << "Adding " << next->ToString() << " to input set";
        inputs.insert(next);
      } else if (next->called_computations().empty() &&
                 !IsPoplarInstruction(PoplarOp::ExecutionCounter, next) &&
                 !next->HasSideEffect()) {
        // TODO(T41424): Remove execution counter check if we consider adding
        // side effect flag to the execution counter instruction.
        for (HloInstruction* op : next->operands()) {
          VLOG(3) << "Checking stateless operand " << op->ToString();
          to_check.push(op);
        }
      } else {
        VLOG(3) << next->ToString()
                << " is not a constant or parameter, ignoring.";
        dependencies_.emplace(key, Dependency{/*valid=*/false, {}});
        return false;
      }
    }
    result = dependencies_
                 .emplace(key, Dependency{/*valid=*/true, std::move(inputs)})
                 .first->second.inputs;
    return true;
  }

 private:
  struct Dependency {
    bool valid;
    ParamSet inputs;
  };
  absl::flat_hash_map<ComputationOutput, Dependency, ComputationOutput::Hash>
      dependencies_;
};

struct ReplacementPlanApplyContext {
  HloInstruction* resource_update;
  HloCloneContext clone_context;
  absl::flat_hash_map<ComputationOutput, HloInstruction*,
                      ComputationOutput::Hash>
      stage_output_clone;
  absl::flat_hash_map<int64, int64> pipeline_input_to_ru_arg_map;

  ReplacementPlanApplyContext(
      HloInstruction* resource_update,
      absl::flat_hash_map<int64, int64>&& pipeline_input_to_ru_arg_map)
      : resource_update(resource_update),
        clone_context(resource_update->parent()->parent()),
        pipeline_input_to_ru_arg_map(std::move(pipeline_input_to_ru_arg_map)) {}
};

class ReplacementPlan {
 private:
  int64 resource_update_param_idx_;
  std::set<int64> pipeline_inputs_;
  struct Replacement {
    HloInstruction* call;
    int64 stage_index;
    int64 output_index;
    ParamSet inputs;
  };
  std::list<Replacement> replacements_;

 public:
  explicit ReplacementPlan(int64 resource_update_param_idx)
      : resource_update_param_idx_(resource_update_param_idx) {}

  const std::set<int64>& GetPipelineInputs() const { return pipeline_inputs_; }

  bool Build(HloInstruction* resource_update_arg, const PipelineStages& stages,
             ElementwiseExpressionDependencyMap& deps) {
    if (resource_update_arg->opcode() != HloOpcode::kGetTupleElement) {
      return false;
    }

    std::queue<HloInstruction*> to_check;
    absl::flat_hash_set<HloInstruction*> visited;
    to_check.push(resource_update_arg);
    while (!to_check.empty()) {
      auto current_arg = to_check.front();
      to_check.pop();
      if (visited.contains(current_arg)) {
        continue;
      }
      visited.insert(current_arg);
      VLOG(3) << "Checking pipeline instruction " << current_arg->ToString();
      if (current_arg->opcode() == HloOpcode::kGetTupleElement) {
        auto call = current_arg->mutable_operand(0);
        if (!IsPipelineStageOrBackwardOp(call)) {
          VLOG(3) << "GTE target is not a pipeline stage.";
          return false;
        }
        HloComputation* call_comp = call->to_apply();
        std::size_t stage_index = std::distance(
            stages.forward.begin(),
            absl::c_find_if(stages.forward,
                            [call_comp](const HloInstruction* fwd) {
                              return fwd->to_apply() == call_comp;
                            }));

        if (stage_index == stages.forward.size()) {
          VLOG(3) << "Unknown stage." << call->ToString();
          return false;
        }

        int64 output_index = current_arg->tuple_index();
        ParamSet inputs;
        if (!deps.GetParametersForOutput(call_comp, output_index, inputs)) {
          VLOG(3) << "Not an elementwise cluster";
          return false;
        }
        replacements_.push_back(
            Replacement{call, stage_index, output_index, inputs});
        for (HloInstruction* input : inputs) {
          auto arg = call->mutable_operand(input->parameter_number());
          VLOG(3) << "Queueing argument " << arg->ToString();
          to_check.push(arg);
        }
      } else if (current_arg->opcode() == HloOpcode::kParameter) {
        VLOG(3) << "Found pipeline argument " << current_arg->ToString();
        pipeline_inputs_.insert(current_arg->parameter_number());
      } else {
        VLOG(3) << "Invalid pipeline instruction " << current_arg->ToString();
        return false;
      }
    }
    if (replacements_.empty()) {
      return false;
    }
    // Sort replacements in stages from first to the last
    absl::c_reverse(replacements_);
    return true;
  }

  Status Apply(ReplacementPlanApplyContext& context) const {
    HloInstruction* resource_update = context.resource_update;
    HloComputation* resource_update_comp = resource_update->to_apply();
    HloCloneContext& clone_context = context.clone_context;
    auto& stage_output_clone = context.stage_output_clone;
    auto& pipeline_input_to_ru_arg_map = context.pipeline_input_to_ru_arg_map;

    for (auto& repl : replacements_) {
      auto comp = repl.call->to_apply();
      auto key = ComputationOutput{comp, repl.output_index};
      if (stage_output_clone.contains(key)) {
        continue;
      }
      VLOG(3) << "Processing the output of " << comp->name() << ":"
              << repl.output_index;
      auto output =
          comp->root_instruction()->mutable_operand(repl.output_index);
      VLOG(3) << "Output: " << output->ToString();

      for (auto input : repl.inputs) {
        auto call_arg = repl.call->mutable_operand(input->parameter_number());
        VLOG(3) << "Mapping first stage input " << input->ToString() << " to "
                << call_arg->ToString();
        // For the first stage accept only pipeline inputs
        // For any other stage accept either pipeline input or GTEs from
        // previous stage (checked above).
        if (repl.stage_index == 0 ||
            call_arg->opcode() != HloOpcode::kGetTupleElement) {
          CHECK_EQ(call_arg->opcode(), HloOpcode::kParameter);
          auto resource_update_param =
              resource_update_comp->parameter_instruction(
                  pipeline_input_to_ru_arg_map.at(
                      call_arg->parameter_number()));
          VLOG(3) << "Found resource update param: "
                  << resource_update_param->ToString();
          clone_context.MapInstruction(input, resource_update_param);
        } else {
          CHECK_EQ(call_arg->opcode(), HloOpcode::kGetTupleElement);
          auto prev_stage = call_arg->mutable_operand(0);
          CHECK_EQ(prev_stage->opcode(), HloOpcode::kCall);
          auto prev_output = stage_output_clone.at(ComputationOutput{
              prev_stage->to_apply(), call_arg->tuple_index()});
          VLOG(3) << "Found output of the previous stage: "
                  << prev_output->ToString();
          clone_context.MapInstruction(input, prev_output);
        }
      }

      HloInstruction* clone = clone_context.FindInstruction(output);
      VLOG(3) << "Finding clone for output " << output->ToString() << " -> "
              << (clone ? clone->ToString() : "null");
      if (!clone) {
        CHECK_NE(output->opcode(), HloOpcode::kParameter);
        TF_ASSIGN_OR_RETURN(
            clone, CloneComputationSubtree(output, resource_update_comp, "",
                                           &clone_context));
        VLOG(3) << "Adding output: " << clone->ToString();
        stage_output_clone[key] = clone;
        clone_context.MapInstruction(output, clone);
      } else {
        VLOG(3) << "Adding previously cloned output: " << clone->ToString();
        stage_output_clone[key] = clone;
      }
    }

    // Replace old argument uses:
    const HloInstruction* resource_update_arg =
        resource_update->operand(resource_update_param_idx_);
    CHECK_EQ(resource_update_arg->opcode(), HloOpcode::kGetTupleElement);
    HloInstruction* last_stage_output = stage_output_clone.at(
        ComputationOutput{resource_update_arg->operand(0)->to_apply(),
                          resource_update_arg->tuple_index()});
    HloInstruction* resource_update_param =
        resource_update_comp->parameter_instruction(resource_update_param_idx_);
    VLOG(2) << "Replace " << resource_update_param->ToString()
            << " uses with the output from the last stage: "
            << resource_update_param_idx_ << ": "
            << last_stage_output->ToString();
    TF_RETURN_IF_ERROR(
        resource_update_param->ReplaceAllUsesWith(last_stage_output));
    return Status::OK();
  }
};

}  // namespace

StatusOr<bool> PipelineResourceUpdateInputOptimizer::OptimizePipeline(
    HloInstruction* pipeline_op) {
  HloComputation* pipeline_comp = pipeline_op->to_apply();

  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  TF_RETURN_IF_ERROR(FixRootInstructions(stages));
  if (!stages.resource_update) {
    return false;
  }

  HloInstruction* resource_update = *stages.resource_update;
  HloComputation* resource_update_comp = resource_update->to_apply();

  // Generate replacement plan for each resource update argument
  // Plan contains a number of stages replacements and dependent stage indices.

  // High level algorithm:
  // For each resource update argument:
  //  - Check that every stage either passes parameter to the output tuple or
  //  - Returns elementwise operations with parameters or constants.
  // For each stage create a Replacement structure with call,
  // output index and dependent input parameter indices.
  // This structure defines elementwise cluster to lower.
  //
  // Also collect pipeline computation inputs to pass them to the resource
  // update if they were not passed already.
  //
  // Clone elementwise clusters starting from the first stage.
  // Use HloCloneContext to connect inputs/outputs of those clusters together,
  // combining clusters into single expression

  ElementwiseExpressionDependencyMap dependencies;
  std::list<ReplacementPlan> plans;
  for (int64 i = 0; i < resource_update->operand_count(); ++i) {
    HloInstruction* ru_arg = resource_update->mutable_operand(i);
    VLOG(2) << "Checking resource update argument " << i << ": "
            << ru_arg->ToString();
    ReplacementPlan repl(i);
    if (repl.Build(ru_arg, stages, dependencies)) {
      VLOG(2) << "Found cluster suitable for lowering.";
      plans.push_back(std::move(repl));
    }
  }

  if (plans.empty()) {
    return false;
  }

  VLOG(1) << "Got " << plans.size() << " resource update arguments to replace.";

  // Map of the pipeline input parameter index to the resource update argument
  // index.
  absl::flat_hash_map<int64, int64> pipeline_input_to_ru_arg_map;
  // Resource update argument and new parameter instruction pair.
  std::vector<std::pair<HloInstruction*, std::unique_ptr<HloInstruction>>>
      new_params;
  auto new_resource_update_operands = resource_update->operands();
  // Collect all new parameters.
  for (auto& plan : plans) {
    for (auto pipeline_input : plan.GetPipelineInputs()) {
      if (pipeline_input_to_ru_arg_map.contains(pipeline_input)) {
        continue;
      }

      auto param = pipeline_comp->parameter_instruction(pipeline_input);
      auto indices = resource_update->OperandIndices(param);
      if (indices.empty()) {
        auto index = resource_update->operand_count() + new_params.size();
        pipeline_input_to_ru_arg_map[pipeline_input] = index;
        new_params.emplace_back(
            param, HloInstruction::CreateParameter(index, param->shape(),
                                                   param->name()));
      } else {
        pipeline_input_to_ru_arg_map[pipeline_input] = indices[0];
      }
    }
  }

  // Adding new parameters to the resource update computation.
  if (!new_params.empty()) {
    VLOG(3) << "Adding " << new_params.size()
            << " new parameter(s) to resource update...";
    HloModule* module = resource_update_comp->parent();
    std::vector<const HloInstruction*> new_param_ptrs;
    for (const auto& p : new_params) {
      new_resource_update_operands.push_back(p.first);
      new_param_ptrs.push_back(p.second.get());
    }
    HloComputation* new_resource_update_comp = module->AddEmbeddedComputation(
        resource_update_comp->CloneWithReplacements({}, new_param_ptrs,
                                                    nullptr));
    resource_update->set_to_apply(new_resource_update_comp);
    TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(resource_update_comp));
    resource_update_comp = new_resource_update_comp;

    CHECK_EQ(resource_update->parent(), pipeline_comp);
    HloInstruction* new_resource_update =
        pipeline_comp->AddInstruction(resource_update->CloneWithNewOperands(
            resource_update->shape(), new_resource_update_operands));
    TF_RETURN_IF_ERROR(
        resource_update->ReplaceAllUsesWith(new_resource_update));
    TF_RETURN_IF_ERROR(pipeline_comp->RemoveInstruction(resource_update));
    resource_update = new_resource_update;
  }

  ReplacementPlanApplyContext context(resource_update,
                                      std::move(pipeline_input_to_ru_arg_map));
  for (auto& plan : plans) {
    TF_RETURN_IF_ERROR(plan.Apply(context));
  }
  return true;
}

StatusOr<bool> PipelineResourceUpdateInputOptimizer::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to optimize.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  HloInstruction* pipeline_op = pipeline_ops[0];
  VLOG(2) << "Before PipelineResourceUpdateInputOptimizer:";
  XLA_VLOG_LINES(2, module->ToString());

  TF_ASSIGN_OR_RETURN(bool changed, OptimizePipeline(pipeline_op));

  if (changed) {
    VLOG(2) << "After PipelineResourceUpdateInputOptimizer:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the Pipeline.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
