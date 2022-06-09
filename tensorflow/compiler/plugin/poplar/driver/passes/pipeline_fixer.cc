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
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_fixer.h"

#include <functional>
#include <list>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <utility>

#include "absl/strings/str_replace.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/elementwise.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/execution_counter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_noop.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace {
namespace m = match;
}
namespace poplarplugin {
using pipeline_config = PoplarBackendConfig::CallConfig::PipelineConfig;

Status PipelineFixer::InsertStatefulNoopsIntoStages() {
  for (auto& stages : {stages_.forward, stages_.backward}) {
    for (HloInstruction* stage : stages) {
      HloComputation* stage_computation = stage->to_apply();
      stage_computation->AddInstruction(CreateStatefulNoop());
    }
  }
  return Status::OK();
}

Status PipelineFixer::UpdateStage(const StageID& stage_id,
                                  HloInstruction* new_stage) {
  if (stage_id.stage_type == StageType::kForward) {
    stages_.forward[stage_id.id] = new_stage;
  } else {
    stages_.backward[stage_id.id] = new_stage;
  }
  return Status::OK();
}

HloInstruction* PipelineFixer::GetStage(const StageID& stage_id) {
  if (stage_id.stage_type == StageType::kForward) {
    return stages_.forward[stage_id.id];
  } else {
    return stages_.backward[stage_id.id];
  }
}

std::vector<HloInstruction*> PipelineFixer::GetOrderedStages() {
  std::vector<HloInstruction*> stages(stages_.forward.size() +
                                      stages_.backward.size());
  absl::c_copy(stages_.forward, stages.begin());
  std::copy(stages_.backward.rbegin(), stages_.backward.rend(),
            std::next(stages.begin(), stages_.forward.size()));
  return stages;
}

namespace {
StatusOr<std::vector<HloInstruction*>> FindClusterToLower(
    HloInstruction* stage, const StageID& stage_id,
    HloInstruction* lowering_root, PipelineDataflowAnalysis* analysis) {
  // Build a cluster of instructions which are lowered and order them.
  // Start from the root and build up the cluster by visiting both
  // operands and users of instructions already in the cluster.
  std::vector<HloInstruction*> ordered_lowering;
  std::vector<const HloValueSet*> value_sets;
  std::queue<HloInstruction*> to_visit;
  absl::flat_hash_set<HloInstruction*> visited;
  to_visit.push(lowering_root);

  while (!to_visit.empty()) {
    HloInstruction* inst = to_visit.front();
    to_visit.pop();

    if (visited.contains(inst)) {
      continue;
    }

    bool ready_to_lower = true;
    for (HloInstruction* operand : inst->operands()) {
      TF_ASSIGN_OR_RETURN(bool operand_needs_lowering,
                          analysis->HasToBeLoweredIntoStage(stage, operand));
      ready_to_lower &= (visited.contains(operand) || !operand_needs_lowering);
    }
    std::vector<HloInstruction*> candidates;
    if (ready_to_lower) {
      // If we are ready to lower an instruction then add its value set
      // and visit its children.
      ordered_lowering.push_back(inst);
      visited.insert(inst);
      value_sets.push_back(&analysis->GetValueSet(inst));
      candidates = inst->users();
    } else {
      // We need to lower operands first.
      candidates = {inst->operands().begin(), inst->operands().end()};
    }

    // Add any instructions which need to be considered for lowering.
    for (HloInstruction* candidate : candidates) {
      if (!visited.contains(candidate)) {
        TF_ASSIGN_OR_RETURN(
            bool needs_lowering,
            analysis->HasToBeLoweredIntoStage(stage, candidate));
        if (needs_lowering) {
          to_visit.push(candidate);
        }
      }
    }
  }
  HloValueSet value_set;
  value_set.AssignUnionOf(value_sets);

  // If the current stage is a forward stage and the value set contains a
  // backward stage then we can't lower this cluster into the fwd stage
  // without violating data flow constraints (i.e. backprop can't be used
  // in the forward pass).
  // We therefore skip and the bwd stage will deal with this.
  auto value_from_bwd = [](const HloValue* value) {
    return IsPipelineStageBackward(value->instruction());
  };

  if (stage_id.stage_type == StageType::kForward &&
      absl::c_any_of(value_set.values(), value_from_bwd)) {
    return std::vector<HloInstruction*>();
  }

  // Verify we can lower this.
  TF_RETURN_IF_ERROR(analysis->VerifyPipelineStageOperands(stage, value_set));
  return ordered_lowering;
}

// Tidy function to remove any dangling outputs from a stage and prevent use
// after free.
Status RemovePipelineStageDeadUsers(HloInstruction* stage,
                                    HloInstructionSet& stage_users) {
  std::vector<HloInstruction*> users = stage->users();
  HloComputation* comp = stage->parent();
  for (HloInstruction* gte : users) {
    CHECK_EQ(gte->opcode(), HloOpcode::kGetTupleElement);
    if (gte->user_count() == 0) {
      stage_users.erase(gte);
      TF_RETURN_IF_ERROR(comp->RemoveInstruction(gte));
    }
  }
  return Status::OK();
}

int64 GetNextStageID(int64& current,
                     const std::vector<HloInstruction*>& stages) {
  if (static_cast<size_t>(current + 1) < stages.size()) {
    current++;
  }
  return GetPipelineStageID(stages[current]);
}
}  // namespace

StatusOr<bool> PipelineFixer::BreakUpElementwiseOperations() {
  bool changed = false;

  TF_ASSIGN_OR_RETURN(auto analysis,
                      PipelineDataflowAnalysis::GetAnalysis(stages_));

  // We only break up elementwise operations which are not inside of stages yet.
  HloComputation* comp = stages_.forward[0]->parent();
  bool progress_made = true;
  while (progress_made) {
    progress_made = false;
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (inst->opcode() != HloOpcode::kAdd) {
        continue;
      }

      // Currently only try to breakup if all the values come from different
      // backward pipeline stages which are placed on the same shard.
      HloValueSet value_set = analysis->GetOperandsValueSet(inst);
      absl::flat_hash_set<int64> backward_pipeline_stages;
      absl::flat_hash_set<int64> shards;
      for (const HloValue* value : value_set.values()) {
        HloInstruction* producer = value->instruction();
        if (IsPipelineStageBackward(producer)) {
          TF_ASSIGN_OR_RETURN(StageID stage_id, analysis->GetStageID(producer));
          TF_ASSIGN_OR_RETURN(const int64 shard,
                              analysis->GetShardForStage(stage_id));

          backward_pipeline_stages.insert(stage_id.id);
          shards.insert(shard);
        }
      }
      // Skip if the backward stages are not unique.
      if (value_set.values().size() != backward_pipeline_stages.size()) {
        continue;
      }
      // Skip if they are not on the same shard.
      if (shards.size() != 1) {
        continue;
      }

      for (HloInstruction* user : inst->users()) {
        const auto indices = user->OperandIndices(inst);
        // Convert Multiply(A, Add(B, C))
        // to Add(Multiply(A, B), Multiply(A, C))
        // Note that the order of operands to multiply can be swapped.
        if (indices.size() == 1 && user->opcode() == HloOpcode::kMultiply) {
          HloInstruction* scale = user->mutable_operand((indices[0] + 1) % 2);
          HloInstruction* add_lhs = inst->mutable_operand(0);
          HloInstruction* add_rhs = inst->mutable_operand(1);
          TF_ASSIGN_OR_RETURN(
              HloInstruction * multiply_lhs,
              MakeBinaryHlo(HloOpcode::kMultiply, scale, add_lhs));
          TF_ASSIGN_OR_RETURN(
              HloInstruction * multiply_rhs,
              MakeBinaryHlo(HloOpcode::kMultiply, scale, add_rhs));
          TF_ASSIGN_OR_RETURN(
              HloInstruction * new_add,
              MakeBinaryHlo(HloOpcode::kAdd, multiply_lhs, multiply_rhs));
          TF_RETURN_IF_ERROR(comp->ReplaceInstruction(user, new_add));

          // Recompute the analysis.
          TF_ASSIGN_OR_RETURN(analysis,
                              PipelineDataflowAnalysis::GetAnalysis(stages_));
          progress_made = true;
          // Need to stop as user instructions have been modified.
          break;
        }
      }

      // Start from beginning as user instructions have been modified.
      if (progress_made) {
        break;
      }
    }
  }

  return changed;
}

namespace pipeline_fixer_util {
bool IsRunningMeanAccumulatorScale(HloInstruction* scale) {
  HloInstruction *counter0, *counter1;
  return Match(scale, m::Multiply(m::Convert(m::Op(&counter0)),
                                  m::Divide(m::ConstantScalar(1),
                                            m::Add(m::Convert(m::Op(&counter1)),
                                                   m::ConstantScalar(1))))) &&
         counter0 == counter1 &&
         IsPoplarInstruction(PoplarOp::ExecutionCounter, counter0) &&
         Cast<HloExecutionCounter>(counter0)->CanLowerIntoPipelineStage();
}

bool IsRunningMeanGradient(HloInstruction* gradient) {
  // If this is running mean then it is expected that the gradient is
  // multiplied by:
  // 1 / (convert(execution_counter) + 1).
  HloInstruction* scale;
  if (!Match(gradient, m::Multiply(m::Op(), m::Broadcast(m::Op(&scale))))) {
    VLOG(3) << "Gradient " << gradient->ToString()
            << " is not a valid running mean gradient as it is not scaled by a "
               "broadcast of a constant.";
    return false;
  }

  // Look through the convert on the scale.
  if (scale->opcode() == HloOpcode::kConvert) {
    scale = scale->mutable_operand(0);
  }

  HloInstruction* counter;
  if (Match(scale,
            m::Divide(m::ConstantScalar(1), m::Add(m::Convert(m::Op(&counter)),
                                                   m::ConstantScalar(1)))) &&
      IsPoplarInstruction(PoplarOp::ExecutionCounter, counter) &&
      Cast<HloExecutionCounter>(counter)->CanLowerIntoPipelineStage()) {
    return true;
  } else {
    VLOG(3) << "Gradient " << gradient->ToString()
            << " is not a valid running mean gradient as the scale is not "
            << "`1 / (execution_counter + 1)`.";
    return false;
  }
}

StatusOr<bool> GradientsNeedRescaling(pipeline_config::Schedule schedule,
                                      int64 accumulator_scale_stage_id,
                                      int64 gradient_stage_id) {
  CHECK_GE(accumulator_scale_stage_id, gradient_stage_id);

  switch (schedule) {
    case pipeline_config::Sequential: {
      // Sequential schedule does not need rescaling as only one stage is
      // executed at a time.
      return false;
    }
    case pipeline_config::Grouped: {
      // For grouped schedule we need to rescale all the gradients which are not
      // scaled in the same pipeline stage as the accumulator.
      return accumulator_scale_stage_id != gradient_stage_id;
    }
    case pipeline_config::Interleaved:
      // Interleaved schedule does not support this - this code should not be
      // reachable as it's not possible to have an interleaved schedule with
      // same variable used in multiple pipeline stages.
      TF_FALLTHROUGH_INTENDED;
    default:
      return FailedPrecondition("Unsupported pipeline schedule.");
  }
}

StatusOr<HloInstruction*> GetGradientScale(HloInstruction* accumulation_count,
                                           int64 accumulator_scale_stage_id,
                                           int64 gradient_stage_id) {
  CHECK_GE(accumulator_scale_stage_id, gradient_stage_id);
  const int64 offset = accumulator_scale_stage_id - gradient_stage_id;
  CHECK(offset);

  // Change the scale of the gradient to:
  // 1 / ((accumulation_count - offset) <= execution_counter ?
  //       accumulation_count : (execution_counter + offset))
  HloComputation* comp = accumulation_count->parent();
  HloInstruction* execution_counter = comp->AddInstruction(
      CreateExecutionCounter(/*lower_into_pipeline_stage=*/true));

  // Generate the predicate expression.
  TF_ASSIGN_OR_RETURN(HloInstruction * ac_minus_offset,
                      MakeBinaryHlo(HloOpcode::kSubtract, accumulation_count,
                                    MakeR0ConstantHlo<int32>(comp, offset)));
  TF_ASSIGN_OR_RETURN(HloInstruction * pred,
                      MakeCompareHlo(ComparisonDirection::kLe, ac_minus_offset,
                                     execution_counter));

  // Generate the on_true of the expression.
  HloInstruction* on_true = MakeConvertToHlo(accumulation_count, F32);

  // Generate the on_false of the expression.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * on_false,
      MakeBinaryHlo(HloOpcode::kAdd, MakeConvertToHlo(execution_counter, F32),
                    MakeR0ConstantHlo<float>(comp, offset)));

  // Create the conditional statement.
  TF_ASSIGN_OR_RETURN(HloInstruction * divisor,
                      MakeSelectHlo(pred, on_true, on_false));

  // Create an inverse.
  return comp->AddInstruction(CreateInverse(divisor));
}

StatusOr<HloInstruction*> RescaleGradient(HloInstruction* gradient,
                                          HloInstruction* scale) {
  HloInstruction* pre_scale_gradient;
  CHECK(
      Match(gradient, m::Multiply(m::Op(&pre_scale_gradient), m::Broadcast())));
  const Shape& gradient_shape = gradient->shape();

  if (gradient_shape.element_type() != scale->shape().element_type()) {
    scale = MakeConvertToHlo(scale, gradient_shape.element_type());
  }

  scale = MakeBroadcastHlo(scale, {}, gradient_shape.dimensions());

  TF_ASSIGN_OR_RETURN(
      gradient, MakeBinaryHlo(HloOpcode::kMultiply, pre_scale_gradient, scale));
  return gradient;
}
}  // namespace pipeline_fixer_util

StatusOr<bool> PipelineFixer::BreakUpGradientAccumulationOperations(
    HloInstruction* accumulation_count) {
  HloComputation* comp = stages_.forward[0]->parent();
  TF_ASSIGN_OR_RETURN(auto analysis,
                      PipelineDataflowAnalysis::GetAnalysis(stages_));

  // A function which detects outputs backward pipeline stages (producers) being
  // accumulated to create a gradient. This usually occurs when a resource
  // variable is used by multiple stages.
  //
  // This can be optimized by accumulating the gradients into the accumulation
  // buffer separately (note that this breaks SSA).
  //
  // For example:
  // bps2 = (...) bwd_pipeline_stage_2(...) stage_id = 2
  // bps2_grad = gte(bps2)
  // ...
  // bps0 = (...) bwd_pipeline_stage_0(...) stage_id = 0
  // bps0_grad = gte(bps0)
  //
  // gradient_buffer = gradient-accumulator-creator()
  // add = add(bsp0, bsp2)
  // gradient = gradient-accumulation-add(gradient_buffer, buffer)
  // accumulated_sink = gradient-accumulator-sink(gradient)
  //
  // Here the `add` operation means that `bps2_grad` will eventually have to be
  // passed as an input into `bwd_pipeline_stage_0` so that it can be
  // accumulated. Instead we create separate accumulation instructions (using
  // the same gradient accumulation buffer) as below:
  // bps2 = (...) bwd_pipeline_stage_2(...) stage_id = 2
  // bps2_grad = gte(bps2)
  // ...
  // bps0 = (...) bwd_pipeline_stage_0(...) stage_id = 0
  // bps0_grad = gte(bps0)
  //
  // gradient_buffer = gradient-accumulator-creator()
  // gradient2 = gradient-accumulation-add(gradient_buffer, bps2_grad)
  // gradient0 = gradient-accumulation-add(gradient_buffer, bps0_grad)
  // accumulated_sink = gradient-accumulator-sink(gradient2, gradient0)

  // Stores all the individual gradients which are accumulated to form a
  // single gradient from different pipeline stages. Note that they are stored
  // in the descending order because the last backward stage is executed first
  // and that stage needs to scale the gradient accumulation buffer.
  using GradientSources =
      std::map<int64, std::vector<HloInstruction*>, std::greater<int64>>;

  struct GradientSourcesInfo {
    GradientSources gradient_sources;
    bool is_running_mean;
  };

  // Find all the gradient accumulation add operations which have gradients
  // originating from different pipeline stages.
  HloInstructionMap<GradientSourcesInfo> gradient_adds_to_break_up;
  for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
    if (!IsPoplarInstruction(PoplarOp::GradientAccumulatorAddWithScale, inst)) {
      continue;
    }

    VLOG(3) << "Trying to break up gradient accumulation for "
            << inst->ToString();

    // Traverse through all the adds and get all the individual gradients.
    GradientSourcesInfo info;

    // Check whether the accumulator is scaled. If it is, check whether it is
    // running mean pattern which can be transformed to generate the same
    // result, otherwise we cannot proceed.
    HloInstruction* grad_accum_scale = inst->mutable_operand(2);
    if (Match(grad_accum_scale, m::ConstantScalar(1)) ||
        Match(grad_accum_scale, m::Convert(m::ConstantScalar(1)))) {
      // If the scale is 1, then we can always break up.
      info.is_running_mean = false;
    } else if (pipeline_fixer_util::IsRunningMeanAccumulatorScale(
                   grad_accum_scale)) {
      info.is_running_mean = true;
    } else {
      VLOG(3) << "Cannot break up accumulation as the scale of the gradient "
                 "accumulator is not 1.0 or running mean.";
      continue;
    }

    std::queue<HloInstruction*> to_visit;
    absl::flat_hash_set<const HloInstruction*> visited;
    to_visit.push(inst->mutable_operand(1));

    bool valid_to_break_up = true;
    while (to_visit.size()) {
      HloInstruction* gradient = to_visit.front();
      to_visit.pop();

      if (visited.contains(gradient)) {
        continue;
      }

      visited.insert(gradient);
      if (gradient->opcode() == HloOpcode::kAdd) {
        to_visit.push(gradient->mutable_operand(0));
        to_visit.push(gradient->mutable_operand(1));
      } else {
        auto value_set = analysis->GetValueSet(gradient);
        // Make sure the value set contains only one backward pipeline stage.
        absl::flat_hash_set<HloInstruction*> pipeline_stages;
        for (auto value : value_set.values()) {
          HloInstruction* defining_instruction = value->defining_instruction();
          if (IsPipelineStageBackward(defining_instruction)) {
            pipeline_stages.insert(defining_instruction);
          }
        }

        if (pipeline_stages.size() != 1) {
          VLOG(3) << "Cannot break up accumulation as " << gradient->ToString()
                  << " has " << pipeline_stages.size()
                  << " pipeline stage sources.";
          valid_to_break_up = false;
          break;
        }

        if (info.is_running_mean &&
            !pipeline_fixer_util::IsRunningMeanGradient(gradient)) {
          valid_to_break_up = false;
          break;
        }

        HloInstruction* pipeline_stage = *std::begin(pipeline_stages);
        TF_ASSIGN_OR_RETURN(auto stage_id,
                            analysis->GetStageID(pipeline_stage));

        info.gradient_sources[stage_id.id].push_back(gradient);
      }
    }

    if (valid_to_break_up && info.gradient_sources.size() > 1) {
      gradient_adds_to_break_up[inst] = info;
    } else {
      VLOG(3) << "Cannot breakup gradients for " << inst->ToString();
    }
  }

  if (gradient_adds_to_break_up.empty()) {
    return false;
  }

  // Create the addition trees for gradients spanning from the same pipeline
  // stage and combine.
  for (auto& gradient_pair : gradient_adds_to_break_up) {
    HloInstruction* accumulation_add = gradient_pair.first;
    HloInstruction* accumulation_buffer = accumulation_add->mutable_operand(0);
    HloInstruction* accumulation_scale = accumulation_add->mutable_operand(2);

    // Find the GradientAccumulatorSink instruction.
    if (accumulation_add->user_count() != 1) {
      return InternalErrorStrCat(
          "Expected the gradient accumulation add to have a single user, but "
          "has ",
          accumulation_add->user_count(), " users.");
    }

    HloInstruction* accumulation_sink = accumulation_add->users()[0];
    if (!IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(
            accumulation_sink)) {
      return InternalErrorStrCat(
          "Expected the gradient accumulation buffer to be used by a "
          "gradient accumulation sink, but it is used by ",
          accumulation_sink->ToString(), " instead.");
    }

    const GradientSourcesInfo& info = gradient_pair.second;
    VLOG(3) << "Breaking up " << accumulation_add->ToString();

    std::vector<HloInstruction*> new_accumulation_adds;
    const int64 first_stage_id = std::begin(info.gradient_sources)->first;

    for (auto& stage_pair : info.gradient_sources) {
      const int64 stage_id = stage_pair.first;
      std::vector<HloInstruction*> gradients = stage_pair.second;
      // Reverse the order of the gradients so that they are combined in the
      // same order as before to prevent changes to the liveness of the graph.
      absl::c_reverse(gradients);
      HloInstruction* gradient_tail = nullptr;

      // When using running mean, the gradients might need to be rescaled
      // based on the schedule.
      TF_ASSIGN_OR_RETURN(bool rescale_gradients,
                          pipeline_fixer_util::GradientsNeedRescaling(
                              schedule_, first_stage_id, stage_id));
      HloInstruction* running_mean_scale = nullptr;

      for (HloInstruction* gradient : gradients) {
        // Rescale gradients for running mean.
        if (info.is_running_mean && rescale_gradients) {
          if (!running_mean_scale) {
            TF_ASSIGN_OR_RETURN(
                running_mean_scale,
                pipeline_fixer_util::GetGradientScale(
                    accumulation_count, first_stage_id, stage_id));
          }
          TF_ASSIGN_OR_RETURN(gradient, pipeline_fixer_util::RescaleGradient(
                                            gradient, running_mean_scale));
        }

        if (gradient_tail) {
          TF_ASSIGN_OR_RETURN(
              gradient_tail,
              MakeBinaryHlo(HloOpcode::kAdd, gradient_tail, gradient));
        } else {
          gradient_tail = gradient;
        }
      }

      // Note that the accumulator should only be scaled once.
      HloInstruction* scale = accumulation_scale;
      if (new_accumulation_adds.size()) {
        scale = comp->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::One(scale->shape().element_type())));
      }

      // Create a new add.
      HloInstruction* new_accumulation_add =
          comp->AddInstruction(CreateGradientAccumulatorAddWithScale(
              accumulation_buffer, gradient_tail, scale));
      new_accumulation_adds.push_back(new_accumulation_add);
      VLOG(3) << "Created partial add " << new_accumulation_add->ToString()
              << " for stage " << stage_id;
    }

    // Create a new sink.
    HloInstruction* new_accumulation_sink = comp->AddInstruction(
        CreateGradientAccumulatorSink(new_accumulation_adds));
    VLOG(3) << "Created a new sink " << new_accumulation_sink->ToString();
    TF_RETURN_IF_ERROR(
        comp->ReplaceInstruction(accumulation_sink, new_accumulation_sink));
  }

  return true;
}

// Lowers any outputs of the stage into the stage.
// Returns true if the stage has changed.
StatusOr<bool> PipelineFixer::LowerPipelineStagesOutputs() {
  bool changed = false;

  TF_ASSIGN_OR_RETURN(auto analysis,
                      PipelineDataflowAnalysis::GetAnalysis(stages_));
  for (HloInstruction* stage : GetOrderedStages()) {
    TF_ASSIGN_OR_RETURN(StageID stage_id, analysis->GetStageID(stage));
    // We can't just iterate over users as we might remove some of them during
    // lowering.
    HloInstructionSet stage_users(stage->users().begin(), stage->users().end());

    // The combined instruction clusters which we want to lower for this stage.
    std::vector<HloInstruction*> all_ordered_lowerings;

    // Find and combine the clusters which we want to lower. The combining is
    // done to reduce calls to PipelineDataflowAnalysis::GetAnalysis, which can
    // be expensive.
    while (!stage_users.empty()) {
      HloInstruction* stage_gte_user = *stage_users.begin();
      stage_users.erase(stage_gte_user);
      CHECK_EQ(stage_gte_user->opcode(), HloOpcode::kGetTupleElement);
      CHECK_EQ(stage_gte_user->user_count(), 1);

      HloInstruction* stage_user = stage_gte_user->users()[0];

      TF_ASSIGN_OR_RETURN(bool needs_lowering,
                          analysis->HasToBeLoweredIntoStage(stage, stage_user));
      if (!needs_lowering) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(
          std::vector<HloInstruction*> ordered_lowering,
          FindClusterToLower(stage, stage_id, stage_user, analysis.get()));
      // Nothing to lower.
      if (ordered_lowering.empty()) {
        continue;
      }
      VLOG(3) << "Lowering outputs for stage " << stage_id;
      changed = true;

      // Prevent use after free in the outer loop.
      absl::c_for_each(ordered_lowering, [&stage_users](HloInstruction* inst) {
        stage_users.erase(inst);
      });

      // Combine the clusters such that we preserve the post ordering.
      for (auto* inst : ordered_lowering) {
        const auto unique = absl::c_find(all_ordered_lowerings, inst) ==
                            all_ordered_lowerings.end();
        if (unique) {
          all_ordered_lowerings.push_back(inst);
        }
      }
    }
    // Lower the instructions into the computation.
    TF_ASSIGN_OR_RETURN(
        stage, AddInstructionsToPipelineStage(stage, all_ordered_lowerings));

    TF_RETURN_IF_ERROR(UpdateStage(stage_id, stage));
    TF_RETURN_IF_ERROR(RemovePipelineStageDeadUsers(stage, stage_users));
    // Recompute the analysis.
    TF_ASSIGN_OR_RETURN(analysis,
                        PipelineDataflowAnalysis::GetAnalysis(stages_));
    // The dataflow in pipelining requires all the tensors to be thread through
    // consecutive pipeline stages. Check if there are any outputs from the
    // previous stage which are used by other stages directly and thread them
    // through this stage.
    // Note that we currently assume that forward stages do not require
    // threading as the Python API does not allow for unthreaded inputs.
    // Note that we do not need to thread outputs which are used by the resource
    // update.
    if (stage_id.stage_type == StageType::kForward) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(StageID previous_stage_id,
                        analysis->GetPreviousStageID(stage));
    HloInstruction* previous_stage = GetStage(previous_stage_id);

    // We remember that a pipeline stage only outputs GTEs and that each
    // GTE has been de-duplicated.
    // Store the tuple index from each user gte.
    HloInstruction* pipeline_root = stage->parent()->root_instruction();
    std::map<int64, HloInstructionSet> output_users;
    for (HloInstruction* gte : previous_stage->users()) {
      CHECK_EQ(gte->opcode(), HloOpcode::kGetTupleElement);
      CHECK_EQ(gte->user_count(), 1);
      int64 tuple_index = gte->tuple_index();
      HloInstruction* gte_user = gte->users()[0];
      // Do not track usage if it is:
      // * the current stage - this has already been lowered,
      // * the root instruction,
      // * the ResourceUpdate.
      // * a gradient accumulation sink - it will be an input to the
      //   ResourceUpdate.
      if (!(gte_user == stage || gte_user == pipeline_root ||
            IsResourceUpdate(gte_user) ||
            IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(gte_user))) {
        output_users[tuple_index].insert(gte);
      }
    }
    if (output_users.empty()) {
      continue;
    }
    VLOG(3) << "Threading values through stage " << stage_id;
    changed = true;

    // For each GTE tuple index, get one of GTEs X, replace all the other GTEs
    // uses with it, and add X as a forced parameter. The lowering will then
    // thread the value through and create GTEs out of this stage for all other
    // pipeline stages (but it will not change uses in the resource update).
    HloInstructionSet forced_parameters;
    for (auto& tuple_index_users_pair : output_users) {
      HloInstructionSet users = tuple_index_users_pair.second;
      HloInstruction* forced_parameter = *users.begin();
      users.erase(forced_parameter);
      forced_parameters.insert(forced_parameter);
      for (HloInstruction* gte : users) {
        TF_RETURN_IF_ERROR(
            gte->parent()->ReplaceInstruction(gte, forced_parameter));
      }
    }

    // Verify we can lower this.
    TF_RETURN_IF_ERROR(analysis->VerifyPipelineStageOperands(
        stage, analysis->GetValueSet(previous_stage)));
    // Lower the instructions into the computation.
    TF_ASSIGN_OR_RETURN(stage, AddInstructionsToPipelineStage(
                                   stage, {}, {}, forced_parameters, false));

    TF_RETURN_IF_ERROR(UpdateStage(stage_id, stage));
    // Recompute the analysis.
    TF_ASSIGN_OR_RETURN(analysis,
                        PipelineDataflowAnalysis::GetAnalysis(stages_));
  }
  return changed;
}

// This function lowers inputs to the pipeline stage which are arguments
// to the stage, but can be lowered (for examples constants etc.).
// Returns true if the stage has changed.
StatusOr<bool> PipelineFixer::LowerPipelineStagesInputs() {
  bool changed = false;

  TF_ASSIGN_OR_RETURN(auto analysis,
                      PipelineDataflowAnalysis::GetAnalysis(stages_));
  for (HloInstruction* stage : GetOrderedStages()) {
    TF_ASSIGN_OR_RETURN(StageID stage_id, analysis->GetStageID(stage));

    // Store parameter number and instruction which needs lowering.
    // Stored in order by parameter number.
    std::map<int64, HloInstruction*> parameters_to_replace;

    // Check if any operands need lowering.
    for (int64 operand_idx = 0; operand_idx != stage->operand_count();
         ++operand_idx) {
      HloInstruction* operand = stage->mutable_operand(operand_idx);
      TF_ASSIGN_OR_RETURN(bool operand_needs_lowering,
                          analysis->HasToBeLoweredIntoStage(stage, operand));
      if (operand_needs_lowering) {
        parameters_to_replace[operand_idx] = operand;
      }
    }

    // Skip if nothing needs lowering.
    if (parameters_to_replace.empty()) {
      continue;
    }
    VLOG(3) << "Lowering inputs for stage " << stage_id;
    changed = true;
    // Build a cluster of instructions which are lowered and order them.
    // Start from the operands and build the cluster by going through operands
    // only.
    std::vector<HloInstruction*> ordered_lowering;
    std::vector<const HloValueSet*> value_sets;
    std::queue<HloInstruction*> to_visit;
    absl::flat_hash_set<HloInstruction*> visited;

    for (auto& pair : parameters_to_replace) {
      to_visit.push(pair.second);
    }

    while (!to_visit.empty()) {
      HloInstruction* inst = to_visit.front();
      to_visit.pop();

      if (visited.contains(inst)) {
        continue;
      }

      ordered_lowering.push_back(inst);
      value_sets.push_back(&analysis->GetValueSet(inst));

      // Add any operand which has to be lowered.
      for (HloInstruction* operand : inst->operands()) {
        if (!visited.contains(operand)) {
          TF_ASSIGN_OR_RETURN(
              bool needs_lowering,
              analysis->HasToBeLoweredIntoStage(stage, operand));
          if (needs_lowering) {
            to_visit.push(operand);
          }
        }
      }
    }
    HloValueSet value_set;
    value_set.AssignUnionOf(value_sets);

    // Verify we can lower this.
    TF_RETURN_IF_ERROR(analysis->VerifyPipelineStageOperands(stage, value_set));

    // Make sure the lowering is in post order.
    absl::c_reverse(ordered_lowering);
    // Lower the instructions into the computation.
    TF_ASSIGN_OR_RETURN(stage,
                        AddInstructionsToPipelineStage(stage, ordered_lowering,
                                                       parameters_to_replace));
    // Check that after lowering the parameters are now unused.
    TF_ASSIGN_OR_RETURN(absl::flat_hash_set<int64> unused_parameters,
                        GetUnusedParametersInCall(stage));
    bool lowered_all_params =
        parameters_to_replace.size() <= unused_parameters.size() &&
        absl::c_all_of(
            parameters_to_replace,
            [&unused_parameters](std::pair<int64, HloInstruction*> replaced) {
              return unused_parameters.contains(replaced.first);
            });
    if (!lowered_all_params) {
      return InternalErrorStrCat("Failed to lower inputs for PipelineStage ",
                                 stage->ToString());
    }
    TF_ASSIGN_OR_RETURN(stage,
                        RemoveParametersFromCall(stage, unused_parameters));
    TF_RETURN_IF_ERROR(UpdateStage(stage_id, stage));
    // Recompute the analysis.
    TF_ASSIGN_OR_RETURN(analysis,
                        PipelineDataflowAnalysis::GetAnalysis(stages_));
  }
  return changed;
}

// Lowers any usages of paramaters into stages.
// Returns true if the stage has changed.
StatusOr<bool> PipelineFixer::LowerParameterUsagesIntoStages() {
  bool changed = false;

  TF_ASSIGN_OR_RETURN(auto analysis,
                      PipelineDataflowAnalysis::GetAnalysis(stages_));
  std::vector<HloInstruction*> ordered_stages = GetOrderedStages();
  // Iterate in reverse order so that we lower any changes affecting parameters
  // into the backward stages first if possible.
  for (auto itr = ordered_stages.rbegin(); itr != ordered_stages.rend();
       ++itr) {
    HloInstruction* stage = *itr;
    TF_ASSIGN_OR_RETURN(StageID stage_id, analysis->GetStageID(stage));
    // Get the corresponding forward stage.
    HloInstruction* fwd_stage = stages_.forward[stage_id.id];
    // Go through the operands to the forward stage.
    HloInstructionSet params_users;
    for (HloInstruction* operand : fwd_stage->operands()) {
      if (operand->opcode() == HloOpcode::kParameter) {
        // For parameters, we add all their non-pipeline stage users as
        // potential things we need to lower into the backward stage.
        absl::c_copy_if(operand->users(),
                        std::inserter(params_users, std::begin(params_users)),
                        [](const HloInstruction* inst) {
                          return !IsPipelineStageOrBackwardOp(inst);
                        });
      }
    }
    while (!params_users.empty()) {
      HloInstruction* param_user = *params_users.begin();
      params_users.erase(param_user);
      TF_ASSIGN_OR_RETURN(bool needs_lowering,
                          analysis->HasToBeLoweredIntoStage(stage, param_user));
      if (!needs_lowering) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(
          std::vector<HloInstruction*> ordered_lowering,
          FindClusterToLower(stage, stage_id, param_user, analysis.get()));
      // Nothing to lower.
      if (ordered_lowering.empty()) {
        continue;
      }
      VLOG(3) << "Lowering parameters for stage " << stage_id;
      changed = true;
      // Prevent use after free in the outer loop.
      absl::c_for_each(ordered_lowering, [&params_users](HloInstruction* inst) {
        params_users.erase(inst);
      });

      // Lower the instructions into the computation.
      TF_ASSIGN_OR_RETURN(
          stage, AddInstructionsToPipelineStage(stage, ordered_lowering));

      TF_RETURN_IF_ERROR(UpdateStage(stage_id, stage));
      TF_RETURN_IF_ERROR(RemovePipelineStageDeadUsers(stage, params_users));
      // Recompute the analysis.
      TF_ASSIGN_OR_RETURN(analysis,
                          PipelineDataflowAnalysis::GetAnalysis(stages_));
    }
  }
  return changed;
}

StatusOr<bool> PipelineFixer::LowerOpsIntoPipelineStages(
    HloInstruction* accumulation_count) {
  // Breakup elementwise operations where the inputs originate from different
  // stages.
  TF_ASSIGN_OR_RETURN(bool breakup_elementwise, BreakUpElementwiseOperations());
  // Breakup gradient accumulation operations where the inputs originate from
  // different stages.
  TF_ASSIGN_OR_RETURN(
      bool breakup_gradients,
      BreakUpGradientAccumulationOperations(accumulation_count));
  // Lower any outputs from stages into stages if possible.
  TF_ASSIGN_OR_RETURN(bool lowered_outputs, LowerPipelineStagesOutputs());
  // Lower any inputs into stages if possible.
  TF_ASSIGN_OR_RETURN(bool lowered_inputs, LowerPipelineStagesInputs());
  // Lower any usages of parameters into stages if possible.
  TF_ASSIGN_OR_RETURN(bool lowered_params_uses,
                      LowerParameterUsagesIntoStages());

  return breakup_elementwise || breakup_gradients || lowered_inputs ||
         lowered_outputs || lowered_params_uses;
}

Status PipelineFixer::RemovePipelineWrapper(HloComputation* pipeline_comp) {
  // We expect the pipeline computation to have a call to a wrapped computation
  // of all the pipeline stages. Find that call.
  absl::InlinedVector<HloInstruction*, 1> inner_calls;
  absl::c_copy_if(
      pipeline_comp->instructions(), std::back_inserter(inner_calls),
      [&](HloInstruction* inst) { return inst->opcode() == HloOpcode::kCall; });
  if (inner_calls.size() != 1) {
    return FailedPrecondition(
        "Expected a single wrapper call inside the Pipeline, but got %d.",
        inner_calls.size());
  }
  HloInstruction* call = inner_calls[0];
  HloComputation* comp_to_inline = call->to_apply();
  TF_RETURN_IF_ERROR(InlineComputation(call, comp_to_inline).status());
  return Status::OK();
}

StatusOr<bool> PipelineFixer::FixConstantGradients(
    int64 batch_serialization_iterations, const HloInstruction* pipeline_inst) {
  if (!stages_.resource_update) {
    return false;
  }
  HloInstruction* resource_update = *stages_.resource_update;
  HloComputation* pipeline_comp = resource_update->parent();

  HloInstruction* gradient_accumulation_inst =
      pipeline_comp->parameter_instruction(
          GetAccumulationCountOperandIndex(pipeline_inst));

  TF_ASSIGN_OR_RETURN(auto analysis,
                      PipelineDataflowAnalysis::GetAnalysis(stages_));
  bool changed = false;

  // Go through all the gradient accumulator sinks.
  for (int64 op_idx = 0; op_idx != resource_update->operand_count(); ++op_idx) {
    HloInstruction* operand = resource_update->mutable_operand(op_idx);
    if (!IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(operand)) {
      continue;
    }
    // We expect the sink to only be an input once to the resource update.
    CHECK_EQ(resource_update->OperandIndices(operand).size(), 1);
    HloGradientAccumulatorSink* sink =
        Cast<HloGradientAccumulatorSink>(operand);
    // If the sink has multiple operands, then the gradients cannot be constant
    // as they come from multiple pipeline stages.
    if (sink->operand_count() != 1) {
      continue;
    }
    HloInstruction* sink_input = operand->mutable_operand(0);
    if (!IsPoplarInstruction(PoplarOp::GradientAccumulatorAddWithScale)(
            sink_input)) {
      continue;
    }
    HloInstruction* lhs = sink_input->mutable_operand(0);
    if (!IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(lhs)) {
      return FailedPrecondition(
          "Expected the first input to the GradientAccumulatorAdd to be a "
          "GradientAccumulatorCreate, but is %s.",
          lhs->ToString().c_str());
    }
    HloInstruction* rhs = sink_input->mutable_operand(1);
    VLOG(3) << "Replacing an accumulated constant gradient with a constant.";

    // Create the constant for the gradient accumulation.
    Literal literal(ShapeUtil::MakeShape(S32, {}));
    literal.PopulateWithValue(static_cast<int>(batch_serialization_iterations));
    HloInstruction* const_multiplier = pipeline_comp->AddInstruction(
        HloInstruction::CreateConstant(std::move(literal)));

    // Multiply by the number of batches accumulating over
    HloInstruction* accum_const_multiplier =
        pipeline_comp->AddInstruction(HloInstruction::CreateBinary(
            ShapeUtil::MakeShape(S32, {}), HloOpcode::kMultiply,
            const_multiplier, gradient_accumulation_inst));

    const_multiplier =
        pipeline_comp->AddInstruction(HloInstruction::CreateBroadcastSequence(
            ShapeUtil::MakeShape(S32, operand->shape().dimensions()),
            accum_const_multiplier, [&](std::unique_ptr<HloInstruction> x) {
              return pipeline_comp->AddInstruction(std::move(x));
            }));

    // Convert it to the right type.
    HloInstruction* cast_multiplier = pipeline_comp->AddInstruction(
        HloInstruction::CreateConvert(operand->shape(), const_multiplier));
    // Multiply the constant gradient by the accumulation factor.
    HloInstruction* multiplied_gradient =
        pipeline_comp->AddInstruction(HloInstruction::CreateBinary(
            operand->shape(), HloOpcode::kMultiply, cast_multiplier, rhs));
    TF_RETURN_IF_ERROR(
        pipeline_comp->ReplaceInstruction(operand, multiplied_gradient));
    // Explicitly remove the gradient buffer allocator as it is stateful.
    TF_RETURN_IF_ERROR(pipeline_comp->ForceRemoveInstruction(lhs));
    changed = true;
  }

  return changed;
}

static StatusOr<std::vector<HloInstruction*>> CreateLoweringOrder(
    HloInstruction* operand, PipelineDataflowAnalysis* analysis,
    const HloInstruction* accumulation_count) {
  // We currently only expect constant gradients to be stageless
  // (because they do not depend on any input, we cannot associate them
  // with a backward stage).
  // Find the cluster of instructions from the operand instruction.
  std::vector<HloInstruction*> ordered_lowering;
  std::vector<const HloValueSet*> value_sets;
  std::queue<HloInstruction*> to_visit;
  absl::flat_hash_set<HloInstruction*> visited;

  to_visit.push(operand);
  while (!to_visit.empty()) {
    HloInstruction* inst = to_visit.front();
    to_visit.pop();

    if (visited.contains(inst)) {
      continue;
    }

    ordered_lowering.push_back(inst);
    value_sets.push_back(&analysis->GetValueSet(inst));

    // Add any operand which has to be lowered.
    for (HloInstruction* operand : inst->operands()) {
      if (!visited.contains(operand)) {
        TF_ASSIGN_OR_RETURN(bool needs_lowering,
                            analysis->HasToBeLowered(operand));
        if (needs_lowering) {
          to_visit.push(operand);
        }
      }
    }
  }

  // We only expect constant gradients to be lowered, therefore we expect no
  // producers in the cluster we are lowering.
  HloValueSet value_set;
  value_set.AssignUnionOf(value_sets);
  if (value_set.values().size()) {
    // If only buffer is gradient accumulation count this is also fine
    if (!(value_set.values().size() == 1 &&
          value_set.GetUniqueValue().defining_instruction()->Identical(
              *accumulation_count))) {
      return FailedPrecondition(
          "Detected input to the ResourceUpdate which should have been "
          "lowered into a Pipeline(Backward)Stage.");
    }
  }
  return ordered_lowering;
}

StatusOr<bool> PipelineFixer::LowerResourceUpdateInputs(
    HloInstruction* accumulation_count) {
  if (!stages_.resource_update) {
    return false;
  }
  HloInstruction* resource_update = *stages_.resource_update;
  HloComputation* resource_update_comp = resource_update->to_apply();

  TF_ASSIGN_OR_RETURN(auto analysis,
                      PipelineDataflowAnalysis::GetAnalysis(stages_));
  absl::flat_hash_set<int64> unused_op_indices;
  // Go through all the operands and lower the ones which need lowering.
  for (int64 op_idx = 0; op_idx != resource_update->operand_count(); ++op_idx) {
    HloInstruction* operand = resource_update->mutable_operand(op_idx);
    TF_ASSIGN_OR_RETURN(bool lower, analysis->HasToBeLowered(operand));
    if (!lower) {
      continue;
    }
    VLOG(3) << "Lowering the operand " << op_idx << " for the ResourceUpdate.";
    TF_ASSIGN_OR_RETURN(
        auto ordered_lowering,
        CreateLoweringOrder(operand, analysis.get(), accumulation_count));

    // Do the lowering of instructions into the resource update.
    absl::c_reverse(ordered_lowering);
    absl::flat_hash_map<HloInstruction*, HloInstruction*>
        old_to_new_computation;
    old_to_new_computation[accumulation_count] =
        GetResourceUpdateNumMiniBatchesInstruction(resource_update);
    for (HloInstruction* old_inst : ordered_lowering) {
      std::vector<HloInstruction*> new_operands(old_inst->operand_count());
      absl::c_transform(old_inst->operands(), new_operands.begin(),
                        [&old_to_new_computation](HloInstruction* old_operand) {
                          return old_to_new_computation.at(old_operand);
                        });
      HloInstruction* new_inst = resource_update_comp->AddInstruction(
          old_inst->CloneWithNewOperands(old_inst->shape(), new_operands));
      old_inst->SetupDerivedInstruction(new_inst);
      old_to_new_computation[old_inst] = new_inst;
    }

    // Replace all uses of the corresponding parameters with the operand
    // instruction.
    HloInstruction* param = resource_update_comp->parameter_instruction(op_idx);
    TF_RETURN_IF_ERROR(
        param->ReplaceAllUsesWith(old_to_new_computation.at(operand)));
    unused_op_indices.insert(op_idx);
  }

  // Remove unused operands.
  TF_RETURN_IF_ERROR(
      RemoveParametersFromCall(resource_update, unused_op_indices).status());

  return unused_op_indices.size();
}

namespace {
HloComputation* CreateEmptyComputation(const std::string& stage_name,
                                       HloModule* module) {
  auto builder = HloComputation::Builder(stage_name + "_func");
  builder.AddInstruction(CreateStatefulNoop());
  auto* root = builder.AddInstruction(HloInstruction::CreateTuple({}));
  return module->AddEmbeddedComputation(builder.Build(root));
}
}  // namespace

Status PipelineFixer::InsertDummyBackwardStages(HloComputation* pipeline_comp) {
  TF_ASSIGN_OR_RETURN(PipelineStages stages,
                      GetPipelineStages(pipeline_comp, false));
  if (stages.backward.empty()) {
    return Status::OK();
  }

  // Find the missing backward stages.
  std::list<int64> missing;
  int64 back_idx = -1;
  size_t stage_id = GetNextStageID(back_idx, stages.backward);
  for (size_t i = 0; i != stages.forward.size(); ++i) {
    if (stage_id == i) {
      stage_id = GetNextStageID(back_idx, stages.backward);
    } else {
      missing.push_front(i);
    }
  }

  // Create dummy stages for the missing indices.
  for (int64 missing_id : missing) {
    HloInstruction* input = stages.forward[missing_id];
    std::string name = input->to_apply()->name() + "_grad";
    TF_RETURN_IF_ERROR(
        CreatePipelineStage(
            pipeline_comp, {},
            CreateEmptyComputation(name, pipeline_comp->parent()),
            PoplarBackendConfig::CallConfig::PipelineStageBackward, missing_id,
            name)
            .status());
  }

  return Status::OK();
}

Status PipelineFixer::FixPipeline(HloInstruction* pipeline_op) {
  HloComputation* pipeline_comp = pipeline_op->to_apply();

  TF_RETURN_IF_ERROR(RemovePipelineWrapper(pipeline_comp));

  TF_RETURN_IF_ERROR(InsertDummyBackwardStages(pipeline_comp));

  TF_ASSIGN_OR_RETURN(stages_, GetPipelineStages(pipeline_comp));

  TF_ASSIGN_OR_RETURN(schedule_, GetPipelineSchedule(pipeline_op));

  if (stages_.recomputation.size()) {
    return FailedPrecondition(
        "PipelineStageRecomputation stages are not allowed in the"
        " PipelineFixer pass.");
  }

  // Go through the stages and insert stateful no-ops to make sure DCE does not
  // remove stages. This is usually caused by the constant propagation in TF2XLA
  // layer.
  // We do not want to remove the stages because later stages of this pass will
  // lower ops/thread ops through stages.
  TF_RETURN_IF_ERROR(InsertStatefulNoopsIntoStages());
  // Clean up the pipelines.
  TupleSimplifier ts(true);
  TF_RETURN_IF_ERROR(ts.Run(pipeline_op->GetModule()).status());
  HloDCE dce;
  TF_RETURN_IF_ERROR(dce.Run(pipeline_op->GetModule()).status());

  // Try and evaluate the accumulation count to be a constant.
  int64_t accumulation_count_idx =
      GetAccumulationCountOperandIndex(pipeline_op);
  HloInstruction* accumulation_count = nullptr;
  auto accumulation_count_value =
      GetConstantValue<int32>(pipeline_op->operand(accumulation_count_idx));
  if (accumulation_count_value) {
    accumulation_count =
        MakeR0ConstantHlo<int32>(pipeline_comp, *accumulation_count_value);
  } else {
    accumulation_count =
        pipeline_comp->parameter_instruction(accumulation_count_idx);
  }

  // Insert GTE edges such that every user of a stage is a GTE.
  TF_RETURN_IF_ERROR(InsertGTEEdges(stages_).status());
  // Duplicate edges - this makes analysis easier.
  TF_RETURN_IF_ERROR(DuplicateGTEEdges(stages_).status());
  // Uniquify computations called by stages.
  TF_RETURN_IF_ERROR(UniquifyPipelineStageCallsites(stages_).status());
  // Make sure that the root of each stage is a tuple.
  TF_RETURN_IF_ERROR(FixRootInstructions(stages_));
  // Verify we can actually try and lower this Pipeline.
  TF_RETURN_IF_ERROR(VerifyPipelineStagesBeforeFixing(stages_));
  // Run the lowering on pipeline stages.
  TF_RETURN_IF_ERROR(LowerOpsIntoPipelineStages(accumulation_count).status());
  // Run the fixing for constant gradients.
  TF_RETURN_IF_ERROR(
      FixConstantGradients(GetPipelineBatchSerializationIterations(pipeline_op),
                           pipeline_op)
          .status());
  // Run the lowering on the resource update.
  TF_RETURN_IF_ERROR(LowerResourceUpdateInputs(accumulation_count).status());
  // Tidy again.
  TF_RETURN_IF_ERROR(dce.Run(pipeline_op->GetModule()).status());
  // Verify the pipeline is now ok to be lowered.
  TF_RETURN_IF_ERROR(VerifyPipelineAfterFixing(pipeline_op));

  return Status::OK();
}

StatusOr<bool> PipelineFixer::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to fix.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  VLOG(2) << "Before fixing the Pipeline stages.";
  XLA_VLOG_LINES(2, module->ToString());

  TF_RETURN_IF_ERROR(FixPipeline(pipeline_ops[0]));

  VLOG(2) << "After fixing the Pipeline stages.";
  XLA_VLOG_LINES(2, module->ToString());
  return true;
}

StatusOr<bool> PipelineFixer::TestFixConstantGradients(
    HloInstruction* pipeline_op, HloComputation* pipeline_comp) {
  TF_ASSIGN_OR_RETURN(stages_, GetPipelineStages(pipeline_comp));
  // Run the fixing for constant gradients.
  return FixConstantGradients(
      GetPipelineBatchSerializationIterations(pipeline_op), pipeline_op);
}

}  // namespace poplarplugin
}  // namespace xla
