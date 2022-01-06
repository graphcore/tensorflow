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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_gradient_accumulation_optimizer.h"

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"

namespace xla {
namespace poplarplugin {
namespace {
StatusOr<int64> GetSingleUseOperandIndex(const HloInstruction* inst,
                                         const HloInstruction* operand) {
  const auto indices = inst->OperandIndices(operand);
  if (indices.size() != 1) {
    return InternalErrorStrCat("Expected the instruction ", operand->ToString(),
                               " to be an operand of ", inst->ToString(),
                               " once, but it was an operand ", indices.size(),
                               " times.");
  }
  return indices[0];
}
}  // namespace

StatusOr<bool> PipelineGradientAccumulationOptimizer::OptimizePipeline(
    HloInstruction* pipeline_op) {
  HloComputation* pipeline_comp = pipeline_op->to_apply();

  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  TF_RETURN_IF_ERROR(FixRootInstructions(stages));
  TF_ASSIGN_OR_RETURN(const auto schedule, GetPipelineSchedule(pipeline_op));

  // There is nothing to optimize if there is no backward stages.
  if (stages.backward.empty()) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(std::vector<PipelinePath> paths,
                      FindPassthroughPipelinePaths(stages, schedule));
  // Only consider backward paths.
  std::vector<PipelinePath> backward_paths;
  absl::c_copy_if(paths, std::back_inserter(backward_paths),
                  [](const PipelinePath& path) -> bool {
                    return path.GetType() == PipelinePath::Type::kBackward;
                  });

  if (backward_paths.empty()) {
    return false;
  }

  for (PipelinePath& path : backward_paths) {
    // See how the begining of the path consumes the operation, if it's consumed
    // by the HloGradientAccumulatorAdd, then we can move the application of
    // this gradient into the producer stage.
    HloInstruction* consumer_stage = path.GetNewConsumerStage();
    CHECK(IsPipelineStageBackward(consumer_stage));
    HloComputation* consumer_comp = consumer_stage->to_apply();
    const int64 parameter_number = path.GetInputsPath()[0];
    HloInstruction* parameter_instruction =
        consumer_comp->parameter_instruction(parameter_number);

    // Only continue if the instruction has at most one user at any point
    // and it is used by HloGradientAccumulatorAdd on the rhs at the end.
    // Track the instructions and the operands.
    std::vector<HloInstruction*> gradient_path = {parameter_instruction};
    std::vector<int64> gradient_operand_path;

    bool look_at_users = true;
    while (look_at_users) {
      look_at_users = false;
      // Traverse any adds.
      std::vector<HloInstruction*> users = gradient_path.back()->users();
      if (users.size() == 1) {
        HloInstruction* user = users[0];
        const auto indices = user->OperandIndices(gradient_path.back());
        if (indices.size() == 1 && user->opcode() == HloOpcode::kAdd) {
          gradient_operand_path.push_back(indices[0]);
          gradient_path.push_back(user);
          look_at_users = true;
        }
      }
    }

    // If the end of the path does not have a single GradientAccumulatorAdd user
    // then this path cannot be used.
    HloInstruction* end_of_path = gradient_path.back();
    std::vector<HloInstruction*> end_of_path_users = end_of_path->users();
    if (!(end_of_path_users.size() == 1 &&
          IsPoplarInstruction(PoplarOp::GradientAccumulatorAddWithScale)(
              end_of_path_users[0]))) {
      continue;
    }
    auto const_opt = GetConstantValue<float>(end_of_path_users[0]->operand(2));
    CHECK(const_opt && *const_opt == 1.0f);

    HloInstruction* gradient_accumulator_add = end_of_path_users[0];

    // Find the GradientAccumulatorCreate instruction given the path.
    // It is expected to be the lhs of the GradientAccumulatorAdd, which is just
    // a parameter instruction.
    const HloInstruction* gradient_accumulator_add_lhs =
        gradient_accumulator_add->operand(0);
    if (gradient_accumulator_add_lhs->opcode() != HloOpcode::kParameter) {
      return InternalErrorStrCat(
          "Expected the LHS operand of the gradient accumulation add to be a "
          "parameter, but is ",
          gradient_accumulator_add_lhs->ToString(), ".");
    }
    // Get the parameter and make sure it is the gradient accumulation buffer
    // creation op in the outer scope.
    const int64 accumulator_parameter_number =
        gradient_accumulator_add_lhs->parameter_number();
    HloInstruction* gradient_accumulator_create =
        consumer_stage->mutable_operand(accumulator_parameter_number);
    if (!IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(
            gradient_accumulator_create)) {
      return InternalErrorStrCat(
          "Expected the input to the gradient accumulation add to be a "
          "gradient accumulation buffer, but is ",
          gradient_accumulator_create->ToString(), ".");
    }

    // Find the GradientAccumulatorSink instruction.
    if (gradient_accumulator_add->user_count() != 1) {
      return InternalErrorStrCat(
          "Expected the gradient accumulation add to have a single user, but "
          "has ",
          gradient_accumulator_add->user_count(), " users.");
    }
    HloInstruction* consumer_stage_output =
        gradient_accumulator_add->users()[0];
    if (consumer_stage_output->opcode() != HloOpcode::kTuple ||
        consumer_stage_output != consumer_comp->root_instruction()) {
      return FailedPrecondition(
          "Expected the gradient accumulation add have been consumed by a root "
          "tuple instruction.");
    }
    // Get the output index in the tuple.
    TF_ASSIGN_OR_RETURN(const int64 consumer_output_index,
                        GetSingleUseOperandIndex(consumer_stage_output,
                                                 gradient_accumulator_add));
    absl::flat_hash_set<HloInstruction*> gtes;
    for (HloInstruction* consumer_user : consumer_stage->users()) {
      CHECK_EQ(consumer_user->opcode(), HloOpcode::kGetTupleElement);
      if (consumer_user->tuple_index() == consumer_output_index) {
        gtes.insert(consumer_user);
      }
    }
    if (gtes.size() != 1) {
      return InternalErrorStrCat(
          "Expected the gradient accumulation buffer to only have a "
          "single user, but it has ",
          gtes.size(), " users.");
    }
    HloInstruction* consumer_output = *std::begin(gtes);
    if (consumer_output->user_count() != 1) {
      return InternalErrorStrCat(
          "Expected the gradient accumulation buffer to only have a single "
          "user, but it has ",
          consumer_output->user_count(), " users.");
    }
    HloInstruction* gradient_accumulation_sink = consumer_output->users()[0];
    if (!IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(
            gradient_accumulation_sink)) {
      return InternalErrorStrCat(
          "Expected the gradient accumulation buffer to be used by a "
          "gradient accumulation sink, but it is used by ",
          gradient_accumulation_sink->ToString(), " instead.");
    }
    TF_ASSIGN_OR_RETURN(
        const int64 consumer_sink_operand_index,
        GetSingleUseOperandIndex(gradient_accumulation_sink, consumer_output));

    VLOG(2) << "Parameter instruction " << parameter_instruction->ToString()
            << " is used by the gradient accumulator add "
            << gradient_accumulator_add->ToString()
            << " which uses the gradient accumulation buffer "
            << gradient_accumulator_create->ToString()
            << ". Shortcuting the data path to save memory.";

    // Get the parameter user and remove the parameter from the gradient
    // calculation in the customer stage by replacing it with zeros - this will
    // be optimized out.
    HloInstruction* parameter_user = parameter_instruction->users()[0];
    HloInstruction* zeros =
        BroadcastZeros(consumer_comp, parameter_user->shape().element_type(),
                       parameter_user->shape().dimensions());
    TF_RETURN_IF_ERROR(
        parameter_instruction->ReplaceUseWith(parameter_user, zeros));

    // Get where the stage is currently used - i.e. where it is being threaded
    // through.
    HloInstruction* old_user = path.GetOldConsumerStage();
    // Get the GTE instruction.
    HloInstruction* old_user_operand =
        old_user->mutable_operand(path.GetInputsPath().back());
    CHECK_EQ(old_user_operand->opcode(), HloOpcode::kGetTupleElement);

    // Get the producer stage where this gradient comes from.
    const int64 producer_output_index = old_user_operand->tuple_index();
    HloInstruction* producer_stage = old_user_operand->mutable_operand(0);
    CHECK(IsPipelineStageBackward(producer_stage));

    // Accumulate the gradients in the producer pipeline.
    auto producer_parameter_indices =
        producer_stage->OperandIndices(gradient_accumulator_create);
    switch (producer_parameter_indices.size()) {
      case 0: {
        // The gradient accumulation buffer is not used inside of the producer
        // pipeline stage.
        // Clone the GTE to make sure each GTE has a single user.
        HloInstruction* gte_clone =
            pipeline_comp->AddInstruction(old_user_operand->Clone());
        // Create a gradient add instruction.
        // TODO(T46014, T46015) : Add appropriate accumulator scale
        TF_ASSIGN_OR_RETURN(
            Literal literal,
            LiteralUtil::CreateR0(1.0f).Convert(
                gradient_accumulator_create->shape().element_type()));
        auto* one = pipeline_comp->AddInstruction(
            HloInstruction::CreateConstant(std::move(literal)));
        HloInstruction* accumulator_add =
            pipeline_comp->AddInstruction(CreateGradientAccumulatorAddWithScale(
                gradient_accumulator_create, gte_clone, one));

        // Create a new sink instruction such that it has the output of the
        // accumulator as an operand to merge the buffers into a single one.
        // Add the accumulator add as an operand to the sink instruction.
        auto new_operands = gradient_accumulation_sink->operands();
        new_operands.push_back(accumulator_add);

        // Clone the sink with new operands.
        HloInstruction* new_sink = pipeline_comp->AddInstruction(
            gradient_accumulation_sink->CloneWithNewOperands(
                gradient_accumulation_sink->shape(), new_operands));
        gradient_accumulation_sink->SetupDerivedInstruction(new_sink);

        // Replace the sink with a new one.
        TF_RETURN_IF_ERROR(pipeline_comp->ReplaceInstruction(
            gradient_accumulation_sink, new_sink));

        // Lower the accumulator add into the pipeline stage.
        TF_RETURN_IF_ERROR(AddInstructionsToPipelineStage(
                               producer_stage, {one, accumulator_add})
                               .status());
        break;
      }
      case 1: {
        // The gradient accumulation buffer is already used by the producer
        // pipeline stage.
        // Find the existing gradient accumulation add instruction inside of the
        // producer computation, and add the producer to it.
        const uint64 gradient_accumulation_parameter_index =
            producer_parameter_indices[0];
        HloComputation* producer_comp = producer_stage->to_apply();
        HloInstruction* producer_parameter =
            producer_comp->parameter_instruction(
                gradient_accumulation_parameter_index);
        if (producer_parameter->user_count() != 1) {
          return InternalErrorStrCat(
              "Expected the gradient accumulation creator to have a single "
              "user, but has ",
              producer_parameter->user_count(), " users.");
        }

        HloInstruction* producer_gradient_accumulator_add =
            producer_parameter->users()[0];
        if (!IsPoplarInstruction(PoplarOp::GradientAccumulatorAddWithScale)(
                producer_gradient_accumulator_add)) {
          return FailedPrecondition(
              "Expected the gradient accumulation creator to have been "
              "consumed by a gradient accumulation add.");
        }
        // Get the producer.
        HloInstruction* root = producer_comp->root_instruction();
        CHECK_EQ(root->opcode(), HloOpcode::kTuple);
        HloInstruction* producer = root->mutable_operand(producer_output_index);
        // Given the accumulation add, take it's current RHS, add the producer
        // to it, and then replace the new accumulated gradients as the operand
        // of the gradient accumulation add.
        HloInstruction* rhs =
            producer_gradient_accumulator_add->mutable_operand(1);
        TF_ASSIGN_OR_RETURN(HloInstruction * new_add,
                            MakeBinaryHlo(HloOpcode::kAdd, rhs, producer));
        TF_RETURN_IF_ERROR(
            producer_gradient_accumulator_add->ReplaceOperandWith(1, new_add));
        break;
      }
      default: {
        return FailedPrecondition(
            "Detected a gradient accumulation buffer which is used as multiple "
            "inputs.");
      }
    }
    return true;
  }
  return false;
}

StatusOr<bool> PipelineGradientAccumulationOptimizer::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to optimize.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  HloInstruction* pipeline_op = pipeline_ops[0];
  VLOG(2) << "Before PipelineGradientAccumulationOptimizer:";
  XLA_VLOG_LINES(2, module->ToString());
  TF_ASSIGN_OR_RETURN(bool changed, OptimizePipeline(pipeline_op));

  if (changed) {
    VLOG(2) << "After PipelineGradientAccumulationOptimizer:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the module.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
