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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_verifier.h"

#include <memory>

#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {
// Function to check that inst has single sharding and that it matches the
// pipeline sharding.
Status HasCompatiblePipelineSharding(const HloSharding& expected,
                                     const HloInstruction* inst) {
  if (!inst->has_sharding()) {
    return InternalErrorStrCat("Expected for ", inst->ToString(),
                               " to have sharding information");
  }
  auto optional_sharding = inst->sharding().ExtractSingleSharding();
  if (!optional_sharding) {
    return InternalErrorStrCat("Expected the sharding for ", inst->ToString(),
                               " to be singular.");
  }
  if (expected != *optional_sharding) {
    return InternalErrorStrCat("Expected all the sharding in ",
                               inst->ToString(), " to be compatible with ",
                               expected.ToString(), ".");
  } else {
    return Status::OK();
  }
}
}  // namespace

PipelineVerifier::PipelineVerifier(bool allow_recomputation)
    : allow_recomputation_(allow_recomputation) {}

Status PipelineVerifier::VerifyGradientAccumulation(HloModule* module,
                                                    CallGraph* call_graph) {
  for (const auto& computation : module->computations()) {
    if (IsPopOpsFusion(computation)) {
      continue;
    }

    for (const HloInstruction* inst : computation->instructions()) {
      if (!IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(inst)) {
        continue;
      }

      if (inst->user_count() == 0) {
        return InternalErrorStrCat("Expected the gradient accumulation buffer ",
                                   inst->ToString(),
                                   " to have at least one user.");
      }

      // We expect all the backward stages to be lowered inplace in order to
      // make sure there is only one gradient accumulation buffer if it is used
      // by multiple stages.
      const bool expect_lowered_inplace = inst->user_count() > 1;

      // We expect the gradient accumulation creators to only be used by
      // backward pipeline stages residing on the same shard.
      for (HloInstruction* user : inst->users()) {
        const auto indices = user->OperandIndices(inst);
        if (indices.size() != 1) {
          return InternalErrorStrCat(
              "Expected the gradient accumulation buffer to only appear as an "
              "opperand once, but it is used ",
              indices.size(), " times.");
        }
        if (!IsPipelineStageBackward(user)) {
          return InternalErrorStrCat(
              "Expected the gradient accumulation buffer to only be used by "
              "backward pipeline stages, but detected ",
              inst->ToString(), " as a user.");
        }
        if (*user->sharding_unique_device() !=
            *inst->sharding_unique_device()) {
          return InternalError(
              "Expected the gradient accumulation buffer and the backward "
              "pipeline stage to have compatible sharding.");
        }
        if (expect_lowered_inplace && !IsLoweredInplace(user)) {
          return InternalErrorStrCat("Expected the pipeline backward stage ",
                                     user->ToString(),
                                     " to have been lowered inplace.");
        }
        HloComputation* pipeline_stage_comp = user->to_apply();

        // Inside of the backward pipeline stage, we expect the gradient
        // accumulator to be used serially, with the final use in the root
        // tuple. We expect all the uses to be inplace on the buffer.
        int64 output_index = indices[0];
        HloInstruction* inner_user =
            pipeline_stage_comp->parameter_instruction(output_index);
        do {
          if (inner_user->user_count() != 1) {
            return InternalErrorStrCat(
                "Expected the gradient accumulation buffer to be used "
                "serially, but detected ",
                inner_user->user_count(), " users.");
          }
          HloInstruction* next_user = inner_user->users()[0];
          const auto next_user_indices = next_user->OperandIndices(inner_user);

          if (next_user_indices.size() != 1) {
            return InternalErrorStrCat(
                "Expected the gradient accumulation buffer to only appear as "
                "an opperand once, but it is used ",
                next_user_indices.size(), " times.");
          }
          auto inplace_modifier = GetInplaceModifier(inner_user);
          if (!inplace_modifier) {
            return InternalError(
                "Expected the gradient accumulation buffer to be used "
                "inplace.");
          }
          if (*inplace_modifier != next_user) {
            return InternalErrorStrCat(
                "Expected the gradient accumulation inplace user to be ",
                next_user->ToString(), " but it was ",
                (*inplace_modifier)->ToString(), ".");
          }
          inner_user = next_user;
          output_index = next_user_indices[0];
        } while (pipeline_stage_comp->root_instruction() != inner_user);
        CHECK_EQ(inner_user->opcode(), HloOpcode::kTuple);

        // We expect the output at the gradient accumulation buffer location
        // to be only used (via a GTE) by the gradient accumulation sink
        // instruction.
        absl::flat_hash_set<HloInstruction*> gtes;
        for (HloInstruction* stage_user : user->users()) {
          CHECK_EQ(stage_user->opcode(), HloOpcode::kGetTupleElement);
          if (stage_user->tuple_index() == output_index) {
            gtes.insert(stage_user);
          }
        }
        if (gtes.size() != 1) {
          return InternalErrorStrCat(
              "Expected the gradient accumulation buffer to only have a "
              "single user, but it has ",
              gtes.size(), " users.");
        }

        // We expect the sink instruction to only be used by the resource
        // update function.
        HloInstruction* gte = *std::begin(gtes);
        if (gte->user_count() != 1) {
          return InternalErrorStrCat(
              "Expected the gradient accumulation buffer to only have a single "
              "user, but it has ",
              gte->user_count(), " users.");
        }

        if (!IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(
                gte->users()[0])) {
          return InternalErrorStrCat(
              "Expected the gradient accumulation buffer to be used by a "
              "gradient accumulation sink, but it is used by ",
              gte->users()[0]->ToString(), " instead.");
        }
      }
    }
  }
  return Status::OK();
}

Status PipelineVerifier::VerifyPipeline(HloInstruction* pipeline_op,
                                        CallGraph* call_graph) {
  HloComputation* pipeline_computation = pipeline_op->to_apply();
  TF_ASSIGN_OR_RETURN(PipelineStages stages,
                      GetPipelineStages(pipeline_computation));
  TF_ASSIGN_OR_RETURN(
      auto analysis, PipelineDataflowAnalysis::GetAnalysis(
                         stages, true, true, true, allow_recomputation_, true));

  // Make sure all the instructions in the pipeline_computation do not require
  // lowering. Note that the lowering checks validity of the usage.
  for (HloInstruction* inst : pipeline_computation->instructions()) {
    TF_ASSIGN_OR_RETURN(bool needs_lowering, analysis->HasToBeLowered(inst));
    if (needs_lowering) {
      return InternalErrorStrCat(
          "Detected instruction ", inst->ToString(),
          " which should have been lowered into a PipelineStage.");
    }
    // Verify sharding for the stages.
    if (IsAnyPipelineStageOp(inst)) {
      if (!inst->has_sharding()) {
        return InternalErrorStrCat("Expected for ", inst->ToString(),
                                   " to have sharding information");
      }
      const HloSharding expected_sharding =
          *inst->sharding().ExtractSingleSharding();
      // Check that the sharding of all operands matches the output sharding.
      for (HloInstruction* operand : inst->operands()) {
        TF_RETURN_IF_ERROR(
            HasCompatiblePipelineSharding(expected_sharding, operand));
      }
      // Check that all the users have the correct sharding too.
      for (HloInstruction* user : inst->users()) {
        TF_RETURN_IF_ERROR(
            HasCompatiblePipelineSharding(expected_sharding, user));
      }
      // Check that instructions in all visited subcomputations have correct
      // sharding too.
      // Get all the computations called.
      TF_ASSIGN_OR_RETURN(absl::flat_hash_set<HloComputation*> called_in_stage,
                          GetAllComputationsCalledBy(inst, call_graph));
      for (HloComputation* comp : called_in_stage) {
        for (HloInstruction* inst : comp->instructions()) {
          TF_RETURN_IF_ERROR(
              HasCompatiblePipelineSharding(expected_sharding, inst));
        }
      }
    } else if (IsPipelineResourceUpdate(inst)) {
      // Issue a warning if there are any gradient accumulation operations in
      // the resource update.
      TF_ASSIGN_OR_RETURN(absl::flat_hash_set<HloComputation*> called_in_stage,
                          GetAllComputationsCalledBy(inst, call_graph));
      for (HloComputation* comp : called_in_stage) {
        for (HloInstruction* inst : comp->instructions()) {
          if (IsPoplarInstruction(PoplarOp::StatefulGradientAccumulate)(inst) ||
              IsPoplarInstruction(
                  PoplarOp::StatefulGradientAccumulateAndAllReduce)(inst) ||
              IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(inst) ||
              IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(inst)) {
            LOG(INFO)
                << "Detected a gradient accumulation operation inside "
                   "the resource update part of the pipeline which might "
                   "lead to unintended behaviour. Please note that unless "
                   "continuous weight update has been enabled, the pipelining "
                   "operation automatically handles and generates gradient "
                   "accumulation operations.";
          }
        }
      }
    }
  }

  // We only check sharding if there are inputs/outputs to the pipeline.
  const bool check_sharding = pipeline_op->operand_count() ||
                              !ShapeUtil::IsEmptyTuple(pipeline_op->shape());
  if (check_sharding) {
    // Verify that the input/output have the same sharding.
    std::vector<HloSharding> input_sharding;
    for (const HloInstruction* operand : pipeline_op->operands()) {
      if (!operand->has_sharding()) {
        return InternalErrorStrCat("Expected for ", operand->ToString(),
                                   " to have sharding information");
      }
      const HloSharding& sharding = operand->sharding();
      if (sharding.IsTuple()) {
        absl::c_copy(sharding.tuple_elements(),
                     std::back_inserter(input_sharding));
      } else {
        input_sharding.push_back(sharding);
      }
    }

    if (!pipeline_op->has_sharding()) {
      return InternalErrorStrCat("Expected for ", pipeline_op->ToString(),
                                 " to have sharding information");
    }
    if (input_sharding != pipeline_op->sharding().tuple_elements()) {
      return InternalErrorStrCat(
          "Expected the sharding of inputs and outputs of pipeline ",
          pipeline_op->name(), " to match.");
    }
  }

  return Status::OK();
}

StatusOr<bool> PipelineVerifier::Run(HloModule* module) {
  // First verify the usage of PipelineStatefulGradientAccumulate.
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  TF_RETURN_IF_ERROR(VerifyGradientAccumulation(module, call_graph.get()));
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> pipeline_ops,
                      GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to verify.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  VLOG(2) << "Verifing the Pipelines.";
  XLA_VLOG_LINES(2, module->ToString());

  // This pass does not change anything.
  TF_RETURN_IF_ERROR(VerifyPipeline(pipeline_ops[0], call_graph.get()));
  return false;
}

}  // namespace poplarplugin
}  // namespace xla
