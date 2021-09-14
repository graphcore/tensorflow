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
  if (expected != *optional_sharding &&
      expected.GetUniqueDevice() != Devices::All) {
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
    } else if (IsResourceUpdate(inst)) {
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
  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
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
