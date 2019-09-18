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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {
StatusOr<bool> AddCopyIfParamterModifiedInplace(HloInstruction* call,
                                                int64 parameter_number) {
  HloComputation* comp = call->to_apply();
  HloInstruction* parameter = comp->parameter_instruction(parameter_number);
  if (IsOutputModifiedInplace(parameter)) {
    VLOG(1) << "Inserting a copy for stage " << call->ToString()
            << " parameter number " << parameter_number;
    // Insert a copy from the the parameter.
    HloInstruction* copy = comp->AddInstruction(HloInstruction::CreateUnary(
        parameter->shape(), HloOpcode::kCopy, parameter));
    parameter->SetupDerivedInstruction(copy);
    TF_RETURN_IF_ERROR(parameter->ReplaceAllUsesWith(copy));
    return true;
  }
  return false;
}

StatusOr<bool> InsertForwardStageCopies(PipelineStages& stages) {
  bool changed = false;
  for (HloInstruction* stage : stages.forward) {
    // Go through all the operands to the stage, and insert copies if necessary
    // to make sure no parameter is modified inplace.
    // We could alternatively mark the modifying instruction not inplace, but
    // that might result in higher memory usage (for example a tuple going into
    // a loop is now not inplace).
    // TODO(T10387)
    for (int64 op_idx = 0; op_idx != stage->operand_count(); ++op_idx) {
      if (stage->operand(op_idx)->opcode() != HloOpcode::kParameter) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(bool added,
                          AddCopyIfParamterModifiedInplace(stage, op_idx));
      changed |= added;
    }
  }
  return changed;
}

StatusOr<bool> InsertReadOnlyVariableCopies(HloInstruction* pipeline_op) {
  bool changed = false;
  HloComputation* pipeline_comp = pipeline_op->to_apply();
  HloInstruction* root = pipeline_comp->root_instruction();
  // Secondly, we make sure that if a parameter of the pipeline is an output
  // of the pipeline as well then it is not modified. TF2XLA lowering of
  // pipelines expects these to not be modified.
  if (root->operand_count() != pipeline_comp->num_parameters()) {
    return FailedPrecondition(
        "Expected the Pipeline to have %d outputs, but has %d",
        root->operand_count(), pipeline_comp->num_parameters());
  }
  for (int64 param_idx = 0; param_idx != pipeline_comp->num_parameters();
       ++param_idx) {
    HloInstruction* param_inst =
        pipeline_comp->parameter_instruction(param_idx);
    if (root->operand(param_idx) != param_inst) {
      // Don't need to do anything, the value is modified.
      continue;
    }
    for (HloInstruction* user : param_inst->users()) {
      if (user == root) {
        continue;
      }
      CHECK(IsPipelineStageOrBackwardOp(user));
      // Go through each use of the parameter in the user and insert kCopy
      // instructions if necessary.
      for (int64 index : user->OperandIndices(param_inst)) {
        TF_ASSIGN_OR_RETURN(bool added,
                            AddCopyIfParamterModifiedInplace(user, index));
        changed |= added;
      }
    }
  }
  return changed;
}

StatusOr<bool> InsertIntraIPUCopies(PipelineStages& stages) {
  bool changed = false;
  // Go through all the inputs to stages, if they are GTEs (which must be from
  // the previous stage, otherwise the input would have been an inter IPU copy
  // or a FIFO), then insert a copy to make sure stages are not modifying the
  // same tensor.
  for (auto& stages : {stages.forward, stages.backward}) {
    for (HloInstruction* stage : stages) {
      for (int64 op_idx = 0; op_idx != stage->operand_count(); ++op_idx) {
        const HloInstruction* operand = stage->operand(op_idx);
        if (operand->opcode() == HloOpcode::kGetTupleElement) {
          CHECK(IsPipelineStageOrBackwardOp(operand->operand(0)));
          TF_ASSIGN_OR_RETURN(bool added,
                              AddCopyIfParamterModifiedInplace(stage, op_idx));
          changed |= added;
        }
      }
    }
  }
  return changed;
}
}  // namespace

StatusOr<bool> PipelineCopyInserter::InsertInPipeline(
    HloInstruction* pipeline_op) {
  HloComputation* pipeline_comp = pipeline_op->to_apply();
  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  // We first make sure that forward stages do not modify parameters inplace.
  TF_ASSIGN_OR_RETURN(bool inserted_fwd_copies,
                      InsertForwardStageCopies(stages));
  // Insert copies for any read-only pipeline inputs.
  TF_ASSIGN_OR_RETURN(bool inserted_ro_copies,
                      InsertReadOnlyVariableCopies(pipeline_op));
  // Insert intra IPU copies for any outputs from the previous stage on the same
  // device.
  TF_ASSIGN_OR_RETURN(bool inserted_intra_copies, InsertIntraIPUCopies(stages));
  return inserted_fwd_copies || inserted_ro_copies || inserted_intra_copies;
}

StatusOr<bool> PipelineCopyInserter::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> pipeline_ops,
                      GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to fix.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  VLOG(2) << "Before PipelineCopyInserter:";
  XLA_VLOG_LINES(2, module->ToString(HloPrintOptions::ShortParsable()));

  TF_ASSIGN_OR_RETURN(bool changed, InsertInPipeline(pipeline_ops[0]));

  if (changed) {
    VLOG(2) << "After PipelineCopyInserter:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the Pipeline.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
