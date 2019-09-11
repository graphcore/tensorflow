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

StatusOr<bool> PipelineCopyInserter::InsertInPipeline(
    HloInstruction* pipeline_op) {
  bool changed = false;
  HloComputation* pipeline_comp = pipeline_op->to_apply();
  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  // We only do this for the forward stages.
  for (HloInstruction* stage : stages.forward) {
    HloComputation* stage_comp = stage->to_apply();
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
      HloInstruction* parameter = stage_comp->parameter_instruction(op_idx);
      if (IsOutputModifiedInplace(parameter)) {
        // Insert a copy from the the parameter.
        HloInstruction* copy =
            stage_comp->AddInstruction(HloInstruction::CreateUnary(
                parameter->shape(), HloOpcode::kCopy, parameter));
        parameter->SetupDerivedInstruction(copy);
        TF_RETURN_IF_ERROR(parameter->ReplaceAllUsesWith(copy));
        changed = true;
      }
    }
  }
  return changed;
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
