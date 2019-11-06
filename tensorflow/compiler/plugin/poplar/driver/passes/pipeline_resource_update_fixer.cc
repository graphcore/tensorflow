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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_resource_update_fixer.h"

#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {
// Make sure the root of the pipeline computation is a tuple.
StatusOr<bool> FixRoot(PipelineStages& stages) {
  HloInstruction* resource_update = *stages.resource_update;
  HloComputation* pipeline_comp = resource_update->parent();
  HloInstruction* pipeline_comp_root = pipeline_comp->root_instruction();

  // We do not need to fix the root if it's not the resource update.
  if (pipeline_comp_root != resource_update) {
    if (pipeline_comp_root->opcode() != HloOpcode::kTuple) {
      return InternalErrorStrCat(
          "Expected the root of the Pipeline computation to be a tuple "
          "instruction.");
    }
    return false;
  }
  // Add a GTE for each output from the resource update.
  const int64 num_elements =
      ShapeUtil::TupleElementCount(resource_update->shape());
  std::vector<HloInstruction*> gtes(num_elements);
  for (int64 idx = 0; idx != num_elements; ++idx) {
    TF_ASSIGN_OR_RETURN(gtes[idx],
                        MakeGetTupleElementHlo(resource_update, idx));
  }
  HloInstruction* tuple_root =
      pipeline_comp->AddInstruction(HloInstruction::CreateTuple(gtes));
  pipeline_comp->set_root_instruction(tuple_root);
  return true;
}
}  // namespace

StatusOr<bool> PipelineResourceUpdateFixer::FixPipeline(
    HloInstruction* pipeline_op) {
  HloComputation* pipeline_comp = pipeline_op->to_apply();

  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  if (!stages.resource_update) {
    return false;
  }
  // Make sure the root is in the expected format.
  TF_ASSIGN_OR_RETURN(bool root_changed, FixRoot(stages));

  return root_changed;
}

StatusOr<bool> PipelineResourceUpdateFixer::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> pipeline_ops,
                      GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to fix.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  VLOG(2) << "Before PipelineResourceUpdateFixer:";
  XLA_VLOG_LINES(2, module->ToString());

  TF_ASSIGN_OR_RETURN(bool changed, FixPipeline(pipeline_ops[0]));

  if (changed) {
    VLOG(2) << "After PipelineResourceUpdateFixer:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the Pipeline.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
