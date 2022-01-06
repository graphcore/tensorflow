/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/fix_root_instruction.h"

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"

namespace xla {
namespace poplarplugin {

StatusOr<bool> FixRootInstructionsPass::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to fix.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  VLOG(2) << "Before fixing the Pipeline root instructions.";
  XLA_VLOG_LINES(2, module->ToString());

  HloInstruction* pipeline = pipeline_ops[0];

  TF_ASSIGN_OR_RETURN(auto stages_, GetPipelineStages(pipeline->to_apply()));
  TF_RETURN_IF_ERROR(FixRootInstructions(stages_));
  TF_ASSIGN_OR_RETURN(bool changed, FixRootInstruction(pipeline->to_apply()));
  TF_RETURN_IF_ERROR(
      HloDCE().RunOnComputation(pipeline->to_apply(), false).status());

  VLOG(2) << "After fixing the Pipeline root instructions.";
  XLA_VLOG_LINES(2, module->ToString());
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
