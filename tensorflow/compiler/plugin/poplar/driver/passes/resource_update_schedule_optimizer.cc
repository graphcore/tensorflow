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

#include "tensorflow/compiler/plugin/poplar/driver/passes/resource_update_schedule_optimizer.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {

StatusOr<bool> ResourceUpdateScheduleOptimizer::OptimizeResourceUpdate(
    HloInstruction* resource_update_op) {
  bool changed = false;
  HloComputation* comp = resource_update_op->to_apply();
  auto reachability_map = HloReachabilityMap::Build(comp);

  // Find all the inter IPU copies.
  std::vector<HloInstruction*> inter_ipu_copies;
  absl::c_copy_if(comp->MakeInstructionPostOrder(),
                  std::back_inserter(inter_ipu_copies),
                  [](HloInstruction* inst) -> bool {
                    return IsPoplarInstruction(PoplarOp::InterIpuCopy)(inst);
                  });

  // Find all the inputs to the computation.
  std::vector<HloInstruction*> comp_inputs;
  absl::c_copy_if(
      comp->MakeInstructionPostOrder(), std::back_inserter(comp_inputs),
      [](HloInstruction* inst) -> bool { return inst->operand_count() == 0; });

  for (HloInstruction* inter_ipu_copy : inter_ipu_copies) {
    for (HloInstruction* comp_input : comp_inputs) {
      if (!reachability_map->IsReachable(comp_input, inter_ipu_copy)) {
        TF_RETURN_IF_ERROR(inter_ipu_copy->AddControlDependencyTo(comp_input));
        reachability_map->UpdateReachabilityThroughInstruction(comp_input);
        changed = true;
      }
    }
  }
  return changed;
}

StatusOr<bool> ResourceUpdateScheduleOptimizer::Run(HloModule* module) {
  VLOG(2) << "Before ResourceUpdateScheduleOptimizer:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;

  // Run it for pipelines.
  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
  for (HloInstruction* pipeline_op : pipeline_ops) {
    HloComputation* pipeline_comp = pipeline_op->to_apply();
    TF_ASSIGN_OR_RETURN(PipelineStages stages,
                        GetPipelineStages(pipeline_comp));
    if (stages.resource_update) {
      TF_ASSIGN_OR_RETURN(bool changed_pipeline,
                          OptimizeResourceUpdate(*stages.resource_update));
      changed |= changed_pipeline;
    }
  }

  if (changed) {
    VLOG(2) << "After ResourceUpdateScheduleOptimizer:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
