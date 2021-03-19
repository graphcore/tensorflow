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

#include "tensorflow/compiler/plugin/poplar/driver/passes/fusion_inliner.h"

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
namespace poplarplugin {
namespace {
Status InlineFusion(HloInstruction* fusion) {
  HloComputation* fusion_comp = fusion->fused_instructions_computation();
  HloComputation* comp = fusion->parent();
  return InlineComputation(fusion, fusion_comp).status();
}
}  // namespace

FusionInliner::FusionInliner(std::function<bool(HloInstruction*)> predicate)
    : predicate_(predicate) {}

StatusOr<bool> FusionInliner::Run(HloModule* module) {
  VLOG(2) << "Before FusionInliner:";
  XLA_VLOG_LINES(2, module->ToString());
  bool changed = false;

  std::vector<HloInstruction*> to_inline;

  for (auto comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (auto inst : comp->MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kFusion && predicate_(inst)) {
        to_inline.push_back(inst);
      }
    }
  }

  if (to_inline.size()) {
    for (auto inst : to_inline) {
      TF_RETURN_IF_ERROR(InlineFusion(inst));
    }

    VLOG(2) << "After FusionInliner:";
    XLA_VLOG_LINES(2, module->ToString());
    return true;
  } else {
    VLOG(2) << "No changes were made to the module.";
    return false;
  }
}

}  // namespace poplarplugin
}  // namespace xla
