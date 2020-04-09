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

#include "tensorflow/compiler/plugin/poplar/driver/passes/gather_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {
namespace {
bool IsConvertableGather(const HloInstruction* inst) {
  return IsMultiSliceGather(inst);
}

StatusOr<bool> ReplaceGather(HloInstruction* gather) {
  HloComputation* computation = gather->parent();
  auto operands = gather->operands();

  HloInstruction* multi_slice = computation->AddInstruction(
      CreateMultiSlice(gather->shape(), operands[0], operands[1]));

  gather->SetupDerivedInstruction(multi_slice);
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(gather, multi_slice));
  return true;
}
}  // namespace

StatusOr<bool> GatherSimplifier::Run(HloModule* module) {
  bool changed = false;
  VLOG(2) << "Before the GatherSimplifier:";
  XLA_VLOG_LINES(2, module->ToString());

  for (auto comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }
    // Go through instructions in post order to make sure we do not change
    // operands.
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (IsConvertableGather(inst)) {
        TF_ASSIGN_OR_RETURN(bool replaced, ReplaceGather(inst));
        changed |= replaced;
      }
    }
  }

  if (changed) {
    VLOG(2) << "After the GatherSimplifier:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "There were no changes.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
