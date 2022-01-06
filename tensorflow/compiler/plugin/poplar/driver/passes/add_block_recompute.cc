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

#include "tensorflow/compiler/plugin/poplar/driver/passes/add_block_recompute.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/recompute.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

namespace {
// Be conservative and block all ops which have side effects or sub
// computations.
bool ShouldBlockInstruction(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kParameter ||
         inst->opcode() == HloOpcode::kConstant || inst->HasSideEffect() ||
         !inst->called_computations().empty();
}
}  // namespace

StatusOr<bool> AddBlockRecompute::Run(HloModule* module) {
  bool result = false;

  for (auto comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (auto inst : comp->MakeInstructionPostOrder()) {
      if (ShouldBlockInstruction(inst)) {
        auto block = comp->AddInstruction(CreateBlockRecompute(inst));
        inst->SetupDerivedInstruction(block);
        TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(block));
        result = true;
      }
    }
  }

  return result;
}

}  // namespace poplarplugin
}  // namespace xla
