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

#include "tensorflow/compiler/plugin/poplar/driver/passes/scatter_simplifier.h"
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
bool IsConvertableScatter(const HloInstruction* inst) {
  return IsMultiUpdateScatter(inst) || IsMultiUpdateAddScatter(inst);
}

StatusOr<bool> ReplaceScatter(HloInstruction* scatter) {
  HloComputation* computation = scatter->parent();

  auto dim_numbers = scatter->scatter_dimension_numbers();
  const std::size_t index_vector_dim = dim_numbers.index_vector_dim();
  const unsigned update_dim = dim_numbers.update_window_dims()[0];
  const bool is_update_add = IsMultiUpdateAddScatter(scatter);

  HloInstruction* multi_update;
  auto operands = scatter->operands();
  if (is_update_add) {
    // We use one for the scale.
    HloInstruction* one =
        computation->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::One(scatter->shape().element_type())));
    operands.push_back(one);
    multi_update = computation->AddInstruction(CreateMultiUpdateAdd(
        scatter->shape(), operands, index_vector_dim, update_dim));
  } else {
    multi_update = computation->AddInstruction(CreateMultiUpdate(
        scatter->shape(), operands, index_vector_dim, update_dim));
  }
  scatter->SetupDerivedInstruction(multi_update);
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(scatter, multi_update));
  return true;
}
}  // namespace

StatusOr<bool> ScatterSimplifier::Run(HloModule* module) {
  bool changed = false;
  VLOG(2) << "Before the ScatterSimplifier:";
  XLA_VLOG_LINES(2, module->ToString());

  for (auto comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    // Go through instructions in post order to make sure we do not change
    // operands.
    auto insts = comp->MakeInstructionPostOrder();
    for (HloInstruction* inst : insts) {
      if (IsConvertableScatter(inst)) {
        TF_ASSIGN_OR_RETURN(bool replaced, ReplaceScatter(inst));
        changed |= replaced;
      }
    }
  }

  if (changed) {
    VLOG(2) << "After the ScatterSimplifier:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "There were no changes.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
