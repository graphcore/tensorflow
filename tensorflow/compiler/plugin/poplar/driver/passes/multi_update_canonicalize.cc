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

#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_update_canonicalize.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {
namespace {
bool IsMultiUpdate(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::MultiUpdate)(inst);
}

bool IsMultiUpdateAdd(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(inst);
}

StatusOr<HloInstruction*> MoveInstructionDimensionToBack(
    HloInstruction* inst, std::size_t dim_to_move) {
  std::vector<int64> permutation(inst->shape().rank());
  for (size_t i = 0, next_idx = 0; i != permutation.size(); ++i) {
    if (i != dim_to_move) {
      permutation[next_idx++] = i;
    }
  }
  permutation.back() = dim_to_move;
  TF_ASSIGN_OR_RETURN(HloInstruction * new_inst,
                      MakeTransposeHlo(inst, permutation));
  inst->SetupDerivedInstruction(new_inst);
  return new_inst;
}

StatusOr<HloInstruction*> CollapseAllButLastDimension(HloInstruction* inst) {
  HloComputation* computation = inst->parent();
  const Shape inst_shape = inst->shape();
  const int64 last_dim = inst_shape.dimensions(inst_shape.rank() - 1);
  const Shape new_inst_shape = ShapeUtil::MakeShape(
      inst_shape.element_type(),
      {ShapeUtil::ElementsIn(inst_shape) / last_dim, last_dim});
  HloInstruction* new_inst = computation->AddInstruction(
      HloInstruction::CreateReshape(new_inst_shape, inst));
  inst->SetupDerivedInstruction(new_inst);
  return new_inst;
}

StatusOr<bool> ReplaceMultiUpdate(HloInstruction* inst) {
  HloMultiUpdateInstruction* multi_update =
      Cast<HloMultiUpdateInstruction>(inst);
  bool changed = false;
  HloInstruction* operand = multi_update->mutable_operand(0);
  HloInstruction* indices = multi_update->mutable_operand(1);
  HloInstruction* updates = multi_update->mutable_operand(2);
  HloComputation* computation = inst->parent();
  std::size_t index_vector_dim = multi_update->GetIndexVectorDimension();
  std::size_t update_dim = multi_update->GetUpdateSliceDimension();
  const uint32 serialization_factor = multi_update->GetSerializationFactor();

  // First check if we need to add an extra dimension for the index_vector_dim
  // so that it is no longer explicit.
  if (static_cast<size_t>(indices->shape().rank()) == index_vector_dim) {
    Shape new_indices_shape = indices->shape();
    ShapeUtil::AppendMajorDimension(1, &new_indices_shape);
    HloInstruction* new_indices = computation->AddInstruction(
        HloInstruction::CreateReshape(new_indices_shape, indices));
    indices->SetupDerivedInstruction(new_indices);
    indices = new_indices;
    changed = true;
  }

  // Move index_vector_dim to the back.
  if ((indices->shape().rank() - 1) != static_cast<int64>(index_vector_dim)) {
    TF_ASSIGN_OR_RETURN(
        indices, MoveInstructionDimensionToBack(indices, index_vector_dim));
    index_vector_dim = indices->shape().rank() - 1;
    changed = true;
  }

  // Collapse all but the last dimension of indices.
  if (indices->shape().rank() != 2) {
    TF_ASSIGN_OR_RETURN(indices, CollapseAllButLastDimension(indices));
    index_vector_dim = 1;
    changed = true;
  }

  // Move the update_dim to the back.
  if ((updates->shape().rank() - 1) != static_cast<int64>(update_dim)) {
    TF_ASSIGN_OR_RETURN(updates,
                        MoveInstructionDimensionToBack(updates, update_dim));
    update_dim = updates->shape().rank() - 1;
    changed = true;
  }

  // Collapse all but the last dimension of updates.
  if (updates->shape().rank() != 2) {
    TF_ASSIGN_OR_RETURN(updates, CollapseAllButLastDimension(updates));
    update_dim = 1;
    changed = true;
  }

  if (changed) {
    HloInstruction* new_inst;
    if (IsMultiUpdate(inst)) {
      new_inst = computation->AddInstruction(CreateMultiUpdate(
          operand->shape(), {operand, indices, updates}, index_vector_dim,
          update_dim, serialization_factor));
    } else {
      HloInstruction* scale = inst->mutable_operand(3);
      new_inst = computation->AddInstruction(CreateMultiUpdateAdd(
          operand->shape(), {operand, indices, updates, scale},
          index_vector_dim, update_dim, serialization_factor));
    }
    inst->SetupDerivedInstruction(new_inst);
    TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(new_inst));
    TF_RETURN_IF_ERROR(computation->RemoveInstruction(inst));
  }
  return changed;
}
}  // namespace

StatusOr<bool> MultiUpdateCanonicalize::Run(HloModule* module) {
  bool changed = false;
  VLOG(2) << "Before the MultiUpdateCanonicalize:";
  XLA_VLOG_LINES(2, module->ToString());

  for (HloComputation* comp : module->MakeNonfusionComputations()) {
    // Go through instructions in post order to make sure we do not change
    // operands.
    auto insts = comp->MakeInstructionPostOrder();
    for (HloInstruction* inst : insts) {
      if (IsPoplibsHloCustomOp(inst) &&
          (IsMultiUpdate(inst) || IsMultiUpdateAdd(inst))) {
        TF_ASSIGN_OR_RETURN(bool replaced, ReplaceMultiUpdate(inst));
        changed |= replaced;
      }
    }
  }

  if (changed) {
    VLOG(2) << "After the MultiUpdateCanonicalize:";
    XLA_VLOG_LINES(2, module->ToString());
    // LOG(FATAL) << module->ToString();
  } else {
    VLOG(2) << "There were no changes.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
