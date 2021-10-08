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

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {
// TODO(T45278) popops::multiUpdate and popops::multiUpdateAdd only supports the
// 2D case.
bool CheckValidMultiUpdateAttributes(const HloScatterInstruction* inst) {
  const Shape operand_shape = inst->operand(0)->shape();
  const Shape indices_shape = inst->operand(1)->shape();
  const Shape updates_shape = inst->operand(2)->shape();
  const auto dim_numbers = inst->scatter_dimension_numbers();
  const auto update_window_dims = dim_numbers.update_window_dims();
  const auto inserted_window_dims = dim_numbers.inserted_window_dims();
  const auto scatter_dims_to_operand_dims =
      dim_numbers.scatter_dims_to_operand_dims();
  const auto index_vector_dim = dim_numbers.index_vector_dim();
  const uint64 index_dim_size =
      indices_shape.rank() == index_vector_dim
          ? 1
          : indices_shape.dimensions(index_vector_dim);
  return operand_shape.rank() == 2 && index_dim_size == 1 &&
         scatter_dims_to_operand_dims.size() == 1 &&
         scatter_dims_to_operand_dims[0] == 0 &&
         inserted_window_dims.size() == 1 && inserted_window_dims[0] == 0 &&
         update_window_dims.size() == 1 &&
         update_window_dims[0] == (updates_shape.rank() - 1);
}

bool IsMultiUpdateScatter(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kScatter) {
    const HloScatterInstruction* scatter = Cast<HloScatterInstruction>(inst);
    const HloInstruction* root = inst->to_apply()->root_instruction();
    return Match(root, m::Parameter(1)) &&
           CheckValidMultiUpdateAttributes(scatter);
  }
  return false;
}

bool IsMultiUpdateAddScatter(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kScatter) {
    const HloScatterInstruction* scatter = Cast<HloScatterInstruction>(inst);
    const HloInstruction* root = inst->to_apply()->root_instruction();
    return Match(root, m::Add(m::Parameter(0), m::Parameter(1))) &&
           CheckValidMultiUpdateAttributes(scatter);
  }
  return false;
}

bool IsConvertableScatter(const HloInstruction* inst) {
  return IsMultiUpdateScatter(inst) || IsMultiUpdateAddScatter(inst);
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

StatusOr<bool> ReplaceScatter(HloInstruction* scatter) {
  VLOG(3) << "Simplifying " << scatter->ToString();

  HloComputation* computation = scatter->parent();
  auto dim_numbers = scatter->scatter_dimension_numbers();
  const int64 index_vector_dim = dim_numbers.index_vector_dim();
  const bool is_update_add = IsMultiUpdateAddScatter(scatter);
  int64 update_dim = dim_numbers.update_window_dims()[0];

  HloInstruction* operand = scatter->mutable_operand(0);
  HloInstruction* indices = scatter->mutable_operand(1);
  HloInstruction* updates = scatter->mutable_operand(2);

  // If the indices are scalar then this is a dynamic-update-slice kind of
  // scatter.
  if (ShapeUtil::IsScalar(indices->shape())) {
    TF_ASSIGN_OR_RETURN(updates, PrependDegenerateDims(updates, 1));
    update_dim += 1;
  }

  // Reshape the indices into a 2D shape [num_lookups, 1].
  const Shape& indices_shape = indices->shape();
  const Shape new_indices_shape = ShapeUtil::MakeShape(
      indices_shape.element_type(), {ShapeUtil::ElementsIn(indices_shape), 1});
  TF_ASSIGN_OR_RETURN(indices, MakeReshapeHlo(new_indices_shape, indices));

  // Move the update_dim to the back.
  if ((updates->shape().rank() - 1) != update_dim) {
    TF_ASSIGN_OR_RETURN(updates,
                        MoveInstructionDimensionToBack(updates, update_dim));
    update_dim = updates->shape().rank() - 1;
  }

  // Collapse all but the last dimension of updates.
  if (updates->shape().rank() != 2) {
    TF_ASSIGN_OR_RETURN(updates, CollapseAllButLastDimension(updates));
    update_dim = 1;
  }

  HloInstruction* multi_update;
  if (is_update_add) {
    // We use one for the scale.
    HloInstruction* one =
        computation->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::One(scatter->shape().element_type())));
    multi_update = computation->AddInstruction(CreateMultiUpdateAdd(
        scatter->shape(), {operand, indices, updates, one}, update_dim));
  } else {
    multi_update = computation->AddInstruction(CreateMultiUpdate(
        scatter->shape(), {operand, indices, updates}, update_dim));
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
