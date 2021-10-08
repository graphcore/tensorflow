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
absl::optional<int64> GetScatterDimension(
    int64 rank, absl::Span<const int64> update_window_dims) {
  std::vector<int64> all_dims(rank);
  absl::c_iota(all_dims, 0);

  std::vector<int64> scatter_dims;
  absl::c_set_difference(all_dims, update_window_dims,
                         std::inserter(scatter_dims, scatter_dims.begin()));
  if (scatter_dims.size() == 1) {
    return scatter_dims[0];
  }
  return absl::nullopt;
}

bool CheckValidMultiUpdateAttributes(const HloInstruction* inst) {
  const Shape& operand_shape = inst->operand(0)->shape();
  const Shape& indices_shape = inst->operand(1)->shape();
  const Shape& updates_shape = inst->operand(2)->shape();
  const auto& dim_numbers = inst->scatter_dimension_numbers();
  const auto& update_window_dims = dim_numbers.update_window_dims();
  const auto& inserted_window_dims = dim_numbers.inserted_window_dims();
  const auto& scatter_dims_to_operand_dims =
      dim_numbers.scatter_dims_to_operand_dims();
  const auto index_vector_dim = dim_numbers.index_vector_dim();
  const uint64 index_dim_size =
      indices_shape.rank() == index_vector_dim
          ? 1
          : indices_shape.dimensions(index_vector_dim);

  if (updates_shape.rank() == 0) {
    return false;
  }

  if (updates_shape.rank() != operand_shape.rank()) {
    return false;
  }

  if (index_dim_size != 1) {
    return false;
  }

  if (update_window_dims.size() != (updates_shape.rank() - 1)) {
    return false;
  }

  auto scatter_dimension_opt = GetScatterDimension(
      updates_shape.rank(), AsInt64Slice(update_window_dims));
  if (!scatter_dimension_opt) {
    return false;
  }

  const int64 scatter_dimension = *scatter_dimension_opt;

  if (ShapeUtil::DeleteDimension(scatter_dimension, updates_shape) !=
      ShapeUtil::DeleteDimension(scatter_dimension, operand_shape)) {
    return false;
  }

  if (scatter_dims_to_operand_dims.size() != 1) {
    return false;
  }

  if (scatter_dims_to_operand_dims[0] != 0) {
    return false;
  }

  if (inserted_window_dims.size() != 1) {
    return false;
  }

  if (inserted_window_dims[0] != 0) {
    return false;
  }

  return true;
}

bool IsMultiUpdateScatter(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kScatter) {
    const HloScatterInstruction* scatter = Cast<HloScatterInstruction>(inst);
    const HloInstruction* root = inst->to_apply()->root_instruction();
    return Match(root, m::Parameter(1));
  }
  return false;
}

bool IsMultiUpdateAddScatter(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kScatter) {
    const HloScatterInstruction* scatter = Cast<HloScatterInstruction>(inst);
    const HloInstruction* root = inst->to_apply()->root_instruction();
    return Match(root, m::Add(m::Parameter(0), m::Parameter(1)));
  }
  return false;
}

bool IsConvertableScatter(const HloInstruction* inst) {
  return IsMultiUpdateScatter(inst) || IsMultiUpdateAddScatter(inst);
}

std::vector<int64> GetPermutation(const HloInstruction* inst,
                                  std::size_t dim_to_move) {
  std::vector<int64> permutation(inst->shape().rank());
  permutation[0] = dim_to_move;
  for (size_t i = 0, next_idx = 1; i != permutation.size(); ++i) {
    if (i != dim_to_move) {
      permutation[next_idx++] = i;
    }
  }
  return permutation;
}

StatusOr<HloInstruction*> CollapseAllButZerothDimension(HloInstruction* inst) {
  HloComputation* computation = inst->parent();
  const Shape inst_shape = inst->shape();
  const int64 zero_dim = inst_shape.dimensions(0);
  const Shape new_shape = ShapeUtil::MakeShape(
      inst_shape.element_type(),
      {zero_dim, ShapeUtil::ElementsIn(inst_shape) / zero_dim});
  return MakeReshapeHlo(new_shape, inst);
}

StatusOr<bool> ReplaceScatter(HloInstruction* scatter) {
  VLOG(3) << "Simplifying " << scatter->ToString();

  HloComputation* computation = scatter->parent();
  auto dim_numbers = scatter->scatter_dimension_numbers();
  const int64 index_vector_dim = dim_numbers.index_vector_dim();
  const auto& update_window_dims = dim_numbers.update_window_dims();
  const bool is_update_add = IsMultiUpdateAddScatter(scatter);

  HloInstruction* operand = scatter->mutable_operand(0);
  HloInstruction* indices = scatter->mutable_operand(1);
  HloInstruction* updates = scatter->mutable_operand(2);

  // Reshape the indices into a 2D shape [num_lookups, 1].
  const Shape& indices_shape = indices->shape();
  const Shape new_indices_shape = ShapeUtil::MakeShape(
      indices_shape.element_type(), {ShapeUtil::ElementsIn(indices_shape), 1});
  TF_ASSIGN_OR_RETURN(indices, MakeReshapeHlo(new_indices_shape, indices));

  const int64 scatter_dimension = *GetScatterDimension(
      updates->shape().rank(), AsInt64Slice(update_window_dims));

  const std::vector<int64> permutation =
      GetPermutation(operand, scatter_dimension);
  const std::vector<int64> invert_permutation =
      InvertPermutations<int64>(permutation);

  // Move the scatter dimension to the front.
  TF_ASSIGN_OR_RETURN(operand, MakeTransposeHlo(operand, permutation));
  TF_ASSIGN_OR_RETURN(updates, MakeTransposeHlo(updates, permutation));

  const Shape pre_flatten_operand_shape = operand->shape();

  // Collapse all but the zeroth dimension.
  TF_ASSIGN_OR_RETURN(operand, CollapseAllButZerothDimension(operand));
  TF_ASSIGN_OR_RETURN(updates, CollapseAllButZerothDimension(updates));

  HloInstruction* multi_update;
  if (is_update_add) {
    // We use one for the scale.
    HloInstruction* one =
        computation->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::One(scatter->shape().element_type())));
    multi_update = computation->AddInstruction(CreateMultiUpdateAdd(
        operand->shape(), {operand, indices, updates, one}));
  } else {
    multi_update = computation->AddInstruction(
        CreateMultiUpdate(operand->shape(), {operand, indices, updates}));
  }
  scatter->SetupDerivedInstruction(multi_update);

  // Uncollapse the dimensions.
  TF_ASSIGN_OR_RETURN(multi_update,
                      MakeReshapeHlo(pre_flatten_operand_shape, multi_update));
  // Undo the transpose.
  TF_ASSIGN_OR_RETURN(multi_update,
                      MakeTransposeHlo(multi_update, invert_permutation));

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
      if (IsConvertableScatter(inst) && CheckValidMultiUpdateAttributes(inst)) {
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
