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
#include "tensorflow/compiler/xla/service/shape_inference.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {
namespace {

bool CheckValidMultiSliceAttributes(const HloGatherInstruction* inst) {
  const Shape& output_shape = inst->shape();
  const Shape& operand_shape = inst->operand(0)->shape();
  const Shape& start_indices = inst->operand(1)->shape();
  const auto& dim_numbers = inst->gather_dimension_numbers();
  const auto& offset_dims = dim_numbers.offset_dims();
  const auto& start_index_map = dim_numbers.start_index_map();
  const auto& collapsed_slice_dims = dim_numbers.collapsed_slice_dims();
  const auto& index_vector_dim = dim_numbers.index_vector_dim();
  const auto& slice_sizes = inst->gather_slice_sizes();

  const uint64 index_dim_size =
      start_indices.rank() == index_vector_dim
          ? 1
          : start_indices.dimensions(index_vector_dim);

  if (index_dim_size != 1) {
    return false;
  }

  const int64 num_dims =
      ShapeUtil::IsScalar(operand_shape) ? 1 : operand_shape.rank();

  if (slice_sizes.size() != num_dims) {
    return false;
  }

  if (collapsed_slice_dims.size() != 1) {
    return false;
  }

  const int64 collapsed_slice_dim = collapsed_slice_dims[0];

  // Collapsed axis of slice sizes must have dimension 1.
  if (slice_sizes[collapsed_slice_dim] != 1) {
    return false;
  }

  for (int64 dim = 0; dim != num_dims; ++dim) {
    if (dim == collapsed_slice_dim) {
      continue;
    }

    // Non collapsed axis of operand shape should be same as
    // non collapsed axis of slice sizes.
    if (operand_shape.dimensions(dim) != slice_sizes[dim]) {
      return false;
    }
  }

  // Size of offset dims must be 1 less than the operand rank.
  if (offset_dims.size() != num_dims - 1) {
    return false;
  }

  for (int64 dim = 0, j = 0; dim != num_dims; ++dim) {
    if (dim == collapsed_slice_dim) {
      continue;
    }

    if (output_shape.dimensions(offset_dims[j++]) != slice_sizes[dim]) {
      return false;
    }
  }

  std::vector<int64> incremental_start_index_map(start_index_map.size());
  absl::c_iota(incremental_start_index_map, 0);
  if (!absl::c_equal(incremental_start_index_map, start_index_map)) {
    return false;
  }

  // TODO(T14037): only slicing along the zeroth dimension is currently
  // supported.
  if (collapsed_slice_dim != 0) {
    return false;
  }

  return true;
}

bool IsConvertableGather(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kGather) {
    const HloGatherInstruction* gather = Cast<HloGatherInstruction>(inst);
    return CheckValidMultiSliceAttributes(gather);
  }
  return false;
}

StatusOr<bool> ReplaceGather(HloInstruction* gather) {
  HloComputation* computation = gather->parent();
  HloInstruction* values = gather->mutable_operand(0);
  HloInstruction* indices = gather->mutable_operand(1);

  const Shape& gather_shape = gather->shape();
  const Shape& values_shape = values->shape();

  // Collapse the non sliced dimensions for values.
  const std::vector<int64> new_values_dims =
      ShapeUtil::IsScalar(values_shape)
          ? std::vector<int64>{1, 1}
          : std::vector<int64>{values_shape.dimensions(0),
                               ShapeUtil::ElementsIn(values_shape) /
                                   values_shape.dimensions(0)};
  TF_ASSIGN_OR_RETURN(values, MakeReshapeHlo(new_values_dims, values));

  // Create the multi slice.
  const std::vector<int64> multi_slice_dims =
      ShapeUtil::IsScalar(gather_shape)
          ? std::vector<int64>{1, 1}
          : std::vector<int64>{gather_shape.dimensions(0),
                               ShapeUtil::ElementsIn(gather_shape) /
                                   gather_shape.dimensions(0)};
  const Shape multi_slice_shape =
      ShapeUtil::MakeShape(gather_shape.element_type(), multi_slice_dims);

  HloInstruction* multi_slice = computation->AddInstruction(
      CreateMultiSlice(multi_slice_shape, values, indices));
  gather->SetupDerivedInstruction(multi_slice);

  // Restore the non sliced dimensions.
  TF_ASSIGN_OR_RETURN(multi_slice, MakeReshapeHlo(gather_shape, multi_slice));

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
