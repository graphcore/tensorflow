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

#include <functional>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/slice_util.h"
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

bool IsValidGather(const HloGatherInstruction* gather) {
  // Evaluates whether the attributes of the given gather instruction are valid
  // according to the xla gather semantics spec.
  const Shape& output_shape = gather->shape();
  const Shape& operand_shape = gather->operand(0)->shape();
  const Shape& indices_shape = gather->operand(1)->shape();
  const auto& dim_numbers = gather->gather_dimension_numbers();
  const auto& offset_dims = dim_numbers.offset_dims();
  const auto& collapsed_slice_dims = dim_numbers.collapsed_slice_dims();
  const auto& start_index_map = dim_numbers.start_index_map();
  const auto& index_vector_dim = dim_numbers.index_vector_dim();
  const auto& slice_sizes = gather->gather_slice_sizes();

  const bool multiple_slice_dims = start_index_map.size() > 1;

  const int64 num_dims_operand =
      ShapeUtil::IsScalar(operand_shape) ? 1 : operand_shape.rank();

  const int64 num_dims_indices =
      ShapeUtil::IsScalar(indices_shape) ? 1 : indices_shape.rank();

  const int64 num_dims_output =
      ShapeUtil::IsScalar(output_shape) ? 1 : output_shape.rank();

  if (slice_sizes.size() != num_dims_operand) {
    return false;
  }

  if (multiple_slice_dims) {
    if (index_vector_dim < 0 || index_vector_dim >= num_dims_indices) {
      // If index vector dim must refer to a real dimension in the indices.
      return false;
    }
    if (indices_shape.dimensions(index_vector_dim) != start_index_map.size()) {
      // The index dimension of the indices shape must be the same size as the
      // number of slice dimensions.
      return false;
    }
    for (int64 i = 1; i != start_index_map.size(); ++i) {
      // Slice dimensions must be in ascending order.
      if (start_index_map[i - 1] >= start_index_map[i]) {
        return false;
      }
    }
  }

  auto in_bounds = [](int64 min, int64 max) {
    return [min, max](int64 n) { return n >= min && n < max; };
  };
  auto in_container = [](absl::Span<const int64> container) {
    return [container](int64 n) {
      return absl::c_find(container, n) != container.end();
    };
  };

  if (!absl::c_all_of(start_index_map, in_bounds(0, num_dims_operand))) {
    // Slice dims must refer to real dims in the data.
    return false;
  }
  if (!absl::c_all_of(offset_dims, in_bounds(0, num_dims_output))) {
    // Offset dims must refer to real dims in the output.
    return false;
  }
  std::vector<int64> slice_dims(start_index_map.begin(), start_index_map.end());
  if (!absl::c_all_of(collapsed_slice_dims, in_container(slice_dims))) {
    // Collapsed slice dims should only contain slice dims.
    return false;
  }

  return true;
}

bool IsConvertableToMultiSlice(const HloGatherInstruction* gather) {
  // Check that the gather instruction can be converted to a multi-slice.
  const Shape& operand_shape = gather->operand(0)->shape();
  const Shape& indices_shape = gather->operand(1)->shape();
  const auto& dim_numbers = gather->gather_dimension_numbers();
  const auto& start_index_map = dim_numbers.start_index_map();
  const auto& slice_sizes = gather->gather_slice_sizes();

  const int64 num_dims_operand =
      ShapeUtil::IsScalar(operand_shape) ? 1 : operand_shape.rank();

  switch (operand_shape.element_type()) {
    case PrimitiveType::S32:
    case PrimitiveType::F16:
    case PrimitiveType::F32:
      break;
    default:
      return false;
  }
  if (indices_shape.element_type() != PrimitiveType::S32) {
    return false;
  }

  // MultiSlice does not currently support non-standard slice sizes.
  for (int64 dim = 0; dim != num_dims_operand; ++dim) {
    if (absl::c_find(start_index_map, dim) != start_index_map.end()) {
      // The slice size must be 1 in the axes we are indexing.
      if (slice_sizes[dim] != 1) {
        return false;
      }
    } else {
      // The slice size must match the input shape in the axes we are not
      // indexing.
      if (slice_sizes[dim] != operand_shape.dimensions(dim)) {
        return false;
      }
    }
  }

  return true;
}

StatusOr<bool> ReplaceGather(HloGatherInstruction* gather) {
  HloComputation* computation = gather->parent();
  HloInstruction* values = gather->mutable_operand(0);
  HloInstruction* indices = gather->mutable_operand(1);

  HloInstruction* transformed_values = values;
  HloInstruction* transformed_indices = indices;

  const auto& dim_numbers = gather->gather_dimension_numbers();
  const auto& start_index_map = dim_numbers.start_index_map();
  const auto& offset_dims = dim_numbers.offset_dims();
  const auto& index_vector_dim = dim_numbers.index_vector_dim();
  int64 num_slice_dims = start_index_map.size();
  int64 num_offset_dims = offset_dims.size();

  std::vector<int64> iota_indices(num_slice_dims);
  absl::c_iota(iota_indices, 0);
  bool pre_transpose = !absl::c_equal(start_index_map, iota_indices);

  bool reduce_indices = num_slice_dims > 1;

  std::vector<int64> iota_offsets(num_offset_dims);
  absl::c_iota(iota_offsets, gather->shape().rank() - num_offset_dims);
  bool post_transpose = !absl::c_equal(offset_dims, iota_offsets);

  if (pre_transpose) {
    // If the slice dims are not already at the front of the input shape,
    // transpose input so that they are.
    std::vector<int64> slice_dims(start_index_map.begin(),
                                  start_index_map.end());
    TF_ASSIGN_OR_RETURN(transformed_values,
                        TransposeToFront(transformed_values, slice_dims));
  }

  // Reshape values to rank 2 where dim0 is the sliced dim.
  const auto& dims = transformed_values->shape().dimensions();
  std::vector<int64> flattened_dims(2);
  flattened_dims[0] = std::accumulate(
      dims.begin(), dims.begin() + num_slice_dims, 1, std::multiplies<int64>());
  flattened_dims[1] = std::accumulate(dims.begin() + num_slice_dims, dims.end(),
                                      1, std::multiplies<int64>());
  TF_ASSIGN_OR_RETURN(transformed_values,
                      ReshapeIfDifferent(transformed_values, flattened_dims));

  if (reduce_indices) {
    // Reduce the indices now that we have flattened the slice dims.
    // When there are multiple slice dims, the index tensor has an extra
    // dimension of size num_slice_dims, as we need an index per slice dim
    // rather than a single index. Now we have flattened the slice dims, we need
    // to reduce these indices into a single index for each slice.
    std::vector<int64> sizes(num_slice_dims);
    for (int64 i = 0; i != num_slice_dims; ++i) {
      sizes[i] = values->shape().dimensions(start_index_map[i]);
    }

    TF_ASSIGN_OR_RETURN(
        transformed_indices,
        ReduceIndices(transformed_indices, index_vector_dim, sizes));
  }

  // Flatten indices shape.
  if (transformed_indices->shape().rank() > 1) {
    // Don't bother reshaping rank 0 or 1 indices.
    TF_ASSIGN_OR_RETURN(transformed_indices, Flatten(transformed_indices));
  }

  // Create the multi slice.
  std::vector<int64> multi_slice_dims(2);
  // Use ElementsIn as it handles scalar and 1d indices tensors.
  multi_slice_dims[0] = ShapeUtil::ElementsIn(transformed_indices->shape());
  multi_slice_dims[1] = transformed_values->shape().dimensions(1);

  const Shape& gather_shape = gather->shape();
  const Shape multi_slice_shape =
      ShapeUtil::MakeShape(gather_shape.element_type(), multi_slice_dims);

  HloInstruction* multi_slice = computation->AddInstruction(CreateMultiSlice(
      multi_slice_shape, transformed_values, transformed_indices));
  gather->SetupDerivedInstruction(multi_slice);

  if (post_transpose) {
    // Create transposed version of the original gather shape by
    // transposing the index dimensions to the front.
    std::vector<int64> permutations;
    for (int64 dim = 0; dim != gather_shape.rank(); ++dim) {
      if (absl::c_find(offset_dims, dim) == offset_dims.end()) {
        permutations.push_back(dim);
      }
    }
    absl::c_copy(offset_dims, std::back_inserter(permutations));
    Shape untransposed_shape = ShapeUtil::PermuteDimensions(
        InversePermutation(permutations), gather_shape);

    // Reshape result to the transposed shape.
    TF_ASSIGN_OR_RETURN(multi_slice,
                        ReshapeIfDifferent(multi_slice, untransposed_shape));

    // Untranspose the result.
    TF_ASSIGN_OR_RETURN(multi_slice,
                        InverseTranspose(multi_slice, permutations));
  } else {
    // Restore the non sliced dimensions.
    TF_ASSIGN_OR_RETURN(multi_slice,
                        ReshapeIfDifferent(multi_slice, gather_shape));
  }

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
      if (inst->opcode() == HloOpcode::kGather) {
        auto* gather = Cast<HloGatherInstruction>(inst);
        if (!IsValidGather(gather)) {
          continue;
        }
        if (!IsConvertableToMultiSlice(gather)) {
          continue;
        }
        TF_ASSIGN_OR_RETURN(bool replaced, ReplaceGather(gather));
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
