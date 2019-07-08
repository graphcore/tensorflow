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
#include <algorithm>
#include <limits>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Scatter.hpp>

namespace xla {
namespace poplarplugin {
namespace {

// Transposes the given scatter_indices such that the index_vector_dim becomes
// the most-minor dimension.
poplar::Tensor TransposeIndexVectorDimToLast(poplar::Tensor indices,
                                             unsigned index_vector_dim) {
  if (indices.rank() == index_vector_dim) {
    return indices;
  }

  if (indices.rank() == index_vector_dim + 1) {
    return indices;
  }

  std::vector<unsigned> permutation(indices.rank());

  const auto front = std::begin(permutation);
  const auto mid = std::next(std::begin(permutation), index_vector_dim);
  const auto back = std::end(permutation) - 1;

  std::iota(front, mid, 0);
  std::iota(mid, back, index_vector_dim + 1);
  *back = index_vector_dim;

  return indices.dimShuffle(permutation);
}

// Canonicalizes the scatter_indices tensor in order to keep them uniform while
// performing the scatter operation.
poplar::Tensor CanonicalizeScatterIndices(poplar::Tensor scatter_indices,
                                          unsigned index_vector_dim) {
  // Transpose the non-index-vector dimensions to the front.
  poplar::Tensor scatter_indices_t =
      TransposeIndexVectorDimToLast(scatter_indices, index_vector_dim);

  const bool indices_are_scalar = scatter_indices_t.rank() == index_vector_dim;

  // The number of dimensions in scatter_indices that are index dimensions.
  const std::size_t index_dims_in_scatter_indices = indices_are_scalar ? 0 : 1;

  // If there is only one index (i.e. scatter_indices has rank 1 and this
  // scatter is really just a dynamic update slice) add a leading degenerate
  // dimension for uniformity.  Otherwise create a "collapsed" leading dimension
  // that subsumes all of the non-index-vector dimensions.
  std::vector<std::size_t> shape = scatter_indices_t.shape();
  if (shape.size() == index_dims_in_scatter_indices) {
    shape.insert(shape.begin(), 1);
    return scatter_indices_t.reshape(shape);
  }

  if (indices_are_scalar) {
    return scatter_indices_t.reshape({scatter_indices_t.numElements(), 1});
  }

  // Collapse all but the dimensions (0 or 1) in scatter_indices containing
  // the index vectors.
  std::vector<std::size_t> new_shape = {
      scatter_indices_t.numElements() / shape.back(), shape.back()};
  return scatter_indices_t.reshape(new_shape);
}

// Permutes the `updates` tensor such that all the scatter dims appear in the
// major dimensions and all the window dimensions appear in the minor
// dimensions.
poplar::Tensor PermuteScatterAndWindowDims(
    poplar::Tensor updates, const std::vector<unsigned> update_window_dims) {
  std::vector<unsigned> permutation(updates.rank());

  std::iota(std::begin(permutation), std::end(permutation), 0);

  const auto is_window_dim = [&update_window_dims](unsigned dim) {
    return !std::binary_search(std::begin(update_window_dims),
                               std::end(update_window_dims), dim);
  };

  std::stable_partition(std::begin(permutation), std::end(permutation),
                        is_window_dim);

  return updates.dimShuffle(permutation);
}

// Expands or contracts the scatter indices in the updates tensor.
poplar::Tensor AdjustScatterDims(std::vector<std::size_t> scatter_indices_shape,
                                 poplar::Tensor updates,
                                 unsigned index_vector_dim) {
  unsigned rank = scatter_indices_shape.size();

  if (index_vector_dim < scatter_indices_shape.size()) {
    rank--;
  }

  auto shape = updates.shape();
  if (rank == 0) {
    shape.insert(shape.begin(), 1);

    // If there are no scatter dims, this must be a dynamic-update-slice kind of
    // scatter. In this case, we prepend a degenerate dimension to work
    // uniformly in the while loop.
    return updates.reshape(shape);
  }

  auto begin = std::begin(shape);
  auto collapse = std::next(std::begin(shape), rank);
  auto end = std::end(shape);

  std::vector<std::size_t> new_shape;
  new_shape.push_back(
      std::accumulate(begin, collapse, 1, std::multiplies<std::size_t>()));
  new_shape.insert(std::end(new_shape), collapse, end);

  return updates.reshape(new_shape);
}

enum class UpdateMode { Replace, Accumulate };
void MultiUpdateInternal(poplar::Graph& graph, poplar::Tensor operand,
                         const poplar::Tensor& indices,
                         const poplar::Tensor& updates,
                         std::size_t index_vector_dim,
                         std::vector<unsigned> update_window_dims,
                         poplar::program::Sequence& prog,
                         const std::string& debug_prefix, UpdateMode mode) {
  // If the updates tensor is empty, there is no need to update the operand. We
  // can return the operand as is.
  if (updates.numElements() == 0) {
    return;
  }

  // Canonicalize the scatter_indices, after which the size of its most-major
  // dimension must be same as the while loop trip count.
  poplar::Tensor canonical_scatter_iIndices =
      CanonicalizeScatterIndices(indices, index_vector_dim);

  // Canonicalize the updates, after which the size of its most-major dimension
  // must be same as the while loop trip count.
  poplar::Tensor canonical_updates =
      PermuteScatterAndWindowDims(updates, update_window_dims);
  poplar::Tensor adjusted_canonical_updates =
      AdjustScatterDims(indices.shape(), canonical_updates, index_vector_dim)
          .expand({1});

  // Since we can assume we are in the 2D case, the only possible solution is
  // transpose.
  if (update_window_dims[0] == 0) {
    operand = operand.transpose();
  }

  if (mode == UpdateMode::Replace) {
    popops::multiUpdate(
        graph, operand, adjusted_canonical_updates,
        canonical_scatter_iIndices.reinterpret(poplar::UNSIGNED_INT), {0}, {1},
        prog, debug_prefix);
  } else {
    poplar::Tensor scale = graph.addConstant(operand.elementType(), {}, 1,
                                             debug_prefix + "/const_1_scale");
    graph.setTileMapping(scale, 0);
    popops::multiUpdateAdd(
        graph, operand, adjusted_canonical_updates,
        canonical_scatter_iIndices.reinterpret(poplar::UNSIGNED_INT), scale,
        {0}, {1}, prog, debug_prefix);
  }
}

bool CheckValidAttributes(const HloScatterInstruction* inst) {
  const auto dim_numbers = inst->scatter_dimension_numbers();
  const auto update_window_dims = dim_numbers.update_window_dims();
  const auto inserted_window_dims = dim_numbers.inserted_window_dims();
  const auto scatter_dims_to_operand_dims =
      dim_numbers.scatter_dims_to_operand_dims();

  return (inst->operand(0)->shape().rank() != 2) ||
         (inst->operand(2)->shape().rank() != 2) ||
         (scatter_dims_to_operand_dims.size() != 1) ||
         (inserted_window_dims.size() != 1 || (update_window_dims.size()) != 1);
}
}  // namespace

StatusOr<poplar::program::Program> CreateMultiUpdate(
    CompilerResources& res, const HloScatterInstruction* inst,
    TensorMap& tensor_map) {
  const auto dim_numbers = inst->scatter_dimension_numbers();
  const auto update_window_dims = dim_numbers.update_window_dims();
  const auto index_vector_dim = dim_numbers.index_vector_dim();

  // TODO popops::multiUpdate and popops::multiUpdateAdd only supports the 2D
  // case. Fallback to scatter.
  if (CheckValidAttributes(inst)) {
    return CreateScatter(res, inst, tensor_map);
  }

  poplar::program::Sequence prog;
  poplar::Graph& graph = GetGraph(res, inst);

  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
                      FindInplaceOutputTensors(tensor_map, res, inst, prog));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor operand = inputs[0][0];

  TF_ASSIGN_OR_RETURN(poplar::Tensor indices,
                      FindInstructionInput(tensor_map, res, inst, 1, prog));

  TF_ASSIGN_OR_RETURN(poplar::Tensor updates,
                      FindInstructionInput(tensor_map, res, inst, 2, prog));

  VLOG(1) << "Processing " << inst->name() << " as multiUpdate";

  MultiUpdateInternal(graph, operand, indices, updates, index_vector_dim,
                      {update_window_dims.begin(), update_window_dims.end()},
                      prog, GetDebugName(inst), UpdateMode::Replace);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand));

  return prog;
}

StatusOr<poplar::program::Program> CreateMultiUpdateAdd(
    CompilerResources& res, const HloScatterInstruction* inst,
    TensorMap& tensor_map) {
  const auto dim_numbers = inst->scatter_dimension_numbers();
  const auto update_window_dims = dim_numbers.update_window_dims();
  const auto index_vector_dim = dim_numbers.index_vector_dim();

  // TODO popops::multiUpdate and popops::multiUpdateAdd only supports the 2D
  // case. Fallback to scatter.
  if (CheckValidAttributes(inst)) {
    return CreateScatter(res, inst, tensor_map);
  }

  poplar::program::Sequence prog;
  poplar::Graph& graph = GetGraph(res, inst);

  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
                      FindInplaceOutputTensors(tensor_map, res, inst, prog));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor operand = inputs[0][0];

  TF_ASSIGN_OR_RETURN(poplar::Tensor indices,
                      FindInstructionInput(tensor_map, res, inst, 1, prog));

  TF_ASSIGN_OR_RETURN(poplar::Tensor updates,
                      FindInstructionInput(tensor_map, res, inst, 2, prog));

  VLOG(1) << "Processing " << inst->name() << " as multiUpdateAdd";

  MultiUpdateInternal(graph, operand, indices, updates, index_vector_dim,
                      {update_window_dims.begin(), update_window_dims.end()},
                      prog, GetDebugName(inst), UpdateMode::Accumulate);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand));

  return prog;
}

}  // namespace poplarplugin
}  // namespace xla
