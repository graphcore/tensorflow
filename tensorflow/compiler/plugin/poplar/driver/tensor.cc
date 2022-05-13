/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"

#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include <poplar/OptionFlags.hpp>
#include <popops/HostSliceTensor.hpp>
#include <poputil/TileMapping.hpp>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/driver_types.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conversions.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/mapping_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matmul_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/ml_type_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/deferred_visitor.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/bcast.h"
#include "tensorflow/stream_executor/lib/status.h"

using ::absl::StrCat;
using ::tensorflow::str_util::Join;

namespace xla {
namespace poplarplugin {
namespace {
// Adds a tensor which is linearly mapped across the tiles.
DriverTensor AddLinearlyMappedTensor(
    DriverGraph& graph, const poplar::Type poplar_type,
    const std::vector<std::size_t>& shape,
    const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Allocating a linearly mapped tensor "
          << debug_name_and_id.getPathName() << " "
          << absl::StrJoin(shape, ", ");
  DriverTensor out = graph.addVariable(poplar_type, shape, {debug_name_and_id});
  poputil::mapTensorLinearly(graph, out);
  return out;
}

// Adds a tensor which is linearly mapped across the tiles, however the starting
// tile depends on previous allocations.
DriverTensor AddLinearlyMappedTensorWithOffset(
    DriverGraph& graph, const poplar::Type poplar_type,
    const std::vector<std::size_t>& shape,
    const poplar::DebugNameAndId& debug_name_and_id,
    CompilerResources& resources) {
  VLOG(1) << "Allocating a linearly mapped tensor with an offset "
          << debug_name_and_id.getPathName() << " "
          << absl::StrJoin(shape, ", ");
  DriverTensor out = graph.addVariable(poplar_type, shape, {debug_name_and_id});
  MappingHelper::MapTensorLinearly(resources.linear_mapping_state, graph, out);
  return out;
}

bool ShouldRebalanceTensor(DriverGraph& graph, const DriverTensor& tensor) {
  // Do not rebalance if the tensor doesn't have aliasing.
  if (tensor.isParallelWriteable()) {
    return false;
  }

  // Rebalance if the tensor has more elements than there are tiles.
  return tensor.numElements() > graph.getTarget().getNumTiles();
}

DriverTensor RebalanceTensorIfRequired(
    DriverGraph& graph, CompilerResources& res, poplar::program::Sequence& seq,
    const DriverTensor& tensor, const poplar::DebugNameAndId& debug_name_and_id,
    bool always_add_copy = false) {
  DriverTensor rebalanced_tensor = tensor;
  bool rebalance =
      always_add_copy ? true : ShouldRebalanceTensor(graph, tensor);

  if (rebalance) {
    rebalanced_tensor = TensorCloneAndRebalanceAliasing(graph, res, tensor,
                                                        {debug_name_and_id});
    seq.add(poplar::program::Copy(tensor, rebalanced_tensor, false,
                                  {debug_name_and_id}));
  }
  return rebalanced_tensor;
}

TensorOrRemoteBufferVector GetTensorsMaybeExpand(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    poplar::program::Sequence& seq, bool expand_aliasing,
    const poplar::DebugNameAndId& debug_name_and_id,
    absl::optional<int64> opt_tensors_start = absl::nullopt,
    absl::optional<int64> opt_tensors_end = absl::nullopt) {
  // Allocate any tensors which were deferred.
  DeferredAllocations::AllocateIfExists(res, inst, opt_tensors_start,
                                        opt_tensors_end);

  TensorMap::NamedTensorLocationVector tensor_vector =
      map.FindInstructionNamedTensorLocations(inst, opt_tensors_start,
                                              opt_tensors_end);
  TensorOrRemoteBufferVector outputs;
  for (auto tensor : tensor_vector) {
    CHECK(tensor.tensor.IsTensor() || tensor.tensor.IsRemoteBuffer() ||
          tensor.tensor.IsOpaque());
    if (tensor.tensor.IsTensor()) {
      if (expand_aliasing) {
        auto& graph = GetGraphWithOutputIndex(
            res, inst, tensor.location.flattened_output_tuple_index);
        tensor.tensor = RebalanceTensorIfRequired(
            graph, res, seq, tensor.tensor, debug_name_and_id);
      }
      TF_CHECK_OK(map.UpdateTensor(tensor.location, tensor.tensor));
    }
    outputs.push_back(tensor.tensor);
  }
  return outputs;
}

}  // namespace

StatusOr<poplar::Type> PoplarDataType(const xla::PrimitiveType& element_type) {
  switch (element_type) {
    case PRED:
      return poplar::BOOL;
    case S8:
      return poplar::SIGNED_CHAR;
    case U8:
      return poplar::UNSIGNED_CHAR;
    case S16:
      return poplar::SHORT;
    case S32:
    case S64:
      return poplar::INT;
    case U32:
    case U64:
      return poplar::UNSIGNED_INT;
    case F16:
      return poplar::HALF;
    case F32:
      return poplar::FLOAT;
    default:
      return xla::FailedPrecondition("unsupported primitive type in Poplar %s",
                                     PrimitiveType_Name(element_type));
  }
}

StatusOr<poplar::Type> PoplarDataType(const xla::Shape& shape) {
  return PoplarDataType(shape.element_type());
}

std::vector<size_t> PoplarShapeFromXlaShape(const xla::Shape& xla_shape) {
  std::vector<size_t> shape;
  for (auto d : xla_shape.dimensions()) {
    shape.push_back(d);
  }
  return shape;
}

// Concatenate all tensors into a single tensor.
DriverTensor ConcatenateTensors(const std::vector<DriverTensor>& tensors,
                                int64 dimension) {
  std::vector<snap::Tensor> snap_tensors(tensors.size());
  absl::c_copy(tensors, snap_tensors.begin());
  auto ret = snap::concat(snap_tensors, dimension);
  return ret;
}

DriverTensor FlattenAndConcatenateTensors(
    const std::vector<DriverTensor>& tensors) {
  std::vector<DriverTensor> flat_tensors(tensors.size());
  absl::c_transform(
      tensors, flat_tensors.begin(),
      [&](const DriverTensor& tensor) { return tensor.flatten(); });
  auto ret = ConcatenateTensors(flat_tensors);
  return ret;
}

StatusOr<DriverTensor> SliceTensor(
    DriverTensor tensor_to_slice,
    const HloInstruction::InstructionVector& slices, int64 slice_index,
    int64 dimension) {
  size_t offset = 0;
  for (auto slice : slices) {
    if (!slice->shape().IsArray()) {
      return xla::FailedPrecondition("SliceTensor - Shape not supported.");
    }
    const size_t tensor_size = slice->shape().dimensions(dimension);
    if (slice_index == 0) {
      return tensor_to_slice.slice(offset, offset + tensor_size, dimension);
    }
    offset += tensor_size;
    slice_index--;
  }
  return xla::InvalidArgument("Slice index out of bounds");
}

std::vector<DriverTensor> SliceTensorIntoTensorsLike(
    DriverTensor tensor_to_slice,
    const std::vector<DriverTensor>& like_tensors) {
  std::vector<DriverTensor> output_tensors(like_tensors.size());
  for (size_t i = 0; i < like_tensors.size(); ++i) {
    auto tensor = like_tensors[i];
    auto output_tensor = tensor_to_slice.slice(0, tensor.numElements(), 0);
    tensor_to_slice = tensor_to_slice.slice(tensor.numElements(),
                                            tensor_to_slice.numElements(), 0);
    output_tensors[i] = output_tensor.reshape(tensor.shape());
  }
  return output_tensors;
}

xla::Shape XlaShapeFromPoplarShape(PrimitiveType element_type,
                                   const std::vector<size_t>& poplar_shape) {
  xla::Shape shape;
  shape.set_element_type(element_type);
  for (int64 dimension : poplar_shape) {
    shape.add_dimensions(dimension);
  }
  LayoutUtil::SetToDefaultLayout(&shape);
  return shape;
}

DriverTensor ConvertToDeviceLayout(const Shape& shape,
                                   const DriverTensor& tensor) {
  // Reshape then dimshuffle
  DriverTensor out = tensor;
  if (!LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    unsigned int rank = tensor.rank();
    std::vector<std::size_t> dim(rank);
    std::vector<unsigned int> shuffle(rank);
    for (unsigned int i = 0; i < rank; i++) {
      shuffle[shape.layout().minor_to_major(i)] = rank - i - 1;
      dim[rank - i - 1] = tensor.dim(shape.layout().minor_to_major(i));
    }

    out = out.reshape(dim);
    out = out.dimShuffle(shuffle);
  }
  return out;
}

StatusOr<DriverTensor> CreateIndicesTensor(
    DriverGraph& graph, const popops::SlicePlan& plan,
    const xla::Shape& xla_indices_shape,
    const poplar::DebugNameAndId& debug_name_and_id) {
  std::vector<size_t> indices_shape =
      PoplarShapeFromXlaShape(xla_indices_shape);
  TF_ASSIGN_OR_RETURN(poplar::Type indices_type,
                      PoplarDataType(xla_indices_shape));
  const auto num_indices =
      std::accumulate(indices_shape.begin(), indices_shape.end(),
                      std::size_t(1), std::multiplies<std::size_t>());
  return DriverTensor(popops::createIndicesTensor(graph, {0}, num_indices, plan,
                                                  {}, {debug_name_and_id})
                          .reshape(indices_shape)
                          .reinterpret(indices_type),
                      graph);
}

StatusOr<DriverTensor> AddHostCopyTensor(
    DriverGraph& graph, const poplar::DebugNameAndId& debug_name_and_id,
    const xla::Shape& shape) {
  TF_ASSIGN_OR_RETURN(poplar::Type poplar_type, PoplarDataType(shape));
  const std::vector<std::size_t> poplar_shape = PoplarShapeFromXlaShape(shape);

  // Passing isRead=false gives better tile balance.
  return DriverTensor(popops::createHostTransferableTensor(
                          graph, poplar_type, poplar_shape,
                          /*isRead=*/false, {debug_name_and_id}),
                      graph);
}

DriverTensor ConvertFromDeviceLayout(const Shape& shape,
                                     const DriverTensor& tensor) {
  // Dimshuffle then reshape
  DriverTensor out = tensor;
  if (!LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    unsigned int rank = tensor.rank();
    std::vector<unsigned int> shuffle(rank);
    for (unsigned int i = 0; i < rank; i++) {
      shuffle[rank - i - 1] = shape.layout().minor_to_major(i);
    }
    out = out.dimShuffle(shuffle);
    out = out.reshape(tensor.shape());
  }
  return out;
}

StatusOr<DriverTensor> AddPlainTensor(DriverGraph& graph,
                                      const poplar::DebugContext& debug_context,
                                      const xla::Shape& shape,
                                      CompilerResources& resources,
                                      bool offset) {
  PoplarOpDefDebugInfo debug_info(debug_context, "AddPlainTensor");

  std::vector<std::size_t> dim = PoplarShapeFromXlaShape(shape);
  TF_ASSIGN_OR_RETURN(poplar::Type poplar_type, PoplarDataType(shape));
  if (offset) {
    return AddLinearlyMappedTensorWithOffset(graph, poplar_type, dim,
                                             {debug_info}, resources);
  } else {
    return AddLinearlyMappedTensor(graph, poplar_type, dim, {debug_info});
  }
}

StatusOr<DriverTensor> AddIndicesTensor(
    DriverGraph& graph, const poplar::DebugNameAndId& debug_name_and_id,
    const xla::Shape& shape, CompilerResources& resources) {
  return CreateIndicesTensor(graph, popops::SlicePlan(), shape,
                             {debug_name_and_id});
}

template <typename IIter1, typename IIter2, typename OIter, typename Zipper>
static void zip(IIter1 ibegin1, IIter1 iend1, IIter2 ibegin2, OIter obegin,
                Zipper zipper) {
  for (; ibegin1 != iend1; ++ibegin1, ++ibegin2, ++obegin) {
    *obegin = zipper(*ibegin1, *ibegin2);
  }
}

// Find a value for G s.t. D / G <= T, and G | D.
static StatusOr<std::size_t> FindG(const std::size_t D, const std::size_t T) {
  for (std::size_t g = (D + T - 1) / T; g <= D; ++g) {
    if (D % g == 0) {
      return g;
    }
  }

  return tensorflow::errors::FailedPrecondition(
      "Cannot find a value of G that is both a factor of D and satisfies D / G "
      "<= T");
}

// Find the sequence dimension, if there is one
static StatusOr<std::size_t> FindSeqDim(const xla::Shape& shape_xla,
                                        const xla::Shape& slice_shape_xla) {
  const auto shape = PoplarShapeFromXlaShape(shape_xla);
  const auto slice_shape = PoplarShapeFromXlaShape(slice_shape_xla);
  const auto volume =
      std::accumulate(shape.begin(), shape.end(), std::size_t(1),
                      std::multiplies<std::size_t>());
  const auto slice_volume =
      std::accumulate(slice_shape.begin(), slice_shape.end(), std::size_t(1),
                      std::multiplies<std::size_t>());

  // If the desired shape is 1D, then no special work is required.
  // If the slice shape is the same as the input shape, this is just a copy
  if (shape_xla.rank() > 1 && shape != slice_shape && volume > 1 &&
      slice_volume > 1) {
    // Calculate the element-wise ratio between the slice the input rank
    std::vector<float> dimension_ratios(shape.size());
    zip(slice_shape.begin(), slice_shape.end(), shape.begin(),
        dimension_ratios.begin(), std::divides<float>());

    // Assumes the sequence dimension is the dimension with the smallest ratio
    // between the input and the slice.
    return std::distance(
        dimension_ratios.begin(),
        std::min_element(dimension_ratios.begin(), dimension_ratios.end()));
  }

  return tensorflow::errors::FailedPrecondition(
      "Cannot compute slice sequence dimension");
}

namespace {
int64 HighestFactorLessOrEqualThanN(int64 to_factor, int64 n) {
  CHECK_GE(to_factor, n);
  int64 factor = n;
  while (to_factor % factor) {
    factor--;
  }
  return factor;
}
}  // namespace

DriverTensor CreateTensorFromSlice(
    DriverGraph& graph, const DriverTensor& slice, int64 slice_dimension,
    int64 output_size, CompilerResources& resources,
    const poplar::DebugNameAndId& debug_name_and_id) {
  const int64 slice_size = slice.dim(slice_dimension);
  // Find the higher number of rows we can slice out.
  const int64 rows_to_clone =
      HighestFactorLessOrEqualThanN(output_size, slice_size);

  // Slice out the right number of rows from the slice.
  DriverTensor adjusted_slice = slice;
  if (rows_to_clone != slice_size) {
    adjusted_slice = adjusted_slice.slice(0, rows_to_clone, slice_dimension);
  }

  // Calculate how many slices are required for the input tensor.
  const int64 number_of_clones = output_size / rows_to_clone;
  std::vector<DriverTensor> output_slices(number_of_clones);

  // Make a clone of the slice.
  output_slices[0] = TensorCloneAndRebalanceAliasing(
      graph, resources, adjusted_slice, {debug_name_and_id});

  // The remaining slices will just clone the layout of the first slice.
  for (int64 i = 1; i != number_of_clones; ++i) {
    output_slices[i] = graph.clone(output_slices[0], {debug_name_and_id});
  }

  // Concatentate all the slices into a single tensor.
  return ConcatenateTensors(output_slices, slice_dimension);
}

/**
 * Revert the change in tensor shape from a custom HLO Poplar instruction. This
 * is used in `xla::poplarplugin::PathTransform`.
 */
static StatusOr<DriverTensor> RevertPoplarCustomOpTensorTransformation(
    DriverGraph& graph, const HloInstruction* inst,
    CompilerResources& resources, DriverTensor in,
    const poplar::DebugNameAndId& debug_name_and_id) {
  const auto opcode = GetPoplarCustomOp(inst);

  if (!opcode.has_value()) {
    return in;
  }

  switch (opcode.value()) {
    case PoplarOp::StaticMultiSlice: {
      const auto slice_inst = Cast<HloStaticMultiSliceInstruction>(inst);

      // Start by getting the indices in the slice that correspond to
      // unique slices from the input tensor. For example, if the input
      // slice indices were `[3, 3, 2]`, then the unique indices in the
      // slice would be `[0, 2]` (the index `1` is omitted because it
      // corresponds to a duplicated slice from the input).
      const auto input_indices = slice_inst->GetIndices();
      std::vector<unsigned> unique_slice_indices;
      absl::flat_hash_set<int64> visited_input_indices;
      for (unsigned i = 0; i < input_indices.size(); i++) {
        if (!visited_input_indices.count(input_indices[i])) {
          unique_slice_indices.push_back(i);
          visited_input_indices.insert(input_indices[i]);
        }
      }

      // Modify the slice instruction output such that it only contains
      // unique slices.
      in = ConcatenateTensors(in.slices(unique_slice_indices), 0);

      // Finally, create the input tensor from the output slice.
      in = CreateTensorFromSlice(graph, in, 0,
                                 slice_inst->operand(0)->shape().dimensions(0),
                                 resources, {debug_name_and_id});

      break;
    }
    default: {
      // All other instructions in the path do not modify the shape
      break;
    }
  }

  return in;
}

static StatusOr<DriverTensor> PathTransform(
    DriverGraph& graph, const TensorLocation& source,
    CompilerResources& resources, DriverTensor in, int64 input_index,
    const std::vector<const HloInstruction*>& path,
    const poplar::DebugNameAndId& debug_name_and_id) {
  // Now revert any transformations required by the path from the source to
  // the target

  for (auto i = path.rbegin(); i != path.rend(); ++i) {
    const HloInstruction* inst = *i;
    switch (inst->opcode()) {
      case HloOpcode::kTranspose: {
        auto optional_permutation =
            convert_array<std::vector<unsigned>>(inst->dimensions());
        if (!optional_permutation) {
          return xla::FailedPrecondition(
              "PathTransform - cannot cast permutation.");
        }
        std::vector<unsigned> permutation = *optional_permutation;
        std::vector<unsigned> shuffle(permutation.size());
        for (unsigned int d = 0; d < permutation.size(); d++) {
          shuffle[permutation[d]] = d;
        }
        in = in.dimShuffle(shuffle);
        break;
      }
      case HloOpcode::kReshape: {
        std::vector<size_t> dims(
            PoplarShapeFromXlaShape(inst->operand(0)->shape()));
        in = in.reshape(dims);
        break;
      }
      case HloOpcode::kConvert: {
        TF_ASSIGN_OR_RETURN(auto poplar_type,
                            PoplarDataType(inst->operand(0)->shape()));
        in = graph.clone(poplar_type, in, {debug_name_and_id});
        break;
      }
      case HloOpcode::kConcatenate: {
        const HloInstruction* operand =
            i + 1 == path.rend() ? source.instruction : *(i + 1);
        const int64 slice_index = inst->operand_index(operand);
        TF_ASSIGN_OR_RETURN(in, SliceTensor(in, inst->operands(), slice_index,
                                            inst->concatenate_dimension()));
        TF_ASSIGN_OR_RETURN(auto poplar_type,
                            PoplarDataType(inst->operand(0)->shape()));
        in = graph.clone(poplar_type, in, {debug_name_and_id});
        break;
      };
      case HloOpcode::kPad: {
        TF_ASSIGN_OR_RETURN(in, UnpadTensor(inst->padding_config(), in));
        break;
      }
      case HloOpcode::kFusion: {
        if (IsPopOpsFusion(inst, "zero_pad")) {
          TF_ASSIGN_OR_RETURN(in,
                              UnpadTensor(inst->fused_instructions_computation()
                                              ->root_instruction()
                                              ->padding_config(),
                                          in));
        }
        break;
      }
      case HloOpcode::kSlice: {
        std::vector<int64> slice_dimensions;
        for (int64 i = 0; i != inst->shape().rank(); ++i) {
          if (inst->shape().dimensions(i) !=
              inst->operand(0)->shape().dimensions(i)) {
            slice_dimensions.push_back(i);
          }
        }
        if (slice_dimensions.size() != 1) {
          return InternalErrorStrCat(
              "Expected to allocate for a slice operation in a single "
              "dimension, instead got a slice in ",
              slice_dimensions.size(), " dimensions.");
        }
        const int64 slice_dimension = slice_dimensions[0];
        in = CreateTensorFromSlice(
            graph, in, slice_dimension,
            inst->operand(0)->shape().dimensions(slice_dimension), resources,
            {debug_name_and_id});
        break;
      }
      case HloOpcode::kReduce: {
        std::vector<size_t> slice_dims(inst->dimensions().begin(),
                                       inst->dimensions().end());
        // Sort to ensure we expand the tensor at the slice dims in order.
        absl::c_sort(slice_dims);

        std::vector<size_t> slice_dim_sizes;
        slice_dim_sizes.reserve(slice_dims.size());
        for (size_t slice_dim : slice_dims) {
          slice_dim_sizes.push_back(
              inst->operand(0)->shape().dimensions(slice_dim));
          // Expand the tensor in the reduce dimensions so it matches the rank
          // of the input (resembling a slice of the input).
          in = in.expand({slice_dim});
        }

        // Broadcast the layout and tile mapping across the reduce dimensions.
        in = DriverTensor(
            popops::createSliceableTensorFromSlice(
                graph, in, slice_dims, slice_dim_sizes, {debug_name_and_id}),
            graph);
        break;
      }
      case HloOpcode::kCustomCall: {
        TF_ASSIGN_OR_RETURN(in,
                            RevertPoplarCustomOpTensorTransformation(
                                graph, inst, resources, in, debug_name_and_id));
        break;
      }
      default: {
        // All other instructions in the path do not modify the shape
        break;
      }
    }
  }

  return in;
}

StatusOr<DriverTensor> AddDynamicSliceTensor(
    DriverGraph& graph, const poplar::DebugNameAndId& debug_name_and_id,
    const xla::Shape& shape_xla, const xla::Shape& slice_shape_xla) {
  const SliceInfo slice_info = GetSliceInfo(shape_xla, slice_shape_xla);

  if (slice_info.sliced_dims.size() == static_cast<size_t>(shape_xla.rank())) {
    // Use the old dynamic slice allocator when we are slicing in all
    // dimensions.
    // TODO Remove this special case once T8594 is fixed.
    DriverTensor unused;
    return AddDynamicSliceTensor(graph, debug_name_and_id, shape_xla,
                                 slice_shape_xla, unused);
  } else {
    TF_ASSIGN_OR_RETURN(auto poplar_type, PoplarDataType(shape_xla));
    const auto input_shape = PoplarShapeFromXlaShape(shape_xla);
    return DriverTensor(
        popops::createSliceableTensor(
            graph, poplar_type, input_shape, slice_info.sliced_dims,
            slice_info.slice_sizes, 0, {debug_name_and_id}),
        graph);
  }
}

StatusOr<DriverTensor> AddDynamicUpdateSliceTensor(
    DriverGraph& graph, const poplar::DebugNameAndId& debug_name_and_id,
    const xla::Shape& input_shape_xla, const xla::Shape& update_shape_xla) {
  const SliceInfo slice_info = GetSliceInfo(input_shape_xla, update_shape_xla);

  if (slice_info.sliced_dims.size() ==
      static_cast<size_t>(update_shape_xla.rank())) {
    // Use the old dynamic slice allocator when we are slicing in all
    // dimensions.
    // TODO Remove this special case once T8594 is fixed.
    DriverTensor unused;
    return AddDynamicSliceTensor(graph, {debug_name_and_id}, update_shape_xla,
                                 update_shape_xla, unused);
  } else {
    TF_ASSIGN_OR_RETURN(auto poplar_type, PoplarDataType(update_shape_xla));
    const auto update_shape = PoplarShapeFromXlaShape(update_shape_xla);
    return DriverTensor(
        popops::createSliceableTensor(
            graph, poplar_type, update_shape, slice_info.sliced_dims,
            slice_info.slice_sizes, 0, {debug_name_and_id}),
        graph);
  }
}

StatusOr<DriverTensor> AddDynamicSliceTensor(
    DriverGraph& graph, const poplar::DebugNameAndId& debug_name_and_id,
    const xla::Shape& shape_xla, const xla::Shape& slice_shape_xla,
    DriverTensor& physical_layout) {
  const auto shape = PoplarShapeFromXlaShape(shape_xla);
  TF_ASSIGN_OR_RETURN(poplar::Type poplar_type, PoplarDataType(shape_xla));
  const auto volume =
      std::accumulate(shape.begin(), shape.end(), std::size_t(1),
                      std::multiplies<std::size_t>());

  // If we are able to compute the sequence_dimension
  const auto sequence_dimension_status = FindSeqDim(shape_xla, slice_shape_xla);
  if (!sequence_dimension_status.ok()) {
    physical_layout =
        AddLinearlyMappedTensor(graph, poplar_type, shape, {debug_name_and_id});
    return physical_layout;
  }

  const auto sequence_dimension = sequence_dimension_status.ValueOrDie();

  // Create a tensor of the form [D/G, S, G] where D is the product of the N-1
  // dimensions that are not the sequence dimension, S is the size of the
  // sequence dimension, and G is a factor of D chosen to ensure that
  // D/G <= T, where T is the number of tiles.
  const auto T = graph.getTarget().getNumTiles();
  const auto D = volume / shape[sequence_dimension];
  const auto S = shape[sequence_dimension];
  const auto G_status = FindG(D, T);
  if (!G_status.ok()) {
    physical_layout =
        AddLinearlyMappedTensor(graph, poplar_type, shape, {debug_name_and_id});
    return physical_layout;
  }

  const auto G = G_status.ValueOrDie();
  if (D == G) {
    physical_layout =
        AddLinearlyMappedTensor(graph, poplar_type, shape, {debug_name_and_id});
    return physical_layout;
  }

  // If a value for G was found
  DriverTensor out = graph.addVariable(poplar_type, {D / G, S, G},
                                       poplar::DebugContext{debug_name_and_id});
  physical_layout = out;

  // Map the sequence dimension across the tiles
  for (std::size_t i = 0; i < out.dim(0); ++i) {
    graph.setTileMapping(out[i], i);
  }

  // Reshape, with the sequence dimension being the last dimension
  auto shape_tmp = shape;
  std::swap(shape_tmp[sequence_dimension], shape_tmp.back());
  out = out.reshape(shape_tmp);

  // Shuffle the dimensions back into the desired order
  // out.dimSwap(sequence_dimension, shape.size() - 1)
  std::vector<unsigned> permutation(shape.size());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation[sequence_dimension], permutation.back());
  out = out.dimShuffle(permutation);

  return out;
}

StatusOr<DriverTensor> AddScatterTensor(
    DriverGraph& graph, const poplar::DebugNameAndId& debug_name_and_id,
    const xla::Shape& shape_xla, const xla::Shape& slice_shape_xla) {
  return AddDynamicSliceTensor(graph, debug_name_and_id, shape_xla,
                               slice_shape_xla);
}

static StatusOr<DriverTensor> AddLeftMatMul(
    DriverGraph& graph, CompilerResources& resources,
    const poplar::DebugNameAndId& debug_name_and_id,
    const TensorTarget& tensor_target, const xla::Shape& shape) {
  TF_ASSIGN_OR_RETURN(poplar::Type type, PoplarDataType(shape));
  const HloInstruction* target = tensor_target.tgt;
  auto dot_dims = target->dot_dimension_numbers();

  // Find the permutations
  std::vector<int64> permutations;
  Shape shuffled_shape, a_shape;

  // Collapse the LHS dimensions down to [Batch, M, Contracting]
  std::tie(a_shape, shuffled_shape, permutations) =
      LeftMatMulPrepare(target->operand(0)->shape(), dot_dims);
  auto b_shape = std::get<0>(RightMatMulPrepare(
      target->operand(1)->shape(), target->dot_dimension_numbers()));

  TF_ASSIGN_OR_RETURN(const poplar::OptionFlags opts,
                      GetMatMulOptionsForInst(target, resources));

  auto result = DriverTensor(
      poplin::createMatMulGroupedInputLHS(
          graph, type, type, PoplarShapeFromXlaShape(a_shape),
          PoplarShapeFromXlaShape(b_shape), {debug_name_and_id, "lhs"}, opts,
          &resources.planning_cache),
      graph);

  // Unpack matrix
  result = result.reshape(PoplarShapeFromXlaShape(shuffled_shape));
  // Permute the matrix dimensions back to the XLA shape
  // Note: the permutations vector was generated for an XLA shape
  // therefore it is already inverted.
  result = result.dimShuffle(ToUnsignedVector(permutations));
  return result;
}

static StatusOr<DriverTensor> AddRightMatMul(
    DriverGraph& graph, CompilerResources& resources,
    const poplar::DebugNameAndId& debug_name_and_id,
    const TensorTarget& tensor_target, const xla::Shape& shape) {
  TF_ASSIGN_OR_RETURN(poplar::Type type, PoplarDataType(shape));
  const HloInstruction* target = tensor_target.tgt;
  auto dot_dims = target->dot_dimension_numbers();

  // Find the permutations
  std::vector<int64> permutations;
  Shape shuffled_shape, b_shape;

  // Collapse the LHS dimensions down to [Batch, Contracting, N]
  std::tie(b_shape, shuffled_shape, permutations) =
      RightMatMulPrepare(target->operand(1)->shape(), dot_dims);
  auto a_shape = std::get<0>(LeftMatMulPrepare(
      target->operand(0)->shape(), target->dot_dimension_numbers()));

  TF_ASSIGN_OR_RETURN(const poplar::OptionFlags opts,
                      GetMatMulOptionsForInst(target, resources));

  // Support RHS being sliceable on the output dimension.
  bool make_output_dimension_sliceable = false;
  if (tensor_target.permutation && tensor_target.sliceable_dimension) {
    const std::vector<int64> permutation = *tensor_target.permutation;
    const int64 sliceable_dimension = *tensor_target.sliceable_dimension;

    // Get the dimension in the output which is sliced.
    const int64 sliceable_output_dimension = permutation[sliceable_dimension];

    // We want the matmul to be sliceable if sliceable_output_dimension is one
    // of the output dimensions - the output dimensions are all the dimensions
    // after the batch and contracting dimensions.
    const int64 first_output_dimension =
        dot_dims.rhs_batch_dimensions_size() +
        dot_dims.rhs_contracting_dimensions_size();

    for (int64 d = first_output_dimension; d != permutations.size(); ++d) {
      make_output_dimension_sliceable |=
          permutations[d] == sliceable_output_dimension;
    }
  }

  DriverTensor result;
  std::vector<size_t> poplar_a_shape = PoplarShapeFromXlaShape(a_shape);
  std::vector<size_t> poplar_b_shape = PoplarShapeFromXlaShape(b_shape);
  if (make_output_dimension_sliceable) {
    VLOG(2) << "Allocating the tensor " << debug_name_and_id.getPathName()
            << " as sliceable.";
    // Swap the contracting dimension and output dimension on b.
    std::swap(poplar_b_shape[1], poplar_b_shape[2]);
    // Set the correct contracting dimension in a as well.
    poplar_a_shape[2] = poplar_b_shape[1];

    result = DriverTensor(
        poplin::createMatMulGroupedInputRHS(
            graph, type, type, poplar_a_shape, poplar_b_shape,
            {debug_name_and_id, "rhs"}, opts, &resources.planning_cache),
        graph);

    // Move the contracting dimension and output dimension into the right
    // locations.
    result = result.dimShuffle({0, 2, 1});
  } else {
    result = DriverTensor(
        poplin::createMatMulGroupedInputRHS(
            graph, type, type, poplar_a_shape, poplar_b_shape,
            {debug_name_and_id, "rhs"}, opts, &resources.planning_cache),
        graph);
  }

  // Unpack matrix
  result = result.reshape(PoplarShapeFromXlaShape(shuffled_shape));
  // Permute the matrix dimensions back to the XLA shape
  // Note: the permutations vector was generated for an XLA shape
  // therefore it is already inverted.
  result = result.dimShuffle(ToUnsignedVector(permutations));
  return result;
}

static StatusOr<DriverTensor> AddElementwiseBinary(
    DriverGraph& graph, CompilerResources& res,
    const poplar::DebugNameAndId& debug_name_and_id, const xla::Shape& shape,
    const HloInstruction* layout, uint64 layout_output_idx,
    const TensorMap& tensor_map) {
  TF_ASSIGN_OR_RETURN(TensorVector outputs,
                      FindInstructionOutputTensors(tensor_map, res, layout));

  if (layout_output_idx < 0 || outputs.size() <= layout_output_idx) {
    return xla::FailedPrecondition(
        "Elementwise %s layout input not found for %s", layout->name(),
        debug_name_and_id.getPathName());
  }

  DriverTensor other_side = outputs[layout_output_idx];

  TF_ASSIGN_OR_RETURN(auto poplar_type, PoplarDataType(shape));

  // Special case of handling the name of the input for AddElementwiseBinary
  // We are cloning a tensor `other_side` to be the input tensor
  // In this case it is more useful to keep the debug name of the `other_side`
  // tensor than the name of the HloInstruction (i.e. /arg_32).
  //
  // Instead of passing the debug_name_and_id to the following methods we
  // we pass an empty name and it will then use the name other `other_side`

  DriverTensor output =
      TensorCloneAndRebalanceAliasing(graph, res, other_side, {});

  if (output.elementType() != poplar_type) {
    output = graph.clone(poplar_type, output, {});
  }

  return output;
}

bool HasTensorAllocationTarget(const TensorLocation& src,
                               const CompilerResources& resources) {
  auto& tensor_allocation_map = resources.annotations.tensor_allocation_map;
  return tensor_allocation_map.find(src) != tensor_allocation_map.end();
}

StatusOr<DriverTensor> AddTensorForTarget(
    DriverGraph& graph, const TensorLocation& source,
    const TensorTarget& tensor_target, CompilerResources& resources,
    const TensorMap& tensor_map, const poplar::DebugContext& debug_context) {
  PoplarOpDefDebugInfo debug_info(debug_context, "AddTensorForTarget");

  const auto* target = tensor_target.tgt;
  const auto input_index = tensor_target.input_index;
  auto tshape = target->operand(input_index)->shape();
  const auto optional_layout = tensor_target.layout;
  const auto optional_layout_output_idx = tensor_target.layout_output_idx;
  VLOG(3) << "Allocation target " << target->ToString() << " index "
          << input_index;

  DriverTensor out;
  if (IsPopOpsElementwiseBinary(target)) {
    TF_ASSIGN_OR_RETURN(
        out, AddElementwiseBinary(graph, resources, {debug_info}, tshape,
                                  *optional_layout, *optional_layout_output_idx,
                                  tensor_map));
  } else {
    const bool is_poplar_custom_op = GetPoplarCustomOp(target).has_value();
    const bool is_hlo_op_with_allocator = HloOpManager::HasOp(target->opcode());

    if (is_poplar_custom_op) {
      TF_ASSIGN_OR_RETURN(
          out, AllocatePoplarOpTensor(graph, resources, {debug_info},
                                      tensor_target, tshape, tensor_map));
    } else if (is_hlo_op_with_allocator) {
      TF_ASSIGN_OR_RETURN(
          out, AllocateHloOpTensor(graph, resources, {debug_info},
                                   tensor_target, tshape, tensor_map));
    } else {
      const std::string error_msg =
          absl::StrCat("Invalid operand for tensor allocation on ",
                       debug_info.getPathName());
      switch (target->opcode()) {
        case HloOpcode::kDot: {
          switch (input_index) {
            case 0: {
              TF_ASSIGN_OR_RETURN(out,
                                  AddLeftMatMul(graph, resources, {debug_info},
                                                tensor_target, tshape));
              break;
            }
            case 1: {
              TF_ASSIGN_OR_RETURN(out,
                                  AddRightMatMul(graph, resources, {debug_info},
                                                 tensor_target, tshape));
              break;
            }
            default:
              return xla::FailedPrecondition("%s", error_msg);
          }
          break;
        }
        case HloOpcode::kDynamicSlice: {
          switch (input_index) {
            case 0: {
              TF_ASSIGN_OR_RETURN(
                  out, AddDynamicSliceTensor(graph, {debug_info}, tshape,
                                             target->shape()));
              break;
            }
            default:
              return xla::FailedPrecondition("%s", error_msg);
          }
          break;
        }
        case HloOpcode::kDynamicUpdateSlice: {
          switch (input_index) {
            case 0: {
              TF_ASSIGN_OR_RETURN(
                  out, AddDynamicSliceTensor(graph, {debug_info}, tshape,
                                             target->operand(1)->shape()));
              break;
            }
            case 1: {
              TF_ASSIGN_OR_RETURN(
                  out, AddDynamicUpdateSliceTensor(graph, {debug_info},
                                                   target->operand(0)->shape(),
                                                   tshape));
              break;
            }
            default:
              return xla::FailedPrecondition("%s", error_msg);
          }

          break;
        }
        default: { return xla::FailedPrecondition("%s", error_msg); }
      }
    }
  }

  TF_ASSIGN_OR_RETURN(out,
                      PathTransform(graph, source, resources, out, input_index,
                                    tensor_target.backward_path, {debug_info}));

  const auto shapes = FlattenedXlaShape(source.instruction->shape());
  TF_ASSIGN_OR_RETURN(
      auto src_type,
      PoplarDataType(shapes[source.flattened_output_tuple_index]));

  if (src_type != out.elementType()) {
    return graph.clone(src_type, out, {debug_info});
  }

  return out;
}

StatusOr<DriverTensor> AddTensor(DriverGraph& graph, const TensorLocation& src,
                                 const xla::Shape& shape,
                                 CompilerResources& resources,
                                 const TensorMap& tensor_map,
                                 const poplar::DebugContext& debug_context) {
  PoplarOpDefDebugInfo debug_info(debug_context, "AddTensor");
  DriverTensor out;

  auto itr = resources.annotations.tensor_allocation_map.find(src);
  if (itr != resources.annotations.tensor_allocation_map.end()) {
    VLOG(3) << "Adding a tensor with layout for ("
            << src.instruction->ToString() << ", "
            << src.flattened_output_tuple_index << ").";
    TF_ASSIGN_OR_RETURN(
        out, AddTensorForTarget(graph, src, itr->second, resources, tensor_map,
                                {debug_info}));

  } else {
    TF_ASSIGN_OR_RETURN(out,
                        AddPlainTensor(graph, {debug_info}, shape, resources));
  }
  return out;
}

namespace {
template <typename TYPE>
void SetInitialTensorValueImpl(DriverGraph& graph, DriverTensor& tensor,
                               const xla::Literal& literal) {
  const TYPE* data(static_cast<const TYPE*>(literal.untyped_data()));
  size_t element_count = literal.element_count();
  poplar::ArrayRef<TYPE> array(data, element_count);
  graph.setInitialValue<TYPE>(tensor, array);
}

void SetFp16InitialTensorValueImpl(DriverGraph& graph, DriverTensor& tensor,
                                   const xla::Literal& literal) {
  const uint16_t* data(static_cast<const uint16_t*>(literal.untyped_data()));
  size_t element_count = literal.element_count();
  poplar::ArrayRef<uint16_t> array(data, element_count);
  graph.setInitialValueHalf(tensor, array);
}

void Set64BitInitialTensorValueImpl(DriverGraph& graph, DriverTensor& tensor,
                                    const xla::Literal& literal) {
  size_t element_count = literal.element_count();
  const void* data(static_cast<const void*>(literal.untyped_data()));
  std::vector<char> converted =
      ConvInt64ToInt32(data, element_count * sizeof(int64), 0);

  int32* data32 = reinterpret_cast<int32*>(converted.data());
  poplar::ArrayRef<int32> array(data32, element_count);
  graph.setInitialValue<int>(tensor, array);
}
}  // namespace

DriverTensor TensorCloneAndRebalanceAliasing(
    DriverGraph& graph, CompilerResources& res, const DriverTensor& tensor,
    const poplar::DebugNameAndId& debug_name_and_id) {
  uint64 offset = res.linear_mapping_state[&graph.getPoplarGraph()];
  poplar::Tensor dst;
  std::tie(dst, offset) = poputil::cloneAndExpandAliasing(
      graph.getPoplarGraph(), tensor, offset, debug_name_and_id);
  res.linear_mapping_state[&graph.getPoplarGraph()] = offset;
  return {dst, graph};
}

Status SetInitialTensorValue(DriverGraph& graph, DriverTensor& tensor,
                             const xla::Literal& literal) {
  const auto type = literal.shape().element_type();
  switch (type) {
    case PRED:
      SetInitialTensorValueImpl<bool>(graph, tensor, literal);
      break;
    case S8:
      SetInitialTensorValueImpl<char>(graph, tensor, literal);
      break;
    case U8:
      SetInitialTensorValueImpl<unsigned char>(graph, tensor, literal);
      break;
    case S32:
      SetInitialTensorValueImpl<int>(graph, tensor, literal);
      break;
    case U32:
      SetInitialTensorValueImpl<unsigned>(graph, tensor, literal);
      break;
    case U64:
    case S64:
      Set64BitInitialTensorValueImpl(graph, tensor, literal);
      break;
    case F16:
      SetFp16InitialTensorValueImpl(graph, tensor, literal);
      break;
    case F32:
      SetInitialTensorValueImpl<float>(graph, tensor, literal);
      break;
    default:
      return xla::InternalErrorStrCat(
          StrCat("Unsupported type when calling SetInitialTensorValue ",
                 primitive_util::LowercasePrimitiveTypeName(type)));
  }
  return Status::OK();
}

namespace {

template <typename TYPE>
DriverTensor CreateConstantTensorImpl(
    DriverGraph& graph, const xla::Literal& literal, const xla::Shape& shape,
    const poplar::Type& type, const poplar::DebugNameAndId& debug_name_and_id) {
  int64 num_elements(ShapeUtil::ElementsIn(literal.shape()));
  std::vector<std::size_t> dim = PoplarShapeFromXlaShape(shape);
  const TYPE* data(static_cast<const TYPE*>(literal.untyped_data()));

  DriverTensor tensor;
  if (num_elements == 0) {
    tensor = graph.addConstant(type, {0}, (TYPE)0, {debug_name_and_id});
  } else if (num_elements == 1) {
    tensor = graph.addConstant(type, dim, data[0], {debug_name_and_id});
  } else {
    tensor = graph.addConstant(type, dim, data, {debug_name_and_id});
  }
  graph.setTileMapping(tensor, 0);

  return ConvertToDeviceLayout(literal.shape(), tensor);
}

DriverTensor CreateFp16ConstantTensorImpl(
    DriverGraph& graph, const xla::Literal& literal, const xla::Shape& shape,
    const poplar::Type& type, const poplar::DebugNameAndId& debug_name_and_id) {
  int64 num_elements(ShapeUtil::ElementsIn(literal.shape()));
  std::vector<std::size_t> dim = PoplarShapeFromXlaShape(shape);
  const uint16_t* data(static_cast<const uint16_t*>(literal.untyped_data()));

  DriverTensor tensor;
  if (num_elements == 0) {
    tensor = graph.addConstantHalf(type, {0}, static_cast<uint16_t>(0),
                                   {debug_name_and_id});
  } else if (num_elements == 1) {
    tensor = graph.addConstantHalf(type, dim, data[0], {debug_name_and_id});
  } else {
    tensor = graph.addConstantHalf(type, dim, data, {debug_name_and_id});
  }
  graph.setTileMapping(tensor, 0);

  return ConvertToDeviceLayout(literal.shape(), tensor);
}

DriverTensor Create64BitConstantTensorImpl(
    DriverGraph& graph, const xla::Literal& literal, const xla::Shape& shape,
    const poplar::Type& type, const poplar::DebugNameAndId& debug_name_and_id) {
  int64 num_elements(ShapeUtil::ElementsIn(literal.shape()));
  std::vector<std::size_t> dim = PoplarShapeFromXlaShape(shape);
  const void* data(static_cast<const void*>(literal.untyped_data()));

  std::vector<char> converted =
      ConvInt64ToInt32(data, num_elements * sizeof(int64), 0);

  const int32* data32 = reinterpret_cast<const int32*>(converted.data());

  DriverTensor tensor;
  if (num_elements == 0) {
    tensor = graph.addConstant(type, {0}, static_cast<int32>(0),
                               {debug_name_and_id});
  } else if (num_elements == 1) {
    tensor = graph.addConstant(type, dim, data32[0], {debug_name_and_id});
  } else {
    tensor = graph.addConstant(type, dim, data32, {debug_name_and_id});
  }
  graph.setTileMapping(tensor, 0);

  return ConvertToDeviceLayout(literal.shape(), tensor);
}
}  // namespace

StatusOr<DriverTensor> CreateConstantTensor(
    DriverGraph& graph, const xla::Literal& literal, const xla::Shape& shape,
    const poplar::Type& poplar_type,
    const poplar::DebugNameAndId& debug_name_and_id) {
  const auto type = literal.shape().element_type();
  switch (type) {
    case PRED:
      return CreateConstantTensorImpl<bool>(graph, literal, shape, poplar_type,
                                            debug_name_and_id);
    case S8:
      return CreateConstantTensorImpl<char>(graph, literal, shape, poplar_type,
                                            debug_name_and_id);
    case U8:
      return CreateConstantTensorImpl<unsigned char>(
          graph, literal, shape, poplar_type, debug_name_and_id);
    case S32:
      return CreateConstantTensorImpl<int>(graph, literal, shape, poplar_type,
                                           debug_name_and_id);
    case U32:
      return CreateConstantTensorImpl<unsigned>(graph, literal, shape,
                                                poplar_type, debug_name_and_id);
    case U64:
    case S64:
      return Create64BitConstantTensorImpl(graph, literal, shape, poplar_type,
                                           debug_name_and_id);
    case F16:
      return CreateFp16ConstantTensorImpl(graph, literal, shape, poplar_type,
                                          debug_name_and_id);
    case F32:
      return CreateConstantTensorImpl<float>(graph, literal, shape, poplar_type,
                                             debug_name_and_id);
    default:
      return xla::InternalErrorStrCat(
          StrCat("Unsupported type when calling CreateConstantTensor ",
                 primitive_util::LowercasePrimitiveTypeName(type)));
  }
}

StatusOr<DriverTensor> AddConstantTensor(
    DriverGraph& graph, const TensorLocation& src, const xla::Shape& shape,
    const xla::Literal& literal, CompilerResources& resources,
    const TensorMap& tensor_map, const poplar::DebugContext& debug_context) {
  PoplarOpDefDebugInfo debug_info(debug_context, "AddConstantTensor");

  if (HasTensorAllocationTarget(src, resources)) {
    auto tensor_target = resources.annotations.tensor_allocation_map.find(src);
    const auto* target = tensor_target->second.tgt;

    if (ShapeUtil::ElementsIn(target->shape()) ==
        ShapeUtil::ElementsIn(shape)) {
      TF_ASSIGN_OR_RETURN(
          DriverTensor tensor,
          AddTensor(graph, src, shape, resources, tensor_map, {debug_info}));
      TF_RETURN_IF_ERROR(SetInitialTensorValue(graph, tensor, literal));
      return ConvertToDeviceLayout(literal.shape(), tensor);
    }
  }

  if (ShapeUtil::ElementsIn(literal.shape()) > 32) {
    TF_ASSIGN_OR_RETURN(
        DriverTensor tensor,
        AddTensor(graph, src, shape, resources, tensor_map, {debug_info}));
    TF_RETURN_IF_ERROR(SetInitialTensorValue(graph, tensor, literal));
    return ConvertToDeviceLayout(literal.shape(), tensor);
  }

  TF_ASSIGN_OR_RETURN(poplar::Type type, PoplarDataType(literal.shape()));
  TF_ASSIGN_OR_RETURN(
      DriverTensor tensor,
      CreateConstantTensor(graph, literal, shape, type, {debug_info}));
  return tensor.reshape(PoplarShapeFromXlaShape(shape));
}

template <typename T>
DriverTensor TileTensor(const T& multiples, const DriverTensor& in) {
  DriverTensor out = in;
  for (unsigned d = 0; d < multiples.size(); d++) {
    int m = multiples[d];
    out = out.broadcast(m, d);
  }
  return out;
}

template DriverTensor TileTensor<tensorflow::BCast::Vec>(
    const tensorflow::BCast::Vec&, const DriverTensor&);

template DriverTensor TileTensor<std::vector<std::size_t>>(
    const std::vector<std::size_t>&, const DriverTensor&);

StatusOr<DriverTensor> UnpadTensor(const PaddingConfig& cfg,
                                   const DriverTensor& in) {
  DriverTensor out = in;
  for (unsigned d = 0; d < in.rank(); d++) {
    std::vector<std::size_t> shape(out.shape());
    // Remove front and back padding first:
    size_t begin = cfg.dimensions(d).edge_padding_low();
    if (static_cast<int64>(shape[d]) < cfg.dimensions(d).edge_padding_high()) {
      return xla::FailedPrecondition(
          "Edge %zu is larger than padded edge %zu for dimension %u",
          cfg.dimensions(d).edge_padding_high(), shape[d], d);
    }
    size_t end = shape[d] - cfg.dimensions(d).edge_padding_high();
    if (begin > end) {
      return xla::FailedPrecondition(
          "Unpadded end %zu is before beginning %zu for dimension %u", end,
          begin, d);
    }
    if (begin > 0 || end != shape[d]) {
      out = out.slice(begin, end, d);
      CHECK_EQ(end - begin, out.shape()[d]);
      shape[d] = end - begin;
    }

    // Remove interior padding:
    if (cfg.dimensions(d).interior_padding() > 0 && shape[d] > 0) {
      out = out.subSample(cfg.dimensions(d).interior_padding() + 1, d);
      CHECK_EQ(
          (out.shape()[d] - 1) * (cfg.dimensions(d).interior_padding() + 1) + 1,
          shape[d]);
    }
  }

  return out;
}

StatusOr<DriverTensor> PadTensor(const PaddingConfig& cfg,
                                 const DriverTensor& in,
                                 const DriverTensor& pad) {
  if (pad.numElements() != 1) {
    return xla::FailedPrecondition(
        "PadTensor: pad tensor is not single valued");
  }

  DriverTensor p(pad.reshape(std::vector<std::size_t>(in.rank(), 1)));

  DriverTensor out = in;
  for (unsigned d = 0; d < in.rank(); d++) {
    std::vector<std::size_t> shape(out.shape());

    if (cfg.dimensions(d).interior_padding() > 0 && shape[d] > 0) {
      shape[d] = cfg.dimensions(d).interior_padding();
      DriverTensor padded = TileTensor(shape, p);
      DriverTensor interleaved = out.slice(0, 1, d);
      for (unsigned int slice = 1; slice < out.dim(d); slice++) {
        interleaved = ConcatenateTensors({interleaved, padded}, d);
        interleaved = ConcatenateTensors(
            {interleaved, out.slice(slice, slice + 1, d)}, d);
      }
      out = interleaved;
    }

    if (cfg.dimensions(d).edge_padding_low() > 0) {
      shape[d] = cfg.dimensions(d).edge_padding_low();
      DriverTensor padded = TileTensor(shape, p);
      out = ConcatenateTensors({padded, out}, d);
    }

    if (cfg.dimensions(d).edge_padding_high() > 0) {
      shape[d] = cfg.dimensions(d).edge_padding_high();
      DriverTensor padded = TileTensor(shape, p);
      out = ConcatenateTensors({out, padded}, d);
    }
  }

  return out;
}

StatusOr<DriverTensor> ReverseTensor(const DriverTensor& in,
                                     const std::vector<int64>& dimensions) {
  DriverTensor out = in;
  if (in.numElements() > 0) {
    for (int64 d : dimensions) {
      out = out.reverse(d);
    }
  }
  return out;
}

StatusOr<DriverTensor> BroadcastTensor(const DriverTensor& in,
                                       const xla::Shape& out,
                                       const std::vector<int64>& dimensions) {
  if (PoplarShapeMatchesXLAShape(in, out)) {
    return in;
  }

  auto optional_bcast_shape =
      convert_array<tensorflow::BCast::Vec>(out.dimensions());
  if (!optional_bcast_shape) {
    return xla::FailedPrecondition(
        "BroadcastTensor - cannot cast output shape.");
  }
  tensorflow::BCast::Vec bcast_shape = *optional_bcast_shape;

  tensorflow::BCast::Vec tensor_shape(out.rank(), 1);
  if (dimensions.size() > 0) {
    for (size_t d = 0; d < dimensions.size(); d++) {
      tensor_shape[dimensions[d]] = in.dim(d);
    }
  } else {
    for (size_t d = 0; d < in.rank(); d++) {
      tensor_shape[d] = in.dim(d);
    }
  }

  tensorflow::BCast bcast(tensor_shape, bcast_shape);
  if (!bcast.IsValid()) {
    return xla::FailedPrecondition("Incompatible broadcast from (%s) to (%s)",
                                   Join(tensor_shape, ",").c_str(),
                                   Join(bcast_shape, ",").c_str());
  }

  DriverTensor o = in;
  auto optional_bcast_x_shape =
      convert_array<std::vector<size_t>>(bcast.x_reshape());
  if (!optional_bcast_x_shape) {
    return xla::FailedPrecondition(
        "BroadcastTensor - cannot cast broadcast shape.");
  }
  std::vector<size_t> bcast_x_shape = *optional_bcast_x_shape;
  o = in.reshape(bcast_x_shape);
  o = TileTensor(bcast.x_bcast(), o);
  return o.reshape(PoplarShapeFromXlaShape(out));
}

bool PoplarShapeMatchesXLAShape(const DriverTensor& tensor,
                                const xla::Shape& shape) {
  VLOG(5) << "Checking Tensor shape";
  if (tensor.rank() != shape.rank()) return false;
  for (size_t d = 0; d < tensor.rank(); d++) {
    if (tensor.dim(d) != (unsigned)shape.dimensions(d)) return false;
  }

  return true;
}

bool PoplarShapeMatchesXLAShape(TensorOrRemoteBuffer torb,
                                const xla::Shape& shape,
                                CompilerResources& resources) {
  VLOG(5) << "Checking TensorOrRemoteBuffer shape";
  if (torb.IsTensor()) {
    return PoplarShapeMatchesXLAShape(torb.AsTensor(), shape);
  }

  auto& remote_buffer = torb.AsRemoteBufferHolder();

  const std::size_t merged_element_count =
      remote_buffer.GetNumElements() * remote_buffer.GetRepeats();
  CHECK_GT(torb.NumMerged(), 0);
  CHECK_EQ(merged_element_count % torb.NumMerged(), 0);
  std::size_t element_count = merged_element_count / torb.NumMerged();

  if (torb.IsReplicaPartitioned()) {
    element_count *= resources.partition_replication_factor;

    // Check the remote buffer shape is correct, allowing for padding and CBR
    // extra elements.
    return element_count >= ShapeUtil::ElementsIn(shape);
  }

  // Check the remote buffer shape is correct.
  return element_count == ShapeUtil::ElementsIn(shape);
}

std::pair<int64, int64> FindTupleInputIndices(const HloInstruction* tuple,
                                              int64 n) {
  int64 start = 0;
  for (int64 i = 0; i < n; i++) {
    start += CountShapes(tuple->operand(i)->shape());
  }
  int64 end = start + CountShapes(tuple->operand(n)->shape());
  return std::make_pair(start, end);
}

std::pair<int64, int64> FindGetTupleElementTupleIndices(
    const HloInstruction* inst) {
  const auto* gte = Cast<HloGetTupleElementInstruction>(inst);
  const HloInstruction* tuple = inst->operand(0);
  const Shape& shape = tuple->shape();
  int64 start = 0;
  for (int64 i = 0; i < gte->tuple_index(); i++) {
    start += CountShapes(ShapeUtil::GetTupleElementShape(shape, i));
  }
  int64 end = start + CountShapes(ShapeUtil::GetTupleElementShape(
                          shape, gte->tuple_index()));
  return std::make_pair(start, end);
}

namespace {
StatusOr<TensorVector> TensorOrRemoteBufferVectorToTensorVector(
    const TensorOrRemoteBufferVector& tv, const std::string& context = {}) {
  TensorVector result;

  for (int i = 0; i < tv.size(); ++i) {
    if (!tv[i].IsTensor()) {
      return tensorflow::errors::FailedPrecondition(
          "Expected all members of the TensorOrRemoteBufferVector to be Poplar "
          "tensors, but input " +
              std::to_string(i) + " is not: ",
          context);
    }

    result.push_back(tv[i]);
  }

  return result;
}
}  // namespace

StatusOr<TensorVector> FindInstructionInputTensorsInRange(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    int64 input, std::pair<int64, int64> range, poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id, bool expand_aliasing) {
  const HloInstruction* operand = inst->operand(input);

  TensorOrRemoteBufferVector inputs =
      GetTensorsMaybeExpand(map, res, operand, seq, expand_aliasing,
                            debug_name_and_id, range.first, range.second);

  return TensorOrRemoteBufferVectorToTensorVector(
      inputs, "FindInstructionInputTensorsInRange on " + inst->name());
}

TensorOrRemoteBufferVector FindInstructionInputsInRange(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    int64 input, std::pair<int64, int64> range, poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id, bool expand_aliasing) {
  const HloInstruction* operand = inst->operand(input);

  return GetTensorsMaybeExpand(map, res, operand, seq, expand_aliasing,
                               debug_name_and_id, range.first, range.second);
}

StatusOr<DriverTensor> FindInstructionInput(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    int64 input, poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id, bool expand_aliasing) {
  const HloInstruction* operand = inst->operand(input);

  TensorOrRemoteBufferVector inputs = GetTensorsMaybeExpand(
      map, res, operand, seq, expand_aliasing, debug_name_and_id, 0, 1);

  if (inputs.size() == 0) {
    return tensorflow::errors::Unknown(
        StrCat("[Poplar] Couldn't find input ", input, " for ", inst->name()));
  }

  inputs.resize(1);

  TF_ASSIGN_OR_RETURN(TensorVector result,
                      TensorOrRemoteBufferVectorToTensorVector(
                          inputs, "FindInstructionInput on " + inst->name()));

  return result[0];
}

TensorOrRemoteBufferVector FindInstructionInputs(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    int64 input, poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id, bool expand_aliasing) {
  const HloInstruction* operand = inst->operand(input);
  return GetTensorsMaybeExpand(map, res, operand, seq, expand_aliasing,
                               debug_name_and_id);
}

StatusOr<TensorVector> FindInstructionInputTensors(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    int64 input, poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id, bool expand_aliasing) {
  const HloInstruction* operand = inst->operand(input);
  TensorOrRemoteBufferVector inputs = GetTensorsMaybeExpand(
      map, res, operand, seq, expand_aliasing, debug_name_and_id);

  return TensorOrRemoteBufferVectorToTensorVector(
      inputs, "FindInstructionInputTensors on " + inst->name());
}

TensorOrRemoteBufferVector FindInstructionOutputs(const TensorMap& map,
                                                  CompilerResources& res,
                                                  const HloInstruction* inst) {
  DeferredAllocations::AllocateIfExists(res, inst);
  return map.FindInstructionOutputs(inst);
}

StatusOr<TensorVector> FindInstructionOutputTensors(
    const TensorMap& map, CompilerResources& res, const HloInstruction* inst) {
  DeferredAllocations::AllocateIfExists(res, inst);
  return map.FindInstructionOutputTensors(inst);
}

StatusOr<TensorVector> FindInstructionOutputTensorsInRange(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    std::pair<int64, int64> range) {
  DeferredAllocations::AllocateIfExists(res, inst, range.first, range.second);
  return map.FindInstructionOutputTensors(inst, range.first, range.second);
}

StatusOr<TensorOrRemoteBufferVector> FindInstructionOutputsInRange(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    std::pair<int64, int64> range) {
  DeferredAllocations::AllocateIfExists(res, inst, range.first, range.second);
  return map.FindInstructionOutputs(inst, range.first, range.second);
}

StatusOr<TensorVector> FindExpandedInstructionOutputsInRange(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    std::pair<int64, int64> range, poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return TensorOrRemoteBufferVectorToTensorVector(
      GetTensorsMaybeExpand(map, res, inst, seq, true, debug_name_and_id,
                            range.first, range.second),
      "FindExpandedInstructionOutputsInRange on " + inst->name());
}

bool AreInplaceOutputTensorsWritable(TensorMap& map, CompilerResources& res,
                                     const HloInstruction* inst) {
  if (!IsLoweredInplace(inst)) {
    return false;
  }

  // Check that the instruction description is for an inplace read/write
  // operation.
  auto inplace_description = GetInplaceDescription(inst);
  if (inplace_description.GetType() != HloInstructionType::kInplaceReadWrite) {
    return false;
  }

  // Get all the input tensors for all the inplace operands
  auto inplace_indexes = inplace_description.GetInplaceOperandIndices();

  std::vector<TensorOrRemoteBufferVector> tensor_vectors(
      inplace_indexes.size());
  for (uint64 i = 0; i < inplace_indexes.size(); i++) {
    tensor_vectors[i] =
        FindInstructionOutputs(map, res, inst->operand(inplace_indexes[i]));
  }

  // Go through all the inplace tensors and check they are all parallel
  // writeable.
  for (auto tensor_vector : tensor_vectors) {
    for (auto tensor : tensor_vector) {
      if (tensor.IsTensor() && !tensor.AsTensor().isParallelWriteable()) {
        return false;
      }
    }
  }

  return true;
}

StatusOr<TensorVectors> FindInplaceOutputTensors(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id, bool expand_aliasing) {
  TF_ASSIGN_OR_RETURN(auto result_with_remote_buffers,
                      FindInplaceOutputs(map, res, inst, seq, debug_name_and_id,
                                         expand_aliasing));

  TensorVectors result;
  result.reserve(result_with_remote_buffers.size());

  for (auto& result_with_remote_buffer : result_with_remote_buffers) {
    TF_ASSIGN_OR_RETURN(auto tensor_vector,
                        TensorOrRemoteBufferVectorToTensorVector(
                            result_with_remote_buffer,
                            "FindInplaceOutputTensors on " + inst->name()));
    result.push_back(tensor_vector);
  }

  return result;
}

StatusOr<TensorOrRemoteBufferVectors> FindInplaceOutputs(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id, bool expand_aliasing) {
  // Check that the instruction description is for an inplace operation.
  auto inplace_description = GetInplaceDescription(inst);
  if (!inplace_description.IsInplaceType()) {
    LOG(FATAL) << "Trying to execute " << inst->name()
               << " as an inplace operation, but it is not.";
  }

  // Get all the input tensors for all the inplace operands
  auto inplace_indices = inplace_description.GetInplaceOperandIndices();
  const bool require_parallel_writeable =
      inplace_description.GetType() == HloInstructionType::kInplaceReadWrite &&
      !inplace_description.AllowNonInplaceLowering();

  TensorOrRemoteBufferVectors tensors(inplace_indices.size());
  for (uint64 i = 0; i < inplace_indices.size(); i++) {
    uint64 inplace_index = inplace_indices[i];
    tensors[i] = FindInstructionInputs(map, res, inst, inplace_index, seq,
                                       debug_name_and_id, expand_aliasing);
    for (int64 j = 0; j < tensors[i].size(); ++j) {
      if (tensors[i][j].IsTensor()) {
        DriverTensor tensor = tensors[i][j].AsTensor();
        if (require_parallel_writeable && !tensor.isParallelWriteable()) {
          DriverGraph& graph =
              GetGraphWithOutputIndex(res, inst->operand(inplace_index), j);
          DriverTensor out = TensorCloneAndRebalanceAliasing(
              graph, res, tensor, {debug_name_and_id, "clone"});
          seq.add(poplar::program::Copy(tensor, out, false, debug_name_and_id));
          tensors[i][j] = out;
        }
      }
    }
  }

  return tensors;
}

Status AddOutput(TensorMap& map, const HloInstruction* inst, int64 n,
                 const TensorOrRemoteBuffer& torb) {
  return map.AddOutput(inst, n, torb);
}

Status AddOutputTensor(TensorMap& map, const HloInstruction* inst, int64 n,
                       const DriverTensor& tensor) {
  return map.AddOutputTensor(inst, n, tensor);
}

Status AddOutputOpaque(TensorMap& map, const HloInstruction* inst, int64 n,
                       absl::any token) {
  return map.AddOutputOpaque(inst, n, token);
}

}  // namespace poplarplugin
}  // namespace xla
