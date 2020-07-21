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
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Encoding.hpp>
#include <popops/Pad.hpp>
#include <popops/SelectScalarFromRows.hpp>
#include <popops/UpdateScalarInRows.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {
StatusOr<poplar::Tensor> ReinterpretAsUnsigned(poplar::Tensor tensor) {
  if (tensor.elementType() == poplar::INT) {
    return tensor.reinterpret(poplar::UNSIGNED_INT);
  } else if (tensor.elementType() == poplar::UNSIGNED_INT) {
    return tensor;
  } else {
    return xla::InvalidArgumentStrCat(
        "Reinterpret - unsupported tensor type ",
        std::string(tensor.elementType().toString()));
  }
}

// Helper for Dynamic(Update)Slice where we recognize dynamic and constant slice
// dimensions.
struct DynamicSliceHelper {
  DynamicSliceHelper(const HloDynamicIndexInstruction* inst) {
    auto index_operands = inst->index_operands();
    const Shape input_shape = inst->operand(0)->shape();

    // Get the slice sizes in the dynamic and constant dimensions.
    std::vector<int64> slice_sizes;
    if (inst->opcode() == HloOpcode::kDynamicSlice) {
      auto* dyn_slice = Cast<HloDynamicSliceInstruction>(inst);
      slice_sizes = dyn_slice->dynamic_slice_sizes();
    } else {
      auto* dyn_update_slice = Cast<HloDynamicUpdateSliceInstruction>(inst);
      slice_sizes = dyn_update_slice->operand(1)->shape().dimensions();
    }

    Shape dynamic_slice_shape = input_shape;
    Shape constant_slice_shape = input_shape;

    // For each operand find whether it is a slice dimension.
    for (uint64 dim = 0; dim != index_operands.size(); ++dim) {
      size_t slice_size = slice_sizes[dim];
      if (input_shape.dimensions(dim) != static_cast<int64>(slice_size)) {
        if (index_operands[dim]->opcode() == HloOpcode::kConstant) {
          constant_slice_shape.set_dimensions(dim, slice_size);
          has_constant_slice = true;
        } else {
          dynamic_slice_shape.set_dimensions(dim, slice_size);
          has_dynamic_slice = true;
        }
      }
    }
    dynamic_slice_info = GetSliceInfo(input_shape, dynamic_slice_shape);
    constant_slice_info = GetSliceInfo(input_shape, constant_slice_shape);
  }

  DynamicSliceHelper() = delete;

  SliceInfo dynamic_slice_info;
  SliceInfo constant_slice_info;
  bool has_constant_slice = false;
  bool has_dynamic_slice = false;
};

StatusOr<poplar::Tensor> SliceTensorConstant(
    const poplar::Tensor& to_slice,
    const DynamicSliceHelper& dynamic_slice_helper,
    const HloDynamicIndexInstruction* inst) {
  auto first_index = inst->first_index_operand_number();
  poplar::Tensor sliced = to_slice;

  if (dynamic_slice_helper.has_constant_slice) {
    const SliceInfo& constant_slice_info =
        dynamic_slice_helper.constant_slice_info;
    std::vector<size_t> slices_start(to_slice.rank(), 0);
    std::vector<size_t> slices_end = to_slice.shape();
    for (size_t i = 0; i != constant_slice_info.sliced_dims.size(); ++i) {
      const size_t dim = constant_slice_info.sliced_dims[i];
      const size_t slice_size = constant_slice_info.slice_sizes[i];
      TF_ASSIGN_OR_RETURN(size_t slice_start,
                          LiteralScalarToNativeType<uint64>(
                              inst->operand(first_index + dim)->literal()));
      slices_start[dim] = slice_start;
      slices_end[dim] = slice_start + slice_size;
    }
    sliced = sliced.slice(slices_start, slices_end);
  }
  return sliced;
}

StatusOr<poplar::Tensor> GetDynamicSliceOffsets(
    const HloDynamicIndexInstruction* inst,
    const DynamicSliceHelper& dynamic_slice_helper,
    poplar::program::Sequence& seq, CompilerResources& res,
    TensorMap& tensor_map) {
  auto first_index = inst->first_index_operand_number();

  const SliceInfo& dynamic_slice_info = dynamic_slice_helper.dynamic_slice_info;
  const size_t num_dynamic_dims = dynamic_slice_info.sliced_dims.size();

  if (num_dynamic_dims == 0) {
    return InternalError(
        "A dynamic slice requires at least one dimension to slice on.");
  }

  std::vector<poplar::Tensor> slice_starts(num_dynamic_dims);

  for (size_t d = 0; d != num_dynamic_dims; ++d) {
    const size_t dim = dynamic_slice_info.sliced_dims[d];
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor slice_start,
        FindInstructionInput(tensor_map, res, inst, first_index + dim, seq));
    TF_ASSIGN_OR_RETURN(slice_starts[d],
                        ReinterpretAsUnsigned(slice_start.reshape({1})));
  }
  return poplar::concat(slice_starts);
}
}  // namespace

StatusOr<poplar::program::Program> CreateDynamicUpdateSliceOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  auto* dynamic_inst = Cast<HloDynamicIndexInstruction>(inst);
  poplar::Graph& graph = GetGraph(res, dynamic_inst);

  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(
      TensorVectors inputs,
      FindInplaceOutputTensors(tensor_map, res, dynamic_inst, seq));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor input = inputs[0][0];

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor update,
      FindInstructionInput(tensor_map, res, dynamic_inst, 1, seq));

  auto dynamic_slice_helper = DynamicSliceHelper(dynamic_inst);
  // First slice the input tensor on the const dimensions.
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor sliced_input,
      SliceTensorConstant(input, dynamic_slice_helper, dynamic_inst));

  // Do the dynamic slice - if there is no dynamic slicing involved, just copy
  // the update into the input.
  if (dynamic_slice_helper.has_dynamic_slice) {
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor slice_indices,
        GetDynamicSliceOffsets(dynamic_inst, dynamic_slice_helper, seq, res,
                               tensor_map));
    const SliceInfo& dynamic_slice_info =
        dynamic_slice_helper.dynamic_slice_info;
    popops::dynamicUpdate(graph, sliced_input, update, slice_indices,
                          dynamic_slice_info.sliced_dims,
                          dynamic_slice_info.slice_sizes, seq,
                          GetDebugName(dynamic_inst));
  } else {
    seq.add(poplar::program::Copy(update, sliced_input));
  }

  TF_CHECK_OK(AddOutputTensor(tensor_map, dynamic_inst, 0, input));

  return seq;
}

StatusOr<poplar::program::Program> CreateDynamicSliceOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  auto* dynamic_inst = Cast<HloDynamicIndexInstruction>(inst);
  poplar::Graph& graph = GetGraph(res, dynamic_inst);

  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor input,
      FindInstructionInput(tensor_map, res, dynamic_inst, 0, seq));

  auto dynamic_slice_helper = DynamicSliceHelper(dynamic_inst);
  // First slice the input tensor on the const dimensions.
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor sliced_input,
      SliceTensorConstant(input, dynamic_slice_helper, dynamic_inst));

  // Do the dynamic slice - if there is no dynamic slicing involved then just
  // duplicate the sliced input.
  poplar::Tensor out;
  if (dynamic_slice_helper.has_dynamic_slice) {
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor slice_indices,
        GetDynamicSliceOffsets(dynamic_inst, dynamic_slice_helper, seq, res,
                               tensor_map));
    const SliceInfo& dynamic_slice_info =
        dynamic_slice_helper.dynamic_slice_info;
    out = popops::dynamicSlice(
        graph, sliced_input, slice_indices, dynamic_slice_info.sliced_dims,
        dynamic_slice_info.slice_sizes, seq, GetDebugName(dynamic_inst));
  } else {
    out = poputil::duplicate(graph, sliced_input, seq,
                             GetDebugName(dynamic_inst));
  }

  TF_CHECK_OK(AddOutputTensor(tensor_map, dynamic_inst, 0, out));

  return seq;
}

StatusOr<poplar::program::Program> CreateIota(CompilerResources& res,
                                              const HloInstruction* inst,
                                              const xla::Shape& output_shape,
                                              TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  poplar::Graph& graph = GetGraph(res, inst);

  auto iota_inst = Cast<HloIotaInstruction>(inst);
  const auto iota_dimension = iota_inst->iota_dimension();

  // Get iota length.
  const int64 iota_length = output_shape.dimensions(iota_dimension);
  switch (output_shape.element_type()) {
    case S64: {
      if (!convert_scalar<int32>(iota_length)) {
        return xla::UnimplementedStrCat(
            "Iota - trying to create an iota of length ", iota_length,
            " but only 31-bit integer lengths are supported for signed types.");
      }
    }
    case U64: {
      if (!convert_scalar<uint32>(iota_length)) {
        return xla::UnimplementedStrCat(
            "Iota - trying to create an iota of length ", iota_length,
            " but only 32-bit integer lengths are supported for unsigned "
            "types.");
      }
    }
    default:
      break;
  }

  // Get the iota shape.
  const bool is_signed = ShapeUtil::ElementIsSigned(output_shape);
  auto iota_xla_type = is_signed ? S32 : U32;
  auto iota_shape = ShapeUtil::MakeShape(iota_xla_type, {iota_length});

  auto name = GetDebugName(inst);

  // Create a tensor which stores the iota.
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor iota_tensor,
      AddPlainTensor(graph, name + "/InitialIotaTensor", iota_shape, res));
  // Do the Iota.
  if (is_signed) {
    popops::iota(graph, iota_tensor, 0, seq, name + "/IotaSigned");
  } else {
    popops::iota(graph, iota_tensor, 0U, seq, name + "/IotaUnsigned");
  }
  // Cast it to the right type if the types don't match.
  TF_ASSIGN_OR_RETURN(poplar::Type iota_type, PoplarDataType(iota_shape));
  TF_ASSIGN_OR_RETURN(poplar::Type output_type, PoplarDataType(output_shape));
  poplar::Tensor casted = iota_type != output_type
                              ? popops::cast(graph, iota_tensor, output_type,
                                             seq, name + "/IotaCast")
                              : iota_tensor;

  // Broadcast it to the right shape given the iota dimension.
  TF_ASSIGN_OR_RETURN(poplar::Tensor out,
                      BroadcastTensor(casted, output_shape, {iota_dimension}));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

StatusOr<poplar::program::Program> CreateCopy(CompilerResources& res,
                                              const HloInstruction* inst,
                                              const xla::Shape& output_shape,
                                              TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  poplar::Graph& graph = GetGraph(res, inst);
  TensorVector inputs = FindInstructionInputs(tensor_map, res, inst, 0, seq);

  for (int64 tuple_idx = 0; tuple_idx != static_cast<int64>(inputs.size());
       ++tuple_idx) {
    poplar::Tensor out = poputil::duplicate(
        graph, inputs[tuple_idx], seq,
        absl::StrCat(GetDebugName(inst), "/", tuple_idx),
        poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, tuple_idx, out));
  }
  return seq;
}

StatusOr<poplar::program::Program> CreateSlice(CompilerResources& res,
                                               const HloInstruction* inst,
                                               const xla::Shape& output_shape,
                                               TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  poplar::Graph& graph = GetGraph(res, inst);
  poplar::Tensor input;
  poplar::Tensor output;

  // Handle this according to inplaceness.
  const bool is_inplace = IsLoweredInplace(inst);
  if (is_inplace) {
    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, seq, false));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    input = inputs[0][0];
  } else {
    TF_ASSIGN_OR_RETURN(
        input, FindInstructionInput(tensor_map, res, inst, 0, seq, false));
  }

  auto optional_begin =
      convert_array<std::vector<size_t>>(inst->slice_starts());
  if (!optional_begin) {
    return xla::FailedPrecondition("HandleSlice - cannot cast slice starts.");
  }
  std::vector<size_t> begin = *optional_begin;

  auto optional_end = convert_array<std::vector<size_t>>(inst->slice_limits());
  if (!optional_end) {
    return xla::FailedPrecondition("HandleSlice - cannot cast slice limits.");
  }
  std::vector<size_t> end = *optional_end;

  std::vector<int64> strides(inst->slice_strides());
  bool simple(true);
  for (std::size_t s : strides) {
    simple &= (s == 1);
  }
  if (simple) {
    input = input.slice(begin, end);
  } else {
    for (size_t d = 0; d < strides.size(); d++) {
      int64 s = strides[d];
      if (s > 0) {
        input = input.slice(begin[d], end[d], d);
        input = input.subSample(strides[d], d);
      } else {
        input = input.slice(end[d] + 1, begin[d] + 1, d);
        input = input.reverse(d);
        input = input.subSample(-strides[d], d);
      }
    }
  }
  if (is_inplace) {
    // This operation is inplace, just pass it through.
    output = input;
  } else {
    // Clone the unaliased regions.
    output = poputil::duplicate(
        graph, input, seq, GetDebugName(inst),
        poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
  }
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));
  return seq;
}

StatusOr<poplar::program::Program> CreateZeroPadOp(CompilerResources& res,
                                                   const HloInstruction* inst,
                                                   const xla::Shape& output,
                                                   TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  poplar::Graph& graph = GetGraph(res, inst);

  const HloInstruction* root = inst->fused_expression_root();
  const PaddingConfig& cfg(root->padding_config());

  TF_ASSIGN_OR_RETURN(
      TensorVectors inputs,
      FindInplaceOutputTensors(tensor_map, res, inst, seq, false));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor in = inputs[0][0];

  std::vector<std::ptrdiff_t> paddingLower;
  std::vector<std::ptrdiff_t> paddingUpper;
  for (auto& d : cfg.dimensions()) {
    paddingLower.push_back(d.edge_padding_low());
    paddingUpper.push_back(d.edge_padding_high());
  }
  poplar::Tensor zero = graph.addConstant(in.elementType(), {}, 0,
                                          GetDebugName(inst) + "/ZeroPad");
  graph.setTileMapping(zero, 0);
  poplar::Tensor out = popops::pad(graph, in, paddingLower, paddingUpper, zero);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return seq;
}

StatusOr<poplar::program::Program> CreateSelectScalarFromRows(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  poplar::Tensor params;
  TF_ASSIGN_OR_RETURN(params,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));
  poplar::Tensor indices;
  TF_ASSIGN_OR_RETURN(indices,
                      FindInstructionInput(tensor_map, res, inst, 1, seq));
  TF_ASSIGN_OR_RETURN(indices, ReinterpretAsUnsigned(indices));

  poplar::Tensor out = popops::selectScalarFromRows(graph, params, indices, seq,
                                                    GetDebugName(inst));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

StatusOr<poplar::program::Program> CreateUpdateScalarInRows(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                      FindInplaceOutputTensors(tensor_map, res, inst, seq));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor params = inputs[0][0];

  TF_ASSIGN_OR_RETURN(poplar::Tensor indices,
                      FindInstructionInput(tensor_map, res, inst, 1, seq));
  TF_ASSIGN_OR_RETURN(indices, ReinterpretAsUnsigned(indices));

  popops::updateScalarInRows(graph, params, indices, seq, GetDebugName(inst));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, params));

  return seq;
}

StatusOr<poplar::program::Program> CreateTuple(CompilerResources& res,
                                               const HloInstruction* inst,
                                               TensorMap& tensor_map,
                                               bool expand_aliasing,
                                               bool preserve_aliases) {
  poplar::program::Sequence seq;
  TF_ASSIGN_OR_RETURN(
      TensorVectors inputs,
      FindInplaceOutputTensors(tensor_map, res, inst, seq, expand_aliasing,
                               preserve_aliases));
  CHECK_EQ(inputs.size(), inst->operand_count());
  uint64 n = 0;
  for (uint64 i = 0; i < inputs.size(); i++) {
    CHECK_EQ(inputs[i].size(), CountShapes(inst->operand(i)->shape()));
    for (uint64 j = 0; j < inputs[i].size(); j++) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, n, inputs[i][j]));
      n++;
    }
  }
  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
