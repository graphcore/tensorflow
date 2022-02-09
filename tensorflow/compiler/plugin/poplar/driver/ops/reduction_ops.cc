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
#include <gcl/Collectives.hpp>
#include <limits>
#include <poplar/TensorCloneMethod.hpp>
#include <popnn/Pooling.hpp>
#include <popnn/PoolingDef.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Pad.hpp>
#include <popops/Reduce.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/reduction_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/window_util.h"

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {

static const std::string a_conn("a");
static const std::string b_conn("b");
static const std::string out_conn("out");

static const std::string reduction_add("ReductionAdd");
static const std::string reduction_mul("ReductionMul");
static const std::string reduction_max("ReductionMax");
static const std::string reduction_min("ReductionMin");
static const std::string reduction_and("ReductionAnd");
static const std::string reduction_or("ReductionOr");

static const std::string reduction_ge("SelectionGe");
static const std::string reduction_gt("SelectionGt");
static const std::string reduction_le("SelectionLe");
static const std::string reduction_lt("SelectionLt");

static const std::string unknown("Unknown");

bool IsReducibleArithmetic(const HloComputation* computation) {
  HloInstruction* root(computation->root_instruction());
  if (!hlo_query::AllOperandsAreParameters(*root)) {
    return false;
  }

  switch (root->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kMultiply:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
      return true;
    default:
      return false;
  }
}

bool IsSimpleSelection(const HloComputation* computation) {
  HloInstruction* root(computation->root_instruction());
  if (!hlo_query::AllOperandsAreParameters(*root)) {
    return false;
  }

  if (root->opcode() != HloOpcode::kCompare) {
    return false;
  }

  switch (root->comparison_direction()) {
    case ComparisonDirection::kGe:
    case ComparisonDirection::kGt:
    case ComparisonDirection::kLe:
    case ComparisonDirection::kLt:
      return true;
    default:
      return false;
  }
}

bool IsPoplibsPool(const HloInstruction* inst,
                   const HloComputation* computation) {
  HloInstruction* root(computation->root_instruction());
  if (!hlo_query::AllOperandsAreParameters(*root)) {
    return false;
  }

  switch (root->opcode()) {
    case HloOpcode::kMaximum:
    case HloOpcode::kAdd:
      break;
    default:
      return false;
  }

  if (inst->shape().rank() != 4) {
    return false;
  }

  const Window& window = inst->window();
  if (window_util::HasDilation(window)) {
    return false;
  }

  unsigned reduction_count = 0;
  for (int64 i = 0; i < window.dimensions_size(); i++) {
    auto& d = window.dimensions(i);
    if (d.size() != 1 || d.stride() != 1 || d.padding_low() != 0 ||
        d.padding_high() != 0) {
      reduction_count++;
    }
  }

  return (reduction_count <= 2);
}

static const std::string& ReductionVertexBaseName(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kAdd:
      return reduction_add;
    case HloOpcode::kMultiply:
      return reduction_mul;
    case HloOpcode::kMaximum:
      return reduction_max;
    case HloOpcode::kMinimum:
      return reduction_min;
    case HloOpcode::kAnd:
      return reduction_and;
    case HloOpcode::kOr:
      return reduction_or;
    default:
      // Cannot reach here
      return unknown;
  }
}

static const std::string& SelectionVertexBaseName(const HloInstruction* inst) {
  switch (inst->comparison_direction()) {
    case ComparisonDirection::kGe:
      return reduction_ge;
    case ComparisonDirection::kGt:
      return reduction_gt;
    case ComparisonDirection::kLe:
      return reduction_le;
    case ComparisonDirection::kLt:
      return reduction_lt;
    default:
      // Cannot reach here
      return unknown;
  }
}

static std::vector<int64> MaxWindowOverlap(const Window& window) {
  std::vector<int64> overlap;
  for (auto& d : window.dimensions()) {
    int64 o = ((d.size() + d.stride() - 1) / d.stride());
    overlap.push_back(o);
  }
  return overlap;
}

template <typename Tpos, typename Tlimit>
static std::size_t GetOverlapLayerNum(const Tpos& pos, const Tlimit& limit) {
  std::size_t layer = 0;
  std::size_t mult = 1;
  for (size_t d = 0; d < pos.size(); d++) {
    std::size_t v = (pos[d] % limit[d]) * mult;
    layer += v;
    mult *= limit[d];
  }
  return layer;
}

std::set<unsigned int> GetPoolingReductionDims(const Window& window) {
  std::set<unsigned int> reduction_dims;
  for (int64 i = 0; i < window.dimensions_size(); i++) {
    auto& d = window.dimensions(i);
    if (d.size() != 1 || d.stride() != 1 || d.padding_low() != 0 ||
        d.padding_high() != 0) {
      reduction_dims.insert(i);
    }
  }

  // If not enough reduction dimensions, we add more
  // If the input is an N-D tensor, we need N-2 "reduction dimensions"
  int dims_from_end = 1;
  while (reduction_dims.size() < window.dimensions_size() - 2) {
    int dim_candidate = window.dimensions_size() - dims_from_end;
    if (reduction_dims.count(dim_candidate) == 0) {
      reduction_dims.insert(dim_candidate);
    }
    dims_from_end += 1;
  }

  return reduction_dims;
}

static std::vector<unsigned int> GetShuffleInputDimensionsForPoplar(
    const Window& window, const std::set<unsigned int> reduction_dims) {
  std::vector<unsigned int> shuffle_in;
  for (int i = 0; i < window.dimensions_size(); i++) {
    if (reduction_dims.count(i) == 0) {
      shuffle_in.push_back(i);
    }
  }
  shuffle_in.insert(shuffle_in.end(), reduction_dims.begin(),
                    reduction_dims.end());
  return shuffle_in;
}

static std::vector<unsigned int> GetShuffleOutputDimensionsForPoplar(
    const std::vector<unsigned int> shuffle_in) {
  std::vector<unsigned int> shuffle_out(shuffle_in.size());
  for (size_t i = 0; i < shuffle_in.size(); i++) {
    shuffle_out[shuffle_in[i]] = i;
  }
  return shuffle_out;
}

static poplar::Type GetReductionType(const popnn::PoolingType& pooling_type,
                                     const poplar::Type& input_type) {
  switch (pooling_type) {
    case popnn::PoolingType::AVG:
    case popnn::PoolingType::SUM:
      return (input_type == poplar::HALF) ? poplar::FLOAT : input_type;
    case popnn::PoolingType::MAX:
    default:
      return input_type;
  }
  return input_type;
}

static popnn::pooling::PoolParams GetPoplibsPoolParams(
    const popnn::PoolingType& pooling_type, const Window& window,
    const std::vector<std::size_t>& input_shape,
    const std::set<unsigned int>& reduction_dims,
    const poplar::Type& input_data_type) {
  // TODO assume here that batch dimension and the channel dimension order
  // doesn't actually matter - it's just the non field dimensions.
  const auto batch_size = input_shape.front();
  const auto num_channels = input_shape[1];
  std::vector<std::size_t> input_field_shape(std::next(input_shape.begin(), 2),
                                             input_shape.end());

  std::vector<std::size_t> kernel_shape;
  std::vector<unsigned> stride;
  std::vector<int> padding_lower;
  std::vector<int> padding_upper;

  for (auto reduction_dim : reduction_dims) {
    auto& d = window.dimensions(reduction_dim);
    kernel_shape.push_back((std::size_t)d.size());
    stride.push_back((unsigned)d.stride());
    padding_lower.push_back((int)d.padding_low());
    padding_upper.push_back((int)d.padding_high());
  }

  return {pooling_type, input_field_shape, kernel_shape,
          stride,       padding_lower,     padding_upper,
          num_channels, batch_size,        input_data_type};
}
StatusOr<poplar::program::Sequence> CreateSimpleReduction(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return CreateSimpleReduction(res, inst, inst, output_shape, tensor_map,
                               /*with_scale=*/false, debug_name_and_id);
}

StatusOr<poplar::program::Sequence> CreateSimpleReduction(
    CompilerResources& res, const HloInstruction* inst,
    const HloInstruction* reduce_inst, const xla::Shape& output_shape,
    TensorMap& tensor_map, bool with_scale,
    const poplar::DebugNameAndId& debug_name_and_id) {
  TF_ASSIGN_OR_RETURN(popops::Operation reduction_operation,
                      GetPoplibsReductionOperation(reduce_inst));
  return CreateSimpleReduction(res, reduction_operation, inst, reduce_inst,
                               output_shape, tensor_map, with_scale,
                               debug_name_and_id);
}

StatusOr<poplar::program::Sequence> CreateSimpleReduction(
    CompilerResources& res, popops::Operation reduction_operation,
    const HloInstruction* inst, const HloInstruction* reduce_inst,
    const xla::Shape& output_shape, TensorMap& tensor_map, bool with_scale,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::program::Sequence seq({}, debug_name_and_id);
  poplar::Tensor out;

  poplar::Graph& graph = GetGraph(res, inst);

  if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
    TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, res, inst, 1, seq,
                                                  debug_name_and_id));
    TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out, inst->shape(), {}));
  } else {
    // Find the input tensors
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor to_reduce,
        FindInstructionInput(tensor_map, res, inst, 0, seq, debug_name_and_id));

    const HloInstruction* root = reduce_inst->to_apply()->root_instruction();

    std::vector<std::size_t> reduction_dims;
    for (auto d : reduce_inst->dimensions()) {
      reduction_dims.push_back(d);
    }

    if (ShapeUtil::ElementsIn(output_shape) == 1) {
      // If output is scalar, map it linearly with res.linear_mapping_state
      TF_ASSIGN_OR_RETURN(
          out, AddPlainTensor(graph, {debug_name_and_id, "out"}, output_shape,
                              res, /*offset=*/true));
    } else if (HasTensorAllocationTarget(TensorLocation{inst, 0}, res)) {
      // Check if there is an allocation for the reduction output.
      TF_ASSIGN_OR_RETURN(
          out, AddTensor(graph, TensorLocation{inst, 0}, output_shape, res,
                         tensor_map, {debug_name_and_id, "out"}));
    } else {
      TF_ASSIGN_OR_RETURN(auto type, PoplarDataType(inst->shape()));
      const auto shape = PoplarShapeFromXlaShape(inst->shape());
      out = graph.addVariable(type, shape, {debug_name_and_id, "out"});

      const auto to_reduce_mapping = graph.getTileMapping(to_reduce);
      std::vector<unsigned> tiles;
      for (auto i = 0ul; i < to_reduce_mapping.size(); ++i) {
        if (!to_reduce_mapping[i].empty()) {
          tiles.push_back(i);
        }
      }

      // Map the reduce output to the same number of tiles preventing subatomic
      // stores.
      const auto get_lcm = [](uint32 a, uint32 b) {
        if (a > b) {
          return (a / tensorflow::MathUtil::GCD<uint32>(a, b)) * b;
        } else if (a < b) {
          return (b / tensorflow::MathUtil::GCD<uint32>(b, a)) * a;
        } else {
          return a;
        }
      };

      const uint32 element_size =
          graph.getTarget().getTypeSize(out.elementType());
      const uint32 element_bound =
          get_lcm(element_size, graph.getTarget().getAtomicStoreGranularity()) /
          element_size;

      uint32 grain_size =
          std::max<uint32>(element_bound, out.numElements() / tiles.size());
      grain_size += grain_size % element_bound;

      poputil::mapTensorLinearly(graph, out, 0, grain_size);
    }

    popops::ReduceParams reduce_params(reduction_operation);
    if (with_scale) {
      TF_ASSIGN_OR_RETURN(poplar::Tensor scale,
                          FindInstructionInput(tensor_map, res, inst, 2, seq,
                                               debug_name_and_id));
      reduce_params = popops::ReduceParams(reduction_operation, false, scale);
    }

    popops::reduceWithOutput(graph, to_reduce, out, reduction_dims,
                             reduce_params, seq, {debug_name_and_id});

    // Apply initial value
    Literal identity_literal =
        GetIdentityConstantLiteral(root, inst->shape().element_type());

    auto* init_inst = inst->operand(1);
    if (!(init_inst->IsConstant() &&
          init_inst->literal() == identity_literal)) {
      TF_ASSIGN_OR_RETURN(poplar::Tensor init_val,
                          FindInstructionInput(tensor_map, res, inst, 1, seq,
                                               debug_name_and_id));

      // Create a binary op with the scatter_root opcode
      TF_ASSIGN_OR_RETURN(init_val, BroadcastTensor(init_val, output_shape));

      TF_ASSIGN_OR_RETURN(popops::expr::BinaryOpType op, LookupBinaryFn(root));

      popops::mapInPlace(graph, op, out, init_val, seq,
                         {debug_name_and_id, "initval"});
    }
  }
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

StatusOr<poplar::program::Sequence> CreateSimpleWindowReduction(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::program::Sequence seq({}, debug_name_and_id);
  poplar::Tensor out;

  poplar::Graph& graph = GetGraph(res, inst);

  if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
    TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, res, inst, 1, seq,
                                                  debug_name_and_id));
    TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out, inst->shape(), {}));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  } else {
    // Find the input tensors
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor to_reduce,
        FindInstructionInput(tensor_map, res, inst, 0, seq, debug_name_and_id));

    // Find the type and vertex
    HloInstruction* root(inst->to_apply()->root_instruction());
    const std::string vertex_name =
        templateVertex(ReductionVertexBaseName(root), to_reduce.elementType());

    const Window& window(inst->window());

    // Allocate the output tensor
    TF_ASSIGN_OR_RETURN(
        out, AddTensor(graph, TensorLocation{inst, 0}, output_shape, res,
                       tensor_map, {debug_name_and_id, "out"}));
    poplar::Tensor out_flat = out.flatten();

    auto cs = graph.addComputeSet({debug_name_and_id});
    const int64 N = ShapeUtil::ElementsIn(output_shape);
    const int64 rank = output_shape.rank();

    // Find the within window strides in case subsampling is required.
    std::vector<uint32> strides(rank, 1);
    for (int64 d = 0; d != rank; d++) {
      const auto& dim = window.dimensions(d);
      if (dim.window_dilation() > dim.base_dilation()) {
        bool valid = false;
        for (int64 i = 0; i != dim.size(); ++i) {
          if ((((i + 1) * dim.window_dilation()) % dim.base_dilation()) == 0) {
            strides[d] =
                ((i + 1) * dim.window_dilation()) / dim.base_dilation();
            valid = true;
            break;
          }
        }
        if (!valid) {
          return FailedPrecondition(
              "Cannot compute a valid stride for a reduction window.");
        }
      }
    }

    // Vector for walking the window through the tensor
    std::vector<std::size_t> pos(rank, 0);
    for (int64 i = 0; i != N; ++i) {
      // Find the window boundries.
      std::vector<std::size_t> start(rank);
      std::vector<std::size_t> end(rank);
      bool valid = true;
      for (int64 d = 0; d != rank; d++) {
        const auto& dim = window.dimensions(d);
        const size_t max_idx = to_reduce.dim(d);

        int32 start_idx = pos[d] * dim.stride() - dim.padding_low();
        int32 end_idx = start_idx + (dim.size() - 1) * dim.window_dilation();

        auto is_valid_idx = [&dim, d, max_idx](int32 idx) {
          return idx >= 0 && (idx % dim.base_dilation()) == 0 &&
                 (idx / dim.base_dilation()) < max_idx;
        };

        // Find the first valid start index.
        for (int64 i = 0; i != dim.size(); ++i) {
          if (!is_valid_idx(start_idx)) {
            start_idx += dim.window_dilation();
          } else {
            break;
          }
        }
        // Find the last valid index.
        for (int64 i = 0; i != dim.size(); ++i) {
          if (!is_valid_idx(end_idx)) {
            end_idx -= dim.window_dilation();
          } else {
            break;
          }
        }

        if (!is_valid_idx(start_idx) || !is_valid_idx(end_idx)) {
          valid = false;
          break;
        }

        start[d] = start_idx / dim.base_dilation();
        end[d] = end_idx / dim.base_dilation() + 1;
      }

      if (!valid) {
        std::fill_n(start.data(), rank, 0);
        std::fill_n(end.data(), rank, 0);
      }
      poplar::Tensor tensor_window = to_reduce.slice(start, end);

      if (valid) {
        for (int64 d = 0; d != rank; d++) {
          if (strides[d] > 1) {
            tensor_window = tensor_window.subSample(strides[d], d);
          }
        }
      }

      poplar::Tensor output = out_flat.slice(i, i + 1).reshape({});

      // Create the vertex.
      auto v = graph.addVertex(
          cs, vertex_name, {{"a", tensor_window.flatten()}, {"out", output}});
      graph.setTileMapping(v, (i / graph.getTarget().getNumWorkerContexts()) %
                                  graph.getTarget().getNumTiles());
      graph.setPerfEstimate(v, 1);

      // Advance the window.
      for (int64 d = rank - 1; d >= 0; d--) {
        pos[d]++;
        if (pos[d] < output_shape.dimensions(d)) {
          break;
        } else {
          pos[d] = 0;
        }
      }
    }

    seq.add(poplar::program::Execute(cs, {debug_name_and_id}));

    // Apply initial value
    Literal identity_literal =
        GetIdentityConstantLiteral(root, inst->shape().element_type());
    auto* init_inst = inst->operand(1);
    if (!(init_inst->IsConstant() &&
          init_inst->literal() == identity_literal)) {
      TF_ASSIGN_OR_RETURN(poplar::Tensor init_val,
                          FindInstructionInput(tensor_map, res, inst, 1, seq,
                                               debug_name_and_id));

      // Create a binary op with the reduce window opcode.
      TF_ASSIGN_OR_RETURN(init_val, BroadcastTensor(init_val, output_shape));

      TF_ASSIGN_OR_RETURN(popops::expr::BinaryOpType op, LookupBinaryFn(root));

      popops::mapInPlace(graph, op, out, init_val, seq,
                         {debug_name_and_id, "initval"});
    }
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  }

  return seq;
}

StatusOr<poplar::program::Sequence> CreatePoplibsWindowReduction(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
    poplar::program::Sequence prog({}, debug_name_and_id);
    TF_ASSIGN_OR_RETURN(poplar::Tensor out,
                        FindInstructionInput(tensor_map, res, inst, 1, prog,
                                             debug_name_and_id));
    TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out, inst->shape(), {}));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
    return prog;
  } else {
    popnn::PoolingType reduction_type;
    CHECK_EQ(inst->opcode(), HloOpcode::kReduceWindow);
    switch (inst->to_apply()->root_instruction()->opcode()) {
      case HloOpcode::kMaximum: {
        reduction_type = popnn::PoolingType::MAX;
        break;
      }
      case HloOpcode::kAdd: {
        reduction_type = popnn::PoolingType::SUM;
        break;
      }
      default: {
        return xla::FailedPrecondition("Unsupported window reduction %s.",
                                       inst->name());
      }
    }

    return CreatePoplibsPooling(res, inst, tensor_map, reduction_type,
                                inst->window(), debug_name_and_id, inst);
  }
}

StatusOr<poplar::program::Sequence> CreatePoplibsPooling(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    popnn::PoolingType pooling_type, const Window& window,
    const poplar::DebugNameAndId& debug_name_and_id,
    absl::optional<const HloInstruction*> optional_reduction_op) {
  poplar::Graph& graph = GetGraph(res, inst);
  poplar::program::Sequence prog({}, debug_name_and_id);

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor to_reduce,
      FindInstructionInput(tensor_map, res, inst, 0, prog, debug_name_and_id));
  auto reduction_dims = GetPoolingReductionDims(window);

  if (reduction_dims.size() == 0) {
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, to_reduce));
    return prog;
  }

  const auto shuffle_in =
      GetShuffleInputDimensionsForPoplar(window, reduction_dims);
  to_reduce = to_reduce.dimShuffle(shuffle_in);

  // TODO(T7321) The default expected behaviour is to do any partial
  // calculations in FP32, however popnn:pooling currently does not support
  // having an FP16 input and FP32 partials. We therefore upcast the input if
  // required and then downcast the result.
  auto input_type = to_reduce.elementType();
  auto reduction_type = GetReductionType(pooling_type, input_type);
  const bool cast_required = input_type != reduction_type;
  // Do the cast to make sure partials are at higher precision.
  if (cast_required) {
    to_reduce = popops::cast(graph, to_reduce, reduction_type, prog,
                             {debug_name_and_id, "PreCast"});
  }
  auto pool_params = GetPoplibsPoolParams(
      pooling_type, window, to_reduce.shape(), reduction_dims, reduction_type);

  poplar::Tensor out =
      popnn::pooling::pool(graph, pool_params, to_reduce, prog,
                           {debug_name_and_id}, res.default_pooling_options);

  // Do the cast to to the original input type.
  if (cast_required) {
    out = popops::cast(graph, out, input_type, prog,
                       {debug_name_and_id, "PostCast"});
  }

  if (optional_reduction_op) {
    // We apply the initial_value of the pooling in the non-default base
    // case. This needs to be after poplibs pool, as it does not support the
    // non-default base case.
    const HloInstruction* reduction_op = *optional_reduction_op;
    const HloInstruction* reducing_op =
        reduction_op->to_apply()->root_instruction();
    const HloInstruction* initial_value = reduction_op->operand(1);

    // What is the default base case for the reduction_op, MAX: -largest, SUM:
    // 0, etc.
    Literal identity_literal =
        GetIdentityConstantLiteral(reducing_op, inst->shape().element_type());

    // Apply the base case if necessary
    if (!(initial_value->IsConstant() &&
          initial_value->literal() == identity_literal)) {
      TF_ASSIGN_OR_RETURN(poplar::Tensor init_val,
                          FindInstructionInput(tensor_map, res, inst, 1, prog,
                                               debug_name_and_id));
      init_val = init_val.reshape({1})
                     .broadcast(out.numElements(), 0)
                     .reshape(out.shape());
      TF_ASSIGN_OR_RETURN(popops::expr::BinaryOpType op,
                          LookupBinaryFn(reducing_op));
      popops::mapInPlace(graph, op, out, init_val, prog,
                         {debug_name_and_id, "initval"});
    }
  }

  const auto shuffle_out = GetShuffleOutputDimensionsForPoplar(shuffle_in);
  out = out.dimShuffle(shuffle_out);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return prog;
}

StatusOr<poplar::program::Sequence> CreatePoplibsMaxPoolGrad(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    const Window& window, const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::program::Sequence seq({}, debug_name_and_id);
  poplar::Graph& graph = GetGraph(res, inst);

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor input,
      FindInstructionInput(tensor_map, res, inst, 0, seq, debug_name_and_id));
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor output,
      FindInstructionInput(tensor_map, res, inst, 1, seq, debug_name_and_id));
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor output_grad,
      FindInstructionInput(tensor_map, res, inst, 2, seq, debug_name_and_id));

  const auto reduction_dims = GetPoolingReductionDims(window);

  const auto shuffle_in =
      GetShuffleInputDimensionsForPoplar(window, reduction_dims);

  input = input.dimShuffle(shuffle_in);
  output_grad = output_grad.dimShuffle(shuffle_in);
  output = output.dimShuffle(shuffle_in);

  auto pool_params =
      GetPoplibsPoolParams(popnn::PoolingType::MAX, window, input.shape(),
                           reduction_dims, input.elementType());

  poplar::Tensor out = popnn::pooling::poolInputGradient(
      graph, pool_params, input, output, output_grad, false, seq,
      {debug_name_and_id}, res.default_pooling_options);
  // Shuffle back
  const auto shuffle_out = GetShuffleOutputDimensionsForPoplar(shuffle_in);
  out = out.dimShuffle(shuffle_out);
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return seq;
}

StatusOr<poplar::program::Sequence> CreatePoplibsPoolingGrad(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    popnn::PoolingType pooling_type, const Window& window,
    const poplar::DebugNameAndId& debug_name_and_id) {
  if (pooling_type == popnn::PoolingType::MAX) {
    return xla::FailedPrecondition("Calling invalid function for MaxPoolGrad.");
  }

  poplar::program::Sequence prog({}, debug_name_and_id);
  poplar::Graph& graph = GetGraph(res, inst);

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor output_grad,
      FindInstructionInput(tensor_map, res, inst, 0, prog, debug_name_and_id));
  // Get the input shape (same shape as the input grad i.e. the output shape).
  auto optional_input_shape =
      convert_array<std::vector<std::size_t>>(inst->shape().dimensions());
  if (!optional_input_shape) {
    return xla::FailedPrecondition(
        "CreatePoplibsPoolingGrad - cannot cast input_shape.");
  }
  std::vector<std::size_t> input_shape = *optional_input_shape;

  const auto reduction_dims = GetPoolingReductionDims(window);

  const auto shuffle_in =
      GetShuffleInputDimensionsForPoplar(window, reduction_dims);
  // Shuffle the input shape and the output grad tensor into a NC... shape.
  std::vector<std::size_t> input_shape_shuffled(input_shape.size());
  for (uint64 i = 0; i < input_shape.size(); i++) {
    input_shape_shuffled[i] = input_shape[shuffle_in[i]];
  }
  output_grad = output_grad.dimShuffle(shuffle_in);

  // TODO(T7321) The default expected behaviour is to do any partial
  // calculations in FP32, however popnn:pooling currently does not support
  // having an FP16 input and FP32 partials. We therefore upcast the input if
  // required and then downcast the result.
  auto output_grad_type = output_grad.elementType();
  auto reduction_type = GetReductionType(pooling_type, output_grad_type);
  const bool cast_required = output_grad_type != reduction_type;
  if (cast_required) {
    output_grad = popops::cast(graph, output_grad, reduction_type, prog,
                               {debug_name_and_id, "PreCast"});
  }

  auto pool_params =
      GetPoplibsPoolParams(pooling_type, window, input_shape_shuffled,
                           reduction_dims, reduction_type);
  poplar::Tensor out = popnn::pooling::poolInputGradient(
      graph, pool_params, 1, output_grad, prog, {debug_name_and_id},
      res.default_pooling_options);

  if (cast_required) {
    out = popops::cast(graph, out, output_grad_type, prog,
                       {debug_name_and_id, "PreCast"});
  }

  // Shuffle back
  const auto shuffle_out = GetShuffleOutputDimensionsForPoplar(shuffle_in);
  out = out.dimShuffle(shuffle_out);
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return prog;
}

StatusOr<poplar::program::Sequence> CreateSimpleSelectAndScatter(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::Tensor out;
  poplar::program::Sequence prog({}, debug_name_and_id);

  poplar::Graph& graph = GetGraph(res, inst);

  // Find the input tensors
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor operand,
      FindInstructionInput(tensor_map, res, inst, 0, prog, debug_name_and_id));

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor source,
      FindInstructionInput(tensor_map, res, inst, 1, prog, debug_name_and_id));

  HloInstruction* select_root(inst->select()->root_instruction());
  HloInstruction* scatter_root(inst->scatter()->root_instruction());

  /*
   * Selection
   */

  std::string select_vertex_name = templateVertex(
      SelectionVertexBaseName(select_root), operand.elementType());

  const Window& window(inst->window());

  std::vector<int64> overlap(MaxWindowOverlap(window));
  int64 overlap_count(std::accumulate(overlap.begin(), overlap.end(), 1,
                                      [](int64 a, int64 b) { return a * b; }));

  // Create a partials tensor for reduction
  std::vector<std::size_t> poplar_shape = operand.shape();
  poplar_shape.push_back(1);

  poplar::Tensor extended_operand = operand.reshape(poplar_shape);
  poplar::Tensor partial =
      graph.clone(extended_operand, {debug_name_and_id, "partial"});

  for (int64 i = 1; i < overlap_count; i++) {
    partial = poplar::concat(
        partial, graph.clone(extended_operand, {debug_name_and_id, "partial"}),
        partial.rank() - 1);
  }

  xla::Shape partial_shape(output_shape);
  partial_shape.add_dimensions(overlap_count);
  LayoutUtil::ClearLayout(&partial_shape);
  partial_shape.mutable_layout()->set_format(DENSE);

  Literal identity_literal =
      GetIdentityConstantLiteral(scatter_root, inst->shape().element_type());

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor identity_val,
      AddConstantTensor(graph, TensorLocation{inst, 0}, partial_shape,
                        identity_literal, res, tensor_map,
                        {debug_name_and_id, "identity_val"}));
  prog.add(
      poplar::program::Copy(identity_val, partial, false, {debug_name_and_id}));

  // Find the number of windows in each dimension
  std::vector<unsigned> window_count(output_shape.rank());
  for (int64 d = 0; d < window.dimensions().size(); d++) {
    std::size_t input_dim(operand.dim(d));
    input_dim += window.dimensions(d).padding_low();
    input_dim += window.dimensions(d).padding_high();

    window_count[d] = window_util::StridedBound(
        input_dim, window.dimensions(d).size(), window.dimensions(d).stride());
  }

  auto select_cs = graph.addComputeSet({debug_name_and_id, "select"});
  prog.add(poplar::program::Execute(select_cs, {debug_name_and_id}));

  const unsigned long num_windows = source.numElements();

  unsigned dim_count(operand.rank());

  // Vector for walking the window through the tensor
  std::vector<std::size_t> pos(dim_count, 0);

  // Slice boundaries
  std::vector<std::size_t> start_in(dim_count);
  std::vector<std::size_t> end_in(dim_count);

  std::vector<std::size_t> start_par(dim_count + 1);
  std::vector<std::size_t> end_par(dim_count + 1);

  for (unsigned i = 0; i < num_windows; ++i) {
    // Find the windows
    for (unsigned d = 0; d < dim_count; d++) {
      const auto& wd(window.dimensions(d));

      int s(pos[d] * wd.stride() - wd.padding_low());
      int e(s + wd.size());
      start_in[d] = std::min(std::max(s, 0), (int)operand.dim(d));
      end_in[d] = std::min(std::max(e, 0), (int)operand.dim(d));

      start_par[d] = start_in[d];
      end_par[d] = end_in[d];
    }
    start_par[dim_count] = GetOverlapLayerNum(pos, overlap);
    end_par[dim_count] = start_par[dim_count] + 1;

    poplar::Tensor w_in = operand.slice(start_in, end_in).flatten();
    poplar::Tensor w_par = partial.slice(start_par, end_par).flatten();
    poplar::Tensor s = source.index(pos);

    auto m = graph.getTileMapping(w_in);
    unsigned int tile_with_max_elements = 0;
    std::size_t max_elements = 0;
    for (unsigned int t = 0; t < m.size(); t++) {
      std::size_t element_count = 0;
      for (auto interval : m[t]) {
        element_count += interval.size();
      }
      if (element_count > max_elements) {
        max_elements = element_count;
        tile_with_max_elements = t;
      }
    }

    // Create the vertex
    auto v = graph.addVertex(select_cs, select_vertex_name,
                             {{"a", w_in}, {"b", s}, {"out", w_par}});
    TF_RETURN_IF_ERROR(SetVertexField(graph, v["initval"], identity_literal));
    graph.setTileMapping(v, tile_with_max_elements);
    graph.setPerfEstimate(v, 1);

    // Advance the window
    for (int d = dim_count - 1; d >= 0; d--) {
      pos[d]++;
      if (pos[d] < window_count[d]) break;
      pos[d] = 0;
    }
  }

  /*
   * Reduction
   */
  TF_ASSIGN_OR_RETURN(popops::Operation op, GetPoplibsReductionOperation(inst));

  std::vector<std::size_t> reduction_dims;
  reduction_dims.push_back(partial.rank() - 1);

  out = popops::reduce(graph, partial, reduction_dims, op, prog,
                       {debug_name_and_id, "reduce"});

  /*
   * Initial value application
   */
  auto* init_inst = inst->operand(2);
  if (!(init_inst->IsConstant() && init_inst->literal() == identity_literal)) {
    TF_ASSIGN_OR_RETURN(poplar::Tensor init_val,
                        FindInstructionInput(tensor_map, res, inst, 2, prog,
                                             debug_name_and_id));

    // Create a binary op with the scatter_root opcode
    TF_ASSIGN_OR_RETURN(init_val, BroadcastTensor(init_val, output_shape));

    TF_ASSIGN_OR_RETURN(popops::expr::BinaryOpType op,
                        LookupBinaryFn(scatter_root));

    popops::mapInPlace(graph, op, out, init_val, prog,
                       {debug_name_and_id, "initval"});
  }

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return prog;
}

StatusOr<poplar::program::Sequence> CreateReplicatedAllReduce(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const gcl::CollectiveOperator op,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::program::Sequence seq({}, debug_name_and_id);
  poplar::Graph& graph = GetGraph(res, inst);

  TF_ASSIGN_OR_RETURN(
      auto tensors,
      FindInplaceOutputTensors(tensor_map, res, inst, seq, debug_name_and_id));
  CHECK_EQ(tensors.size(), inst->operand_count());
  std::vector<poplar::Tensor> flat_tensors;
  for (auto operand_tensors : tensors) {
    flat_tensors.insert(flat_tensors.end(), operand_tensors.begin(),
                        operand_tensors.end());
  }

  // Only do the all reduce when there are multiple replicas.
  if (res.replication_factor > 1) {
    TF_ASSIGN_OR_RETURN(
        const auto replica_groups,
        PoplarReplicaGroups::FromXlaReplicaGroups(inst->replica_groups()));

    TF_ASSIGN_OR_RETURN(const auto gcl_comm_group,
                        ToGclCommGroup(replica_groups, res));

    // Use multi-tensor allReduce to reduce them all at the same time even if
    // they have different types.
    gcl::allReduceInPlaceCrossReplica(GetMasterGraph(res), flat_tensors, op,
                                      seq, gcl_comm_group, {debug_name_and_id},
                                      GetReplicatedCollectiveOptions(res));
  }

  for (int64 i = 0; i != flat_tensors.size(); ++i) {
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, flat_tensors[i]));
  }

  return seq;
}

StatusOr<poplar::program::Sequence> CreateReplicatedAllToAll(
    CompilerResources& res, const HloInstruction* inst, const xla::Shape&,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::program::Sequence seq({}, debug_name_and_id);
  poplar::Graph& graph = GetGraph(res, inst);

  // Re-concat the incoming tensor slices.
  std::vector<poplar::Tensor> incoming_slices(inst->operand_count());
  for (int i = 0; i < inst->operand_count(); ++i) {
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor input,
        FindInstructionInput(tensor_map, res, inst, i, seq, debug_name_and_id));
    incoming_slices[i] = input.expand({0});
  }

  // Create a single tensor target.
  poplar::Tensor target_input = poplar::concat(incoming_slices);

  // Reshape it to be in the shape [num_splits][split_shape]
  std::vector<std::size_t> sub_tensor_shape = target_input[0].shape();
  sub_tensor_shape.insert(sub_tensor_shape.begin(), inst->operand_count());

  target_input = target_input.reshape(sub_tensor_shape);

  poplar::Tensor output_tensor;

  // If we aren't part of a replicated graph, then just duplicate the tensor.
  if (res.replication_factor < 2) {
    output_tensor = poputil::duplicate(
        graph, target_input, seq, {debug_name_and_id},
        poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
  } else {
    // Perfom the actual Replica->Replica exchange version.
    output_tensor =
        gcl::allToAllCrossReplica(graph, target_input, seq, {debug_name_and_id},
                                  GetReplicatedCollectiveOptions(res));
  }

  for (int i = 0; i < inst->operand_count(); ++i) {
    // Add each slice of the tensor as an output and expand to be
    // [1][output_shape] to match what XLA expects.
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, output_tensor[i]));
  }

  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
