/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <poplin/MatMul.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/slice_apply.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/generic_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matmul_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/ml_type_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/util/bcast.h"

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {

static const std::string a_conn("a");
static const std::string b_conn("b");
static const std::string c_conn("c");
static const std::string out_conn("out");

#define POPLAR_OPCODE(O, N) \
  case HloOpcode::O:        \
    return std::string(N)
#define UNUSED_OPCODE(O) \
  case HloOpcode::O:     \
    break;

StatusOr<popops::expr::UnaryOpType> LookupUnaryFn(const HloInstruction* inst) {
  HloOpcode opcode = inst->opcode();
  switch (opcode) {
    case HloOpcode::kAbs:
      return popops::expr::UnaryOpType::ABSOLUTE;
    case HloOpcode::kCbrt:
      return popops::expr::UnaryOpType::CBRT;
    case HloOpcode::kCeil:
      return popops::expr::UnaryOpType::CEIL;
    case HloOpcode::kClz:
      return popops::expr::UnaryOpType::COUNT_LEADING_ZEROS;
    case HloOpcode::kCos:
      return popops::expr::UnaryOpType::COS;
    case HloOpcode::kExp:
      return popops::expr::UnaryOpType::EXPONENT;
    case HloOpcode::kExpm1:
      return popops::expr::UnaryOpType::EXPONENT_MINUS_ONE;
    case HloOpcode::kFloor:
      return popops::expr::UnaryOpType::FLOOR;
    case HloOpcode::kLogistic:
      return popops::expr::UnaryOpType::SIGMOID;
    case HloOpcode::kLog:
      return popops::expr::UnaryOpType::LOGARITHM;
    case HloOpcode::kLog1p:
      return popops::expr::UnaryOpType::LOGARITHM_ONE_PLUS;
    case HloOpcode::kNegate:
      return popops::expr::UnaryOpType::NEGATE;
    case HloOpcode::kPopulationCount:
      return popops::expr::UnaryOpType::POPCOUNT;
    case HloOpcode::kRoundNearestAfz:
      return popops::expr::UnaryOpType::ROUND;
    case HloOpcode::kRsqrt:
      return popops::expr::UnaryOpType::RSQRT;
    case HloOpcode::kSign:
      return popops::expr::UnaryOpType::SIGNUM;
    case HloOpcode::kSin:
      return popops::expr::UnaryOpType::SIN;
    case HloOpcode::kSqrt:
      return popops::expr::UnaryOpType::SQRT;
    case HloOpcode::kTanh:
      return popops::expr::UnaryOpType::TANH;
    case HloOpcode::kIsFinite:
      return popops::expr::UnaryOpType::IS_FINITE;
    case HloOpcode::kCustomCall: {
      if (IsPoplarInstruction(PoplarOp::Square, inst)) {
        return popops::expr::UnaryOpType::SQUARE;
      }
      if (IsPoplarInstruction(PoplarOp::Inverse, inst)) {
        return popops::expr::UnaryOpType::INVERSE;
      }
      if (IsPoplarInstruction(PoplarOp::Erf, inst)) {
        return popops::expr::UnaryOpType::ERF;
      }
      TF_FALLTHROUGH_INTENDED;
    }
    default:
      break;
  }

  if (opcode == HloOpcode::kNot) {
    if (inst->shape().element_type() == PRED) {
      return popops::expr::UnaryOpType::LOGICAL_NOT;
    } else {
      return popops::expr::UnaryOpType::BITWISE_NOT;
    }
  }

  return tensorflow::errors::Unknown(
      StrCat("[Poplar] Invalid opcode lookup ", HloOpcodeString(opcode)));
}

StatusOr<popops::expr::BinaryOpType> LookupComparisonFn(
    const HloInstruction* inst) {
  auto direction = inst->comparison_direction();
  switch (direction) {
    case ComparisonDirection::kEq:
      return popops::expr::BinaryOpType::EQUAL;
    case ComparisonDirection::kGt:
      return popops::expr::BinaryOpType::GREATER_THAN;
    case ComparisonDirection::kGe:
      return popops::expr::BinaryOpType::GREATER_THAN_EQUAL;
    case ComparisonDirection::kLt:
      return popops::expr::BinaryOpType::LESS_THAN;
    case ComparisonDirection::kLe:
      return popops::expr::BinaryOpType::LESS_THAN_EQUAL;
    case ComparisonDirection::kNe:
      return popops::expr::BinaryOpType::NOT_EQUAL;
    default:
      break;
  }

  return tensorflow::errors::Unknown(
      StrCat("[Poplar] Invalid opcode lookup ",
             ComparisonDirectionToString(direction)));
}

StatusOr<popops::expr::BinaryOpType> LookupBinaryFn(
    const HloInstruction* inst) {
  HloOpcode opcode = inst->opcode();
  bool is_slice_apply = IsPoplarInstruction(PoplarOp::SliceApply)(inst);
  if (is_slice_apply) {
    const HloSliceApply* slice_apply = Cast<HloSliceApply>(inst);
    opcode = slice_apply->GetOperation();
  }

  switch (opcode) {
    case HloOpcode::kAdd:
      return popops::expr::BinaryOpType::ADD;
    case HloOpcode::kAtan2:
      return popops::expr::BinaryOpType::ATAN2;
    case HloOpcode::kDivide:
      return popops::expr::BinaryOpType::DIVIDE;
    case HloOpcode::kMaximum:
      return popops::expr::BinaryOpType::MAXIMUM;
    case HloOpcode::kMinimum:
      return popops::expr::BinaryOpType::MINIMUM;
    case HloOpcode::kMultiply:
      return popops::expr::BinaryOpType::MULTIPLY;
    case HloOpcode::kPower:
      return popops::expr::BinaryOpType::POWER;
    case HloOpcode::kRemainder:
      return popops::expr::BinaryOpType::REMAINDER;
    case HloOpcode::kShiftLeft:
      return popops::expr::BinaryOpType::SHIFT_LEFT;
    case HloOpcode::kShiftRightArithmetic:
      return popops::expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND;
    case HloOpcode::kShiftRightLogical:
      return popops::expr::BinaryOpType::SHIFT_RIGHT;
    case HloOpcode::kSubtract:
      return popops::expr::BinaryOpType::SUBTRACT;
    case HloOpcode::kCompare:
      if (!is_slice_apply) {
        return LookupComparisonFn(inst);
      }
    default:
      break;
  }

  if (opcode == HloOpcode::kAnd) {
    if (inst->shape().element_type() == PRED) {
      return popops::expr::BinaryOpType::LOGICAL_AND;
    } else {
      return popops::expr::BinaryOpType::BITWISE_AND;
    }
  }

  if (opcode == HloOpcode::kOr) {
    if (inst->shape().element_type() == PRED) {
      return popops::expr::BinaryOpType::LOGICAL_OR;
    } else {
      return popops::expr::BinaryOpType::BITWISE_OR;
    }
  }

  if (opcode == HloOpcode::kXor) {
    if (inst->shape().element_type() == PRED) {
      return popops::expr::BinaryOpType::NOT_EQUAL;
    } else {
      return popops::expr::BinaryOpType::BITWISE_XOR;
    }
  }

  return tensorflow::errors::Unknown(
      StrCat("[Poplar] Invalid opcode lookup ", HloOpcodeString(opcode)));
}

StatusOr<popops::expr::TernaryOpType> LookupTernaryFn(
    const HloInstruction* inst) {
  const HloOpcode opcode = inst->opcode();
  switch (opcode) {
    case HloOpcode::kClamp:
      return popops::expr::TernaryOpType::CLAMP;
    case HloOpcode::kSelect:
      return popops::expr::TernaryOpType::SELECT;
    default:
      return tensorflow::errors::Unknown(
          StrCat("[Poplar] Invalid opcode lookup ", HloOpcodeString(opcode)));
  }
}

StatusOr<poplar::program::Sequence> CreateTupleSelectOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq({}, debug_name_and_id);

  TF_ASSIGN_OR_RETURN(poplar::Tensor pred,
                      FindInstructionInput(tensor_map, res, inst, 0, seq,
                                           debug_name_and_id, false));

  TF_ASSIGN_OR_RETURN(TensorVector in0,
                      FindInstructionInputTensors(tensor_map, res, inst, 1, seq,
                                                  debug_name_and_id, false));
  TF_ASSIGN_OR_RETURN(TensorVector in1,
                      FindInstructionInputTensors(tensor_map, res, inst, 2, seq,
                                                  debug_name_and_id, false));

  if (in0.size() != in1.size()) {
    return xla::FailedPrecondition("Mismatching tuple sizes on %s",
                                   inst->name().c_str());
  }

  for (unsigned int i = 0; i < in0.size(); i++) {
    poplar::Tensor p = pred;
    poplar::Tensor i0 = in0[i];
    poplar::Tensor i1 = in1[i];

    if (p.numElements() == 1) {
      p = p.reshape({1});
      p = p.broadcast(i0.numElements(), 0);
      p = p.reshape(i0.shape());
    }

    poplar::Tensor out = popops::map(graph, popops::expr::TernaryOpType::SELECT,
                                     i0, i1, p, seq, {debug_name_and_id});

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, out));
  }

  return seq;
}

namespace {
template <typename T>
Status DoScaledInplaceConstantOrTensor(
    poplar::Graph& graph, poplar::Tensor& lhs, poplar::Tensor& rhs, T scale,
    poplar::program::Sequence& prog, const HloOpcode op_type,
    const poplar::DebugNameAndId& debug_name_and_id) {
  // Call the inplace op
  switch (op_type) {
    case HloOpcode::kAdd: {
      popops::scaledAddTo(graph, lhs, rhs, scale, prog, {debug_name_and_id});
      break;
    }
    case HloOpcode::kSubtract: {
      popops::scaledSubtractFrom(graph, lhs, rhs, scale, prog,
                                 {debug_name_and_id});
      break;
    }
    default: {
      return xla::FailedPrecondition("Unsupported scaled inplace op: %s",
                                     debug_name_and_id.getPathName());
    }
  }
  return Status::OK();
}

template <typename T>
Status DoScaledInplaceConstantOrTensor(
    poplar::Graph& graph, poplar::Tensor& tensor_a, T scale_a,
    poplar::Tensor& tensor_b, T scale_b, poplar::program::Sequence& prog,
    const HloOpcode op_type, const poplar::DebugNameAndId& debug_name_and_id) {
  // Call the inplace op
  switch (op_type) {
    case HloOpcode::kAdd: {
      popops::scaledAddTo(graph, tensor_a, scale_a, tensor_b, scale_b, prog,
                          {debug_name_and_id});
      break;
    }
    case HloOpcode::kSubtract: {
      popops::scaledSubtractFrom(graph, tensor_a, scale_a, tensor_b, scale_b,
                                 prog, {debug_name_and_id});
      break;
    }
    default: {
      return xla::FailedPrecondition("Unsupported scaled inplace op: %s",
                                     debug_name_and_id.getPathName());
    }
  }
  return Status::OK();
}
}  // namespace

Status ScaledInplaceConstantOrTensor(
    poplar::Graph& graph, poplar::Tensor& lhs, poplar::Tensor& rhs,
    const double scale, poplar::program::Sequence& prog,
    const HloOpcode op_type, const poplar::DebugNameAndId& debug_name_and_id) {
  return DoScaledInplaceConstantOrTensor(graph, lhs, rhs, scale, prog, op_type,
                                         debug_name_and_id);
}

Status ScaledInplaceConstantOrTensor(
    poplar::Graph& graph, poplar::Tensor& lhs, poplar::Tensor& rhs,
    poplar::Tensor& scale, poplar::program::Sequence& prog,
    const HloOpcode op_type, const poplar::DebugNameAndId& debug_name_and_id) {
  return DoScaledInplaceConstantOrTensor(graph, lhs, rhs, scale, prog, op_type,
                                         debug_name_and_id);
}

Status ScaledInplaceConstantOrTensor(
    poplar::Graph& graph, poplar::Tensor& tensor_a, const double scale_a,
    poplar::Tensor& tensor_b, const double scale_b,
    poplar::program::Sequence& prog, const HloOpcode op_type,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return DoScaledInplaceConstantOrTensor(graph, tensor_a, scale_a, tensor_b,
                                         scale_b, prog, op_type,
                                         debug_name_and_id);
}

Status ScaledInplaceConstantOrTensor(
    poplar::Graph& graph, poplar::Tensor& tensor_a, poplar::Tensor& scale_a,
    poplar::Tensor& tensor_b, poplar::Tensor& scale_b,
    poplar::program::Sequence& prog, const HloOpcode op_type,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return DoScaledInplaceConstantOrTensor(graph, tensor_a, scale_a, tensor_b,
                                         scale_b, prog, op_type,
                                         debug_name_and_id);
}

StatusOr<poplar::program::Sequence> CreateMatMulForDotOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq({}, debug_name_and_id);

  CHECK_EQ(inst->opcode(), HloOpcode::kDot);

  // Do not expand aliasing when creating a cached op - the input will be
  // reallocated if required.
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor arg_lhs,
      FindInstructionInput(tensor_map, res, inst, 0, seq, debug_name_and_id,
                           /*expand_aliasing*/ false));
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor arg_rhs,
      FindInstructionInput(tensor_map, res, inst, 1, seq, debug_name_and_id,
                           /*expand_aliasing*/ false));

  const DotDimensionNumbers dot_dims = inst->dot_dimension_numbers();
  TF_ASSIGN_OR_RETURN(const std::string dot_type_s, GetMLTypeAsString(inst));
  TF_ASSIGN_OR_RETURN(const poplar::OptionFlags opts,
                      GetMatMulOptionsForInst(inst, res));

  // Created a cached dot.
  auto func = [&graph, &res, &output_shape, dot_dims, dot_type_s, &opts,
               debug_name_and_id](std::vector<poplar::Tensor>& args,
                                  poplar::program::Sequence& prog) {
    poplar::Tensor lhs = args[0];
    poplar::Tensor rhs = args[1];

    auto lhs_reduction_dimensions = dot_dims.lhs_contracting_dimensions();
    auto rhs_reduction_dimensions = dot_dims.rhs_contracting_dimensions();
    auto lhs_batch_dimensions = dot_dims.lhs_batch_dimensions();
    auto rhs_batch_dimensions = dot_dims.rhs_batch_dimensions();

    // DimShuffle the LHS to [Batch..., M..., Contracting...]
    std::vector<unsigned> lhs_permutation =
        LeftMatMulPermutations(lhs.shape(), dot_dims);
    lhs = lhs.dimShuffle(lhs_permutation);

    // DimShuffle the RHS to [Batch..., Contracting..., N...]
    std::vector<unsigned> rhs_permutation =
        RightMatMulPermutations(rhs.shape(), dot_dims);
    rhs = rhs.dimShuffle(rhs_permutation);

    // Collapse the LHS dimensions down to [Batch, M, Contracting]
    lhs = lhs.reshape(LeftMatMulPackShape(lhs.shape(), dot_dims));

    // Collapse the RHS dimensions down to [Batch, Contracting, N]
    rhs = rhs.reshape(RightMatMulPackShape(rhs.shape(), dot_dims));

    if (VLOG_IS_ON(2)) {
      std::stringstream stream;
      poplin::matMulGroupedReportPlan(stream, graph, lhs.elementType(),
                                      lhs.elementType(), lhs.shape(),
                                      rhs.shape(), opts, &res.matmul_cache);
      VLOG(2) << "MatMul " << debug_name_and_id.getPathName() << ". Type "
              << dot_type_s << (res.clear_matmul_pass_type ? " (cleared)" : "")
              << ". Plan " << stream.str();
      for (auto opt : opts) {
        VLOG(2) << "- option: " << opt.first << " = " << opt.second;
      }
    }

    args[2] =
        poplin::matMulGrouped(graph, lhs, rhs, prog, lhs.elementType(),
                              {debug_name_and_id}, opts, &res.matmul_cache);
    // Reshape to XLA shape
    args[2] = args[2].reshape(PoplarShapeFromXlaShape(output_shape));
  };

  poplar::Tensor output;
  std::vector<poplar::Tensor> args = {arg_lhs, arg_rhs, output};
  poputil::graphfn::Signature sig = {poputil::graphfn::input(arg_lhs, "lhs"),
                                     poputil::graphfn::input(arg_rhs, "rhs"),
                                     poputil::graphfn::created("output")};
  TF_RETURN_IF_ERROR(res.graph_cache.ExecuteCached(inst, graph, res, seq, func,
                                                   sig, args, {0, 1}));

  output = args[2];

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));

  return seq;
}

StatusOr<poplar::program::Sequence> CreateCastOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq({}, debug_name_and_id);

  // TODO(T16423) - Do not expand aliasing when casting.
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor in,
      FindInstructionInput(tensor_map, res, inst, 0, seq, debug_name_and_id));

  TF_ASSIGN_OR_RETURN(poplar::Type poplar_type, PoplarDataType(output_shape));

  poplar::Tensor out =
      popops::cast(graph, in, poplar_type, seq, {debug_name_and_id});

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}
}  // namespace poplarplugin
}  // namespace xla
