/* Copyright 2018-2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include <functional>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/norm.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/pooling.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/relu.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/sigmoid.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/window_util.h"
namespace xla {

namespace m = match;

namespace poplarplugin {
namespace {
// TODO popops::multiUpdate and popops::multiUpdateAdd only supports the 2D
// case.
bool CheckValidAttributes(const HloScatterInstruction* inst) {
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

bool CheckValidAttributesForGather(const HloGatherInstruction* inst) {
  const Shape operand_shape = inst->operand(0)->shape();
  const Shape start_indices = inst->operand(1)->shape();
  const auto dim_numbers = inst->gather_dimension_numbers();
  const auto offset_dims = dim_numbers.offset_dims();
  const auto collapsed_slice_dims = dim_numbers.collapsed_slice_dims();
  const auto start_index_map = dim_numbers.start_index_map();
  const auto index_vector_dim = dim_numbers.index_vector_dim();
  const auto slice_sizes = inst->gather_slice_sizes();

  const uint64 index_dim_size =
      start_indices.rank() == index_vector_dim
          ? 1
          : start_indices.dimensions(index_vector_dim);

  // Currently we only allow operand rank 2.
  if (operand_shape.rank() != 2) {
    return false;
  }

  // For this case vector of collapsed dimensions should have size 1.
  if (collapsed_slice_dims.size() != 1) {
    return false;
  }

  // Allow collapsed axis to be only 0 as multiSlice
  // would not handle 1 at the moment. Next task also allow 1
  // and do multiSlice(transpose(operand)).
  int collapsed_slice_dim = collapsed_slice_dims[0];
  if (collapsed_slice_dim != 0) {
    return false;
  }

  // Non collapsed axis is orthogonal to collapsed one, can be 0 or 1.
  int non_collapsed_dim = 1 - collapsed_slice_dim;

  // Non collapsed axis of operand shape should be same as
  // non collapsed axis of slice sizes.
  if (operand_shape.dimensions(non_collapsed_dim) !=
      slice_sizes[non_collapsed_dim]) {
    return false;
  }

  // Collapsed axis of slice sizes must have dimension 1.
  if (slice_sizes[collapsed_slice_dim] != 1) {
    return false;
  }

  // Size of offset dims must be 1.
  if (offset_dims.size() != 1) {
    return false;
  }

  // Offset axis must be same as non collapsed axis.
  if (offset_dims[0] != non_collapsed_dim) {
    return false;
  }

  if (index_dim_size != 1) {
    return false;
  }

  return true;
}

}  // namespace

bool IsRandomNormal(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kRng &&
         inst->random_distribution() == RandomDistribution::RNG_NORMAL;
}

bool IsRandomUniform(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kRng &&
         inst->random_distribution() == RandomDistribution::RNG_UNIFORM;
}

bool IsConstantZero(const HloInstruction* inst) {
  return inst->IsConstant() && !ShapeUtil::IsZeroElementArray(inst->shape()) &&
         inst->literal().IsAll(0);
}

bool IsConstantOne(const HloInstruction* inst) {
  return inst->IsConstant() && !ShapeUtil::IsZeroElementArray(inst->shape()) &&
         inst->literal().IsAll(1);
}

bool IsExternalPadding(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kPad) {
    return false;
  }

  const PaddingConfig& cfg(inst->padding_config());
  for (auto& d : cfg.dimensions()) {
    if (d.interior_padding() > 0) return false;
  }
  return true;
}

bool Is2DReductionWindow(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kReduceWindow) {
    return false;
  }

  const Window& window(inst->window());
  int reduction_count = 0;
  for (int64 i = 0; i < window.dimensions_size(); i++) {
    if (window.dimensions(i).size() != 1 ||
        window.dimensions(i).stride() != 1 ||
        window.dimensions(i).padding_low() != 0 ||
        window.dimensions(i).padding_high() != 0) {
      reduction_count++;
    }
  }
  return reduction_count == 2;
}

bool IsFloat(const HloInstruction* inst) {
  return ShapeUtil::ElementIsFloating(inst->shape());
}

bool IsScalar(const HloInstruction* inst) {
  return ShapeUtil::IsScalar(inst->shape());
}

bool IsFloatScalar(const HloInstruction* inst) {
  return IsScalar(inst) && IsFloat(inst);
}

bool IsScalarConstant(const HloInstruction* inst) {
  return IsScalar(inst) && inst->IsConstant();
}

bool IsFloatScalarConstant(const HloInstruction* inst) {
  return IsScalarConstant(inst) && IsFloat(inst);
}

bool IsScalarIntegerConstant(const HloInstruction* inst) {
  return IsScalarConstant(inst) && ShapeUtil::ElementIsIntegral(inst->shape());
}

bool IsConvFilterTranspose(const HloInstruction* inst) {
  // If this reverse feeds a convolution and it is reversing the
  // spatial dimensions of the convolution, then we can use the
  // special 'reverse spatial dimensions' feature of the convolution
  // to achieve the reverse
  if (inst->users().size() != 1) return false;
  const std::vector<int64>& rev(inst->dimensions());

  HloInstruction* conv = inst->users()[0];
  if (conv->opcode() != HloOpcode::kConvolution) {
    return false;
  }
  const ConvolutionDimensionNumbers& d(conv->convolution_dimension_numbers());

  if (rev.size() != static_cast<size_t>(d.kernel_spatial_dimensions_size()))
    return false;
  for (size_t i = 0; i < rev.size(); i++) {
    if (d.kernel_spatial_dimensions(i) != rev[i]) return false;
  }

  return true;
}

bool IsBiasReduce(const HloInstruction* inst) {
  HloInstruction* root(inst->to_apply()->root_instruction());
  if (!hlo_query::AllOperandsAreParameters(*root)) {
    return false;
  }
  if (root->opcode() != HloOpcode::kAdd) {
    return false;
  }

  if (inst->shape().rank() != 1) return false;

  const std::vector<int64>& dims(inst->dimensions());
  if (static_cast<int64>(dims.size()) != inst->operand(0)->shape().rank() - 1) {
    return false;
  }
  return true;
}

bool IsOutputFeed(const HloInstruction* inst) {
  HloInstruction* root = inst->parent()->root_instruction();
  if (inst == root) return true;
  if (inst->user_count() != 1) return false;
  if (inst->users()[0] == root) return true;
  return false;
}

bool Is1DVector(const HloInstruction* inst) {
  return inst->shape().rank() == 1;
}

bool IsExpandingReshape(const HloInstruction* inst) {
  return ShapeUtil::TrueRank(inst->shape()) == 1;
}

bool IsF16(const HloInstruction* inst) {
  return inst->shape().element_type() == PrimitiveType::F16;
}

bool IsF32(const HloInstruction* inst) {
  return inst->shape().element_type() == PrimitiveType::F32;
}

bool IsF16OrF32(const HloInstruction* inst) {
  return IsF16(inst) || IsF32(inst);
}

bool IsF32ToF16Convert(const HloInstruction* inst) {
  return IsF16(inst) && IsF32(inst->operand(0));
}

bool IsF16ToF32Convert(const HloInstruction* inst) {
  return IsF32(inst) && IsF16(inst->operand(0));
}

bool IsPopOpsConvolution(const HloInstruction* inst) {
  if (IsPopOpsFusion(inst, "depthwise_conv")) return true;
  if (IsPopOpsFusion(inst, "conv_with_reverse")) return true;
  if (IsPopOpsFusion(inst, "depthwise_filter")) return true;
  return false;
}

bool IsPopOpsConvolutionWithReverse(const HloInstruction* inst) {
  return (IsPopOpsFusion(inst, "conv_with_reverse"));
}

bool IsOpWithWindowNoBaseDilation(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kConvolution:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kSelectAndScatter:
      return !window_util::HasBaseDilation(inst->window());
    default:
      return false;
  }
}

bool IsOpWithWindowNoStride(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kConvolution:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kSelectAndScatter:
      return !window_util::HasStride(inst->window());
    default:
      return false;
  }
}

bool IsPaddingReduceWindow(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kReduceWindow) return false;
  // it's an identity, which means the window is of dim 1x...x1 and root of
  // to_apply computation is a parameter num 1
  const auto window = inst->window();
  for (const auto dim : window.dimensions()) {
    if (dim.size() != 1) return false;
  }
  const auto* root = inst->to_apply()->root_instruction();
  if (root->opcode() != HloOpcode::kParameter) return false;
  return root->parameter_number() == 1;
}

bool IsBiasAdd(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kAdd) {
    return false;
  }
  const auto& op_shape = inst->operand(0)->shape();
  const auto& bias_shape = inst->operand(1)->shape();
  if (op_shape.rank() != bias_shape.rank()) {
    return false;
  }

  // Go through the bias shape, if the dimension size is 1, then the dimension
  // of the op doesn't matter, otherwise they have to match.
  for (int64 i = 0; i < bias_shape.rank(); i++) {
    int64 bias_dim = ShapeUtil::GetDimension(bias_shape, i);
    if (bias_dim != 1) {
      int64 op_dim = ShapeUtil::GetDimension(op_shape, i);
      if (bias_dim != op_dim) {
        return false;
      }
    }
  }
  return true;
}

bool IsAddOrSubtract(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kAdd ||
         inst->opcode() == HloOpcode::kSubtract;
}

bool IsMultiplyOrDivide(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kMultiply ||
         inst->opcode() == HloOpcode::kDivide;
}

bool IsPopOpsBiasAdd(const HloInstruction* inst) {
  return IsPopOpsFusion(inst, "matmul_biasadd") ||
         IsPopOpsFusion(inst, "conv_biasadd");
}

bool IsPopOpsElementwise(const HloInstruction* inst) {
  if (auto* poplar_inst = DynCast<HloPoplarInstruction>(inst)) {
    return poplar_inst->IsPopOpsElementwise();
  }
  return IsPopOpsFusion(inst, "scaled_inplace") || inst->IsElementwise();
}

bool IsPopOpsElementwiseBinary(const HloInstruction* inst) {
  // Scaled inplace is a special case because it has 3 operands but the 3rd one
  // is always constant - we consider it a binary op.
  return (IsPopOpsElementwise(inst) && inst->operand_count() == 2) ||
         IsPopOpsFusion(inst, "scaled_inplace");
}

bool IsNormInference(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kBatchNormInference ||
         IsPoplarInstruction(PoplarOp::GroupNormInference)(inst);
}

bool IsNormTraining(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kBatchNormTraining ||
         IsPoplarInstruction(PoplarOp::GroupNormTraining)(inst);
}

bool IsNormInferenceOrTraining(const HloInstruction* inst) {
  return IsNormTraining(inst) || IsNormInference(inst);
}

bool IsNormGradient(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kBatchNormGrad ||
         IsPoplarInstruction(PoplarOp::GroupNormGrad)(inst);
}

bool IsNonLinearity(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::Relu)(inst) ||
         IsPoplarInstruction(PoplarOp::Sigmoid)(inst);
}

bool IsNonLinearityGradient(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::ReluGrad)(inst) ||
         IsPoplarInstruction(PoplarOp::SigmoidGrad)(inst);
}

bool IsCompareEqual(const HloInstruction* inst) {
  return inst->comparison_direction() == ComparisonDirection::kEq;
}

bool IsSupportedAllReduce(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kAllReduce) {
    const HloInstruction* root = inst->to_apply()->root_instruction();
    return Match(root, m::Add(m::Parameter(0), m::Parameter(1)));
  }
  return false;
}

bool IsMultiUpdateScatter(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kScatter) {
    const HloScatterInstruction* scatter = Cast<HloScatterInstruction>(inst);
    const HloInstruction* root = inst->to_apply()->root_instruction();
    return Match(root, m::Parameter(1)) && CheckValidAttributes(scatter);
  }
  return false;
}

bool IsMultiUpdateAddScatter(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kScatter) {
    const HloScatterInstruction* scatter = Cast<HloScatterInstruction>(inst);
    const HloInstruction* root = inst->to_apply()->root_instruction();
    return Match(root, m::Add(m::Parameter(0), m::Parameter(1))) &&
           CheckValidAttributes(scatter);
  }
  return false;
}

bool IsMultiSliceGather(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kGather) {
    const HloGatherInstruction* gather = Cast<HloGatherInstruction>(inst);
    return CheckValidAttributesForGather(gather);
  }
  return false;
}

bool IsAnySliceApply(const HloInstruction* inst) {
  for (auto op : {PoplarOp::SliceApply, PoplarOp::SliceApplyaXbY,
                  PoplarOp::SliceApplyabY, PoplarOp::SliceApplyaXb}) {
    if (IsPoplarInstruction(op)(inst)) {
      return true;
    }
  }
  return false;
}

std::function<bool(const HloInstruction*)> IsPoplarInstruction(PoplarOp op) {
  return [op](const HloInstruction* inst) -> bool {
    return IsPoplibsHloCustomOp(inst) &&
           inst->custom_call_target() == PoplarOp_Name(op);
  };
}

}  // namespace poplarplugin
}  // namespace xla
