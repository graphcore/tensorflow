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
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include <functional>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/norm.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/pooling.h"
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
bool HasSingleUser(const HloInstruction* inst) {
  return inst->user_count() == 1;
}

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
  return IsPoplarInstruction(PoplarOp::ConvWithReverse, inst);
}

bool IsPopOpsConvolutionWithReverse(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::ConvWithReverse, inst);
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

bool IsAdd(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kAdd;
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

bool IsAnyScaledInplace(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::ScaledInplaceXbY)(inst) ||
         IsPoplarInstruction(PoplarOp::ScaledInplaceaXbY)(inst);
}

bool IsPopOpsElementwise(const HloInstruction* inst) {
  if (auto* poplar_inst = DynCast<HloPoplarInstruction>(inst)) {
    return poplar_inst->IsPopOpsElementwise();
  }
  return IsAnyScaledInplace(inst) || inst->IsElementwise();
}

bool IsPopOpsElementwiseBinary(const HloInstruction* inst) {
  // Scaled inplace is a special case because it has 3 operands but the 3rd one
  // is always constant - we consider it a binary op.
  return (IsPopOpsElementwise(inst) && inst->operand_count() == 2) ||
         IsAnyScaledInplace(inst);
}

bool IsPopOpsElementwiseBinaryOperandsDifferent(const HloInstruction* inst) {
  // Check binary operation has same argument as both binary inputs
  return IsPopOpsElementwiseBinary(inst) &&
         inst->operand(0) != inst->operand(1);
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

bool IsCompareEqual(const HloInstruction* inst) {
  return inst->comparison_direction() == ComparisonDirection::kEq;
}

bool IsCompareNotEqual(const HloInstruction* inst) {
  return inst->comparison_direction() == ComparisonDirection::kNe;
}

bool IsCompareLess(const HloInstruction* inst) {
  return inst->comparison_direction() == ComparisonDirection::kLt;
}

bool IsCompareLessOrEqual(const HloInstruction* inst) {
  return inst->comparison_direction() == ComparisonDirection::kLe;
}

bool IsCompareGreater(const HloInstruction* inst) {
  return inst->comparison_direction() == ComparisonDirection::kGt;
}

bool IsCompareGreaterOrEqual(const HloInstruction* inst) {
  return inst->comparison_direction() == ComparisonDirection::kGe;
}

bool IsSupportedAllReduce(const HloInstruction* inst) {
  return IsAllReduceAdd(inst) || IsAllReduceMean(inst);
}

bool IsMultiSliceOrUpdate(const HloInstruction* inst) {
  for (auto op : {PoplarOp::MultiSlice, PoplarOp::MultiUpdate,
                  PoplarOp::MultiUpdateAdd}) {
    if (IsPoplarInstruction(op)(inst)) {
      return true;
    }
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

bool IsReductionFusion(const HloInstruction* inst) {
  return IsPopOpsFusion(inst, "reduction_fp16_input") ||
         IsPopOpsFusion(inst, "reduction_square_add");
}

bool IsWideConstant(const HloInstruction* inst) {
  return IsPopOpsFusion(inst, "wide_const");
}

bool IsWideConstantZero(const HloInstruction* inst) {
  if (IsWideConstant(inst)) {
    const HloInstruction* fusion_root = inst->fused_expression_root();
    return IsConstantZero(fusion_root->operand(0));
  }
  if (inst->opcode() == HloOpcode::kBroadcast) {
    auto dims = inst->dimensions();
    return dims.size() == 0 ? IsConstantZero(inst->operand(0)) : false;
  }
  return IsConstantZero(inst);
}

bool IsUniformSingleDimSlice(const HloInstruction* slice) {
  // All the strides are 1.
  if (absl::c_any_of(slice->slice_strides(),
                     [](int64 stride) { return stride != 1; })) {
    return false;
  }
  // Only one dimension is sliced.
  int64 num_sliced_dims = 0;
  for (int64 i = 0; i != slice->shape().rank(); ++i) {
    if (slice->shape().dimensions(i) !=
        slice->operand(0)->shape().dimensions(i)) {
      num_sliced_dims++;
    }
  }
  return num_sliced_dims == 1;
}

bool IsSingleElement(const HloInstruction* inst) {
  return ShapeUtil::ElementsIn(inst->shape()) == 1;
}

namespace {
bool IsReduceWithRootOp(const HloInstruction* inst, HloOpcode opcode) {
  if (inst->opcode() == HloOpcode::kReduce) {
    HloInstruction* root(inst->to_apply()->root_instruction());
    if (!hlo_query::AllOperandsAreParameters(*root)) {
      return false;
    }
    return root->opcode() == opcode;
  }
  return false;
}
}  // namespace

bool IsGlobalAllReduce(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kAllReduce &&
         inst->replica_groups().empty();
}

bool IsReduceAdd(const HloInstruction* inst) {
  return IsReduceWithRootOp(inst, HloOpcode::kAdd);
}

bool IsReduceAddOrMultiply(const HloInstruction* inst) {
  return IsReduceWithRootOp(inst, HloOpcode::kAdd) ||
         IsReduceWithRootOp(inst, HloOpcode::kMultiply);
}

bool IsSerializedGradientAccumulation(const HloInstruction* inst) {
  return IsPopOpsFusion(inst, "serialized_gradient_accumulation");
}

std::function<bool(const HloInstruction*)> IsPoplarInstruction(PoplarOp op) {
  return [op](const HloInstruction* inst) -> bool {
    return IsPoplibsHloCustomOp(inst) &&
           inst->custom_call_target() == PoplarOp_Name(op);
  };
}

bool IsAllReduceAdd(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kAllReduce) {
    const HloInstruction* root = inst->to_apply()->root_instruction();
    return Match(root, m::Add(m::Parameter(0), m::Parameter(1)));
  }
  return false;
}

bool IsAllReduceMean(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kAllReduce) {
    return false;
  }

  const HloInstruction* root = inst->to_apply()->root_instruction();
  return Match(root, m::Add(m::Parameter(0),
                            m::Divide(m::Parameter(1), m::ConstantScalar())));
}

}  // namespace poplarplugin
}  // namespace xla
