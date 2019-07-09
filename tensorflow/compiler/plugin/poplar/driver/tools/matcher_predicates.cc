#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/norm.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/pooling.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/relu.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/sigmoid.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

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
  const auto dim_numbers = inst->scatter_dimension_numbers();
  const auto update_window_dims = dim_numbers.update_window_dims();
  const auto inserted_window_dims = dim_numbers.inserted_window_dims();
  const auto scatter_dims_to_operand_dims =
      dim_numbers.scatter_dims_to_operand_dims();

  return !(
      (inst->operand(0)->shape().rank() != 2) ||
      (inst->operand(2)->shape().rank() != 2) ||
      (scatter_dims_to_operand_dims.size() != 1) ||
      (inserted_window_dims.size() != 1 || (update_window_dims.size()) != 1));
}
}  // namespace

static bool IsAllFloatValue(const HloInstruction* inst, const double value) {
  return !ShapeUtil::IsZeroElementArray(inst->shape()) &&
         inst->literal().IsAllFloat(value);
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
  return !ShapeUtil::IsZeroElementArray(inst->shape()) &&
         inst->literal().IsAll(0);
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

  if (rev.size() != d.kernel_spatial_dimensions_size()) return false;
  for (int64 i = 0; i < rev.size(); i++) {
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
  if (dims.size() != inst->operand(0)->shape().rank() - 1) {
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

bool IsPopOpsBiasAdd(const HloInstruction* inst) {
  return IsPopOpsFusion(inst, "matmul_biasadd") ||
         IsPopOpsFusion(inst, "conv_biasadd");
}

bool IsPopOpsElementwise(const HloInstruction* inst) {
  if (auto* poplar_inst = DynCast<HloPoplarInstruction>(inst)) {
    return poplar_inst->IsPopOpsElementwise();
  }
  return IsPopOpsBiasAdd(inst) || IsPopOpsFusion(inst, "scaled_inplace") ||
         inst->IsElementwise();
}

bool IsPopOpsElementwiseBinary(const HloInstruction* inst) {
  // Scaled inplace is a special case because it has 3 operands but the 3rd one
  // is always constant - we consider it a binary op.
  return (IsPopOpsElementwise(inst) && inst->operand_count() == 2) ||
         IsPopOpsFusion(inst, "scaled_inplace");
}

bool IsNormInference(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kBatchNormInference ||
         DynCast<HloGroupNormInstruction>(inst);
}

bool IsNormTraining(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kBatchNormTraining ||
         DynCast<HloGroupNormTrainInstruction>(inst);
}

bool IsNormInferenceOrTraining(const HloInstruction* inst) {
  return IsNormTraining(inst) || IsNormInference(inst);
}

bool IsNormGradient(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kBatchNormGrad ||
         DynCast<HloGroupNormGradInstruction>(inst);
}

bool IsNonLinearity(const HloInstruction* inst) {
  return DynCast<HloReluInstruction>(inst) != nullptr ||
         DynCast<HloSigmoidInstruction>(inst) != nullptr;
}

bool IsNonLinearityGradient(const HloInstruction* inst) {
  return DynCast<HloReluGradInstruction>(inst) != nullptr ||
         DynCast<HloSigmoidGradInstruction>(inst) != nullptr;
}

bool IsCompareEqual(const HloInstruction* inst) {
  return inst->comparison_direction() == ComparisonDirection::kEq;
}

bool IsSupportedAllReduce(const HloInstruction* inst) {
  if (auto all_reduce = DynCast<HloAllReduceInstruction>(inst)) {
    auto root = all_reduce->to_apply()->root_instruction();
    return Match(root, m::Add(m::Parameter(0), m::Parameter(1)));
  }
  return false;
}

bool IsMultiUpdate(const HloInstruction* inst) {
  if (auto scatter = DynCast<HloScatterInstruction>(inst)) {
    auto root_inst = scatter->to_apply()->root_instruction();
    return Match(root_inst, m::Parameter(1)) && CheckValidAttributes(scatter);
  }
  return false;
}

bool IsMultiUpdateAdd(const HloInstruction* inst) {
  if (auto scatter = DynCast<HloScatterInstruction>(inst)) {
    auto root_inst = scatter->to_apply()->root_instruction();
    return Match(root_inst, m::Add(m::Parameter(0), m::Parameter(1))) &&
           CheckValidAttributes(scatter);
  }
  return false;
}

}  // namespace poplarplugin
}  // namespace xla
