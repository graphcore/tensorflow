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
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_util.h"

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/conv_with_reverse.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/weights_transpose_chans_flip_xy.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hash.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/window_util.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

#include "google/protobuf/util/message_differencer.h"

namespace xla {
namespace poplarplugin {
namespace {
bool IsCustomConvolutionInstruction(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::WeightsTransposeChansFlipXY, inst) ||
         IsPoplarInstruction(PoplarOp::ConvWithReverse, inst) ||
         IsPoplarInstruction(PoplarOp::F8Conv2D, inst) ||
         IsPoplarInstruction(PoplarOp::F8Conv3D, inst);
}
}  // namespace

StatusOr<Window> GetConvolutionWindow(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kFusion) {
    TF_ASSIGN_OR_RETURN(auto cfg, inst->backend_config<PoplarBackendConfig>());
    return cfg.fusion_config().window();
  } else if (inst->opcode() == HloOpcode::kCustomCall) {
    if (IsCustomConvolutionInstruction(inst)) {
      return inst->window();
    } else {
      return InternalError(
          "Trying to access window on non "
          "HloWeightsTransposeChansFlipXYInstruction"
          "operation.");
    }
  } else {
    if (!CastOrNull<HloConvolutionInstruction>(inst)) {
      return InternalError(
          "Trying to access convolution window on a non convolution "
          "operation.");
    }
    return inst->window();
  }
}

StatusOr<ConvolutionDimensionNumbers> GetConvolutionDims(
    const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kFusion) {
    TF_ASSIGN_OR_RETURN(auto cfg, inst->backend_config<PoplarBackendConfig>());
    return cfg.fusion_config().dimension_numbers();
  } else if (inst->opcode() == HloOpcode::kCustomCall) {
    if (IsCustomConvolutionInstruction(inst)) {
      return inst->convolution_dimension_numbers();
    } else {
      return InternalError(
          "Trying to access convolution_dimension_numbers on a non "
          "HloWeightsTransposeChansFlipXYInstruction.");
    }
  } else {
    if (!CastOrNull<HloConvolutionInstruction>(inst)) {
      return InternalError(
          "Trying to access convolution dimension numbers on a non "
          "convolution operation.");
    }
    return inst->convolution_dimension_numbers();
  }
}

StatusOr<int64_t> GetFeatureGroupCount(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kFusion) {
    TF_ASSIGN_OR_RETURN(auto cfg, inst->backend_config<PoplarBackendConfig>());
    return cfg.fusion_config().feature_group_count();
  } else if (inst->opcode() == HloOpcode::kCustomCall) {
    if (IsCustomConvolutionInstruction(inst)) {
      return inst->feature_group_count();
    } else {
      return InternalError(
          "Trying to access feature_group_count on non "
          "HloWeightsTransposeChansFlipXYInstruction.");
    }
  } else {
    if (!CastOrNull<HloConvolutionInstruction>(inst)) {
      return InternalError(
          "Trying to access convolution feature group count numbers "
          "on a non convolution operation.");
    }
    return inst->feature_group_count();
  }
}

StatusOr<int64_t> GetBatchGroupCount(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kFusion) {
    TF_ASSIGN_OR_RETURN(auto cfg, inst->backend_config<PoplarBackendConfig>());
    return cfg.fusion_config().batch_group_count();
  } else if (inst->opcode() == HloOpcode::kCustomCall) {
    if (IsCustomConvolutionInstruction(inst)) {
      return inst->batch_group_count();
    } else {
      LOG(FATAL) << "Trying to access batch_group_count on non "
                    "HloWeightsTransposeChansFlipXYInstruction.";
    }
  } else {
    if (!CastOrNull<HloConvolutionInstruction>(inst)) {
      LOG(FATAL) << "Trying to access convolution batch group count numbers on "
                    "a non convolution operation.";
    }
    return inst->batch_group_count();
  }
}

StatusOr<PrecisionConfig> GetPrecisionConfig(const HloInstruction* inst) {
  if (IsPopOpsFusion(inst, "conv_with_reverse")) {
    auto* conv_inst =
        inst->fused_instructions_computation()->root_instruction();
    return conv_inst->precision_config();
  } else if (inst->opcode() == HloOpcode::kConvolution) {
    return inst->precision_config();
  } else if (IsPoplarInstruction(PoplarOp::ConvWithReverse, inst)) {
    auto conv_with_reverse = Cast<HloConvWithReverse>(inst);
    return conv_with_reverse->GetPrecisionConfig();
  } else {
    return FailedPrecondition(
        "Trying to access precision configuration "
        " on a non convolution operation.");
  }
}

bool ForwardBackwardConvolutionDimensionNumbersMatch(
    const ConvolutionDimensionNumbers& fwd,
    const ConvolutionDimensionNumbers& bwd) {
  return google::protobuf::util::MessageDifferencer::Equivalent(
      FlipConvolutionDimensionNumbersFeatureAxis(fwd), bwd);
}

ConvolutionDimensionNumbers FlipConvolutionDimensionNumbersFeatureAxis(
    const ConvolutionDimensionNumbers& dims) {
  auto result = dims;
  auto output_dim = dims.kernel_output_feature_dimension();
  auto input_dim = dims.kernel_input_feature_dimension();
  result.set_kernel_output_feature_dimension(input_dim);
  result.set_kernel_input_feature_dimension(output_dim);
  return result;
}

// Copied from tensorflow/compiler/xla/service/hlo_instructions.cc.
std::string PrecisionConfigToString(const PrecisionConfig& precision_config) {
  if (absl::c_all_of(precision_config.operand_precision(), [](int32 precision) {
        return static_cast<PrecisionConfig::Precision>(precision) ==
               PrecisionConfig::DEFAULT;
      })) {
    return "";
  }

  return absl::StrCat(
      "operand_precision={",
      absl::StrJoin(
          precision_config.operand_precision(), ",",
          [](string* out, int32 precision) {
            CHECK(PrecisionConfig::Precision_IsValid(precision)) << precision;
            absl::StrAppend(
                out, PrecisionToString(
                         static_cast<PrecisionConfig::Precision>(precision)));
          }),
      "}");
}

}  // namespace poplarplugin
}  // namespace xla

namespace std {
std::size_t hash<xla::Window>::operator()(const xla::Window& window) const {
  return xla::poplarplugin::hash_util::hash(xla::window_util::ToString(window));
}

std::size_t hash<xla::PrecisionConfig>::operator()(
    const xla::PrecisionConfig& precision_config) const {
  std::string hashing_value;
  precision_config.SerializeToString(&hashing_value);

  return xla::poplarplugin::hash_util::hash(hashing_value);
}

std::size_t hash<xla::ConvolutionDimensionNumbers>::operator()(
    const xla::ConvolutionDimensionNumbers& dimension_numbers) const {
  return xla::poplarplugin::hash_util::hash(
      xla::ConvolutionDimensionNumbersToString(dimension_numbers));
}
}  // namespace std
