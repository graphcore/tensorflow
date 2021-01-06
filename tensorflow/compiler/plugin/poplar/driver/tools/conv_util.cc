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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/weights_transpose_chans_flip_xy.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {
namespace poplarplugin {

Window GetConvolutionWindow(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kFusion) {
    auto cfg = inst->backend_config<PoplarBackendConfig>();
    return cfg.ValueOrDie().fusion_config().window();
  } else if (inst->opcode() == HloOpcode::kCustomCall) {
    if (IsPoplarInstruction(PoplarOp::WeightsTransposeChansFlipXY)(inst)) {
      auto inst_wtxy = Cast<HloWeightsTransposeChansFlipXYInstruction>(inst);
      return inst_wtxy->window();
    } else {
      LOG(FATAL) << "Trying to access window on non "
                    "HloWeightsTransposeChansFlipXYInstruction"
                    "operation.";
    }
  } else {
    if (!CastOrNull<HloConvolutionInstruction>(inst)) {
      LOG(FATAL) << "Trying to access convolution window on a non convolution "
                    "operation.";
    }
    return inst->window();
  }
}

ConvolutionDimensionNumbers GetConvolutionDims(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kFusion) {
    auto cfg = inst->backend_config<PoplarBackendConfig>();
    return cfg.ValueOrDie().fusion_config().dimension_numbers();
  } else if (inst->opcode() == HloOpcode::kCustomCall) {
    if (IsPoplarInstruction(PoplarOp::WeightsTransposeChansFlipXY)(inst)) {
      auto inst_wtxy = Cast<HloWeightsTransposeChansFlipXYInstruction>(inst);
      return inst_wtxy->convolution_dimension_numbers();
    } else {
      LOG(FATAL) << "Trying to access convolution_dimension_numbersn on a non "
                    "HloWeightsTransposeChansFlipXYInstruction.";
    }
  } else {
    if (!CastOrNull<HloConvolutionInstruction>(inst)) {
      LOG(FATAL) << "Trying to access convolution dimension numbers on a non "
                    "convolution operation.";
    }
    return inst->convolution_dimension_numbers();
  }
}

int64 GetFeatureGroupCount(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kFusion) {
    auto cfg = inst->backend_config<PoplarBackendConfig>();
    return cfg.ValueOrDie().fusion_config().feature_group_count();
  } else if (inst->opcode() == HloOpcode::kCustomCall) {
    if (IsPoplarInstruction(PoplarOp::WeightsTransposeChansFlipXY)(inst)) {
      auto inst_wtxy = Cast<HloWeightsTransposeChansFlipXYInstruction>(inst);
      return inst_wtxy->feature_group_count();
    } else {
      LOG(FATAL) << "Trying to access feature_group_count on non "
                    "HloWeightsTransposeChansFlipXYInstruction.";
    }
  } else {
    if (!CastOrNull<HloConvolutionInstruction>(inst)) {
      LOG(FATAL) << "Trying to access convolution feature group count numbers "
                    "on a non convolution operation.";
    }
    return inst->feature_group_count();
  }
}

int64 GetBatchGroupCount(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kFusion) {
    auto cfg = inst->backend_config<PoplarBackendConfig>();
    return cfg.ValueOrDie().fusion_config().batch_group_count();
  } else {
    if (!CastOrNull<HloConvolutionInstruction>(inst)) {
      LOG(FATAL) << "Trying to access convolution batch group count numbers on "
                    "a non convolution operation.";
    }
    return inst->batch_group_count();
  }
}

}  // namespace poplarplugin
}  // namespace xla
