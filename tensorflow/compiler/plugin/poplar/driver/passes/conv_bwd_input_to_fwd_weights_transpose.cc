/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/conv_bwd_input_to_fwd_weights_transpose.h"

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/weights_transpose_chans_flip_xy.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/ml_type_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {
namespace {
Status SaveBackendConfigForWT(HloInstruction* old_inst,
                              HloInstruction* new_inst) {
  TF_ASSIGN_OR_RETURN(PoplarBackendConfig old_inst_config,
                      old_inst->backend_config<PoplarBackendConfig>());
  TF_ASSIGN_OR_RETURN(PoplarBackendConfig new_inst_config,
                      new_inst->backend_config<PoplarBackendConfig>());
  new_inst_config.mutable_convolution_options()->CopyFrom(
      old_inst_config.convolution_options());
  new_inst->set_backend_config(new_inst_config);

  return Status::OK();
}

StatusOr<bool> ReplaceConvolutionWithReverse(
    HloInstruction* inst_conv_with_reverse) {
  HloComputation* comp = inst_conv_with_reverse->parent();
  HloModule* module = comp->parent();
  HloInstruction* conv_input = inst_conv_with_reverse->mutable_operand(0);
  HloInstruction* conv_kernel = inst_conv_with_reverse->mutable_operand(1);

  ConvolutionDimensionNumbers conv_dimension_numbers =
      GetConvolutionDims(inst_conv_with_reverse);

  std::vector<size_t> conv_input_shape(conv_input->shape().dimensions().begin(),
                                       conv_input->shape().dimensions().end());
  std::vector<size_t> conv_output_shape(
      inst_conv_with_reverse->shape().dimensions().begin(),
      inst_conv_with_reverse->shape().dimensions().end());

  Window window = GetConvolutionWindow(inst_conv_with_reverse);
  int64 feature_group_count = GetFeatureGroupCount(inst_conv_with_reverse);

  HloInstruction* weights_transpose_flip =
      comp->AddInstruction(CreateHloWeightsTransposeChansFlipXY(
          conv_kernel, conv_dimension_numbers, conv_input_shape,
          conv_output_shape, window, feature_group_count));

  inst_conv_with_reverse->SetupDerivedInstruction(weights_transpose_flip);

  SaveBackendConfigForWT(inst_conv_with_reverse, weights_transpose_flip);

  for (HloInstruction* predecessor :
       inst_conv_with_reverse->control_predecessors()) {
    TF_RETURN_IF_ERROR(
        predecessor->AddControlDependencyTo(weights_transpose_flip));
  }

  int64 batch_group_count = GetBatchGroupCount(inst_conv_with_reverse);
  HloInstruction* root_inst =
      inst_conv_with_reverse->fused_instructions_computation()
          ->root_instruction();

  HloInstruction* conv_fwd =
      comp->AddInstruction(HloInstruction::CreateConvolve(
          inst_conv_with_reverse->shape(), conv_input, weights_transpose_flip,
          feature_group_count, batch_group_count, window,
          FlipConvolutionDimensionNumbersFeatureAxis(conv_dimension_numbers),
          root_inst->precision_config()));

  inst_conv_with_reverse->SetupDerivedInstruction(conv_fwd);
  if (inst_conv_with_reverse->has_sharding()) {
    HloSharding new_sharding = HloSharding::AssignDevice(
        *inst_conv_with_reverse->sharding_unique_device());
    weights_transpose_flip->set_sharding(new_sharding);
    conv_fwd->set_sharding(new_sharding);
  }

  for (HloInstruction* successor :
       inst_conv_with_reverse->control_successors()) {
    TF_RETURN_IF_ERROR(conv_fwd->AddControlDependencyTo(successor));
  }

  TF_ASSIGN_OR_RETURN(MLType ml_type, GetMLType(inst_conv_with_reverse));

  // In order to reuse backward convolution we have to make it equivalent to
  // forward, i.e. flip weights. GetConvolutionOptionsForInst will get ml_type
  // from the instruction if not specified explicitly. Information about the
  // type of the pass is set in the poplar convolution option flags. See
  // GetConvolutionOptionsForInst for future reference.

  if (ml_type == MLType::TRAINING_BWD) {
    TF_RETURN_IF_ERROR(SetInstructionMLType(conv_fwd, MLType::TRAINING_FWD));
    TF_RETURN_IF_ERROR(
        SetInstructionMLType(weights_transpose_flip, MLType::TRAINING_FWD));
  } else {
    TF_RETURN_IF_ERROR(SetInstructionMLType(conv_fwd, ml_type));
    TF_RETURN_IF_ERROR(SetInstructionMLType(weights_transpose_flip, ml_type));
  }
  auto called_computations = inst_conv_with_reverse->called_computations();

  TF_RETURN_IF_ERROR(inst_conv_with_reverse->DropAllControlDeps());
  TF_RETURN_IF_ERROR(
      comp->ReplaceInstruction(inst_conv_with_reverse, conv_fwd));

  for (HloComputation* fusion_comp : called_computations) {
    TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(fusion_comp));
  }

  return true;
}

bool ForwardBackwardConvolutionMatch(HloInstruction* fwd, HloInstruction* bwd) {
  const int64 operand_count = fwd->operand_count();
  if (operand_count != bwd->operand_count() || operand_count == 0) {
    return false;
  }

  if (!ShapeUtil::Compatible(fwd->shape(), bwd->shape()) ||
      !fwd->has_compatible_sharding(bwd)) {
    return false;
  }

  for (int64 i = 0; i < operand_count; ++i) {
    if (!ShapeUtil::Compatible(fwd->operand(i)->shape(),
                               bwd->operand(i)->shape())) {
      return false;
    }
  }

  auto fwd_dims = GetConvolutionDims(fwd);
  auto bwd_dims = GetConvolutionDims(bwd);
  return ForwardBackwardConvolutionDimensionNumbersMatch(fwd_dims, bwd_dims);
}

}  // namespace

StatusOr<bool> ConvBwdInputToFwdWeightsTranspose::Run(HloModule* module) {
  bool changed = false;
  VLOG(2) << "Before the ConvBwdInputToFwdWeightsTranspose:";
  XLA_VLOG_LINES(2, module->ToString());

  std::vector<HloInstruction*> to_replace;
  std::vector<HloInstruction*> fwd_conv;

  for (auto comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(MLType ml_type, GetMLType(inst));

      if (IsPopOpsConvolutionWithReverse(inst) &&
          ml_type == MLType::TRAINING_BWD) {
        to_replace.push_back(inst);
      }

      if (ml_type == MLType::TRAINING_FWD &&
          (inst->opcode() == HloOpcode::kConvolution ||
           IsPopOpsFusion(inst, "depthwise_conv") ||
           IsPopOpsFusion(inst, "depthwise_filter"))) {
        fwd_conv.push_back(inst);
      }
    }
  }

  for (HloInstruction* inst : to_replace) {
    bool replace = false;
    VLOG(2) << "Found backward convolution " << inst->ToString();
    VLOG(2) << "Dimension numbers "
            << ConvolutionDimensionNumbersToString(GetConvolutionDims(inst));
    for (HloInstruction* fwd : fwd_conv) {
      VLOG(2) << "Comparing against forward convolution " << fwd->ToString();
      VLOG(2) << "Dimension numbers "
              << ConvolutionDimensionNumbersToString(GetConvolutionDims(fwd));
      if (ForwardBackwardConvolutionMatch(fwd, inst)) {
        VLOG(2) << "Found matching forward convolution: " << fwd->ToString();
        replace = true;
        break;
      }
    }
    if (replace) {
      VLOG(2) << "Replacing " << inst->name()
              << " with WeightsTransposeChansFlipXY/forward convolution.";
      TF_ASSIGN_OR_RETURN(bool replaced, ReplaceConvolutionWithReverse(inst));
      changed |= replaced;
    }
  }

  if (changed) {
    VLOG(2) << "After the ConvBwdInputToFwdWeightsTranspose:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "There were no changes.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
