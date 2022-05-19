/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "google/protobuf/util/message_differencer.h"

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

  TF_ASSIGN_OR_RETURN(ConvolutionDimensionNumbers conv_dimension_numbers,
                      GetConvolutionDims(inst_conv_with_reverse));

  TF_ASSIGN_OR_RETURN(Window window,
                      GetConvolutionWindow(inst_conv_with_reverse));
  TF_ASSIGN_OR_RETURN(int64_t feature_group_count,
                      GetFeatureGroupCount(inst_conv_with_reverse));

  HloInstruction* weights_transpose_flip =
      comp->AddInstruction(CreateHloWeightsTransposeChansFlipXY(
          conv_kernel, conv_dimension_numbers, conv_input->shape(),
          inst_conv_with_reverse->shape(), window, feature_group_count));

  inst_conv_with_reverse->SetupDerivedInstruction(weights_transpose_flip);

  SaveBackendConfigForWT(inst_conv_with_reverse, weights_transpose_flip);

  for (HloInstruction* predecessor :
       inst_conv_with_reverse->control_predecessors()) {
    TF_RETURN_IF_ERROR(
        predecessor->AddControlDependencyTo(weights_transpose_flip));
  }

  TF_ASSIGN_OR_RETURN(int64_t batch_group_count,
                      GetBatchGroupCount(inst_conv_with_reverse));

  TF_ASSIGN_OR_RETURN(auto precision_config,
                      GetPrecisionConfig(inst_conv_with_reverse));

  TF_RETURN_IF_ERROR(
      conv_input->AddControlDependencyTo(weights_transpose_flip));

  HloInstruction* conv_fwd =
      comp->AddInstruction(HloInstruction::CreateConvolve(
          inst_conv_with_reverse->shape(), conv_input, weights_transpose_flip,
          feature_group_count, batch_group_count, window,
          FlipConvolutionDimensionNumbersFeatureAxis(conv_dimension_numbers),
          precision_config));

  inst_conv_with_reverse->SetupDerivedInstruction(conv_fwd);
  if (inst_conv_with_reverse->has_sharding()) {
    const HloSharding& new_sharding = inst_conv_with_reverse->sharding();
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

StatusOr<bool> ForwardBackwardConvolutionMatch(HloInstruction* fwd,
                                               HloInstruction* bwd) {
  const int64_t operand_count = fwd->operand_count();
  if (operand_count != bwd->operand_count() || operand_count == 0) {
    return false;
  }

  if (!ShapeUtil::Compatible(fwd->shape(), bwd->operand(0)->shape()) ||
      !fwd->has_compatible_sharding(bwd)) {
    return false;
  }

  for (int64_t i = 0; i < operand_count; ++i) {
    if (!ShapeUtil::Compatible(
            fwd->operand(i)->shape(),
            i == 0 ? bwd->shape() : bwd->operand(i)->shape())) {
      return false;
    }
  }

  TF_ASSIGN_OR_RETURN(auto fwd_window, GetConvolutionWindow(fwd));
  TF_ASSIGN_OR_RETURN(auto bwd_window, GetConvolutionWindow(bwd));
  if (!google::protobuf::util::MessageDifferencer::Equivalent(fwd_window,
                                                              bwd_window)) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(auto fwd_dims, GetConvolutionDims(fwd));
  TF_ASSIGN_OR_RETURN(auto bwd_dims, GetConvolutionDims(bwd));
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
          inst->opcode() == HloOpcode::kConvolution) {
        fwd_conv.push_back(inst);
      }
    }
  }

  for (HloInstruction* inst : to_replace) {
    bool replace = false;
    VLOG(2) << "Found backward convolution " << inst->ToString();
    for (HloInstruction* fwd : fwd_conv) {
      VLOG(2) << "Comparing against forward convolution " << fwd->ToString();
      TF_ASSIGN_OR_RETURN(bool match,
                          ForwardBackwardConvolutionMatch(fwd, inst));
      if (match) {
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
