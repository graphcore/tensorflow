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
#include "tensorflow/compiler/plugin/poplar/driver/poplar_passes/convolution_preplanning.h"

#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/ml_type_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include <poplar/Target.hpp>

namespace xla {
namespace poplarplugin {

/*
 * Visit all non-fused operations in the whole module looking for convolutions,
 * and add the parameters and the options for that convolution to the set
 *  of things to pass to the poplibs convolution pre-planner.
 */

StatusOr<bool> ConvolutionPreplanning::Run(HloModule* module) {
  VLOG(2) << "Preplanning convolution operations.";
  preplan_convs.clear();
  option_flags_store.clear();

  for (auto* comp : module->computations()) {
    if (!IsPopOpsFusion(comp)) {
      for (HloInstruction* inst : comp->instructions()) {
        if (inst->opcode() == HloOpcode::kConvolution) {
          TF_RETURN_IF_ERROR(StorePreplanConv(inst, 0, 1));
        } else if (IsPopOpsConvolution(inst)) {
          TF_RETURN_IF_ERROR(StorePreplanConv(inst, 0, 1));
        } else if (IsPopOpsFusion(inst, "conv_scaled_inplace")) {
          TF_RETURN_IF_ERROR(StorePreplanConv(inst, 1, 2));
        }
      }
    }
  }

  poplin::preplanConvolutions(preplan_convs, resources_.convolution_cache);
  return false;
}

Status ConvolutionPreplanning::StorePreplanConv(const HloInstruction* inst,
                                                int64 input_index,
                                                int64 kernel_index) {
  const poplar::Target& target = GetGraph(resources_, inst).getTarget();
  TF_ASSIGN_OR_RETURN(
      const poplin::ConvParams conv_params,
      GetConvolutionParameters(inst, input_index, kernel_index));

  TF_ASSIGN_OR_RETURN(poplar::OptionFlags option_flags,
                      GetConvolutionOptionsForInst(inst, resources_));

  option_flags_store.push_back(option_flags);
  preplan_convs.insert(
      std::make_tuple(&target, conv_params, &(option_flags_store.back())));
  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla
