/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops/conv_ops_helpers.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_poplar_util.h"

namespace xla {
namespace poplarplugin {
namespace helper {

StatusOr<DriverTensor> AddConvolutionInput(
    DriverGraph& graph, const HloInstruction* target,
    CompilerResources& resources, const poplar::DebugInfo& debug_info) {
  TF_ASSIGN_OR_RETURN(poplin::ConvParams params,
                      GetConvolutionParameters(target, 0, 1));

  TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                      GetConvolutionOptionsForInst(target, resources));

  auto out = DriverTensor(poplin::createInput(
      graph, params, {debug_info, "input"}, opts, &resources.planning_cache));

  auto o = ShuffleConvolutionInputToTensorflow(target, out);

  return o;
}

StatusOr<DriverTensor> AddConvolutionWeights(
    DriverGraph& graph, const HloInstruction* target,
    CompilerResources& resources, const poplar::DebugInfo& debug_info) {
  TF_ASSIGN_OR_RETURN(poplin::ConvParams params,
                      GetConvolutionParameters(target, 0, 1));

  TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                      GetConvolutionOptionsForInst(target, resources));

  auto out = DriverTensor(poplin::createWeights(
      graph, params, {debug_info, "weights"}, opts, &resources.planning_cache));

  out = RemoveGroupsDimensionFromWeights(params, out);

  return ShuffleConvolutionWeightsToTensorflow(target, out);
}

}  // namespace helper
}  // namespace poplarplugin
}  // namespace xla
