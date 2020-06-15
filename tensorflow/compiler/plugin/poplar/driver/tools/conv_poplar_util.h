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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CONV_POPLAR_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CONV_POPLAR_UTIL_H_

#include <vector>

#include <poplin/Convolution.hpp>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace poplarplugin {
class HloMultiConvInstruction;

StatusOr<poplin::ConvParams> GetConvolutionParameters(
    const HloInstruction* operand_op, int64 input_index, int64 kernel_index);

StatusOr<poplin::ConvParams> GetConvolutionParametersForWeightsTranspose(
    const HloInstruction* inst, const std::vector<size_t>& conv_input_shape,
    const std::vector<size_t>& conv_output_shape);

StatusOr<std::vector<poplin::ConvParams>> GetConvolutionParametersForMultiConv(
    const HloMultiConvInstruction* inst);

poplar::Tensor ShuffleConvolutionInputToPoplar(const HloInstruction* inst,
                                               const poplar::Tensor& tensor);

poplar::Tensor ShuffleConvolutionInputToPoplar(
    const ConvolutionDimensionNumbers& dims, const poplar::Tensor& tensor);

poplar::Tensor ShuffleConvolutionOutputToPoplar(const HloInstruction* inst,
                                                const poplar::Tensor& tensor);

poplar::Tensor ShuffleConvolutionOutputToPoplar(
    const ConvolutionDimensionNumbers& dims, const poplar::Tensor& tensor);

poplar::Tensor ShuffleConvolutionWeightsToPoplar(const HloInstruction* inst,
                                                 const poplar::Tensor& tensor,
                                                 bool swap_features);

poplar::Tensor ShuffleConvolutionWeightsToPoplar(
    const ConvolutionDimensionNumbers& dims, const poplar::Tensor& tensor,
    bool swap_features);

poplar::Tensor ShuffleConvolutionInputToTensorflow(
    const HloInstruction* inst, const poplar::Tensor& tensor);

poplar::Tensor ShuffleConvolutionInputToTensorflow(
    const ConvolutionDimensionNumbers& dims, const poplar::Tensor& tensor);

poplar::Tensor ShuffleConvolutionWeightsToTensorflow(
    const HloInstruction* inst, const poplar::Tensor& tensor);

poplar::Tensor ShuffleConvolutionWeightsToTensorflow(
    const ConvolutionDimensionNumbers& dims, const poplar::Tensor& tensor);

poplar::Tensor ShuffleConvolutionOutputToTensorflow(
    const HloInstruction* inst, const poplar::Tensor& tensor);

poplar::Tensor ShuffleConvolutionOutputToTensorflow(
    const ConvolutionDimensionNumbers& dims, const poplar::Tensor& tensor);

poplar::Tensor AddGroupsDimensionToWeights(const poplin::ConvParams& p,
                                           const poplar::Tensor& t,
                                           bool flipped);

poplar::Tensor RemoveGroupsDimensionFromWeights(const poplin::ConvParams& p,
                                                const poplar::Tensor& t);
}  // namespace poplarplugin
}  // namespace xla

#endif
