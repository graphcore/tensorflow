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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MATCHER_PREDICATES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MATCHER_PREDICATES_H_

#include <functional>

#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {

class HloInstruction;

namespace poplarplugin {

bool IsRandomNormal(const HloInstruction*);
bool IsRandomUniform(const HloInstruction*);
bool IsCompareEqual(const HloInstruction*);
bool IsConstantZero(const HloInstruction*);
bool IsConstantOne(const HloInstruction*);
bool IsWideConstantZero(const HloInstruction*);
bool IsExternalPadding(const HloInstruction*);
bool IsFloat(const HloInstruction*);
bool IsScalar(const HloInstruction*);
bool IsFloatScalar(const HloInstruction*);
bool IsScalarConstant(const HloInstruction*);
bool IsFloatScalarConstant(const HloInstruction*);
bool IsScalarIntegerConstant(const HloInstruction*);
bool IsConvFilterTranspose(const HloInstruction*);
bool IsBiasReduce(const HloInstruction*);
bool IsOutputFeed(const HloInstruction*);
bool Is1DVector(const HloInstruction*);
bool IsExpandingReshape(const HloInstruction*);
bool IsF16(const HloInstruction*);
bool IsF32(const HloInstruction*);
bool IsF16OrF32(const HloInstruction*);
bool IsF32ToF16Convert(const HloInstruction*);
bool IsF16ToF32Convert(const HloInstruction*);
bool IsPopOpsConvolution(const HloInstruction*);
bool IsPopOpsConvolutionWithReverse(const HloInstruction*);
bool IsOpWithWindowNoBaseDilation(const HloInstruction*);
bool IsOpWithWindowNoStride(const HloInstruction*);
bool IsPaddingReduceWindow(const HloInstruction*);
bool IsAddOrSubtract(const HloInstruction*);
bool IsMultiplyOrDivide(const HloInstruction*);
bool IsBiasAdd(const HloInstruction*);
bool IsPopOpsBiasAdd(const xla::HloInstruction*);
bool IsPopOpsElementwise(const xla::HloInstruction*);
bool IsPopOpsElementwiseBinary(const xla::HloInstruction*);
bool IsPopOpsElementwiseBinaryOperandsDifferent(const xla::HloInstruction*);
bool IsNormInference(const xla::HloInstruction*);
bool IsNormTraining(const xla::HloInstruction*);
bool IsNormInferenceOrTraining(const xla::HloInstruction*);
bool IsNormGradient(const xla::HloInstruction*);
bool IsNonLinearityGradient(const xla::HloInstruction*);
bool IsNonLinearity(const xla::HloInstruction*);
bool IsSupportedAllReduce(const HloInstruction*);
bool IsMultiUpdateScatter(const HloInstruction*);
bool IsMultiUpdateAddScatter(const HloInstruction*);
bool IsMultiSliceGather(const HloInstruction*);
bool IsAnySliceApply(const HloInstruction*);
bool IsUniformSingleDimSlice(const HloInstruction*);
bool IsSingleElement(const HloInstruction*);
bool IsReduceAddOrMultiply(const HloInstruction*);
bool IsSerializedGradientAccumulation(const HloInstruction*);
/**
 * Construct a unary predicate which checks if a given HloInstruction is a
 * custom Poplibs instruction of a specified type.
 *
 * @param lib The library to capture and compare against.
 *
 * @returns The unary predicate.
 */
std::function<bool(const HloInstruction*)> IsPoplarInstruction(PoplarOp op);
}  // namespace poplarplugin
}  // namespace xla

#endif
