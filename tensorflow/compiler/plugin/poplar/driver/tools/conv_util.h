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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CONV_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CONV_UTIL_H_

#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace poplarplugin {

StatusOr<Window> GetConvolutionWindow(const HloInstruction* inst);

StatusOr<ConvolutionDimensionNumbers> GetConvolutionDims(
    const HloInstruction* inst);

StatusOr<int64> GetFeatureGroupCount(const HloInstruction* inst);

StatusOr<int64> GetBatchGroupCount(const HloInstruction* inst);

// Checks that forward and backward convolution dimension number match: input
// and output feature dimensions are swapped and the rest of the parameters are
// equivalent.
bool ForwardBackwardConvolutionDimensionNumbersMatch(
    const ConvolutionDimensionNumbers& fwd,
    const ConvolutionDimensionNumbers& bwd);

// Returns ConvolutionDimensionNumbers with kernel input/output feature
// dimension flipped.
ConvolutionDimensionNumbers FlipConvolutionDimensionNumbersFeatureAxis(
    const ConvolutionDimensionNumbers& dims);

}  // namespace poplarplugin
}  // namespace xla

#endif
