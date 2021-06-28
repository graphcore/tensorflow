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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/weights_transpose_chans_flip_xy.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace poplarplugin {

namespace {
Shape GetWeightsShape(const Shape& shape,
                      const ConvolutionDimensionNumbers& conv_dims) {
  std::vector<int64> dims;
  absl::c_copy(shape.dimensions(), std::back_inserter(dims));
  std::swap(dims[conv_dims.kernel_input_feature_dimension()],
            dims[conv_dims.kernel_output_feature_dimension()]);
  return ShapeUtil::MakeShape(shape.element_type(), dims);
}
}  // namespace

HloWeightsTransposeChansFlipXYInstruction::
    HloWeightsTransposeChansFlipXYInstruction(
        HloInstruction* operand,
        const ConvolutionDimensionNumbers& conv_dimension_numbers,
        const std::vector<size_t>& conv_input_shape,
        const std::vector<size_t>& conv_output_shape, xla::Window window,
        int64 feature_group_count)
    : HloPoplarInstruction(
          GetWeightsShape(operand->shape(), conv_dimension_numbers), {operand},
          PoplarOp::WeightsTransposeChansFlipXY),
      conv_input_shape_(conv_input_shape),
      conv_output_shape_(conv_output_shape) {
  set_convolution_dimension_numbers(conv_dimension_numbers);
  set_window(window);
  set_feature_group_count(feature_group_count);
}

const std::vector<size_t>&
HloWeightsTransposeChansFlipXYInstruction::ConvInputShape() const {
  return conv_input_shape_;
}

const std::vector<size_t>&
HloWeightsTransposeChansFlipXYInstruction::ConvOutputShape() const {
  return conv_output_shape_;
}

absl::flat_hash_set<int64>
HloWeightsTransposeChansFlipXYInstruction::AllocatingIndices() const {
  return {0};
}

bool HloWeightsTransposeChansFlipXYInstruction::AllocatingOutput() const {
  return false;
}

absl::flat_hash_map<int64, int64>
HloWeightsTransposeChansFlipXYInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions
HloWeightsTransposeChansFlipXYInstruction::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloWeightsTransposeChansFlipXYInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults
HloWeightsTransposeChansFlipXYInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloWeightsTransposeChansFlipXYInstruction::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction>
HloWeightsTransposeChansFlipXYInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloWeightsTransposeChansFlipXYInstruction>(
      new_operands[0], convolution_dimension_numbers(), ConvInputShape(),
      ConvOutputShape(), window(), feature_group_count());
}

std::vector<std::string>
HloWeightsTransposeChansFlipXYInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> extra;
  const ConvolutionDimensionNumbers& conv_dimension_numbers =
      convolution_dimension_numbers();

  extra.push_back(absl::StrCat(
      "dim_labels=",
      ConvolutionDimensionNumbersToString(conv_dimension_numbers)));

  extra.push_back(absl::StrCat("window=", window_util::ToString(window())));
  extra.push_back(
      absl::StrCat("conv_input_shape_=",
                   "{ " + absl::StrJoin(conv_input_shape_, ", ") + "}"));
  extra.push_back(
      absl::StrCat("conv_output_shape_=",
                   "{ " + absl::StrJoin(conv_output_shape_, ", ") + "}"));

  return extra;
}

std::unique_ptr<HloInstruction> CreateHloWeightsTransposeChansFlipXY(
    HloInstruction* operand,
    const ConvolutionDimensionNumbers& conv_dimension_numbers,
    const std::vector<size_t>& conv_input_shape,
    const std::vector<size_t>& conv_output_shape, xla::Window window,
    int64 feature_group_count) {
  return absl::make_unique<HloWeightsTransposeChansFlipXYInstruction>(
      operand, conv_dimension_numbers, conv_input_shape, conv_output_shape,
      window, feature_group_count);
}

}  // namespace poplarplugin
}  // namespace xla
