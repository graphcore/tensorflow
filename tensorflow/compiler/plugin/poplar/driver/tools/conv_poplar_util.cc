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
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_util.h"

namespace xla {
namespace poplarplugin {
namespace {

StatusOr<poplin::ConvParams> GetConvolutionParametersCore(
    const HloInstruction* inst, const std::vector<size_t>& input_dims,
    const std::vector<size_t>& kernel_dims,
    const std::vector<size_t>& output_dims, int64 n_g,
    const xla::ConvolutionDimensionNumbers& dims, const Window& window,
    poplar::Type dtype) {
  unsigned int n_b = input_dims[dims.input_batch_dimension()];
  unsigned int n_i = input_dims[dims.input_feature_dimension()];
  unsigned int n_j = kernel_dims[dims.kernel_input_feature_dimension()];
  unsigned int n_o = output_dims[dims.output_feature_dimension()];
  unsigned int n_p = kernel_dims[dims.kernel_output_feature_dimension()];

  if ((n_i >= n_j) && (n_o >= n_p)) {
    // Forward and backward passes
    if (n_g != (n_i / n_j) * (n_o / n_p)) {
      LOG(WARNING) << "Mismatch of the feature group for convolution "
                   << inst->name();
    }
    n_i = n_i / n_g;
    n_o = n_o / n_g;
  } else {
    // Weight update
    n_g = (n_j / n_i) * (n_p / n_o);
    n_b = n_b / n_g;
  }

  std::vector<std::size_t> n_s;
  std::vector<std::size_t> f_s;
  std::vector<unsigned int> w_s;
  std::vector<unsigned int> p_l;
  std::vector<unsigned int> p_u;
  std::vector<unsigned int> t_l;
  std::vector<unsigned int> t_u;
  std::vector<unsigned int> d_i;
  std::vector<unsigned int> d_w;
  std::vector<unsigned int> zeros;
  std::vector<bool> flipInput;
  std::vector<bool> flipKernel;

  for (int64 i = 0; i < window.dimensions().size(); i++) {
    n_s.push_back(input_dims[dims.input_spatial_dimensions(i)]);
    f_s.push_back(kernel_dims[dims.kernel_spatial_dimensions(i)]);
    w_s.push_back(window.dimensions(i).stride());
    flipInput.push_back(false);
    flipKernel.push_back(window.dimensions(i).window_reversal());
    if (window.dimensions(i).padding_low() < 0) {
      unsigned int p = -window.dimensions(i).padding_low();
      unsigned int d = window.dimensions(i).base_dilation();
      unsigned int trunc = (p + d - 1) / d;
      unsigned int pad = p % d;
      t_l.push_back(trunc);
      p_l.push_back(pad);
    } else {
      p_l.push_back(window.dimensions(i).padding_low());
      t_l.push_back(0);
    }
    if (window.dimensions(i).padding_high() < 0) {
      unsigned int p = -window.dimensions(i).padding_high();
      unsigned int d = window.dimensions(i).base_dilation();
      unsigned int trunc = (p + d - 1) / d;
      unsigned int pad = p % d;
      t_u.push_back(trunc);
      p_u.push_back(pad);
    } else {
      p_u.push_back(window.dimensions(i).padding_high());
      t_u.push_back(0);
    }
    d_i.push_back(window.dimensions(i).base_dilation());
    d_w.push_back(window.dimensions(i).window_dilation());
    zeros.push_back(0);
  }

  poplin::ConvParams params(dtype, dtype, n_b, n_s, f_s, n_i, n_o, n_g,
                            {t_l, t_u, d_i, p_l, p_u, flipInput},
                            {zeros, zeros, d_w, zeros, zeros, flipKernel},
                            {zeros, zeros, w_s, zeros, zeros});

  return params;
}

}  // namespace

StatusOr<poplin::ConvParams> GetConvolutionParameters(
    const HloInstruction* inst, int64 input_index, int64 kernel_index) {
  const Shape& input = inst->operand(input_index)->shape();
  const Shape& kernel = inst->operand(kernel_index)->shape();
  const Shape& output = inst->shape();

  TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(input));

  std::vector<size_t> input_dims = PoplarShapeFromXlaShape(input);
  std::vector<size_t> kernel_dims = PoplarShapeFromXlaShape(kernel);
  std::vector<size_t> output_dims = PoplarShapeFromXlaShape(output);

  const Window& window = GetConvolutionWindow(inst);
  unsigned int n_g = GetFeatureGroupCount(inst);
  const auto& dims = GetConvolutionDims(inst);

  return GetConvolutionParametersCore(inst, input_dims, kernel_dims,
                                      output_dims, n_g, dims, window, dtype);
}

StatusOr<poplin::ConvParams> GetConvolutionParametersForWeightsTranspose(
    const HloInstruction* inst, const std::vector<size_t>& conv_input_shape,
    const std::vector<size_t>& conv_output_shape) {
  const Shape& kernel = inst->operand(0)->shape();
  const Window& window = GetConvolutionWindow(inst);

  TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(kernel));

  std::vector<size_t> kernel_shape = PoplarShapeFromXlaShape(kernel);

  const auto& dims = GetConvolutionDims(inst);
  unsigned int n_g = GetFeatureGroupCount(inst);

  return GetConvolutionParametersCore(inst, conv_input_shape, kernel_shape,
                                      conv_output_shape, n_g, dims, window,
                                      dtype);
}

poplar::Tensor ShuffleConvolutionInputToPoplar(
    const ConvolutionDimensionNumbers& dims, const poplar::Tensor& tensor) {
  std::vector<unsigned int> shuffle(2 + dims.input_spatial_dimensions_size());
  shuffle[0] = dims.input_batch_dimension();
  shuffle[1] = dims.input_feature_dimension();
  for (int64 i = 0; i < dims.input_spatial_dimensions_size(); i++) {
    shuffle[2 + i] = dims.input_spatial_dimensions(i);
  }

  return tensor.dimShuffle(shuffle);
}

poplar::Tensor ShuffleConvolutionInputToPoplar(const HloInstruction* inst,
                                               const poplar::Tensor& tensor) {
  const ConvolutionDimensionNumbers& d(GetConvolutionDims(inst));
  return ShuffleConvolutionInputToPoplar(d, tensor);
}

poplar::Tensor ShuffleConvolutionOutputToPoplar(
    const ConvolutionDimensionNumbers& dims, const poplar::Tensor& tensor) {
  std::vector<unsigned int> shuffle(2 +
                                    dims.output_spatial_dimensions().size());
  shuffle[0] = dims.output_batch_dimension();
  shuffle[1] = dims.output_feature_dimension();
  for (int64 i = 0; i < dims.output_spatial_dimensions().size(); ++i) {
    shuffle[2 + i] = dims.output_spatial_dimensions(i);
  }

  return tensor.dimShuffle(shuffle);
}

poplar::Tensor ShuffleConvolutionOutputToPoplar(const HloInstruction* inst,
                                                const poplar::Tensor& tensor) {
  const ConvolutionDimensionNumbers& d(GetConvolutionDims(inst));
  return ShuffleConvolutionOutputToPoplar(d, tensor);
}

poplar::Tensor ShuffleConvolutionWeightsToPoplar(
    const ConvolutionDimensionNumbers& dims, const poplar::Tensor& tensor,
    bool swap_features) {
  std::vector<unsigned int> shuffle(2 + dims.kernel_spatial_dimensions_size());
  if (swap_features) {
    shuffle[0] = dims.kernel_input_feature_dimension();
    shuffle[1] = dims.kernel_output_feature_dimension();
  } else {
    shuffle[0] = dims.kernel_output_feature_dimension();
    shuffle[1] = dims.kernel_input_feature_dimension();
  }
  for (int64 i = 0; i < dims.kernel_spatial_dimensions_size(); i++) {
    shuffle[2 + i] = dims.kernel_spatial_dimensions(i);
  }

  return tensor.dimShuffle(shuffle);
}

poplar::Tensor ShuffleConvolutionWeightsToPoplar(const HloInstruction* inst,
                                                 const poplar::Tensor& tensor,
                                                 bool swap_features) {
  const ConvolutionDimensionNumbers& d(GetConvolutionDims(inst));
  return ShuffleConvolutionWeightsToPoplar(d, tensor, swap_features);
}

poplar::Tensor ShuffleConvolutionInputToTensorflow(
    const ConvolutionDimensionNumbers& dims, const poplar::Tensor& tensor) {
  std::vector<unsigned int> shuffle(2 + dims.input_spatial_dimensions_size());
  shuffle[dims.input_batch_dimension()] = 0;
  shuffle[dims.input_feature_dimension()] = 1;
  for (int64 i = 0; i < dims.input_spatial_dimensions_size(); i++) {
    shuffle[dims.input_spatial_dimensions(i)] = i + 2;
  }

  return tensor.dimShuffle(shuffle);
}

poplar::Tensor ShuffleConvolutionInputToTensorflow(
    const HloInstruction* inst, const poplar::Tensor& tensor) {
  const ConvolutionDimensionNumbers& d(GetConvolutionDims(inst));
  return ShuffleConvolutionInputToTensorflow(d, tensor);
}

poplar::Tensor ShuffleConvolutionWeightsToTensorflow(
    const ConvolutionDimensionNumbers& dims, const poplar::Tensor& tensor) {
  std::vector<unsigned int> shuffle(2 + dims.kernel_spatial_dimensions_size());
  shuffle[dims.kernel_output_feature_dimension()] = 0;
  shuffle[dims.kernel_input_feature_dimension()] = 1;
  for (int64 i = 0; i < dims.kernel_spatial_dimensions_size(); i++) {
    shuffle[dims.kernel_spatial_dimensions(i)] = i + 2;
  }

  return tensor.dimShuffle(shuffle);
}

poplar::Tensor ShuffleConvolutionWeightsToTensorflow(
    const HloInstruction* inst, const poplar::Tensor& tensor) {
  const ConvolutionDimensionNumbers& d(GetConvolutionDims(inst));
  return ShuffleConvolutionWeightsToTensorflow(d, tensor);
}

poplar::Tensor ShuffleConvolutionOutputToTensorflow(
    const ConvolutionDimensionNumbers& dims, const poplar::Tensor& tensor) {
  std::vector<unsigned int> shuffle(2 + dims.output_spatial_dimensions_size());
  shuffle[dims.output_batch_dimension()] = 0;
  shuffle[dims.output_feature_dimension()] = 1;
  for (int64 i = 0; i < dims.output_spatial_dimensions_size(); i++) {
    shuffle[dims.output_spatial_dimensions(i)] = i + 2;
  }

  return tensor.dimShuffle(shuffle);
}

poplar::Tensor ShuffleConvolutionOutputToTensorflow(
    const HloInstruction* inst, const poplar::Tensor& tensor) {
  const ConvolutionDimensionNumbers& d(GetConvolutionDims(inst));
  return ShuffleConvolutionOutputToTensorflow(d, tensor);
}

// This function operates on the poplibs format weights (GOI...)
poplar::Tensor AddGroupsDimensionToWeights(const poplin::ConvParams& p,
                                           const poplar::Tensor& t,
                                           bool flipped) {
  poplar::Tensor out = t;

  unsigned int out_dim = flipped ? 1 : 0;
  unsigned int in_dim = 1 - out_dim;

  if (p.getNumConvGroups() == 1) {
    // Non-grouped case
    return out.reshapePartial(0, 0, {1});
  } else {
    unsigned int chan_div[2];
    chan_div[in_dim] = out.dim(in_dim) / p.getNumInputChansPerConvGroup();
    chan_div[out_dim] = out.dim(out_dim) / p.getNumOutputChansPerConvGroup();

    // OI... ->(GO)(GI)...
    out = out.reshapePartial(0, 2,
                             {chan_div[0], out.dim(0) / chan_div[0],
                              chan_div[1], out.dim(1) / chan_div[1]});

    // (GO)(GI)... -> (GG)OI...
    out = out.dimShufflePartial({2}, {1});

    // (GG)OI... -> GOI...
    return out.reshapePartial(0, 2, {out.dim(0) * out.dim(1)});
  }
}

// This function operates on the poplibs format weights (GOI...)
poplar::Tensor RemoveGroupsDimensionFromWeights(const poplin::ConvParams& p,
                                                const poplar::Tensor& t) {
  poplar::Tensor out = t;
  return out.reshapePartial(0, 2, {out.dim(0) * out.dim(1)});
}

}  // namespace poplarplugin
}  // namespace xla
