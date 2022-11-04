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

#include <algorithm>
#include <utility>
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_conv.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/weights_transpose_chans_flip_xy.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/window_util.h"
#include "tensorflow/compiler/xla/window_util.h"
namespace xla {
namespace poplarplugin {
namespace {

StatusOr<poplin::ConvParams> GetConvolutionParametersCore(
    const HloInstruction* inst, const std::vector<size_t>& input_dims,
    const std::vector<size_t>& kernel_dims,
    const std::vector<size_t>& output_dims, int64_t f_g, int64_t b_g,
    const xla::ConvolutionDimensionNumbers& dims, const Window& window,
    poplar::Type input_dtype, poplar::Type output_dtype) {
  if (f_g > 1 && b_g > 1) {
    return xla::FailedPrecondition(
        "Poplar doesn't support grouping in batch and feature dimensions on ",
        inst->name());
  }

  unsigned int n_b = input_dims[dims.input_batch_dimension()];
  unsigned int n_i = input_dims[dims.input_feature_dimension()];
  unsigned int n_o = output_dims[dims.output_feature_dimension()];

  // Convolution groups
  //
  // A grouped convolution is where there are G weight tensors, each one
  // operating on one section of the input filters I.  Call i=I/G the input
  // channels per group.  In the output, there are O channels, composed of a
  // concatenation of G independent parts.  Call o=O/G, the output channels
  // per group.
  //
  // You could store these G weight tensors independently, as G lots of
  // [i, o] or packed into a larger tensor.  TF core packs the G weight
  // tensors into this shape [i*G, o].  XLA uses this packing shape
  // [i, o*G], while poplibs uses [G, o, i].
  //
  // In the code below, n_i and n_o start off being the size of the input
  // and output tensor channels, but become the poplibs channels per group
  // required by the convolution parameters structure.  Ie. they start off
  // as I and O, but become i and o.
  //
  // Note the above shapes leave out the spatial dimensions for brevity.

  // Grouped in the filter dimension.
  if (f_g > 1) {
    n_i = n_i / f_g;
    n_o = n_o / f_g;
  }

  // Grouped in the batch dimension.
  if (b_g > 1) {
    n_b = n_b / b_g;
    n_o = n_o / b_g;
  }

  // Create spatial dimension padding and striding.
  std::vector<std::size_t> n_s;
  std::vector<std::size_t> f_s;
  std::vector<unsigned int> out_s;
  std::vector<unsigned int> in_p_l;
  std::vector<unsigned int> in_p_u;
  std::vector<unsigned int> in_t_l;
  std::vector<unsigned int> in_t_u;
  std::vector<unsigned int> in_d;
  std::vector<unsigned int> w_d;
  std::vector<unsigned int> zeros;
  std::vector<bool> flipInput;
  std::vector<bool> flipKernel;
  std::vector<unsigned int> out_t_l;
  std::vector<unsigned int> out_t_u;

  for (int64_t i = 0; i < window.dimensions().size(); i++) {
    n_s.push_back(input_dims[dims.input_spatial_dimensions(i)]);
    f_s.push_back(kernel_dims[dims.kernel_spatial_dimensions(i)]);
    out_s.push_back(window.dimensions(i).stride());
    flipInput.push_back(false);
    flipKernel.push_back(window.dimensions(i).window_reversal());
    if (int p = window.dimensions(i).padding_low(); p < 0) {
      int t = -p;
      p = 0;
      in_p_l.push_back(p);
      in_t_l.push_back(0);
      out_t_l.push_back(t);
    } else {
      in_p_l.push_back(window.dimensions(i).padding_low());
      in_t_l.push_back(0);
      out_t_l.push_back(0);
    }
    if (int p = window.dimensions(i).padding_high(); p < 0) {
      int t = -p;
      p = 0;
      in_p_u.push_back(p);
      in_t_u.push_back(0);
      out_t_u.push_back(t);
    } else {
      in_p_u.push_back(window.dimensions(i).padding_high());
      in_t_u.push_back(0);
      out_t_u.push_back(0);
    }
    in_d.push_back(window.dimensions(i).base_dilation());
    w_d.push_back(window.dimensions(i).window_dilation());
    zeros.push_back(0);
  }

  auto n_g = std::max(f_g, b_g);

  poplin::ConvParams params(input_dtype, output_dtype, n_b, n_s, f_s, n_i, n_o,
                            n_g,
                            {in_t_l, in_t_u, in_d, in_p_l, in_p_u, flipInput},
                            {zeros, zeros, w_d, zeros, zeros, flipKernel},
                            {out_t_l, out_t_u, out_s, zeros, zeros});
  return params.canonicalize();
}

}  // namespace

StatusOr<poplin::ConvParams> GetConvolutionParameters(
    const HloInstruction* inst, int64_t input_index, int64_t kernel_index) {
  const Shape& input = inst->operand(input_index)->shape();
  const Shape& kernel = inst->operand(kernel_index)->shape();
  Shape output = inst->shape();

  // If we are using F8 convolution, then the output will be a tuple
  // of the form (data, metadata). In this function, we are only
  // interested in `data`, not `metadata`.
  if (output.IsTuple()) {
    output = ShapeUtil::GetTupleElementShape(output, 0);
  }

  TF_ASSIGN_OR_RETURN(poplar::Type input_dtype, PoplarDataType(input));
  if (input_dtype == poplar::UNSIGNED_CHAR) {
    input_dtype = poplar::QUARTER;
  }

  TF_ASSIGN_OR_RETURN(poplar::Type output_dtype, PoplarDataType(output));

  std::vector<size_t> input_dims = PoplarShapeFromXlaShape(input);
  std::vector<size_t> kernel_dims = PoplarShapeFromXlaShape(kernel);
  std::vector<size_t> output_dims = PoplarShapeFromXlaShape(output);

  TF_ASSIGN_OR_RETURN(Window window, GetConvolutionWindow(inst));

  TF_ASSIGN_OR_RETURN(auto dims, GetConvolutionDims(inst));

  TF_ASSIGN_OR_RETURN(unsigned int f_g, GetFeatureGroupCount(inst));
  TF_ASSIGN_OR_RETURN(unsigned int b_g, GetBatchGroupCount(inst));

  return GetConvolutionParametersCore(inst, input_dims, kernel_dims,
                                      output_dims, f_g, b_g, dims, window,
                                      input_dtype, output_dtype);
}

StatusOr<poplin::ConvParams> GetConvolutionParametersForWeightsTranspose(
    const HloWeightsTransposeChansFlipXYInstruction* inst) {
  const Shape& input = inst->ConvInputShape();
  const Shape& kernel = inst->operand(0)->shape();
  const Shape& output = inst->ConvOutputShape();

  TF_ASSIGN_OR_RETURN(Window window, GetConvolutionWindow(inst));

  TF_ASSIGN_OR_RETURN(poplar::Type input_dtype, PoplarDataType(input));
  TF_ASSIGN_OR_RETURN(poplar::Type output_dtype, PoplarDataType(output));

  std::vector<size_t> input_dims = PoplarShapeFromXlaShape(input);
  std::vector<size_t> kernel_dims = PoplarShapeFromXlaShape(kernel);
  std::vector<size_t> output_dims = PoplarShapeFromXlaShape(output);

  TF_ASSIGN_OR_RETURN(auto dims, GetConvolutionDims(inst));
  TF_ASSIGN_OR_RETURN(unsigned int f_g, GetFeatureGroupCount(inst));
  TF_ASSIGN_OR_RETURN(unsigned int b_g, GetBatchGroupCount(inst));

  return GetConvolutionParametersCore(inst, input_dims, kernel_dims,
                                      output_dims, f_g, b_g, dims, window,
                                      input_dtype, output_dtype);
}

StatusOr<std::vector<poplin::ConvParams>> GetConvolutionParametersForMultiConv(
    const HloMultiConvInstruction* inst) {
  const auto& convolution_specs = inst->GetConvolutionSpecs();

  std::vector<poplin::ConvParams> params(convolution_specs.size());
  for (int64_t i = 0; i != convolution_specs.size(); ++i) {
    const Shape& input = inst->operand(i)->shape();
    const Shape& kernel = inst->operand(i + convolution_specs.size())->shape();
    const Shape& output = ShapeUtil::GetTupleElementShape(inst->shape(), i);

    TF_ASSIGN_OR_RETURN(poplar::Type input_dtype, PoplarDataType(input));
    TF_ASSIGN_OR_RETURN(poplar::Type output_dtype, PoplarDataType(output));

    std::vector<size_t> input_dims = PoplarShapeFromXlaShape(input);
    std::vector<size_t> kernel_dims = PoplarShapeFromXlaShape(kernel);
    std::vector<size_t> output_dims = PoplarShapeFromXlaShape(output);

    const auto& convolution_spec = convolution_specs[i];
    const Window& window = convolution_spec.window;
    unsigned int f_g = convolution_spec.feature_group_count;
    unsigned int b_g = convolution_spec.batch_group_count;
    const auto& dims = convolution_spec.dims;

    TF_ASSIGN_OR_RETURN(
        params[i], GetConvolutionParametersCore(
                       inst, input_dims, kernel_dims, output_dims, f_g, b_g,
                       dims, window, input_dtype, output_dtype));
  }
  return params;
}

// Convert TF/XLA format tensor (with dims labelled by the sructure
// ConvolutionDimensionNumbers), into a Poplar format tensor, always
// Batch, Features, Y, X, ...
poplar::Tensor ShuffleConvolutionInputToPoplar(
    int64_t group_count, const ConvolutionDimensionNumbers& dims,
    const poplar::Tensor& tensor) {
  std::vector<unsigned int> shuffle(2 + dims.input_spatial_dimensions_size());
  shuffle[0] = dims.input_batch_dimension();
  shuffle[1] = dims.input_feature_dimension();
  for (int64_t i = 0; i < dims.input_spatial_dimensions_size(); i++) {
    shuffle[2 + i] = dims.input_spatial_dimensions(i);
  }

  auto out = tensor.dimShuffle(shuffle);

  // Move 'G' parts of the I to B (because B is the reducing dimension)
  out = out.reshapePartial(0, 1, {group_count, out.dim(0) / group_count});
  out = out.dimShufflePartial({0}, {1});
  out = out.reshapePartial(1, 3, {out.dim(1) * out.dim(2)});
  return out;
}

StatusOr<poplar::Tensor> ShuffleConvolutionInputToPoplar(
    const HloInstruction* inst, const poplar::Tensor& tensor) {
  TF_ASSIGN_OR_RETURN(auto group_count, GetBatchGroupCount(inst));
  TF_ASSIGN_OR_RETURN(ConvolutionDimensionNumbers d, GetConvolutionDims(inst));
  return ShuffleConvolutionInputToPoplar(group_count, d, tensor);
}

// Do the inverse operation to ShuffleConvolutionInputToPoplar
poplar::Tensor ShuffleConvolutionOutputToPoplar(
    const ConvolutionDimensionNumbers& dims, const poplar::Tensor& tensor) {
  std::vector<unsigned int> shuffle(2 +
                                    dims.output_spatial_dimensions().size());
  shuffle[0] = dims.output_batch_dimension();
  shuffle[1] = dims.output_feature_dimension();
  for (int64_t i = 0; i < dims.output_spatial_dimensions().size(); ++i) {
    shuffle[2 + i] = dims.output_spatial_dimensions(i);
  }

  return tensor.dimShuffle(shuffle);
}

StatusOr<poplar::Tensor> ShuffleConvolutionOutputToPoplar(
    const HloInstruction* inst, const poplar::Tensor& tensor) {
  TF_ASSIGN_OR_RETURN(ConvolutionDimensionNumbers d, GetConvolutionDims(inst));
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
  for (int64_t i = 0; i < dims.kernel_spatial_dimensions_size(); i++) {
    shuffle[2 + i] = dims.kernel_spatial_dimensions(i);
  }

  return tensor.dimShuffle(shuffle);
}

StatusOr<poplar::Tensor> ShuffleConvolutionWeightsToPoplar(
    const HloInstruction* inst, const poplar::Tensor& tensor,
    bool swap_features) {
  TF_ASSIGN_OR_RETURN(ConvolutionDimensionNumbers d, GetConvolutionDims(inst));
  return ShuffleConvolutionWeightsToPoplar(d, tensor, swap_features);
}

DriverTensor ShuffleConvolutionInputToTensorflow(
    int64_t group_count, const ConvolutionDimensionNumbers& dims,
    const DriverTensor& tensor) {
  // Move 'G' parts of the B back to I
  const unsigned n_g = group_count;
  DriverTensor out = tensor.reshapePartial(1, 2, {n_g, tensor.dim(1) / n_g});
  out = out.dimShufflePartial({1}, {0});
  out = out.reshapePartial(0, 2, {out.dim(0) * out.dim(1)});

  std::vector<unsigned int> shuffle(2 + dims.input_spatial_dimensions_size());
  shuffle[dims.input_batch_dimension()] = 0;
  shuffle[dims.input_feature_dimension()] = 1;
  for (int64_t i = 0; i < dims.input_spatial_dimensions_size(); i++) {
    shuffle[dims.input_spatial_dimensions(i)] = i + 2;
  }

  return out.dimShuffle(shuffle);
}

StatusOr<DriverTensor> ShuffleConvolutionInputToTensorflow(
    const HloInstruction* inst, const DriverTensor& tensor) {
  TF_ASSIGN_OR_RETURN(int64_t group_count, GetBatchGroupCount(inst));
  TF_ASSIGN_OR_RETURN(ConvolutionDimensionNumbers d, GetConvolutionDims(inst));
  return ShuffleConvolutionInputToTensorflow(group_count, d, tensor);
}

DriverTensor ShuffleConvolutionWeightsToTensorflow(
    const ConvolutionDimensionNumbers& dims, const DriverTensor& tensor,
    bool swap_features) {
  std::vector<unsigned int> shuffle(2 + dims.kernel_spatial_dimensions_size());
  int out_dim = dims.kernel_output_feature_dimension();
  int in_dim = dims.kernel_input_feature_dimension();
  shuffle[out_dim] = swap_features ? 1 : 0;
  shuffle[in_dim] = swap_features ? 0 : 1;
  for (int64_t i = 0; i < dims.kernel_spatial_dimensions_size(); i++) {
    shuffle[dims.kernel_spatial_dimensions(i)] = i + 2;
  }
  return tensor.dimShuffle(shuffle);
}

StatusOr<DriverTensor> ShuffleConvolutionWeightsToTensorflow(
    const HloInstruction* inst, const DriverTensor& tensor,
    bool swap_features) {
  TF_ASSIGN_OR_RETURN(ConvolutionDimensionNumbers d, GetConvolutionDims(inst));
  return ShuffleConvolutionWeightsToTensorflow(d, tensor, swap_features);
}

poplar::Tensor ShuffleConvolutionOutputToTensorflow(
    const ConvolutionDimensionNumbers& dims, const poplar::Tensor& tensor) {
  std::vector<unsigned int> shuffle(2 + dims.output_spatial_dimensions_size());
  shuffle[dims.output_batch_dimension()] = 0;
  shuffle[dims.output_feature_dimension()] = 1;
  for (int64_t i = 0; i < dims.output_spatial_dimensions_size(); i++) {
    shuffle[dims.output_spatial_dimensions(i)] = i + 2;
  }

  return tensor.dimShuffle(shuffle);
}

StatusOr<poplar::Tensor> ShuffleConvolutionOutputToTensorflow(
    const HloInstruction* inst, const poplar::Tensor& tensor) {
  TF_ASSIGN_OR_RETURN(ConvolutionDimensionNumbers d, GetConvolutionDims(inst));
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
DriverTensor RemoveGroupsDimensionFromWeights(const poplin::ConvParams& p,
                                              const DriverTensor& t) {
  return t.reshapePartial(0, 2, {t.dim(0) * t.dim(1)});
}

}  // namespace poplarplugin
}  // namespace xla
