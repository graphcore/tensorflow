# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
F8 operations
~~~~~~~~~~~~~
"""

from enum import IntEnum

from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import cast
from tensorflow.python.ops.nn_ops import _get_sequence
import tensorflow as tf
import numpy as np


class Format(IntEnum):
  """
  Format describes bit layout of F8 type.

  Attributes:
    F143: 1 sign bit, 4 bit of significand and 3 bits of exponent.

    F152: 1 sign bit, 5 bit of significand and 2 bits of exponent.

  """
  F143 = 0
  F152 = 1


def create_metadata(fmt, scale=0):
  return (128 if fmt == Format.F143 else 0) | (scale & 0x3f)


class QuarterTensor:
  """
  Represents a tensor with data type fp8.
  """
  def __init__(self, data, metadata):
    """
    Constructs a quarter tensor from the values tensor and metadata.

    Args:
      data: A tensor with data type uint8. Should generally be the output
        of a call to `convert_to_f8`.
      metadata: The metadata for this quarter tensor, should be the output
        of a call to `create_metadata`.
    """
    self.data = self._maybe_get_tf_variable(data)
    self.metadata = metadata

  def _maybe_get_tf_variable(self, data):
    if isinstance(data, np.ndarray):
      data = tf.Variable(data)
    if data.dtype != "uint8":
      raise TypeError(
          "Trying to set/update QuarterTensor data with a tensor of type "
          f"{data.dtype}, but only uint8 are supported. Check that data "
          "is a value returned by `convert_to_f8`")
    return data

  def numpy(self):
    """Returns a numpy representation of the tensor
    """
    return [self.data.numpy(), self.metadata]

  def assign(self, new_values, **kwargs):
    """Assigns new values to the tensor

    Args:
      new_values: An array of format [data, metadata] that
        should be the output of `QuarterTensor.numpy`.
    """
    self.data = self._maybe_get_tf_variable(new_values[0])
    self.metadata = new_values[1]

  def __eq__(self, other):
    return self.numpy() == other.numpy()

  @property
  def shape(self):
    return self.data.shape

  @property
  def dtype(self):
    return tf.uint8

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return self.data[index]

  def __iter__(self):
    return iter([self.data, self.metadata])


def convert_to_f8(values, metadata, name=None):
  """
  Converts given values to f8 representation.

  Args:
    values: Any tensor of any type
    metadata: metadata created by create_metadata

    name: Optional op name.

  Returns:
    (output, metadata) tuple of uint8 output and metadata.
  """

  if not (values.dtype is dtypes.float16):
    values = cast(values, dtypes.float16)

  if isinstance(metadata, np.ndarray) or isinstance(metadata, int):
    metadata = ops.convert_to_tensor(metadata, dtype=dtypes.uint8)
  v, m = gen_popops_ops.ipu_convert_to_f8(values, metadata, name=name)
  return QuarterTensor(v, m)


def convert_from_f8(packed_input, dtype=dtypes.half, name=None):
  """
  Converts packed f8 representation to tensor of type dtype

  Args:
    packed_input: result of convert_to_f8 or any other f8 op.
    dtype: output tensor type. Default is half because it's hw
        accelerated and would not require extra cast.
    name: Optional op name.

  Returns:
    Tensor with type dtype with unpacked f8 values.
  """
  values, metadata = packed_input
  if isinstance(metadata, np.ndarray) or isinstance(metadata, int):
    metadata = ops.convert_to_tensor(metadata, dtype=dtypes.uint8)
  values = gen_popops_ops.ipu_convert_from_f8(values, metadata, name=name)
  if values.dtype != dtype:
    values = cast(values, dtype=dtype)
  return values


def canonicalise_input(inp, expected_rank=3, name="MatMulCanonicalise"):
  result = None
  # input expects data and metadata, if input is type without metadata just put in junk
  if isinstance(inp, QuarterTensor):
    result = [inp.data, inp.metadata]
  else:
    result = [
        ops.convert_to_tensor(inp),
        ops.convert_to_tensor(0, dtype=dtypes.uint8)
    ]
  if not result[0].shape.is_fully_defined():
    raise Exception(f"Input shape must be fully defined: {result[0]}")
  # XLA op always expects grouped matmul so if doesn't have group dimension then add it
  if len(result[0].shape) == 2:
    result[0] = array_ops.expand_dims(result[0], axis=0, name=name)
  result[0].shape.assert_has_rank(expected_rank)
  return result


def canonicalise_output(output, lhs):
  if isinstance(lhs, QuarterTensor):
    lhs = lhs.data
  if (len(lhs.shape) == 2):
    out = array_ops.squeeze(output[0], axis=0)
  else:
    out = output[0]
  return out


def f8_matmul(lhs, rhs, name="f8_matmul"):
  """
  Performs a matmul on the 2 inputs tensors supporting element type fp8.

  Args:
    lhs: Left hand side of matmul, can be tensor or quarter tensor.
    rhs: Right hand side of matmul, can be tensor or quarter tensor.

  Returns:
    Tensor with type float16.
  """
  inputs = canonicalise_input(lhs) + canonicalise_input(rhs)
  values = gen_popops_ops.ipu_f8_matmul(lhs=inputs[0],
                                        lhs_meta=inputs[1],
                                        rhs=inputs[2],
                                        rhs_meta=inputs[3],
                                        name=name)
  return canonicalise_output(values, lhs)


def f8_conv_1d(inputs,
               filters,
               strides,
               padding,
               data_format="NWC",
               dilations=[1],
               name="f8_conv_1d"):
  """
  Performs a 1D convolution on the 2 inputs tensors supporting element type fp8.

  Args:
    inputs: A `Tensor` or `QuarterTensor` of at least rank-3. M
    filters: A `Tensor` or `QuarterTensor` of rank at least 3.
    strides: An int or list of `ints` that has length `1` or `3`.  The number of
      entries by which the filter is moved right at each step.
    padding: 'SAME' or 'VALID'
    data_format: An optional `string` from `"NWC", "NCW"`.
      The data is stored in the order of `batch_shape + [in_width,
      in_channels]`.  The `"NCW"` format stores data as `batch_shape +
      [in_channels, in_width]`.
      Defaults to `"NWC"`.
    dilations: An int or list of `ints` that has length `1` or `3` which
      defaults to 1. The dilation factor for each dimension of input. If set to
      `k > 1`, there will be `k-1` skipped cells between each filter element on
      that dimension. Dilations in the batch and depth dimensions must be 1.
    name: A name for the operation (optional).
      Defaults to 'f8_conv_1d'.

  Returns:
    `Tensor` of type `float16`.
  """
  # Reshape input to make this a 2d conv.
  if data_format == "NWC":
    data_format = "NHWC"
    spatial_start_dim = -3
    channel_index = 2
  elif data_format == "NCW":
    data_format = "NCHW"
    spatial_start_dim = -2
    channel_index = 1
  else:
    raise ValueError("`data_format` must be 'NWC' or 'NCW'. "
                     f"Received: data_format={data_format}")

  strides = [1] + _get_sequence(strides, 1, channel_index, "stride")
  dilations = [1] + _get_sequence(dilations, 1, channel_index, "dilations")

  if isinstance(inputs, QuarterTensor):
    inputs.data = array_ops.expand_dims(inputs.data, spatial_start_dim)
  else:
    inputs = array_ops.expand_dims(inputs, spatial_start_dim)

  if isinstance(filters, QuarterTensor):
    filters.data = array_ops.expand_dims(filters.data, 0)
  else:
    filters = array_ops.expand_dims(filters, 0)

  y = f8_conv_2d(inputs,
                 filters,
                 strides,
                 padding,
                 data_format=data_format,
                 dilations=dilations,
                 name=name)

  return array_ops.squeeze(y, [spatial_start_dim])


def f8_conv_2d(inputs,
               filters,
               strides,
               padding,
               data_format="NHWC",
               dilations=[1, 1, 1, 1],
               name="f8_conv_2d"):
  """
  Performs a 2D convolution on the 2 inputs tensors supporting element type fp8.

  Args:
    inputs: A `Tensor` or `QuarterTensor`. 
      A Tensor of rank at least 4. The dimension order is interpreted according
      to the value of `data_format`; with the all-but-inner-3 dimensions acting
      as batch dimensions. See below for details.
    filters: A `Tensor` or `QuarterTensor`.
      A 4-D tensor of shape
      `[filter_height, filter_width, in_channels, out_channels]`
    strides: An int or list of `ints` that has length `1`, `2` or `4`.  The
      stride of the sliding window for each dimension of `input`. If a single
      value is given it is replicated in the `H` and `W` dimension. By default
      the `N` and `C` dimensions are set to 1. The dimension order is determined
      by the value of `data_format`, see below for details.
    padding: Either the `string` `"SAME"` or `"VALID"` indicating the type of
      padding algorithm to use, or a list indicating the explicit paddings at
      the start and end of each dimension. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information. When explicit padding is used and data_format is
      `"NHWC"`, this should be in the form `[[0, 0], [pad_top, pad_bottom],
      [pad_left, pad_right], [0, 0]]`. When explicit padding used and
      data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          `batch_shape + [height, width, channels]`.
      Alternatively, the format could be "NCHW", the data storage order of:
          `batch_shape + [channels, height, width]`.
      Defaults to `"NHWC"`.
    dilations: An int or list of `ints` that has length `1`, `2` or `4`.
      The dilation factor for each dimension of`input`. If a
      single value is given it is replicated in the `H` and `W` dimension. By
      default the `N` and `C` dimensions are set to 1. If set to k > 1, there
      will be k-1 skipped cells between each filter element on that dimension.
      The dimension order is determined by the value of `data_format`, see above
      for details. Dilations in the batch and depth dimensions if a 4-d tensor
      must be 1.
      Defaults to None.
    name: A name for the operation (optional).
      Defaults to 'f8_conv_2d'.

  Returns:
    `Tensor` of type `float16`.
  """
  x = canonicalise_input(inputs, expected_rank=4)
  k = canonicalise_input(filters, expected_rank=4)

  values = gen_popops_ops.ipu_f8_conv2d(inputs=x[0],
                                        filters=k[0],
                                        inputs_meta=x[1],
                                        filters_meta=k[1],
                                        strides=strides,
                                        padding=padding,
                                        data_format=data_format,
                                        dilations=dilations,
                                        name=name)
  return canonicalise_output(values, inputs)


def f8_conv_3d(inputs,
               filters,
               strides,
               padding,
               data_format="NDHWC",
               dilations=[1, 1, 1, 1, 1],
               name="f8_conv_3d"):
  """
  Performs a 3d convolution on the 2 inputs tensors supporting element type fp8.

  Args:
    inputs: A `Tensor` or `QuarterTensor` of shape
      `[batch, in_depth, in_height, in_width, in_channels]`.
    filters: A `Tensor` or `QuarterTensor`. Must have the same type as `input`.
      Shape
      `[filter_depth, filter_height, filter_width, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filters`.
    strides: A list of ints that has length >= 5. 1-D `Tensor` of length 5.
      The stride of the sliding window for each dimension of input.
      Must have `strides[0] = strides[4] = 1`.
    padding: A string from: `"SAME"`, `"VALID"`.
      The type of padding algorithm to use.
    data_format: An optional string from: `"NDHWC"`, `"NCDHW"`.
      The data format of the input and output data. With the default format
      `"NDHWC"`, the data is stored in the order of:
      `[batch, in_depth, in_height, in_width, in_channels]`. Alternatively,
      the format could be `"NCDHW"`, the data storage order is:
      `[batch, in_channels, in_depth, in_height, in_width]`.
      Defaults to "NDHWC".
    dilations: An optional list of ints. 1-D tensor of length 5. The dilation,
      factor for each dimension of input. If set to `k > 1`, there will
      be `k-1` skipped cells between each filter element on that dimension.
      The dimension order is determined by the value of data_format, see above
      for details. Dilations in the batch and depth dimensions must be 1.
      Defaults to [1, 1, 1, 1, 1].
    name: A name for the operation (optional).
      Defaults to 'f8_conv_3d'.

  Returns:
    `Tensor` of type `float16`.
  """
  x = canonicalise_input(inputs, expected_rank=5)
  k = canonicalise_input(filters, expected_rank=5)

  values = gen_popops_ops.ipu_f8_conv3d(inputs=x[0],
                                        filters=k[0],
                                        inputs_meta=x[1],
                                        filters_meta=k[1],
                                        strides=strides,
                                        padding=padding,
                                        data_format=data_format,
                                        dilations=dilations,
                                        name=name)
  return canonicalise_output(values, inputs)
