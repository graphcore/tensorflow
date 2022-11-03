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
import tensorflow as tf


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
    self.data = self.maybe_get_tf_variable(data)
    self.metadata = metadata

  def maybe_get_tf_variable(self, data):
    result = data
    if not isinstance(data, tf.Variable) and not isinstance(data, tf.Tensor):
      result = tf.Variable(data)
    if result.dtype != "uint8":
      raise TypeError(
          "Trying to set/update QuarterTensor data with a tensor of type "
          f"{result.dtype}, but only uint8 are supported. Check that data "
          "is a value returned by `convert_to_f8`")
    return result

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
    self.data = self.maybe_get_tf_variable(new_values[0])
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
  metadata = ops.convert_to_tensor(metadata, dtype=dtypes.uint8)
  values = gen_popops_ops.ipu_convert_from_f8(values, metadata, name=name)
  if values.dtype != dtype:
    values = cast(values, dtype=dtype)
  return values


def canonicalise_input(inp):
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
    result[0] = array_ops.expand_dims(result[0],
                                      axis=0,
                                      name="MatMulCanonicalise")
  result[0].shape.assert_has_rank(3)
  return result


def canonicalise_output(output, lhs):
  if isinstance(lhs, QuarterTensor):
    lhs = lhs.data
  if (len(lhs.shape) == 2):
    out = array_ops.squeeze(output[0], axis=0)
  else:
    out = output[0]
  return out


"""
Performs a matmul on the 2 inputs tensors supporting element type fp8.

Args:
  lhs: Left hand side of matmul, can be tensor or quarter tensor.
  rhs: Right hand side of matmul, can be tensor or quarter tensor.

Returns:
  Tensor with type float16.
"""


def f8_matmul(lhs, rhs, name="f8_matmul"):
  inputs = canonicalise_input(lhs) + canonicalise_input(rhs)
  values = gen_popops_ops.ipu_f8_matmul(lhs=inputs[0],
                                        lhs_meta=inputs[1],
                                        rhs=inputs[2],
                                        rhs_meta=inputs[3],
                                        name=name)
  return canonicalise_output(values, lhs)
