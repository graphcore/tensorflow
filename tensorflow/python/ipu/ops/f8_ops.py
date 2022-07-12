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
  if values.dtype == dtypes.uint8:
    return values, metadata

  metadata = ops.convert_to_tensor(metadata, dtype=dtypes.uint8)
  return gen_popops_ops.ipu_convert_to_f8(values, metadata, name=name)


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
  return gen_popops_ops.ipu_convert_from_f8(values,
                                            metadata,
                                            dtype=dtype,
                                            name=name)
