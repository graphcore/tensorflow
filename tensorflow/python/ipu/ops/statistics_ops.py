# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
Statistics operators
~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def fixed_width_bins(inputs, n_bins):
  """This op generates evenly spaced levels for histogram binning derived
  from the value range of `inputs`.

  Args:
    inputs: A rank-1 tensor of values over which to compute binning levels.
    n_bins: The number of bins required.

  Returns:
    A rank-1 tensor of binning values.
  """
  min_level = math_ops.reduce_min(inputs)
  max_level = math_ops.reduce_max(inputs)
  return math_ops.linspace(min_level, max_level, n_bins)


def histogram_normalize(hist):
  """This op normalizes a histogram.

  Args:
    hist: The histogram to be normalized.

  Returns:
    The normalized histogram.
  """
  return hist / math_ops.reduce_sum(hist, axis=0)


def histogram(inputs, levels, absolute_of_input=False):
  """This op generates a histogram of `inputs` over the fixed width bins
  defined by `levels`.

  Args:
    inputs: A rank-1 tensor of values over which to compute binning levels.
    levels: The number of bins required.
    absolute_of_input: If True, bin on input magnitude (absolute value).
      Default is False.

  Returns:
    A rank-1 histogram tensor.
  """
  inputs = ops.convert_to_tensor(inputs)
  levels = ops.convert_to_tensor(levels)

  # Check dtypes.
  type_check = lambda x: x.dtype in (dtypes.float16, dtypes.float32)
  if not type_check(inputs) or not type_check(levels):
    raise ValueError("Only float16 and float32 types are supported for "
                     "histogram computation.")

  # Check ranks.
  if len(inputs.shape) != 1 or len(levels.shape) != 1:
    raise ValueError("histogram expects rank-1 tensor inputs.")

  return gen_popops_ops.ipu_histogram(inputs,
                                      levels,
                                      absolute_of_input=absolute_of_input)


def histogram_update(hist, inputs, levels, absolute_of_input=False):
  """This op updates the histogram `hist` over the fixed width bins
  defined by `levels` for new `inputs`.

  Args:
    inputs: A rank-1 tensor of values over which to compute binning levels.
    levels: The number of bins required.
    absolute_of_input: If True, bin on input magnitude (absolute value).
      Default is False.

  Returns:
    The updated rank-1 histogram tensor, `hist`.
  """
  hist = ops.convert_to_tensor(hist)
  inputs = ops.convert_to_tensor(inputs)
  levels = ops.convert_to_tensor(levels)

  # Check dtypes.
  if hist.dtype != dtypes.float32:
    raise ValueError("hist must be of float32 type")

  type_check = lambda x: x.dtype in (dtypes.float16, dtypes.float32)
  if not type_check(hist) or not type_check(inputs) or not type_check(levels):
    raise ValueError(
        "Only float16 and float32 types are supported for histogram update.")

  # Check ranks.
  if len(hist.shape) != 1 or len(inputs.shape) != 1 or len(levels.shape) != 1:
    raise ValueError("histogram_update expects rank-1 tensor inputs.")

  # Check hist and levels shapes.
  if hist.shape[0] != levels.shape[0] + 1:
    raise ValueError("hist and levels shapes are incompatible. "
                     "For n levels, hist must have n+1 elements.")

  return gen_popops_ops.ipu_histogram_update(
      hist, inputs, levels, absolute_of_input=absolute_of_input)
