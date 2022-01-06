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
Image operations
~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def normalise_image(image, channel_offsets, channel_scales, scale=1,
                    name=None):
  """
  Pad an image to have 4 channel dimensions and normalise it according to the
  following formula:

  .. code-block:: python

    image = (image[c] * scale - channel_offsets[c]) * channel_scales[c]

  for each of the `c` channels in the image.

  Args:
    image: An `[X,Y,Z,3]` tensor, where the channels are the innermost
           dimension. Must be `uint8`, `float32` or `float16`.
    channel_offsets: A `[3]` array or tensor of offsets for the channels.
    channel_scales: A `[3]` array or tensor of scales for the channels.
    scale: A scalar constant that will scale the image before channel
           normalization. Defaults to 1.
    name: Optional op name.

  Returns:
    An `[X,Y,Z,4]` tensor with the same type as the input `image`, except
    `uint8` inputs where the output is `float16`.
  """

  if image.dtype not in [dtypes.float16, dtypes.float32, dtypes.uint8]:
    raise TypeError("The input `image` to `normalise_image` must be either"
                    " float16, float32 or uint8.")

  # The poplibs op outputs the same type as the image, except in the case of
  # uint8, where it casts to half.
  output_type = dtypes.float16 if image.dtype == dtypes.uint8 else image.dtype

  if isinstance(channel_offsets, ops.Tensor):
    channel_offsets = math_ops.cast(channel_offsets, output_type)
  if isinstance(channel_scales, ops.Tensor):
    channel_scales = math_ops.cast(channel_scales, output_type)

  channel_offsets = ops.convert_to_tensor(channel_offsets, dtype=output_type)
  channel_scales = ops.convert_to_tensor(channel_scales, dtype=output_type)

  return gen_popops_ops.normalise_image(image,
                                        channel_offsets,
                                        channel_scales,
                                        scale=scale,
                                        name=name)
