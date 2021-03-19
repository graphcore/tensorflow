# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
Dataset wrappers
~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_dataset_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops


class BufferDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` which makes sure there is a multiple of `buffer_size` number of
  elements available."""
  def __init__(self, input_dataset, buffer_size):
    """A `Dataset` which makes sure there is a multiple of `buffer_size` number of
      elements available.

    Args:
      input_dataset: The input dataset.
      buffer_size: The number of dataset elements which will be available.
    """
    self._input_dataset = input_dataset
    self._buffer_size = ops.convert_to_tensor(buffer_size,
                                              dtype=dtypes.int64,
                                              name="buffer_size")
    variant_tensor = gen_dataset_ops.buffer_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        buffer_size=self._buffer_size,
        **self._flat_structure)
    super(BufferDataset, self).__init__(input_dataset, variant_tensor)
