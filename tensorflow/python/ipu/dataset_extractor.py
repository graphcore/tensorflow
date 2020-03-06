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
# =============================================================================
"""
Dataset extractor
~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_dataset_benchmark
from tensorflow.python.data.ops import dataset_ops


def dataset_extractor(dataset, num_elements, filename, print_stats=True):
  """Allows the user to extract tensors from a `tf.data.Dataset`.

    Args:
      dataset: An instance of `tf.data.Dataset` to extract elements from.
      num_elements: The number of elements to extract from the dataset.
      filename: Where to save the extracted elements to.
      print_stats: Whether to print progress messages to the
        console.

    Note:
      One file per tuple element will be created.
      For example if the infeed is made of tuples (image, label) and
       `filename`="/tmp/infeed.bin" then the files generated will be:
      /tmp/infeed.0.bin (Will contain all the images) and
      /tmp/infeed.1.bin (Will contain all the labels).

    Returns:
      The operation that will save the elements of the infeed to file.

    Raises:
      TypeError: if `dataset` is not an instance of `tf.data.Dataset`.
      ValueError: if `num_elements` is less than 1.
    """
  if num_elements < 1:
    return ValueError("Expected `num_elements` to be at least 1.")

  if not isinstance(dataset, dataset_ops.DatasetV2):
    return TypeError("Expected `dataset` argument to be of type "
                     "`tf.data.Dataset`, but got %s "
                     "instead." % (str(dataset)))

  try:
    dataset_variant = dataset._variant_tensor  # pylint: disable=protected-access
  except TypeError:
    dataset_variant = dataset._as_variant_tensor  # pylint: disable=protected-access

  struct = dataset_ops.get_structure(dataset)
  names = list(struct.keys()) if isinstance(struct, dict) else []

  return gen_dataset_benchmark.dataset_extractor(dataset_variant, print_stats,
                                                 num_elements, filename, names,
                                                 **dataset._flat_structure)  # pylint: disable=protected-access
