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
Dataset benchmarking
~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_dataset_benchmark
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.data.ops import dataset_ops


def dataset_benchmark(dataset,
                      number_of_epochs,
                      elements_per_epochs,
                      print_stats=True,
                      apply_debug_options=True,
                      do_memcpy=True):
  """Allows the user to benchmark performance of a `tf.data.Dataset`.

    Args:
      dataset: An instance of `tf.data.Dataset` which will be benchmarked.
      number_of_epochs: The number of epochs this dataset will be run for.
      elements_per_epochs: The number of elements there are in each epoch.
      print_stats: Whether to print statistics about the performance to the
        console.
      apply_debug_options: Whether to apply debug options.
      do_memcpy: Whether to perform a `memcpy` operation which simulates a
        dataset buffer being copied to a Poplar managed buffer.

    Returns:
      A JSON string with performance statistics, which records the following
      metrics every epoch:
        * `elements_processed` - number of elements processed.
        * `total_bytes_processed` - total number of bytes which was processed.
        * `time_elapsed` - the time it took (in seconds) for the epoch to
          complete.
        * `elements_per_second` - number of elements processed per second.
        * `bandwidth` - the bandwidth achieved, measured in GB/s.

    The JSON string returned can be parsed into a native Python JSON library
    (see https://docs.python.org/3/library/json.html).

    Raises:
      TypeError: if `dataset` is not an instance of `tf.data.Dataset`.
      ValueError: if `number_of_epochs` or `elements_per_epochs` is less than 1.
    """
  if number_of_epochs < 1:
    return ValueError("Expected `number_of_epochs` to be at least 1.")
  if elements_per_epochs < 1:
    return ValueError("Expected `elements_per_epochs` to be at least 1.")

  if not isinstance(dataset, dataset_ops.DatasetV2):
    return TypeError("Expected `dataset` argument to be of type "
                     "`tf.data.Dataset`, but got %s "
                     "instead." % (str(dataset)))

  if apply_debug_options:
    dataset = dataset._apply_debug_options()  # pylint: disable=protected-access

  try:
    dataset_variant = dataset._variant_tensor  # pylint: disable=protected-access
  except TypeError:
    dataset_variant = dataset._as_variant_tensor  # pylint: disable=protected-access

  return gen_dataset_benchmark.dataset_benchmark(dataset_variant, print_stats,
                                                 do_memcpy, number_of_epochs,
                                                 elements_per_epochs,
                                                 **dataset._flat_structure)  # pylint: disable=protected-access


def infeed_benchmark(infeed_queue,
                     number_of_epochs,
                     elements_per_epochs,
                     print_stats=True,
                     do_memcpy=True):
  """Allows the user to benchmark performance of an
    `ipu.ipu_infeed_queue.IPUInfeedQueue`.

    Args:
      infeed_queue: An instance of `ipu.ipu_infeed_queue.IPUInfeedQueue` which
        will be benchmarked.
      number_of_epochs: The number of epochs this infeed queue will be run for.
      elements_per_epochs: The number of elements there are in each epoch.
      print_stats: Whether to print statistics about the performance to the
        console.
      do_memcpy: Whether to perform a `memcpy` operation which simulates a
        dataset buffer being copied to a Poplar managed buffer.

    Returns:
      A JSON string with performance statistics, which records the following
      metrics every epoch:
        * `elements_processed` - number of elements processed.
        * `total_bytes_processed` - total number of bytes which was processed.
        * `time_elapsed` - the time it took (in seconds) for the epoch to
          complete.
        * `elements_per_second` - number of elements processed per second.
        * `bandwidth` - the bandwidth achieved, measured in GB/s.

    The JSON string returned can be parsed into a native Python JSON library
    (see https://docs.python.org/3/library/json.html).

    Raises:
      TypeError: if `infeed_queue` is not an instance of
        `ipu.ipu_infeed_queue.IPUInfeedQueue`.
      ValueError: if `number_of_epochs` or `elements_per_epochs` is less than 1.
    """
  if not isinstance(infeed_queue, ipu_infeed_queue.IPUInfeedQueue):
    return TypeError("Expected `infeed_queue` argument to be of type "
                     "`ipu.ipu_infeed_queue.IPUInfeedQueue`, but got %s "
                     "instead." % (str(infeed_queue)))
  # Don't need to apply options because the infeed queue already applies them.
  apply_debug_options = False
  return dataset_benchmark(
      infeed_queue._dataset,  # pylint: disable=protected-access
      number_of_epochs,
      elements_per_epochs,
      print_stats=print_stats,
      apply_debug_options=apply_debug_options,
      do_memcpy=do_memcpy)
