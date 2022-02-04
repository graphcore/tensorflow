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
import threading
import time

from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.ipu.keras.extensions import keras_util


class PollingThread(threading.Thread):
  def __init__(self,
               output_iterator,
               num_steps,
               replication_factor,
               batch_begin_fn,
               batch_end_fn,
               unpack_step_results=False):
    super().__init__()

    self._cancelled = threading.Event()
    self._result = None
    self._output_iterator = output_iterator
    self._num_steps = num_steps
    self._replication_factor = replication_factor
    self._batch_begin_fn = batch_begin_fn
    self._batch_end_fn = batch_end_fn
    self._unpack_step_results = unpack_step_results

    # The polling mechanism works as follows:
    # 1. Until there is data, wait for `initial_wait_time` seconds.
    # 2. Get the time stamp when the outfeed first has data (first_timestamp).
    # 3. Get the time stamp when the outfeed has data for the second time
    #    (second_timestamp).
    # 4. Update the `wait_time` to:
    #    min((second_timestamp - first_timestamp) /
    #        (number_of_samples_processed * fudge_factor),
    #        initial_wait_time)
    self._initial_wait_time = 0.001
    self._first_timestamp = None
    self._second_timestamp = None
    self._fudge_factor = 1.9
    self._wait_time = self._initial_wait_time

  def postprocess(self, num_samples_processed):
    """Functions which should be called after an iteration of a polling loop is
    complete. If no results were processed, a sleep is inserted.

    Args:
      num_samples_processed (int): The number of batches that has been
      processed.
    """

    if num_samples_processed:
      if not self._first_timestamp:
        self._first_timestamp = time.time()
      elif not self._second_timestamp:
        self._second_timestamp = time.time()
        self._wait_time = min(
            (self._second_timestamp - self._first_timestamp) /
            (num_samples_processed * self._fudge_factor),
            self._initial_wait_time)
    else:
      time.sleep(self._wait_time)

  def cancel(self):
    """A thread should only be cancelled when an exception occurs."""
    self._cancelled.set()
    self.join()

  def cancelled(self):
    return self._cancelled.is_set()

  def get_result(self):
    return self._result

  def _iterate_over_replica_results(self, data):  # pylint: disable=missing-yield-type-doc
    """Function which slices out the per replica results.

    Args:
      data (tf.Tensor): Output to iterate over from outfeed queue.
    Yields:
      `tf.Tensor` The input tensor restructured according to the replication
      factor.
    """
    if self._replication_factor == 1:
      yield data
      return

    # Each tensor has an extra dimension
    for replica in range(self._replication_factor):
      yield nest.map_structure(lambda t: t[replica], data)  # pylint: disable=cell-var-from-loop

  def run(self):
    step = 0
    end_step = self._num_steps

    while step < end_step:
      # Check whether the thread was cancelled (could be an exception).
      if self.cancelled():
        return

      begin_step = step

      # Get the data (including replication).
      for all_data in self._output_iterator:
        # Get each step outputs.
        for data in self._iterate_over_replica_results(all_data):
          # Callback `_on_batch_begin()`.
          self._batch_begin_fn(step)

          data = data[0] if self._unpack_step_results else data

          # Callback `_on_batch_end()`.
          self._batch_end_fn(step, data)

          self._result = data

        step += 1

      self.postprocess(step - begin_step)


class PollingThreadPredict(PollingThread):
  """Optimized version of the PollingThread for predict function."""
  def run(self):
    step = 0
    end_step = self._num_steps

    while step < end_step:
      # Check whether the thread was cancelled (could be an exception).
      if self.cancelled():
        return

      results = self._output_iterator.dequeue()
      flat_results = nest.flatten(results)
      # Skip if no results are ready.
      if not flat_results:
        self.postprocess(0)
        continue

      num_iterations = array_ops.shape(flat_results[0]).numpy()[0]
      if not num_iterations:
        self.postprocess(0)
        continue

      begin_step = step

      # Call the callback for each step.
      for iteration in range(num_iterations):
        all_replicas_data_flat = [result[iteration] for result in flat_results]

        for flat_data in self._iterate_over_replica_results(
            all_replicas_data_flat):

          # Callback `_on_batch_begin()`.
          self._batch_begin_fn(step)

          data = nest.pack_sequence_as(results, flat_data)

          # Callback `_on_batch_end()`.
          self._batch_end_fn(step, data)

        step += 1

      # Append the results.
      merged_results = keras_util.merge_into_batch_dimension(
          results, self._replication_factor)
      if self._result is None:
        self._result = [merged_results]
      else:
        self._result.append(merged_results)

      self.postprocess(step - begin_step)
