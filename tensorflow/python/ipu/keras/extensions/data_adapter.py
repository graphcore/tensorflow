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

import math
import numpy as np
from typing import Generator, Iterator
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.ops import variables

# Counter to keep track of number of log entries per token.
_counter_per_token = {}


def _call_counter(token):  #pylint: disable=missing-type-doc,missing-return-type-doc
  """Wrapper for _counter_per_token.

  Args:
    token: The token for which to look up the count.

  Returns:
    The number of times this function has been called with
    *token* as an argument (starting at 0)
  """
  global _counter_per_token  # pylint: disable=global-variable-not-assigned
  _counter_per_token[token] = 1 + _counter_per_token.get(token, -1)
  return _counter_per_token[token]


class IPUDataHandler(data_adapter.DataHandler):
  """Handles iterating over epoch-level Iterator objects on IPU."""
  def __init__(  # pylint: disable=super-init-not-called
      self,
      x,
      y=None,
      sample_weight=None,
      batch_size=None,
      steps_per_epoch=None,
      initial_epoch=0,
      epochs=1,
      shuffle=False,
      class_weight=None,
      max_queue_size=10,
      workers=1,
      use_multiprocessing=False,
      model=None,
      steps_per_execution=None,
      replication_factor=1):

    self._initial_epoch = initial_epoch
    self._epochs = epochs
    self._model = model

    if steps_per_execution is None:
      self._steps_per_execution = variables.Variable(
          1,
          dtype='int64',
          aggregation=variables.VariableAggregationV2.ONLY_FIRST_REPLICA)
      self._steps_per_execution_value = 1
    else:
      self._steps_per_execution = steps_per_execution
      self._steps_per_execution_value = steps_per_execution.numpy().item()

    numpy_warning = ("{} is of type `np.ndarray`. This will be cast to "
                     "`tf.Tensor` during every call to: {}. If you plan to "
                     "call any of these functions multiple times in your "
                     "program, it is recommended to pre-emptively cast to "
                     "`tf.Tensor` to avoid the repeated computation.")
    if isinstance(x, np.ndarray) and _call_counter("x_is_ndarray") == 1:
      logging.warn(
          numpy_warning.format("x", "`fit()`, `predict()` and `evaluate()`"))
    if (y is not None and isinstance(y, np.ndarray)
        and _call_counter("y_is_ndarray") == 1):
      logging.warn(numpy_warning.format("y", "`fit()` and `evaluate()`"))
    adapter_cls = data_adapter.select_data_adapter(x, y)
    self._adapter = adapter_cls(
        x,
        y,
        batch_size=batch_size,
        steps=steps_per_epoch,
        epochs=epochs - initial_epoch,
        sample_weights=sample_weight,
        shuffle=shuffle,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        distribution_strategy=ds_context.get_strategy(),
        model=model)

    if isinstance(self._adapter, data_adapter.GeneratorDataAdapter):
      hint = ("use `tf.data.Dataset.from_generator()` to specify the shape of "
              "your data.")
      if isinstance(x, Iterator) and not isinstance(x, Generator):
        hint = ("wrap your iterator inside a generator like so:\n\n"
                "def generator:\n"
                "    while True:\n"
                "        yield next(iterator)\n\n"
                "and " + hint)
      raise ValueError(
          "The provided set of data is of type `{}` which is not "
          "compatible with the IPU's requirement to know the data shape ahead "
          "of time. Please {}".format(type(x).__name__, hint))

    dataset = self._get_and_post_process_dataset(class_weight)

    self._replication_factor = replication_factor
    self._inferred_steps = self._infer_steps(steps_per_epoch, dataset)
    self._steps_per_epoch = steps_per_epoch

    self._validate_dataset(dataset)

    self._dataset = dataset
    self._current_step = 0
    self._step_increment = self._steps_per_execution_value - 1
    self._insufficient_data = False

    self._validate_data_handler()

  def _validate_data_handler(self):
    super()._validate_data_handler()

    if self.steps_per_execution_value > self.inferred_steps:
      logging.warn(
          "`steps_per_execution` has been set to {} but the dataset "
          "provided{} only contains {} batches. Using {} as "
          "`steps_per_execution`.".format(
              self.steps_per_execution_value,
              " to this replica" if self._replication_factor > 1 else "",
              self.inferred_steps, self.inferred_steps))

    with self._truncate_execution_to_epoch():
      if self.inferred_steps == 0:
        steps_per_replica = math.ceil(
            super()._infer_steps(None, self._dataset) /
            self._replication_factor)
        raise ValueError(
            "The provided dataset contains {} items, but all {} replicas "
            "are requested to run {} steps each. Make sure the dataset "
            "contains at least {} items.".format(
                len(self._dataset), self._replication_factor,
                steps_per_replica,
                steps_per_replica * self._replication_factor))
      elif self.inferred_steps % self._steps_per_execution != 0:
        raise ValueError(
            "`steps_per_execution` must be a divisor of the number of batches "
            "in the dataset provided{}.".format(
                " to this replica" if self._replication_factor > 1 else ""))

  def _infer_steps(self, steps, dataset):
    """Infers steps_per_epoch needed to loop through a dataset."""
    steps = super()._infer_steps(steps, dataset)
    if steps is None:
      raise ValueError(
          "Could not infer the size of the data. You must specify the number "
          "of steps to run.")
    if steps % self._replication_factor:
      logging.warn(
          "Dataset of length {} is being evenly distributed between {} "
          "replicas. The remaining {} batch{} will be dropped.".format(
              len(dataset), self._replication_factor,
              steps % self._replication_factor,
              "es" if steps % self._replication_factor > 1 else ""))

    return int(steps // self._replication_factor)

  def _get_and_post_process_dataset(self, class_weight):
    original_dataset = self._adapter.get_dataset()
    dataset = original_dataset

    # Check whether any prefetching should be done.
    prefetch_buffer = _get_prefetch_attribute(dataset)

    if class_weight:
      dataset = dataset.map(
          data_adapter._make_class_weight_map_fn(class_weight))  # pylint: disable=protected-access

    if self._adapter.has_partial_batch():
      raise ValueError(
          "The provided set of data has a partial batch, which could result in "
          "data being dropped. Either adjust the `batch_size` of your model "
          "(currently set to {}) or pad your dataset to ensure there are no "
          "partial batches.".format(self._adapter.batch_size()))

    dataset = _autocast_dataset(dataset)

    # Check whether the dataset should be prefetched.
    if dataset != original_dataset and prefetch_buffer is not None:
      dataset = dataset.prefetch(prefetch_buffer)

    return dataset

  def _validate_dataset(self, dataset):
    # Validate the size of the dataset.
    dataset_size = cardinality.cardinality(dataset)
    if dataset_size == cardinality.UNKNOWN:
      logging.info(
          "The provided set of data has an unknown size. This can result in "
          "runtime errors if not enough data is provided during execution.")
    elif dataset_size == cardinality.INFINITE:
      pass
    else:
      # If the size is known, it must provide enough data.
      if self._adapter.should_recreate_iterator():
        total_steps = self._inferred_steps
      else:
        total_steps = (self._epochs -
                       self._initial_epoch) * self._inferred_steps

      if total_steps > dataset_size:
        raise ValueError(
            "Your input does not have enough data. Make sure that your dataset "
            "or generator can generate at least {} batches (currently it can "
            "only generate {} batches). You may need to use the repeat() "
            "function when building your dataset.".format(
                total_steps, dataset_size))

    # Validate that the dataset has a shape which can be handled by infeeds.
    for spec in nest.flatten(dataset.element_spec):
      if not spec.shape.is_fully_defined():
        raise ValueError(
            "The provided set of data contains a shape {} which is not fully "
            "defined. Executing on IPU requires all dataset elements to have "
            "fully defined shapes. If using batch() make sure to set "
            "`drop_remainder=True`.".format(spec.shape))

  @property
  def steps_per_execution_value(self):
    return self._steps_per_execution_value

  @property
  def element_spec(self):
    return self._dataset.element_spec

  @property
  def batch_size(self):
    batch_size = self._adapter.batch_size()
    if batch_size is None and self.element_spec:
      element_spec = nest.flatten(self.element_spec)[0]
      if element_spec.shape:
        batch_size = element_spec.shape[0]
    return batch_size

  def set_replication_factor(self, value):
    self._replication_factor = value
    self._inferred_steps = self._infer_steps(self._steps_per_epoch,
                                             self._dataset)
    self._validate_dataset(self._dataset)
    self._validate_data_handler()

  def enumerate_epochs_with_reuse(self, manager, mode, infeed_kwargs):
    """Yields `(epoch, InfeedQueue)`."""
    with self._truncate_execution_to_epoch():
      data_iterator = manager.get_infeed(mode, self._dataset, infeed_kwargs)
      for epoch in range(self._initial_epoch, self._epochs):
        if self._insufficient_data:  # Set by `catch_stop_iteration`.
          break
        if self._adapter.should_recreate_iterator():
          data_iterator = manager.get_infeed(mode, self._dataset,
                                             infeed_kwargs)
        yield epoch, data_iterator
        self._adapter.on_epoch_end()


def _get_prefetch_attribute(dataset):
  if isinstance(dataset, dataset_ops.PrefetchDataset):
    return dataset._buffer_size  # pylint: disable=protected-access
  elif (isinstance(dataset, dataset_ops.DatasetV1Adapter)
        and isinstance(dataset._dataset, dataset_ops.PrefetchDataset)):  # pylint: disable=protected-access
    return dataset._dataset._buffer_size  # pylint: disable=protected-access
  return None


def _autocast_dataset(dataset):
  """Automatically downcast fp64 to fp32 when `v2_dtype_behavior_enabled()`."""
  element_spec = dataset.element_spec
  if (not base_layer_utils.v2_dtype_behavior_enabled()
      or not any(spec.dtype == dtypes.float64
                 for spec in nest.flatten(dataset.element_spec))):
    return dataset

  strip_tuple = isinstance(element_spec, tensor_spec.TensorSpec)

  def autocast_structure(*structure):
    def autocast_tensor(tensor):
      if tensor.dtype == dtypes.float64:
        return math_ops.cast(tensor, dtypes.float32)
      return tensor

    mapped = nest.map_structure(autocast_tensor, structure)
    return mapped[0] if strip_tuple else mapped

  return dataset.map(autocast_structure)
