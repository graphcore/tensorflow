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
# =============================================================================
"""
IPU specific Keras Model extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import copy

from tensorflow.python.ipu.keras.extensions import data_adapter as ipu_data_adapter
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device as tf_device
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import utils
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import training as training_module
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.profiler import trace
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc


class ModelExtension(base_layer.KerasExtension):
  def __init__(self):
    self._ipu_train_function = None
    self._ipu_test_function = None
    self._ipu_predict_function = None
    self._replication_factor = None
    self._use_synthetic_data = utils.use_synthetic_data_for(
        utils.SyntheticDataCategory.Outfeed)

  def _get_shard_count(self):
    """Returns how many shards/IPUs the model is parallelized over."""
    raise NotImplementedError

  def _is_pipeline(self):
    """Returns whether the model is pipelined."""
    raise self._get_shard_count() > 1

  def _check_mode(self):
    if self.run_eagerly:
      raise RuntimeError(
          "Keras models cannot run eagerly when using `IPUStrategy`. Set "
          "`run_eagerly=False` when calling `compile`.")

  def _get_replication_factor(self):
    """Get the replication of the model."""
    if self._replication_factor is None:
      device_string = self.distribute_strategy.extended.non_slot_devices(None)
      current_device = tf_device.DeviceSpec.from_string(device_string)

      if current_device.device_type != "IPU":
        raise ValueError(self.__class__.__name__ +
                         " can only be used on an IPU device.")

      num_ipus = utils.get_num_of_ipus_in_device(device_string)
      shard_count = self._get_shard_count()
      if self._get_shard_count() > num_ipus:
        raise ValueError(
            "Current device has {} IPUs attached, however the current model "
            "requires a multiple of {} IPUs.".format(num_ipus, shard_count))
      self._replication_factor = num_ipus // shard_count

    return self._replication_factor

  def _get_steps_per_execution_per_replica(self, inferred_steps,
                                           original_steps_per_execution_value,
                                           data_steps_per_execution_value):
    if self._steps_per_execution is None:
      self._configure_steps_per_execution(1)

    replication_factor = self._get_replication_factor()

    if data_steps_per_execution_value % replication_factor != 0:
      if data_steps_per_execution_value != original_steps_per_execution_value:
        truncation_message = \
            " (truncated from {} due to {} steps per epoch)".format(
                original_steps_per_execution_value, inferred_steps)
      else:
        truncation_message = ""
      raise RuntimeError(
          "Currently `steps_per_execution` is set to {}{} and the current IPU "
          "system configuration and model configuration means that your Keras "
          "model will automatically execute in a data-parallel fashion across "
          "{} replicas. However the number of replicas needs to divide "
          "`steps_per_execution`. Either make sure that `steps_per_execution` "
          "is a multiple of {} or adjust your IPU system configuration to "
          "reduce the number of IPUs used for this IPUStrategy.".format(
              data_steps_per_execution_value, truncation_message,
              replication_factor, replication_factor))

    return data_steps_per_execution_value // replication_factor

  def _reset_ipu_options(self):
    """Function which resets any internal state of the extension when
    configuration changes."""
    self._ipu_train_function = None
    self._ipu_test_function = None
    self._ipu_predict_function = None
    self._replication_factor = None

  def _assert_weights_created_supported(self):
    return True

  def _assert_weights_created_delegate(self):
    if not self.built:
      raise ValueError('Weights for model %s have not yet been created. '
                       'Weights are created when the Model is first called on '
                       'inputs or `build()` is called with an `input_shape`.' %
                       self.name)

  def _reset_compile_cache_supported(self):
    return True

  def _reset_compile_cache_delegate(self):
    self._reset_ipu_options()
    return self._reset_compile_cache(__extension_delegate=False)

  def _get_output_iterator(self, outfeed_queue, replication_factor,
                           num_steps_per_replica):
    """Returns an iterator which is to be used for accessing output data
    consumed by the callbacks.

    When using synthetic data, creates an iterator which will return the right
    number of steps of data. If not synthetic it will just return the outfeed
    queue.
    """
    if self._use_synthetic_data:
      return _SyntheticDataGenerator(outfeed_queue, replication_factor,
                                     num_steps_per_replica)
    return outfeed_queue

  def _make_single_ipu_train_function(self):
    @def_function.function(experimental_compile=True)
    def train_function(steps_per_execution, iterator, outfeed):
      for _ in math_ops.range(steps_per_execution):
        outfeed.enqueue(self.train_step(next(iterator)))

    return train_function

  def _make_single_ipu_test_function(self):
    @def_function.function(experimental_compile=True)
    def test_function(steps_per_execution, iterator, outfeed):
      for _ in math_ops.range(steps_per_execution):
        outfeed.enqueue(self.test_step(next(iterator)))

    return test_function

  def _make_single_ipu_predict_function(self):
    @def_function.function(experimental_compile=True)
    def predict_function(steps_per_execution, iterator, outfeed):
      for _ in math_ops.range(steps_per_execution):
        outfeed.enqueue(self.predict_step(next(iterator)))

    return predict_function

  def _make_ipu_train_function(self):
    if self._ipu_train_function is None:
      if self._is_pipelined():
        raise NotImplementedError
      else:
        self._ipu_train_function = self._make_single_ipu_train_function()

    return self._ipu_train_function

  def _make_ipu_test_function(self):
    if self._ipu_test_function is None:
      if self._is_pipelined():
        raise NotImplementedError
      else:
        self._ipu_test_function = self._make_single_ipu_test_function()

    return self._ipu_test_function

  def _make_ipu_predict_function(self):
    if self._ipu_predict_function is None:
      if self._is_pipelined():
        raise NotImplementedError
      else:
        self._ipu_predict_function = self._make_single_ipu_predict_function()

    return self._ipu_predict_function

  def _fit_supported(self, *args, **kwargs):  # pylint:disable=unused-argument
    return True

  def _fit_delegate(self,
                    x=None,
                    y=None,
                    batch_size=None,
                    epochs=1,
                    verbose=1,
                    callbacks=None,
                    validation_split=0.,
                    validation_data=None,
                    shuffle=True,
                    class_weight=None,
                    sample_weight=None,
                    initial_epoch=0,
                    steps_per_epoch=None,
                    validation_steps=None,
                    validation_batch_size=None,
                    validation_freq=1,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False):
    base_layer.keras_api_gauge.get_cell('fit').set(True)
    # Legacy graph support is contained in `training_v1.Model`.
    version_utils.disallow_legacy_graph('Model', 'fit')
    self._assert_compile_was_called()
    self._check_call_args('fit')
    training_module._disallow_inside_tf_function('fit')  # pylint: disable=protected-access

    self._check_mode()
    replication_factor = self._get_replication_factor()

    if validation_split:
      # Create the validation data using the training data. Only supported for
      # `Tensor` and `NumPy` input.
      (x, y,
       sample_weight), validation_data = (data_adapter.train_validation_split(
           (x, y, sample_weight), validation_split=validation_split))

    if validation_data:
      val_x, val_y, val_sample_weight = (
          data_adapter.unpack_x_y_sample_weight(validation_data))

    with self.distribute_strategy.scope(), \
         training_utils.RespectCompiledTrainableState(self):
      # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
      data_handler = ipu_data_adapter.IPUDataHandler(
          x=x,
          y=y,
          sample_weight=sample_weight,
          batch_size=batch_size,
          steps_per_epoch=steps_per_epoch,
          initial_epoch=initial_epoch,
          epochs=epochs,
          shuffle=shuffle,
          class_weight=class_weight,
          max_queue_size=max_queue_size,
          workers=workers,
          use_multiprocessing=use_multiprocessing,
          model=self,
          steps_per_execution=self._steps_per_execution)

      # Container that configures and calls `tf.keras.Callback`s.
      if not isinstance(callbacks, callbacks_module.CallbackList):
        callbacks = callbacks_module.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self,
            verbose=verbose,
            epochs=epochs,
            steps=data_handler.inferred_steps)

      self.stop_training = False
      self._train_counter.assign(0)
      callbacks.on_train_begin()
      train_function = self._make_ipu_train_function()
      training_logs = None

      data_handler._initial_epoch = (  # pylint: disable=protected-access
          self._maybe_load_initial_epoch_from_ckpt(initial_epoch))
      logs = None

      original_steps_per_execution_value = \
          data_handler.steps_per_execution_value
      outfeed = ipu_outfeed_queue.ScopedIPUOutfeedQueue()
      for epoch, iterator in data_handler.enumerate_epochs():
        inferred_steps = data_handler.inferred_steps
        steps_per_execution_value = data_handler.steps_per_execution_value
        steps_per_execution_per_replica = \
          self._get_steps_per_execution_per_replica(
              inferred_steps, original_steps_per_execution_value,
              steps_per_execution_value)

        self.reset_metrics()
        callbacks.on_epoch_begin(epoch)

        for step in data_handler.steps():
          end_step = step + data_handler.step_increment
          with trace.Trace('train',
                           epoch_num=epoch,
                           step_num=step,
                           batch_size=batch_size,
                           _r=1):
            self.distribute_strategy.run(train_function,
                                         args=(steps_per_execution_per_replica,
                                               iterator, outfeed))
            self._train_counter.assign_add(steps_per_execution_value)

            output_iterator = self._get_output_iterator(
                outfeed, replication_factor, steps_per_execution_per_replica)

            current_step = step
            for replica_data in output_iterator:
              for data in _iterate_over_replica_results(
                  replica_data, replication_factor):
                callbacks.on_train_batch_begin(current_step)
                logs = data
                training_module.write_scalar_summaries(logs, step=current_step)
                callbacks.on_train_batch_end(current_step, logs)
                current_step += 1

              if current_step == end_step:
                break

          if self.stop_training:
            break

        if logs is None:
          raise ValueError('Expect x to be a non-empty array or dataset.')
        epoch_logs = copy.copy(logs)

        # Run validation.
        if validation_data and self._should_eval(epoch, validation_freq):
          # Create data_handler for evaluation and cache it.
          if getattr(self, '_eval_data_handler', None) is None:
            self._fit_frame = tf_inspect.currentframe()
            self._eval_data_handler = ipu_data_adapter.IPUDataHandler(
                x=val_x,
                y=val_y,
                sample_weight=val_sample_weight,
                batch_size=validation_batch_size or batch_size,
                steps_per_epoch=validation_steps,
                initial_epoch=0,
                epochs=1,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=self,
                steps_per_execution=self._steps_per_execution)
          val_logs = self.evaluate(x=val_x,
                                   y=val_y,
                                   sample_weight=val_sample_weight,
                                   batch_size=validation_batch_size
                                   or batch_size,
                                   steps=validation_steps,
                                   callbacks=callbacks,
                                   max_queue_size=max_queue_size,
                                   workers=workers,
                                   use_multiprocessing=use_multiprocessing,
                                   return_dict=True)
          val_logs = {'val_' + name: val for name, val in val_logs.items()}
          epoch_logs.update(val_logs)

        callbacks.on_epoch_end(epoch, epoch_logs)
        training_logs = epoch_logs
        if self.stop_training:
          break

      # If eval data_hanlder exists, delete it after all epochs are done.
      if getattr(self, '_eval_data_handler', None) is not None:
        del self._eval_data_handler
        del self._fit_frame
      callbacks.on_train_end(logs=training_logs)
      return self.history

  def _evaluate_supported(self, *args, **kwargs):  # pylint:disable=unused-argument
    return True

  def _evaluate_delegate(self,
                         x=None,
                         y=None,
                         batch_size=None,
                         verbose=1,
                         sample_weight=None,
                         steps=None,
                         callbacks=None,
                         max_queue_size=10,
                         workers=1,
                         use_multiprocessing=False,
                         return_dict=False):

    base_layer.keras_api_gauge.get_cell('evaluate').set(True)
    version_utils.disallow_legacy_graph('Model', 'evaluate')
    self._assert_compile_was_called()
    self._check_call_args('evaluate')
    training_module._disallow_inside_tf_function('evaluate')  # pylint: disable=protected-access

    self._check_mode()
    replication_factor = self._get_replication_factor()

    with self.distribute_strategy.scope():
      # Use cached evaluation data only when it's called in `Model.fit`
      if (getattr(self, '_fit_frame', None) is not None
          and tf_inspect.currentframe().f_back is self._fit_frame
          and getattr(self, '_eval_data_handler', None) is not None):
        data_handler = self._eval_data_handler
      else:
        # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
        data_handler = ipu_data_adapter.IPUDataHandler(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps,
            initial_epoch=0,
            epochs=1,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            model=self,
            steps_per_execution=self._steps_per_execution)

      # Container that configures and calls `tf.keras.Callback`s.
      if not isinstance(callbacks, callbacks_module.CallbackList):
        callbacks = callbacks_module.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self,
            verbose=verbose,
            epochs=1,
            steps=data_handler.inferred_steps)

      logs = {}
      test_function = self._make_ipu_test_function()

      self._test_counter.assign(0)
      callbacks.on_test_begin()

      original_steps_per_execution_value = \
          data_handler.steps_per_execution_value
      outfeed = ipu_outfeed_queue.ScopedIPUOutfeedQueue()
      for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
        inferred_steps = data_handler.inferred_steps
        steps_per_execution_value = data_handler.steps_per_execution_value
        steps_per_execution_per_replica = \
          self._get_steps_per_execution_per_replica(
              inferred_steps, original_steps_per_execution_value,
              steps_per_execution_value)

        self.reset_metrics()

        for step in data_handler.steps():
          end_step = step + data_handler.step_increment
          with trace.Trace('test', step_num=step, _r=1):
            self.distribute_strategy.run(test_function,
                                         args=(steps_per_execution_per_replica,
                                               iterator, outfeed))
            self._test_counter.assign_add(steps_per_execution_value)

            output_iterator = self._get_output_iterator(
                outfeed, replication_factor, steps_per_execution_per_replica)
            current_step = step
            for replica_data in output_iterator:
              for data in _iterate_over_replica_results(
                  replica_data, replication_factor):
                callbacks.on_test_batch_begin(current_step)
                logs = data
                callbacks.on_test_batch_end(current_step, logs)
                current_step += 1

              if current_step == end_step:
                break

      logs = tf_utils.to_numpy_or_python_type(logs)
      callbacks.on_test_end(logs=logs)

      if return_dict:
        return logs
      else:
        results = []
        for name in self.metrics_names:
          if name in logs:
            results.append(logs[name])
        for key in sorted(logs.keys()):
          if key not in self.metrics_names:
            results.append(logs[key])
        if len(results) == 1:
          return results[0]
        return results

  def _predict_supported(self, *args, **kwargs):  # pylint:disable=unused-argument
    return True

  def _predict_delegate(self,
                        x,
                        batch_size=None,
                        verbose=0,
                        steps=None,
                        callbacks=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False):
    base_layer.keras_api_gauge.get_cell('predict').set(True)
    version_utils.disallow_legacy_graph('Model', 'predict')
    self._check_call_args('predict')
    training_module._disallow_inside_tf_function('predict')  # pylint: disable=protected-access

    self._check_mode()
    replication_factor = self._get_replication_factor()

    outputs = None
    with self.distribute_strategy.scope():
      # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
      data_handler = ipu_data_adapter.IPUDataHandler(
          x=x,
          batch_size=batch_size,
          steps_per_epoch=steps,
          initial_epoch=0,
          epochs=1,
          max_queue_size=max_queue_size,
          workers=workers,
          use_multiprocessing=use_multiprocessing,
          model=self,
          steps_per_execution=self._steps_per_execution)

      # Container that configures and calls `tf.keras.Callback`s.
      if not isinstance(callbacks, callbacks_module.CallbackList):
        callbacks = callbacks_module.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self,
            verbose=verbose,
            epochs=1,
            steps=data_handler.inferred_steps)

      predict_function = self._make_ipu_predict_function()
      self._predict_counter.assign(0)
      callbacks.on_predict_begin()
      batch_outputs = None

      original_steps_per_execution_value = \
          data_handler.steps_per_execution_value
      outfeed = ipu_outfeed_queue.ScopedIPUOutfeedQueue()
      for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
        inferred_steps = data_handler.inferred_steps
        steps_per_execution_value = data_handler.steps_per_execution_value
        steps_per_execution_per_replica = \
          self._get_steps_per_execution_per_replica(
              inferred_steps, original_steps_per_execution_value,
              steps_per_execution_value)

        for step in data_handler.steps():
          end_step = step + data_handler.step_increment
          self.distribute_strategy.run(predict_function,
                                       args=(steps_per_execution_per_replica,
                                             iterator, outfeed))
          self._predict_counter.assign_add(steps_per_execution_value)

          output_iterator = self._get_output_iterator(
              outfeed, replication_factor, steps_per_execution_per_replica)

          current_step = step
          for replica_data in output_iterator:
            for data in _iterate_over_replica_results(replica_data,
                                                      replication_factor):
              callbacks.on_predict_batch_begin(current_step)
              batch_outputs = data
              if outputs is None:
                outputs = nest.map_structure(
                    lambda batch_output: [batch_output], batch_outputs)
              else:
                nest.map_structure_up_to(
                    batch_outputs,
                    lambda output, batch_output: output.append(batch_output),
                    outputs, batch_outputs)

              callbacks.on_predict_batch_end(current_step,
                                             {'outputs': batch_outputs})
              current_step += 1

            if current_step == end_step:
              break

      if batch_outputs is None:
        raise ValueError('Expect x to be a non-empty array or dataset.')
      callbacks.on_predict_end()
    all_outputs = nest.map_structure_up_to(batch_outputs,
                                           training_module.concat, outputs)
    return tf_utils.to_numpy_or_python_type(all_outputs)


class _SyntheticDataGenerator(collections_abc.Iterator):
  """An iterator for generating synthetic data."""
  def __init__(self, outfeed_queue, replication_factor, num_steps):
    shapes = outfeed_queue._flat_shapes  # pylint: disable=protected-access
    dtypes = outfeed_queue._flat_types  # pylint: disable=protected-access
    if replication_factor > 1:
      shapes = [[replication_factor] + shape.as_list() for shape in shapes]
    flat_buffers = [
        array_ops.zeros(shape, dtype) for shape, dtype in zip(shapes, dtypes)
    ]
    self._dummy_data = nest.pack_sequence_as(
        outfeed_queue._structure,  # pylint: disable=protected-access
        flat_buffers)
    self._num_steps = num_steps
    self._step = 0

  def __iter__(self):
    return self

  def __next__(self):
    if self._step == self._num_steps:
      raise StopIteration
    self._step += 1
    return self._dummy_data


def _iterate_over_replica_results(data, replication_factor):
  """Function which slices out the per replica results."""
  if replication_factor == 1:
    yield data
    return

  # Each tensor has an extra dimension
  flat_data = nest.flatten(data)
  for i in range(replication_factor):
    x = nest.pack_sequence_as(data, [t[i] for t in flat_data])
    yield x
