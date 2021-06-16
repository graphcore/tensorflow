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
Keras Model interfaces for IPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from functools import partial
import copy
import inspect
import math
import numpy as np
import weakref

from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.keras.layer_replacement import IPULayerReplacer
from tensorflow.python.ipu.ops import functional_ops
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import Model as KerasModel
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import data_adapter
# TODO(T31437): Fix up Keras API.
# from tensorflow.python.keras.engine import network
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.engine import training as keras_training
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ipu.keras.optimizers import _KerasOptimizerWrapper
from tensorflow.python.ipu.keras.optimizers import _TensorflowOptimizerWrapper
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops.losses import util as tf_losses_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.ops import math_ops


def _validate_args(kwargs, fn):
  blacklist = [
      "sample_weight_mode", "weighted_metrics", "target_tensors", "distribute",
      "run_eagerly", "validation_split", "validation_data", "class_weight",
      "sample_weight", "validation_steps", "validation_freq", "max_queue_size",
      "workers", "use_multiprocessing"
  ]

  if 'y' in kwargs:
    raise ValueError(
        "Labels should be provided by the 'x' DataSet containing a tuple.")

  if 'batch_size' in kwargs:
    raise ValueError("Do not specify `batch_size` in IPU Keras models."
                     " Use the Dataset.batch() method to apply batching"
                     " at the input dataset level.")

  bad_args = list(filter(lambda x: x in kwargs, blacklist))
  if bad_args:
    raise NotImplementedError(
        "IPU Keras models do not support these parameters to " + fn + "(): " +
        ", ".join(bad_args))


def _validate_dataset_element_count(ds, count, fn_name):
  structure = dataset_ops.get_structure(ds)
  num_elements = len(nest.flatten(structure))
  if num_elements != count:
    raise ValueError(
        "%s requires a dataset with a structure containing %d element(s), but "
        "got %d element(s) instead." % (fn_name, count, num_elements))


def _get_dataset_and_count(x, y, batch_size):
  adapter_cls = data_adapter.select_data_adapter(x, y)
  adapter = adapter_cls(x, y, batch_size=batch_size)

  dataset = adapter.get_dataset()
  original_dataset = dataset
  size = adapter.get_size()

  if adapter.has_partial_batch():
    dataset = dataset.unbatch()
    # Remove the partial batch from the dataset.
    dataset = dataset.batch(batch_size, drop_remainder=True)

    size -= 1

  dataset = _autocast_dataset(dataset)

  # Check whether the dataset should be prefetched.
  prefetch_buffer = None
  if isinstance(original_dataset, dataset_ops.PrefetchDataset):
    prefetch_buffer = original_dataset._buffer_size  # pylint: disable=protected-access
  elif (
      isinstance(original_dataset, dataset_ops.DatasetV1Adapter)
      and isinstance(original_dataset._dataset, dataset_ops.PrefetchDataset)):  # pylint: disable=protected-access
    prefetch_buffer = original_dataset._dataset._buffer_size  # pylint: disable=protected-access

  if prefetch_buffer is not None:
    dataset = dataset.prefetch(prefetch_buffer)

  return dataset, size


def _autocast_dataset(dataset):
  if (not base_layer_utils.v2_dtype_behavior_enabled()
      or not any(spec.dtype == dtypes.float64
                 for spec in nest.flatten(dataset.element_spec))):
    return dataset

  def autocast_structure(*structure):
    def autocast_tensor(tensor):
      if tensor.dtype == dtypes.float64:
        return math_ops.cast(tensor, dtypes.float32)
      return tensor

    return nest.map_structure(autocast_tensor, structure)

  return dataset.map(autocast_structure)


def _handle_renamed_arg(old_arg, new_arg, old_arg_name, new_arg_name,
                        is_supplied_fn):
  if is_supplied_fn(old_arg):
    calling_class = inspect.stack()[1][0].f_locals["self"].__class__
    logging.warning(
        f"From {calling_class.__name__}: Argument '{old_arg_name}' is"
        f" deprecated, use '{new_arg_name}' instead.")
    if is_supplied_fn(new_arg):
      raise ValueError(f"Arguments {old_arg_name} and {new_arg_name}"
                       " cannot be used together.")
    return old_arg
  return new_arg


class _IpuModelBase(KerasModel):
  """Base class for IPU Keras models"""
  def __init__(self,
               gradient_accumulation_count,
               shard_count,
               layer_replacement=False,
               **kwargs):
    name = kwargs.pop("name", None)
    super(_IpuModelBase, self).__init__(dtype=None, name=name)

    self.args = kwargs
    self.gradient_accumulation_count = gradient_accumulation_count

    self.built = False
    self.history = None
    self.infeed = None
    self.outfeed = None
    self.last_ds = None
    self.last_mode = None
    self._per_mode_loop_fns = {}

    # Round the shard count to the next power of two
    self.shard_count = 2**int(math.ceil(math.log2(shard_count)))
    self._replication_factor = None

    self._layer_replacer = IPULayerReplacer() if layer_replacement else None

  def build(self, input_shape):
    pass

  # pylint: disable=arguments-differ
  def call(self, _):
    raise ValueError(self.__class__.__name__ +
                     " can only be called through the `fit`, "
                     "`evaluate` or `predict` interfaces.")

  @trackable.no_automatic_dependency_tracking
  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              **kwargs):
    if isinstance(optimizer, optimizers.Optimizer):
      raise ValueError(
          "Optimizer must be a native TensorFlow or Keras V2 optimizer"
          " found in tensorflow.keras.optimizer_v2.")

    if not isinstance(loss_weights, (list, type(None))):
      raise ValueError("loss_weights can only be specified as a list.")

    _validate_args(kwargs, "compile")

    # If a previous run has been done then the data feeds will need to
    # be reset.
    self.infeed = None
    self.outfeed = None

    return super(_IpuModelBase, self).compile(optimizer=optimizer,
                                              loss=loss,
                                              metrics=metrics,
                                              loss_weights=loss_weights,
                                              **kwargs)

  # This is a duplication of the graph compilation section of the method
  # Model.compile in training.py
  def _add_loss(self, targets):
    self._is_compiled = True

    target_tensors = targets if isinstance(targets,
                                           (list, tuple)) else [targets]

    target_tensors = self._process_target_tensor_for_compile(target_tensors)

    # Prepare list of loss functions, same size of model outputs.
    self.loss_functions = training_utils_v1.prepare_loss_functions(
        self.loss, self.output_names)

    self._training_endpoints = []
    for o, n, l, t in zip(self.outputs, self.output_names, self.loss_functions,
                          target_tensors):
      endpoint = keras_training._TrainingEndpoint(o, n, l)  # pylint: disable=protected-access
      endpoint.create_training_target(t, run_eagerly=self.run_eagerly)
      self._training_endpoints.append(endpoint)

    # Prepare list loss weights, same size of model outputs.
    training_utils_v1.prepare_loss_weights(self._training_endpoints,
                                           self.loss_weights)

    masks = self._prepare_output_masks()

    # Create the stateful metrics operations only once.
    if not hasattr(self, "_per_output_metrics"):
      self._cache_output_metric_attributes(self._compile_metrics, None)

      # Set metric attributes for each output.
      self._set_metric_attributes()

    # Invoke metric functions (unweighted) for all the outputs.
    metrics = self._handle_metrics(
        self.outputs,
        targets=self._targets,
        skip_target_masks=self._prepare_skip_target_masks(),
        masks=masks)

    # Prepare sample weight modes. List with the same length as model outputs.
    training_utils_v1.prepare_sample_weight_modes(self._training_endpoints,
                                                  None)

    # Creates the model loss
    self.total_loss = self._prepare_total_loss(masks)
    return [self.total_loss] + metrics

  def _get_internal_run_loop(self, mode, run_loop_kwargs):
    # Cache the built run_loops w.r.t the mode and run kwargs.
    key = (mode, tuple(sorted(run_loop_kwargs.items())))
    if key not in self._per_mode_loop_fns:
      fn = partial(
          self.__class__._internal_run_loop,  # pylint: disable=protected-access
          self,
          run_loop_kwargs=run_loop_kwargs)
      self._per_mode_loop_fns[key] = def_function.function(
          fn, autograph=False, experimental_compile=True)
    return self._per_mode_loop_fns[key]

  def _internal_run_loop(self,
                         infeed_queue,
                         outfeed_queue,
                         repeat_count,
                         mode,
                         run_loop_kwargs=None):
    raise NotImplementedError(
        "_IpuModelBase should not be used directly.  Use PipelinedModel or "
        "Model instead.")

  def _get_wrapped_optimizer(self):
    opt = self.optimizer

    # Unwrap native TF optimizers from a Keras wrapper
    if isinstance(opt, optimizers.TFOptimizer):
      return _TensorflowOptimizerWrapper(self, opt.optimizer)

    # Convert native Keras optimizers to TF optimizers
    elif isinstance(opt, optimizer_v2.OptimizerV2):
      return _KerasOptimizerWrapper(self, opt)

    # Other optimizer types are not supported
    else:
      raise ValueError(
          "Only Keras optimizer_v2.Optimizer and TensorFlow native"
          " training.Optimizer subclasses are supported.")

  @trackable.no_automatic_dependency_tracking
  def _do_internal(self, mode, ds, size, epochs, verbose, callbacks,
                   initial_epoch, steps_per_epoch, steps_per_run,
                   prefetch_depth, **kwargs):
    run_loop_kwargs = kwargs.pop("run_loop_kwargs", {})
    self.args = kwargs

    # Figure out if we need to recreate the iterator after each epoch.
    recreate_iterator = False
    require_steps_per_epoch = False
    verify_dataset_length = False
    derived_steps_per_epoch = None

    dataset_length = K.get_value(cardinality.cardinality(ds))
    if dataset_length == cardinality.INFINITE:
      # An infinite dataset can be walked over indefinitely, but the user
      # must specify how many steps there are in each epoch.
      require_steps_per_epoch = True
    elif dataset_length == cardinality.UNKNOWN:
      # An unknown length of dataset must be restarted on each epoch, and
      # the user must specify the number of steps per epoch.
      require_steps_per_epoch = True
      recreate_iterator = True
    else:
      # A known length of dataset must be restarted on each epoch, but the
      # user doesn't have to specify the number of steps per epoch.
      recreate_iterator = True
      verify_dataset_length = True

    def check_dataset_not_exhausted(length):
      if steps_per_epoch * self.gradient_accumulation_count > length:
        steps_string = "Steps per epoch" if mode == ModeKeys.TRAIN else "Steps"
        raise ValueError(
            steps_string + " times gradient accumulation count (%d x %d) is"
            " greater than the number of mini-batches in the dataset (%d)." %
            (steps_per_epoch, self.gradient_accumulation_count, length))

    def derive_steps_per_epoch(size):
      if steps_per_run is None:
        _steps_per_epoch_per_replica = size // (
            self.gradient_accumulation_count * self.replication_factor)
        if _steps_per_epoch_per_replica == 0:
          raise ValueError(
              "The number of mini-batches in the dataset (%d) must be at least"
              " the gradient accumulation count (%d) multiplied by the"
              " replication factor (%d)." %
              (size, self.gradient_accumulation_count,
               self.replication_factor))
        _derived_steps_per_epoch = (_steps_per_epoch_per_replica *
                                    self.replication_factor)
        if size % (self.gradient_accumulation_count *
                   self.replication_factor) != 0:
          # Log a warning if samples have been dropped
          if mode == ModeKeys.TRAIN:
            steps_string = "steps per epoch (steps_per_epoch)"
          else:
            steps_string = "steps"
          logging.warning(
              ' The number of mini-batches in the dataset (%d) must be a'
              ' multiple of the gradient accumulation count (%d) multiplied by'
              ' the replication factor (%d). Samples have been dropped to give'
              ' a dataset of %d mini-batches. Adjust the size of the supplied'
              ' dataset or specify the %s if you do not want this to happen.' %
              (size, self.gradient_accumulation_count, self.replication_factor,
               _derived_steps_per_epoch * self.gradient_accumulation_count,
               steps_string))
      else:
        _steps_per_epoch_per_replica = size // (
            self.gradient_accumulation_count * self.replication_factor *
            steps_per_run)
        if _steps_per_epoch_per_replica == 0:
          raise ValueError(
              "The number of mini-batches in the dataset (%d) must be at least"
              " the gradient accumulation count (%d) multiplied by the"
              " replication factor (%d) multiplied by steps_per_run (%d)." %
              (size, self.gradient_accumulation_count, self.replication_factor,
               steps_per_run))
        _derived_steps_per_epoch = (_steps_per_epoch_per_replica *
                                    self.replication_factor * steps_per_run)
        if size % (self.gradient_accumulation_count * self.replication_factor *
                   steps_per_run) != 0:
          # Log a warning if samples have been dropped
          if mode == ModeKeys.TRAIN:
            steps_string = "steps per epoch (steps_per_epoch)"
          else:
            steps_string = "steps"
          logging.warning(
              ' The number of mini-batches in the dataset (%d) must be a'
              ' multiple of the gradient accumulation count (%d) multiplied by'
              ' the replication factor (%d) multiplied by steps_per_run (%d).'
              ' Samples have been dropped to give a dataset of %d mini-batches.'
              ' Adjust the size of the supplied dataset or specify the %s if'
              ' you do not want this to happen.' %
              (size, self.gradient_accumulation_count, self.replication_factor,
               steps_per_run, _derived_steps_per_epoch *
               self.gradient_accumulation_count, steps_string))
      return _derived_steps_per_epoch

    if require_steps_per_epoch:
      if not steps_per_epoch:
        if size is None:
          raise ValueError(
              "When using an infinitely repeating dataset, you must provide the"
              " number of steps per epoch (steps_per_epoch).")
        else:
          # cardinality.UNKNOWN but known size
          # This applies to Numpy arrays that have been converted into a
          # dataset by _get_dataset_and_count()
          derived_steps_per_epoch = derive_steps_per_epoch(size)
      else:
        if size is not None:
          check_dataset_not_exhausted(size)

    if steps_per_epoch is not None:
      if steps_per_run is not None:
        if steps_per_epoch % (steps_per_run * self.replication_factor) != 0:
          if mode == ModeKeys.TRAIN:
            steps_string = "number of steps in an epoch 'steps_per_epoch'"
            loop_type = "training"
          else:
            steps_string = "number of steps 'steps'"
            if mode == ModeKeys.PREDICT:
              loop_type = "inference"
            else:
              loop_type = "evaluation"
          raise ValueError(
              self.__class__.__name__ + " requires the " + steps_string +
              " (%d) to be evenly divisible by the number of steps per"
              " execution of the on-device %s loop"
              " 'steps_per_run' (%d) multiplied by the replication factor"
              " (%d)." % (steps_per_epoch, loop_type, steps_per_run,
                          self.replication_factor))
      elif steps_per_epoch % self.replication_factor != 0:
        if mode == ModeKeys.TRAIN:
          raise ValueError(
              self.__class__.__name__ + " requires the number of steps in an"
              " epoch 'steps_per_epoch' (%d) to be evenly divisible by the"
              " replication factor (%d)." %
              (steps_per_epoch, self.replication_factor))
        else:
          raise ValueError(
              self.__class__.__name__ + " requires the number of steps"
              " 'steps' (%d) to be evenly divisible by the replication factor"
              " (%d)." % (steps_per_epoch, self.replication_factor))

    # If there is a fixed length of dataset, and the user has also specified
    # a steps_per_epoch, then check that this won't exhaust the dataset.
    if verify_dataset_length and steps_per_epoch:
      check_dataset_not_exhausted(dataset_length)

    # Find out how many mini-batches, steps, repeats, and outer loops.
    if steps_per_epoch is not None:
      mini_batches_per_epoch = steps_per_epoch
    elif derived_steps_per_epoch is not None:
      mini_batches_per_epoch = derived_steps_per_epoch
    else:
      mini_batches_per_epoch = None

    if mini_batches_per_epoch is not None:
      mini_batches_per_epoch *= self.gradient_accumulation_count

    # If mini_batches_per_epoch is None then this will infer the value to use,
    # else it will check that the value is valid.
    steps_name_string = 'steps_per_epoch' if mode == ModeKeys.TRAIN else "steps"
    mini_batches_per_epoch = training_utils_v1.infer_steps_for_dataset(
        self, ds, mini_batches_per_epoch, epochs, steps_name=steps_name_string)

    # These errors can only occur when steps_per_epoch is not passed in and the
    # dataset is of finite cardinality. In that case mini_batches_per_epoch is
    # set to the number of mini-batches in the dataset by
    # infer_steps_for_dataset().
    if steps_per_run is None:
      if mini_batches_per_epoch % (self.gradient_accumulation_count *
                                   self.replication_factor) != 0:
        raise ValueError(
            self.__class__.__name__ + " requires the number of mini-batches in"
            " the dataset (%d) to be evenly divisible by the gradient"
            " accumulation count (%d) multiplied by the replication factor (%d)"
            % (mini_batches_per_epoch, self.gradient_accumulation_count,
               self.replication_factor))
    else:
      if mini_batches_per_epoch % (self.gradient_accumulation_count *
                                   self.replication_factor *
                                   steps_per_run) != 0:
        raise ValueError(
            self.__class__.__name__ + " requires the number of mini-batches in"
            " the dataset (%d) to be evenly divisible by the product of"
            " 'steps_per_run' (%d), the gradient accumulation count (%d), and"
            " the replication factor (%d)." %
            (mini_batches_per_epoch, steps_per_run,
             self.gradient_accumulation_count, self.replication_factor))

    steps_per_epoch_per_replica = (
        mini_batches_per_epoch /
        (self.gradient_accumulation_count * self.replication_factor))
    if not steps_per_run:
      steps_per_run = steps_per_epoch_per_replica

    outer_loop_count = int(steps_per_epoch_per_replica / steps_per_run)

    total_batches = mini_batches_per_epoch * (epochs - initial_epoch)

    # Prepare for progress reporting.
    callbacks = cbks.configure_callbacks(
        callbacks,
        self,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch_per_replica,
        verbose=verbose,
        count_mode='steps',
        mode=mode)

    # If the dataset or mode has changed, then we need to recreate the feeds
    if not self.last_ds or self.last_ds() != ds or self.last_mode != mode:
      self.infeed = None
      self.outfeed = None
      self.last_ds = weakref.ref(ds)
      self.last_mode = mode

    # Create infeed and outfeed
    if not self.infeed or not self.outfeed:
      self.infeed = ipu_infeed_queue.IPUInfeedQueue(
          ds,
          prefetch_depth=prefetch_depth)
      self.outfeed = ipu_outfeed_queue.IPUOutfeedQueue()

    initial_epoch = self._maybe_load_initial_epoch_from_ckpt(
        initial_epoch, mode)

    callbacks.on_train_begin(mode)

    # Ask the poplar executor to create a dataset iterator
    self.infeed.initializer  # pylint: disable=pointless-statement

    # Aggregator for combining the various outputs/metrics together
    if mode != ModeKeys.PREDICT:
      aggregator = training_utils_v1.MetricsAggregator(
          use_steps=True, steps=mini_batches_per_epoch)
    else:
      aggregator = training_utils_v1.OutputsAggregator(use_steps=True,
                                                       steps=total_batches)

    # Outer loop
    try:
      for epoch in range(initial_epoch, epochs):
        if callbacks.model.stop_training:
          break

        epoch_logs = {}
        callbacks.on_epoch_begin(epoch, epoch_logs)

        # Clear metrics
        self.reset_metrics()

        for run in range(outer_loop_count):

          batch_num = run * steps_per_run
          batch_logs = {
              'batch': batch_num,
              'size': 1,
              'num_steps': steps_per_run
          }
          callbacks.on_batch_begin(batch_num, batch_logs)

          # Create and run the core graph.
          strategy = distribution_strategy_context.get_strategy()
          # Pass the run_loop_kwargs through to allow children to pass kwargs
          # from their fit/evaluate/predict calls to their _internal_run_loop.
          func = self._get_internal_run_loop(mode, run_loop_kwargs)
          strategy.run(func,
                       args=[self.infeed, self.outfeed, steps_per_run, mode])

          # Send an end of batches
          callbacks.on_batch_end(batch_num, batch_logs)

          # After the first call we can update the callbacks to include
          # the metrics.
          if epoch == initial_epoch and run == 0:
            cbks.set_callback_parameters(
                callbacks,
                self,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch_per_replica,
                verbose=verbose,
                mode=mode)

        # Restart the iterator at the end of the epoch if necessary
        if recreate_iterator:
          self.infeed.deleter  # pylint: disable=pointless-statement
          self.infeed.initializer  # pylint: disable=pointless-statement

        # Fetch the outfeed for the history
        if utils.use_synthetic_data_for(utils.SyntheticDataCategory.Outfeed):
          empty_results = self.outfeed.dequeue()
          results = []
          for empty_result in empty_results:
            shape = empty_result.shape.as_list()
            # The first dimension is the number of iterations
            shape[0] = mini_batches_per_epoch
            dtype = empty_result.dtype.as_numpy_dtype()
            results.append(np.full(shape, np.nan, dtype=dtype))
        else:
          results = self.outfeed.dequeue()

          # For fit() and evaluate() the shape of results is
          #   (1+num_metrics, steps_per_epoch X GA)
          # or with replication:
          #   (1+num_metrics, steps_per_epoch x GA / RF, RF)
          #
          # For predict() the shape of results is
          #   (num_outputs, steps_per_epoch x GA, batch_size, output_shape)
          # or with replication:
          #   (num_outputs, RF, steps_per_epoch x GA/RF, batch_size, output_shape)
          #
          # where steps_per_epoch is the value passed to this function (or derived)
          #       GA is gradient accumulation count
          #       RF is replication factor
          #       output_shape may have multiple dimensions
          results = [
              map(lambda x: x.numpy(), r) for r in nest.flatten(results)
          ]

        results = zip(*results)

        if self.replication_factor > 1:
          # "Transpose" all the outfeed elements.
          def gen(results):
            for t in results:
              for i in range(self.replication_factor):
                yield tuple(x[i] for x in t)

          results = gen(results)

        # Get the final loss and metrics
        r = next(results)

        aggregator.create(r)
        aggregator.aggregate(r)

        for r in results:
          aggregator.aggregate(r)

        aggregator.finalize()
        results = aggregator.results

        # Store only the final losses/metrics for the epoch log.
        # Loss is the average across the epoch.
        # For other metrics, if there is replication, the value
        # is the final value for one of the replicas.
        cbks.make_logs(self, epoch_logs, results, mode)
        callbacks.on_epoch_end(epoch, epoch_logs)

      callbacks.on_train_end(mode)

    # Close the infeed and outfeed queues at the end of the epoch
    finally:
      try:
        self.infeed.deleter  # pylint: disable=pointless-statement
        self.outfeed.deleter  # pylint: disable=pointless-statement
      except NotFoundError as e:
        if str(e).startswith("Outfeed with id="):
          pass

    # fit() method returns the history object
    if mode == ModeKeys.TRAIN:
      return self.history

    # evaluate() and predict() return the aggregated results
    return aggregator.results

  @trackable.no_automatic_dependency_tracking
  def fit(self,
          x,
          y,
          batch_size,
          epochs,
          verbose,
          callbacks,
          shuffle,
          initial_epoch,
          steps_per_epoch,
          steps_per_run,
          prefetch_depth=None,
          **kwargs):

    if batch_size and isinstance(x, dataset_ops.DatasetV2):
      raise ValueError("Do not specify `batch_size` in " +
                       self.__class__.__name__ + ".fit(). Use the"
                       " Dataset.batch() method to apply batching at the input"
                       " dataset level.")

    ds, size = _get_dataset_and_count(x, y, batch_size)

    _validate_args(kwargs, "fit")
    _validate_dataset_element_count(ds, self._num_inputs + self._num_outputs,
                                    "fit")

    self._assert_compile_was_called()

    return self._do_internal(ModeKeys.TRAIN, ds, size, epochs, verbose,
                             callbacks, initial_epoch, steps_per_epoch,
                             steps_per_run, prefetch_depth, **kwargs)

  def evaluate(self,
               x=None,
               y=None,
               batch_size=None,
               verbose=1,
               steps=None,
               callbacks=None,
               steps_per_run=None,
               prefetch_depth=None,
               **kwargs):
    if batch_size and isinstance(x, dataset_ops.DatasetV2):
      raise ValueError("Do not specify `batch_size` in " +
                       self.__class__.__name__ + ".evaluate(). Use the "
                       "Dataset.batch() method to apply batching at the input "
                       "dataset level.")

    ds, size = _get_dataset_and_count(x, y, batch_size)

    _validate_args(kwargs, "evaluate")
    _validate_dataset_element_count(ds, self._num_inputs + self._num_outputs,
                                    "evaluate")

    self._assert_compile_was_called()

    return self._do_internal(ModeKeys.TEST, ds, size, 1, verbose, callbacks, 0,
                             steps, steps_per_run, prefetch_depth, **kwargs)

  def predict(self,
              x,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              steps_per_run=None,
              prefetch_depth=None,
              **kwargs):
    if batch_size and isinstance(x, dataset_ops.DatasetV2):
      raise ValueError("Do not specify `batch_size` in " +
                       self.__class__.__name__ + ".predict(). Use the "
                       "Dataset.batch() method to apply batching at the input "
                       "dataset level.")

    ds, size = _get_dataset_and_count(x, None, batch_size)

    _validate_args(kwargs, "predict")
    _validate_dataset_element_count(ds, self._num_inputs, "predict")

    result = self._do_internal(ModeKeys.PREDICT, ds, size, 1, verbose,
                               callbacks, 0, steps, steps_per_run,
                               prefetch_depth, **kwargs)

    # Sequential models only support a single output
    # but Model/PipelineModel may have multiple outputs
    if len(result) == 1:
      return result[0]

    # Convert from tuple to list to match output from Keras Model
    return list(result)

  @property
  def replication_factor(self):
    if not self._replication_factor:
      strategy = distribution_strategy_context.get_strategy()
      device_string = strategy.extended.non_slot_devices(None)
      current_device = tf_device.DeviceSpec.from_string(device_string)

      if current_device.device_type != "IPU":
        raise ValueError(self.__class__.__name__ +
                         " can only be used on an IPU device.")

      num_ipus = utils.get_num_of_ipus_in_device(device_string)
      if self.shard_count > num_ipus:
        raise ValueError(
            "Current device has %d IPUs attached, however the current model "
            "requires a multiple of %d IPUs." % (num_ipus, self.shard_count))

      self._replication_factor = int(num_ipus / self.shard_count)

    return self._replication_factor


class IPUSequential(_IpuModelBase):
  """A Keras Sequential class specifically targeting the IPU. This is
  similar to the Keras Sequential model class, but it also supports the
  accumulation of gradient deltas, and an on-device training/inference loop.

  There are some limitations with this Sequential class compared to the
  standard Keras Sequential class:

  - Keras V1 optimizers cannot be used.
  - Loss weightings can only be specified as a list, not a callable.
  - Weighted metrics, target tensors and sample weight mode are not supported.
  - Validation cannot be performed as part of the `fit` loop.
  - The model cannot be called using the __call__() interface.
  - The model cannot be saved using the `save` interface.

  Example:

  .. code-block:: python

    dataset = ...

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = ipu.keras.Sequential([
        keras.layers.Dense(4),
        keras.layers.Dense(4),
        keras.layers.Dense(4),
      ])

      m.compile('sgd', loss='mse')

      m.fit(dataset, steps_per_epoch=144)

  """
  def __init__(self,
               layers=None,
               gradient_accumulation_count=1,
               gradient_accumulation_dtype=None,
               layer_replacement=False,
               accumulation_count=1,
               accumulation_dtype=None):
    """
    Creates a Keras sequential model, optimized to run on the IPU.

    Args:
        layers: A Python list of Keras Layers.
        gradient_accumulation_count: The number of mini-batches to process
            while accumulating their gradients, before running a
            parameter/weight update step.
        gradient_accumulation_dtype: The data type used for the gradient
          accumulation buffer. One of:

          - `None`: Use an accumulator of the same type as the variable type.
          - A `DType`: Use this type for all the accumulators.
          - A callable that takes the variable and returns a `DType`: Allows
            specifying the accumulator type on a per-variable basis.

          The gradients passed to `Optimizer.apply_gradients` will have the
          dtype requested here. If that dtype is different from the variable
          dtype a cast is needed at some point to make them compatible. This can
          be done by using a custom optimizer.
        layer_replacement: If enabled (True), Keras layers will be substituted
          with IPU Keras implementations, when possible.
    """

    super().__init__(gradient_accumulation_count=gradient_accumulation_count,
                     shard_count=1,
                     layer_replacement=layer_replacement)

    if not isinstance(layers, list):
      raise ValueError("IPU Sequential requires a list of Layers.")

    for s in layers:
      if not isinstance(s, Layer):
        raise ValueError("An IPU Sequential's list of Layers may only contain"
                         " Keras Layers.")

    self.gradient_accumulation_count = gradient_accumulation_count
    self.gradient_accumulation_dtype = gradient_accumulation_dtype
    self.model_layers = layers
    self._num_inputs = 1
    self._num_outputs = 1

  def build(self, input_shape):
    """Builds the model based on input shapes received.

    Args:
     input_shape: Single tuple, TensorShape, or list of shapes, where shapes
         are tuples, integers, or TensorShapes.
    """
    s = input_shape
    for l in self.model_layers:
      l.build(s)
      s = l.compute_output_shape(s)
    self.built = True

  @trackable.no_automatic_dependency_tracking
  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              **kwargs):
    """
    This provides the same functionality as the Keras Sequential ``compile``
    method.

    Certain features are not supported by the IPU Sequential class:

      - sample_weight_mode
      - weighted_metrics
      - target_tensors

    Note that loss weights can only be specified as a list.

    Args:
      optimizer: String (name of optimizer) or optimizer instance. See
        `tf.keras.optimizers`. An instance of a subclass of
        `tensorflow.python.training.optimizer` can also be used.
      loss: String (name of objective function), objective function or
        `tf.keras.losses.Loss` instance. See `tf.keras.losses`.
        IPU-specific loss classes can also be used. See the documentation in
        :py:mod:`tensorflow.python.ipu.keras.losses` for usage instructions.
        An objective function is any callable with the signature
        `scalar_loss = fn(y_true, y_pred)`. If the model has multiple outputs,
        you can use a different loss on each output by passing a dictionary or
        a list of losses. The loss value that will be minimized by the model
        will then be the sum of all individual losses.
      metrics: List of metrics to be evaluated by the model during training and
        testing. Typically you will use `metrics=['accuracy']`. To specify
        different metrics for different outputs of a multi-output model, you
        could pass a dictionary, such as `metrics={'output_a': 'accuracy',
        'output_b': ['accuracy', 'mse']}`, or a list (`len = len(outputs)`) of
        lists of metrics such as
        `metrics=[['accuracy'], ['accuracy', 'mse']]` or `metrics=['accuracy',
        ['accuracy', 'mse']]`.
      loss_weights: Optional list specifying scalar coefficients (Python floats)
        to weight the loss contributions of different model outputs. The loss
        value that will be minimized by the model will then be the weighted sum
        of all individual losses, weighted by the loss_weights coefficients.
        The list is expected to have a 1:1 mapping to the model's outputs.
    Raises:
      ValueError: if there are invalid arguments.
    """
    return super().compile(optimizer, loss, metrics, loss_weights, **kwargs)

  def _internal_run_loop(self,
                         infeed_queue,
                         outfeed_queue,
                         repeat_count,
                         mode,
                         run_loop_kwargs=None):
    training = mode == ModeKeys.TRAIN

    def main_body(inputs):

      if not self.inputs:
        self._set_input_attrs(inputs)

      x = inputs
      for l in self.model_layers:
        kwargs = {}
        argspec = tf_inspect.getfullargspec(l.call).args
        if 'training' in argspec:
          kwargs['training'] = training
        x = l(x, **kwargs)

      return x

    def inference_body(inputs):
      x = main_body(inputs)

      outfeed_queue.enqueue([x])
      return []

    def training_body(inputs, targets):

      x = main_body(inputs)
      self._set_output_attrs(x)

      l = self._add_loss(targets)

      outfeed_queue.enqueue(l)

      if not self.trainable_weights:
        raise ValueError(
            "Sequential must have at least one trainable parameter.")

      if training:
        opt = self._get_wrapped_optimizer()
        if self.gradient_accumulation_count > 1:
          opt = gradient_accumulation_optimizer.GradientAccumulationOptimizerV2(
              opt,
              self.gradient_accumulation_count,
              dtype=self.gradient_accumulation_dtype)

        # Get gradients and apply them to the trainable variables
        grads_and_vars = opt.compute_gradients(l[0], self.trainable_variables)
        opt.apply_gradients(grads_and_vars)

      return []

    def body(*args):
      fn = functional_ops.outlined_function(
          inference_body if mode == ModeKeys.PREDICT else training_body)
      return fn(*args)

    result = loops.repeat(int(repeat_count * self.gradient_accumulation_count),
                          body,
                          infeed_queue=infeed_queue)

    return result.outputs

  # pylint: disable=arguments-differ
  @trackable.no_automatic_dependency_tracking
  def fit(self,
          x=None,
          y=None,
          *,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          shuffle=True,
          initial_epoch=0,
          steps_per_epoch=None,
          steps_per_run=None,
          prefetch_depth=None,
          **kwargs):  # pylint: disable=useless-super-delegation
    """
    This provides equivalent functionality to the Keras Sequential `fit` method.

    Note that `batch_size` here is the number of samples that is processed on
    each replica in each forward pass. This is referred to as the mini-batch
    size. Prepare Dataset input on this basis.

    Each step (per replica) will process mini-batch multiplied by gradient
    accumulation count samples before updating the weights. Therefore, the
    effective batch size for a weight update is the mini-batch size multiplied
    by the gradient accumulation count multiplied by the replication factor.

    The number of weight update steps per epoch is the `steps_per_epoch` value
    divided by the replication factor, and this is the number of steps that
    will be shown in the progress bar.

    For a finite dataset the iterator over the data will be reset at the start
    of each epoch. This means that the dataset does not need to be repeated
    `epochs` times if `steps_per_epoch` is not specified. It also means that if
    a small value for `steps_per_epoch` is supplied then not all samples will be
    used.

    A shuffled Dataset should be supplied. Non-dataset inputs (as described in
    the parameters section below) for `x` and `y` will be accepted but will not
    be shuffled, and this may lead to over-fitting.

    Input/Target data of the following types will be converted into a Dataset
    internally based on the batch_size, dropping any partial batch: Numpy array
    (or list of arrays), TensorFlow tensor (or list of tensors) or dict.

    Only the parameters documented below are supported.

    Args:
      x: Input data.
        It could be:

        - A Numpy array (or array-like), or a list of arrays (in case the model
          has multiple inputs).
        - A TensorFlow tensor, or a list of tensors (in case the model has
          multiple inputs).
        - A dict mapping input names to the corresponding array/tensors, if the
          model has named inputs.
        - A `tf.data` dataset. This must return a tuple of `(inputs, targets)`.
      y: Target data. Like the input data `x`, it could be either Numpy array(s)
        or TensorFlow tensor(s). It should be consistent with `x` (you cannot
        have Numpy inputs and tensor targets, or tensor inputs and Numpy
        targets). If `x` is a dataset then `y` must not be specified (since
        targets will be obtained from `x`).
      batch_size: Integer or `None`. The mini-batch size to use for input data
        supplied as Numpy array(s) or TensorFlow tensor(s). If `x` is a dataset
        then `batch_size` must not be specified.
      epochs: Integer. Number of epochs to train the model. The number of steps
        performed per epoch is defined by the `steps_per_epoch` parameter, or
        calculated according to the constraints described below.
        Note that in conjunction with `initial_epoch`, `epochs` is to be
        understood as "final epoch". The model is not trained for a number of
        iterations given by `epochs`, but merely until the epoch of index
        `epochs` is reached.
      verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar,
        2 = one line per epoch. Note that the progress bar is not particularly
        useful when logged to a file, so verbose=2 is recommended when not
        running interactively (for example, in a production environment).
      callbacks: List of `keras.callbacks.Callback` instances. List of callbacks
        to apply during training. See `tf.keras.callbacks` in the TensorFlow
        documentation.
      shuffle: **NOT SUPPORTED**. This will be supported in a future release.
      initial_epoch: Integer. Epoch at which to start training (useful for
        resuming a previous training run).
      steps_per_epoch: Integer or `None`. Specifies the total number of steps to
        be performed per epoch.
        The following constraints apply:

        - If `steps_per_run` is specified then the value for `steps_per_epoch`
          must be evenly divisible by `steps_per_run` multiplied by the
          replication factor. Otherwise it must be divisible by the
          replication factor.
        - For an infinitely repeating dataset a value for `steps_per_epoch`
          must be specified.
        - For a finite dataset if `steps_per_epoch` is specified then it must
          contain at least mini-batch size * gradient accumulation count *
          `steps` samples.
        - For a dataset of known finite length a value for `steps_per_epoch`
          will be calculated if no value is specified. The number of
          samples in the dataset must be a multiple of the mini-batch size
          multiplied by the gradient accumulation count multiplied by the
          replication factor (multiplied by `steps_per_run` if it is
          specified).
        - For array or tensor inputs a value for `steps_per_epoch` will be
          calculated if no value is specified. If the number of samples provided
          is not a multiple of the mini-batch size multiplied by the gradient
          accumulation count multiplied by the replication factor (multiplied by
          `steps_per_run` if it is specified) then samples will be dropped when
          deriving a value for `steps_per_epoch` and a warning will be logged.
      steps_per_run: Integer or `None`. Specifies how many steps will be
        performed per replica on each hardware execution.
        If not specified this will be set to `steps_per_epoch` (which will
        be calculated if not specified) divided by the replication factor.
        The value of 'steps_per_epoch' (if specified) must be evenly
        divisible by `steps_per_run` multiplied by the replication factor.
      prefetch_depth: Integer or `None`. The `prefetch_depth` to be used by the
        :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue`
        that is created internally by this function. See the
        :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue`
        documentation.
    Returns:
      A `History` object. Its `History.history` attribute is a record of
      training loss values and metrics values at successive epochs.
    Raises:
      ValueError: if there are invalid arguments.
    """
    return super().fit(x,
                       y,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       callbacks=callbacks,
                       shuffle=shuffle,
                       initial_epoch=initial_epoch,
                       steps_per_epoch=steps_per_epoch,
                       steps_per_run=steps_per_run,
                       prefetch_depth=prefetch_depth,
                       **kwargs)

  # pylint: disable=arguments-differ
  def evaluate(self,
               x=None,
               y=None,
               *,
               batch_size=None,
               verbose=1,
               steps=None,
               callbacks=None,
               steps_per_run=None,
               prefetch_depth=None,
               **kwargs):  # pylint: disable=useless-super-delegation
    """
    This provides equivalent functionality to the Keras Sequential `evaluate`
    method.

    Note that `batch_size` here is the number of samples that is processed on
    each replica in each forward pass. This is referred to as the mini-batch
    size. Prepare Dataset input on this basis.

    Each step (per replica) will process mini-batch multiplied by gradient
    accumulation count samples. Therefore, the effective batch size is the
    mini-batch size multiplied by the gradient accumulation count multiplied by
    the replication factor.

    Input/Target data of the following types will be converted into a Dataset
    internally based on the batch_size, dropping any partial batch: Numpy array
    (or list of arrays), TensorFlow tensor (or list of tensors) or dict.

    Only the parameters documented below are supported.

    Args:
      x: Input data. It could be:

        - A Numpy array (or array-like), or a list of arrays (in case the model
          has multiple inputs).
        - A TensorFlow tensor, or a list of tensors (in case the model has
          multiple inputs).
        - A dict mapping input names to the corresponding array/tensors, if the
          model has named inputs.
        - A `tf.data` dataset. This must return a tuple of `(inputs, targets)`.
      y: Target data. Like the input data `x`, it could be either Numpy array(s)
        or TensorFlow tensor(s). It should be consistent with `x` (you cannot
        have Numpy inputs and tensor targets, or tensor inputs and Numpy
        targets). If `x` is a dataset then `y` must not be specified (since
        targets will be obtained from `x`).
      batch_size: Integer or `None`. The mini-batch size to use for input data
        supplied as Numpy array(s) or TensorFlow tensor(s). If `x` is a dataset
        then `batch_size` must not be specified.
      verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar. **HAS NO
        EFFECT** - the progress bar is not displayed. This will be corrected in
        a future release.
      steps: Integer or `None`. Specifies the total number of steps to be
        performed. The following constraints apply:

        - If `steps_per_run` is specified then the value for `steps`
          must be evenly divisible by `steps_per_run` multiplied by the
          replication factor. Otherwise it must be divisible by the
          replication factor.
        - For an infinitely repeating dataset a value for `steps`
          must be specified.
        - For a finite dataset if `steps` is specified then it must contain at
          least mini-batch size * gradient accumulation count * `steps` samples.
          For a dataset of known finite length a value for `steps`
          will be calculated if no value is specified. The number of
          samples in the dataset must be a multiple of the mini-batch size
          multiplied by the gradient accumulation count multiplied by the
          replication factor (multiplied by `steps_per_run` if it is specified).
        - For array or tensor inputs a value for `steps` will be calculated
          if no value is specified. If the number of samples provided is not a
          multiple of the mini-batch size multiplied by the gradient
          accumulation count multiplied by the replication factor (multiplied by
          `steps_per_run` if it is specified) then samples will be dropped when
          deriving a value for `steps` and a warning will be logged.
      callbacks: List of keras.callbacks.Callback instances. List of callbacks
        to apply during evaluation. **KNOWN ISSUE**: `evaluate` currently
        calls the callback functions applicable to `fit` rather than those
        applicable to `evaluate`. This will be corrected in a future release.
      steps_per_run: Integer or `None`. Specifies how many steps will be
        performed per replica on each hardware execution.
        If not specified this will be set to `steps` (which will be calculated
        if not specified) divided by the replication factor.
        The value of `steps` (if specified) must be evenly divisible by
        `steps_per_run` multiplied by the replication factor.
      prefetch_depth: Integer or `None`. The `prefetch_depth` to be used by the
        :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue`
        that is created internally by this function. See the
        :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue`
        documentation.
    Returns:
      Scalar test loss (if the model has a single output and no metrics) or list
      of scalars (if the model has multiple outputs and/or metrics). The
      attribute model.metrics_names will give you the display labels for the
      scalar outputs.
    Raises:
      ValueError: if there are invalid arguments.
    """
    return super().evaluate(x,
                            y,
                            batch_size=batch_size,
                            verbose=verbose,
                            steps=steps,
                            callbacks=callbacks,
                            steps_per_run=steps_per_run,
                            prefetch_depth=prefetch_depth,
                            **kwargs)

  # pylint: disable=arguments-differ
  def predict(self,
              x,
              *,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              steps_per_run=None,
              prefetch_depth=None,
              **kwargs):  # pylint: disable=useless-super-delegation
    """
    This provides equivalent functionality to the Keras Sequential `predict`
    method.

    Note that `batch_size` here is the number of samples that is processed on
    each replica in each forward pass. This is referred to as the mini-batch
    size. Prepare Dataset input on this basis.

    Each step (per replica) will process mini-batch multiplied by gradient
    accumulation count samples. Therefore, the effective batch size is the
    mini-batch size multiplied by the gradient accumulation count multiplied by
    the replication factor.

    Input/Target data of the following types will be converted into a Dataset
    internally based on the batch_size, dropping any partial batch: Numpy array
    (or list of arrays), TensorFlow tensor (or list of tensors) or dict.

    Only the parameters documented below are supported.

    Args:
      x: Input data. It could be:

        - A Numpy array (or array-like), or a list of arrays (in case the model
          has multiple inputs).
        - A TensorFlow tensor, or a list of tensors (in case the model has
          multiple inputs).
        - A dict mapping input names to the corresponding array/tensors, if the
          model has named inputs.
        - A `tf.data` dataset. This must return a tuple of `(inputs, targets)`.
      batch_size: Integer or `None`. The mini-batch size to use for input data
        supplied as Numpy array(s) or TensorFlow tensor(s). If `x` is a dataset
        then `batch_size` must not be specified.
      verbose: Verbosity mode, 0 or 1. **HAS NO EFFECT**. This will be corrected
        in a future release.
      steps: Integer or `None`. Specifies the total number of steps to be
        performed.
        The following constraints apply:

        - If `steps_per_run` is specified then the value for `steps`
          must be evenly divisible by `steps_per_run` multiplied by the
          replication factor. Otherwise it must be divisible by the
          replication factor.
        - For an infinitely repeating dataset a value for `steps`
          must be specified.
        - For a finite dataset if `steps` is specified then it must contain at
          least mini-batch size * gradient accumulation count * `steps` samples.
          For a dataset of known finite length a value for `steps`
          will be calculated if no value is specified. The number of
          samples in the dataset must be a multiple of the mini-batch size
          multiplied by the gradient accumulation count multiplied by the
          replication factor (multiplied by `steps_per_run` if it is specified).
        - For array or tensor inputs a value for `steps` will be calculated
          if no value is specified. If the number of samples provided is not a
          multiple of the mini-batch size multiplied by the gradient
          accumulation count multiplied by the replication factor (multiplied by
          `steps_per_run` if it is specified) then samples will be dropped when
          deriving a value for `steps` and a warning will be logged.
      callbacks: List of keras.callbacks.Callback instances. List of callbacks
        to apply during evaluation. **KNOWN ISSUE**: `predict` currently
        calls the callback functions applicable to `fit` rather than those
        applicable to `predict`. This will be corrected in a future release.
      steps_per_run: Integer or `None`. Specifies how many steps will be
        performed per replica on each hardware execution.
        If not specified this will be set to `steps` (which will be calculated
        if not specified) divided by the replication factor.
        The value of `steps` (if specified) must be evenly divisible by
        `steps_per_run` multiplied by the replication factor.
      prefetch_depth: Integer or `None`. The `prefetch_depth` to be used by the
        :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue`
        that is created internally by this function. See the
        :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue`
        documentation.
    Returns:
      Numpy array(s) of predictions.
    Raises:
      ValueError: if there are invalid arguments.
    """
    return super().predict(x,
                           batch_size=batch_size,
                           verbose=verbose,
                           steps=steps,
                           callbacks=callbacks,
                           steps_per_run=steps_per_run,
                           prefetch_depth=prefetch_depth,
                           **kwargs)

  def save(self,
           filepath,
           overwrite=True,
           include_optimizer=True,
           save_format=None,
           signatures=None,
           options=None):
    """ IPU Keras models do not support the `save` interface.
    """
    raise NotImplementedError(
        "IPU Keras models do not support the `save` interface.")


class IPUModel(_IpuModelBase):
  """A Keras Model class specifically targeting the IPU.  This is
  similar to the Keras Model class, but it also supports the accumulation of
  gradient deltas, and an on-device training/inference loop.

  There are some limitations with the IPU Model class compared to the standard
  Keras Model class:

  - Keras V1 optimizers cannot be used.
  - Loss weightings can only be specified as a list, not a callable.
  - Weighted metrics, target tensors and sample weight mode are not supported.
  - Validation cannot be performed as part of the `fit` loop.
  - The model cannot be called using the __call__() interface.
  - The model cannot be saved using the `save` interface.

  Example:

  .. code-block:: python

    dataset = ...

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      inputs = keras.Input(shape=(784,))

      # Add some more vertices to the graph.
      x = keras.layers.Dense(64, activation="relu")(inputs)
      x = keras.layers.Dense(64, activation="relu")(x)
      x = keras.layers.Dense(10)(x)

      model = ipu.keras.Model(inputs=inputs, outputs=x)
      model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"])

      model.fit(dataset, epochs=2, steps_per_epoch=128)

  """
  def __init__(self,
               *args,
               gradient_accumulation_count=1,
               gradient_accumulation_dtype=None,
               layer_replacement=False,
               accumulation_count=1,
               accumulation_dtype=None,
               **kwargs):
    """
    Creates a Keras model, optimized to run on the IPU.

    ``inputs`` and ``outputs`` must be passed in as either arguments or keyword
    arguments.

    Args:
        gradient_accumulation_count: The number of mini-batches to process
            while accumulating their gradients, before running a
            parameter/weight update step.
        gradient_accumulation_dtype: The data type used for the gradient
          accumulation buffer. One of:

          - `None`: Use an accumulator of the same type as the variable type.
          - A `DType`: Use this type for all the accumulators.
          - A callable that takes the variable and returns a `DType`: Allows
            specifying the accumulator type on a per-variable basis.

          The gradients passed to `Optimizer.apply_gradients` will have the
          dtype requested here. If that dtype is different from the variable
          dtype a cast is needed at some point to make them compatible. This can
          be done by using a custom optimizer.
        layer_replacement: If enabled (True), Keras layers will be substituted
          with IPU Keras implementations, when possible.
    """

    super().__init__(gradient_accumulation_count=gradient_accumulation_count,
                     shard_count=1,
                     layer_replacement=layer_replacement,
                     **kwargs)

    self.gradient_accumulation_dtype = gradient_accumulation_dtype

    # Signature detection
    if len(args) == 2 - sum(['inputs' in kwargs, 'outputs' in kwargs]):
      self._init_network(*args, **kwargs)
    else:
      raise ValueError("Model was not provided with 'inputs' and 'outputs'")

    # Substitute layers for IPU equivalents if enabled.
    if self._layer_replacer:
      for layer in self._layers:
        layer = self._layer_replacer(layer)

    self._num_inputs = len(self.inputs)
    self._num_outputs = len(self.outputs)

  @trackable.no_automatic_dependency_tracking
  def _init_network(self, inputs, outputs, name=None, **kwargs):
    generic_utils.validate_kwargs(
        kwargs, {"trainable"},
        "Functional models may only specify `name` and `trainable` keyword"
        " arguments during initialization. Got an unexpected argument:")
    # Normalize and set self.inputs, self.outputs.
    if isinstance(inputs, list) and len(nest.flatten(inputs)) == 1:
      inputs = inputs[0]
    if isinstance(outputs, list) and len(nest.flatten(outputs)) == 1:
      outputs = outputs[0]
    self._nested_outputs = outputs
    self._nested_inputs = inputs
    self.inputs = nest.flatten(inputs)
    self.outputs = nest.flatten(outputs)

    if not all(hasattr(tensor, '_keras_history') for tensor in self.outputs):
      base_layer_utils.create_keras_history(self._nested_outputs)

    self._base_init(name=name, **kwargs)
    self._validate_graph_inputs_and_outputs()

    self._input_layers = []
    self._output_layers = []

    # Store the output coordinates to map a node in the graph to an output.
    self._output_coordinates = []

    # Build self._output_layers:
    for x in self.outputs:
      layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
      self._output_layers.append(layer)
      self._output_coordinates.append((layer, node_index, tensor_index))

    # Build self._input_layers:
    for x in self.inputs:
      layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
      # It's supposed to be an input layer, so only one node
      # and one tensor output.
      assert node_index == 0
      assert tensor_index == 0
      self._input_layers.append(layer)

    # A Model does not create weights of its own, thus it is already built.
    self.built = True
    self._is_graph_network = True

    # Keep track of the network's nodes and layers.
    # TODO(T31437): Fix up Keras API.
    # nodes, nodes_by_depth, layers, _ = network._map_graph_network(  # pylint: disable=protected-access
    #     self.inputs, self.outputs)
    nodes, nodes_by_depth, layers = (None, None, None)
    assert False

    self._network_nodes = nodes
    self._nodes_by_depth = nodes_by_depth
    self._layers = layers
    self._layer_call_argspecs = {}
    for layer in self._layers:
      self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)
      layer._attribute_sentinel.add_parent(self._attribute_sentinel)  # pylint: disable=protected-access

    self._track_layers(self._layers)

    # Create the node linking internal inputs to internal outputs.
    node_module.Node(outbound_layer=self,
                     inbound_layers=[],
                     node_indices=[],
                     tensor_indices=[],
                     input_tensors=self._nested_inputs,
                     output_tensors=self._nested_outputs)

    self.output_names = []
    self.input_names = [layer.name for layer in self._input_layers]
    self._set_output_names()

    self._create_post_order()

  def _create_post_order(self):
    self._post_order_node_execution = []
    # Set of reference tensors which were computed.
    computed_set = set()

    # Execute input layers first.
    for op, layer in zip(self.inputs, self._input_layers):
      assert len(layer.inbound_nodes) == 1
      self._post_order_node_execution.append(layer.inbound_nodes[0])
      computed_set.add(str(id(op)))

    depth_keys = list(self._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    # Remove the input layers as they have already been computed.
    depth_keys = depth_keys[1:]

    for depth in depth_keys:
      nodes = self._nodes_by_depth[depth]

      for node in nodes:
        # Check all the node inputs have been executed.
        if all(
            str(id(tensor)) in computed_set
            for tensor in nest.flatten(node.input_tensors)):
          if not node in self._post_order_node_execution:
            self._post_order_node_execution.append(node)
            # Update computed_set.
            computed_set.update(
                [str(id(x)) for x in nest.flatten(node.output_tensors)])
    assert len(self._post_order_node_execution) == len(self._network_nodes)

  def _execute_layer_node(self, node, training, tensor_dict):
    convert_kwargs_to_constants = True
    # This is always a single layer, never a list.
    layer = node.outbound_layer

    # Call layer (reapplying ops to new inputs).
    computed_tensors = nest.map_structure(lambda t: tensor_dict[str(id(t))],
                                          node.input_tensors)

    # Ensure `training` arg propagation if applicable.
    kwargs = copy.copy(node.arguments) if node.arguments else {}
    if convert_kwargs_to_constants:
      # TODO(T31437): Fix up Keras API.
      # kwargs = network._map_tensors_to_constants(kwargs)  # pylint: disable=protected-access
      kwargs = None
      assert False

    argspec = self._layer_call_argspecs[layer].args
    if 'training' in argspec:
      kwargs.setdefault('training', training)
      if (isinstance(kwargs['training'], ops.Tensor)
          and kwargs['training'] in K._GRAPH_LEARNING_PHASES.values()):  # pylint: disable=protected-access
        kwargs['training'] = training  # Materialize placeholder.

    # Map Keras tensors in kwargs to their computed value.
    def _map_tensor_if_from_keras_layer(t):
      if isinstance(t, ops.Tensor) and hasattr(t, '_keras_history'):
        t_id = str(id(t))
        return tensor_dict[t_id]
      return t

    kwargs = nest.map_structure(_map_tensor_if_from_keras_layer, kwargs)

    # Compute outputs.
    output_tensors = layer(computed_tensors, **kwargs)

    # Update tensor_dict.
    for x, y in zip(nest.flatten(node.output_tensors),
                    nest.flatten(output_tensors)):
      tensor_dict[str(id(x))] = y

  def _get_output_tensors(self, tensor_dict):
    output_tensors = []
    for output_layer, node_index, tensor_index in self._output_coordinates:
      # Map the output node tensor to an output in the graph.
      output = nest.flatten(
          output_layer.get_output_at(node_index))[tensor_index]

      assert str(
          id(output)) in tensor_dict, "Could not compute output " + str(output)
      tensor = tensor_dict[str(id(output))]
      output_tensors.append(tensor)

    output_tensors = nest.pack_sequence_as(self._nested_outputs,
                                           output_tensors)
    return output_tensors

  # pylint: disable=arguments-differ
  def _add_loss(self, outputs, targets):
    assert len(outputs) == len(self.outputs)
    assert len(targets) == len(self.outputs)

    masks = self._prepare_output_masks()

    # Invoke metric functions (unweighted) for all the outputs.
    metrics = self._handle_metrics(
        outputs,
        targets=targets,
        skip_target_masks=self._prepare_skip_target_masks(),
        masks=masks)

    # Loss calculation logic taken from training.py and adapted to take the in
    # pipeline outputs and targets.
    total_loss = None
    output_losses = []
    with K.name_scope('loss'):
      for endpoint, mask, y_true, y_pred in zip(self._training_endpoints,
                                                masks, targets, outputs):
        if endpoint.should_skip_target():
          continue
        loss_fn = endpoint.loss_fn
        loss_weight = endpoint.loss_weight
        loss_name = endpoint.loss_name()
        sample_weight = endpoint.sample_weight

        with K.name_scope(loss_name):
          if mask is not None:
            mask = math_ops.cast(mask, y_pred.dtype)
            # Update weights with mask.
            if sample_weight is None:
              sample_weight = mask
            else:
              # Update dimensions of weights to match with mask if possible.
              mask, _, sample_weight = (
                  tf_losses_utils.squeeze_or_expand_dimensions(
                      mask, sample_weight=sample_weight))
              sample_weight *= mask

          if hasattr(loss_fn, 'reduction'):
            per_sample_losses = loss_fn.call(y_true, y_pred)
            weighted_losses = losses_utils.compute_weighted_loss(
                per_sample_losses,
                sample_weight=sample_weight,
                reduction=losses_utils.ReductionV2.NONE)
            loss_reduction = loss_fn.reduction

            # `AUTO` loss reduction defaults to `SUM_OVER_BATCH_SIZE` for all
            # compile use cases.
            if loss_reduction == losses_utils.ReductionV2.AUTO:
              loss_reduction = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE

            # Compute the stateless loss value.
            output_loss = losses_utils.reduce_weighted_loss(
                weighted_losses, reduction=loss_reduction)
          else:
            # Compute the stateless loss value for a custom loss class.
            # Here we assume that the class takes care of loss reduction
            # because if this class returns a vector value we cannot
            # differentiate between use case where a custom optimizer
            # expects a vector loss value vs unreduced per-sample loss value.
            output_loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
            loss_reduction = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE

        # If the number of outputs is 1 then we don't append the loss metric
        # associated with each model output. When there are multiple outputs
        # associated with a model, each output's loss is calculated and returned
        # as part of the loss_metrics.
        if len(self.outputs) > 1:
          # Keep track of the stateful output loss result.
          output_losses.append(endpoint.output_loss_metric(output_loss))

        # Scale output loss for distribution. For custom losses we assume
        # reduction was mean.
        if loss_reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE:
          output_loss = losses_utils.scale_loss_for_distribution(output_loss)

        if total_loss is None:
          total_loss = loss_weight * output_loss
        else:
          total_loss += loss_weight * output_loss
      if total_loss is None:
        if not self.losses:
          raise ValueError('The model cannot be compiled '
                           'because it has no loss to optimize.')
        else:
          total_loss = 0.

      # Add regularization penalties and other layer-specific losses.
      custom_losses = self.get_losses_for(None) + self.get_losses_for(
          self.inputs)
      if custom_losses:
        raise ValueError('Custom layer losses are not supported.')
    self.total_loss = total_loss

    losses_and_metrics = [self.total_loss] + output_losses + metrics
    return losses_and_metrics

  def _internal_run_loop(self,
                         infeed_queue,
                         outfeed_queue,
                         repeat_count,
                         mode,
                         run_loop_kwargs=None):
    training = mode == ModeKeys.TRAIN

    def main_body(inputs):
      inputs = nest.flatten(inputs)
      assert len(inputs) == len(self.inputs)

      # Dictionary mapping reference tensors to computed tensors.
      tensor_dict = {}

      # "Execute" the input layers
      for op, layer, tensor in zip(self.inputs, self._input_layers, inputs):
        tensor_dict[str(id(op))] = layer(tensor)
        if isinstance(op, ops.Tensor) and isinstance(tensor, ops.Tensor):
          try:
            tensor.set_shape(tensor.shape.merge_with(op.shape))
          except ValueError:
            logging.warning(
                'Model was constructed with shape {} for input {}, but it was '
                're-called on a Tensor with incompatible shape {}.'.format(
                    op, op.shape, tensor.shape))

      # Execute the remaining nodes.
      for node in self._post_order_node_execution[len(inputs):]:
        self._execute_layer_node(node, training, tensor_dict)

      output_tensors = self._get_output_tensors(tensor_dict)
      return output_tensors

    def inference_body(*args):
      x = main_body(args)
      if isinstance(x, (list, tuple)):
        outfeed_queue.enqueue(x)
      else:
        outfeed_queue.enqueue([x])
      return []

    def training_body(*args):
      n_inputs = len(self.inputs)
      inputs = list(args[:n_inputs])
      targets = list(args[n_inputs:])

      outputs = main_body(inputs)
      losses_and_metrics = self._add_loss(nest.flatten(outputs), targets)
      outfeed_queue.enqueue(losses_and_metrics)

      if not self.trainable_weights:
        raise ValueError("Model must have at least one trainable parameter.")

      if training:
        opt = self._get_wrapped_optimizer()
        if self.gradient_accumulation_count > 1:
          opt = gradient_accumulation_optimizer.GradientAccumulationOptimizerV2(
              opt,
              self.gradient_accumulation_count,
              dtype=self.gradient_accumulation_dtype)

        grads_and_vars = opt.compute_gradients(losses_and_metrics[0],
                                               self.trainable_variables)
        opt.apply_gradients(grads_and_vars)

      return []

    def body(*args, **kwargs):
      # Flatten all the arguments.
      args = nest.flatten(args) + nest.flatten(kwargs)
      fn = functional_ops.outlined_function(
          inference_body if mode == ModeKeys.PREDICT else training_body)
      return fn(*args)

    result = loops.repeat(int(repeat_count * self.gradient_accumulation_count),
                          body,
                          infeed_queue=infeed_queue)

    return result.outputs

  def build(self, input_shape):
    """Builds the model based on input shapes received.

    Args:
     input_shape: Single tuple, TensorShape, or list of shapes, where shapes
         are tuples, integers, or TensorShapes.
    """

    # A Model does not create weights of its own, thus it is already built.
    assert self._is_graph_network
    self.built = True
    return

  @trackable.no_automatic_dependency_tracking
  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              **kwargs):
    """
    This provides the same functionality as the Keras Model ``compile`` method.

    Certain features are not supported by the IPU Model:

      - sample_weight_mode
      - weighted_metrics
      - target_tensors

    Note that loss weights can only be specified as a list.

    Args:
      optimizer: String (name of optimizer) or optimizer instance. See
        `tf.keras.optimizers`. An instance of a subclass of
        `tensorflow.python.training.optimizer` can also be used.
      loss: String (name of objective function), objective function or
        `tf.keras.losses.Loss` instance. See `tf.keras.losses`.
        IPU-specific loss classes can also be used. See the documentation in
        :py:mod:`tensorflow.python.ipu.keras.losses` for usage instructions.
        An objective function is any callable with the signature
        `scalar_loss = fn(y_true, y_pred)`. If the model has multiple outputs,
        you can use a different loss on each output by passing a dictionary or
        a list of losses. The loss value that will be minimized by the model
        will then be the sum of all individual losses.
      metrics: List of metrics to be evaluated by the model during training and
        testing. Typically you will use `metrics=['accuracy']`. To specify
        different metrics for different outputs of a multi-output model, you
        could pass a dictionary, such as `metrics={'output_a': 'accuracy',
        'output_b': ['accuracy', 'mse']}`, or a list (`len = len(outputs)`) of
        lists of metrics such as
        `metrics=[['accuracy'], ['accuracy', 'mse']]` or `metrics=['accuracy',
        ['accuracy', 'mse']]`.
      loss_weights: Optional list specifying scalar coefficients (Python floats)
        to weight the loss contributions of different model outputs. The loss
        value that will be minimized by the model will then be the weighted sum
        of all individual losses, weighted by the loss_weights coefficients.
        The list is expected to have a 1:1 mapping to the model's outputs.
    Raises:
      ValueError: if there are invalid arguments.
    """
    return super().compile(optimizer, loss, metrics, loss_weights, **kwargs)

  # pylint: disable=arguments-differ
  @trackable.no_automatic_dependency_tracking
  def fit(self,
          x=None,
          y=None,
          *,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          shuffle=True,
          initial_epoch=0,
          steps_per_epoch=None,
          steps_per_run=None,
          prefetch_depth=None,
          **kwargs):  # pylint: disable=useless-super-delegation
    """
    This provides equivalent functionality to the Keras Model `fit` method.

    Note that `batch_size` here is the number of samples that is processed on
    each replica in each forward pass. This is referred to as the mini-batch
    size. Prepare Dataset input on this basis.

    Each step (per replica) will process mini-batch multiplied by gradient
    accumulation count samples before updating the weights. Therefore, the
    effective batch size for a weight update is the mini-batch size multiplied
    by the gradient accumulation count multiplied by the replication factor.

    The number of weight update steps per epoch is the `steps_per_epoch` value
    divided by the replication factor, and this is the number of steps that
    will be shown in the progress bar.

    For a finite dataset the iterator over the data will be reset at the start
    of each epoch. This means that the dataset does not need to be repeated
    `epochs` times if `steps_per_epoch` is not specified. It also means that if
    a small value for `steps_per_epoch` is supplied then not all samples will be
    used.

    A shuffled Dataset should be supplied. Non-dataset inputs (as described in
    the parameters section below) for `x` and `y` will be accepted but will not
    be shuffled, and this may lead to over-fitting.

    Input/Target data of the following types will be converted into a Dataset
    internally based on the batch_size, dropping any partial batch: Numpy array
    (or list of arrays), TensorFlow tensor (or list of tensors) or dict.

    Only the parameters documented below are supported.

    Args:
      x: Input data.
        It could be:

        - A Numpy array (or array-like), or a list of arrays (in case the model
          has multiple inputs).
        - A TensorFlow tensor, or a list of tensors (in case the model has
          multiple inputs).
        - A dict mapping input names to the corresponding array/tensors, if the
          model has named inputs.
        - A `tf.data` dataset. This must return a tuple of `(inputs, targets)`.
      y: Target data. Like the input data `x`, it could be either Numpy array(s)
        or TensorFlow tensor(s). It should be consistent with `x` (you cannot
        have Numpy inputs and tensor targets, or tensor inputs and Numpy
        targets). If `x` is a dataset then `y` must not be specified (since
        targets will be obtained from `x`).
      batch_size: Integer or `None`. The mini-batch size to use for input data
        supplied as Numpy array(s) or TensorFlow tensor(s). If `x` is a dataset
        then `batch_size` must not be specified.
      epochs: Integer. Number of epochs to train the model. The number of steps
        performed per epoch is defined by the `steps_per_epoch` parameter, or
        calculated according to the constraints described below.
        Note that in conjunction with `initial_epoch`, `epochs` is to be
        understood as "final epoch". The model is not trained for a number of
        iterations given by `epochs`, but merely until the epoch of index
        `epochs` is reached.
      verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar,
        2 = one line per epoch. Note that the progress bar is not particularly
        useful when logged to a file, so verbose=2 is recommended when not
        running interactively (for example, in a production environment).
      callbacks: List of `keras.callbacks.Callback` instances. List of callbacks
        to apply during training. See `tf.keras.callbacks` in the TensorFlow
        documentation.
      shuffle: **NOT SUPPORTED**. This will be supported in a future release.
      initial_epoch: Integer. Epoch at which to start training (useful for
        resuming a previous training run).
      steps_per_epoch: Integer or `None`. Specifies the total number of steps to
        be performed per epoch.
        The following constraints apply:

        - If `steps_per_run` is specified then the value for `steps_per_epoch`
          must be evenly divisible by `steps_per_run` multiplied by the
          replication factor. Otherwise it must be divisible by the
          replication factor.
        - For an infinitely repeating dataset a value for `steps_per_epoch`
          must be specified.
        - For a finite dataset if `steps_per_epoch` is specified then it must
          contain at least mini-batch size * gradient accumulation count *
          `steps` samples.
        - For a dataset of known finite length a value for `steps_per_epoch`
          will be calculated if no value is specified. The number of
          samples in the dataset must be a multiple of the mini-batch size
          multiplied by the gradient accumulation count multiplied by the
          replication factor (multiplied by `steps_per_run` if it is
          specified).
        - For array or tensor inputs a value for `steps_per_epoch` will be
          calculated if no value is specified. If the number of samples provided
          is not a multiple of the mini-batch size multiplied by the gradient
          accumulation count multiplied by the replication factor (multiplied by
          `steps_per_run` if it is specified) then samples will be dropped when
          deriving a value for `steps_per_epoch` and a warning will be logged.
      steps_per_run: Integer or `None`. Specifies how many steps will be
        performed per replica on each hardware execution.
        If not specified this will be set to `steps_per_epoch` (which will
        be calculated if not specified) divided by the replication factor.
        The value of 'steps_per_epoch' (if specified) must be evenly
        divisible by `steps_per_run` multiplied by the replication factor.
      prefetch_depth: Integer or `None`. The `prefetch_depth` to be used by the
        :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue`
        that is created internally by this function. See the
        :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue`
        documentation.
    Returns:
      A `History` object. Its `History.history` attribute is a record of
      training loss values and metrics values at successive epochs.
    Raises:
      ValueError: if there are invalid arguments.
    """
    return super().fit(x,
                       y,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       callbacks=callbacks,
                       shuffle=shuffle,
                       initial_epoch=initial_epoch,
                       steps_per_epoch=steps_per_epoch,
                       steps_per_run=steps_per_run,
                       prefetch_depth=prefetch_depth,
                       **kwargs)

  # pylint: disable=arguments-differ
  def evaluate(self,
               x=None,
               y=None,
               *,
               batch_size=None,
               verbose=1,
               steps=None,
               callbacks=None,
               steps_per_run=None,
               prefetch_depth=None,
               **kwargs):  # pylint: disable=useless-super-delegation
    """
    This provides equivalent functionality to the Keras Model `evaluate`
    method.

    Note that `batch_size` here is the number of samples that is processed on
    each replica in each forward pass. This is referred to as the mini-batch
    size. Prepare Dataset input on this basis.

    Each step (per replica) will process mini-batch multiplied by gradient
    accumulation count samples. Therefore, the effective batch size is the
    mini-batch size multiplied by the gradient accumulation count multiplied by
    the replication factor.

    Input/Target data of the following types will be converted into a Dataset
    internally based on the batch_size, dropping any partial batch: Numpy array
    (or list of arrays), TensorFlow tensor (or list of tensors) or dict.

    Only the parameters documented below are supported.

    Args:
      x: Input data. It could be:

        - A Numpy array (or array-like), or a list of arrays (in case the model
          has multiple inputs).
        - A TensorFlow tensor, or a list of tensors (in case the model has
          multiple inputs).
        - A dict mapping input names to the corresponding array/tensors, if the
          model has named inputs.
        - A `tf.data` dataset. This must return a tuple of `(inputs, targets)`.
      y: Target data. Like the input data `x`, it could be either Numpy array(s)
        or TensorFlow tensor(s). It should be consistent with `x` (you cannot
        have Numpy inputs and tensor targets, or tensor inputs and Numpy
        targets). If `x` is a dataset then `y` must not be specified (since
        targets will be obtained from `x`).
      batch_size: Integer or `None`. The mini-batch size to use for input data
        supplied as Numpy array(s) or TensorFlow tensor(s). If `x` is a dataset
        then `batch_size` must not be specified.
      verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar. **HAS NO
        EFFECT** - the progress bar is not displayed. This will be corrected in
        a future release.
      steps: Integer or `None`. Specifies the total number of steps to be
        performed. The following constraints apply:

        - If `steps_per_run` is specified then the value for `steps`
          must be evenly divisible by `steps_per_run` multiplied by the
          replication factor. Otherwise it must be divisible by the
          replication factor.
        - For an infinitely repeating dataset a value for `steps`
          must be specified.
        - For a finite dataset if `steps` is specified then it must contain at
          least mini-batch size * gradient accumulation count * `steps` samples.
          For a dataset of known finite length a value for `steps`
          will be calculated if no value is specified. The number of
          samples in the dataset must be a multiple of the mini-batch size
          multiplied by the gradient accumulation count multiplied by the
          replication factor (multiplied by `steps_per_run` if it is specified).
        - For array or tensor inputs a value for `steps` will be calculated
          if no value is specified. If the number of samples provided is not a
          multiple of the mini-batch size multiplied by the gradient
          accumulation count multiplied by the replication factor (multiplied by
          `steps_per_run` if it is specified) then samples will be dropped when
          deriving a value for `steps` and a warning will be logged.
      callbacks: List of keras.callbacks.Callback instances. List of callbacks
        to apply during evaluation. **KNOWN ISSUE**: `evaluate` currently
        calls the callback functions applicable to `fit` rather than those
        applicable to `evaluate`. This will be corrected in a future release.
      steps_per_run: Integer or `None`. Specifies how many steps will be
        performed per replica on each hardware execution.
        If not specified this will be set to `steps` (which will be calculated
        if not specified) divided by the replication factor.
        The value of `steps` (if specified) must be evenly divisible by
        `steps_per_run` multiplied by the replication factor.
      prefetch_depth: Integer or `None`. The `prefetch_depth` to be used by the
        :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue`
        that is created internally by this function. See the
        :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue`
        documentation.
    Returns:
      Scalar test loss (if the model has a single output and no metrics) or list
      of scalars (if the model has multiple outputs and/or metrics). The
      attribute model.metrics_names will give you the display labels for the
      scalar outputs.
    Raises:
      ValueError: if there are invalid arguments.
    """
    return super().evaluate(x,
                            y,
                            batch_size=batch_size,
                            verbose=verbose,
                            steps=steps,
                            callbacks=callbacks,
                            steps_per_run=steps_per_run,
                            prefetch_depth=prefetch_depth,
                            **kwargs)

  # pylint: disable=arguments-differ
  def predict(self,
              x,
              *,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              steps_per_run=None,
              prefetch_depth=None,
              **kwargs):  # pylint: disable=useless-super-delegation
    """
    This provides equivalent functionality to the Keras Model `predict` method.

    Note that `batch_size` here is the number of samples that is processed on
    each replica in each forward pass. This is referred to as the mini-batch
    size. Prepare Dataset input on this basis.

    Each step (per replica) will process mini-batch multiplied by gradient
    accumulation count samples. Therefore, the effective batch size is the
    mini-batch size multiplied by the gradient accumulation count multiplied by
    the replication factor.

    Input/Target data of the following types will be converted into a Dataset
    internally based on the batch_size, dropping any partial batch: Numpy array
    (or list of arrays), TensorFlow tensor (or list of tensors) or dict.

    Only the parameters documented below are supported.

    Args:
      x: Input data. It could be:

        - A Numpy array (or array-like), or a list of arrays (in case the model
          has multiple inputs).
        - A TensorFlow tensor, or a list of tensors (in case the model has
          multiple inputs).
        - A dict mapping input names to the corresponding array/tensors, if the
          model has named inputs.
        - A `tf.data` dataset. This must return a tuple of `(inputs, targets)`.
      batch_size: Integer or `None`. The mini-batch size to use for input data
        supplied as Numpy array(s) or TensorFlow tensor(s). If `x` is a dataset
        then `batch_size` must not be specified.
      verbose: Verbosity mode, 0 or 1. **HAS NO EFFECT**. This will be corrected
        in a future release.
      steps: Integer or `None`. Specifies the total number of steps to be
        performed.
        The following constraints apply:

        - If `steps_per_run` is specified then the value for `steps`
          must be evenly divisible by `steps_per_run` multiplied by the
          replication factor. Otherwise it must be divisible by the
          replication factor.
        - For an infinitely repeating dataset a value for `steps`
          must be specified.
        - For a finite dataset if `steps` is specified then it must contain at
          least mini-batch size * gradient accumulation count * `steps` samples.
          For a dataset of known finite length a value for `steps`
          will be calculated if no value is specified. The number of
          samples in the dataset must be a multiple of the mini-batch size
          multiplied by the gradient accumulation count multiplied by the
          replication factor (multiplied by `steps_per_run` if it is specified).
        - For array or tensor inputs a value for `steps` will be calculated
          if no value is specified. If the number of samples provided is not a
          multiple of the mini-batch size multiplied by the gradient
          accumulation count multiplied by the replication factor (multiplied by
          `steps_per_run` if it is specified) then samples will be dropped when
          deriving a value for `steps` and a warning will be logged.
      callbacks: List of keras.callbacks.Callback instances. List of callbacks
        to apply during evaluation. **KNOWN ISSUE**: `predict` currently
        calls the callback functions applicable to `fit` rather than those
        applicable to `predict`. This will be corrected in a future release.
      steps_per_run: Integer or `None`. Specifies how many steps will be
        performed per replica on each hardware execution.
        If not specified this will be set to `steps` (which will be calculated
        if not specified) divided by the replication factor.
        The value of `steps` (if specified) must be evenly divisible by
        `steps_per_run` multiplied by the replication factor.
      prefetch_depth: Integer or `None`. The `prefetch_depth` to be used by the
        :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue`
        that is created internally by this function. See the
        :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue`
        documentation.
    Returns:
      Numpy array(s) of predictions.
    Raises:
      ValueError: if there are invalid arguments.
    """
    return super().predict(x,
                           batch_size=batch_size,
                           verbose=verbose,
                           steps=steps,
                           callbacks=callbacks,
                           steps_per_run=steps_per_run,
                           prefetch_depth=prefetch_depth,
                           **kwargs)

  def save(self,
           filepath,
           overwrite=True,
           include_optimizer=True,
           save_format=None,
           signatures=None,
           options=None):
    """ IPU Keras models do not support the `save` interface.
    """
    raise NotImplementedError(
        "IPU Keras models do not support the `save` interface.")


Model = IPUModel
Sequential = IPUSequential
