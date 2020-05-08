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
import weakref

from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras import backend
from tensorflow.python.keras import Model as KerasModel
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.training.tracking import base as trackable


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
    raise ValueError("Do not specify `batch_size` in IPU Keras models. "
                     "Use the DataSet.batch() method to apply batching "
                     "at the input dataset level.")

  bad_args = list(filter(lambda x: x in kwargs, blacklist))
  if bad_args:
    raise NotImplementedError(
        "IPU Keras models do not support these parameters to " + fn + "(): " +
        ", ".join(bad_args))


class _TensorflowOptimizerWrapper(Optimizer):
  """A class which wraps a standard Tensorflow optimizer, giving it a TF
  optimizer interface, but generating gradients against the Keras Model.
  """
  def __init__(self, model, opt):
    super(_TensorflowOptimizerWrapper, self).__init__(use_locking=False,
                                                      name="optimizer_shim")
    self._model = model
    self._optimizer = opt

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    return self._optimizer.compute_gradients(loss,
                                             self._model.trainable_weights)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    return self._optimizer.apply_gradients(grads_and_vars, global_step, name)

  def _apply_sparse(self, grad, var):
    raise NotImplementedError()

  def _apply_dense(self, grad, var):
    raise NotImplementedError()

  def _resource_apply_dense(self, grad, handle):
    raise NotImplementedError()

  def _resource_apply_sparse(self, grad, handle, indices):
    raise NotImplementedError()


class _KerasOptimizerWrapper(Optimizer):
  """A class which wraps a Keras optimizer, giving it ia TF optimizer interface.
  """
  def __init__(self, model, opt):
    super(_KerasOptimizerWrapper, self).__init__(use_locking=False,
                                                 name="optimizer_shim")
    self._model = model
    self._optimizer = opt

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    grads = self._optimizer.get_gradients(loss, self._model.trainable_weights)
    return zip(grads, self._model.trainable_weights)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    return self._optimizer.apply_gradients(grads_and_vars)

  def _apply_sparse(self, grad, var):
    raise NotImplementedError()

  def _apply_dense(self, grad, var):
    raise NotImplementedError()

  def _resource_apply_dense(self, grad, handle):
    raise NotImplementedError()

  def _resource_apply_sparse(self, grad, handle, indices):
    raise NotImplementedError()


class _IpuModelBase(KerasModel):
  """Base class for IPU Keras models"""
  def __init__(self, accumulation_count, **kwargs):
    name = kwargs.pop("name", None)
    super(_IpuModelBase, self).__init__(dtype=None, name=name)

    self.args = kwargs
    self.accumulation_count = accumulation_count

    self.built = False
    self.history = None
    self.infeed = None
    self.outfeed = None
    self.last_ds = None
    self.last_mode = None
    self.internal_loop_fn = None

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
          "Optimizer must be a native Tensorflow optimizers, or Keras V2 "
          "optimizers, found in tensorflow.keras.optimizer_v2.")

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

    # Prepare list of loss functions, same size of model outputs.
    self.loss_functions = training_utils.prepare_loss_functions(
        self.loss, self.output_names)

    self._training_endpoints = []
    for o, n, l, t in zip(self.outputs, self.output_names, self.loss_functions,
                          target_tensors):
      endpoint = training._TrainingEndpoint(o, n, l)  # pylint: disable=protected-access
      endpoint.create_training_target(t, run_eagerly=self.run_eagerly)
      self._training_endpoints.append(endpoint)

    # Prepare list loss weights, same size of model outputs.
    training_utils.prepare_loss_weights(self._training_endpoints,
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
    training_utils.prepare_sample_weight_modes(self._training_endpoints, None)

    # Creates the model loss
    self.total_loss = self._prepare_total_loss(masks)

    return [self.total_loss] + metrics

  def _get_internal_run_loop(self):
    raise NotImplementedError(
        "_IpuModelBase should not be used directly.  Use PipelinedModel or "
        "Model instead.")

  def _internal_run_loop(self, infeed_queue, outfeed_queue, repeat_count,
                         mode):
    raise NotImplementedError(
        "_IpuModelBase should not be used directly.  Use PipelinedModel or "
        "Model instead.")

  def _get_optimizer(self):
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
          "Only Keras optimizer_v2.Optimizer and Tensorflow native "
          "training.Optimizer subclasses are supported.")

  @trackable.no_automatic_dependency_tracking
  def _do_internal(self, mode, ds, epochs, verbose, callbacks, initial_epoch,
                   steps_per_epoch, steps_per_run, **kwargs):

    self.args = kwargs

    # Figure out if we need to recreate the iterator after each epoch.
    recreate_iterator = False
    require_steps_per_epoch = False
    verify_dataset_length = False

    dataset_length = backend.get_value(cardinality.cardinality(ds))
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

    if require_steps_per_epoch:
      if not steps_per_epoch:
        raise ValueError(
            "When using an infinitely repeating dataset, you must provide "
            "the number of steps per epoch (steps_per_epoch).")

    # Find out how many mini-batches, steps, repeats, and outer loops.
    mini_batches_per_epoch = steps_per_epoch
    if mini_batches_per_epoch is not None:
      mini_batches_per_epoch = mini_batches_per_epoch * self.accumulation_count

    # If there is a fixed length of dataset, and the user has also specified
    # a steps_per_epoch, then check that this won't exhaust the dataset.
    if verify_dataset_length and steps_per_epoch:
      if mini_batches_per_epoch > dataset_length:
        raise ValueError(
            "Steps per epoch times accumulation count (%d x %d) is greater "
            "than the number of samples in the dataset (%d)." %
            (steps_per_epoch, self.accumulation_count, dataset_length))

    mini_batches_per_epoch = training_utils.infer_steps_for_dataset(
        self, ds, mini_batches_per_epoch, epochs, steps_name='steps_per_epoch')

    if mini_batches_per_epoch % self.accumulation_count != 0:
      raise ValueError(
          self.__class__.__name__ + " requires the number of batches in the "
          "dataset (%d) to be a multiple of the accumulated batch size (%d)" %
          (mini_batches_per_epoch, self.accumulation_count))

    steps_per_epoch = mini_batches_per_epoch / self.accumulation_count

    if not steps_per_run:
      steps_per_run = steps_per_epoch

    if steps_per_epoch % steps_per_run != 0:
      raise ValueError(
          self.__class__.__name__ + " requires the number of steps per "
          "execution of the on device training loop 'steps_per_run' (%d) "
          "to be a multiple of the number of steps in the epoch (%d)." %
          (mini_batches_per_epoch, steps_per_epoch))

    outer_loop_count = int(steps_per_epoch / steps_per_run)

    total_samples = mini_batches_per_epoch * (epochs - initial_epoch)

    # Prepare for progress reporting
    callbacks = cbks.configure_callbacks(callbacks,
                                         self,
                                         epochs=epochs,
                                         steps_per_epoch=steps_per_epoch,
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
      self.infeed = ipu_infeed_queue.IPUInfeedQueue(ds, "infeed")
      self.outfeed = ipu_outfeed_queue.IPUOutfeedQueue("outfeed")

    initial_epoch = self._maybe_load_initial_epoch_from_ckpt(
        initial_epoch, mode)

    callbacks.on_train_begin(mode)

    # Ask the poplar executor to create a dataset iterator
    self.infeed.initializer  # pylint: disable=pointless-statement

    # Aggregator for combining the various outputs/metrics together
    if mode != ModeKeys.PREDICT:
      aggregator = training_utils.MetricsAggregator(use_steps=False,
                                                    num_samples=total_samples)
    else:
      aggregator = training_utils.OutputsAggregator(use_steps=False,
                                                    num_samples=total_samples)

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
          func = self._get_internal_run_loop()
          strategy.experimental_run_v2(
              func, args=[self.infeed, self.outfeed, steps_per_run, mode])

          # Send an end of batches
          callbacks.on_batch_end(batch_num, batch_logs)

          # After the first call we can update the callbacks to include
          # the metrics.
          if epoch == initial_epoch and run == 0:
            cbks.set_callback_parameters(callbacks,
                                         self,
                                         epochs=epochs,
                                         steps_per_epoch=steps_per_epoch,
                                         verbose=verbose,
                                         mode=mode)

        # Restart the iterator at the end of the epoch if necessary
        if recreate_iterator:
          self.infeed.deleter  # pylint: disable=pointless-statement
          self.infeed.initializer  # pylint: disable=pointless-statement

        # Fetch the outfeed for the history
        results = self.outfeed.dequeue()
        results = map(lambda x: x.numpy(), results)
        results = enumerate(zip(*results))

        # Get the final loss and metrics
        i, r = next(results)
        aggregator.create(r)
        aggregator.aggregate(r, 0, 1)

        for i, r in results:
          aggregator.aggregate(r, i, i + 1)

        aggregator.finalize()
        results = aggregator.results

        # Store only the final losses/metrics for the epoch log
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
  def fit(self, x, *, epochs, verbose, callbacks, shuffle, initial_epoch,
          steps_per_epoch, steps_per_run, **kwargs):
    if not isinstance(x, dataset_ops.DatasetV2):
      raise ValueError(self.__class__.__name__ +
                       " can only `fit` with a `tf.data.Dataset` "
                       "as input.")

    if 'y' in kwargs:
      raise ValueError(
          "Labels should be provided by the 'x' DataSet containing a tuple.")

    if 'batch_size' in kwargs:
      raise ValueError("Do not specify `batch_size` in " +
                       self.__class__.__name__ + ".fit(). Use the "
                       "DataSet.batch() method to apply batching at the input "
                       "dataset level.")

    _validate_args(kwargs, "fit")

    structure = dataset_ops.get_structure(x)
    if not isinstance(structure, tuple) or len(structure) != 2:
      raise ValueError(
          self.__class__.__name__ + ".fit requires a dataset containing a "
          "tuple of two elements, the data value and the target value.")

    self._assert_compile_was_called()

    return self._do_internal(ModeKeys.TRAIN, x, epochs, verbose, callbacks,
                             initial_epoch, steps_per_epoch, steps_per_run,
                             **kwargs)

  def evaluate(self,
               x=None,
               *,
               verbose=1,
               steps=None,
               callbacks=None,
               steps_per_run=None,
               **kwargs):
    if not isinstance(x, dataset_ops.DatasetV2):
      raise ValueError(self.__class__.__name__ + " can only `evaluate` with a "
                       "`tf.data.Dataset` as input.")

    _validate_args(kwargs, "evaluate")

    structure = dataset_ops.get_structure(x)
    if not isinstance(structure, tuple) or len(structure) != 2:
      raise ValueError(
          self.__class__.__name__ + ".evaluate requires a dataset containing "
          "a tuple of two elements, the data value and the target value.")

    self._assert_compile_was_called()

    return self._do_internal(ModeKeys.TEST, x, 1, verbose, callbacks, 0, steps,
                             steps_per_run, **kwargs)

  def predict(self,
              x,
              *,
              verbose=0,
              steps=None,
              callbacks=None,
              steps_per_run=None,
              **kwargs):
    if not isinstance(x, dataset_ops.DatasetV2):
      raise ValueError(self.__class__.__name__ + " can only `predict` with a "
                       "`tf.data.Dataset` as input.")

    _validate_args(kwargs, "predict")

    structure = dataset_ops.get_structure(x)
    if not (isinstance(structure, tensor_spec.TensorSpec) or
            (isinstance(structure, tensor_spec.TensorSpec)
             and len(structure) != 1)):
      raise ValueError(
          self.__class__.__name__ + ".predict requires a dataset containing "
          "either a tuple of one single data value, or just a single data "
          "value.")

    return self._do_internal(ModeKeys.PREDICT, x, 1, verbose, callbacks, 0,
                             steps, steps_per_run, **kwargs)


class PipelinedModel(_IpuModelBase):
  """Keras Model for encapsulating a pipeline of stages to be run in parallel
  on an IPU system.

  A pipelined model will execute multiple sections (stages) of a model on more
  than one IPU at the same time, by pipelining mini-batches of data through
  the stages.

  It encapsulates the ipu.pipelining_ops.pipeline operation and the associated
  InFeed and OutFeed queues into a class which resembles the Keras Model class
  and provides the `fit` API for training the model.

  The different stages are specified, similarly to the Keras Sequential model,
  as a list in the constructor.  With the PipelinedModel class the list of
  layers becomes a list of lists of layers, where each list contains the layers
  for a particular stage.

  The pipeline depth argument describes the number of mini-batches which are
  sent through the pipeline in a single operation of the pipeline.  The
  effective batch size is therefore the mini-batch size multipled by the
  pipeline depth.

  There are some limitations with the PipelinedModel compared to the standard
  Keras Model.

  - The input must be provided by a tf.DataSet.
  - Keras V1 optimizers cannot be used.
  - Loss weightings can only be specified as a list, not a callable.
  - Weighted metrics, target tensors and sample weight mode are not supported.
  - Validation cannot be performed as part of the `fit` loop.
  - The model cannot be called using the __call__() interface.

  The model will only be constructed after the first call to the `fit` method,
  so a summary of the model will not be possible until after some training
  has occurred.  Related to this, the `build` method does not build the
  model.

  Example:

  .. code-block:: python

    dataset = ...

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu.keras.PipelinedModel([
        [
          keras.layers.Dense(4),
          keras.layers.Dense(4),
          keras.layers.Dense(4),
        ],
        [
          keras.layers.Dense(8),
        ],
      ], pipeline_depth=24)

      m.compile('sgd', loss='mse')

      m.fit(dataset, steps_per_epoch=144)

  """
  def __init__(self, stages=None, pipeline_depth=None, **kwargs):
    """
    Creates a pipelined model.

    Args:
        stages: A python list of lists of Layers.
        pipeline_depth: The number of mini-batches processed by the
                        pipeline on each iteration.
        name: Optional name for the pipeline operation.

    Other arguments are passed to the pipeline operator, for instance
    device_mapping or pipeline_schedule.
    """
    super(PipelinedModel, self).__init__(pipeline_depth, **kwargs)

    if not isinstance(stages, list):
      raise ValueError("An IPU pipeline must take a list of stages, where "
                       "each stage is a list of Keras Layers.")

    for s in stages:
      if not isinstance(s, list):
        raise ValueError("An IPU pipeline may only contain lists of "
                         "stages, where each stage is a list of Keras Layers.")
      for l in s:
        if not isinstance(l, Layer):
          raise ValueError("Each list in the `stages` list must contain "
                           "only Keras Layers.")

    if not pipeline_depth:
      raise ValueError(
          "The pipeline_depth parameter must be specified.  Choose a "
          "pipeline_depth such that pipeline_depth * mini_batch_size is a "
          "good total batch size.  One step of the model will run "
          "pipeline_depth mini-batches through the model, accumulate the "
          "gradients of the errors, and then apply the accumulated gradients "
          "to the model weights once.")

    self.pipeline_depth = pipeline_depth
    self.stages = stages

  @trackable.no_automatic_dependency_tracking
  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              **kwargs):
    """
    This provides the same functionality as the Keras Model ``compile``
    method.

    Certain features are not supported by the IPU PipelinedModel:
    - sample_weight_mode
    - weighted_metrics
    - target_tensors
    """
    return super(PipelinedModel, self).compile(optimizer, loss, metrics,
                                               loss_weights, **kwargs)

  def _get_internal_run_loop(self):
    if not self.internal_loop_fn:
      fn = partial(PipelinedModel._internal_run_loop, self)
      self.internal_loop_fn = def_function.function(fn,
                                                    autograph=False,
                                                    experimental_compile=True)
    return self.internal_loop_fn

  def _internal_run_loop(self, infeed_queue, outfeed_queue, repeat_count,
                         mode):

    # Plain functions to build a stage
    def call_inference_stage(stage_id, inputs):
      # Record the inputs of the first stage
      if stage_id == 0 and not self.inputs:
        self._set_input_attrs(inputs)

      x = inputs
      for l in self.stages[stage_id]:
        x = l(x)

      return x

    def call_training_stage(stage_id, inputs, targets):

      x = call_inference_stage(stage_id, inputs)

      # Recompile the model now that we know the inputs and outputs, and
      # then create the losses and metrics
      if stage_id == len(self.stages) - 1:
        self._set_output_attrs(x)
        return self._add_loss(targets)

      return x, targets

    # Function for generating the optimizer config for pipelines.
    def optimizer_function(loss, *_):

      if not self.trainable_weights:
        raise ValueError("Model must have at least one trainable parameter.")

      opt = self._get_optimizer()

      return pipelining_ops.OptimizerFunctionOutput(opt, loss)

    # The pipeline stages, a set of feed forward functions.
    if mode == ModeKeys.PREDICT:
      stage_fn = call_inference_stage
    else:
      stage_fn = call_training_stage

    stages = []
    for stage_id in range(len(self.stages)):
      stages.append(partial(stage_fn, stage_id))

    opt = optimizer_function if mode == ModeKeys.TRAIN else None

    pipeline = pipelining_ops.pipeline(stages,
                                       pipeline_depth=self.pipeline_depth,
                                       repeat_count=repeat_count,
                                       inputs=[],
                                       infeed_queue=infeed_queue,
                                       outfeed_queue=outfeed_queue,
                                       optimizer_function=opt,
                                       name=self.name,
                                       **self.args)

    return pipeline.outputs

  @trackable.no_automatic_dependency_tracking
  def fit(self,
          x=None,
          *,
          epochs=1,
          verbose=1,
          callbacks=None,
          shuffle=True,
          initial_epoch=0,
          steps_per_epoch=None,
          steps_per_run=None,
          **kwargs):  # pylint: disable=useless-super-delegation
    """
    This provides the same functionality as the Keras Model `fit` method.

    The pipeline itself can be wrapped in a loop in order to execute a larger
    training run in a single call to hardware.  The `steps_per_run` argument
    is needed to describe how many steps should be performed on each hardware
    execution.  The dataset should be able to provide enough samples to run
    for the mini-batch size multiplied by the pipeline depth multiplied by the
    steps_per_run value.  If the dataset is infinite, because it has been
    repeated indefinitely, then this will be ok.
    """
    return super(PipelinedModel, self).fit(x,
                                           epochs=epochs,
                                           verbose=verbose,
                                           callbacks=callbacks,
                                           shuffle=shuffle,
                                           initial_epoch=initial_epoch,
                                           steps_per_epoch=steps_per_epoch,
                                           steps_per_run=steps_per_run,
                                           **kwargs)

  def evaluate(self,
               x=None,
               *,
               verbose=1,
               steps=None,
               callbacks=None,
               steps_per_run=None,
               **kwargs):  # pylint: disable=useless-super-delegation
    """
    This provides the same functionality as the Keras Model `evaluate` method.

    The pipeline itself can be wrapped in a loop in order to execute a larger
    evaluation run in a single call to hardware.  The `steps_per_run` argument
    is needed to describe how many steps should be performed on each hardware
    execution.  The dataset should be able to provide enough samples to run
    for the mini-batch size multiplied by the pipeline depth multiplied by the
    steps_per_run value.  If the dataset is infinite, because it has been
    repeated indefinitely, then this will be ok.
    """
    return super(PipelinedModel, self).evaluate(x,
                                                verbose=verbose,
                                                steps=steps,
                                                callbacks=callbacks,
                                                steps_per_run=steps_per_run,
                                                **kwargs)

  def predict(self,
              x,
              *,
              verbose=0,
              steps=None,
              callbacks=None,
              steps_per_run=None,
              **kwargs):  # pylint: disable=useless-super-delegation
    """
    This provides the same functionality as the Keras Model `predict` method.

    The predict method operates on a DataSet, because the IPU pipelining
    mechanism relies on feeding data to the system in a stream.  So single
    predictions cannot be performed using this method.  Exporting the model
    parameters, and importing them into the same model but not pipelined will
    allow single mini-batches.

    The pipeline itself can be wrapped in a loop in order to execute a larger
    prediction run in a single call to hardware.  The `steps_per_run` argument
    is needed to describe how many steps should be performed on each hardware
    execution.  The dataset should be able to provide enough samples to run
    for the mini-batch size multiplied by the pipeline depth multiplied by the
    steps_per_run value.  If the dataset is infinite, because it has been
    repeated indefinitely, then this will be ok.
    """
    return super(PipelinedModel, self).predict(x,
                                               verbose=verbose,
                                               steps=steps,
                                               callbacks=callbacks,
                                               steps_per_run=steps_per_run,
                                               **kwargs)

  def save(self,
           filepath,
           overwrite=True,
           include_optimizer=True,
           save_format=None,
           signatures=None,
           options=None):
    raise NotImplementedError(
        "IPU models do not support the `save` interface.")


class Model(_IpuModelBase):
  """A Keras Model class specifically tergetting the IPU.  This class is
  similar to the Keras Sequential model class, but it also supports the
  accumulation of gradient deltas, and an on-device training loop.

  There are some limitations with the Model compared to the standard
  Keras Model.

  - The input must be provided by a tf.DataSet.
  - Keras V1 optimizers cannot be used.
  - Loss weightings can only be specified as a list, not a callable.
  - Weighted metrics, target tensors and sample weight mode are not supported.
  - Validation cannot be performed as part of the `fit` loop.
  - The model cannot be called using the __call__() interface.

  The model will only be constructed after the first call to the `fit` method,
  so a summary of the model will not be possible until after some training
  has occurred.  Related to this, the `build` method does not build the
  model.

  Example:

  .. code-block:: python

    dataset = ...

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu.keras.Model([
        keras.layers.Dense(4),
        keras.layers.Dense(4),
        keras.layers.Dense(4),
      ])

      m.compile('sgd', loss='mse')

      m.fit(dataset, steps_per_epoch=144)

  """
  def __init__(self, layers=None, accumulation_count=1):
    """
    Creates a Keras model, optimized to run on the IPU.

    Args:
        layers: A python list of Keras Layers.
        accumulation_count: The number of mini-batches to process
            while accumulating their gradients, before running a
            parameter/weight update step.
    """
    super(Model, self).__init__(accumulation_count)

    if not isinstance(layers, list):
      raise ValueError("An IPU Model must take a list of Layers.")

    for s in layers:
      if not isinstance(s, Layer):
        raise ValueError("An IPU Model may only contain lists of Keras "
                         "Layers.")

    self.accumulation_count = accumulation_count
    self.model_layers = layers

  @trackable.no_automatic_dependency_tracking
  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              **kwargs):
    """
    This provides the same functionality as the Keras Model ``compile``
    method.

    Certain features are not supported by the IPU Model:
    - sample_weight_mode
    - weighted_metrics
    - target_tensors
    """
    return super(Model, self).compile(optimizer, loss, metrics, loss_weights,
                                      **kwargs)

  def _get_internal_run_loop(self):
    if not self.internal_loop_fn:
      fn = partial(Model._internal_run_loop, self)
      self.internal_loop_fn = def_function.function(fn,
                                                    autograph=False,
                                                    experimental_compile=True)
    return self.internal_loop_fn

  def _internal_run_loop(self, infeed_queue, outfeed_queue, repeat_count,
                         mode):
    def main_body(inputs):

      if not self.inputs:
        self._set_input_attrs(inputs)

      x = inputs
      for l in self.model_layers:
        x = l(x)

      return x

    def inference_body(inputs):
      x = main_body(inputs)

      outfeed = outfeed_queue.enqueue([x])
      return outfeed

    def training_body(inputs, targets):

      x = main_body(inputs)
      self._set_output_attrs(x)

      l = self._add_loss(targets)

      outfeed = outfeed_queue.enqueue(l)

      if not self.trainable_weights:
        raise ValueError("Model must have at least one trainable parameter.")

      opt = self._get_optimizer()
      if opt and mode == ModeKeys.TRAIN:

        # If it is gradient accumulation then wrap in that too
        if self.accumulation_count > 1:
          # TODO(T16260) generalize this when we support an alternative
          # gradient accumulation method.
          opt = gradient_accumulation_optimizer.GradientAccumulationOptimizer(
              opt, self.accumulation_count)

        # Get gradients and apply them to the trainable variables
        grads_and_vars = opt.compute_gradients(l[0], self.trainable_variables)
        opt.apply_gradients(grads_and_vars)

      return outfeed

    # The pipeline stages, a set of feed forward functions.
    if mode == ModeKeys.PREDICT:
      body = inference_body
    else:
      body = training_body

    result = loops.repeat(int(repeat_count * self.accumulation_count),
                          body,
                          infeed_queue=infeed_queue)

    return result.outputs

  @trackable.no_automatic_dependency_tracking
  def fit(self,
          x=None,
          *,
          epochs=1,
          verbose=1,
          callbacks=None,
          shuffle=True,
          initial_epoch=0,
          steps_per_epoch=None,
          steps_per_run=None,
          **kwargs):  # pylint: disable=useless-super-delegation
    """
    This provides the same functionality as the Keras Model `fit` method.

    The pipeline itself can be wrapped in a loop in order to execute a larger
    training run in a single call to hardware.  The `steps_per_run` argument
    is needed to describe how many steps should be performed on each hardware
    execution.  The dataset should be able to provide enough samples to run
    for the mini-batch size multiplied by the pipeline depth multiplied by the
    steps_per_run value.  If the dataset is infinite, because it has been
    repeated indefinitely, then this will be ok.
    """
    return super(Model, self).fit(x,
                                  epochs=epochs,
                                  verbose=verbose,
                                  callbacks=callbacks,
                                  shuffle=shuffle,
                                  initial_epoch=initial_epoch,
                                  steps_per_epoch=steps_per_epoch,
                                  steps_per_run=steps_per_run,
                                  **kwargs)

  def evaluate(self,
               x=None,
               *,
               verbose=1,
               steps=None,
               callbacks=None,
               steps_per_run=None,
               **kwargs):  # pylint: disable=useless-super-delegation
    """
    This provides the same functionality as the Keras Model `evaluate` method.

    The pipeline itself can be wrapped in a loop in order to execute a larger
    evaluation run in a single call to hardware.  The `steps_per_run` argument
    is needed to describe how many steps should be performed on each hardware
    execution.  The dataset should be able to provide enough samples to run
    for the mini-batch size multiplied by the pipeline depth multiplied by the
    steps_per_run value.  If the dataset is infinite, because it has been
    repeated indefinitely, then this will be ok.
    """
    return super(Model, self).evaluate(x,
                                       verbose=verbose,
                                       steps=steps,
                                       callbacks=callbacks,
                                       steps_per_run=steps_per_run,
                                       **kwargs)

  def predict(self,
              x,
              *,
              verbose=0,
              steps=None,
              callbacks=None,
              steps_per_run=None,
              **kwargs):  # pylint: disable=useless-super-delegation
    """
    This provides the same functionality as the Keras Model `predict` method.

    The predict method operates on a DataSet, because the IPU pipelining
    mechanism relies on feeding data to the system in a stream.  So single
    predictions cannot be performed using this method.  Exporting the model
    parameters, and importing them into the same model but not pipelined will
    allow single mini-batches.

    The pipeline itself can be wrapped in a loop in order to execute a larger
    prediction run in a single call to hardware.  The `steps_per_run` argument
    is needed to describe how many steps should be performed on each hardware
    execution.  The dataset should be able to provide enough samples to run
    for the mini-batch size multiplied by the pipeline depth multiplied by the
    steps_per_run value.  If the dataset is infinite, because it has been
    repeated indefinitely, then this will be ok.
    """
    return super(Model, self).predict(x,
                                      verbose=verbose,
                                      steps=steps,
                                      callbacks=callbacks,
                                      steps_per_run=steps_per_run,
                                      **kwargs)

  def save(self,
           filepath,
           overwrite=True,
           include_optimizer=True,
           save_format=None,
           signatures=None,
           options=None):
    raise NotImplementedError(
        "IPU models do not support the `save` interface.")
