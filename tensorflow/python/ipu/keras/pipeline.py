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
Keras Pipelined Model interfaces for IPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from collections import OrderedDict
from functools import partial
import math

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu.keras import model as ipu_model
from tensorflow.python.ipu.keras.extensions.functional_extensions import PipelineStage  # pylint: disable=unused-import
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import nest


class PipelineSequential(ipu_model._IpuModelBase):  # pylint: disable=protected-access
  """Keras Model for encapsulating a pipeline of stages to be run in parallel on an IPU system.

  A pipelined model will execute multiple sections (stages) of a model on more
  than one IPU at the same time, by pipelining mini-batches of data through
  the stages.

  It encapsulates the ipu.pipelining_ops.pipeline operation and the associated
  InFeed and OutFeed queues into a class which resembles the Keras Model class
  and provides the `fit` API for training the model.

  The different stages are specified, similarly to the Keras Sequential model,
  as a list in the constructor.  With the PipelineSequential class the list
  of layers becomes a list of lists of layers, where each list contains the
  layers for a particular stage.

  The `gradient_accumulation_count` argument describes the number of
  mini-batches which are sent through the pipeline in a single operation of the
  pipeline.  The effective batch size is therefore the mini-batch size multipled
  by the gradient accumulation count.

  Note that pipelining supports the recomputation of activations for stateless
  ops during the backwards pass. This reduces the number of activations that
  will be stored on the device, saving memory at the expense of additional
  computation. To enable recomputation, use the
  :func:`tensorflow.python.ipu.utils.set_recomputation_options()` function when
  configuring the device.

  Refer to the :py:mod:`tensorflow.python.ipu.pipelining_ops` documentation for
  more details about pipelining.

  There are some limitations with the PipelineSequential class compared to the
  standard Keras Model:

  - Keras V1 optimizers cannot be used.
  - Loss weightings can only be specified as a list, not a callable.
  - Weighted metrics, target tensors and sample weight mode are not supported.
  - Validation cannot be performed as part of the `fit` loop.
  - The model cannot be called using the __call__() interface.
  - It cannot be used in a custom training loop.
  - The model cannot be saved using the `save` interface.

  The model will only be constructed after the first call to the `fit` method,
  so a summary of the model will not be possible until after some training
  has occurred.  Related to this, the `build` method does not build the
  model.

  Example:

  .. code-block:: python

    dataset = ...

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = ipu.keras.PipelineSequential([
        [
          keras.layers.Dense(4),
          keras.layers.Dense(4),
          keras.layers.Dense(4),
        ],
        [
          keras.layers.Dense(8),
        ],
      ], gradient_accumulation_count=24)

      m.compile('sgd', loss='mse')

      m.fit(dataset, steps_per_epoch=144)

  """
  def __init__(self,
               stages,
               gradient_accumulation_count,
               gradient_accumulation_dtype=None,
               batch_serialization_iterations=1,
               device_mapping=None,
               pipeline_schedule=None,
               recomputation_mode=None,
               forward_propagation_stages_poplar_options=None,
               backward_propagation_stages_poplar_options=None,
               weight_update_poplar_options=None,
               offload_weight_update_variables=None,
               replicated_optimizer_state_sharding=False,
               offload_activations=None,
               offload_gradient_accumulation_buffers=None,
               replicated_weight_sharding=None,
               offload_weights=None,
               layer_replacement=False,
               **kwargs):
    """
    Creates a pipelined Sequential model.

    Note that arguments marked with (EXPERIMENTAL) are under active development
    and might not provide representative performance.

    Args:
        stages: A Python list of lists of Layers.
        gradient_accumulation_count: The number of mini-batches processed by
            the pipeline on each iteration.
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
        batch_serialization_iterations: (EXPERIMENTAL) number of times a loop
            executes to compute a batch on each pipeline stage execution.
            Currently only supported with the `PipelineSchedule.Sequential`.
        device_mapping: If provided, a list of length equal to the number of
            computational stages. An element at index `i` in the list
            represents which IPU the computational stage
            `computational_stages[i]` should reside on.
            This can be used to make sure computational stages which share
            `tf.Variable` objects are resident on the same IPU.
        pipeline_schedule: the scheduling algorithm to use for pipeline
            lowering. Must be of type
            :class:`~tensorflow.python.ipu.pipelining_ops.PipelineSchedule`.
        recomputation_mode: the recomputation mode to use for training pipeline
            models. Must be of type
            :class:`~tensorflow.python.ipu.pipelining_ops.RecomputationMode`.
        forward_propagation_stages_poplar_options: If provided, a list of length
            equal to the number of computational stages. Each element is a
            :class:`~tensorflow.python.ipu.pipelining_ops.PipelineStageOptions`
            object which allows for fine grain control
            of the Poplar options for a given forward propagation computational
            stage.
        backward_propagation_stages_poplar_options: If provided, a list of
            length equal to the number of computational stages. Each element is
            a
            :class:`~tensorflow.python.ipu.pipelining_ops.PipelineStageOptions`
            object which allows for fine grained control
            of the Poplar options for a given backward propagation
            computational stage.
        weight_update_poplar_options: If provided, a
            :class:`~tensorflow.python.ipu.pipelining_ops.PipelineStageOptions`
            object which allows for fine grained control of the Poplar options
            for the weight update stage.
        offload_weight_update_variables: When enabled, any `tf.Variable` which
            is only used by the weight update of the pipeline (for example the
            accumulator variable when using the `tf.MomentumOptimizer`), will
            be stored in the remote memory. During the weight update this
            variable will be streamed onto the device and then streamed back
            to the remote memory after it has been updated. Requires the
            machine to be configured with support for `Poplar remote buffers`.
            Offloading variables into remote memory can reduce maximum memory
            liveness, but can also increase the computation time of the weight
            update. When set to `None` the variables will be placed in either
            in-processor or remote memory automatically based on the current
            best placement strategy.
            Note that this option has no effect for inference only pipelines.
        replicated_optimizer_state_sharding: (EXPERIMENTAL) If True, any
            `tf.Variable` which is offloaded (for example the accumulator
            variable when using the `tf.MomentumOptimizer`), will be partitioned
            across the replicas. This can exploit the additional bandwidth of
            the IPU-Links to improve overall throughput.
            Note that this option has no effect for inference-only pipelines.
        offload_activations: When enabled, all the activations for the batches
            which are not being executed by the pipeline stages at the given
            time are stored in remote memory. Requires the machine to be
            configured with support for `Poplar remote buffers`. Offloading
            activations into remote memory can reduce maximum memory liveness,
            but can also increase the computation time as activations have to
            be copied from/to the device(s). When set to `None`, the
            activations might be offloaded when beneficial. This feature is
            currently only supported when the pipeline schedule is
            `PipelineSchedule.Sequential` and
            `batch_serialization_iterations > 1`.
        offload_gradient_accumulation_buffers: (EXPERIMENTAL) When enabled, all
            the gradient accumulation buffers are stored in remote memory.
            Offloading gradient accumulation buffers into remote memory can
            reduce maximum memory liveness, but can also increase the
            computation time as the buffers have to be copied to the device,
            updated and the copied off the device. Requires the machine to be
            configured with support for `Poplar remote buffers`.
            When set to `None`, the `offload_gradient_accumulation_buffers`
            might be offloaded when beneficial.
            Note that this option has no effect for inference-only pipelines.
        replicated_weight_sharding: (EXPERIMENTAL) When enabled and running a
            replicated model any `tf.Variable` objects used by the pipeline
            stage computations (excluding those only used by the weight update)
            will be partitioned across the replicas. Whenever a partitioned
            `tf.Variable` is accessed, it will be first all-gathered across
            replicas to make sure each replica has access to the whole
            `tf.Variable`. This can exploit the additional bandwidth of the
            IPU-Links to improve overall throughput. When set to `None`, the
            activations might be offloaded when beneficial. This feature is
            enabled by default when the pipeline schedule is
            `PipelineSchedule.Sequential` and
            `batch_serialization_iterations > 1`, where this option can reduce
            the memory usage at the cost of extra communication.
        offload_weights: (EXPERIMENTAL) When enabled and
            `replicated_weight_sharding` is enabled, any `tf.Variable` which are
            partitioned across replicas will be stored in
            `Poplar remote buffers`. Offloading variables into remote memory can
            further reduce maximum memory liveness, but can also increase the
            computation time due to extra communication. When set to `None` the
            variables will be placed in either in-processor or remote memory
            automatically based on the current best placement strategy.
        layer_replacement: If enabled (True), Keras layers will be substituted
          with IPU Keras implementations, when possible.
        name: Optional name for the pipeline operation.
    """

    if not isinstance(stages, list) or not stages:
      raise ValueError("An IPU pipeline must take a non-empty list of stages, "
                       "where each stage is a list of Keras Layers.")

    for s in stages:
      if not isinstance(s, list):
        raise ValueError("An IPU pipeline may only contain lists of "
                         "stages, where each stage is a list of Keras Layers.")
      for l in s:
        if not isinstance(l, Layer):
          raise ValueError("Each list in the `stages` list must contain "
                           "only Keras Layers.")

    shard_count = max(device_mapping) + 1 if device_mapping else \
                  len(stages)

    accumulation_count = gradient_accumulation_count * \
      batch_serialization_iterations
    super().__init__(accumulation_count,
                     shard_count,
                     layer_replacement=layer_replacement,
                     **kwargs)

    self.gradient_accumulation_count = gradient_accumulation_count
    self.gradient_accumulation_dtype = gradient_accumulation_dtype
    self.stages = stages

    # Store additional pipeline params.
    self.batch_serialization_iterations = batch_serialization_iterations
    self.device_mapping = device_mapping
    self.pipeline_schedule = pipeline_schedule
    self.recomputation_mode = recomputation_mode
    self.forward_propagation_stages_poplar_options = \
      forward_propagation_stages_poplar_options
    self.backward_propagation_stages_poplar_options = \
      backward_propagation_stages_poplar_options
    self.weight_update_poplar_options = weight_update_poplar_options
    self.offload_weight_update_variables = \
      offload_weight_update_variables
    self.replicated_optimizer_state_sharding = \
      replicated_optimizer_state_sharding
    self.offload_activations = offload_activations
    self.offload_gradient_accumulation_buffers = \
      offload_gradient_accumulation_buffers
    self.replicated_weight_sharding = replicated_weight_sharding
    self.offload_weights = offload_weights
    self._num_inputs = 1
    self._num_outputs = 1

  def build(self, input_shape):
    """Builds the model based on input shapes received.

    Args:
     input_shape: Single tuple, TensorShape, or list of shapes, where shapes
         are tuples, integers, or TensorShapes.
    """
    s = input_shape
    for l in self.layers:
      l.build(s)
      s = l.compute_output_shape(s)
    self.built = True

    if self._layer_replacer:
      for l in self.layers:
        l = self._layer_replacer(l)

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

    Certain features are not supported by the IPU PipelineSequential class:

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
    run_loop_kwargs = run_loop_kwargs or {}
    accumulate_outfeed = run_loop_kwargs.get("accumulate_outfeed", False)
    accumulate_outfeed_dtype = run_loop_kwargs.get("accumulate_outfeed_dtype",
                                                   None)

    # Plain functions to build a stage
    def call_inference_stage(stage_id, inputs):
      # Record the inputs of the first stage
      if stage_id == 0 and not self.inputs:
        self._set_input_attrs(inputs)

      x = inputs
      for l in self.stages[stage_id]:
        stage_kwargs = {}
        argspec = tf_inspect.getfullargspec(l.call).args
        if 'training' in argspec:
          stage_kwargs['training'] = training
        x = l(x, **stage_kwargs)

      return x

    def call_training_stage(stage_id, inputs, targets):

      x = call_inference_stage(stage_id, inputs)

      # Recompile the model now that we know the inputs and outputs, and
      # then create the losses and metrics
      if stage_id == len(self.stages) - 1:
        self._set_output_attrs(x)
        losses_and_metrics = self._add_loss(targets)
        # Normalize metrics by accumulation count if we're accumulating
        if accumulate_outfeed and len(losses_and_metrics) > 1:
          for i in range(1, len(losses_and_metrics)):
            losses_and_metrics[i] /= self.gradient_accumulation_count
        return losses_and_metrics

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

    opt = optimizer_function if training else None

    pipeline = pipelining_ops.pipeline(
        stages,
        gradient_accumulation_count=self.gradient_accumulation_count,
        gradient_accumulation_dtype=self.gradient_accumulation_dtype,
        repeat_count=repeat_count,
        inputs=[],
        infeed_queue=infeed_queue,
        outfeed_queue=outfeed_queue,
        optimizer_function=opt,
        device_mapping=self.device_mapping,
        pipeline_schedule=self.pipeline_schedule,
        recomputation_mode=self.recomputation_mode,
        forward_propagation_stages_poplar_options=self.
        forward_propagation_stages_poplar_options,
        backward_propagation_stages_poplar_options=self.
        backward_propagation_stages_poplar_options,
        weight_update_poplar_options=self.weight_update_poplar_options,
        offload_weight_update_variables=self.offload_weight_update_variables,
        replicated_optimizer_state_sharding=self.
        replicated_optimizer_state_sharding,
        offload_activations=self.offload_activations,
        offload_gradient_accumulation_buffers=self.
        offload_gradient_accumulation_buffers,
        replicated_weight_sharding=self.replicated_weight_sharding,
        offload_weights=self.offload_weights,
        accumulate_outfeed=accumulate_outfeed,
        accumulate_outfeed_dtype=accumulate_outfeed_dtype,
        name=self.name)

    return pipeline.outputs

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
          accumulate_outfeed=False,
          accumulate_outfeed_dtype=None,
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
      accumulate_outfeed: The loss and metrics from the final pipeline
        stage are normally enqueued as soon as they're available. If this
        option is True, the data will instead be accumulated when they're
        available and enqueued at the end of pipeline execution, reducing
        the amount of host <-> device communication. The accumulated metrics are
        normalised by the `gradient_accumulation_count`.
      accumulate_outfeed_dtype: The data type used for the outfeed accumulation
        buffers. One of:

        - `None`: Use an accumulator of the same type as the variable type.
        - A `DType`: Use this type for all the accumulators.
        - A callable that takes the variable and returns a `DType`. Allows
          specifying the accumulator type on a per-variable basis. Variables
          given to this callable will be called "PipelineStage:0",
          "PipelineStage:1", etc. The loss is the first output, then the
          metrics are the consequent outputs, in the order they were
          specified. For example:

          .. code-block:: python

            # Compile the model with MSE loss and two metrics.
            model.compile(opt, loss='mse', metrics=['mse', 'accuracy'])

            def accumulator_dtype_fn(var):
              # The MSE loss is the first output.
              if var.name == "PipelineStage:0":
                return tf.float16
              # The metrics are "PipelineStage:1" and "PipelineStage:2"
              # respectively.
              if var.name == "PipelineStage:1":
                # Accumulate the MSE metric in float32
                return tf.float32
              if var.name == "PipelineStage:2":
                # Accumulate the accuracy in float16
                return tf.float16
              return tf.float32

            model.fit(...
                      accumulate_outfeed=True,
                      accumulate_outfeed_dtype=accumulator_dtype_fn)

    Returns:
      A `History` object. Its `History.history` attribute is a record of
      training loss values and metrics values at successive epochs.
    Raises:
      ValueError: if there are invalid arguments.
    """
    run_loop_kwargs = {
        "accumulate_outfeed": accumulate_outfeed,
        "accumulate_outfeed_dtype": accumulate_outfeed_dtype
    }
    kwargs["run_loop_kwargs"] = run_loop_kwargs
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
               accumulate_outfeed=False,
               accumulate_outfeed_dtype=None,
               **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
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
      accumulate_outfeed: The loss and metrics from the final pipeline
        stage are normally enqueued as soon as they're available. If this
        option is True, the data will instead be accumulated when they're
        available and enqueued at the end of pipeline execution, reducing
        the amount of host <-> device communication. The accumulated metrics are
        normalised by the `gradient_accumulation_count`.
      accumulate_outfeed_dtype: The data type used for the outfeed accumulation
        buffers. One of:

        - `None`: Use an accumulator of the same type as the variable type.
        - A `DType`: Use this type for all the accumulators.
        - A callable that takes the variable and returns a `DType`. Allows
          specifying the accumulator type on a per-variable basis. Variables
          given to this callable will be called "PipelineStage:0",
          "PipelineStage:1", etc. The loss is the first output, then the
          metrics are the consequent outputs, in the order they were
          specified. For example:

          .. code-block:: python

            # Compile the model with MSE loss and two metrics.
            model.compile(opt, loss='mse', metrics=['mse', 'accuracy'])

            def accumulator_dtype_fn(var):
              # The MSE loss is the first output.
              if var.name == "PipelineStage:0":
                return tf.float16
              # The metrics are "PipelineStage:1" and "PipelineStage:2"
              # respectively.
              if var.name == "PipelineStage:1":
                # Accumulate the MSE metric in float32
                return tf.float32
              if var.name == "PipelineStage:2":
                # Accumulate the accuracy in float16
                return tf.float16
              return tf.float32

            model.fit(...
                      accumulate_outfeed=True,
                      accumulate_outfeed_dtype=accumulator_dtype_fn)

    Returns:
      Scalar test loss (if the model has a single output and no metrics) or list
      of scalars (if the model has multiple outputs and/or metrics). The
      attribute model.metrics_names will give you the display labels for the
      scalar outputs.
    Raises:
      ValueError: if there are invalid arguments.
    """
    run_loop_kwargs = {
        "accumulate_outfeed": accumulate_outfeed,
        "accumulate_outfeed_dtype": accumulate_outfeed_dtype
    }
    kwargs["run_loop_kwargs"] = run_loop_kwargs
    return super().evaluate(x,
                            y,
                            batch_size=batch_size,
                            verbose=verbose,
                            steps=steps,
                            callbacks=callbacks,
                            steps_per_run=steps_per_run,
                            prefetch_depth=prefetch_depth,
                            **kwargs)

  def predict(self,
              x,
              *,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              steps_per_run=None,
              prefetch_depth=None,
              **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
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

    This means that single predications cannot be performed using this method.
    Saving the model weights, and loading them into a non-pipelined version of
    the same model will allow single mini-batches (using gradient accumulation
    count = 1).

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


SequentialPipelineModel = deprecation.deprecated_alias(
    deprecated_name="SequentialPipelineModel",
    name="PipelineSequential",
    func_or_class=PipelineSequential)


class PipelineModel(ipu_model.Model):
  """Keras Model for encapsulating a pipeline of stages to be run in
  parallel on an IPU system.

  A pipelined model will execute multiple sections (stages) of a model on more
  than one IPU at the same time, by pipelining mini-batches of data through
  the stages.

  The different stages are specified when defining the graph structure via use
  of the `PipelineStage` context manager. Pipeline stages can be assigned to
  all calls of layer by constructing the layer within a `PipelineStage` scope
  as follows:

  .. code-block:: python

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    input_layer = Input(2)
    with strategy.scope():
      with PipelineStage(0):
        x = Dense(4)(input_layer)

      with PipelineStage(1):
        x = Dense(4)(x)

  Pipeline stages can also be assigned to individual layer calls, as follows:

  .. code-block:: python

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    input_layer = Input(2)
    l = Dense(4)
    with strategy.scope():
      with PipelineStage(0):
        x = l(input_layer)

      with PipelineStage(1):
        x = l(x)

  Pipeline stages assigned to layer calls take precedence over those assigned
  when constructing the layer.

  If a layer which use Variables (such as weights) is assigned to multiple
  pipeline stages, these stages must be mapped to the same device. This can
  be done using the `device_mapping` argument.

  The `gradient_accumulation_count` argument describes the number of
  mini-batches which are sent through the pipeline in a single operation of the
  pipeline. The effective batch size is therefore the mini-batch size multipled
  by the gradient accumulation count.

  Note that pipelining supports the recomputation of activations for stateless
  ops during the backwards pass. This reduces the number of activations that
  will be stored on the device, saving memory at the expense of additional
  computation. To enable recomputation, use the
  :func:`tensorflow.python.ipu.utils.set_recomputation_options()` function when
  configuring the device.

  Refer to the :py:mod:`tensorflow.python.ipu.pipelining_ops` documentation for
  more details about pipelining.

  There are some limitations with the PipelineModel compared to the
  standard Keras Model:

  - Keras V1 optimizers cannot be used.
  - Loss weightings can only be specified as a list, not a callable.
  - Weighted metrics, target tensors and sample weight mode are not supported.
  - Validation cannot be performed as part of the `fit` loop.
  - The model cannot be called using the __call__() interface.
  - It cannot be used in a custom training loop.
  - The model cannot be saved using the `save` interface.

  Example:

  .. code-block:: python

    dataset = ...

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(4)

      with ipu.keras.PipelineStage(0):
        x = keras.layers.Dense(4, activation="relu")(input_layer)
        x = keras.layers.Dense(4, activation="relu")(x)

      with ipu.keras.PipelineStage(1):
        x = keras.layers.Dense(4, activation="relu")(x)
        x = keras.layers.Dense(4, activation="relu")(x)

      with ipu.keras.PipelineStage(2):
        x = keras.layers.Dense(2, activation="relu")(x)

      model = ipu.keras.PipelineModel(inputs=inputs, outputs=x,
                                      gradient_accumulation_count=12)

      model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"])

      model.fit(dataset, epochs=2, steps_per_epoch=128)
  """
  def __init__(self,
               *args,
               gradient_accumulation_count,
               gradient_accumulation_dtype=None,
               batch_serialization_iterations=1,
               device_mapping=None,
               pipeline_schedule=None,
               recomputation_mode=None,
               forward_propagation_stages_poplar_options=None,
               backward_propagation_stages_poplar_options=None,
               weight_update_poplar_options=None,
               offload_weight_update_variables=None,
               replicated_optimizer_state_sharding=False,
               offload_activations=None,
               offload_gradient_accumulation_buffers=None,
               replicated_weight_sharding=None,
               offload_weights=None,
               layer_replacement=False,
               **kwargs):
    """
    Creates a pipelined model (defined via the Keras Functional API).

    Needs to pass in ``inputs`` and ``outputs`` as either arguments or
    keyword arguments.

    Note that arguments marked with (EXPERIMENTAL) are under active development
    and might not provide representative performance.

    Args:
        gradient_accumulation_count: The number of mini-batches processed by
            the pipeline on each iteration.
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
        batch_serialization_iterations: (EXPERIMENTAL) number of times a loop
            executes to compute a batch on each pipeline stage execution.
            Currently only supported with the `PipelineSchedule.Sequential`.
        device_mapping: If provided, a list of length equal to the number of
            computational stages. An element at index `i` in the list
            represents which IPU the computational stage
            `computational_stages[i]` should reside on.
            This can be used to make sure computational stages which share
            `tf.Variable` objects are resident on the same IPU.
        pipeline_schedule: the scheduling algorithm to use for pipeline
            lowering. Must be of type
            :class:`~tensorflow.python.ipu.pipelining_ops.PipelineSchedule`.
        recomputation_mode: the recomputation mode to use for training pipeline
            models. Must be of type
            :class:`~tensorflow.python.ipu.pipelining_ops.RecomputationMode`.
        forward_propagation_stages_poplar_options: If provided, a list of
            length equal to the number of computational stages. Each element is
            a
            :class:`~tensorflow.python.ipu.pipelining_ops.PipelineStageOptions`
            object which allows for fine grain control of
            the Poplar options for a given forward propagation computational
            stage.
        backward_propagation_stages_poplar_options: If provided, a list of
            length equal to the number of computational stages. Each element is
            a
            :class:`~tensorflow.python.ipu.pipelining_ops.PipelineStageOptions`
            object which allows for fine grained control
            of the Poplar options for a given backward propagation computational
            stage.
        weight_update_poplar_options: If provided, a
            :class:`~tensorflow.python.ipu.pipelining_ops.PipelineStageOptions`
            object which allows for fine grained control of the Poplar options
            for the weight update stage.
        offload_weight_update_variables: When enabled, any `tf.Variable` which
            is only used by the weight update of the pipeline (for example the
            accumulator variable when using the `tf.MomentumOptimizer`), will
            be stored in the remote memory. During the weight update this
            variable will be streamed onto the device and then streamed back
            to the remote memory after it has been updated. Requires the
            machine to be configured with support for `Poplar remote buffers`.
            Offloading variables into remote memory can reduce maximum memory
            liveness, but can also increase the computation time of the weight
            update. When set to `None` the variables will be placed in either
            in-processor or remote memory automatically based on the current
            best placement strategy.
            Note that this option has no effect for inference only pipelines.
        replicated_optimizer_state_sharding: (EXPERIMENTAL) If True, any
            `tf.Variable` which is offloaded (for example the accumulator
            variable when using the `tf.MomentumOptimizer`), will be partitioned
            across the replicas. This can exploit the additional bandwidth of
            the IPU-Links to improve overall throughput.
            Note that this option has no effect for inference-only pipelines.
        offload_activations: When enabled, all the activations for the batches
            which are not being executed by the pipeline stages at the given
            time are stored in remote memory. Requires the machine to be
            configured with support for `Poplar remote buffers`. Offloading
            activations into remote memory can reduce maximum memory liveness,
            but can also increase the computation time as activations have to
            be copied from/to the device(s). When set to `None`, the
            activations might be offloaded when beneficial. This feature is
            currently only supported when the pipeline schedule is
            `PipelineSchedule.Sequential` and
            `batch_serialization_iterations > 1`.
        offload_gradient_accumulation_buffers: (EXPERIMENTAL) When enabled, all
            the gradient accumulation buffers are stored in remote memory.
            Offloading gradient accumulation buffers into remote memory can
            reduce maximum memory liveness, but can also increase the
            computation time as the buffers have to be copied to the device,
            updated and the copied off the device. Requires the machine to be
            configured with support for `Poplar remote buffers`.
            When set to `None`, the `offload_gradient_accumulation_buffers`
            might be offloaded when beneficial.
            Note that this option has no effect for inference-only pipelines.
        replicated_weight_sharding: (EXPERIMENTAL) When enabled and running a
            replicated model any `tf.Variable` objects used by the pipeline
            stage computations (excluding those only used by the weight update)
            will be partitioned across the replicas. Whenever a partitioned
            `tf.Variable` is accessed, it will be first all-gathered across
            replicas to make sure each replica has access to the whole
            `tf.Variable`. This can exploit the additional bandwidth of the
            IPU-Links to improve overall throughput. When set to `None`, the
            activations might be offloaded when beneficial. This feature is
            enabled by default when the pipeline schedule is
            `PipelineSchedule.Sequential` and
            `batch_serialization_iterations > 1`, where this option can reduce
            the memory usage at the cost of extra communication.
        offload_weights: (EXPERIMENTAL) When enabled and
            `replicated_weight_sharding` is enabled, any `tf.Variable` which are
            partitioned across replicas will be stored in
            `Poplar remote buffers`. Offloading variables into remote memory can
            further reduce maximum memory liveness, but can also increase the
            computation time due to extra communication. When set to `None` the
            variables will be placed in either in-processor or remote memory
            automatically based on the current best placement strategy.
        layer_replacement: If enabled (True), Keras layers will be substituted
          with IPU Keras implementations, when possible.
        name: Optional name for the pipeline operation.
    """
    accumulation_count = gradient_accumulation_count * \
      batch_serialization_iterations
    super().__init__(*args,
                     gradient_accumulation_count=accumulation_count,
                     layer_replacement=layer_replacement,
                     **kwargs)

    # Mutable attributes will be seen as trainable and e.g. added as Layers.
    # Define them inside this function if you don't want them tracked.
    self._pipeline_init_network()

    # Compute shard count.
    shard_count = max(device_mapping) + 1 if device_mapping else \
                  len(self.stages)
    # Round the shard count to the next power of two.
    self.shard_count = 2**int(math.ceil(math.log2(shard_count)))

    self.gradient_accumulation_count = gradient_accumulation_count
    self.gradient_accumulation_dtype = gradient_accumulation_dtype

    # Store additional pipeline params.
    self.batch_serialization_iterations = batch_serialization_iterations
    self.device_mapping = device_mapping
    self.pipeline_schedule = pipeline_schedule
    self.recomputation_mode = recomputation_mode
    self.forward_propagation_stages_poplar_options = \
      forward_propagation_stages_poplar_options
    self.backward_propagation_stages_poplar_options = \
      backward_propagation_stages_poplar_options
    self.weight_update_poplar_options = weight_update_poplar_options
    self.offload_weight_update_variables = \
      offload_weight_update_variables
    self.replicated_optimizer_state_sharding = \
      replicated_optimizer_state_sharding
    self.offload_activations = offload_activations
    self.offload_gradient_accumulation_buffers = \
      offload_gradient_accumulation_buffers
    self.replicated_weight_sharding = replicated_weight_sharding
    self.offload_weights = offload_weights

  @trackable.no_automatic_dependency_tracking
  def _pipeline_init_network(self):
    # Assign stages to nodes.
    self.stages = self._assign_node_stages()
    self._stage_node_ids = self._get_per_stage_node_ids()

  def _assign_node_stages(self):
    # Get stages for each layer - used for mapping to nodes below.
    nodes_per_stage = {}
    max_stage_id = -1

    num_inputs = len(self._input_layers)
    # The first num_inputs layers are input layers which don't need pipeline
    # stages assigned.
    for node in self._post_order_node_execution[num_inputs:]:
      # Verify that a pipeline stage has been assigned.
      if not hasattr(node, "_pipeline_stage"):
        if not hasattr(node.outbound_layer, "_pipeline_stage"):
          raise ValueError(
              f"All layers of a pipelined model must have an associated "
              f"pipeline stage.\nHowever, {node.outbound_layer} has not been "
              f"assigned to one.\nPipeline stages can be assigned when a "
              f"layer is constructed, or each time a layer is called."
              f"\nDifferent pipeline stages can assigned to each call.")
        node._pipeline_stage = node.outbound_layer._pipeline_stage  # pylint: disable=protected-access

      pipeline_stage = node._pipeline_stage  # pylint: disable=protected-access
      nodes_per_stage.setdefault(pipeline_stage, []).append(node)
      max_stage_id = max(max_stage_id, pipeline_stage)

    # Check that all pipeline stages are visited.
    found_stages = sorted(nodes_per_stage.keys())
    num_stages = max_stage_id + 1

    if found_stages != list(range(num_stages)):
      missing_stages = set(range(num_stages)) - set(found_stages)
      raise ValueError(
          "Pipeline stages in the graph need to be strictly increasing, "
          "found pipeline stages %s, however the following pipeline stages "
          "are missing %s." % (", ".join(str(v)
                                         for v in found_stages), ", ".join(
                                             str(v) for v in missing_stages)))

    # Post order does not take pipeline stages into account, for example
    # multiple pipeline stages might have output layers. Try and reorder the
    # the nodes to preserve post order and to make sure pipeline stages
    # can still be executed in order.
    new_post_order_node_execution = []

    # Set of reference tensors which were computed.
    computed_set = set()
    for op, layer in zip(self.inputs, self._input_layers):
      assert len(layer.inbound_nodes) == 1
      new_post_order_node_execution.append(layer.inbound_nodes[0])
      computed_set.add(str(id(op)))

    # New post order executes all the layers within a pipeline stage and it
    # makes sure that all the layer inputs have already executed.
    for stage_id in range(num_stages):
      for node in nodes_per_stage[stage_id]:
        all_inputs_executed = all(
            str(id(tensor)) in computed_set
            for tensor in nest.flatten(node.input_tensors))
        if not all_inputs_executed:
          raise ValueError(
              "Layer %s in pipeline stage %d has a dependency from a pipeline "
              "stage which has not yet executed. Layers can only use outputs "
              "from current or previous pipeline stages." %
              (node.outbound_layer.name, node._pipeline_stage))  # pylint: disable=protected-access
        new_post_order_node_execution.append(node)
        # Update computed_set.
        computed_set.update(
            [str(id(x)) for x in nest.flatten(node.output_tensors)])

    assert len(new_post_order_node_execution) == len(
        self._post_order_node_execution)
    self._post_order_node_execution = new_post_order_node_execution

    stage_node_ids = list(set(nodes_per_stage.keys()))
    stage_node_ids.sort()
    return stage_node_ids

  def _get_per_stage_node_ids(self):
    assert self.stages

    stage_node_ids = [[]] * len(self.stages)

    for i, n in enumerate(self._post_order_node_execution[len(self.inputs):]):
      s = n._pipeline_stage  # pylint: disable=protected-access
      assert s < len(self.stages)
      stage_node_ids[s].append(i)

    return stage_node_ids

  def _internal_run_loop(self,
                         infeed_queue,
                         outfeed_queue,
                         repeat_count,
                         mode,
                         run_loop_kwargs=None):
    training = mode == ModeKeys.TRAIN
    run_loop_kwargs = run_loop_kwargs or {}
    accumulate_outfeed = run_loop_kwargs.get("accumulate_outfeed", False)
    accumulate_outfeed_dtype = run_loop_kwargs.get("accumulate_outfeed_dtype",
                                                   None)

    # Dictionary mapping reference tensors to computed tensors.
    tensor_dict = OrderedDict()

    def get_inputs_and_targets(*args):
      args = nest.flatten(args)
      num_inputs = len(self.inputs)
      inputs = list(args[:num_inputs])
      targets = list(args[num_inputs:])
      assert len(inputs) == num_inputs

      # "Execute" the input layers
      executed_inputs = []
      for op, layer, tensor in zip(self.inputs, self._input_layers, inputs):
        executed_inputs.append(layer(tensor))
        tensor_dict[str(id(op))] = executed_inputs[-1]
        if isinstance(op, ops.Tensor) and isinstance(tensor, ops.Tensor):
          try:
            tensor.set_shape(tensor.shape.merge_with(op.shape))
          except ValueError:
            logging.warning(
                'Model was constructed with shape {} for input {}, but it '
                'was re-called on a Tensor with incompatible '
                'shape {}.'.format(op, op.shape, tensor.shape))
      return executed_inputs, targets

    def main_body(stage_id, *args):
      if stage_id == self.stages[0]:
        inputs, targets = get_inputs_and_targets(*args)
      else:
        inputs = list(args[:len(tensor_dict)])
        targets = list(args[len(inputs):])

      # Update the tensor dict with the inputs.
      for idx, k in enumerate(tensor_dict):
        tensor_dict[k] = inputs[idx]

      for i in self._stage_node_ids[stage_id]:
        node = self._post_order_node_execution[len(self.inputs) + i]
        if node._pipeline_stage == stage_id:  # pylint: disable=protected-access
          self._execute_layer_node(node, training, tensor_dict)  # pylint: disable=protected-access

      if stage_id == self.stages[-1]:
        return self._get_output_tensors(tensor_dict), targets  # pylint: disable=protected-access
      return list(tensor_dict.values()), targets

    def inference_body(stage_id, *args):
      return main_body(stage_id, *args)[0]

    def training_body(stage_id, *args):
      outputs, targets = main_body(stage_id, *args)
      if stage_id == self.stages[-1]:
        losses_and_metrics = self._add_loss(nest.flatten(outputs), targets)
        # Normalize metrics by accumulation count if we're accumulating
        if accumulate_outfeed and len(losses_and_metrics) > 1:
          for i in range(1, len(losses_and_metrics)):
            losses_and_metrics[i] /= self.gradient_accumulation_count
        return losses_and_metrics
      return outputs + targets

    def optimizer_function(total_loss, *_):
      if not self.trainable_weights:
        raise ValueError("Model must have at least one trainable parameter.")

      opt = self._get_optimizer()
      return pipelining_ops.OptimizerFunctionOutput(opt, total_loss)

    # The pipeline stages, a set of feed forward functions.
    if mode == ModeKeys.PREDICT:
      stage_fn = inference_body
    else:
      stage_fn = training_body

    stages = []
    for stage in self.stages:
      stages.append(partial(stage_fn, stage))

    opt = optimizer_function if training else None

    pipeline = pipelining_ops.pipeline(
        stages,
        gradient_accumulation_count=self.gradient_accumulation_count,
        gradient_accumulation_dtype=self.gradient_accumulation_dtype,
        repeat_count=repeat_count,
        inputs=[],
        infeed_queue=infeed_queue,
        outfeed_queue=outfeed_queue,
        optimizer_function=opt,
        device_mapping=self.device_mapping,
        pipeline_schedule=self.pipeline_schedule,
        recomputation_mode=self.recomputation_mode,
        forward_propagation_stages_poplar_options=self.
        forward_propagation_stages_poplar_options,
        backward_propagation_stages_poplar_options=self.
        backward_propagation_stages_poplar_options,
        weight_update_poplar_options=self.weight_update_poplar_options,
        offload_weight_update_variables=self.offload_weight_update_variables,
        replicated_optimizer_state_sharding=self.
        replicated_optimizer_state_sharding,
        offload_activations=self.offload_activations,
        offload_gradient_accumulation_buffers=self.
        offload_gradient_accumulation_buffers,
        replicated_weight_sharding=self.replicated_weight_sharding,
        offload_weights=self.offload_weights,
        accumulate_outfeed=accumulate_outfeed,
        accumulate_outfeed_dtype=accumulate_outfeed_dtype,
        name=self.name)

    return pipeline.outputs

  @trackable.no_automatic_dependency_tracking
  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              **kwargs):
    """
    This provides the same functionality as the Keras Model ``compile`` method.

    Certain features are not supported by the IPU PipelineModel:

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
          accumulate_outfeed=False,
          accumulate_outfeed_dtype=None,
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
      accumulate_outfeed: The loss and metrics from the final pipeline
        stage are normally enqueued as soon as they're available. If this
        option is True, the data will instead be accumulated when they're
        available and enqueued at the end of pipeline execution, reducing
        the amount of host <-> device communication. The accumulated metrics are
        normalised by the `gradient_accumulation_count`.
      accumulate_outfeed_dtype: The data type used for the outfeed accumulation
        buffers. One of:

        - `None`: Use an accumulator of the same type as the variable type.
        - A `DType`: Use this type for all the accumulators.
        - A callable that takes the variable and returns a `DType`. Allows
          specifying the accumulator type on a per-variable basis. Variables
          given to this callable will be called "PipelineStage:0",
          "PipelineStage:1", etc. The loss is the first output, then the
          metrics are the consequent outputs, in the order they were
          specified. For example:

          .. code-block:: python

            # Compile the model with MSE loss and two metrics.
            model.compile(opt, loss='mse', metrics=['mse', 'accuracy'])

            def accumulator_dtype_fn(var):
              # The MSE loss is the first output.
              if var.name == "PipelineStage:0":
                return tf.float16
              # The metrics are "PipelineStage:1" and "PipelineStage:2"
              # respectively.
              if var.name == "PipelineStage:1":
                # Accumulate the MSE metric in float32
                return tf.float32
              if var.name == "PipelineStage:2":
                # Accumulate the accuracy in float16
                return tf.float16
              return tf.float32

            model.fit(...
                      accumulate_outfeed=True,
                      accumulate_outfeed_dtype=accumulator_dtype_fn)

    Returns:
      A `History` object. Its `History.history` attribute is a record of
      training loss values and metrics values at successive epochs.
    Raises:
      ValueError: if there are invalid arguments.
    """
    run_loop_kwargs = {
        "accumulate_outfeed": accumulate_outfeed,
        "accumulate_outfeed_dtype": accumulate_outfeed_dtype
    }
    kwargs["run_loop_kwargs"] = run_loop_kwargs
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
               accumulate_outfeed=False,
               accumulate_outfeed_dtype=None,
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
      accumulate_outfeed: The loss and metrics from the final pipeline
        stage are normally enqueued as soon as they're available. If this
        option is True, the data will instead be accumulated when they're
        available and enqueued at the end of pipeline execution, reducing
        the amount of host <-> device communication. The accumulated metrics are
        normalised by the `gradient_accumulation_count`.
      accumulate_outfeed_dtype: The data type used for the outfeed accumulation
        buffers. One of:

        - `None`: Use an accumulator of the same type as the variable type.
        - A `DType`: Use this type for all the accumulators.
        - A callable that takes the variable and returns a `DType`. Allows
          specifying the accumulator type on a per-variable basis. Variables
          given to this callable will be called `PipelineStage:0`,
          `PipelineStage:1`, etc. The loss is the first output, then the
          metrics are the consequent outputs, in the order they were
          specified. For example:

          .. code-block:: python

            # Compile the model with MSE loss and two metrics.
            model.compile(opt, loss='mse', metrics=['mse', 'accuracy'])

            def accumulator_dtype_fn(var):
              # The MSE loss is the first output.
              if var.name == "PipelineStage:0":
                return tf.float16
              # The metrics are "PipelineStage:1" and "PipelineStage:2"
              # respectively.
              if var.name == "PipelineStage:1":
                # Accumulate the MSE metric in float32
                return tf.float32
              if var.name == "PipelineStage:2":
                # Accumulate the accuracy in float16
                return tf.float16
              return tf.float32

            model.fit(...
                      accumulate_outfeed=True,
                      accumulate_outfeed_dtype=accumulator_dtype_fn)

    Returns:
      Scalar test loss (if the model has a single output and no metrics) or list
      of scalars (if the model has multiple outputs and/or metrics). The
      attribute model.metrics_names will give you the display labels for the
      scalar outputs.
    Raises:
      ValueError: if there are invalid arguments.
    """
    run_loop_kwargs = {
        "accumulate_outfeed": accumulate_outfeed,
        "accumulate_outfeed_dtype": accumulate_outfeed_dtype
    }
    kwargs["run_loop_kwargs"] = run_loop_kwargs
    return super().evaluate(x,
                            y,
                            batch_size=batch_size,
                            verbose=verbose,
                            steps=steps,
                            callbacks=callbacks,
                            steps_per_run=steps_per_run,
                            prefetch_depth=prefetch_depth,
                            **kwargs)

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

    This means that single predications cannot be performed using this method.
    Saving the model weights, and loading them into a non-pipelined version of
    the same model will allow single mini-batches (using gradient accumulation
    count = 1).

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
