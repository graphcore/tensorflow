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
import weakref
import math

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ipu.keras import model as ipu_model
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import nest


class PipelinedModel:
  def __init__(self, *args, **kwargs):
    raise NotImplementedError(
        "PipelinedModel is no longer used. "
        "For pipelining of ipu.keras.Sequential models, "
        "use ipu.keras.SequentialPipelineModel and for ipu.keras.Model "
        "use ipu.keras.PipelineModel.")


class SequentialPipelineModel(ipu_model._IpuModelBase):  # pylint: disable=protected-access
  """Keras Model for encapsulating a pipeline of stages to be run in parallel
  on an IPU system.

  A pipelined model will execute multiple sections (stages) of a model on more
  than one IPU at the same time, by pipelining mini-batches of data through
  the stages.

  It encapsulates the ipu.pipelining_ops.pipeline operation and the associated
  InFeed and OutFeed queues into a class which resembles the Keras Model class
  and provides the `fit` API for training the model.

  The different stages are specified, similarly to the Keras Sequential model,
  as a list in the constructor.  With the PipelineModel class the list of
  layers becomes a list of lists of layers, where each list contains the layers
  for a particular stage.

  The pipeline depth argument describes the number of mini-batches which are
  sent through the pipeline in a single operation of the pipeline.  The
  effective batch size is therefore the mini-batch size multipled by the
  pipeline depth.

  There are some limitations with the PipelineModel compared to the standard
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
      m = ipu.keras.SequentialPipelineModel([
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
               batch_serialization_iterations=1,
               device_mapping=None,
               pipeline_schedule=None,
               forward_propagation_stages_poplar_options=None,
               backward_propagation_stages_poplar_options=None,
               weight_update_poplar_options=None,
               replicated_optimizer_state_sharding=False,
               offload_activations=None,
               offload_gradient_accumulation_buffers=None,
               replicated_weight_sharding=None,
               offload_weights=None,
               **kwargs):
    """
    Creates a pipelined model.

    Args:
        stages: A python list of lists of Layers.
        gradient_accumulation_count: The number of mini-batches processed by
            the pipeline on each iteration.
        batch_serialization_iterations: number of times a loop executes to
            compute a batch on each pipeline stage execution. Currently only
            supported with the `PipelineSchedule.Sequential`.
        device_mapping: If provided, a list of length equal to the number of
            computational stages. An element at index `i` in the list
            represents which IPU the computational stage
            `computational_stages[i]` should reside on.
            This can be used to make sure computational stages which share
            tf.Variable`s are resident on the same IPU.
        pipeline_schedule: Which scheduling algorithm to use for pipeline
            lowering. Defaults to `PipelineSchedule.Grouped`.
        forward_propagation_stages_poplar_options: If provided, a list of
            length equal to the number of computational stages. Each element is
            a PipelineStageOptions object which allows for fine grain control
            of the Poplar options for a given forward propagation computational
            stage.
        backward_propagation_stages_poplar_options: If provided, a list of
            length equal to the number of computational stages. Each element is
            a PipelineStageOptions object which allows for fine grained control
            of the Poplar options for a given backward propagation
            computational stage.
        weight_update_poplar_options: If provided, a PipelineStageOptions
            object which allows for fine grained control of the Poplar options
            for the weight update stage.
        replicated_optimizer_state_sharding: If True, any `tf.Variable` which
            is offloaded (for example the accumulator variable when using the
            `tf.MomentumOptimizer`), will be partitioned across the replicas.
            This can exploit the additional bandwidth of the IPU-Links to
            improve overall throughput.
            Note that this option has no effect for inference only pipelines.
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
        offload_gradient_accumulation_buffers: When enabled, all the gradient
            accumulation buffers are stored in remote memory. Offloading
            gradient accumulation buffers into remote memory can reduce maximum
            memory liveness, but can also increase the computation time as the
            buffers have to be copied to the device, updated and the copied off
            the device. Requires the machine to be configured with support for
            `Poplar remote buffers`.
            When set to `None`, the `offload_gradient_accumulation_buffers`
            might be offloaded when beneficial.
            Note that this option has no effect for inference only pipelines.
        replicated_weight_sharding: When enabled and running a replicated
            model, any `tf.Variable`s used by the pipeline stage computations
            (excluding those only used by the weight update), will be
            partitioned across the replicas. Whenever the a partitioned
            `tf.Variable` is accessed, it will be first all-gathered across
            replicas to make sure each replica has access to the whole
            `tf.Variable`. This can exploit the additional bandwidth of the
            IPU-Links to improve overall throughput. When set to `None`, the
            activations might be offloaded when beneficial. This feature is
            enabled by default when the pipeline schedule is
            `PipelineSchedule.Sequential` and
            `batch_serialization_iterations > 1`, where this option can reduce
            the memory usage at the cost of extra communication.
        offload_weights: When enabled and `replicated_weight_sharding` is
            enabled, any `tf.Variable` which are partitioned across replicas
            will be stored in `Poplar remote buffers`.  Offloading variables
            into remote memory can further reduce maximum memory liveness, but
            can also increase the computation time due to extra communication.
            When set to `None` the variables will be placed in either
            in-processor or remote memory automatically based on the current
            best placement strategy.
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

    shard_count = max(
        device_mapping if device_mapping else range(len(stages))) + 1
    accumulation_count = gradient_accumulation_count * \
      batch_serialization_iterations
    super().__init__(accumulation_count, shard_count, **kwargs)

    self.gradient_accumulation_count = gradient_accumulation_count
    self.stages = stages

    # Store additional pipeline params.
    self.batch_serialization_iterations = batch_serialization_iterations
    self.device_mapping = device_mapping
    self.pipeline_schedule = pipeline_schedule
    self.forward_propagation_stages_poplar_options = \
      forward_propagation_stages_poplar_options
    self.backward_propagation_stages_poplar_options = \
      backward_propagation_stages_poplar_options
    self.weight_update_poplar_options = weight_update_poplar_options
    self.replicated_optimizer_state_sharding = \
      replicated_optimizer_state_sharding
    self.offload_activations = offload_activations
    self.offload_gradient_accumulation_buffers = \
      offload_gradient_accumulation_buffers
    self.replicated_weight_sharding = replicated_weight_sharding
    self.offload_weights = offload_weights

  def build(self, input_shape):
    s = input_shape
    for l in self.layers:
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
    This provides the same functionality as the Keras Model ``compile``
    method.

    Certain features are not supported by the IPU PipelineModel:
    - sample_weight_mode
    - weighted_metrics
    - target_tensors
    """
    return super().compile(optimizer, loss, metrics, loss_weights, **kwargs)

  def _get_internal_run_loop(self):
    if not self.internal_loop_fn:
      fn = partial(SequentialPipelineModel._internal_run_loop, self)
      self.internal_loop_fn = def_function.function(fn,
                                                    autograph=False,
                                                    experimental_compile=True)
    return self.internal_loop_fn

  def _internal_run_loop(self, infeed_queue, outfeed_queue, repeat_count,
                         mode):
    training = mode == ModeKeys.TRAIN

    # Plain functions to build a stage
    def call_inference_stage(stage_id, inputs):
      # Record the inputs of the first stage
      if stage_id == 0 and not self.inputs:
        self._set_input_attrs(inputs)

      x = inputs
      for l in self.stages[stage_id]:
        kwargs = {}
        argspec = tf_inspect.getfullargspec(l.call).args
        if 'training' in argspec:
          kwargs['training'] = training
        x = l(x, **kwargs)

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

    opt = optimizer_function if training else None

    pipeline = pipelining_ops.pipeline(
        stages,
        gradient_accumulation_count=self.gradient_accumulation_count,
        repeat_count=repeat_count,
        inputs=[],
        infeed_queue=infeed_queue,
        outfeed_queue=outfeed_queue,
        optimizer_function=opt,
        device_mapping=self.device_mapping,
        pipeline_schedule=self.pipeline_schedule,
        forward_propagation_stages_poplar_options=self.
        forward_propagation_stages_poplar_options,
        backward_propagation_stages_poplar_options=self.
        backward_propagation_stages_poplar_options,
        weight_update_poplar_options=self.weight_update_poplar_options,
        replicated_optimizer_state_sharding=self.
        replicated_optimizer_state_sharding,
        offload_activations=self.offload_activations,
        offload_gradient_accumulation_buffers=self.
        offload_gradient_accumulation_buffers,
        replicated_weight_sharding=self.replicated_weight_sharding,
        offload_weights=self.offload_weights,
        name=self.name,
        **self.args)

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
               **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
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
    return super().evaluate(x,
                            y,
                            batch_size=batch_size,
                            verbose=verbose,
                            steps=steps,
                            callbacks=callbacks,
                            steps_per_run=steps_per_run,
                            **kwargs)

  def predict(self,
              x,
              *,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              steps_per_run=None,
              **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
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
    return super().predict(x,
                           batch_size=batch_size,
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


class PipelineStage(object):
  def __init__(self, stage):
    self._stage = stage

  def __enter__(self):
    if self._stage < 0:
      raise ValueError("%d is not a valid pipeline stage.")

    strategy = distribution_strategy_context.get_strategy()
    if not strategy:
      raise RuntimeError("PipelineStage may only be used from "
                         "within an IPUStrategy context.")

    if hasattr(strategy, "_pipeline_stage"):
      raise RuntimeError("Pipeline stages must not be nested.")

    strategy._pipeline_stage = self._stage  # pylint: disable=protected-access

    return self

  def __exit__(self, exception_type, value, traceback):
    strategy = distribution_strategy_context.get_strategy()
    assert strategy and hasattr(strategy, "_pipeline_stage")

    delattr(strategy, "_pipeline_stage")


class PipelineModel(ipu_model.Model):
  """Keras Model for encapsulating a pipeline of stages to be run in
  parallel on an IPU system.

  A pipelined model will execute multiple sections (stages) of a model on more
  than one IPU at the same time, by pipelining mini-batches of data through
  the stages.

  The different stages are specified when defining the graph structure via use
  of the PipelineStage context manager, as follows for a simple two
  stage pipeline:

  .. code-block:: python

    strategy = ipu.ipu_strategy.IPUStrategy()
    input_layer = Input(2)
    with strategy.scope():
      with PipelineStage(0):
        x = Dense(4)(input_layer)

      with PipelineStage(1):
        x = Dense(4)(x)

  The pipeline depth argument describes the number of mini-batches which are
  sent through the pipeline in a single operation of the pipeline.  The
  effective batch size is therefore the mini-batch size multipled by the
  pipeline depth.

  There are some limitations with the PipelineModel compared to the
  standard Keras Model.

  - Keras V1 optimizers cannot be used.
  - Loss weightings can only be specified as a list, not a callable.
  - Weighted metrics, target tensors and sample weight mode are not supported.
  - Validation cannot be performed as part of the `fit` loop.
  - The model cannot be called using the __call__() interface.

  Example:

  .. code-block:: python

    dataset = ...

    strategy = ipu.ipu_strategy.IPUStrategy()
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

      model.fit(training_data, epochs=2, steps_per_epoch=128)
  """
  def __init__(self,
               *args,
               gradient_accumulation_count,
               batch_serialization_iterations=1,
               device_mapping=None,
               pipeline_schedule=None,
               forward_propagation_stages_poplar_options=None,
               backward_propagation_stages_poplar_options=None,
               weight_update_poplar_options=None,
               replicated_optimizer_state_sharding=False,
               offload_activations=None,
               offload_gradient_accumulation_buffers=None,
               replicated_weight_sharding=None,
               offload_weights=None,
               **kwargs):
    """
    Creates a pipelined model (defined via the Keras Functional API).

    Needs to pass in ``inputs`` and ``outputs`` as either arguments or
    keyword arguments.

    Args:
        gradient_accumulation_count: The number of mini-batches processed by
            the pipeline on each iteration.
        batch_serialization_iterations: number of times a loop executes to
        compute a batch on each pipeline stage execution. Currently only
        supported with the `PipelineSchedule.Sequential`.
        device_mapping: If provided, a list of length equal to the number of
            computational stages. An element at index `i` in the list
            represents which IPU the computational stage
            `computational_stages[i]` should reside on.
            This can be used to make sure computational stages which share
            tf.Variable`s are resident on the same IPU.
        pipeline_schedule: Which scheduling algorithm to use for pipeline
            lowering. Defaults to `PipelineSchedule.Grouped`.
        forward_propagation_stages_poplar_options: If provided, a list of
            length equal to the number of computational stages. Each element is
            a PipelineStageOptions object which allows for fine grain control of
            the Poplar options for a given forward propagation computational
            stage.
        backward_propagation_stages_poplar_options: If provided, a list of
            length equal to the number of computational stages. Each element is
            a PipelineStageOptions object which allows for fine grained control
            of the Poplar options for a given backward propagation computational
            stage.
        weight_update_poplar_options: If provided, a PipelineStageOptions
            object which allows for fine grained control of the Poplar options
            for the weight update stage.
        replicated_optimizer_state_sharding: If True, any `tf.Variable` which
            is offloaded (for example the accumulator variable when using the
            `tf.MomentumOptimizer`), will be partitioned across the replicas.
            This can exploit the additional bandwidth of the IPU-Links to
            improve overall throughput.
            Note that this option has no effect for inference only pipelines.
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
        offload_gradient_accumulation_buffers: When enabled, all the gradient
            accumulation buffers are stored in remote memory. Offloading
            gradient accumulation buffers into remote memory can reduce maximum
            memory liveness, but can also increase the computation time as the
            buffers have to be copied to the device, updated and the copied off
            the device. Requires the machine to be configured with support for
            `Poplar remote buffers`.
            When set to `None`, the `offload_gradient_accumulation_buffers`
            might be offloaded when beneficial.
            Note that this option has no effect for inference only pipelines.
        replicated_weight_sharding: When enabled and running a replicated
            model, any `tf.Variable`s used by the pipeline stage computations
            (excluding those only used by the weight update), will be
            partitioned across the replicas. Whenever the a partitioned
            `tf.Variable` is accessed, it will be first all-gathered across
            replicas to make sure each replica has access to the whole
            `tf.Variable`. This can exploit the additional bandwidth of the
            IPU-Links to improve overall throughput. When set to `None`, the
            activations might be offloaded when beneficial. This feature is
            enabled by default when the pipeline schedule is
            `PipelineSchedule.Sequential` and
            `batch_serialization_iterations > 1`, where this option can reduce
            the memory usage at the cost of extra communication.
        offload_weights: When enabled and `replicated_weight_sharding` is
            enabled, any `tf.Variable` which are partitioned across replicas
            will be stored in `Poplar remote buffers`.  Offloading variables
            into remote memory can further reduce maximum memory liveness, but
            can also increase the computation time due to extra communication.
            When set to `None` the variables will be placed in either
            in-processor or remote memory automatically based on the current
            best placement strategy.
        name: Optional name for the pipeline operation.
    """
    accumulation_count = gradient_accumulation_count * \
      batch_serialization_iterations
    super().__init__(*args, accumulation_count=accumulation_count, **kwargs)

    # Assign stages to nodes.
    self.stages = self._assign_node_stages()
    self._stage_node_ids = self._get_per_stage_node_ids()

    # Compute shard count.
    shard_count = max(
        device_mapping if device_mapping else range(len(self.stages))) + 1

    # Round the shard count to the next power of two
    self.shard_count = 2**int(math.log2(shard_count))

    self.gradient_accumulation_count = gradient_accumulation_count

    # Store additional pipeline params.
    self.batch_serialization_iterations = batch_serialization_iterations
    self.device_mapping = device_mapping
    self.pipeline_schedule = pipeline_schedule
    self.forward_propagation_stages_poplar_options = \
      forward_propagation_stages_poplar_options
    self.backward_propagation_stages_poplar_options = \
      backward_propagation_stages_poplar_options
    self.weight_update_poplar_options = weight_update_poplar_options
    self.replicated_optimizer_state_sharding = \
      replicated_optimizer_state_sharding
    self.offload_activations = offload_activations
    self.offload_gradient_accumulation_buffers = \
      offload_gradient_accumulation_buffers
    self.replicated_weight_sharding = replicated_weight_sharding
    self.offload_weights = offload_weights

  def _assign_node_stages(self):
    # Get stages for each layer - used for mapping to nodes below.
    stages = dict()
    for n, layer in enumerate(self._layers):
      # Input layers need not have a pipeline stage assigned -
      # we don't have to wait for a computation to complete to
      # access the "result" of an input layer.
      if isinstance(layer, InputLayer):
        continue

      # Verify that a pipeline stage has been assigned.
      if not hasattr(layer, "_pipeline_stage"):
        raise ValueError(
            "All layers of a pipelined model must have an associated "
            "pipeline stage")

      layer_id = id(layer)
      stages[layer_id] = layer._pipeline_stage  # pylint: disable=protected-access

    # Get the stage for each node from it's corresponding layer (above).
    prev_stage = None
    for n, node in enumerate(self._post_order_node_execution):
      layer = node.outbound_layer
      if isinstance(layer, InputLayer):
        continue

      # Assign the node a pipeline stage.
      layer_id = id(layer)
      assert layer_id in stages
      node._pipeline_stage = stages[layer_id]  # pylint: disable=protected-access

      # Verify stage order.
      stage = node._pipeline_stage  # pylint: disable=protected-access
      if prev_stage and prev_stage > stage:
        raise ValueError(
            "Post order execution node #%d has stage %d, but previous node had "
            "stage %d. The pipeline stage for a node must be greater than or "
            "equal to it's predecessor in the order of execution." %
            (n, stage, prev_stage))
      prev_stage = stage
    stage_node_ids = list(set(stages.values()))
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

  def _internal_run_loop(self, infeed_queue, outfeed_queue, repeat_count,
                         mode):
    training = mode == ModeKeys.TRAIN

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
        return self._get_output_tensors(tensor_dict)  # pylint: disable=protected-access
      return list(tensor_dict.values()) + targets

    def inference_body(stage_id, *args):
      return main_body(stage_id, *args)

    def training_body(stage_id, *args):
      x = main_body(stage_id, *args)
      if stage_id == self.stages[-1]:
        self._set_output_attrs(x)
        targets = args[-len(self.outputs)]
        return self._add_loss(targets)
      return x

    def optimizer_function(loss, *_):
      if not self.trainable_weights:
        raise ValueError("Model must have at least one trainable parameter.")

      opt = self._get_optimizer()
      return pipelining_ops.OptimizerFunctionOutput(opt, loss)

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
        repeat_count=repeat_count,
        inputs=[],
        infeed_queue=infeed_queue,
        outfeed_queue=outfeed_queue,
        optimizer_function=opt,
        device_mapping=self.device_mapping,
        pipeline_schedule=self.pipeline_schedule,
        forward_propagation_stages_poplar_options=self.
        forward_propagation_stages_poplar_options,
        backward_propagation_stages_poplar_options=self.
        backward_propagation_stages_poplar_options,
        weight_update_poplar_options=self.weight_update_poplar_options,
        replicated_optimizer_state_sharding=self.
        replicated_optimizer_state_sharding,
        offload_activations=self.offload_activations,
        offload_gradient_accumulation_buffers=self.
        offload_gradient_accumulation_buffers,
        replicated_weight_sharding=self.replicated_weight_sharding,
        offload_weights=self.offload_weights,
        name=self.name,
        **self.args)

    return pipeline.outputs

  def _get_internal_run_loop(self):
    if not self.internal_loop_fn:
      fn = partial(PipelineModel._internal_run_loop, self)
      self.internal_loop_fn = def_function.function(fn,
                                                    autograph=False,
                                                    experimental_compile=True)
    return self.internal_loop_fn

  @trackable.no_automatic_dependency_tracking
  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              **kwargs):
    """
    This provides the same functionality as the Keras Model ``compile`` method.

    Certain features are not supported by the IPU Pipelined Model:
    - sample_weight_mode
    - weighted_metrics
    - target_tensors
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
    return super().evaluate(x,
                            y,
                            batch_size=batch_size,
                            verbose=verbose,
                            steps=steps,
                            callbacks=callbacks,
                            steps_per_run=steps_per_run,
                            **kwargs)

  def predict(self,
              x,
              *,
              batch_size=None,
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
    return super().predict(x,
                           batch_size=batch_size,
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
