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

from functools import partial
import weakref

from tensorflow.python.eager import def_function
from tensorflow.python.ipu.keras import model as ipu_model
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import tf_inspect


class PipelinedModel(ipu_model._IpuModelBase):  # pylint: disable=protected-access
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

    if not pipeline_depth:
      raise ValueError(
          "The pipeline_depth parameter must be specified.  Choose a "
          "pipeline_depth such that pipeline_depth * mini_batch_size is a "
          "good total batch size.  One step of the model will run "
          "pipeline_depth mini-batches through the model, accumulate the "
          "gradients of the errors, and then apply the accumulated gradients "
          "to the model weights once.")

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

    shard_count = max(kwargs.get("device_mapping", range(len(stages)))) + 1
    super(PipelinedModel, self).__init__(pipeline_depth, shard_count, **kwargs)

    self.pipeline_depth = pipeline_depth
    self.stages = stages

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
    return super(PipelinedModel, self).fit(x,
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
    return super(PipelinedModel, self).evaluate(x,
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
    return super(PipelinedModel, self).predict(x,
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
