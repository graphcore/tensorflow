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
"""
Pipelining operators
~~~~~~~~~~~~~~~~~~~~
"""
# Function captures are based on /tensorflow/python/ops/cond_v2.py

import inspect
from enum import Enum, IntEnum
from functools import reduce
from google.protobuf import json_format
import numpy as np

from tensorflow.compiler.plugin.poplar.driver import backend_config_pb2
from tensorflow.compiler.plugin.poplar.driver import pipeline_config_pb2
from tensorflow.compiler.plugin.poplar.driver import threestate_pb2
from tensorflow.compiler.plugin.poplar.ops import gen_functional_ops
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.eager import backprop
from tensorflow.python.ops import math_ops
from tensorflow.python.ipu import functional_ops
from tensorflow.python.ipu import internal_ops
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import scopes
from tensorflow.python.ipu.ops import op_util
from tensorflow.python.ipu import gradient_accumulation as ga
from tensorflow.python.ipu.eager import backprop as ipu_backprop
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer
from tensorflow.python.util import nest


class PipelineSchedule(IntEnum):
  """
  The PipelineSchedule describes how stages are interleaved on the IPUs
  servicing the pipeline.  The forward and backward passes of each stage
  will execute on the same IPUs.  So, in the core of the pipeline there is a
  choice as to whether to run the forward stages together, or the backward
  stages and the forward stages together.

  Attributes:
    Grouped: This groups the forward passes on multiple IPUs.  This requires
      more memory since activations need to be stored until the backward
      stages run together. However, since forward passes tend to be smaller
      than backward passes, Grouped tends to improve the speed of the
      execution, as different IPUs don't spend so much time waiting for each
      other.

    Interleaved: This schedules the backward passes whenever the forward
      passes have just generated some activations.  Consequently fewer
      activations are required to be stored between the forward and backward
      pipeline stages, so less memory is required.  However, since forward
      and backward stages tend to be very different in terms of execution
      cycles, the overall performance of the pipeline tends to be slower.

    Sequential: This is a debug mode, where the pipeline is scheduled in
      the same way as if it were a sharded model.
  """
  Grouped = 0
  Interleaved = 1
  Sequential = 2


class RecomputationMode(Enum):
  """When working with pipeline models for training, recomputation might be
  required in order to reduce the number of activations being stored on the
  device at any given time.

  This Enum class is used to control the recomputation implementation, with the
  following approaches supported:

  * `Auto`: automatically try and select the best recomputation strategy based
    on the provided model and pipeline schedule.
  * `RecomputeThenBackpropagate`: first recompute all the activations and then
    perform backpropagation. This mode allows for better code reuse as the
    corresponding forward propagation and the recomputation operations can share
    the exact same code. This recomputation mode is supported by
    `PipelineSchedule.Grouped` and `PipelineSchedule.Interleaved` pipeline
    schedules.
    This is the default recomputation mode for `PipelineSchedule.Grouped` and
    `PipelineSchedule.Interleaved` pipeline schedules.
  * `RecomputeAndBackpropagateInterleaved`: recompute and backpropagate
    operations are interleaved together. This mode can help reduce the maximum
    liveness compared to `RecomputeThenBackpropagate` as the backpropagation
    operations can be scheduled as soon as possible, however less code reuse
    will be possible. This recomputation mode is supported by
    `PipelineSchedule.Grouped` and `PipelineSchedule.Sequential` pipeline
    schedules.
    This is the default recomputation mode for the
    `PipelineSchedule.Sequential` pipeline schedule.
  """
  # pylint: disable=line-too-long
  Auto = backend_config_pb2.PoplarBackendConfig.CallConfig.PipelineConfig.Auto
  RecomputeThenBackpropagate = \
    backend_config_pb2.PoplarBackendConfig.CallConfig.PipelineConfig.Recompute_then_backpropagate
  RecomputeAndBackpropagateInterleaved = \
    backend_config_pb2.PoplarBackendConfig.CallConfig.PipelineConfig.RecomputationMode.Recompute_and_backpropagate_interleaved
  # pylint: enable=line-too-long


class OptimizerFunctionOutput:
  """
  A helper class used for returning a structured output from an
  optimizer_function in a pipeline.
  """
  def __init__(self,
               opt,
               loss,
               compute_gradients_args=None,
               compute_gradients_kwargs=None,
               apply_gradients_args=None,
               apply_gradients_kwargs=None,
               variables=None,
               tape=None,
               gradient_capture_context=None,
               captured_gradient_outfeed=None):
    """Creates an OptimizerFunctionOutput object.

    Args:
       opt: An instance of `optimizer.Optimizer` which is used to generate
         the back-propagation and the weight update pipeline stages.
       loss: The loss which is passed to the optimizer when calling
         `compute_gradients`.
       compute_gradients_args: Positional arguments (not including loss) which
         are passed to the `compute_gradients` function.
       compute_gradients_kwargs: Keyword arguments (not including loss) which
         are passed to the `compute_gradients` function.
       apply_gradients_args: Positional arguments (not including grads_and_vars)
         which are passed to the `apply_gradients` function.
       apply_gradients_kwargs: Keyword arguments (not including grads_and_vars)
         which are passed to the `apply_gradients` function.
       variables: A list or tuple of variables to compute gradients with respect
         to when `opt` is an instance of `OptimizerV2`.
       tape: A `GradientTape` for gradient computation when `opt` is an instance
         of `OptimizerV2`.
       gradient_capture_context: An 
       `ipu.eager.backprop.GradientCaptureContext` for accessing gradients
       captured by `ipu.ops.grad_util_ops.capture_upstream_gradients`.
       captured_gradient_outfeed: An `ipu.IPUOutfeedQueue` to which any captured
         gradients are pushed.
    """
    self.opt = opt
    self.loss = loss
    self.compute_gradients_args = \
      compute_gradients_args if compute_gradients_args else tuple()
    self.compute_gradients_kwargs = \
      compute_gradients_kwargs if compute_gradients_kwargs else dict()
    self.apply_gradients_args = \
      apply_gradients_args if apply_gradients_args else tuple()
    self.apply_gradients_kwargs = \
      apply_gradients_kwargs if apply_gradients_kwargs else dict()
    self.variables = variables if variables else tuple()
    self.tape = tape
    self.gradient_capture_context = gradient_capture_context
    self.captured_gradient_outfeed = captured_gradient_outfeed

  @property
  def opt(self):
    return self._opt

  @opt.setter
  def opt(self, value):
    if not isinstance(value, (optimizer.Optimizer, optimizer_v2.OptimizerV2)):
      raise TypeError(
          "OptimizerFunctionOutput.opt must be a TensorFlow Optimizer "
          "or Keras OptimizerV2 object.")
    self._opt = value

  @property
  def loss(self):
    return self._loss

  @loss.setter
  def loss(self, value):
    if not isinstance(value, ops.Tensor):
      raise TypeError(
          "OptimizerFunctionOutput.loss must be a TensorFlow Tensor object.")
    self._loss = value

  @property
  def compute_gradients_args(self):
    return self._compute_gradients_args

  @compute_gradients_args.setter
  def compute_gradients_args(self, value):
    if not isinstance(value, tuple):
      raise TypeError(
          "OptimizerFunctionOutput.compute_gradients_args must be a tuple.")

    if value and isinstance(self.opt, optimizer_v2.OptimizerV2):
      raise ValueError(
          "OptimizerFunctionOutput.compute_gradients_args may not be used "
          "with OptimizerV2 instances.")

    self._compute_gradients_args = value

  @property
  def compute_gradients_kwargs(self):
    return self._compute_gradients_kwargs

  @compute_gradients_kwargs.setter
  def compute_gradients_kwargs(self, value):
    if not isinstance(value, dict):
      raise TypeError(
          "OptimizerFunctionOutput.compute_gradients_kwargs must be a dict.")

    if value and isinstance(self.opt, optimizer_v2.OptimizerV2):
      raise ValueError(
          "OptimizerFunctionOutput.compute_gradients_kwargs may not be used "
          "with OptimizerV2 instances.")

    self._compute_gradients_kwargs = value

  @property
  def apply_gradients_args(self):
    return self._apply_gradients_args

  @apply_gradients_args.setter
  def apply_gradients_args(self, value):
    if not isinstance(value, tuple):
      raise TypeError(
          "OptimizerFunctionOutput.apply_gradients_args must be a tuple.")
    self._apply_gradients_args = value

  @property
  def apply_gradients_kwargs(self):
    return self._apply_gradients_kwargs

  @apply_gradients_kwargs.setter
  def apply_gradients_kwargs(self, value):
    if not isinstance(value, dict):
      raise TypeError(
          "OptimizerFunctionOutput.apply_gradients_kwargs must be a dict.")
    self._apply_gradients_kwargs = value

  @property
  def variables(self):
    return self._variables

  @variables.setter
  def variables(self, value):
    if not isinstance(value, (tuple, list)):
      raise TypeError(
          "OptimizerFunctionOutput.variables must be a tuple or list.")

    if value and not isinstance(self.opt, optimizer_v2.OptimizerV2):
      raise ValueError(
          "OptimizerFunctionOutput.variables may only be used with OptimizerV2."
      )

    if hasattr(self, '_tape') and self.tape:
      raise ValueError("OptimizerFunctionOutput.variables must be empty when "
                       "OptimizerFunctionOutput.tape is used.")

    self._variables = value

  @property
  def tape(self):
    return self._tape

  @tape.setter
  def tape(self, value):
    if value and not isinstance(value, backprop.GradientTape):
      raise TypeError("OptimizerFunctionOutput.tape must be a GradientTape.")

    if value and not isinstance(self.opt, optimizer_v2.OptimizerV2):
      raise ValueError(
          "OptimizerFunctionOutput.tape may only be used with OptimizerV2.")

    if value and hasattr(self, '_variables') and self.variables:
      raise ValueError("OptimizerFunctionOutput.tape may not be used when "
                       "OptimizerFunctionOutput.variables is nonempty.")

    if value and hasattr(self, '_gcc') and self.gradient_capture_context:
      raise ValueError("OptimizerFunctionOutput.tape may not be used when "
                       "OptimizerFunctionOutput.gradient_capture_context "
                       "is nonempty.")

    self._tape = value

  @property
  def gradient_capture_context(self):
    return self._gcc

  @gradient_capture_context.setter
  def gradient_capture_context(self, value):
    if value and not isinstance(value, ipu_backprop.GradientCaptureContext):
      raise TypeError(
          "OptimizerFunctionOutput.gradient_capture_context must be an "
          "ipu.eager.backprop.GradientCollectionContext.")

    if value and hasattr(self, '_tape') and self.tape:
      raise ValueError("OptimizerFunctionOutput.gradient_capture_context "
                       "may not be used when OptimizerFunctionOutput.tape "
                       "is nonempty.")

    self._gcc = value

  @property
  def captured_gradient_outfeed(self):
    return self._cg_outfeed

  @captured_gradient_outfeed.setter
  def captured_gradient_outfeed(self, value):
    if value and not isinstance(value, ipu_outfeed_queue.IPUOutfeedQueue):
      raise TypeError("OptimizerFunctionOutput.captured_gradient_outfeed must "
                      "be an instance of IPUOutfeedQueue.")

    valid_grad_src = False

    if value and self.tape and isinstance(self.tape,
                                          ipu_backprop.GradientCaptureTape):
      valid_grad_src = True

    if value and self.gradient_capture_context:
      valid_grad_src = True

    if value and not valid_grad_src:
      raise ValueError(
          "OptimizerFunctionOutput.captured_gradient_outfeed "
          "may not be used when neither an "
          "ipu.eager.backprop.GradientCollectionContext, nor an "
          "ipu.eager.backprop.GradientCollectionTape is also used.")

    self._cg_outfeed = value


class PipelineStageOptions:
  """
  A helper class which can be used to configure Poplar compilation options (such
  as `availableMemoryProportion` or `partialsType`) inside a pipeline forward,
  backward and weight update stage. This will override the global options set by
  the :ref:`convolution poplar options <convolutions.poplar_options>`,
  :ref:`matmul poplar options <matmuls.poplar_options>`, and
  :ref:`slice poplar options <slices.poplar_options>` in the
  :py:class:`~tensorflow.python.ipu.config.IPUConfig.`.
  """
  def __init__(self,
               convolution_options=None,
               matmul_options=None,
               slice_options=None):
    """Creates an PipelineStageOptions object.

    Args:
      convolution_options: If provided, a dictionary of Poplar option flags for
        all the convolution operations in the stage.
      matmul_options: If provided, a dictionary of Poplar option flags for
        all the matmul operations in the stage.
      slice_options: If provided, a dictionary of Poplar option flags for
        all the slice operations in the stage.
      loss: The loss which is passed to the optimizer.
    """

    convolution_options = convolution_options if convolution_options else {}
    if not isinstance(convolution_options, dict):
      raise TypeError(
          "PipelineStageOptions.convolution_options must be dictionary.")

    matmul_options = matmul_options if matmul_options else {}
    if not isinstance(matmul_options, dict):
      raise TypeError(
          "PipelineStageOptions.matmul_options must be dictionary.")

    slice_options = slice_options if slice_options else {}
    if not isinstance(slice_options, dict):
      raise TypeError("PipelineStageOptions.slice_options must be dictionary.")

    # Add the values from the dicts into the proto.
    self._proto = pipeline_config_pb2.PipelineStagePoplarConfig()
    for (option_name, value) in convolution_options.items():
      opt = self._proto.convolution_options.add()
      opt.option = option_name
      opt.value = value

    for (option_name, value) in matmul_options.items():
      opt = self._proto.matmul_options.add()
      opt.option = option_name
      opt.value = value

    for (option_name, value) in slice_options.items():
      opt = self._proto.slice_options.add()
      opt.option = option_name
      opt.value = value

  def get_proto(self):
    return self._proto


_ALL_DEVICES = -1


def pipeline(computational_stages,
             gradient_accumulation_count=None,
             gradient_accumulation_dtype=None,
             gradient_accumulation_for_captured_grads=True,
             repeat_count=1,
             batch_serialization_iterations=1,
             inputs=None,
             infeed_queue=None,
             outfeed_queue=None,
             optimizer_function=None,
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
             continuous_weight_updates=False,
             outfeed_loss=False,
             accumulate_outfeed=False,
             accumulate_outfeed_dtype=None,
             outfeed_mask=None,
             reduction_method=ga.GradientAccumulationReductionMethod.SUM,
             name=None):
  """
  Sets up a series of computational stages, where the outputs of one stage are
  the inputs to the next one. These stages are then executed in parallel across
  multiple IPUs. This approach can be used to split the model where layer(s)
  are executed on different IPUs.

  The first stage takes the `inputs` and the `infeed_queue` (if provided) as
  its inputs. If the `infeed_queue` is provided, it is automatically dequeued
  (similar to the ipu.loops API) therefore care needs to be taken to make sure
  the signature of the first pipeline stage matches both the arguments from
  `inputs` and the `infeed_queue`, otherwise an error is thrown.

  All tensors which are used in the pipeline which are not TensorFlow
  Variables need to be explicitly passed as inputs to the pipeline. If an
  input does not change its value during the execution of the pipeline op
  (for example hyperparameters such as learning rate), it needs to be passed
  as part of `inputs`. Alternatively, if these values change during execution
  (for example the model processes different batches of data) the input should
  be passed through the `infeed_queue`
  (see :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue`).

  When training a model, an optional `optimizer_function` function can be
  provided. This function takes all the outputs from the last computational
  stage as inputs, and returns an instance of `OptimizerFunctionOutput` that
  is used to generate the backwards pass of the model using the TensorFlow
  Optimizer API. This will internally create corresponding backpropagation
  pipeline stages for each pipeline stage and colocate them such that the
  activations and weights required for the gradient calculation and
  application stay on the device in order to minimise the number of copies
  between IPUs.

  Note that the gradients, which are calculated by the `compute_gradients`
  function, will be accumulated automatically during the execution of the
  pipeline, unless `continuous_weight_updates` is enabled.

  If the last computational stage has any outputs, then an `outfeed_queue`
  (see :class:`~tensorflow.python.ipu.ipu_outfeed_queue.IPUOutfeedQueue`)
  is required and all the outputs from the last computational stage are enqueued
  to the `outfeed_queue`.

  Note that pipelining supports the recomputation of activations for stateless
  ops during the backwards pass. This reduces the number of activations that
  will be stored on the device, saving memory at the expense of additional
  computation. To enable recomputation, use the
  :func:`tensorflow.python.ipu.utils.set_recomputation_options()` function when
  configuring the device.

  For example a simple inference network for the MNIST can be split across two
  IPUs:

  .. code-block:: python

    from tensorflow import keras

    # Create the dataset
    #...

    # Create the data queues from/to IPU.
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    # Create a pipelined model which is split accross two stages.
    def stage1(image):
      partial = keras.layers.Dense(256, activation=tf.nn.relu)(image)
      partial = keras.layers.Dense(128, activation=tf.nn.relu)(partial)
      return partial

    def stage2(partial):
      logits = keras.layers.Dense(10)(partial)
      probabilities = tf.nn.softmax(logits)
      classes = tf.argmax(input=logits, axis=1)
      return probabilities, classes

    def model():
      with variable_scope.variable_scope("vs", use_resource=True):
        pipeline_op = pipelining_ops.pipeline(
                          computational_stages=[stage1, stage2],
                          gradient_accumulation_count=250,
                          repeat_count=2,
                          inputs=[],
                          infeed_queue=infeed_queue,
                          outfeed_queue=outfeed_queue,
                          device_mapping=[3,1],
                          name="Pipeline")
      return pipeline_op

    with ops.device("/device:IPU:0"):
      compiled_model = ipu_compiler.compile(model, inputs=[])

    outfeed_op = outfeed_queue.dequeue()
    with tf.Session() as sess:
      result = sess.run(compiled_model)
      probabilities, classes = sess.run(outfeed_op)

  In this set up, the model is split across two IPUs. By default the first two
  layers would be executed on the first IPU and the third layer and the
  probabilities and classes on the second IPU but here `device_mapping` is
  used to override the default IPU allocation and instead the first two layers
  will be executed on the fourth IPU and the third layer and the probabilities
  and classed on the second IPU.

  This creates a pipeline of depth 250 (specified by the
  `gradient_accumulation_count`), which means each pipeline stage is executed
  250 times.

  This pipeline is then executed 2 times (specified by the `repeat_count`)
  The results of the pipeline (probabilities and classes) are returned to the
  host by the outfeed queue.

  We can also train this network by providing `optimizer_function`:

  .. code-block:: python

    from tensorflow import keras

    # Create the dataset
    #...

    # Create the data queues from/to IPU.
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    # Create a pipelined model which is split accross two stages.
    def stage1(lr, images, labels):
      partial = keras.layers.Dense(256, activation=tf.nn.relu)(images)
      partial = keras.layers.Dense(128, activation=tf.nn.relu)(partial)
      return lr, partial, labels

    def stage2(lr, partial, labels):
      logits = keras.layers.Dense(10)(partial)
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=labels, logits=logits)
      loss = tf.reduce_mean(cross_entropy)
      return lr, loss

    def optimizer_function(lr, loss):
      optimizer = tf.train.GradientDescentOptimizer(lr)
      return pipelining_ops.OptimizerFunctionOutput(optimizer, loss)

    def model(lr):
      with variable_scope.variable_scope("vs", use_resource=True):
        pipeline_op = pipelining_ops.pipeline(
                          computational_stages=[stage1, stage2],
                          gradient_accumulation_count=128,
                          repeat_count=10,
                          inputs=[lr],
                          infeed_queue=infeed_queue,
                          outfeed_queue=outfeed_queue,
                          optimizer_function=optimizer_function,
                          name="Pipeline")
      return pipeline_op

    with ops.device('cpu'):
      lr = tf.placeholder(np.float16, [])

    with ops.device("/device:IPU:0"):
      compiled_model = ipu_compiler.compile(model, inputs=[lr])

    outfeed_op = outfeed_queue.dequeue()
    with tf.Session() as sess:
      result = sess.run(compiled_model, {lr: 0.01})
      losses = sess.run(outfeed_op)

  Here the `tf.train.GradientDescentOptimizer` generates the pipeline stages
  which calculate the gradients and apply them to the weights. Note how the
  loss is returned to the host by the outfeed queue.

  If a model requires multiple computational pipeline stages to access the same
  `tf.Variable`, then all of these computational stages need to be placed on the
  same IPU using the `device_mapping` argument.

  Note that modifying `tf.Variable` values in a pipeline stage and/or during the
  gradient calculation will result in undefined behavior. These variables can
  only be modified by the `apply_gradients` member function of the applied
  Optimizer.

  Note that arguments marked with (EXPERIMENTAL) are under active development
  and might not provide representative performance.

  Args:
    computational_stages: a list of python functions, where each function
      represents a computational pipeline stage. The function takes the
      outputs of the previous pipeline state as its inputs.
    gradient_accumulation_count: the number of times each pipeline stage will
      be executed.
    gradient_accumulation_dtype: The data type used for the gradient
      accumulation buffer. One of:

      - `None`: Use an accumulator of the same type as the variable type.
      - A `DType`: Use this type for all the accumulators.
      - A callable that takes the variable and returns a `DType`: Allows
        specifying the accumulator type on a per-variable basis.

      The gradients passed to `Optimizer.apply_gradients` will have the dtype
      requested here. If that dtype is different from the variable dtype
      a cast is needed at some point to make them compatible. If you want
      to cast the gradients immediately, you can wrap your optimizer in the
      `MapGradientOptimizer` with a `tf.cast`.
    gradient_accumulation_for_captured_grads: If `True`, any captured gradients
      are accumulated before being passed to the optimizer's `apply_gradients`
      method (via its `captured_grads` keyword argument, if it exists).
      If `False`, the "raw", unaccumulated gradients are passed instead.
    repeat_count: the number of times the pipeline will be executed.
    batch_serialization_iterations: (EXPERIMENTAL) number of times a loop
      executes to compute a batch on each pipeline stage execution. Currently
      only supported with the `PipelineSchedule.Sequential`.
    inputs: arguments passed to the first pipeline stage.
    infeed_queue: optional IPUInfeedQueue, if passed, it is dequeued and
      passed as an input in the first pipeline stage.
    outfeed_queue: IPUOutfeedQueue, required if the last computational stage
      has any outputs. The outputs of these are enqueued to this queue and
      they can be accessed on the host.
    optimizer_function: optional Python function which takes the output of the
      last computational stage as parameters and returns an instance of
      `pipelining_ops.OptimizerFunctionOutput` in order to generate the
      back-propagation and weight-update parts of the model suitable for
      training.
    device_mapping: If provided, a list of length equal to the number of
      computational stages. An element at index `i` in the list represents which
      IPU the computational stage `computational_stages[i]` should reside on.
      This can be used to make sure computational stages which share
      `tf.Variable` are resident on the same IPU.
    pipeline_schedule: Which scheduling algorithm to use for pipeline
      lowering. Defaults to `PipelineSchedule.Grouped`.
    recomputation_mode: The recomputation mode to use for training pipeline
      models. Defaults to RecomputationMode.Auto. Only applies if recomputation
      is enabled. This must be done by using the
      :func:`tensorflow.python.ipu.utils.set_recomputation_options` function
      when configuring the device.
    forward_propagation_stages_poplar_options: If provided, a list of length
      equal to the number of computational stages. Each element is a
      PipelineStageOptions object which allows for fine grain control of the
      Poplar options for a given forward propagation computational stage.
    backward_propagation_stages_poplar_options: If provided, a list of length
      equal to the number of computational stages. Each element is a
      PipelineStageOptions object which allows for fine grained control of the
      Poplar options for a given backward propagation computational stage.
    weight_update_poplar_options: If provided, a PipelineStageOptions object
      which allows for fine grained control of the Poplar options for the
      weight update stage.
    offload_weight_update_variables: When enabled, any `tf.Variable` which is
      only used by the weight update of the pipeline (for example the
      accumulator variable when using the `tf.MomentumOptimizer`), will be
      stored in the remote memory. During the weight update this variable will
      be streamed onto the device and then streamed back to the remote memory
      after it has been updated. Requires the machine to be configured with
      support for `Poplar remote buffers`. Offloading variables into remote
      memory can reduce maximum memory liveness, but can also increase the
      computation time of the weight update.
      When set to `None` the variables will be placed in either in-processor or
      remote memory automatically based on the current best placement strategy.
      Note that this option has no effect for inference only pipelines.
    replicated_optimizer_state_sharding: If True, any `tf.Variable` which is
      offloaded (for example the accumulator variable when using the
      `tf.MomentumOptimizer`), will be partitioned across the replicas.
      This can exploit the additional bandwidth of the IPU-Links to improve
      overall throughput, however it might increase the code size and hence
      the model might need adjusting (for example the PopLibs option
      `availableMemoryProportion` might need to be changed).
      Note that this option has no effect for inference only pipelines.
    offload_activations: When enabled, all the activations for the batches which
      are not being executed by the pipeline stages at the given time are stored
      in remote memory. Requires the machine to be configured with support for
      `Poplar remote buffers`. Offloading activations into remote memory can
      reduce maximum memory liveness, but can also increase the computation time
      as activations have to be copied from/to the device(s).
      When set to `None`, the activations might be offloaded when beneficial.
    offload_gradient_accumulation_buffers: (EXPERIMENTAL) When enabled, all the
      gradient accumulation buffers are stored in remote memory. Offloading
      gradient accumulation buffers into remote memory can reduce maximum memory
      liveness, but can also increase the computation time as the buffers have
      to be copied to the device, updated and the copied off the device.
      Requires the machine to be configured with support for `Poplar remote
      buffers`.
      When set to `None`, the `offload_gradient_accumulation_buffers` might be
      offloaded when beneficial.
      Note that this option has no effect for inference only pipelines.
    replicated_weight_sharding: (EXPERIMENTAL) When enabled and running a
      replicated model, any `tf.Variable` used by the pipeline stage
      computations (excluding those only used by the weight update), will be
      partitioned across the replicas.
      Whenever the a partitioned `tf.Variable` is accessed, it will be first
      all-gathered across replicas to make sure each replica has access to the
      whole `tf.Variable`. This can exploit the additional bandwidth of the
      IPU-Links to improve overall throughput.
      When set to `None`, the activations might be offloaded when beneficial.
      This feature is enabled by default when the pipeline schedule is
      `PipelineSchedule.Sequential` and `batch_serialization_iterations > 1`,
      where this option can reduce the memory usage at the cost of extra
      communication.
    offload_weights: (EXPERIMENTAL) When enabled and
      `replicated_weight_sharding` is enabled, any `tf.Variable` which are
      partitioned across replicas will be stored in `Poplar remote buffers`.
      Offloading variables into remote memory can further reduce maximum memory
      liveness, but can also increase the computation time due to extra
      communication. When set to `None` the variables will be placed in either
      in-processor or remote memory automatically based on the current best
      placement strategy.
    continuous_weight_updates: ** CURRENTLY UNIMPLEMENTED ** When training,
      this option will apply the gradients to the resource variables
      immediately, rather than accumulating the gradients and applying them
      at the end of each execution of the pipeline.
    outfeed_loss: If True, the loss given by the `optimizer_function` will
      be enqueued on the outfeed, instead of the outputs from the last
      computational stage. Cannot be set when `outfeed_mask` is set.
    accumulate_outfeed: Data (loss or outputs) is normally enqueued immediately
      after the last computational stage inside the pipeline. If this option is
      True, the data will instead be accumulated and only enqueued once at the
      end of pipeline execution. To use this option, the provided
      `outfeed_queue` must be in the `IPUOutfeedMode` ALL mode
      (see :class:`~tensorflow.python.ipu.ipu_outfeed_queue.IPUOutfeedMode`).
    accumulate_outfeed_dtype: The data type used for the outfeed accumulation
      buffers. One of:

      - `None`: Use an accumulator of the same type as the variable type.
      - A `DType`: Use this type for all the accumulators.
      - A callable that takes the variable and returns a `DType`: Allows
        specifying the accumulator type on a per-variable basis.
    outfeed_mask: If set, a list of booleans of same length as the same number
      of outputs from the last computational stage. If `outfeed_mask[i]`
      evaluates to `False`, then the output at that index is enqueued to the
      outfeed queue, and if it is set to `True` it is not enqueued. Cannot be
      set when `outfeed_loss` is set. Can only be used when `optimizer_function`
      has been set.
    reduction_method: Reduction method to use when accumulating gradients.
      During the iterations in each optimizer step, the computed gradients can
      either be directly summed up or scaled such that we compute a mean of all
      gradients for each variable. Computing a mean avoids potential issues with
      overflow during accumulation especially when using float16, but gives
      smaller gradients and might require adjusting the learning-rate
      accordingly.
      Defaults to `GradientAccumulationReductionMethod.SUM`
      (see :class:`~tensorflow.python.ipu.gradient_accumulation.GradientAccumulationReductionMethod`)  # pylint: disable=line-too-long
    name: name of this pipeline.

  Returns:
    An `Operation` that executes the pipeline.

  """

  name = name if name else "pipeline"

  if gradient_accumulation_count is None:
    raise ValueError("gradient_accumulation_count must be specified.")

  if isinstance(gradient_accumulation_count, int):
    if gradient_accumulation_count < 0:
      raise ValueError("gradient_accumulation_count must be >= 0.")
    if gradient_accumulation_count > np.iinfo(np.int32).max:
      raise ValueError("gradient_accumulation_count must be < max int32.")

  gradient_accumulation_count = math_ops.cast(
      ops.convert_to_tensor(gradient_accumulation_count,
                            dtype_hint=dtypes.int32), dtypes.int32)

  if optimizer_function is not None:
    reduction_method = ga.GradientAccumulationReductionMethod.parse(
        reduction_method)

  if reduction_method != ga.GradientAccumulationReductionMethod.SUM and \
    batch_serialization_iterations != 1:
    raise ValueError('batch_serialization_iterations != 1 is only '
                     'supported when reduction_method is '
                     'GradientAccumulationReductionMethod.SUM')

  # Ensure inputs is a list, without casting inputs to a boolean. Casting
  # a tf.Tensor to a boolean will be interpreted as an operation in the
  # graph by Autograph.
  inputs = inputs if not isinstance(inputs, type(None)) else []
  inputs = functional_ops._convert_to_list(inputs)  # pylint: disable=protected-access
  inputs = ops.convert_n_to_tensor(inputs)

  if continuous_weight_updates:
    raise NotImplementedError(
        "Continuous weight updates are currently not supported.")

  for i, input in enumerate(inputs):
    if input.dtype == dtypes.resource:
      logging.warn("Passing tensor {} by value.".format(str(input)))
      inputs[i] = input.value()

  if pipeline_schedule is None:
    pipeline_schedule = (PipelineSchedule.Sequential
                         if batch_serialization_iterations > 1 else
                         PipelineSchedule.Grouped)

  if not isinstance(pipeline_schedule, PipelineSchedule):
    raise TypeError("The given pipeline_schedule is not a member of the "
                    "PipelineSchedule enumeration.")

  if (batch_serialization_iterations > 1
      and pipeline_schedule != PipelineSchedule.Sequential):
    raise NotImplementedError("Batch serialization is only supported with the "
                              "`Sequential` schedule.")

  if recomputation_mode is None:
    recomputation_mode = RecomputationMode.Auto

  if not isinstance(recomputation_mode, RecomputationMode):
    raise TypeError("The given recomputation_mode is not a member of the "
                    "RecomputationMode enumeration.")

  if device_mapping is None:
    device_mapping = [0] * len(
        computational_stages) if batch_serialization_iterations > 1 else list(
            range(len(computational_stages)))

  if not isinstance(computational_stages, (list, tuple)):
    raise TypeError(
        "computational_stages argument needs to be a list or a tuple.")

  if infeed_queue:
    if not isinstance(infeed_queue, ipu_infeed_queue.IPUInfeedQueue):
      raise TypeError("infeed_queue is not an instance of "
                      "ipu_infeed_queue.IPUInfeedQueue")

  if outfeed_queue:
    if not isinstance(outfeed_queue, ipu_outfeed_queue.IPUOutfeedQueue):
      raise TypeError("outfeed_queue is not an instance of "
                      "ipu_outfeed_queue.IPUOutfeedQueue")

  # We expect at least one stage.
  if len(computational_stages) < 2:
    raise ValueError("Pipeline requires at least two computational stages.")

  if not isinstance(device_mapping, (list, tuple)):
    raise TypeError("device_mapping argument needs to be a list or a tuple.")

  if len(device_mapping) != len(computational_stages):
    raise ValueError(
        "Each stage must be mapped to an IPU: %d mappings != %d stages" %
        (len(device_mapping), len(computational_stages)))

  # TODO(T18660) interleaved schedule does not support multiple stages on the
  # same IPU during training.
  flat_device_mapping = _to_flat_list(device_mapping)
  if pipeline_schedule == PipelineSchedule.Interleaved and \
      optimizer_function and \
      len(flat_device_mapping) != len(set(flat_device_mapping)):
    raise NotImplementedError(
        "The pipelining schedule 'Interleaved' does not currently support "
        "multiple pipeline stages on the same device for training graphs. "
        "Please use a different pipeline schedule.")

  if (pipeline_schedule == PipelineSchedule.Sequential
      and batch_serialization_iterations > 1
      and len(set(device_mapping)) != 1):
    raise NotImplementedError(
        "When using batch serialization, all the pipeline stages need to be "
        "mapped to a single IPU.")

  # Convert some of the binary options into three states.
  offload_activations = op_util.bool_to_three_state(offload_activations)
  offload_gradient_accumulation_buffers = op_util.bool_to_three_state(
      offload_gradient_accumulation_buffers)
  replicated_weight_sharding = op_util.bool_to_three_state(
      replicated_weight_sharding)
  offload_weights = op_util.bool_to_three_state(
      offload_weights, default=replicated_weight_sharding)

  # Function for setting up and validating the per stage Poplar options.
  def validate_stage_options_and_populate_proto(stages_poplar_options,
                                                proto_list, name):
    if stages_poplar_options is None:
      stages_poplar_options = [
          PipelineStageOptions() for i in range(len(computational_stages))
      ]

    if not isinstance(stages_poplar_options, (list, tuple)):
      raise TypeError(
          "%s must be a list or a tuple of PipelineStageOptions objects." %
          (name))

    if len(stages_poplar_options) != len(computational_stages):
      raise ValueError(
          "%s must be a list or a tuple of PipelineStageOptions objects of "
          "length %d (same number as the number of computational stages) but "
          "is %d." %
          (name, len(computational_stages), len(stages_poplar_options)))

    for stage_options in stages_poplar_options:
      if not isinstance(stage_options, PipelineStageOptions):
        raise TypeError(
            "Expected all elements of %s to be of type PipelineStageOptions, "
            "but got %s instead." % (name, str(stage_options)))

    for stage_options in stages_poplar_options:
      proto_list.append(stage_options.get_proto())

  pipeline_poplar_config = pipeline_config_pb2.PipelinePoplarConfig()

  validate_stage_options_and_populate_proto(
      forward_propagation_stages_poplar_options,
      pipeline_poplar_config.forward_stages,
      "forward_propagation_stages_poplar_options")

  if optimizer_function:
    validate_stage_options_and_populate_proto(
        backward_propagation_stages_poplar_options,
        pipeline_poplar_config.backward_stages,
        "backward_propagation_stages_poplar_options")

    if weight_update_poplar_options is None:
      weight_update_poplar_options = PipelineStageOptions()

    if not isinstance(weight_update_poplar_options, PipelineStageOptions):
      raise TypeError(
          "weight_update_poplar_options to be of type PipelineStageOptions, "
          "but got %s instead." % (str(weight_update_poplar_options)))

    pipeline_poplar_config.resource_update.CopyFrom(
        weight_update_poplar_options.get_proto())

  if outfeed_mask and not optimizer_function:
    raise ValueError(
        "An optimizer_function must be provided when outfeed_mask is set.")

  if outfeed_loss and not optimizer_function:
    raise ValueError(
        "An optimizer_function must be provided when outfeed_loss is True.")

  if outfeed_loss and outfeed_mask:
    raise ValueError(
        "Only one of `outfeed_loss` and `outfeed_mask` can be set.")

  if accumulate_outfeed:
    if not outfeed_queue:
      raise ValueError(
          "An outfeed_queue must be provided when accumulate_outfeed is True.")
    feed_mode = outfeed_queue._outfeed_mode  # pylint: disable=protected-access
    if feed_mode != ipu_outfeed_queue.IPUOutfeedMode.ALL:
      raise ValueError(
          "To accumulate the outfeed, it must be in IPUOutfeedMode ALL.")

  if optimizer_function is None and replicated_optimizer_state_sharding:
    logging.warn("replicated_optimizer_state_sharding will have no effect"
                 " since this pipeline is in inference.")
  if optimizer_function is None and offload_weight_update_variables != False:
    logging.warn("offload_weight_update_variables will have no effect"
                 " since this pipeline is in inference.")

  control_outputs = []

  def _pipeline(*args):
    captured_gradient_accumulation_count = args[0]
    outputs = args[1:]
    training = optimizer_function is not None

    outfeed_sinks = []

    def _acc_grad_scale(gac):
      one = np.float32(1.0)
      if reduction_method == ga.GradientAccumulationReductionMethod.SUM:
        accum_scale = one
        grad_scale = None
      elif reduction_method == ga.GradientAccumulationReductionMethod.MEAN:
        accum_scale = one
        grad_scale = one / math_ops.cast(gac, dtypes.float32)
      elif reduction_method == \
          ga.GradientAccumulationReductionMethod.RUNNING_MEAN:
        n = internal_ops.get_current_iteration_counter(
            lower_into_pipeline_stage=True)
        n = math_ops.cast(n, np.float32)
        inv_n_plus_1 = 1.0 / (n + 1)
        accum_scale = n * inv_n_plus_1
        grad_scale = inv_n_plus_1
      else:
        raise ValueError('reduction_method must be SUM, MEAN or RUNNING_MEAN')

      return accum_scale, grad_scale

    def _enqueue_or_accumulate(tensor_or_tensors):
      # Enqueue the outfeed data now or create accumulators for it
      # which will be enqueued later in the resource update.
      if not accumulate_outfeed:
        control_outputs.append(outfeed_queue.enqueue(tensor_or_tensors))
      else:
        tensors = functional_ops._convert_to_list(tensor_or_tensors)  # pylint: disable=protected-access
        for tensor in tensors:

          def create_accumulate(t):
            # Find the data type for the outfeed accumulator.
            dtype = op_util.get_accumulator_dtype(t, accumulate_outfeed_dtype)
            # Create a new t for the accumulator buffer.
            acc = gen_poputil_ops.gradient_accumulator_create_from_shape(
                shape=t.shape, output_type=dtype)
            acc = gen_poputil_ops.gradient_accumulator_add_with_scale(
                acc, t, 1.0)
            sink = gen_poputil_ops.gradient_accumulator_sink(acc)
            return sink

          outfeed_sinks.append(nest.map_structure(create_accumulate, tensor))

    def _extract_captured_grads(opt_fn):
      if not opt_fn:
        return None

      captured_grad_src = opt_fn.tape if opt_fn.tape and isinstance(
          opt_fn.tape, ipu_backprop.GradientCaptureTape) else None

      if not captured_grad_src:
        captured_grad_src = opt_fn.gradient_capture_context if \
          opt_fn.gradient_capture_context else None

      if not captured_grad_src:
        return None

      # Pull out any grads that have been captured by
      # ipu.ops.grad_util_ops.capture_upstream_gradients.
      return captured_grad_src.captured_gradients

    def _accumulate_captured_grads(captured_grads,
                                   gradient_accumulation_dtype,
                                   accum_scale=1.0,
                                   grad_scale=None):
      if not captured_grads:
        return None

      # Accumulate the captured grads.
      return op_util.accumulate_tagged_gradients(captured_grads,
                                                 gradient_accumulation_dtype,
                                                 accum_scale=accum_scale,
                                                 grad_scale=grad_scale)

    # Build all of the forward stage computations.
    for stage_id, stage in enumerate(computational_stages):
      stage_name = name + "_stage_" + str(stage_id)
      final_stage = stage_id == len(computational_stages) - 1
      stage_infeed_queue = infeed_queue if stage_id == 0 else None

      # Enqueue any tensor outputs from the final stage in inference, unless
      # we're accumulating them.
      if final_stage and not optimizer_function and not accumulate_outfeed:
        stage_outfeed_queue = outfeed_queue
      else:
        stage_outfeed_queue = None

      # Build the stage computation.
      outputs = _pipeline_stage(stage,
                                stage_id,
                                device_mapping[stage_id],
                                outputs,
                                training=training,
                                infeed_queue=stage_infeed_queue,
                                outfeed_queue=stage_outfeed_queue,
                                name=stage_name)

    if training:
      outputs = functional_ops._convert_to_list(outputs)  # pylint: disable=protected-access

      # Get the output from the optimizer function
      opt_fn = optimizer_function(*outputs)
      loss = opt_fn.loss
      opt = opt_fn.opt
      compute_gradients_args = opt_fn.compute_gradients_args
      compute_gradients_kwargs = opt_fn.compute_gradients_kwargs
      apply_gradients_args = opt_fn.apply_gradients_args
      apply_gradients_kwargs = opt_fn.apply_gradients_kwargs

      # Enqueue loss or any output tensors to the outfeed.
      if outfeed_loss:
        if not outfeed_queue:
          raise ValueError(
              "An outfeed_queue must be provided when outfeed_loss is True")
        _enqueue_or_accumulate(loss)
      elif outputs:
        # By default don't mask anything.
        local_mask = outfeed_mask if outfeed_mask else [False] * len(outputs)
        if len(outputs) != len(local_mask):
          raise ValueError(
              "The last computational stage has %d outputs, but the "
              "`outfeed_mask` has elements %d - these need to match in size." %
              (len(outputs), len(local_mask)))

        unmasked_outputs = [
            t for mask, t in zip(local_mask, outputs) if not mask
        ]

        if unmasked_outputs:
          if not outfeed_queue:
            raise ValueError(
                "The last computational stage has tensor outputs: %s, but no"
                " outfeed_queue has been provided." %
                (', '.join(str(t) for t in unmasked_outputs)))
          _enqueue_or_accumulate(unmasked_outputs)

      # Call the compute gradients function - this will be automatically put
      # into pipeline stages.
      if isinstance(opt, optimizer.Optimizer):
        grads_and_vars = opt.compute_gradients(loss, *compute_gradients_args,
                                               **compute_gradients_kwargs)
      else:
        if opt_fn.tape:
          trainable_vars = list(opt_fn.tape.watched_variables())
        elif opt_fn.variables:
          trainable_vars = list(opt_fn.variables)
        else:
          raise RuntimeError(
              "No variables to optimize. When using OptimzerV2, "
              "OptimizerFunctionOutput must contain a list of variables "
              "or a GradientTape.")
        grads = opt.get_gradients(loss, trainable_vars)
        if len(grads) != len(trainable_vars):
          raise RuntimeError("Inconsistent gradient and variable counts.")
        grads_and_vars = [(g, v) for g, v in zip(grads, trainable_vars)]

      if reduction_method == ga.GradientAccumulationReductionMethod.SUM:
        accum_scale, grad_scale = \
          _acc_grad_scale(captured_gradient_accumulation_count)
      else:
        deps = list(
            filter(lambda x: x is not None,
                   reduce(lambda l, p: l + [p[0]], grads_and_vars, [])))
        with ops.control_dependencies(deps):
          accum_scale, grad_scale = \
            _acc_grad_scale(captured_gradient_accumulation_count)

      # Insert gradient accumulation ops.
      accumulated_grads_and_vars = op_util.accumulate_gradients(
          grads_and_vars, gradient_accumulation_dtype, accum_scale, grad_scale)

      captured_grads = _extract_captured_grads(opt_fn)

      if gradient_accumulation_for_captured_grads:
        captured_grads = _accumulate_captured_grads(
            captured_grads,
            gradient_accumulation_dtype,
            accum_scale=accum_scale,
            grad_scale=grad_scale)

    elif not isinstance(outputs, ops.Operation) and accumulate_outfeed:
      # In inference, we never expect tensor outputs from the final stage,
      # because they would've been enqueued already inside the stage if we were
      # given an outfeed, unless we're accumulating.
      _enqueue_or_accumulate(outputs)

    # Create a resource update if we need to.
    if training or outfeed_sinks:
      # Create an explicit function call for the apply gradients - note that we
      # allow external caputres here.
      resource_update_ops = []

      def resource_update_(accumulation_count):
        gen_poputil_ops.gradient_accumulation_count(accumulation_count)
        if training:
          # If we have captured gradients, insert them into the
          # optimiser apply_gradients kwargs. Any optimiser that makes
          # use of these gradients can then retrieve them from kwargs.
          # The optimiser must explicitly take 'captured_grads' as a
          # keyword argument.
          kwargs = dict(apply_gradients_kwargs)
          _, _, kw, _ = inspect.getargspec(opt.__class__.apply_gradients)
          if captured_grads and (kw and 'captured_grads' in kw) or\
            (hasattr(opt, 'supports_captured_grads') and \
              opt.supports_captured_grads):
            kwargs['captured_grads'] = captured_grads

          apply_grads = opt.apply_gradients(accumulated_grads_and_vars,
                                            *apply_gradients_args, **kwargs)

          if apply_grads is not None:
            resource_update_ops.append(apply_grads)

          if captured_grads and opt_fn.captured_gradient_outfeed:
            opt_fn.captured_gradient_outfeed.enqueue(captured_grads)

        # Enqueue any accumulated outfeed data
        if outfeed_sinks:
          # Note: unpack if we're outfeeding loss.
          to_enqueue = outfeed_sinks[0] if outfeed_loss else outfeed_sinks
          enqueue = outfeed_queue.enqueue(to_enqueue)
          if enqueue is not None:
            resource_update_ops.append(enqueue)

      outputs = op_util.create_resource_update(
          resource_update_, name, resource_update_ops,
          offload_weight_update_variables, replicated_optimizer_state_sharding,
          captured_gradient_accumulation_count)

    if not isinstance(outputs, ops.Operation):
      if not outfeed_queue:
        raise ValueError(
            "The last computational stage has tensor outputs: %s, but no"
            " outfeed_queue has been provided." % (', '.join(
                str(t) for t in functional_ops._convert_to_list(outputs))))  # pylint: disable=protected-access

      else:
        raise ValueError(
            "Expected the pipeline resource update stage to output a "
            "tf.Operation, got %s instead." % (str(output)))

    control_outputs.append(outputs)

  with ops.name_scope(name) as scope:
    # pylint: disable=protected-access
    # Must pass the gradient_accumulation_count as the first input.
    inputs = [gradient_accumulation_count] + inputs
    try:
      func_graph, captured_args, _ = functional_ops._compile_function(
          _pipeline, inputs, scope, control_outputs)
    except functional_ops._InvalidCaptureException as e:
      raise ValueError(
          "Trying to capture the tensor %s which is not a resource. This tensor"
          " needs to be passed as either part of the `input` or `infeed_queue`"
          " of the pipeline." % (str(e)))
    # pylint: enable=protected-access

    # Create the pipeline and lower the function into XLA.
    with ops.control_dependencies(list(func_graph.control_captures)):
      output = gen_functional_ops.pipeline(
          captured_args,
          to_apply=util.create_new_tf_function(func_graph),
          Tout=func_graph.output_types,
          output_shapes=func_graph.output_shapes,
          batch_serialization_iterations=batch_serialization_iterations,
          repeat_count=repeat_count,
          schedule=int(pipeline_schedule),
          recomputation_mode=backend_config_pb2.PoplarBackendConfig.CallConfig.
          PipelineConfig.RecomputationMode.Name(recomputation_mode.value),
          pipeline_poplar_config=json_format.MessageToJson(
              pipeline_poplar_config),
          offload_activations=offload_activations,
          offload_gradient_accumulation_buffers=
          offload_gradient_accumulation_buffers,
          replicated_weight_sharding=replicated_weight_sharding,
          offload_weights=offload_weights)
    if not isinstance(output, ops.Operation):
      raise ValueError(
          "Expected the pipeline to output a tf.Operation, got %s instead." %
          (str(output)))

    return output


def _to_flat_list(xs):
  if isinstance(xs, list):
    return sum(list(map(_to_flat_list, xs)), [])
  if isinstance(xs, tuple):
    return _to_flat_list(list(xs))
  if isinstance(xs, range):
    return _to_flat_list(list(xs))
  return [xs]


def _pipeline_stage(func,
                    stage_id,
                    device_id,
                    args,
                    training,
                    infeed_queue=None,
                    outfeed_queue=None,
                    parallel_stage=False,
                    name=None):
  """Internal function for compiling a pipeline stage. This should not be called
  directly and doing so will result in undefined behaviour.

  Creates a pipeline stage.

  Args:
    func: function which will be executed as a stage.
    stage_id: Stage number.
    device_id: IPU the stage will be mapped to.
    args: arguments to the function.
    infeed_queue: optional IPUInfeedQueue, if passed, it is dequeued as part of
      this function.
    outfeed_queue: optional IPUOutfeedQueue, if passed, it is enqueued as part
      of this function.
    name: name of this pipeline sage.

  Returns:
    The values after execting func(args), or the control dependency if
    outfeed_queue is not None.

  """
  name = name if name else "pipeline_stage"

  if isinstance(func, list):
    if not isinstance(device_id, list):
      return ValueError(
          "When the pipeline stage is a list of functions, the device mapping"
          " must also be a list of device IDs.")

    if len(func) != len(device_id):
      return ValueError(
          "Pipeline stage list and device_id list must be the same length.")

    if parallel_stage:
      return NotImplementedError(
          "Multiple nested levels of pipeline stages is not supported.")

    def par_stages_f(*args):
      funcs = enumerate(zip(func, device_id))

      def lower_par_stage(fd):
        i, (f, d) = fd
        return _pipeline_stage(f,
                               i,
                               d,
                               args,
                               training,
                               parallel_stage=True,
                               name=name + str(i))

      return _to_flat_list(list(map(lower_par_stage, funcs)))

    return _pipeline_stage(par_stages_f,
                           stage_id,
                           _ALL_DEVICES,
                           args,
                           training,
                           infeed_queue=infeed_queue,
                           outfeed_queue=outfeed_queue,
                           name=name)

  args = functional_ops._convert_to_list(args)  # pylint: disable=protected-access

  func_to_compile = func
  control_outputs = []
  # If we have an infeed, then we wrap the function in another function which
  # dequeues the infeed.
  if infeed_queue:

    def infeed_func_wrapper(*args):
      args = functional_ops._convert_to_list(args)  # pylint: disable=protected-access
      dequeue_ops = functional_ops._convert_to_list(infeed_queue._dequeue())  # pylint: disable=protected-access
      # Deal with the dequeue depending on whether it's a list or dict.
      if (len(dequeue_ops) == 1 and isinstance(dequeue_ops[0], dict)
          and all(isinstance(key, str) for key in dequeue_ops[0])):
        # Only dicts where all keys are strings can be used as kwargs.
        kwargs = dequeue_ops[0]
        return func(*(args), **kwargs)
      return func(*(args + dequeue_ops))

    func_to_compile = infeed_func_wrapper

  # If we have an outfeed, then we wrap the function in another function which
  # enqueues the outfeed.
  if outfeed_queue:
    func = func_to_compile

    def outfeed_func_wrapper(*args, **kwargs):
      outputs = func(*args, **kwargs)
      # Check if there are output tensors - if there are then enqueue them.
      if not isinstance(outputs, ops.Operation):
        if not isinstance(outputs, (dict, ops.Tensor)):
          outputs = functional_ops._convert_to_list(outputs)  # pylint: disable=protected-access
        outputs = outfeed_queue.enqueue(outputs)
      control_outputs.append(outputs)

    func_to_compile = outfeed_func_wrapper

  def gradient_override_wrapper(*args, **kwargs):
    with op_util.gradient_override_scope(training):
      return func_to_compile(*args, **kwargs)

  with ops.name_scope(name) as scope:
    # pylint: disable=protected-access
    try:
      func_graph, captured_args, constant_outputs = \
        functional_ops._compile_function(
          gradient_override_wrapper, args, scope, control_outputs)
    except functional_ops._InvalidCaptureException as e:
      raise ValueError(
          "Trying to capture the tensor %s which is not a resource. This tensor"
          " needs to be passed as either part of the `input` or `infeed_queue`"
          " of the pipeline." % (str(e)))
    # pylint: enable=protected-access

    # Create the pipeline stage and lower the function into XLA.
    with ops.control_dependencies(list(func_graph.control_captures)):
      with scopes.ipu_shard(device_id):
        outputs = gen_functional_ops.pipeline_stage(
            captured_args,
            to_apply=util.create_new_tf_function(func_graph),
            Tout=func_graph.output_types,
            output_shapes=func_graph.output_shapes,
            stage_id=stage_id)
    if isinstance(outputs, ops.Operation):
      return outputs

    outputs = functional_ops._replace_outputs(outputs, constant_outputs)  # pylint: disable=protected-access
    return functional_ops._pack_sequence_as(  # pylint: disable=protected-access
        func_graph.structured_outputs, outputs)


def recomputation_checkpoint(tensors, name=None):
  """Operation for checkpointing values in a computational pipeline stage.
  When recomputation is enabled, these values will not be recomputed and they
  will be stored in memory instead.

  This operation can reduce memory liveness peaks when using recomputation if
  there are too many activations which need to be recomputed before the
  backpropagation operations can be executed.

  This operation should be used with the
  `RecomputationMode.RecomputeAndBackpropagateInterleaved` pipelining
  recomputation mode.
  Note that this operation has no effect when used with the
  `RecomputationMode.RecomputeThenBackpropagate` pipelining
  recomputation mode.

  Args:
    tensors: A tensor or a structure of tensors which should be checkpointed.
    name: name of this operation.

  Returns:
    A tensor or a structure of tensors which matches shape and type of
    tensors.
  """
  inputs = nest.flatten(tensors, expand_composites=True)
  outputs = [
      gen_functional_ops.recomputation_checkpoint(x, name=name) for x in inputs
  ]
  return nest.pack_sequence_as(tensors, outputs, expand_composites=True)
