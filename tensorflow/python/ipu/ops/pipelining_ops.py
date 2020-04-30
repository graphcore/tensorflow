# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
~~~~~~~~~~~~~~~~~~~~~~
"""
# Function captures are based on /tensorflow/python/ops/cond_v2.py

from enum import IntEnum

from google.protobuf import json_format

from tensorflow.compiler.plugin.poplar.driver import pipeline_config_pb2
from tensorflow.compiler.plugin.poplar.ops import gen_functional_ops
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.ipu import functional_ops
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import scopes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer


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


class OptimizerFunctionOutput:
  """
  A helper class used for returning a structured output from an
  optimizer_function in a pipeline.
  """
  def __init__(self, opt, loss):
    """Creates an OptimizerFunctionOutput object.

    Args:
       opt: An instance of `optimizer.Optimizer` which is used to generate
         the back-propagation and the weight update pipeline stages.
       loss: The loss which is passed to the optimizer.
    """
    self.opt = opt
    self.loss = loss

  @property
  def opt(self):
    return self._opt

  @opt.setter
  def opt(self, value):
    if not isinstance(value, optimizer.Optimizer):
      raise TypeError(
          "OptimizerFunctionOutput.opt must be a TensorFlow Optimizer "
          "object.")
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


class PipelineStageOptions:
  """
  A helper class which can be used to configure Poplar compilation options (such
  as 'availableMemoryProportion') inside a pipeline forward, backward and weight
  update stage. This will override the global options set by
  `ipu.utils.set_convolution_options` and `ipu.utils.set_matmul_options`.
  """
  def __init__(self, convolution_options=None, matmul_options=None):
    """Creates an PipelineStageOptions object.

    Args:
      convolution_options: If provided, a dictionary of Poplar option flags for
        all the convolution operations in the stage.
      matmul_options: If provided, a dictionary of Poplar option flags for
        all the matmul operations in the stage.
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

  def get_proto(self):
    return self._proto


def pipeline(computational_stages,
             pipeline_depth,
             repeat_count=1,
             inputs=None,
             infeed_queue=None,
             outfeed_queue=None,
             optimizer_function=None,
             device_mapping=None,
             pipeline_schedule=None,
             forward_propagation_stages_poplar_options=None,
             backward_propagation_stages_poplar_options=None,
             weight_update_poplar_options=None,
             offload_weight_update_variables=True,
             continuous_weight_updates=False,
             outfeed_loss=False,
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
  be passed through the `infeed_queue` (see
  `ipu.ipu_infeed_queue.IPUInfeedQueue`).

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
  (see `ipu.ipu_outfeed_queue.IPUOutfeedQueue`) is required and all the
  outputs from the last computational stage are enqueued to the
  `outfeed_queue`.

  Note that pipelining also supports recomputation, to enable it, use the
  `tensorflow.ipu.utils.set_recomputation_options()` function when configuring
  the device.

  For example a simple inference network for the MNIST can be split across two
  IPUs:

  .. code-block:: python

    from tensorflow import keras

    # Create the dataset
    #...

    # Create the data queues from/to IPU.
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, "infeed")
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("outfeed")

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
                          pipeline_depth=250,
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

  This creates a pipeline of depth 250 (specified by the `pipeline_depth`),
  which means each pipeline stage is executed 250 times.

  This pipeline is then executed 2 times (specified by the `repeat_count`)
  The results of the pipeline (probabilities and classes) are returned to the
  host by the outfeed queue.

  We can also train this network by providing `optimizer_function`:

  .. code-block:: python

    from tensorflow import keras

    # Create the dataset
    #...

    # Create the data queues from/to IPU.
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, "infeed")
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("outfeed")

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
                          pipeline_depth=128,
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

  Args:
    computational_stages: a list of python functions, where each function
      represents a computational pipeline stage. The function takes the
      outputs of the previous pipeline state as its inputs.
    pipeline_depth: the number of times each pipeline stage will be executed.
    repeat_count: the number of times the pipeline will be executed.
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
      `tf.Variable`s are resident on the same IPU.
    pipeline_schedule: Which scheduling algorithm to use for pipeline
      lowering. Defaults to `PipelineSchedule.Grouped`.
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
    offload_weight_update_variables: If True, any `tf.Variable` which is
      only used by the weight update of the pipeline (for example the
      accumulator variable when using the `tf.MomentumOptimizer`), will be
      stored in the remote memory. During the weight update this variable will
      be streamed onto the device and then streamed back to the remote memory
      after it has been updated. Requires the machine to be configured with
      support for `Poplar graph streaming`. Offloading variables into remote
      memory can reduce maximum memory liveness, but can also increase the
      computation time of the weight update. Note that this option has no effect
      for inference only pipelines.
    continuous_weight_updates: ** CURRENTLY UNIMPLEMENTED ** When training,
      this option will apply the gradients to the resource variables
      immediately, rather than accumulating the gradients and applying them
      at the end of each execution of the pipeline.
    outfeed_loss: If True, the loss given by the `optimizer_function` will
      be enqueued on the outfeed, instead of the outputs from the last
      computational stage.
    name: name of this pipeline.

  Returns:
    An `Operation` that executes the pipeline.

  """
  name = name if name else "pipeline"

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

  device_mapping = device_mapping if device_mapping else list(
      range(0, len(computational_stages)))

  if not isinstance(computational_stages, (list, tuple)):
    raise TypeError(
        "computational_stages argument needs to be a list or a tuple.")

  if infeed_queue:
    if not isinstance(infeed_queue, ipu_infeed_queue.IPUInfeedQueue):
      raise TypeError("infeed_queue is not an instance of "
                      "ipu_infeed_queue.IPUOutfeedQueue")

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

  if pipeline_schedule is None:
    pipeline_schedule = PipelineSchedule.Grouped

  if not isinstance(pipeline_schedule, PipelineSchedule):
    raise TypeError("The given pipeline_schedule is not a member of the "
                    "PipelineSchedule enumeration.")

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

  if outfeed_loss and not optimizer_function:
    raise ValueError(
        "An optimizer_function must be provided when outfeed_loss is True")

  control_outputs = []

  def _pipeline(*args):
    outputs = args
    for stage_id, stage in enumerate(computational_stages):
      stage_infeed_queue = infeed_queue if stage_id == 0 else None
      if stage_id == len(computational_stages) - 1 and not optimizer_function:
        stage_outfeed_queue = outfeed_queue
      else:
        stage_outfeed_queue = None

      stage_name = name + "_stage_" + str(stage_id)
      outputs = _pipeline_stage(stage,
                                stage_id,
                                device_mapping[stage_id],
                                outputs,
                                infeed_queue=stage_infeed_queue,
                                outfeed_queue=stage_outfeed_queue,
                                name=stage_name)

    if optimizer_function:
      outputs = functional_ops._convert_to_list(outputs)  # pylint: disable=protected-access

      # Get the output from the optimizer function
      opt_fn = optimizer_function(*outputs)
      loss = opt_fn.loss
      opt = opt_fn.opt

      # Enqueue loss or any output tensors to the outfeed.
      if outfeed_loss:
        if not outfeed_queue:
          raise ValueError(
              "An outfeed_queue must be provided when outfeed_loss is True")
        control_outputs.append(outfeed_queue.enqueue(opt_fn.loss))
      elif outputs:
        if not outfeed_queue:
          raise ValueError(
              "The last computational stage has tensor outputs: %s, but no"
              " outfeed_queue has been provided." %
              (', '.join(str(t) for t in outputs)))
        control_outputs.append(outfeed_queue.enqueue(outputs))

      # Call the compute gradients function - this will be automatically put
      # into pipeline stages.
      grads_and_vars = opt.compute_gradients(loss)
      # Insert gradient accumulation ops.
      accumulated_grads_and_vars = []
      for grad, var in grads_and_vars:
        if grad is not None:
          with ops.colocate_with(grad):
            # Create an accumulator - variable is used as reference for shape/layout.
            accumulator = gen_poputil_ops.gradient_accumulator_create(var)
            # Add the gradients to the accumulator.
            accumulator = gen_poputil_ops.gradient_accumulator_add(
                accumulator, grad)
            # Sink the accumulators.
            grad = gen_poputil_ops.gradient_accumulator_sink(
                accumulator, num_mini_batches=pipeline_depth)
        # Use the accumulated gradients.
        accumulated_grads_and_vars.append((grad, var))

      # Create an explicit function call for the apply gradients - note that we
      # allow external caputres here.
      apply_grad_ops = []

      def resource_update_():
        apply_grads = opt.apply_gradients(accumulated_grads_and_vars)
        apply_grad_ops.append(apply_grads)

      with ops.name_scope(name + "/WU") as scope:
        func_graph, captured_args = functional_ops._compile_function(  # pylint: disable=protected-access
            resource_update_, [], scope, apply_grad_ops, True)

      # Create the pipeline resource update stage and lower the function into XLA.
      with ops.control_dependencies(list(func_graph.control_captures)):
        outputs = gen_functional_ops.pipeline_resource_update(
            captured_args,
            to_apply=util.create_new_tf_function(func_graph),
            Tout=func_graph.output_types,
            output_shapes=func_graph.output_shapes)

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
    try:
      func_graph, captured_args = functional_ops._compile_function(
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
          pipeline_depth=pipeline_depth,
          repeat_count=repeat_count,
          schedule=int(pipeline_schedule),
          offload_weight_update_variables=offload_weight_update_variables,
          pipeline_poplar_config=json_format.MessageToJson(
              pipeline_poplar_config))
    if not isinstance(output, ops.Operation):
      raise ValueError(
          "Expected the pipeline to output a tf.Operation, got %s instead." %
          (str(output)))

    return output


def _pipeline_stage(func,
                    stage_id,
                    device_id,
                    args,
                    infeed_queue=None,
                    outfeed_queue=None,
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
      if len(dequeue_ops) == 1 and isinstance(dequeue_ops[0], dict):
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
        if not isinstance(outputs, dict):
          outputs = functional_ops._convert_to_list(outputs)  # pylint: disable=protected-access
        outputs = outfeed_queue.enqueue(outputs)
      control_outputs.append(outputs)

    func_to_compile = outfeed_func_wrapper

  with ops.name_scope(name) as scope:
    # pylint: disable=protected-access
    try:
      func_graph, captured_args = functional_ops._compile_function(
          func_to_compile, args, scope, control_outputs)
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
    return func_graph_module.pack_sequence_as(func_graph.structured_outputs,
                                              outputs)
