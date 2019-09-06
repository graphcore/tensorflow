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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.plugin.poplar.ops import gen_pipelining_ops
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util_v2 as util


def pipeline(computational_stages,
             repeat_count,
             inputs=None,
             infeed_queue=None,
             outfeed_queue=None,
             optimizer_stage=None,
             name=None):
  """
  Sets up a series of computational stages, where the outputs of one stage are
  the inputs to the next one. These stages are then executed in parallel across
  multiple IPUs. This approach can be used to split the model where layer(s) are
  executed on different IPUs.

  The first stage takes the `inputs` and the `infeed_queue` (if provided) as its
  inputs. If the `infeed_queue` is provided, it is automatically dequeued
  (similar to the ipu.loops API) therefore care needs to be taken to make sure
  the signature of the first pipeline stage matches both the arguments from
  `inputs` and the `infeed_queue`, otherwise an error is thrown.

  All tensors which are used in the pipeline which are not TensorFlow Variables
  need to be explicitly passed as inputs to the pipeline. If an input does not
  change its value during the execution of the pipeline op (for example
  hyperparameters such as learning rate), it needs to be passed as part of
  `inputs`. Alternatively, if these values change during execution (for example
  the model processes different batches of data) the input should be passed
  through the `infeed_queue` (see ipu.ipu_infeed_queue.IPUInfeedQueue`).

  When training a model, an optional `optimizer_stage` function can be provided.
  This function takes all the outputs from the last computational stage and it
  can use them to generate the backwards pass of the model using the TensorFlow
  Optimizer API. This will internally create corresponding backpropagation
  pipeline stages for each pipeline stage and colocate them such that the
  activations and weights required for the gradient calculation and application
  stay on the device in order to minimise the number of copies between IPUs.

  If an `optimizer_stage` is provided and the `optimizer_stage` has any
  tf.Tensor outputs then an `outfeed_queue` (see
  `ipu.ipu_outfeed_queue.IPUOutfeedQueue`) is required and all the tf.Tensor
  outputs from the `optimizer_stage` are enqueued to the `outfeed_queue`.

  Alternatively, if an `optimizer_stage` is not provided and the last
  computational stage has any outputs, then an `outfeed_queue`
  (see `ipu.ipu_outfeed_queue.IPUOutfeedQueue`) is required and all the outputs
  from the last computational stage are enqueued to the `outfeed_queue`.

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
                          repeat_count=250,
                          inputs=[],
                          infeed_queue=infeed_queue,
                          outfeed_queue=outfeed_queue,
                          name="Pipeline")
      return pipeline_op

    with ops.device("/device:IPU:0"):
      compiled_model = ipu_compiler.compile(model, inputs=[])

    outfeed_op = outfeed_queue.dequeue()
    with tf.Session() as sess:
      result = sess.run(compiled_model)
      probabilities, classes = sess.run(outfeed_op)

  In this set up, the model is split across two IPUs with the first two layers
  being executed on the first IPU and the third layer and the probabilities and
  classes are executed on the second IPU. This pipeline is then executed 250
  times (specified by the `repeat_count`) and the resulting probabilities and
  classes are returned to the host by the outfeed queue.

  We can also train this network by providing `optimizer_stage`:

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

    def optimizer_stage(lr, loss):
      optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
      return loss, optimizer

    def model(lr):
      with variable_scope.variable_scope("vs", use_resource=True):
        pipeline_op = pipelining_ops.pipeline(
                          computational_stages=[stage1, stage2],
                          repeat_count=250,
                          inputs=[lr],
                          infeed_queue=infeed_queue,
                          outfeed_queue=outfeed_queue,
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
  which calculate the gradients and apply them to the weights. Note how the loss
  is returned to the host by the outfeed queue.

  Args:
    computational_stages: a list of python functions, where each function
      represents a computational pipeline stage. The function takes the outputs
      of the previous pipeline state as its inputs.
    repeat_count: the number of times the pipeline will be executed.
    inputs: arguments passed to the first pipeline stage.
    infeed_queue: optional IPUInfeedQueue, if passed, it is dequeued and passed
      as an input in the first pipeline stage.
    outfeed_queue: IPUOutfeedQueue, required if the last computational stage or
      optimizer_stage has any outputs. The outputs of these are enqueued to this
      queue and they can be accessed on the host.
    optimizer_stage: optional Python function which takes the output of the last
      computational stage and uses that to call a TensorFlow Optimizer in order
      to generate the backward pass for the model.
    name: name of this pipeline sage.

  Returns:
    An `Operation` that executes the pipeline.

  """
  name = name if name else "pipeline"
  inputs = inputs if inputs else []

  if not isinstance(computational_stages, (list, tuple)):
    raise ValueError(
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

  control_outputs = []

  def _pipeline(*args):
    outputs = args
    for stage_id, stage in enumerate(computational_stages):
      stage_infeed_queue = infeed_queue if stage_id == 0 else None
      if stage_id == len(computational_stages) - 1 and not optimizer_stage:
        stage_outfeed_queue = outfeed_queue
      else:
        stage_outfeed_queue = None

      stage_name = name + "_stage_" + str(stage_id)
      outputs = _pipeline_stage(
          stage,
          stage_id,
          outputs,
          infeed_queue=stage_infeed_queue,
          outfeed_queue=stage_outfeed_queue,
          name=stage_name)
    if optimizer_stage:
      # Apply the optimizer stage
      outputs = optimizer_stage(*_convert_to_list(outputs))
      outputs = _convert_to_list(outputs)
      # Enqueue any output tensors to the outfeed.
      try:
        outputs = [
            o if isinstance(o, ops.Operation) else ops.convert_to_tensor(o)
            for o in outputs
        ]
      except Exception as e:
        raise ValueError(
            "'optimizer_stage' function return values must all either be "
            "tf.Operations or convertible to tf.Tensors. Got error: '%s'" %
            str(e))

      # Separates the returned Operations and Tensors.
      output_operations = [o for o in outputs if isinstance(o, ops.Operation)]
      output_tensors = [o for o in outputs if not isinstance(o, ops.Operation)]

      if outputs != output_tensors + output_operations:
        raise ValueError(
            "optimizer_stage' function must return zero or more  Tensor values "
            "followed by zero or more Operations.")

      if output_tensors:
        # Enqueue the outfeed.
        if not outfeed_queue:
          raise ValueError(
              "The optimizer_stage has tensor outputs: %s, but no outfeed_queue"
              " has been provided." % (', '.join(
                  str(t) for t in output_tensors)))
        output_operations.append(outfeed_queue.enqueue(output_tensors))
      with ops.control_dependencies(output_operations):
        outputs = control_flow_ops.no_op()
    else:
      if not isinstance(outputs, ops.Operation) and not outfeed_queue:
        raise ValueError(
            "The last computational stage has tensor outputs: %s, but no"
            " outfeed_queue has been provided." % (', '.join(
                str(t) for t in _convert_to_list(outputs))))

    if not isinstance(outputs, ops.Operation):
      raise ValueError(
          "Expected the last pipeline stage to output a tf.Operation, "
          "got %s instead." % (str(outputs)))
    control_outputs.append(outputs)

  with ops.name_scope(name) as scope:
    func_graph, captured_args = _compile_function(_pipeline, inputs, scope,
                                                  control_outputs, name)

    # Create the pipeline and lower the function into XLA.
    with ops.control_dependencies(list(func_graph.control_captures)):
      output = gen_pipelining_ops.pipeline(
          captured_args,
          to_apply=util.create_new_tf_function(func_graph),
          Tout=func_graph.output_types,
          output_shapes=func_graph.output_shapes,
          repeat_count=repeat_count)
    if not isinstance(output, ops.Operation):
      raise ValueError(
          "Expected the pipeline to output a tf.Operation, got %s instead." %
          (str(output)))

    return output


def _pipeline_stage(func,
                    stage_id,
                    args,
                    kwargs=None,
                    infeed_queue=None,
                    outfeed_queue=None,
                    name=None):
  """Internal function for compiling a pipeline stage. This should not be called
  directly and doing so will result in undefined behaviour.

  Creates a pipeline stage.

  Args:
    func: function which will be executed as a stage.
    args: arguments to the function.
    kwargs: key-word arguments to the function.
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
  args = _convert_to_list(args)

  func_to_compile = func
  control_outputs = []
  # If we have an infeed, then we wrap the function in another function which
  # dequeues the infeed.
  if infeed_queue:

    def infeed_func_wrapper(*args):
      args = _convert_to_list(args)
      dequeue_ops = _convert_to_list(infeed_queue._dequeue())
      # Deal with the dequeue depending on whether it's a list or dict.
      if len(dequeue_ops) == 1 and isinstance(dequeue_ops[0], dict):
        kwargs = dequeue_ops[0]
        return func(*(args), **kwargs)
      else:
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
        outputs = _convert_to_list(outputs)
        outputs = outfeed_queue.enqueue(outputs)
      control_outputs.append(outputs)

    func_to_compile = outfeed_func_wrapper

  with ops.name_scope(name) as scope:
    func_graph, captured_args = _compile_function(func_to_compile, args, scope,
                                                  control_outputs, name)

    # Create the pipeline stage and lower the function into XLA.
    with ops.control_dependencies(list(func_graph.control_captures)):
      outputs = gen_pipelining_ops.pipeline_stage(
          captured_args,
          to_apply=util.create_new_tf_function(func_graph),
          Tout=func_graph.output_types,
          output_shapes=func_graph.output_shapes,
          stage_id=stage_id)
    if isinstance(outputs, ops.Operation):
      return outputs
    else:
      return func_graph_module.pack_sequence_as(func_graph.structured_outputs,
                                                outputs)


def _compile_function(func, args, scope, control_outputs, name):
  # Automatic control dependencies are added in defuns, but not in v1
  # graphs. Propagate that behavior here.
  add_control_dependencies = ops.get_default_graph()._add_control_dependencies

  func_name = util.unique_fn_name(scope, "func")
  captured_args = [ops.convert_to_tensor(x) for x in args]

  # Compile the function to a graph.
  func_graph = func_graph_module.func_graph_from_py_func(
      func_name,
      func,
      captured_args, {},
      add_control_dependencies=add_control_dependencies)

  # Add the external captures (resources) to arguments.
  for t in func_graph.external_captures:
    if t.dtype != dtypes.resource:
      raise ValueError(
          "Trying to capture the tensor %s which is not a resource. This tensor "
          "needs to be passed as either part of the `input` or `infeed_queue` of "
          "the pipeline." % (t.name))
  captured_args += func_graph.external_captures

  # Add any control outputs.
  func_graph.control_outputs.extend(control_outputs)

  return func_graph, captured_args


def _convert_to_list(xs):
  if not isinstance(xs, (list, tuple)):
    return [xs]
  else:
    return list(xs)
