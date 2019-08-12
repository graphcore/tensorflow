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
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_util_v2 as util


def pipeline(stages, inputs=None, infeed_queue=None, name=None):
  """
  Sets up a series of computational stages, where the output of one element is
  the input of the next one. These stages are then executed in parallel across
  multiple IPUs.
  This approach can be used to split the model where layer(s) are executed on
  different IPUs.

  For example a simple network for the MNIST can be split across two IPUs:

  .. code-block:: python

    from tensorflow import keras

    # Create the dataset
    #...

    # Create the infeed queue.
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, "infeed")

    # Create a pipelined model which is split accross two stages.
    def stage1(image, labels):
        partial = keras.layers.Dense(256, activation=tf.nn.relu)(image)
        partial = keras.layers.Dense(128, activation=tf.nn.relu)(partial)
        return partial, labels

    def stage2(partial, labels):
        logits = keras.layers.Dense(10)(partial)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                              labels=labels, logits=logits)
        loss = tf.reduce_mean(cross_entropy)
        return loss

    def my_net(x):
      with variable_scope.variable_scope("vs", use_resource=True):
        loss = pipelining_ops.pipeline([stage1, stage2], [],
                                       infeed_queue=infeed_queue)
      return loss

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[x])

  In this set up the model is split across two IPUs with the first two layers
  being executed on the first IPU and the thrid layer and the loss is calculated
  on the second IPU.

  It is possible to use the pipeline for training. If we take the previous
  example and use an Optimizer:

  .. code-block:: python

    def my_net(x):
      with variable_scope.variable_scope("vs", use_resource=True):
        loss = pipelining_ops.pipeline([stage1, stage2], [],
                                       infeed_queue=infeed_queue)
      optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
      return loss

  This will internally create a corresponding backwards path pipeline stage
  for each forward path pipeline stage and colocate them such that the
  activations required for the backwards path stay on the device in order to
  minimise the number of copies between IPUs.

  Args:
    stages: a list of python functions, where each function represents a
      piepeline stage. The function takes the outputs of the previous pipeline
      state as its inputs.
    inputs: arguments passed to the first pipeline stage.
    infeed_queue: optional IPUInfeedQueue, if passed, it is dequeued and passed
      as an input in the first pipeline stage.
    name: name of this pipeline sage.

  Returns:
    The values returned from the last pipeline stage.

  """
  name = name if name else "pipeline"
  inputs = inputs if inputs else []

  if not isinstance(stages, (list, tuple)):
    raise ValueError("stages argument needs to be a list or a tuple.")

  # We expect at least one stage.
  if len(stages) < 2:
    raise ValueError("Pipeline requires at least two stages.")

  def _pipeline(*args):
    last_outputs = args
    for i, stage in enumerate(stages):
      stage_infeed_queue = infeed_queue if i == 0 else None
      stage_name = name + "_stage_" + str(i)
      last_outputs = _pipeline_stage(
          stage,
          last_outputs,
          infeed_queue=stage_infeed_queue,
          name=stage_name)
    return last_outputs

  with ops.name_scope(name) as scope:
    func_graph, captured_args = _compile_function(_pipeline, inputs, scope,
                                                  name)

    # Create the pipeline and lower the function into XLA.
    with ops.control_dependencies(list(func_graph.control_captures)):
      tensors = gen_pipelining_ops.pipeline(
          captured_args,
          to_apply=util.create_new_tf_function(func_graph),
          Tout=func_graph.output_types,
          output_shapes=func_graph.output_shapes)

    return func_graph_module.pack_sequence_as(func_graph.structured_outputs,
                                              tensors)


def _pipeline_stage(func, args, kwargs=None, infeed_queue=None, name=None):
  """Internal function for compiling a pipeline stage. This should not be called
  directly and doing so will result in undefined behaviour.

  Creates a pipeline stage.

  Args:
    func: function which will be executed as a stage.
    args: arguments to the function.
    kwargs: key-word arguments to the function.
    infeed_queue: optional IPUInfeedQueue, if passed, it is dequeued as part of
      this function.
    name: name of this pipeline sage.

  Returns:
    The values after execting func(args)

  """
  name = name if name else "pipeline_stage"
  args = _convert_to_list(args)

  func_to_compile = func
  # If we have an infeed, then we wrap the function in another function which
  # dequeues the infeed.
  if infeed_queue:
    if not isinstance(infeed_queue, ipu_infeed_queue.IPUInfeedQueue):
      raise TypeError(
          "infeed_queue is not an instance of ipu_infeed_queue.IPUInfeedQueue")

    def func_wrapper(*args):
      args = _convert_to_list(args)
      dequeue_ops = _convert_to_list(infeed_queue._dequeue())
      # Deal with the dequeue depending on whether it's a list or dict.
      if len(dequeue_ops) == 1 and isinstance(dequeue_ops[0], dict):
        kwargs = dequeue_ops[0]
        return func(*(args), **kwargs)
      else:
        return func(*(args + dequeue_ops))

    func_to_compile = func_wrapper

  with ops.name_scope(name) as scope:
    func_graph, captured_args = _compile_function(func_to_compile, args, scope,
                                                  name)

    # Create the pipeline stage and lower the function into XLA.
    with ops.control_dependencies(list(func_graph.control_captures)):
      tensors = gen_pipelining_ops.pipeline_stage(
          captured_args,
          to_apply=util.create_new_tf_function(func_graph),
          Tout=func_graph.output_types,
          output_shapes=func_graph.output_shapes)

    return func_graph_module.pack_sequence_as(func_graph.structured_outputs,
                                              tensors)


def _compile_function(func, args, scope, name):
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
  captured_args += func_graph.external_captures

  return func_graph, captured_args


def _convert_to_list(xs):
  if not isinstance(xs, (list, tuple)):
    return [xs]
  else:
    return list(xs)
