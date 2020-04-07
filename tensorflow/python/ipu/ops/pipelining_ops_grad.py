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
"""Gradients for Pipelining operators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.plugin.poplar.ops import gen_pipelining_ops
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework.func_graph import FuncGraph
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import custom_gradient
"""
    These gradient function should *never* be called directly.
"""

# Function captures are based on /tensorflow/python/ops/cond_v2.py


class _XlaFuncGradGraph(FuncGraph):
  """FuncGraph for the gradient function of the to_apply function of a
  PipelineStage.

  Handles intermediate values from the forward pass which are required for the
  gradient calculations.

  Attributes:
    op_needs_rewrite: True if any intermediates were captured, meaning the
      forward op needs to be written to output the wrapped intermediates.
  """
  def __init__(self, name, forward_graph):
    super(_XlaFuncGradGraph,
          self).__init__(name,
                         collections=ops.get_default_graph()._collections)  # pylint: disable=protected-access
    self.op_needs_rewrite = False
    self._forward_graph = forward_graph
    # Raw intermediates captured from the forward graph. Populated iff we're in
    # an XLA context.
    self._xla_intermediates = []

  @property
  def xla_intermediates(self):
    """Raw intermediates captured from the forward graph if XLA is enabled."""
    return self._xla_intermediates

  def _capture_helper(self, tensor, name):
    if (tensor.graph is not self._forward_graph
        or id(tensor) in map(id, self._forward_graph.outputs)):
      return super(_XlaFuncGradGraph, self)._capture_helper(tensor, name)

    if control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()):
      # Capture the intermidate so that it can be added as an extra output
      # which will be used in the gradient calculations.
      if id(tensor) not in map(id, self.captures):
        self.xla_intermediates.append(tensor)
        self.op_needs_rewrite = True
    return super(_XlaFuncGradGraph, self)._capture_helper(tensor, name)


@ops.RegisterGradient("PipelineStage")
def _pipeline_stage_grad(op, *grads):
  """The gradient of a PipelineStage op."""
  assert control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph())
  # Get the pipeline stage function to create a gradient for it.
  stage_op = op.outputs[0].op
  stage_id = stage_op.get_attr('stage_id')
  inputs = stage_op.inputs
  input_shapes = [t.shape for t in inputs]
  fdef = stage_op.graph._get_function(
      stage_op.get_attr("to_apply").name).definition

  with op.graph.as_default():
    func_graph = function_def_to_graph.function_def_to_graph(
        fdef, input_shapes)
  for external_t, internal_t in zip(inputs, func_graph.inputs):
    custom_gradient.copy_handle_data(external_t, internal_t)

  func_graph.reset_captures(zip(inputs, func_graph.inputs))

  # Link the op so that the gradient code can use it.
  func_graph._forward_cond = op

  # Note: op.graph != ops.get_default_graph() when we are computing the gradient
  # of a pipeline stage.
  assert func_graph.outer_graph == stage_op.graph

  # Create grad function that compute the gradient of the forward graphs. This
  # function will capture tensors from the forward pass function.
  grad_name = util.unique_grad_fn_name(func_graph.name)
  func_grad_graph = func_graph_module.func_graph_from_py_func(
      grad_name,
      lambda: cond_v2._grad_fn(func_graph, grads), [], {},
      func_graph=_XlaFuncGradGraph(grad_name, func_graph))

  if func_grad_graph.op_needs_rewrite:
    # Modify 'op' to output the intermediates needed by the grad functions. Note
    # that all needed intermediates are wrapped in optionals.
    extra_func_outputs, = cond_v2._make_intermediates_match_xla(
        [func_graph], [func_grad_graph.xla_intermediates])
    func_graph.outputs.extend(extra_func_outputs)

    # Rewrite the forward function so that it outputs the intermediates.
    func_graph.name += "_rewritten"
    stage_op._set_func_attr("to_apply",
                            util.create_new_tf_function(func_graph))
    stage_op._set_type_list_attr("Tout", func_graph.output_types)
    stage_op._set_shape_list_attr("output_shapes", func_graph.output_shapes)
    stage_op._add_outputs([t.dtype for t in extra_func_outputs],
                          [t.shape for t in extra_func_outputs])

  func_grad_inputs = cond_v2._resolve_grad_inputs(func_graph, func_grad_graph)

  outputs = gen_pipelining_ops.pipeline_stage_backward(
      func_grad_inputs,
      to_apply=util.create_new_tf_function(func_grad_graph),
      Tout=func_grad_graph.output_types,
      output_shapes=func_grad_graph.output_shapes,
      stage_id=stage_id)

  return func_graph_module.pack_sequence_as(func_grad_graph.structured_outputs,
                                            outputs)


@ops.RegisterGradient("Pipeline")
def _pipeline_grad(op, *grads):
  """The gradient of a Pipeline op."""
  raise RuntimeError(
      "Attempting to calculate the gradient of a Pipeline which is not allowed."
      " If you are trying to generate the gradients of operations inside the"
      " pipeline, use the `optimizer_stage` (see"
      " tensorflow.python.ipu.pipelining_ops.pipeline for more information).")
