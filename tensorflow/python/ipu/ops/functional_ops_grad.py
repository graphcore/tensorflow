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
"""Gradients for Functional operators."""

from tensorflow.compiler.plugin.poplar.ops import gen_functional_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.func_graph import FuncGraph
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import custom_gradient
from tensorflow.python.util import nest
"""
    These gradient function should *never* be called directly.
"""

# Function captures are based on /tensorflow/python/ops/cond_v2.py


class _XlaFuncGradGraph(FuncGraph):
  """FuncGraph for the gradient function of the to_apply function.

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
    # Maps forward intermediate constant valued tensor's id to the constant
    # created in this graph for that tensor.
    self._captured_constants = {}

  @property
  def xla_intermediates(self):
    """Raw intermediates captured from the forward graph if XLA is enabled."""
    return self._xla_intermediates

  def _capture_helper(self, tensor, name):
    if (tensor.graph is not self._forward_graph
        or any(tensor is t for t in self._forward_graph.outputs)):
      return super(_XlaFuncGradGraph, self)._capture_helper(tensor, name)

    # If `tensor` is a graph-building time constant, we create a constant with
    # the same value in the backward graph instead of capturing it.
    tensor_id = ops.tensor_id(tensor)
    if tensor_id in self._captured_constants:
      return self._captured_constants[tensor_id]
    elif constant_op.is_constant(tensor):
      self._captured_constants[tensor_id] = constant_op.constant(
          tensor_util.constant_value(tensor), dtype=tensor.dtype)
      return self._captured_constants[tensor_id]

    if control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()):
      # Capture the intermidate so that it can be added as an extra output
      # which will be used in the gradient calculations.
      if all(tensor is not capture for capture in self.external_captures):
        self.xla_intermediates.append(tensor)
        self.op_needs_rewrite = True
    return super(_XlaFuncGradGraph, self)._capture_helper(tensor, name)


def _resolve_grad_inputs(graph, grad_graph, op):
  """Returns the tensors to pass as inputs to `grad_graph`.

  The `grad_graph` may have external references to
  1. Its outer graph containing the input gradients. These references are kept
     as is.
  2. Tensors in the forward pass graph. These tensors may not be "live"
     when the gradient is being computed. We replace such references by their
     corresponding tensor in `graph.outer_graph`.

  Args:
    graph: FuncGraph. The forward-pass function.
    grad_graph: FuncGraph. The gradients function.

  Returns:
    A list of inputs tensors to be passed to grad_graph.
  """
  new_inputs = []
  for t in grad_graph.external_captures:
    # `t` must either be in `grad_graph.outer_graph` or in the forward
    # `graph`.
    if t.graph != grad_graph.outer_graph:
      if t.graph == graph:
        for i, output in enumerate(t.graph.outputs):
          if output is t:
            t = op.outputs[i]
            break
      # Note: We rely on the capturing logic of the gradient op graph to
      # correctly capture the tensors in `graph.outer_graph`. This is handled
      # when building the gradient function.
      if t.graph != graph.outer_graph:
        raise ValueError(
            "Attempting to capture tensor %s which is not an output." %
            (str(t)))

    new_inputs.append(t)

  return new_inputs


def _get_gradients_for_function(op, *grads):
  # Note that this function assumes that the op has function graph at attribute `to_apply` which has only single user (this op).
  assert control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph())
  # Get the forward function to create a gradient for it.
  fwd_op = op.outputs[0].op
  inputs = fwd_op.inputs

  # Use the original graph incase any values were captured.
  # We know we can modify this graph as we generate unique Functional graphs.
  func_graph = fwd_op.graph._get_function(  # pylint: disable=protected-access
      fwd_op.get_attr("to_apply").name).graph
  assert func_graph.outer_graph == op.graph
  for external_t, internal_t in zip(inputs, func_graph.inputs):
    custom_gradient.copy_handle_data(external_t, internal_t)

  func_graph.reset_captures(zip(inputs, func_graph.inputs))

  # Note: op.graph != ops.get_default_graph() when we are computing the gradient
  # of a pipeline stage.
  assert func_graph.outer_graph == fwd_op.graph

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
    # pylint: disable=protected-access
    fwd_op._set_func_attr("to_apply", util.create_new_tf_function(func_graph))
    fwd_op._set_type_list_attr("Tout", func_graph.output_types)
    fwd_op._set_shape_list_attr("output_shapes", func_graph.output_shapes)
    fwd_op._add_outputs([t.dtype for t in extra_func_outputs],
                        [t.shape for t in extra_func_outputs])

  func_grad_inputs = _resolve_grad_inputs(func_graph, func_grad_graph, op)
  # pylint: enable=protected-access
  return func_grad_graph, func_grad_inputs


def _pack_sequence_as(structured_outputs, op_outputs):
  """Packs the outputs of a functional op.

  The functions may contain None's in the list of `structured_outputs`.
  `op_outputs` has those outputs missing. So we need to add those Nones to the
  list of `op_outputs` and then pack it in the same structure as
  `structured_outputs`.

  Args:
    structured_outputs: structured_outputs from one of the branch functions.
    op_outputs: List of output tensors of the op.

  Returns:
    `op_outputs` packed like `structured_outputs`.
  """
  outputs_with_nones = []
  counter = 0
  for output in nest.flatten(structured_outputs, expand_composites=True):
    if output is None:
      outputs_with_nones.append(None)
    else:
      outputs_with_nones.append(op_outputs[counter])
      counter += 1
  return func_graph_module.pack_sequence_as(structured_outputs,
                                            outputs_with_nones)


@ops.RegisterGradient("Function")
def _function_grad(op, *grads):
  """The gradient of a Function op."""
  func_grad_graph, func_grad_inputs = _get_gradients_for_function(op, *grads)
  outputs = gen_functional_ops.function(
      func_grad_inputs,
      to_apply=util.create_new_tf_function(func_grad_graph),
      Tout=func_grad_graph.output_types,
      output_shapes=func_grad_graph.output_shapes)

  return _pack_sequence_as(func_grad_graph.structured_outputs, outputs)
