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
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.func_graph import FuncGraph
from tensorflow.python.ipu import functional_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gradients_util
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
      lambda: _grad_fn(func_graph, grads), [], {},
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
    # pylint: enable=protected-access

  func_grad_inputs = _resolve_grad_inputs(func_graph, func_grad_graph, op)
  constant_outputs = functional_ops._get_constant_outputs(  # pylint: disable=protected-access
      func_grad_graph, func_grad_inputs)
  return func_grad_graph, func_grad_inputs, constant_outputs


@ops.RegisterGradient("Function")
def _function_grad(op, *grads):
  """The gradient of a Function op."""
  func_grad_graph, func_grad_inputs, constant_outputs = \
    _get_gradients_for_function(op, *grads)
  outputs = gen_functional_ops.function(
      func_grad_inputs,
      to_apply=util.create_new_tf_function(func_grad_graph),
      Tout=func_grad_graph.output_types,
      output_shapes=func_grad_graph.output_shapes,
      unique_sharding=op.get_attr("unique_sharding"),
      keep_input_layouts=True)

  outputs = functional_ops._replace_outputs(outputs, constant_outputs)  # pylint: disable=protected-access
  return functional_ops._pack_sequence_as(  # pylint: disable=protected-access
      func_grad_graph.structured_outputs, outputs)


def _grad_fn(func_graph, grads):
  """The gradient function for each conditional branch.

  This function builds the gradient graph of the corresponding forward-pass
  conditional branch in `func_graph`. This is done by differentiating
  func_graph's outputs w.r.t. its inputs.

  Args:
    func_graph: FuncGraph. The corresponding forward-pass function.
    grads: The list of input gradient Tensors.

  Returns:
    The output gradient Tensors.
  """
  # Filter out untrainable function outputs.
  assert len(func_graph.outputs) == len(grads)
  ys = []
  grad_ys = []
  for y, grad_y in zip(func_graph.outputs, grads):
    if not backprop_util.IsTrainable(y):
      continue
    ys.append(y)
    grad_ys.append(grad_y)

  # Build the gradient graph. Note that this builds the gradient computation of
  # func_graph in the current graph, which requires capturing tensors from
  # func_graph. The captured func_graph tensors are resolved to external tensors
  # in _resolve_grad_inputs.
  result = gradients_util._GradientsHelper(  # pylint: disable=protected-access
      ys,
      func_graph.inputs,
      grad_ys=grad_ys,
      src_graph=func_graph)

  return result
