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
Functional operators
~~~~~~~~~~~~~~~~~~~~~~
"""
# Function captures are based on /tensorflow/python/ops/cond_v2.py

from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.compiler.plugin.poplar.ops import gen_functional_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.ipu import scopes
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.util import nest


def outlined_function(func=None,
                      unique_sharding=False,
                      keep_input_layouts=None,
                      name=None):
  """
  An outlined function is a block of organized, reusable code which is used to
  perform a single action. Functions provide better modularity for your
  application and a high degree of code reusing which can decrease the memory
  usage at the expense of passing the arguments around.

  Functions can be used by models constrained by memory which have common
  structures or to serialize some large operations.

  If the provided function contains any stateful operations, such as stateful
  random number generation, then the function cannot be reused and it will be
  inlined automatically.

  See the documentation for more details and examples.

  Args:
    func: A python function which takes a list of positional arguments only. All
      the arguments must be `tf.Tensor`-like objects, or be convertible to them.
      See the documentation for examples of how to pass non `tf.Tensor`-like
      objects to the functions.
      The function provided must return at least one `tf.Tensor`-like object.
    unique_sharding: Makes sure that all function inputs are copied to a single
      device before the function call is executed. Enabling this can increase
      performance as any inter IPU communication can be more efficiently
      scheduled and any duplicated copies can be elided.
    keep_input_layouts: A hint to decide whether to keep the layouts of the
      function inputs when calling the function or re-allocate them based on the
      operations inside the function. Reallocating them can improve the
      performance, but it can also increase the IPU code size. When set to
      'None', this option will be decided automatically.
    name: The name of the function.

  Returns:
    An `Operation` that executes the function.

  """
  name = name if name else "function"

  def decorated(inner_func):
    def func_wrapper(*args):
      args = _convert_to_list(args)
      with ops.name_scope(name) as scope:
        func_graph, captured_args = _compile_function(
            inner_func, args, scope, [], allow_external_captures=True)

        with ops.control_dependencies(list(func_graph.control_captures)):
          outputs = gen_functional_ops.function(
              captured_args,
              to_apply=util.create_new_tf_function(func_graph),
              Tout=func_graph.output_types,
              output_shapes=func_graph.output_shapes,
              unique_sharding=unique_sharding,
              keep_input_layouts=keep_input_layouts,
              name=name)

          # pack_sequence_as requires a list of Tensors, but the gen_ operation
          # returns an Operation under some circumstances (probably when that
          # list would be empty)
          if isinstance(outputs, ops.Operation):
            outputs = outputs.outputs

        return _pack_sequence_as(func_graph.structured_outputs, outputs)

    return func_wrapper

  if func is not None:
    return decorated(func)

  return decorated


class _InvalidCaptureException(Exception):
  pass


def _compile_function(func,
                      args,
                      scope,
                      control_outputs,
                      allow_external_captures=False):
  parent_graph = ops.get_default_graph()
  # Automatic control dependencies are added in defuns, but not in v1
  # graphs. Propagate that behavior here.
  add_control_dependencies = parent_graph._add_control_dependencies  # pylint: disable=protected-access

  # Functions inherit frontend attributes and the gradient override map from the
  # parent graph.
  proto = xla_data_pb2.FrontendAttributes()
  value = parent_graph._attr_scope_map.get(scopes.FRONTEND_ATTRIBUTES_NAME)  # pylint: disable=protected-access
  if value:
    proto.ParseFromString(value.s)
  attribute = attr_value_pb2.AttrValue(s=proto.SerializeToString())
  gradient_override_map = dict(parent_graph._gradient_override_map)  # pylint: disable=protected-access

  def func_wrapper(*args, **kwargs):
    # Add the frontend attributes to the current attributes.
    g = ops.get_default_graph()
    attributes = dict(g._attr_scope_map)  # pylint: disable=protected-access
    attributes[scopes.FRONTEND_ATTRIBUTES_NAME] = attribute

    with g._attr_scope(attributes):  # pylint: disable=protected-access
      with g.gradient_override_map(gradient_override_map):
        return func(*args, **kwargs)

  func_name = util.unique_fn_name(scope, "func")
  captured_args = ops.convert_n_to_tensor(args)

  # Compile the function to a graph.
  func_graph = func_graph_module.func_graph_from_py_func(
      func_name,
      func_wrapper,
      captured_args, {},
      add_control_dependencies=add_control_dependencies)

  # Add the external captures (resources) to arguments.
  for t in func_graph.external_captures:
    if not allow_external_captures and t.dtype != dtypes.resource:
      raise _InvalidCaptureException(t.name)
  captured_args += func_graph.external_captures

  # Add any control outputs.  Autograph will add control outputs to the graph
  # automatically, so only add ones which are not already present.
  for o in control_outputs:
    if not o in func_graph.control_outputs:
      func_graph.control_outputs.extend([o])

  # Fix shape inference for the gradients and extract_outside_compilation_pass.
  for op in func_graph.get_operations():
    output_shapes = [out.get_shape() for out in op.outputs]
    # pylint: disable=protected-access
    op._set_shape_list_attr("_output_shapes", output_shapes)
    op._set_shape_list_attr("_xla_inferred_shapes", output_shapes)
    # pylint: enable=protected-access

  return func_graph, captured_args


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


def _convert_to_list(xs):
  if not isinstance(xs, (list, tuple)):
    return [xs]
  return list(xs)
