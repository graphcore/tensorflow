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
Custom operations
~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.ipu.vertex_edsl import PlaceholderVertexExpr
from tensorflow.python.ipu.vertex_edsl import DefaultNameSource


def codelet_expression_op(vertex_expression, *args):
  """Add a custom fused elementwise expression operation to the graph.

  The automatic gradient calculation in TensorFlow does not have visibility
  of the operations performed by this function and so this operation cannot
  be used for training.

  In the following example, the Python function ``my_custom_op()`` provides
  the expression, and the arguments ``a``, ``b`` and ``c`` are the three
  inputs from other parts of the TensorFlow graph.

  .. code-block:: python

    def my_custom_op(x, y, z):
        return x * x + y * z

    ipu.custom_ops.codelet_expression_op(my_custom_op, a, b, c)

  Args:
    vertex_expression: A Python function that defines the codelet expression.
    args: The tensor inputs to the expression.

  Returns:
   The Tensor which is a result of applying the elementwise operation
  """
  dtype = args[0].dtype
  placeholders = map(lambda x: PlaceholderVertexExpr("in" + str(x), None),
                     range(0, len(args)))
  concrete_expression = vertex_expression(*placeholders)
  expr = concrete_expression.lower(DefaultNameSource())
  return gen_poputil_ops.codelet_expression_op(input=args,
                                               dtype=dtype,
                                               source=expr)


def _validate_inputs_with_gradients(inputs_with_gradients, inputs):
  if inputs_with_gradients is None:
    return list(range(0, len(inputs)))
  if isinstance(inputs_with_gradients, list):
    return inputs_with_gradients
  return list(inputs_with_gradients)


def precompiled_user_op(inputs,
                        library_path,
                        gp_path="",
                        outs=None,
                        name="UserOp",
                        op_name="Build",
                        separate_gradients=False,
                        inputs_with_gradients=None,
                        attributes=None,
                        gradient_attributes=None):
  """Call the Poplar function located in the shared library at `library_path`
  as part of the normal TensorFlow execution with the given `inputs`.

  The shape and type of the output should be specified by `outs`. If it is
  `None` it will default to no output. `outs` should be a dictionary with two
  elements like this:

  .. code-block:: python

    outs = {
      "output_types": [my_types_as_a_list],
      "output_shapes": [my_shapes_as_a_list],
    }

  Args:
    inputs: The tensor inputs to the operation.
    library_path: The path to the shared object file that contains the
      functions to build the Poplar operation in the graph.
    gp_path: The path to a precompiled codelet file, if you have one.
    outs: A dictionary describing the output tensor shapes and types.
    name: The name of the operation in TensorFlow.
    op_name: The prefix of the functions inside the shared object file. This
      defaults to 'Build'.
    separate_gradients:  When set to true, multiple gradient ops will be
      generated, one for each input. When false, a single gradient op will be
      generated, which should produce the partial derivatives for all inputs
      (or all inputs specified in `inputs_with_gradients`).
    inputs_with_gradients: A list of input indices. If this is defined
      then the op will only calculate derivatives for the specified inputs.
    attributes: An optional string object which is passed as an argument to the
      build function. Allows you to specify function attributes which were not
      known at the compile time of the C++ Poplar function. Can be used to pass
      a JSON or ProtoBuf serialized string to the Poplar function for ease of
      use. See the documention for examples.
    gradient_attributes: The same as `attributes`, however this is passed as the
      `attributes` argument to the gradient operation (if training).

  Returns:
    The array of tensor outputs.

  """

  if outs is None:
    outs = {
        "output_types": [],
        "output_shapes": [],
    }

  inputs_with_gradients = _validate_inputs_with_gradients(
      inputs_with_gradients, inputs)

  if attributes and not isinstance(attributes, str):
    raise ValueError("Expected attributes to be a 'str' type, but was %s." %
                     type(attributes))

  if gradient_attributes and not isinstance(gradient_attributes, str):
    raise ValueError(
        "Expected gradient_attributes to be a 'str' type, but was %s." %
        type(gradient_attributes))

  return gen_poputil_ops.ipu_user_op(
      inputs,
      library_path=library_path,
      gp_path=gp_path,
      op_name=op_name,
      name=name,
      separate_gradients=separate_gradients,
      gradient_size=0,
      partial_derivative_index=0,
      inputs_with_gradients=inputs_with_gradients,
      attributes=attributes,
      gradient_attributes=gradient_attributes,
      **outs)


def cpu_user_operation(inputs,
                       library_path,
                       outs=None,
                       name="UserOp",
                       op_name="Callback",
                       separate_gradients=False,
                       inputs_with_gradients=None,
                       attributes=None,
                       gradient_attributes=None):
  """
  Call the CPU function located in the shared library at `library_path`
  as part of the normal TensorFlow execution with the given `inputs`
  copied from the IPU to the CPU, and the outputs are copied back to the
  IPU afterwards.

  The shape and type of the outputs should be specified by `outs`. If it is
  `None` it will default to no output. `outs` should be a dictionary with
  two elements like so:

  .. code-block:: python

    outs = {
      "output_types": [my_types_as_a_list],
      "output_shapes": [my_shapes_as_a_list],
    }

  Args:
    inputs: The tensor inputs to the operation.
    library_path: The path to the shared object that contains the functions
      to execute the operation.
    outs: A dictionary describing the output tensor shapes and types.
    name: The name of the operation.
    op_name: The prefix of the functions inside the shared object file. This
      defaults to 'Callback'.
    separate_gradients:  When set to `True`, multiple gradient ops will be
      generated, one for each input. When `False`, a single gradient op will be
      generated, which should produce the partial derivatives for all inputs.
    inputs_with_gradients: A list of input indices. If this is defined
      then the op will only calculate derivatives for the specified inputs.
    attributes: An optional string object which is passed as an argument to the
      Poplar function. Allows you to specify function attributes which were not
      known at the compile time of the C++ Poplar function. Can be used to pass
      a JSON or ProtoBuf serialized string to the Poplar function for ease of
      use. See the documention for examples.
    gradient_attributes: Same as `attribute`, however this is passed as the
      `attribute` to the gradient operations (if training.)

  Returns:
    The array of tensor outputs.
  """

  if outs is None:
    outs = {
        "output_types": [],
        "output_shapes": [],
    }

  inputs_with_gradients = _validate_inputs_with_gradients(
      inputs_with_gradients, inputs)

  if attributes and not isinstance(attributes, str):
    raise ValueError("Expected attributes to be a 'str' type, but was %s." %
                     type(attributes))

  if gradient_attributes and not isinstance(gradient_attributes, str):
    raise ValueError(
        "Expected gradient_attributes to be a 'str' type, but was %s." %
        type(gradient_attributes))

  return gen_poputil_ops.ipu_user_read_write_op(
      inputs,
      library_path=library_path,
      op_name=op_name,
      name=name,
      separate_gradients=separate_gradients,
      gradient_size=0,
      partial_derivative_index=0,
      inputs_with_gradients=inputs_with_gradients,
      attributes=attributes,
      gradient_attributes=gradient_attributes,
      **outs)
