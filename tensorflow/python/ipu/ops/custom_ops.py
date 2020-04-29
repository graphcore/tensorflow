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

  Note that no autograd is done on this fused operation because the autograd
  code does not understand the internal structure of the fused codelet.

  Args:
    vertex_expression: A python function that defines the codelet expression.
    args: Tensor inputs to the expression.

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
                        inputs_with_gradients=None):
  """Call the poplar function located in the shared library at 'library_path'
  as part of the normal tensorflow execution with the given 'inputs'.

  The shape and type of the output should be specified by 'outs'. If it is None
  it will default to no output. 'outs' should be a dictionary with two
  elements like so:

  outs = {
    "output_types": [my_types_as_a_list],
    "output_shapes": [my_shapes_as_a_list],
  }

  Args:
    inputs: The tensor inputs to the operation.
    library_path: The path to the shared object that contains the functions
      to build the Poplar operation in the graph.
    gp_path: The path to the precompiled codelet file.
    outs: A dictionary describing the output tensor shapes and types.
    name: The name of the operation.
    op_name: The prefix of the functions inside the shard object file. This
      defaults to 'Build'.
    separate_gradients:  When set to true, multiple gradient ops will be
      generated, one for each input.  When false, a single gradient op will be
      generated, which should produce the partial derivatives for all inputs.
    inputs_with_gradients: When set, produce derivatives only for specified
      inputs. List of input indices expected.

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

  return gen_poputil_ops.ipu_user_op(
      inputs,
      library_path=library_path,
      gp_path=gp_path,
      op_name=op_name,
      name=name,
      separate_gradients=separate_gradients,
      is_gradient=False,
      partial_derivative_index=0,
      inputs_with_gradients=inputs_with_gradients,
      **outs)


def cpu_user_operation(inputs,
                       library_path,
                       outs=None,
                       name="UserOp",
                       op_name="Callback",
                       separate_gradients=False,
                       inputs_with_gradients=None):
  """Call the CPU function located in the shared library at 'library_path'
    as part of the normal tensorflow execution with the given 'inputs'
    copied from the IPU to the CPU, and the outputs are copied back to the
    IPU afterwards,

    The shape and type of the outputs should be specified by 'outs'. If it is
    None it will default to no output.  'outs' should be a dictionary with
    two elements like so:

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
    op_name: The prefix of the functions inside the shard object file. This
      defaults to 'Callback'.
    separate_gradients:  When set to true, multiple gradient ops will be
      generated, one for each input.  When false, a single gradient op will be
      generated, which should produce the partial derivatives for all inputs.
    inputs_with_gradients: When set, produce derivatives only for specified
      inputs. List of input indices expected.

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

  return gen_poputil_ops.ipu_user_read_write_op(
      inputs,
      library_path=library_path,
      op_name=op_name,
      name=name,
      separate_gradients=separate_gradients,
      is_gradient=False,
      partial_derivative_index=0,
      inputs_with_gradients=inputs_with_gradients,
      **outs)
