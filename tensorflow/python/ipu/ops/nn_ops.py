# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
Popnn primitive neural network operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from google.protobuf import json_format

from tensorflow.compiler.plugin.poplar.driver import backend_config_pb2
from tensorflow.compiler.plugin.poplar.driver import option_flag_pb2
from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
from tensorflow.compiler.plugin.poplar.ops import gen_functional_ops
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.ipu import functional_ops
from tensorflow.python.ipu import scopes
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import nn_grad


def gelu(x, name=None):
  """This targets the PopLibs Popnn gelu operation, optimised for execution
  on the IPU.

  Args:
    x: The input tensor.
    name: Optional op name.

  Returns:
    A `Tensor`. Has the same type the input tensor.
  """

  return gen_popnn_ops.ipu_gelu(x, name=name)


def multi_conv(func=None, options=None):
  """A function decorator for generating multi-convolution operations.
  Multi-convolutions allow for a set of data-independent convolutions to be
  executed in parallel. Executing convolutions in parallel can lead to an
  increase in the data throughput.

  The ``multi_conv`` function decorator is a convenient way to generate
  multi-convolutions - it detects all the convolution operations inside of the
  decorated function and executes them in parallel.

  For example:

  .. code-block:: python

    from tensorflow import keras
    from tensorflow.python import ipu

    @ipu.nn_ops.multi_conv
    def convs(x, y, z):
      x = keras.layers.DepthwiseConv2D(8, 2, depth_multiplier=2)(x)
      y = keras.layers.DepthwiseConv2D(16, 4, depth_multiplier=2)(y)
      z = keras.layers.Conv2D(8, 3)(z)
      return x, y, z

  Will detect and execute the three convolutions ``x``, ``y`` and ``z`` in
  parallel.
  Note that any operations which are not convolutions, such as bias add
  operations, will be executed in the same way as if they were not inside of a
  ``multi_conv`` decorated function.

  It is also possible to set PopLibs multi-convolution options using this
  decorator.

  For example:

  .. code-block:: python

    from tensorflow import keras
    from tensorflow.python import ipu

    @ipu.nn_ops.multi_conv(options={"perConvReservedTiles":"50"})
    def convs(x, y, z):
      x = keras.layers.DepthwiseConv2D(8, 2, depth_multiplier=2)(x)
      y = keras.layers.DepthwiseConv2D(16, 4, depth_multiplier=2)(y)
      z = keras.layers.Conv2D(8, 3)(z)
      return x, y, z

  See the PopLibs documention for the list of all available flags.
  Note that these options will also be applied to the gradient operations
  generated during backpropagation.

  Args:
    func: A python function which takes a list of positional arguments only. All
      the arguments must be `tf.Tensor`-like objects, or be convertible to them.
      The function provided must return at least one `tf.Tensor`-like object.
    options: A dictionary of Poplar option flags for multi-convolution. See the
      multi-convolution PopLibs documentation for available flags.
  """
  def decorated(inner_func):
    def multi_conv_wrapper(*args):
      inner_options = options if options else {}

      if not isinstance(inner_options, dict):
        raise TypeError(
            "Expected the multi_conv `options` to be a `dict`, but got %s "
            "instead." % (str(inner_options)))

      option_proto = option_flag_pb2.PoplarOptionFlags()
      for key, value in inner_options.items():
        flag = option_proto.flags.add()
        flag.option = key
        flag.value = value

      def func_wrapper(*args):
        with ops.get_default_graph().as_default() as g:
          with g.gradient_override_map(_gradient_override_map):
            return inner_func(*args)

      args = functional_ops._convert_to_list(args)  # pylint: disable=protected-access
      with ops.name_scope("multi_conv") as scope:
        func_graph, captured_args = functional_ops._compile_function(  # pylint: disable=protected-access
            func_wrapper,
            args,
            scope, [],
            allow_external_captures=True)

        with ops.control_dependencies(list(func_graph.control_captures)):
          outputs = gen_functional_ops.multi_conv(
              captured_args,
              to_apply=util.create_new_tf_function(func_graph),
              Tout=func_graph.output_types,
              output_shapes=func_graph.output_shapes,
              option_flags=json_format.MessageToJson(option_proto))

      return func_graph_module.pack_sequence_as(func_graph.structured_outputs,
                                                outputs)

    return multi_conv_wrapper

  if func is not None:
    return decorated(func)

  return decorated


def _SetMlType(op, ml_type):
  attrs = xla_data_pb2.FrontendAttributes()
  attr_name = backend_config_pb2.FrontendAttributeId.Name(
      backend_config_pb2.FrontendAttributeId.ML_TYPE)
  attrs.map[attr_name] = ml_type
  serial_attrs = attrs.SerializeToString()
  op._set_attr(  # pylint: disable=protected-access
      scopes.FRONTEND_ATTRIBUTES_NAME,
      attr_value_pb2.AttrValue(s=serial_attrs))


def _SetOpAsBwd(grad):
  type_name = backend_config_pb2.MLType.Name(backend_config_pb2.TRAINING_BWD)
  _SetMlType(grad.op, type_name)


def _SetOpAsWU(grad):
  type_name = backend_config_pb2.MLType.Name(backend_config_pb2.TRAINING_WU)
  _SetMlType(grad.op, type_name)


# Override all the convolution operation gradients so that they can be annotated
# with the "ML type".
@ops.RegisterGradient("CustomConv2D")
def _CustomConv2DGrad(op, grad):
  grads = nn_grad._Conv2DGrad(op, grad)  # pylint: disable=protected-access
  assert len(grads) == 2
  _SetOpAsBwd(grads[0])
  _SetOpAsWU(grads[1])
  return grads


@ops.RegisterGradient("CustomConv2DBackpropInput")
def _CustomConv2DBackpropInputGrad(op, grad):
  grads = nn_grad._Conv2DBackpropInputGrad(op, grad)  # pylint: disable=protected-access
  assert len(grads) == 3
  _SetOpAsBwd(grads[1])
  _SetOpAsWU(grads[2])
  return grads


@ops.RegisterGradient("CustomConv2DBackpropFilter")
def _CustomConv2DBackpropFilterGrad(op, grad):
  grads = nn_grad._Conv2DBackpropFilterGrad(op, grad)  # pylint: disable=protected-access
  assert len(grads) == 3
  _SetOpAsBwd(grads[0])
  _SetOpAsWU(grads[2])
  return grads


@ops.RegisterGradient("CustomDepthwiseConv2dNative")
def _CustomDepthwiseConv2dNativeGrad(op, grad):
  grads = nn_grad._DepthwiseConv2dNativeGrad(op, grad)  # pylint: disable=protected-access
  assert len(grads) == 2
  _SetOpAsBwd(grads[0])
  _SetOpAsWU(grads[1])
  return grads


@ops.RegisterGradient("CustomDepthwiseConv2dNativeBackpropInput")
def _CustomDepthwiseConv2dNativeBackpropInputGrad(op, grad):
  grads = nn_grad._DepthwiseConv2dNativeBackpropInputGrad(op, grad)  # pylint: disable=protected-access
  assert len(grads) == 3
  _SetOpAsBwd(grads[1])
  _SetOpAsWU(grads[2])
  return grads


@ops.RegisterGradient("CustomDepthwiseConv2dNativeBackpropFilter")
def _CustomDepthwiseConv2dNativeBackpropFilterGrad(op, grad):
  grads = nn_grad._DepthwiseConv2dNativeBackpropFilterGrad(op, grad)  # pylint: disable=protected-access
  assert len(grads) == 3
  _SetOpAsBwd(grads[0])
  _SetOpAsWU(grads[2])
  return grads


@ops.RegisterGradient("CustomConv3D")
def _CustomConv3DGrad(op, grad):
  grads = nn_grad._Conv3DGrad(op, grad)  # pylint: disable=protected-access
  assert len(grads) == 2
  _SetOpAsBwd(grads[0])
  _SetOpAsWU(grads[1])
  return grads


@ops.RegisterGradient("CustomConv3DBackpropInputV2")
def _CustomConv3DBackpropInputGrad(op, grad):
  grads = nn_grad._Conv3DBackpropInputGrad(op, grad)  # pylint: disable=protected-access
  assert len(grads) == 3
  _SetOpAsBwd(grads[1])
  _SetOpAsWU(grads[2])
  return grads


@ops.RegisterGradient("CustomConv3DBackpropFilterV2")
def _CustomConv3DBackpropFilterGrad(op, grad):
  grads = nn_grad._Conv3DBackpropFilterGrad(op, grad)  # pylint: disable=protected-access
  assert len(grads) == 3
  _SetOpAsBwd(grads[0])
  _SetOpAsWU(grads[2])
  return grads


# Map the TF convolution ops to gradient wrapper functions.
_gradient_override_map = {
    "Conv2D":
    "CustomConv2D",
    "Conv2DBackpropInput":
    "CustomConv2DBackpropInput",
    "Conv2DBackpropFilter":
    "CustomConv2DBackpropFilter",
    "Conv3D":
    "CustomConv3D",
    "Conv3DBackpropInputV2":
    "CustomConv3DBackpropInputV2",
    "Conv3DBackpropFilterV2":
    "CustomConv3DBackpropFilterV2",
    "DepthwiseConv2dNative":
    "CustomDepthwiseConv2dNative",
    "DepthwiseConv2dNativeBackpropInput":
    "CustomDepthwiseConv2dNativeBackpropInput",
    "DepthwiseConv2dNativeBackpropFilter":
    "CustomDepthwiseConv2dNativeBackpropFilter"
}
