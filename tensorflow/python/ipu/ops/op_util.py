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
Utilities for IPU ops
~~~~~~~~~~~~~~~~~~~~~
"""
import six

from tensorflow.compiler.plugin.poplar.driver import backend_config_pb2
from tensorflow.compiler.plugin.poplar.driver import threestate_pb2
from tensorflow.compiler.plugin.poplar.ops import gen_functional_ops
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.ops import math_ops
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ipu import functional_ops
from tensorflow.python.ipu import scopes
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import math_grad
from tensorflow.python.ops import nn_grad
from tensorflow.python.util import tf_contextlib


def SetMlType(op, ml_type):
  if context.executing_eagerly():
    return op
  if ml_type:
    operation = op if isinstance(op, ops.Operation) else op.op
    attrs = xla_data_pb2.FrontendAttributes()
    attr_name = backend_config_pb2.FrontendAttributeId.Name(
        backend_config_pb2.FrontendAttributeId.ML_TYPE)
    attrs.map[attr_name] = backend_config_pb2.MLType.Name(ml_type)
    serial_attrs = attrs.SerializeToString()
    operation._set_attr(  # pylint: disable=protected-access
        scopes.FRONTEND_ATTRIBUTES_NAME,
        attr_value_pb2.AttrValue(s=serial_attrs))
  return op


def SetOpAsFwd(op):
  return SetMlType(op, backend_config_pb2.TRAINING_FWD)


def SetOpAsBwd(op):
  return SetMlType(op, backend_config_pb2.TRAINING_BWD)


def SetOpAsWU(op):
  return SetMlType(op, backend_config_pb2.TRAINING_WU)


# Override all the convolution operation gradients so that they can be annotated
# with the "ML type".
@ops.RegisterGradient("CustomConv2D")
def _CustomConv2DGrad(op, grad):
  grads = nn_grad._Conv2DGrad(op, grad)  # pylint: disable=protected-access
  assert len(grads) == 2
  SetOpAsFwd(op)
  SetOpAsBwd(grads[0])
  SetOpAsWU(grads[1])
  return grads


@ops.RegisterGradient("CustomConv2DBackpropInput")
def _CustomConv2DBackpropInputGrad(op, grad):
  grads = nn_grad._Conv2DBackpropInputGrad(op, grad)  # pylint: disable=protected-access
  assert len(grads) == 3
  SetOpAsFwd(op)
  SetOpAsBwd(grads[1])
  SetOpAsWU(grads[2])
  return grads


@ops.RegisterGradient("CustomConv2DBackpropFilter")
def _CustomConv2DBackpropFilterGrad(op, grad):
  grads = nn_grad._Conv2DBackpropFilterGrad(op, grad)  # pylint: disable=protected-access
  assert len(grads) == 3
  SetOpAsFwd(op)
  SetOpAsBwd(grads[0])
  SetOpAsWU(grads[2])
  return grads


@ops.RegisterGradient("CustomDepthwiseConv2dNative")
def _CustomDepthwiseConv2dNativeGrad(op, grad):
  grads = nn_grad._DepthwiseConv2dNativeGrad(op, grad)  # pylint: disable=protected-access
  assert len(grads) == 2
  SetOpAsFwd(op)
  SetOpAsBwd(grads[0])
  SetOpAsWU(grads[1])
  return grads


@ops.RegisterGradient("CustomDepthwiseConv2dNativeBackpropInput")
def _CustomDepthwiseConv2dNativeBackpropInputGrad(op, grad):
  grads = nn_grad._DepthwiseConv2dNativeBackpropInputGrad(op, grad)  # pylint: disable=protected-access
  assert len(grads) == 3
  SetOpAsFwd(op)
  SetOpAsBwd(grads[1])
  SetOpAsWU(grads[2])
  return grads


@ops.RegisterGradient("CustomDepthwiseConv2dNativeBackpropFilter")
def _CustomDepthwiseConv2dNativeBackpropFilterGrad(op, grad):
  grads = nn_grad._DepthwiseConv2dNativeBackpropFilterGrad(op, grad)  # pylint: disable=protected-access
  assert len(grads) == 3
  SetOpAsFwd(op)
  SetOpAsBwd(grads[0])
  SetOpAsWU(grads[2])
  return grads


@ops.RegisterGradient("CustomConv3D")
def _CustomConv3DGrad(op, grad):
  grads = nn_grad._Conv3DGrad(op, grad)  # pylint: disable=protected-access
  assert len(grads) == 2
  SetOpAsFwd(op)
  SetOpAsBwd(grads[0])
  SetOpAsWU(grads[1])
  return grads


@ops.RegisterGradient("CustomConv3DBackpropInputV2")
def _CustomConv3DBackpropInputGrad(op, grad):
  grads = nn_grad._Conv3DBackpropInputGrad(op, grad)  # pylint: disable=protected-access
  assert len(grads) == 3
  SetOpAsFwd(op)
  SetOpAsBwd(grads[1])
  SetOpAsWU(grads[2])
  return grads


@ops.RegisterGradient("CustomConv3DBackpropFilterV2")
def _CustomConv3DBackpropFilterGrad(op, grad):
  grads = nn_grad._Conv3DBackpropFilterGrad(op, grad)  # pylint: disable=protected-access
  assert len(grads) == 3
  SetOpAsFwd(op)
  SetOpAsBwd(grads[0])
  SetOpAsWU(grads[2])
  return grads


def conv_gradients_override_map():
  return {
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


def HandleMatMulGrads(grads):
  assert len(grads) == 2

  # Batched matmul might have batch dimension reductions.
  def look_through_reshape_reduce(output):
    if output.op.type == "Reshape":
      output = output.op.inputs[0]
    if output.op.type == "Sum":
      output = output.op.inputs[0]
    return output

  SetOpAsBwd(look_through_reshape_reduce(grads[0]))
  SetOpAsWU(look_through_reshape_reduce(grads[1]))
  return grads


@ops.RegisterGradient("CustomMatMul")
def _CustomMatMulGrad(op, grad):
  grads = math_grad._MatMulGrad(op, grad)  # pylint: disable=protected-access
  SetOpAsFwd(op)
  return HandleMatMulGrads(grads)


@ops.RegisterGradient("CustomBatchMatMul")
def _CustomBatchMatMulGrad(op, grad):
  grads = math_grad._BatchMatMul(op, grad)  # pylint: disable=protected-access
  SetOpAsFwd(op)
  return HandleMatMulGrads(grads)


@ops.RegisterGradient("CustomBatchMatMulV2")
def _CustomBatchMatMulV2Grad(op, grad):
  grads = math_grad._BatchMatMulV2(op, grad)  # pylint: disable=protected-access
  SetOpAsFwd(op)
  return HandleMatMulGrads(grads)


def matmul_gradients_override_map():
  return {
      "MatMul": "CustomMatMul",
      "BatchMatMul": "CustomBatchMatMul",
      "BatchMatMulV2": "CustomBatchMatMulV2",
  }


def gradients_override_map():
  return {**conv_gradients_override_map(), **matmul_gradients_override_map()}


@tf_contextlib.contextmanager
def gradient_override_scope(training):
  """Scope which configures any operations which need to be aware of whether
  they are an operation in forward or backward propagation, and if the latter,
  make sure that the gradient operations are annotated as a gradient with
  respect to activations or as a gradient with respect to the weights.

  Args:
    training: whether this is a training graph.

  Returns:
     A context
  """
  with scopes.frontend_attribute(
      backend_config_pb2.FrontendAttributeId.Name(backend_config_pb2.ML_TYPE),
      backend_config_pb2.MLType.Name(
          backend_config_pb2.TRAINING_FWD if training else backend_config_pb2.
          INFERENCE_FWD)):
    with ops.get_default_graph().as_default() as g:
      with g.gradient_override_map(gradients_override_map()):
        yield


def get_accumulator_dtype(variable, dtype_override):
  """Get the accumulator dtype for the given variable."""
  if dtype_override is None:
    return variable.dtype

  # Note that a `DType` is callable, so only try to call it if validation fails.
  try:
    return dtypes.as_dtype(dtype_override)
  except TypeError:
    if callable(dtype_override):
      return dtypes.as_dtype(dtype_override(variable))
    else:
      raise


_activation_modules = set(
    ['tensorflow.python.keras.activations', 'tensorflow.python.ops.math_ops'])


def get_activation_name(identifier):
  "Get activation name from string or activation function object"
  if isinstance(identifier, six.string_types):
    return identifier
  elif callable(identifier):
    if identifier.__module__ not in _activation_modules:
      raise TypeError('Unrecognized function : '
                      f'{identifier.__module__}.{identifier.__name__}')
    return identifier.__name__

  raise TypeError(
      f'Could not interpret activation function identifier: {repr(identifier)}'
  )


def bool_to_three_state(value, default=None):
  if value is None:
    return default if default else threestate_pb2.ThreeState.Name(
        threestate_pb2.THREESTATE_UNDEFINED)
  elif value:
    return threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_ON)
  return threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_OFF)


def accumulate_gradients(grads_and_vars,
                         gradient_accumulation_dtype,
                         accum_scale=1.0,
                         grad_scale=None):
  """
  Create ops that accumulate the gradients in `grads_and_vars`.
  Returns gradient accumulation ops which, when executed, accumulate gradients
  onto their gradient accumulation buffers.

  Args:
    grads_and_vars: List of (gradient, variable) pairs as returned by
      an optimizer's compute_gradients() function.
    gradient_accumulation_dtype: The data type used for the gradient
      accumulation buffer. One of:
    grad_scale: Value to scale gradients with.

      - `None`: Use an accumulator of the same type as the variable type.
      - A `DType`: Use this type for all the accumulators.
      - A callable that takes the variable and returns a `DType`: Allows
        specifying the accumulator type on a per-variable basis.
  """
  accumulated_grads_and_vars = []
  for grad, var in grads_and_vars:
    if grad is not None:
      with ops.colocate_with(grad):
        # Find the data type for the accumulator.
        dtype = get_accumulator_dtype(var, gradient_accumulation_dtype)

        # Create an accumulator - variable is used as reference for shape/layout.
        accumulator = gen_poputil_ops.gradient_accumulator_create(
            var, output_type=dtype)

        # Add the gradients to the accumulator.
        if grad_scale is not None:
          if grad.dtype == dtypes.float32:
            grad = grad * grad_scale
          else:
            grad_dtype = grad.dtype
            grad = math_ops.cast(
                math_ops.cast(grad, dtypes.float32) * grad_scale, grad_dtype)
            # These casts are used to leave the gradient scale in float32, and will be removed
            # in case 4/10 in SerializeGradientAccumulate before being passed to the
            # ScaledInplaceXbY and ScaledInplaceaXbY ops

        accumulator = gen_poputil_ops.gradient_accumulator_add_with_scale(
            accumulator, grad, accum_scale)

        # Sink the accumulators.
        grad = gen_poputil_ops.gradient_accumulator_sink(accumulator)
    # Use the accumulated gradients.
    accumulated_grads_and_vars.append((grad, var))
  return accumulated_grads_and_vars


def create_resource_update(fn, name, control_outputs,
                           offload_weight_update_variables,
                           replicated_optimizer_state_sharding,
                           gradient_accumulation_count):
  """
  Creates a resource update computation in the XLA graph from the Python
  function `fn`.
  Returns the computation outputs.

  Args:
    - fn: A Python callable which will be converted into the XLA resource
          update.
    - name: Base name of the XLA resource update.
    - control_outputs: Operations which must have been executed before the XLA
      resource update can be said to have been executed.
    - offload_weight_update_variables: When enabled, the `tf.Variable`s that
      are only used by the XLA resource update generated by this function will
      be stored in remote memory. During the weight update, these variables will
      be streamed onto the device and then streamed back to remote memory after
      they have been updated. Requires the machine to be configured with support
      for `Poplar remote buffers`. Offloading variables into remote memory can
      reduce maximum memory liveness, but can also increase the computation time
      of the weight update. When set to `None` the variables will be placed in
      either in-processor or remote memory automatically based on the current
      best placement.
    - replicated_optimizer_state_sharding: If True, any `tf.Variable` which is
      offloaded via `offload_weight_update_variables` will be partitioned across
      the replicas. This can exploit the additional bandwidth of the IPU-Links
      and the parallelism of the replicas to improve overall throughput.

  """
  offload_weight_update_variables = bool_to_three_state(
      offload_weight_update_variables,
      threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_UNDEFINED))
  replicated_optimizer_state_sharding = bool_to_three_state(
      replicated_optimizer_state_sharding,
      default=offload_weight_update_variables)

  rts_on = replicated_optimizer_state_sharding != 'THREESTATE_OFF'
  offload_on = offload_weight_update_variables != 'THREESTATE_OFF'
  if rts_on and not offload_on:
    raise ValueError(
        "To use replicated_optimizer_state_sharding, optimizer state must be"
        " offloaded by using offload_weight_update_variables.")

  # Create the resource update and lower the function into XLA.
  with ops.name_scope(name + "/WU") as scope:
    func_graph, captured_args, constant_outputs = \
      functional_ops._compile_function(  # pylint: disable=protected-access
          fn, [gradient_accumulation_count], scope, control_outputs, True)

  # Add a call op to the graph that calls the resource update.
  with ops.control_dependencies(list(func_graph.control_captures)):
    outputs = gen_functional_ops.resource_update(
        captured_args,
        to_apply=util.create_new_tf_function(func_graph),
        Tout=func_graph.output_types,
        output_shapes=func_graph.output_shapes,
        offload_weight_update_variables=offload_weight_update_variables,
        replicated_optimizer_state_sharding=replicated_optimizer_state_sharding
    )
    outputs = functional_ops._replace_outputs(outputs, constant_outputs)  # pylint: disable=protected-access

  return outputs
