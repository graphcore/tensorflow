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
# =============================================================================
"""
Scoping contexts
~~~~~~~~~~~~~~~~
"""
import numpy as np

from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ipu import sharding
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops.variable_scope import variable_scope
from tensorflow.python.util import tf_contextlib
from tensorflow.compiler.plugin.poplar.driver import backend_config_pb2
from tensorflow.compiler.plugin.poplar.driver import threestate_pb2
from tensorflow.compiler.xla.python_api import types

FRONTEND_ATTRIBUTES_NAME = "_XlaFrontendAttributes"
OUTSIDE_COMPILATION_NAME = "_xla_outside_compilation"


@tf_contextlib.contextmanager
def ipu_jit_scope(ipu_scope):
  """Provides a scope for compilation of operations.

  If you would like to compile several sets of operations together, then this
  can provide that mechanism.

  Args:
    ipu_scope: A name to differentiate between different JIT scopes

  Returns:
     A context
  """

  scope = "jit_scope_ipu_" + str(ipu_scope)
  attrs = {
      "_XlaCompile": attr_value_pb2.AttrValue(b=True),
      "_XlaSeparateCompiledGradients": attr_value_pb2.AttrValue(b=False),
      "_XlaScope": attr_value_pb2.AttrValue(s=scope.encode())
  }

  # pylint: disable=protected-access
  with ops.get_default_graph()._attr_scope(attrs):
    yield
  # pylint: enable=protected-access


@tf_contextlib.contextmanager
def ipu_scope(device):
  """Provides a scope for placing operations onto a particular IPU/IPU cluster.

  Args:
    device: The name of the TensorFlow device, such as '/device:IPU:0'

  Returns:
    A context
  """

  with variable_scope('', use_resource=True):
    with ops.device(device):
      with ipu_jit_scope(0) as scope:
        yield scope


@tf_contextlib.contextmanager
def outside_compilation_scope(name="outside"):
  """Provides a scope for placing operations on the host, outside the current
  compilation scope. The operations will be placed on the default host device.
  This allows for offloading computations from the IPU to the host, which can
  be useful for operations that are not supported or suitable for execution on
  the IPU.

  Example:

  .. code-block:: python

    def my_net(a):
      with ipu_scope("/device:IPU:0"):
        b = a * a
        with outside_compilation_scope():
          c = b + 2  # Placed on the host.
        d = b + c
        return d

  Args:
    name: A name for the outside compilation scope.

  Returns:
    A context
  """
  graph = ops.get_default_graph()

  if not control_flow_util.GraphOrParentsInXlaContext(graph):
    raise ValueError(
        "outside_compilation_scope is only allowed in XLA context")

  current_attrs = graph._attr_scope_map  # pylint: disable=protected-access
  if OUTSIDE_COMPILATION_NAME in current_attrs:
    raise ValueError("Illegal nesting of outside_compilation_scope")

  unique_name = graph.unique_name(name, mark_as_used=True)
  attr_value = attr_value_pb2.AttrValue(s=unique_name.encode())
  attrs = {OUTSIDE_COMPILATION_NAME: attr_value}

  # Use a name scope to reduce the risk of op name collisions when
  # moving ops from the current graph to the outside graph.
  with ops.name_scope(unique_name), \
      graph._attr_scope(attrs):  # pylint: disable=protected-access
    yield


@tf_contextlib.contextmanager
def ipu_shard(index):
  """Control sharding for a set of operations.

  Provides a scope which targets operations onto a particular shard (IPU) of a
  multi-IPU sharded device. Gradients created from these operations will
  also be put onto the same shard. Consequently an `ipu_shard` scope
  enclosing a call to `tf.gradients` or `tf.GradientTape.gradient` won't change
  the sharding of the backwards ops.

  Args:
    index: The index of the IPU on which to place the enclosed operations.

  Returns:
     A context
  """

  if hasattr(index, '__iter__'):
    ipus = index
  else:
    ipus = [index]

  proto = xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.MAXIMAL,
                                  tile_assignment_devices=ipus)
  attr_value = attr_value_pb2.AttrValue(s=proto.SerializeToString())
  attrs = {"_XlaSharding": attr_value}

  # pylint: disable=protected-access

  old_operations = ops.get_default_graph().get_operations()

  with ops.get_default_graph()._attr_scope(attrs):
    yield

  operations = ops.get_default_graph().get_operations()
  new_operations = [op for op in operations if op not in old_operations]
  # Make sure that the gradients of the new Ops also have the same sharding.
  # There's a context manager that does the same thing, but this way we dont
  # accidentally override what would have been the intended gradient function.
  _make_gradients_sharded(new_operations)

  # pylint: enable=protected-access


@tf_contextlib.contextmanager
def frontend_attribute(attribute_name, attribute_value, restore_to=None):
  """Sets the specified scope attribute to the specified value in the graph.

  Args:
    attribute_name:  Name of the attribute.
    attribute_value: Attribute's value as a string.
    restore_to:      If at the end of the scope the attribute was to be
                     undefined sets it to this value instead.

  Returns:
    A context
  """

  saved = xla_data_pb2.FrontendAttributes()
  proto = xla_data_pb2.FrontendAttributes()
  # pylint: disable=protected-access
  attr_value = ops.get_default_graph()._attr_scope_map.get(
      FRONTEND_ATTRIBUTES_NAME)
  # pylint: enable=protected-access
  if attr_value:
    proto.ParseFromString(attr_value.s)
    saved.ParseFromString(attr_value.s)
  proto.map[attribute_name] = attribute_value

  attr_value = attr_value_pb2.AttrValue(s=proto.SerializeToString())
  attrs = {FRONTEND_ATTRIBUTES_NAME: attr_value}

  # pylint: disable=protected-access
  graph = ops.get_default_graph()
  with graph._attr_scope(attrs):
    yield

  if restore_to is not None and attribute_name not in saved.map:
    saved.map[attribute_name] = restore_to
    graph._attr_scope_map[FRONTEND_ATTRIBUTES_NAME] = attr_value_pb2.AttrValue(
        s=saved.SerializeToString())
  # pylint: enable=protected-access


@tf_contextlib.contextmanager
def stochastic_rounding(override):
  """Control stochastic rounding for a set of operations.

  EXPERIMENTAL - there are no guarantees that the stochastic rounding provided
  will be used and therefore this should not be used.

  Args:
    override: if True then stochastic rounding will be used, otherwise it will
      be disabled for this set of operations.

  Returns:
     A context
  """
  with frontend_attribute(
      backend_config_pb2.FrontendAttributeId.Name(
          backend_config_pb2.STOCHASTIC_ROUNDING),
      threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_ON if override
                                     else threestate_pb2.THREESTATE_OFF),
      threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_UNDEFINED)):
    yield


@tf_contextlib.contextmanager
def partials_type(override_type):
  """Override the default type used to store intermediate results by
  convolution and matrix mutliply operations.

  EXPERIMENTAL - there are no guarantees that the partials type provided will be
  used and therefore this should not be used.

  Args:
    override_type: Numpy type of the partials (float16 or float32)

  Returns:
     A context
  """
  xla_type = types.MAP_DTYPE_TO_RECORD[str(
      np.dtype(override_type))].primitive_type
  if xla_type not in [xla_data_pb2.F16, xla_data_pb2.F32]:
    raise ValueError("Only support float16, float32, provided %s" %
                     np.dtype(override_type))
  with frontend_attribute(
      backend_config_pb2.FrontendAttributeId.Name(
          backend_config_pb2.PARTIALS_TYPE),
      xla_data_pb2.PrimitiveType.Name(xla_type),
      xla_data_pb2.PrimitiveType.Name(xla_data_pb2.PRIMITIVE_TYPE_INVALID)):
    yield


class _ShardedGradientWrapper:  # pylint: disable=missing-type-doc,missing-param-doc
  """ Utility type for wrapping a call to an ops gradient function.
  """
  def __init__(self, gradient_fn):
    self._gradient_fn = gradient_fn

  def __call__(self, op, *args, **kwargs):
    shard = sharding.get_sharding(op)
    assert shard is not None, "Expected forward Op to be sharded."

    with ipu_shard(shard):
      return self._gradient_fn(op, *args, **kwargs)


def _make_gradients_sharded(operations):
  for op in operations:
    gradient_fn = _get_gradient_function(op)
    has_gradient = gradient_fn is not None

    # Pipeline backward stage cant be explicitly sharded, its implicitly
    # sharded based on the corresponding forward stage.
    if has_gradient and op.type != "PipelineStage":

      # An Op might already have a shared gradient if ipu_shard scopes
      # are nested.
      already_sharded = isinstance(gradient_fn, _ShardedGradientWrapper)
      if not already_sharded:
        # Using _ShardedGradientWrapper instead of a lambd/functools.partial so we can
        # easily check whether something is already wrapped.
        # There are several ways of setting the gradient function for an Op
        # but this method has the highest precedence.
        op._gradient_function = _ShardedGradientWrapper(  # pylint: disable=protected-access
            gradient_fn)


def _get_gradient_function(op):
  try:
    gradient_fn = ops.get_gradient_function(op)
  except LookupError:
    return None

  return gradient_fn
