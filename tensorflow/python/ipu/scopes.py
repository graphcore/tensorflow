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
Scoping contexts for IPUs
~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import numpy as np

from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops.variable_scope import variable_scope
from tensorflow.python.util import tf_contextlib
from tensorflow.compiler.plugin.poplar.driver import backend_config_pb2
from tensorflow.compiler.xla.python_api import types

FRONTEND_ATTRIBUTES_NAME = "_XlaFrontendAttributes"


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
    device: The name of the Tensorflow device, eg '/device:IPU:0'

  Returns:
    A context
  """

  with variable_scope('', use_resource=True):
    with ops.device(device):
      with ipu_jit_scope(0) as scope:
        yield scope


@tf_contextlib.contextmanager
def ipu_shard(index):
  """Control sharding for a set of operations.

  Provides a scope which targets operations onto a particular shard (IPU) of a
  multi-IPU sharded device.

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
  with ops.get_default_graph()._attr_scope(attrs):
    yield
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

  if restore_to != None and attribute_name not in saved.map:
    saved.map[attribute_name] = restore_to
    graph._attr_scope_map[FRONTEND_ATTRIBUTES_NAME] = attr_value_pb2.AttrValue(
        s=saved.SerializeToString())
  # pylint: enable=protected-access


@tf_contextlib.contextmanager
def stochastic_rounding(override):
  """Control stochastic rounding for a set of operations.

  Manually sets the stochastic rounding method to use.

  Returns:
     A context
  """
  with frontend_attribute(
      backend_config_pb2.FrontendAttributeId.Name(
          backend_config_pb2.FrontendAttributeId.STOCHASTIC_ROUNDING),
      backend_config_pb2.StochasticRounding.Name(
          backend_config_pb2.StochasticRounding.FORCE_ON
          if override else backend_config_pb2.StochasticRounding.FORCE_OFF),
      backend_config_pb2.StochasticRounding.Name(
          backend_config_pb2.StochasticRounding.NOT_SET)):
    yield


@tf_contextlib.contextmanager
def partials_type(override_type):
  """Override the default type used to store intermediate results by some operations.

  Args:
    override_type: Numpy type of the partials (float16 or float32)

  Returns:
     A context
  """
  xla_type = types.MAP_DTYPE_TO_RECORD[str(
      np.dtype(override_type))].primitive_type
  if xla_type not in [
      xla_data_pb2.PrimitiveType.F16, xla_data_pb2.PrimitiveType.F32
  ]:
    raise ValueError("Only support float16, float32, provided %s" % dtype)
  with frontend_attribute(
      backend_config_pb2.FrontendAttributeId.Name(
          backend_config_pb2.FrontendAttributeId.PARTIALS_TYPE),
      xla_data_pb2.PrimitiveType.Name(xla_type),
      xla_data_pb2.PrimitiveType.Name(
          xla_data_pb2.PrimitiveType.PRIMITIVE_TYPE_INVALID)):
    yield
