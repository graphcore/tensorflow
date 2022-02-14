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
General utilities
~~~~~~~~~~~~~~~~~
"""
import os
import time
import numpy as np

from tensorflow.compiler.plugin.poplar.driver.config_pb2 import IpuOptions
# Adds the enum SyntheticDataCategory into the scope so it can be imported from
# this file. It is required for calling use_synthetic_data_for.
from tensorflow.compiler.plugin.poplar.driver.config_pb2 import SyntheticDataCategory  # pylint: disable=W0611
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.driver import config_pb2
from tensorflow.compiler.plugin.poplar.driver import threestate_pb2
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
# pylint: disable=unused-import
# These imports are only here to make it easier for the Tensorflow Wheel users
# to use these functions:
# ```
# from tensorflow.python import ipu
# ...
# ipu.utils.export_variables_from_live_session(...)
# ```
from tensorflow.compiler.plugin.poplar.tools.tensorflow_weights_extractor import (
    export_variables_from_live_session, export_variables_from_live_model,
    import_data_in_live_session, import_data_in_live_model)
from tensorflow.compat.v1 import executing_eagerly
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import values
from tensorflow.python.framework import ops
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import dataset_extractor
from tensorflow.python.ipu.config import IPUConfig, SelectionOrder, ExecutionProfileType, DeviceConnectionType, MergeRemoteBuffersBehaviour, SchedulingAlgorithm, get_ipu_config, configure_ipu_system
# pylint: enable=unused-import
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import deprecation


def get_num_of_ipus_in_device(ipu_device, device="cpu"):
  """Get the number of physical IPUs

  Args:
    ipu_device: The IPU device for which to get the number of devices for.
    device: The CPU device which is local to the IPU hardware.

  Returns:
    A number of physical IPUs configured for a particular TF device.
  """

  g = ops.Graph()
  with g.as_default():
    with ops.device(device):
      cfg_op = gen_ipu_ops.ipu_get_num_devices(ipu_device)

  with session_lib.Session(graph=g) as sess:
    return sess.run(cfg_op)


def running_on_ipu_model():
  """ Check if XLA is configured to run on the ipu model.

  Returns:
    True if XLA is configured to run on the ipu model.
    False if XLA is configured to run on real hardware.
  """
  return "--use_ipu_model" in os.environ.get("TF_POPLAR_FLAGS", "")


def reset_ipu_seed(seed,
                   device="/device:IPU:0",
                   cpu_device="cpu",
                   experimental_identical_replicas=False):
  """Reset the seed used to generate stateful random numbers and perform
  stochastic rounding.

  Args:
    seed: The new random number generator seed.
    device: The device to which the seed will be applied.
    cpu_device: The CPU device which is on the same hardware to the IPU device.
    experimental_identical_replicas: Whether to seed all the local replicas
      identically. Note that to generate identical sequences of random numbers
      on all replicas, the Poplar engine option `"target.deterministicWorkers"`
      must also be set to `"portable"`. Also note that for multi-replica
      distribution with multiple processes, the same seed must be passed to
      each process to ensure that all the replicas globally get the same seed.
      WARNING: This flag is experimental and subject to change.

  Returns:
    None
  """
  g = ops.Graph()
  with g.as_default():
    with ops.device(cpu_device):
      cfg_op = gen_ipu_ops.ipu_reset_seed(
          device, seed, identical_replicas=experimental_identical_replicas)

  with session_lib.Session(graph=g) as sess:
    sess.run(cfg_op)


def move_variable_initialization_to_cpu(graph=None):
  """For all variables in the VARIABLES collection, move any initialization
  ops onto the CPU.

  Args:
    graph: Operations are moved around on this graph.  The default graph will be
           used if not specified.

  Returns:
    None
  """
  if not graph:
    graph = ops.get_default_graph()

  with ops.device("/device:CPU:0"):
    control_flow_ops.no_op(name="cpu")
  variables = []
  for v in graph.get_collection('variables'):
    # We assume a distribution strategy knows better how to
    # initialize its own variables, so skip those.
    if not isinstance(v, values.DistributedVariable):
      variables.append(v)

  def _uses_resource(op):
    """ Helper to determine if an op uses a resource """
    return any(input_tensor.dtype == 'resource' for input_tensor in op.inputs)

  init_ops = []
  dep_ops = [v.initializer.inputs[1].op for v in variables]
  visited = set()

  # Depth-first search up the graph starting from all variables in VARIABLES
  # Place all touched ops on the CPU, but do not touch or search ops that use
  # resource tensors, otherwise device colocation could be violated.
  while dep_ops:
    op = dep_ops.pop()
    if op not in visited and not _uses_resource(op):
      visited.add(op)
      init_ops += [op]
      dep_ops += [x.op for x in op.inputs]

  # pylint: disable=protected-access
  for op in init_ops:
    op._set_device('/device:CPU:0')
    op._set_attr(
        '_class',
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(
            s=[b'loc:@cpu'])))
    op._set_attr('_XlaCompile', attr_value_pb2.AttrValue(b=False))
    op._set_attr('_XlaScope', attr_value_pb2.AttrValue(s=b''))
  # pylint: enable=protected-access

  return


def export_dataset_to_file(dataset_or_infeed,
                           output_filename,
                           num_elements,
                           feed_name="",
                           apply_options=True):
  """Export as binary `num_elements` from the given `infeed` to the specified
  `output_filename`.

  If the infeed elements are tuples then one file per tuple element will be
  created.
  For example, if `dataset` looks like

  .. code-block:: python

    [{ "a": A_0, "b": B_0}, { "a": A_1, "b": B_1}, ...]

  then `export_dataset_to_file(dataset, "my_dataset.bin", 100)` will generate:

  .. code-block:: python

    my_dataset.0.bin   # Contains tensors [ A_0, A_1, ..., A_99]
    my_dataset.1.bin   # Contains tensors [ B_0, B_1, ..., B_99]

  Args:
    dataset_or_infeed: An unary dataset with the same input and output
      structure or an `IPUInfeedQueue`.
    output_filename: Where to export the tensors to.
    num_elements: Number of elements to export from the dataset.
    feed_name: Specify the feed name.
    apply_options: Whether to apply optimization options which can improve the
      dataset performance.
  """
  assert isinstance(dataset_or_infeed,
                    (dataset_ops.Dataset, ipu_infeed_queue.IPUInfeedQueue))
  if isinstance(dataset_or_infeed, ipu_infeed_queue.IPUInfeedQueue):
    dataset = dataset_or_infeed._dataset  # pylint: disable=protected-access
    feed_name = feed_name or dataset_or_infeed._id  # pylint: disable=protected-access
  else:
    dataset = dataset_or_infeed
  if apply_options:
    dataset = dataset._apply_options()  # pylint: disable=protected-access

  extractor = dataset_extractor.dataset_extractor(dataset, num_elements,
                                                  output_filename, feed_name)
  with ops.device("cpu"), session_lib.Session() as sess:
    sess.run(extractor)


def export_inputs_to_file(inputs, output_filename, feed_dict):
  """Export as binary the list of `inputs` provided to the specified
  `output_filename`.

  Args:
    inputs: List of graph inputs to export.
    output_filename: Where to export the tensors to.
    feed_dict: Feed dictionary containing the inputs' values.
  """

  with ops.device("cpu"), session_lib.Session() as sess:
    names = [i.name for i in inputs]
    sess.run(
        dataset_extractor.export_variables(inputs, names, output_filename),
        feed_dict)


def use_synthetic_data_for(synthetic_data_category):
  """Get whether synthetic data is being used for the given category.

  Args:
    synthetic_data_category: A SyntheticDataCategory enum value.

  Returns:
    A bool indicating the result.
  """

  op = gen_ipu_ops.ipu_use_synthetic_data_for(
      synthetic_data_category=synthetic_data_category)

  if executing_eagerly():
    return op.numpy()[0]
  return session_lib.Session().run(op)[0]
