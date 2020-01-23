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
# ===================================================================
"""
Distributed training with IPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.compiler.xla import xla
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import values
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ipu import scopes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import variable_scope


class IPUMultiWorkerStrategy(distribute_lib.StrategyV1):
  """This is a distribution strategy for synchronous training using
  IPUs on multiple workers with between-graph replication.

  It places variables on the host device of each worker, and uses
  multi-worker all-reduce to to keep the variables in sync, using
  TensorFlow's implementation of collective operations over gRPC.

  It is the responsibility of the user to place the operations on the
  IPU, while this strategy will make sure that the variables are kept
  on the host and in sync between the multiple workers.

  When used during training with an `Optimizer`, this means that the
  variables will be streamed from the host to the IPU when needed,
  and that the gradients will be streamed back to the host and then
  all-reduced across the workers. Then the workers will do identical
  updates to their copies of the variables. In other words,
  `optimizer.compute_gradients()` is done on the device, while
  `optimizer.apply_gradients()` is done on the host. All the "slot"
  variables used by the optimizer (e.g. the momentum accumulator)
  are kept only in host memory and never used on the device, saving
  device memory.

  The default behavior is to sync (allreduce) the variables when
  they are written (sync-on-write). This is a good choice when
  reads are at least as common as writes. However, for variables
  where writes are more common than reads (like metrics or population
  statistics in batch normalization layers), it is beneficial to
  only sync (allreduce) the variables when they are read
  (sync-on-read). In both cases, it is important that all the workers
  participate in the sync, otherwise progress will be blocked.
  Take special care in the latter case (with sync-on-read variables),
  because it implies that all the workers need to read these variables
  at the same time. For example, it implies that all the workers must
  checkpoint the model at the same time.
  """
  def __init__(self, cluster_resolver):
    super().__init__(IPUMultiWorkerExtended(self, cluster_resolver))


def _current_device():
  return constant_op.constant(1.).device


def _is_inside_compilation():
  graph = ops.get_default_graph()
  attrs = graph._attr_scope_map  # pylint: disable=protected-access

  is_in_xla_context = control_flow_util.GraphOrParentsInXlaContext(graph)
  is_outside_compilation = scopes.OUTSIDE_COMPILATION_NAME in attrs

  return is_in_xla_context and not is_outside_compilation


class IPUSyncOnReadVariable(values.SyncOnReadVariable):  # pylint: disable=abstract-method
  pass


class IPUMirroredVariable(values.MirroredVariable):  # pylint: disable=abstract-method
  pass


class IPUMultiWorkerExtended(
    collective_all_reduce_strategy.CollectiveAllReduceExtended):
  def __init__(self, container_strategy, cluster_resolver):
    super().__init__(
        container_strategy,
        communication=cross_device_ops_lib.CollectiveCommunication.RING,
        cluster_resolver=cluster_resolver)

    devices = self._device_map.all_devices
    if len(devices) != 1:
      raise ValueError("Expected one host device per worker")
    self._host_device = devices[0]

    # The default is to force the variables on the host device.
    self._allow_variable_placement = False

  def experimental_allow_variable_placement(self):
    self._allow_variable_placement = True

  def _get_variable_creator_initial_value(self, replica_id, device,
                                          primary_var, **kwargs):
    assert replica_id == 0
    assert device is not None
    assert primary_var is None

    def initial_value_fn():  # pylint: disable=g-missing-docstring
      # Only the first device participates in the broadcast of initial values.
      group_key = self._collective_keys.get_group_key([device])
      group_size = self._num_workers
      collective_instance_key = (
          self._collective_keys.get_variable_instance_key())

      # Override colocation and XLA attributes for initializers.
      attrs = {
          "_class": attr_value_pb2.AttrValue(s=b'loc:@cpu'),
          "_XlaCompile": attr_value_pb2.AttrValue(b=False),
          "_XlaScope": attr_value_pb2.AttrValue(s=b''),
      }
      with ops.device(device), \
          ops.get_default_graph()._attr_scope(attrs):  # pylint: disable=protected-access
        initial_value = kwargs["initial_value"]
        if callable(initial_value):
          initial_value = initial_value()
        assert not callable(initial_value)
        initial_value = ops.convert_to_tensor(initial_value,
                                              dtype=kwargs.get("dtype", None))

        if self._num_workers > 1:
          if self._is_chief:
            bcast_send = collective_ops.broadcast_send(
                initial_value, initial_value.shape, initial_value.dtype,
                group_size, group_key, collective_instance_key)
            with ops.control_dependencies([bcast_send]):
              return array_ops.identity(initial_value)
          else:
            return collective_ops.broadcast_recv(initial_value.shape,
                                                 initial_value.dtype,
                                                 group_size, group_key,
                                                 collective_instance_key)
        return initial_value

    return initial_value_fn

  def _create_variable(self, next_creator, *args, **kwargs):
    colocate_with = kwargs.pop("colocate_with", None)
    if colocate_with is None:
      device_map = self._device_map
      logical_device = 0
    elif isinstance(colocate_with, numpy_dataset.SingleDevice):
      with ops.device(colocate_with.device):
        return next_creator(*args, **kwargs)
    else:
      device_map = colocate_with.device_map
      logical_device = colocate_with.logical_device

    def _real_creator(devices, *args, **kwargs):
      assert len(devices) == 1
      assert devices[0] == self._host_device

      # The chief worker will initialize and broadcast the value to
      # the other workers.
      kwargs["initial_value"] = self._get_variable_creator_initial_value(
          replica_id=0,  # First (and only) replica on each worker.
          device=self._host_device,
          primary_var=None,
          **kwargs)

      if self._allow_variable_placement:
        # Skip device placement override below.
        return [next_creator(*args, **kwargs)]

      # For sync-on-read variables we do not override the device
      # placement, allowing them to be placed on the IPU. They will
      # be transfered and reduced on the hosts only when read.
      synchronization = kwargs.get("synchronization")
      if synchronization == variable_scope.VariableSynchronization.ON_READ:
        return [next_creator(*args, **kwargs)]

      # Cache a snapshot of the variable on the current device,
      # otherwise the XLA cluster containing the ops consuming the
      # variable might be moved to the host to be colocated with it.
      kwargs["caching_device"] = _current_device()

      # In case we are inside an ipu_jit_scope, we need to override it
      # to disable XLA for variable initialization on the host.
      disable_xla = {
          "_XlaCompile": attr_value_pb2.AttrValue(b=False),
          "_XlaScope": attr_value_pb2.AttrValue(s=b''),
      }

      graph = ops.get_default_graph()
      with ops.device(self._host_device), \
          graph._attr_scope(disable_xla):  # pylint: disable=protected-access
        return [next_creator(*args, **kwargs)]

    # For tf1: use distribute_lib.create_mirrored_variable
    return values.create_mirrored_variable(
        self._container_strategy(), device_map, logical_device, _real_creator,
        IPUMirroredVariable, IPUSyncOnReadVariable, *args, **kwargs)

  def read_var(self, var):
    return var.read_value()

  def _reduce_to(self, reduce_op, value, destinations):
    if isinstance(value, values.DistributedValues):
      assert len(value.values) == 1
      value = value.values[0]

    if _is_inside_compilation():
      # Escape the compilation scope and place the reduction on the host.
      with scopes.outside_compilation_scope("reduce_to"):
        return super()._reduce_to(reduce_op, value, destinations)

    # Make sure the reduction is done on the host device
    # by wrapping the inputs in an identity op on that device.
    with ops.device(self._host_device):
      value = array_ops.identity(value, name="reduce_to")

    return super()._reduce_to(reduce_op, value, destinations)

  def _batch_reduce_to(self, reduce_op, value_destination_pairs):
    if _is_inside_compilation():
      # Escape the compilation scope and place the reduction on the host.
      with scopes.outside_compilation_scope("batch_reduce"):
        return super()._batch_reduce_to(reduce_op, value_destination_pairs)

    # Make sure the reduction is done on the host device
    # by wrapping the inputs in an identity op on that device.
    with ops.device(self._host_device):
      value_destination_pairs = [(array_ops.identity(v,
                                                     name="batch_reduce"), d)
                                 for (v, d) in value_destination_pairs]

    return super()._batch_reduce_to(reduce_op, value_destination_pairs)

  def _call_for_each_replica(self, fn, args, kwargs):
    with distribute_lib.ReplicaContext(
        self._container_strategy(), replica_id_in_sync_group=0), \
        ops.device(self._host_device):
      return fn(*args, **kwargs)

  def _validate_colocate_with_variable(self, colocate_with_variable):
    if colocate_with_variable.device != self._host_device:
      raise ValueError("Unexpected colocated variable device: {}".format(
          colocate_with_variable.device))
