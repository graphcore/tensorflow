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
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import values
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ipu import scopes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import variable_scope


class IPUMultiWorkerStrategy(distribute_lib.StrategyV1):
  """This is a distribution strategy for synchronous training using
  IPUs on multiple workers with between-graph replication.

  By default variables and ops are placed on the IPU of each worker,
  but variables can optionally be placed on the host by setting
  `variables_on_host=True`. In any case, this strategy will make
  sure that variables are kept in sync between the workers by
  performing multi-worker reductions.

  The multi-worker reductions are done using TensorFlow's
  implementation of collective operations over gRPC.

  **Variable synchronization**

  The default behavior is to sync (allreduce) the variables when
  they are written (sync-on-write). This is a good choice when
  reads are at least as common as writes. However, for variables
  where writes are more common than reads (like metrics or population
  statistics in batch normalization layers), it is beneficial to
  only sync (allreduce) the variables when they are read
  (sync-on-read).

  In both cases, it is important that all the workers participate
  in the sync, otherwise progress will be blocked. Take special care
  in the latter case (with sync-on-read variables), because it implies
  that all the workers need to read these variables at the same time.
  For example, it implies that all the workers must checkpoint the
  model at the same time.

  Sync-on-read variables are placed on the IPU even when variables
  were requested placed on the host (with `variables_on_host=True`),
  because it allows the ops to update the variables directly on the
  IPU without any host involvement. Only when the variable is read,
  it is streamed to the host and allreduced there.

  **Weight updates**

  When used during training with an `Optimizer`, there is an implicit
  allreduce in the `optimizer.apply_gradients()` function (which is
  called from `optimizer.minimize()`). This will automatically cause
  the gradients to be streamed to the host of each worker, allreduced
  between the workers, and then streamed back to the IPU of each worker,
  where identical weight updates are performed (keeping the workers in
  sync).

  When variables are placed on the host, the weight updates should
  also be placed on the host. In other words, the
  `optimizer.compute_gradients()` call should be placed on the IPU,
  while the `optimizer.apply_gradients()` call should be placed
  on the host. This must be done explicitly. In this scenario all
  the "slot" variables used by the optimizer (e.g. the momentum
  accumulator) are then also kept only in host memory and never
  used on the IPU, saving IPU memory.
  """
  def __init__(self,
               cluster_resolver,
               ipu_device="/device:IPU:0",
               variables_on_host=False):
    super().__init__(
        IPUMultiWorkerExtended(self, cluster_resolver, ipu_device,
                               variables_on_host))


def _is_inside_compilation():
  graph = ops.get_default_graph()
  attrs = graph._attr_scope_map  # pylint: disable=protected-access

  is_in_xla_context = control_flow_util.GraphOrParentsInXlaContext(graph)
  is_outside_compilation = scopes.OUTSIDE_COMPILATION_NAME in attrs

  return is_in_xla_context and not is_outside_compilation


def _ipu_device_for_host(ipu_device_string, host_device_string):
  ipu_device = device_lib.DeviceSpec.from_string(ipu_device_string)
  host_device = device_lib.DeviceSpec.from_string(host_device_string)

  # Take distributed info from the host and device info from the IPU.
  ipu_for_host = device_lib.DeviceSpec(job=host_device.job,
                                       replica=host_device.replica,
                                       task=host_device.task,
                                       device_type=ipu_device.device_type,
                                       device_index=ipu_device.device_index)

  return ipu_for_host.to_string()


class IPUSyncOnReadVariable(values.SyncOnReadVariable):  # pylint: disable=abstract-method
  pass


class IPUMirroredVariable(values.MirroredVariable):  # pylint: disable=abstract-method
  pass


class IPUMultiWorkerExtended(
    collective_all_reduce_strategy.CollectiveAllReduceExtended):
  def __init__(self, container_strategy, cluster_resolver, ipu_device,
               variables_on_host):
    super().__init__(
        container_strategy,
        communication=cross_device_ops_lib.CollectiveCommunication.RING,
        cluster_resolver=cluster_resolver)

    host_devices = self._device_map.all_devices
    if len(host_devices) != 1:
      raise ValueError("Expected one host device per worker")

    self._host_device = host_devices[0]
    self._ipu_device = _ipu_device_for_host(ipu_device, self._host_device)
    self._variables_on_host = variables_on_host

    if variables_on_host:
      self._variable_device = self._host_device
    else:
      self._variable_device = self._ipu_device

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
      device_map = values.ReplicaDeviceMap([self._variable_device])
      logical_device = 0
    elif isinstance(colocate_with, numpy_dataset.SingleDevice):
      with ops.device(colocate_with.device):
        return next_creator(*args, **kwargs)
    else:
      device_map = colocate_with.device_map
      logical_device = colocate_with.logical_device

    def _real_creator(devices, *args, **kwargs):
      assert len(devices) == 1
      assert devices[0] == self._variable_device

      # The chief worker will initialize and broadcast the value to
      # the other workers. Always done on the host.
      kwargs["initial_value"] = self._get_variable_creator_initial_value(
          replica_id=0,  # First (and only) replica on each worker.
          device=self._host_device,
          primary_var=None,
          **kwargs)

      # We always place sync-on-read variables on the IPU. They will
      # be transfered and reduced on the hosts only when read.
      synchronization = kwargs.get("synchronization")
      if (not self._variables_on_host or
          synchronization == variable_scope.VariableSynchronization.ON_READ):
        with ops.device(self._ipu_device):
          return [next_creator(*args, **kwargs)]

      # Cache a snapshot of the variable on the IPU device,
      # otherwise the XLA cluster containing the ops consuming the
      # variable might be moved to the host to be colocated with it.
      kwargs["caching_device"] = self._ipu_device

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

    # For tf2: use values.create_mirrored_variable
    return distribute_lib.create_mirrored_variable(
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
        ops.device(self._ipu_device):
      return fn(*args, **kwargs)

  def _validate_colocate_with_variable(self, colocate_with_variable):
    if colocate_with_variable.device != self._variable_device:
      raise ValueError("Unexpected colocated variable device: {}".format(
          colocate_with_variable.device))
