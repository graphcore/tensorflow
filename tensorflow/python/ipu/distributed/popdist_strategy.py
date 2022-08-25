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
import popdist

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ipu.ops import cross_replica_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import server_lib
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.ipu.distributed import Sum, Average, \
    allreduce as hvd_allreduce, \
    broadcast as hvd_broadcast


def _to_horovod_op(reduce_op):
  if reduce_op == reduce_util.ReduceOp.SUM:
    return Sum
  if reduce_op == reduce_util.ReduceOp.MEAN:
    return Average

  raise ValueError("Unsupported reduce op: {}".format(reduce_op))


def _is_current_device_ipu():
  current_device = tf_device.DeviceSpec.from_string(device_util.current())
  return current_device.device_type == "IPU"


class PopDistStrategy(distribute_lib.StrategyV1):
  """This is a distribution strategy for multi-replica distribution
  that uses compiled communications with GCL for reductions over IPU
  links and gateway links, while using Horovod for broadcasting of
  the initial values of variables to all processes, or when a
  reduction is requested with a CPU as the current device.

  This is the recommended distribution strategy when using PopDist and PopRun.
  The GCL reductions will then be performed across all the global replicas in
  the application.
  """
  def __init__(self,
               ipu_device="/device:IPU:0",
               add_ipu_cross_replica_reductions=True):
    popdist.init()

    # We create an empty cluster here since we will not be using gRPC for communication.
    # All the communication is delegated to either GCL or Horovod (MPI) below.
    cluster_resolver = cluster_resolver_lib.SimpleClusterResolver(
        server_lib.ClusterSpec({}))

    super().__init__(
        PopDistExtendedV1(self, cluster_resolver, ipu_device,
                          add_ipu_cross_replica_reductions))

  def update_ipu_config(self, config):
    """Update the given IPU configuration with the multi-replica
    distribution options.

    Args:
      config: The IPUConfig instance to update.

    Returns:
      The IPUConfig instance.
    """
    config.experimental.multi_replica_distribution.process_count = \
      popdist.getNumInstances()
    config.experimental.multi_replica_distribution.process_index = \
      popdist.getInstanceIndex()


def _ipu_device_for_host(ipu_device_string, host_device_string):
  ipu_device = tf_device.DeviceSpec.from_string(ipu_device_string)
  host_device = tf_device.DeviceSpec.from_string(host_device_string)

  # Take distributed info from the host and device info from the IPU.
  ipu_for_host = tf_device.DeviceSpec(job=host_device.job,
                                      replica=host_device.replica,
                                      task=host_device.task,
                                      device_type=ipu_device.device_type,
                                      device_index=ipu_device.device_index)

  return ipu_for_host.to_string()


class IPUSyncOnReadVariable(values.SyncOnReadVariable):  # pylint: disable=abstract-method
  pass


class IPUMirroredVariable(values.MirroredVariable):  # pylint: disable=abstract-method
  pass


class PopDistExtendedV1(
    collective_all_reduce_strategy.CollectiveAllReduceExtended):
  def __init__(self, container_strategy, cluster_resolver, ipu_device,
               add_ipu_cross_replica_reductions):

    super().__init__(
        container_strategy,
        communication=cross_device_ops_lib.CollectiveCommunication.RING,
        cluster_resolver=cluster_resolver)

    self._num_workers = popdist.getNumInstances()
    self._add_ipu_cross_replica_reductions = add_ipu_cross_replica_reductions

    host_devices = self._device_map.all_devices
    if len(host_devices) != 1:
      raise ValueError("Expected one host device per worker")

    self._host_device = host_devices[0]
    self._ipu_device = _ipu_device_for_host(ipu_device, self._host_device)
    self._variables_on_host = False

    self._variable_device = self._ipu_device

  def _get_variable_creator_initial_value(self, replica_id, device,
                                          primary_var, **kwargs):
    assert replica_id == 0
    assert device is not None
    assert primary_var is None

    def initial_value_fn():  # pylint: disable=g-missing-docstring
      # Override colocation and XLA attributes for initializers.
      colocation_list = attr_value_pb2.AttrValue.ListValue(s=[b'loc:@cpu'])
      attrs = {
          "_class": attr_value_pb2.AttrValue(list=colocation_list),
          "_XlaCompile": attr_value_pb2.AttrValue(b=False),
          "_XlaScope": attr_value_pb2.AttrValue(s=b''),
      }
      with ops.device(device), ops.get_default_graph()._attr_scope(attrs):  # pylint: disable=protected-access
        initial_value = kwargs["initial_value"]
        if callable(initial_value):
          initial_value = initial_value()
        assert not callable(initial_value)
        initial_value = ops.convert_to_tensor(initial_value,
                                              dtype=kwargs.get("dtype", None))
        return self._broadcast_implementation(initial_value, device)

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
      with ops.device(self._host_device), graph._attr_scope(disable_xla):  # pylint: disable=protected-access
        return [next_creator(*args, **kwargs)]

    # For tf2: use values.create_mirrored_variable
    return distribute_lib.create_mirrored_variable(
        self._container_strategy(), device_map, logical_device, _real_creator,
        IPUMirroredVariable, IPUSyncOnReadVariable, *args, **kwargs)

  def read_var(self, var):
    return var.read_value()

  def _reduce_to(self, reduce_op, value, destinations):
    del destinations

    if isinstance(value, values.DistributedValues):
      assert len(value.values) == 1
      value = value.values[0]

    if not _is_current_device_ipu():
      return hvd_allreduce(value, op=_to_horovod_op(reduce_op))

    # On the IPU, do reduction with GCL if requested.
    if not self._add_ipu_cross_replica_reductions:
      return value

    if reduce_op not in (reduce_util.ReduceOp.SUM, reduce_util.ReduceOp.MEAN):
      raise ValueError("Unsupported reduce op: {}".format(reduce_op))

    result = cross_replica_ops.cross_replica_sum(value)

    if reduce_op == reduce_util.ReduceOp.MEAN:
      result = gen_poputil_ops.ipu_replication_normalise(result)

    return result

  def _batch_reduce_to(self, reduce_op, value_destination_pairs):
    return [
        self.reduce_to(reduce_op, t, destinations=v)
        for t, v in value_destination_pairs
    ]

  def _call_for_each_replica(self, fn, args, kwargs):
    with distribute_lib.ReplicaContext(
        self._container_strategy(), replica_id_in_sync_group=0), \
        ops.device(self._ipu_device):
      return fn(*args, **kwargs)

  def _validate_colocate_with_variable(self, colocate_with_variable):
    if colocate_with_variable.device != self._variable_device:
      raise ValueError("Unexpected colocated variable device: {}".format(
          colocate_with_variable.device))

  def _reduce_implementation(self, reduce_op, value, destinations):
    # This is an extension point for overriding, try to keep a stable API.
    return super()._reduce_to(reduce_op, value, destinations)

  def _batch_reduce_implementation(self, reduce_op, value_destination_pairs):
    # This is an extension point for overriding, try to keep a stable API.
    return super()._batch_reduce_to(reduce_op, value_destination_pairs)

  def _broadcast_implementation(self, initial_value, device):
    if tf_device.DeviceSpec.from_string(device).device_type != "CPU":
      raise RuntimeError(
          "Can only broadcast on CPU, but got device {}".format(device))

    return hvd_broadcast(initial_value, root_rank=0)
