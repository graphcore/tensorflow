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
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.framework import device as device_lib
from tensorflow.python.distribute.cluster_resolver import \
    cluster_resolver as cluster_resolver_lib
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ipu import keras_extensions
from tensorflow.python.ipu.ops import cross_replica_ops
from tensorflow.python.ipu.distributed import host_collective_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import server_lib
from tensorflow.python.training.tracking import base as trackable


def _is_current_device_ipu():
  current_device = tf_device.DeviceSpec.from_string(device_util.current())
  return current_device.device_type == "IPU"


class PopDistStrategy(distribute_lib.StrategyV1,
                      keras_extensions.KerasExtensions):
  """This is a distribution strategy for multi-replica distribution
  that uses compiled communications with GCL for reductions over
  IPU-Links and GW-Links, across all the global replicas in the
  application. This is the recommended distribution strategy when using
  :ref:`PopDist and PopRun <poprun-user-guide:index>`.

  PopDist is used for host communication, for example when broadcasting
  initial values of variables to all processes. Another example is when
  a reduction is requested with a CPU as the current device.
  """
  _collective_key_base = 0

  def __init__(self,
               ipu_device="/device:IPU:0",
               add_ipu_cross_replica_reductions=True,
               enable_dataset_iterators=True,
               enable_keras_extensions=True):
    popdist.init()

    # We create an empty cluster here since we will not be using gRPC for communication.
    # All the communication is delegated to either GCL or Horovod (MPI) below.
    cluster_resolver = cluster_resolver_lib.SimpleClusterResolver(
        server_lib.ClusterSpec({}))

    super().__init__(
        PopDistExtendedV1(self, cluster_resolver, ipu_device,
                          add_ipu_cross_replica_reductions))
    keras_extensions.KerasExtensions.__init__(self, enable_dataset_iterators,
                                              enable_keras_extensions)

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

  @property
  def supports_loss_scaling(self):
    return True


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


class IPUDistributedVariable(values.DistributedVariable):  # pylint: disable=abstract-method
  pass


class IPUSyncOnReadVariable(values.SyncOnReadVariable):  # pylint: disable=abstract-method
  pass


class IPUMirroredVariable(values.MirroredVariable):  # pylint: disable=abstract-method
  pass


IPU_VARIABLE_CLASS_MAPPING = {
    "VariableClass": IPUDistributedVariable,
    variable_scope.VariableSynchronization.ON_WRITE: IPUMirroredVariable,
    variable_scope.VariableSynchronization.ON_READ: IPUSyncOnReadVariable,
}


class IPUOnWritePolicy(values.OnWritePolicy):  # pylint: disable=abstract-method
  pass


class IPUOnReadPolicy(values.OnReadPolicy):  # pylint: disable=abstract-method
  pass


IPU_VARIABLE_POLICY_MAPPING = {
    variable_scope.VariableSynchronization.ON_WRITE: IPUOnWritePolicy,
    variable_scope.VariableSynchronization.ON_READ: IPUOnReadPolicy,
}


class PopDistExtendedV1(
    collective_all_reduce_strategy.CollectiveAllReduceExtended):
  def __init__(self, container_strategy, cluster_resolver, ipu_device,
               add_ipu_cross_replica_reductions):
    communication_options = collective_util.Options(
        implementation=cross_device_ops_lib.CollectiveCommunication.RING)

    super().__init__(container_strategy,
                     cluster_resolver=cluster_resolver,
                     communication_options=communication_options)

    self._num_workers = popdist.getNumInstances()
    self._add_ipu_cross_replica_reductions = add_ipu_cross_replica_reductions

    host_devices = self._devices
    if len(host_devices) != 1:
      raise ValueError("Expected one host device per instance?")

    self._host_device = host_devices[0]
    self._ipu_device = _ipu_device_for_host(ipu_device, self._host_device)
    self._variables_on_host = False

    self._variable_device = self._ipu_device

    # By default the functional graphs are not retraced and therefore device
    # information is not lowered to ops which means distribution strategies do
    # not work.
    self._retrace_functions_for_each_device = True

  @property
  def _num_replicas_in_sync(self):
    return popdist.getNumTotalReplicas()

  def non_slot_devices(self, var_list):
    del var_list
    return self._ipu_device

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
      with ops.device(device), \
          ops.get_default_graph()._attr_scope(attrs):  # pylint: disable=protected-access
        initial_value = kwargs["initial_value"]
        if callable(initial_value):
          initial_value = initial_value()
        if isinstance(initial_value, trackable.CheckpointInitialValue):
          initial_value = initial_value.wrapped_value
        assert not callable(initial_value)
        initial_value = ops.convert_to_tensor(initial_value,
                                              dtype=kwargs.get("dtype", None))
        return self._broadcast_implementation(initial_value, device)

    return initial_value_fn

  def _create_variable(self, next_creator, **kwargs):
    colocate_with = kwargs.pop("colocate_with", None)
    if colocate_with is None:
      devices = [self._variable_device]
    elif isinstance(colocate_with, numpy_dataset.SingleDevice):
      with ops.device(colocate_with.device):
        return next_creator(**kwargs)
    else:
      devices = colocate_with._devices  # pylint: disable=protected-access

    def _real_creator(**kwargs):
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
          return [next_creator(**kwargs)]

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
        return [next_creator(**kwargs)]

    return distribute_utils.create_mirrored_variable(
        self._container_strategy(), _real_creator, IPU_VARIABLE_CLASS_MAPPING,
        IPU_VARIABLE_POLICY_MAPPING, **kwargs)

  def read_var(self, var):
    return var.read_value()

  def _reduce_to(self, reduce_op, value, destinations, options):
    del destinations
    del options

    if isinstance(value, values.DistributedValues):
      assert len(value.values) == 1
      value = value.values[0]

    if not _is_current_device_ipu():
      return host_collective_ops.all_reduce(value, reduce_op)

    # On the IPU, do reduction with GCL if requested.
    if not self._add_ipu_cross_replica_reductions:
      return value

    if reduce_op not in (reduce_util.ReduceOp.SUM, reduce_util.ReduceOp.MEAN):
      raise ValueError("Unsupported reduce op: {}".format(reduce_op))

    if reduce_op == reduce_util.ReduceOp.MEAN:
      result = cross_replica_ops.cross_replica_mean(value)
    else:
      result = cross_replica_ops.cross_replica_sum(value)

    return result

  def _batch_reduce_to(self, reduce_op, value_destination_pairs, options):
    return [
        self.reduce_to(reduce_op, t, destinations=v, options=options)
        for t, v in value_destination_pairs
    ]

  def _call_for_each_replica(self, fn, args, kwargs):
    with distribute_lib.ReplicaContext(self._container_strategy(),
                                       replica_id_in_sync_group=0), ops.device(
                                           self._ipu_device):
      # Make sure it is compiled as a single engine when called in graph mode.
      # This is similar to the mechanism used by xla.compile.
      xla_context = control_flow_ops.XLAControlFlowContext()
      try:
        xla_context.Enter()
        outputs = fn(*args, **kwargs)
      finally:
        xla_context.Exit()

      return outputs

  def _validate_colocate_with_variable(self, colocate_with_variable):
    if colocate_with_variable.device != self._variable_device:
      raise ValueError("Unexpected colocated variable device: {}".format(
          colocate_with_variable.device))

  def _reduce_implementation(self, reduce_op, value, destinations, options):
    # This is an extension point for overriding, try to keep a stable API.
    return super()._reduce_to(reduce_op, value, destinations, options)

  def _batch_reduce_implementation(self, reduce_op, value_destination_pairs,
                                   options):
    # This is an extension point for overriding, try to keep a stable API.
    return super()._batch_reduce_to(reduce_op, value_destination_pairs,
                                    options)

  def _broadcast_implementation(self, initial_value, device):
    if tf_device.DeviceSpec.from_string(device).device_type != "CPU":
      raise RuntimeError(
          "Can only broadcast on CPU, but got device {}".format(device))

    return host_collective_ops.broadcast(initial_value)
