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
IPUMultiWorkerStrategy
~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import tape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest


class IPUMultiWorkerStrategy(distribute_lib.StrategyV1):
  """This is a distribution strategy for synchronous training using
  IPUs on multiple workers with between-graph replication.

  It places variables on the host device of each worker, and does
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
  variables used by the optimizer (e.g. the accumulator in momentum)
  are kept only in host memory and never used on the device, saving
  device memory.

  It does not yet work with the `IPUEstimator` which tries to compile
  one big XLA cluster for everything, which conflicts with how this
  strategy will split out the `optimizer.apply_gradients()` part and
  place it on the host.
  """
  def __init__(self, cluster_resolver):
    super(IPUMultiWorkerStrategy,
          self).__init__(IPUMultiWorkerExtended(self, cluster_resolver))


def current_device():
  return constant_op.constant(1.).device


class IPUMultiWorkerExtended(
    collective_all_reduce_strategy.CollectiveAllReduceExtended):
  def __init__(self, container_strategy, cluster_resolver):
    super(IPUMultiWorkerExtended, self).__init__(
        container_strategy,
        communication=cross_device_ops_lib.CollectiveCommunication.RING,
        cluster_resolver=cluster_resolver)

    devices = self._device_map.all_devices
    if len(devices) != 1:
      raise ValueError("Expected one device per worker for variables")
    self._variable_device = devices[0]

  def _create_variable(self, next_creator, *args, **kwargs):
    # Cache a snapshot of the variable on the current device,
    # otherwise the XLA cluster containing the ops consuming the
    # variable might be moved to the host to be colocated with it.
    kwargs["caching_device"] = current_device()

    # In case we are inside an ipu_jit_scope, we need to override it
    # to disable XLA for variable initialization on the host.
    disable_xla = {"_XlaCompile": attr_value_pb2.AttrValue(b=False)}

    graph = ops.get_default_graph()
    with ops.device(self._variable_device), \
        graph._attr_scope(disable_xla):  # pylint: disable=protected-access

      # The chief worker will initialize and broadcast the value to
      # the other workers.
      kwargs["initial_value"] = self._get_variable_creator_initial_value(
          replica_id=0,  # First replica on each worker.
          device=self._variable_device,
          primary_var=None,
          **kwargs)

      # Don't record operations (e.g. other variable reads) during
      # variable creation.
      with tape.stop_recording():
        return next_creator(*args, **kwargs)

  def _update(self, var, fn, args, kwargs, group):
    # The implementations of _update() and _update_non_slot() are identical
    # except _update() passes `var` as the first argument to `fn()`.
    return self._update_non_slot(var, fn, (var, ) + tuple(args), kwargs, group)

  def _update_non_slot(self, colocate_with, fn, args, kwargs, should_group):
    with ops.device(self._variable_device), \
        distribute_lib.UpdateContext(colocate_with):
      result = fn(*args, **kwargs)
      if should_group:
        return result
      return nest.map_structure(self._local_results, result)

  def read_var(self, var):
    return var.read_value()

  def _reduce_to(self, reduce_op, value, destinations):
    # Make sure the reduction is done on the variable device
    # by wrapping the inputs in an identity op on that device.
    with ops.device(self._variable_device):
      value = array_ops.identity(value)
    return super(IPUMultiWorkerExtended,
                 self)._reduce_to(reduce_op, value, destinations)

  def _batch_reduce_to(self, reduce_op, value_destination_pairs):
    # Make sure the reduction is done on the variable device
    # by wrapping the inputs in an identity op on that device.
    with ops.device(self._variable_device):
      value_destination_pairs = [(array_ops.identity(v), d)
                                 for (v, d) in value_destination_pairs]
    return super(IPUMultiWorkerExtended,
                 self)._batch_reduce_to(reduce_op, value_destination_pairs)

  def _call_for_each_replica(self, fn, args, kwargs):
    with distribute_lib.ReplicaContext(
        self._container_strategy(), replica_id_in_sync_group=self._task_id), \
        ops.device(self._variable_device):
      return fn(*args, **kwargs)

  def _validate_colocate_with_variable(self, colocate_with_variable):
    if colocate_with_variable.device != self._variable_device:
      raise ValueError("Unexpected colocated variable device: {}".format(
          colocate_with_variable.device))
