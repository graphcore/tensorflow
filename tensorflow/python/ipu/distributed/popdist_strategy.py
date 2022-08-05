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

from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.framework import device as tf_device
from tensorflow.python.ipu.ipu_multi_worker_strategy import \
    IPUMultiWorkerExtended
from tensorflow.python.ipu.ops import cross_replica_ops
from tensorflow.python.training import server_lib
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.ipu.horovod import Sum, Average, \
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


class PopDistExtendedV1(IPUMultiWorkerExtended):
  def __init__(self, container_strategy, cluster_resolver, ipu_device,
               add_ipu_cross_replica_reductions):
    super().__init__(container_strategy,
                     cluster_resolver,
                     ipu_device,
                     variables_on_host=False)
    self._num_workers = popdist.getNumInstances()
    self._add_ipu_cross_replica_reductions = add_ipu_cross_replica_reductions

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

  def _broadcast_implementation(self, initial_value, device):
    if tf_device.DeviceSpec.from_string(device).device_type != "CPU":
      raise RuntimeError(
          "Can only broadcast on CPU, but got device {}".format(device))

    return hvd_broadcast(initial_value, root_rank=0)
