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
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.framework import device as tf_device
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu.horovod import Sum, Average, size, rank, allreduce as hvd_allreduce, broadcast as hvd_broadcast
from tensorflow.python.ipu.ipu_multi_worker_strategy import IPUMultiWorkerExtended
from tensorflow.python.ipu.ops import cross_replica_ops
from tensorflow.python.training import server_lib


def _to_horovod_op(reduce_op):
  if reduce_op == reduce_util.ReduceOp.SUM:
    return Sum
  if reduce_op == reduce_util.ReduceOp.MEAN:
    return Average

  raise ValueError("Unsupported reduce op: {}".format(reduce_op))


def _is_current_device_ipu():
  current_device = tf_device.DeviceSpec.from_string(device_util.current())
  return current_device.device_type == "IPU"


class IPUMultiReplicaStrategy(distribute_lib.StrategyV1):
  """This is a distribution strategy for multi-replica distribution
  that uses compiled communications with GCL for reductions over IPU
  links and gateway links, while using Horovod for broadcasting of
  the initial values of variables to all processes, or when a
  reduction is requested with a CPU as the current device.
  """
  def __init__(self, ipu_device="/device:IPU:0"):
    # We create an empty cluster here since we will not be using gRPC for communication.
    # All the communication is delegated to either GCL or Horovod (MPI) below.
    cluster_resolver = cluster_resolver_lib.SimpleClusterResolver(
        server_lib.ClusterSpec({}))

    super().__init__(
        IPUMultiReplicaExtended(self, cluster_resolver, ipu_device))

  def update_ipu_config(self, config):
    """Update the given IPU configuration with the multi-replica
    distribution options.

    Args:
      config: The IpuOptions configuration protobuf to update.

    Returns:
      The IpuOptions configuration protobuf.
    """
    return ipu_utils.set_experimental_multi_replica_distribution_options(
        config, process_count=size(), process_index=rank())


class IPUMultiReplicaExtended(IPUMultiWorkerExtended):
  def __init__(self, container_strategy, cluster_resolver, ipu_device):
    super().__init__(container_strategy,
                     cluster_resolver,
                     ipu_device,
                     variables_on_host=False)
    self._num_workers = size()

  def _reduce_to(self, reduce_op, value, destinations):
    del destinations

    if not _is_current_device_ipu():
      # If not on IPU, use Horovod for the reduction.
      return hvd_allreduce(value, op=_to_horovod_op(reduce_op))

    # On IPU we do a compiled reduction with GCL.
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
