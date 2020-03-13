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
# ===================================================================
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import values
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest


class IPUStrategy(distribute_lib.StrategyV1):
  def __init__(self, ipu_device="/device:IPU:0", cpu_device="/device:CPU:0"):
    super().__init__(IPUExtended(self, ipu_device, cpu_device))


def _get_variable_creator_initial_value(device, **kwargs):
  def initial_value_fn():
    with ops.device(device):
      initial_value = kwargs["initial_value"]
      if callable(initial_value):
        initial_value = initial_value()
      assert not callable(initial_value)
      return ops.convert_to_tensor(initial_value,
                                   dtype=kwargs.get("dtype", None))

  return initial_value_fn


class IPUExtended(distribute_lib.StrategyExtendedV1):  # pylint: disable=abstract-method
  # Not all abstract methods are implemented; implement as needed.
  # See _DefaultDistributionExtended for dummy implementations.

  def __init__(self, container_strategy, ipu_device, cpu_device):
    super().__init__(container_strategy)
    self._ipu_device = ipu_device
    self._cpu_device = cpu_device

    device_map = values.ReplicaDeviceMap([self._cpu_device])

    worker_device_pairs = [("", [self._cpu_device])]
    self._input_workers = input_lib.InputWorkers(device_map,
                                                 worker_device_pairs)

  def _create_variable(self, next_creator, *args, **kwargs):
    # Place initializer on the CPU.
    kwargs["initial_value"] = _get_variable_creator_initial_value(
        device=self._cpu_device, **kwargs)

    # Place variable on the IPU.
    with ops.device(self._ipu_device):
      return next_creator(*args, **kwargs)

  def _local_results(self, distributed_value):
    return (distributed_value,)

  def value_container(self, value):
    return value

  def _call_for_each_replica(self, fn, args, kwargs):
    with distribute_lib.ReplicaContext(self._container_strategy(),
                                       replica_id_in_sync_group=0), \
        ops.device(self._ipu_device):
      # Make sure it is compiled as a single engine when called in graph mode.
      # This is similar to the mechanism used by xla.compile.
      xla_context = control_flow_ops.XLAControlFlowContext()
      try:
        xla_context.Enter()
        return fn(*args, **kwargs)
      finally:
        xla_context.Exit()

  def _reduce_to(self, reduce_op, value, destinations):
    del reduce_op, destinations
    return value

  def _update(self, var, fn, args, kwargs, group):
    return self._update_non_slot(var, fn, (var,) + tuple(args), kwargs, group)

  def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
    with distribute_lib.UpdateContext(colocate_with):
      result = fn(*args, **kwargs)
      if group:
        return result
      return nest.map_structure(self._local_results, result)

  @property
  def _num_replicas_in_sync(self):
    return 1

  def _in_multi_worker_mode(self):
    return False

  def _global_batch_size(self):
    return True

  def _experimental_distribute_dataset(self, dataset):
    return input_lib.get_distributed_dataset(dataset, self._input_workers,
                                             self._container_strategy())

  def non_slot_devices(self, var_list):
    return self._ipu_device
