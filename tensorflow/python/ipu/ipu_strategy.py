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
"""
Distribution strategy for a single system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ipu.ops import cross_replica_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest


class IPUStrategy(distribute_lib.StrategyV1):
  """This is a distribution strategy for targeting a system with one
  or more IPUs.

  Creating variables and Keras models within the scope of the
  IPUStrategy will ensure that they are placed on the IPU.

  A tf.function can be executed on the IPU by calling it from the
  `experimental_run_v2` function.

  Variables will automatically be placed onto the IPUs, but the
  initializers for the variables will be performed on the CPU
  device.

  .. code-block:: python

    from tensorflow.python import ipu

    # Create an IPU distribution strategy
    strategy = ipu.ipu_strategy.IPUStrategy()

    with strategy.scope():
        
        # Instantiate a keras model here
        m = MyModel()

        # And train it
        m.fit(...)

        # Or call a tf.function
        res = strategy.experimental_run_v2(my_fn, [...])
  

  """
  def __init__(self, ipu_device="/device:IPU:0", cpu_device="/device:CPU:0"):
    """Create a new IPUStrategy.

    Args:
      ipu_device: The Tensorflow device representing the IPUs.
      cpu_device: The Tensorflow device for the CPU.
    """
    super().__init__(IPUExtended(self, ipu_device, cpu_device))

  def experimental_run_v2(self, fn, args=(), kwargs=None):
    _validate_experimental_run_v2_function(fn)
    return super().experimental_run_v2(fn, args, kwargs)


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


def _is_current_device_ipu():
  current_device = tf_device.DeviceSpec.from_string(device_util.current())
  return current_device.device_type == "IPU"


def _validate_experimental_run_v2_function(fn):
  if context.executing_eagerly() and not isinstance(
      fn, (def_function.Function, function.ConcreteFunction)):
    raise ValueError(
        "IPUStrategy.experimental_run_v2(fn, ...) does not support eager "
        "execution. Either convert `fn` into a tf.function or consider "
        "calling strategy.experimental_run_v2 inside a tf.function.")


_UNSUPPORTED_DTYPES = (dtypes.float64,)


def _validate_dtypes(tensors, name):
  for t in tensors:
    if t.dtype in _UNSUPPORTED_DTYPES:
      raise TypeError("Unsupported data type for {}: {}".format(
          name, t.dtype.name))


def _validate_function_for_arguments(fn, args, kwargs):
  if isinstance(fn, def_function.Function):
    concrete_fn = fn.get_concrete_function(*args, **kwargs)
    _validate_dtypes(concrete_fn.inputs, "input")
    _validate_dtypes(concrete_fn.outputs, "output")


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
        _validate_function_for_arguments(fn, args, kwargs)
        return fn(*args, **kwargs)
      finally:
        xla_context.Exit()

  def _reduce_to(self, reduce_op, value, destinations):
    del destinations

    if not _is_current_device_ipu():
      return value

    if reduce_op not in (reduce_util.ReduceOp.SUM, reduce_util.ReduceOp.MEAN):
      raise ValueError("Unsupported reduce op: {}".format(reduce_op))

    result = cross_replica_ops.cross_replica_sum(value)

    if reduce_op == reduce_util.ReduceOp.MEAN:
      result = gen_poputil_ops.ipu_replication_normalise(result)

    return result

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
