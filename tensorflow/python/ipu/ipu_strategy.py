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

from collections import OrderedDict
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import sequential
from tensorflow.python.ipu.ops import cross_replica_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest
from tensorflow.python.ipu.keras.extensions import functional_extensions
from tensorflow.python.ipu.keras.extensions import sequential_extensions
from tensorflow.python.ipu import ipu_infeed_queue


class IPUStrategyV1(distribute_lib.StrategyV1):
  """This is a distribution strategy for targeting a system with one
  or more IPUs.

  Creating variables and Keras models within the scope of the
  IPUStrategyV1 will ensure that they are placed on the IPU.

  A tf.function can be executed on the IPU by calling it from the `run`
  function.

  Variables will automatically be placed onto the IPUs, but the
  initializers for the variables will be performed on the CPU
  device.

  .. code-block:: python

    from tensorflow.python import ipu

    # Create an IPU distribution strategy
    strategy = ipu.ipu_strategy.IPUStrategyV1()

    with strategy.scope():
        
        # Instantiate a keras model here
        m = MyModel()

        # And train it
        m.fit(...)

        # Or call a tf.function
        res = strategy.run(my_fn, [...])


  """

  _enable_legacy_iterators = True

  def __init__(self,
               ipu_device="/device:IPU:0",
               cpu_device="/device:CPU:0",
               enable_dataset_iterators=True,
               enable_keras_extensions=True):
    """Create a new IPUStrategyV1.

    Args:
      ipu_device: The TensorFlow device representing the IPUs.
      cpu_device: The TensorFlow device for the CPU.
      enable_dataset_iterators: Whether to create IPUStrategy specific dataset
        iterators inside of this strategy scope or whether to use standard
        dataset iterators.
      enable_keras_extensions: Whether to enable IPU specific Keras extensions
        to improve Keras performance when using IPUs.
    """
    super().__init__(IPUExtendedV1(self, ipu_device, cpu_device))
    self._enable_iterators = enable_dataset_iterators
    self._enable_keras_extensions = enable_keras_extensions
    self._keras_extensions = OrderedDict()
    # Insert Sequential before Functional as Sequential models inherit from
    # Functional models.
    self._register_keras_extension(sequential.Sequential,
                                   sequential_extensions.SequentialExtension)
    self._register_keras_extension(functional.Functional,
                                   functional_extensions.FunctionalExtension)

  def run(self, fn, args=(), kwargs=None, options=None):
    _validate_run_function(fn)
    return super().run(fn, args, kwargs, options)

  def experimental_run_v2(self, *args, **kwargs):
    raise RuntimeError(
      "`experimental_run_v2` has been deleted. Use `run` instead.")

  @property
  def _device_ordinal(self):
    device_string = self.extended.non_slot_devices(None)
    current_device = tf_device.DeviceSpec.from_string(device_string)
    return current_device.device_index

  def _enable_dataset_iterators(self):
    return context.executing_eagerly() and self._enable_iterators

  def _create_dataset_iterator(self, dataset):
    assert self._enable_dataset_iterators()
    return ipu_infeed_queue.IPUOwnedIterator(dataset=dataset)  # pylint: disable=protected-access

  def _register_keras_extension(self, class_type, extension):
    self._keras_extensions[class_type] = extension

  def _delete_keras_extension(self, class_type):
    self._keras_extensions.pop(class_type, None)

  def _patch_keras_extension(self, instance):
    if not self._enable_keras_extensions:
      return

    for class_type, extension in self._keras_extensions.items():
      if isinstance(instance, class_type):
        if isinstance(instance, base_layer.KerasExtension):
          if not isinstance(instance, extension):
            raise RuntimeError(
                "KerasExtension patching failed - already patched with a "
                "different extension.")
          break

        # Patch in the extension.
        # Note that we keep the name as Keras sometimes does __name__ checks.
        cls = instance.__class__
        instance.__class__ = cls.__class__(cls.__name__, (cls, extension), {})
        extension.__init__(instance)
        break


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


def _validate_run_function(fn):
  if context.executing_eagerly() and not isinstance(
      fn, (def_function.Function, function.ConcreteFunction)):
    raise ValueError(
        "IPUStrategyV1.run(fn, ...) does not support eager "
        "execution. Either convert `fn` into a tf.function or consider "
        "calling strategy.run inside a tf.function.")


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


class IPUExtendedV1(distribute_lib.StrategyExtendedV1):  # pylint: disable=abstract-method
  # Not all abstract methods are implemented; implement as needed.
  # See _DefaultDistributionExtended for dummy implementations.

  def __init__(self, container_strategy, ipu_device, cpu_device):
    super().__init__(container_strategy)
    self._ipu_device = ipu_device
    self._cpu_device = cpu_device
    worker_device_pairs = [("", [self._cpu_device])]
    self._input_workers = input_lib.InputWorkers(worker_device_pairs)

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
        outputs = fn(*args, **kwargs)
      finally:
        xla_context.Exit()

      # Insert a sync at the end of the execution to make sure the execution
      # has finished even if there are no outputs.
      with context.eager_mode():
        gen_poputil_ops.device_sync()

      return outputs

  def _reduce_to(self, reduce_op, value, destinations, options):
    del destinations
    del options

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

  def _make_input_fn_iterator(
      self,
      input_fn,
      replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
    del replication_mode
    return input_lib.InputFunctionIterator(input_fn, self._input_workers,
                                           [distribute_lib.InputContext()],
                                           self._container_strategy())

  def _experimental_distribute_dataset(self, dataset, options):
    del options
    return input_lib.get_distributed_dataset(
        dataset,
        self._input_workers,
        self._container_strategy(),
        num_replicas_in_sync=self._num_replicas_in_sync)

  def non_slot_devices(self, var_list):
    del var_list
    return self._ipu_device

  def _get_local_replica_id(self, replica_id_in_sync_group):
    return replica_id_in_sync_group

  @property
  def worker_devices(self):
    return (self._ipu_device,)

  @property
  def parameter_devices(self):
    return (self._ipu_device,)

  @property
  def _support_per_replica_values(self):
    return False


# Export the alias for backwards compability.
IPUStrategy = IPUStrategyV1
