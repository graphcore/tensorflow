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
# =============================================================================
"""
Optimizer wrappers which perform local gradient accumulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.driver import threestate_pb2
from tensorflow.compiler.plugin.poplar.ops import gen_functional_ops
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ipu import functional_ops
from tensorflow.python.ipu.ops import op_util
from tensorflow.python.ipu.optimizers import IpuOptimizer
from tensorflow.python.ipu.optimizers import cross_replica_optimizer


class GradientAccumulationOptimizerV2(IpuOptimizer):  # pylint: disable=abstract-method
  """An optimizer where instead of performing the weight update for every batch,
  gradients across multiple batches are accumulated. After multiple batches
  have been processed, their accumulated gradients are used to compute the
  weight update.

  This feature of neural networks allows us to simulate bigger batch sizes. For
  example if we have a model of batch size 16 and we accumulate the gradients
  of 4 batches, this simulates an input batch of size 64.

  Unlike 'GradientAccumulationOptimizer', this optimizer can be used to wrap any
  other TensorFlow optimizer.

  See the :ref:`gradient-accumulation` section in the documention for more
  details.
  """
  def __init__(self,
               opt,
               num_mini_batches,
               offload_weight_update_variables=None,
               replicated_optimizer_state_sharding=False,
               dtype=None,
               name="GradientAccumulationOptimizerV2"):
    """Construct a Gradient Accumulation Optimizer V2.

    Args:
      opt: An existing `Optimizer` to encapsulate.
      num_mini_batches: Number of mini-batches the gradients will be accumulated
        for.
      offload_weight_update_variables: When enabled, any `tf.Variable` which is
        only used by the weight update of the pipeline (for example the
        accumulator variable when using the `tf.MomentumOptimizer`), will be
        stored in the remote memory. During the weight update this variable will
        be streamed onto the device and then streamed back to the remote memory
        after it has been updated. Requires the machine to be configured with
        support for `Poplar remote buffers`. Offloading variables into remote
        memory can reduce maximum memory liveness, but can also increase the
        computation time of the weight update.
        When set to `None` the variables will be placed in either in-processor
        or remote memory automatically based on the current best placement
        strategy.
      replicated_optimizer_state_sharding: If True, any any `tf.Variable` which
        is offloaded will be partitioned across the replicas. A collective
        all-gather will be inserted to restore the tensor on each replica.
        If `None`, this value will match the value of
        `offload_weight_update_variables`.
      dtype: The data type used for the gradient accumulation buffer.
        One of:
          - `None`: Use an accumulator of the same type as the variable type.
          - A `DType`: Use this type for all the accumulators.
          - A callable that takes the variable and returns a `DType`: Allows
            specifying the accumulator type on a per-variable basis.

        The gradients passed to `Optimizer.apply_gradients` will have the dtype
        requested here. If that dtype is different from the variable dtype
        a cast is needed at some point to make them compatible. If you want
        to cast the gradients immediately, you can wrap your optimizer in the
        `MapGradientOptimizer` with a `tf.cast`.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientAccumulationOptimizerV2".
    """
    super().__init__(opt, name=name)
    self._opt = opt

    if num_mini_batches < 1:
      raise ValueError("num_mini_batches must be a positive number.")

    self._num_mini_batches = num_mini_batches

    def bool_to_three_state(value, default):
      if value is None:
        return default
      elif value:
        return threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_ON)
      return threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_OFF)

    self._offload_weight_update_variables = bool_to_three_state(
        offload_weight_update_variables,
        threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_UNDEFINED))
    self._replicated_optimizer_state_sharding = bool_to_three_state(
        replicated_optimizer_state_sharding,
        self._offload_weight_update_variables)

    self._dtype = dtype

  @staticmethod
  def create_accumulated_grads(grad, var, dtype, num_mini_batches):
    if grad is None:
      return (grad, var)
    with ops.colocate_with(grad):
      # Find the data type for the accumulator.
      dtype = op_util.get_accumulator_dtype(var, dtype)
      # Create an accumulator - variable is used as reference for shape/layout.
      accumulator = gen_poputil_ops.gradient_accumulator_create(
          var, output_type=dtype)
      # Add the gradients to the accumulator.
      accumulator = gen_poputil_ops.gradient_accumulator_add(accumulator, grad)
      # Sink the accumulators.
      grad = gen_poputil_ops.gradient_accumulator_sink(accumulator)
      return (grad, var)

  @staticmethod
  def apply_gradient_accumulation(resource_update_, name, apply_grad_ops,
                                  offload_weight_update_variables,
                                  replicated_optimizer_state_sharding,
                                  num_mini_batches):
    with ops.name_scope(name + "/WU") as scope:
      func_graph, captured_args, constant_outputs = \
        functional_ops._compile_function(  # pylint: disable=protected-access
          resource_update_, [], scope, apply_grad_ops, True)

    # Create the resource update and lower the function into XLA.
    with ops.control_dependencies(list(func_graph.control_captures)):
      outputs = gen_functional_ops.resource_update(
          captured_args,
          to_apply=util.create_new_tf_function(func_graph),
          Tout=func_graph.output_types,
          output_shapes=func_graph.output_shapes,
          offload_weight_update_variables=offload_weight_update_variables,
          replicated_optimizer_state_sharding=
          replicated_optimizer_state_sharding,
          num_batches_to_accumulate=num_mini_batches)
      outputs = functional_ops._replace_outputs(outputs, constant_outputs)  # pylint: disable=protected-access

    return outputs

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      An `Operation` that applies the gradients. If `global_step` was not None,
      that operation also increments `global_step`.

    Raises:
      ValueError: If the grads_and_vars is malformed.
    """

    accumulated_grads_and_vars = list(
        map(
            lambda x: self.create_accumulated_grads(x[0], x[
                1], self._dtype, self._num_mini_batches), grads_and_vars))

    # Create an explicit function call for the apply gradients - note that we
    # allow external captures here.
    apply_grad_ops = []

    def resource_update_():
      apply_grads = self._opt.apply_gradients(accumulated_grads_and_vars,
                                              global_step, name)
      apply_grad_ops.append(apply_grads)

    return self.apply_gradient_accumulation(
        resource_update_, self._opt.get_name(), apply_grad_ops,
        self._offload_weight_update_variables,
        self._replicated_optimizer_state_sharding, self._num_mini_batches)


class CrossReplicaGradientAccumulationOptimizerV2(IpuOptimizer):  # pylint: disable=abstract-method
  """An optimizer where instead of performing the weight update for every batch,
  gradients across multiple batches are accumulated. After multiple batches
  have been processed, their accumulated gradients are then reduced accross the
  replicas before being used to compute the weight update.

  This feature of neural networks allows us to simulate bigger batch sizes. For
  example if we have a model of batch size 16 and we accumulate the gradients
  of 4 batches, this simulates an input batch of size 64.

  This optimizer is similar to GradientAccumulationOptimizerV2, however using
  this optimizer guarantees that the accumulated gradients will only be
  exchanged between IPUs when the gradients are applied to the weights, and
  hence reduces the number of cross-IPU gradient exchanges by a factor of
  'num_mini_batches'.
  """
  def __init__(self,
               opt,
               num_mini_batches,
               offload_weight_update_variables=None,
               replicated_optimizer_state_sharding=False,
               dtype=None,
               name="CrossReplicaGradientAccumulationOptimizerV2"):
    """Construct a Cross Replica Gradient Accumulation Optimizer V2.

    Args:
      opt: An existing `Optimizer` to encapsulate.
      num_mini_batches: Number of mini-batches the gradients will be accumulated
        for.
      offload_weight_update_variables: If True, any `tf.Variable` which is
        only used by the weight update of the model (for example the accumulator
        variable when using the `tf.MomentumOptimizer`), will be stored in the
        remote memory. During the weight update this variable will be streamed
        onto the device and then streamed back to the remote memory after it has
        been updated. Requires the machine to be configured with support for
        `Poplar remote buffers`. Offloading variables into remote memory can
        reduce maximum memory liveness, but can also increase the computation
        time of the weight update.
      replicated_optimizer_state_sharding: If True, any any `tf.Variable` which
        is offloaded will be partitioned across the replicas. A collective
        all-gather will be inserted to restore the tensor on each replica.
        If `None`, this value will match the value of
        `offload_weight_update_variables`.
      dtype: The data type used for the gradient accumulation buffer.
        One of:
          - `None`: Use an accumulator of the same type as the variable type.
          - A `DType`: Use this type for all the accumulators.
          - A callable that takes the variable and returns a `DType`: Allows
            specifying the accumulator type on a per-variable basis.

        The gradients passed to `Optimizer.apply_gradients` will have the dtype
        requested here. If that dtype is different from the variable dtype
        a cast is needed at some point to make them compatible. If you want
        to cast the gradients immediately, you can wrap your optimizer in the
        `MapGradientOptimizer` with a `tf.cast`.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "CrossReplicaGradientAccumulationOptimizerV2".
    """
    if num_mini_batches < 1:
      raise ValueError("num_mini_batches must be a positive number.")

    # Internally we just wrap the optimizer in a GradientAccumulationOptimizer and CrossReplicaOptimizer.
    opt = GradientAccumulationOptimizerV2(
        cross_replica_optimizer.CrossReplicaOptimizer(opt), num_mini_batches,
        offload_weight_update_variables, replicated_optimizer_state_sharding,
        dtype, name)

    super().__init__(opt, name)


class GradientAccumulationOptimizer(IpuOptimizer):
  """An optimizer where instead of performing the weight update for every batch,
  gradients across multiple batches are accumulated. After multiple batches
  have been processed, their accumulated gradients are used to compute the
  weight update.

  This feature of neural networks allows us to simulate bigger batch sizes. For
  example if we have a model of batch size 16 and we accumulate the gradients
  of 4 batches, this simulates an input batch of size 64.

  This optimizer supports `tf.train.GradientDescentOptimizer` and
  `tf.train.MomentumOptimizer` only. All other optimizers should use
  `GradientAccumulationOptimizerV2`.
  """
  def __init__(self,
               opt,
               num_mini_batches,
               verify_usage=True,
               name="GradientAccumulationOptimizer"):
    """Construct a Gradient Accumulation Optimizer.

    Args:
      opt: An existing `Optimizer` to encapsulate.
      num_mini_batches: Number of mini-batches the gradients will be accumulated
        for.
      verify_usage: The current gradient accumulation supports the
        `GradientDescentOptimizer` and `MomentumOptimizer` optimizers.
        Any other usages of this optimizer might results in incorrect results.
        This option can be used to disable this check.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientAccumulationOptimizer".
    """

    super(GradientAccumulationOptimizer, self).__init__(opt, name)

    if num_mini_batches < 1:
      raise ValueError("num_mini_batches must be a positive number.")

    self._num_mini_batches = num_mini_batches
    self._verify_usage = verify_usage

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      An `Operation` that applies the gradients. If `global_step` was not None,
      that operation also increments `global_step`.

    Raises:
      ValueError: If the grads_and_vars is malformed.
    """
    summed_grads_and_vars = []
    for (grad, var) in grads_and_vars:
      if grad is None:
        summed_grads_and_vars.append((grad, var))
      else:
        with ops.colocate_with(grad):
          summed_grads_and_vars.append(
              (gen_poputil_ops.ipu_stateful_gradient_accumulate(
                  grad,
                  num_mini_batches=self._num_mini_batches,
                  verify_usage=self._verify_usage), var))
    return self._opt.apply_gradients(summed_grads_and_vars, global_step, name)


class CrossReplicaGradientAccumulationOptimizer(IpuOptimizer):
  """An optimizer where instead of performing the weight update for every batch,
  gradients across multiple batches are accumulated. After multiple batches
  have been processed, their accumulated gradients are then reduced accross the
  replicas before being used to compute the weight update.

  This feature of neural networks allows us to simulate bigger batch sizes. For
  example if we have a model of batch size 16 and we accumulate the gradients
  of 4 batches, this simulates an input batch of size 64.

  This optimizer is similar to GradientAccumulationOptimizer, however using this
  optimizer guarantees that the accumulated gradients will only be exchanged
  between IPUs when the accumulated gradients are back-propagated through the
  network.
  """
  def __init__(self,
               opt,
               num_mini_batches,
               verify_usage=True,
               name="CrossReplicaGradientAccumulationOptimizer"):
    """Construct a Cross Replica Gradient Accumulation Optimizer.

    Args:
      opt: An existing `Optimizer` to encapsulate.
      num_mini_batches: Number of mini-batches the gradients will be accumulated
        for.
      verify_usage: The current gradient accumulation supports the
        `GradientDescentOptimizer` and `MomentumOptimizer` optimizers.
        Any other usages of this optimizer might results in incorrect results.
        This option can be used to disable this check.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "CrossReplicaGradientAccumulationOptimizer".
    """
    if num_mini_batches < 1:
      raise ValueError("num_mini_batches must be a positive number.")

    # Internally we just wrap the optimizer in a GradientAccumulationOptimizer and CrossReplicaOptimizer.
    opt = cross_replica_optimizer.CrossReplicaOptimizer(
        GradientAccumulationOptimizer(opt, num_mini_batches, verify_usage))
    super(CrossReplicaGradientAccumulationOptimizer, self).__init__(opt, name)
