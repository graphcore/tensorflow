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

import numpy as np

from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.framework import ops
from tensorflow.python.ipu.ops import op_util
from tensorflow.python.ipu.optimizers import IpuOptimizer
from tensorflow.python.ipu.optimizers import cross_replica_optimizer
from tensorflow.python.ipu import internal_ops
from tensorflow.python.ipu.gradient_accumulation import GradientAccumulationReductionMethod
from tensorflow.python.ops import math_ops


def _compute_scales(reduction_method, num_mini_batches):
  if reduction_method == GradientAccumulationReductionMethod.SUM:
    accum_scale = 1.0
    grad_scale = None
  elif reduction_method == GradientAccumulationReductionMethod.MEAN:
    accum_scale = 1.0
    grad_scale = 1.0 / num_mini_batches
  elif reduction_method == \
      GradientAccumulationReductionMethod.RUNNING_MEAN:
    n = internal_ops.get_current_iteration_counter()
    n = n % num_mini_batches
    n = math_ops.cast(n, np.float32)
    accum_scale = n / (n + 1)
    grad_scale = 1.0 / (n + 1)
  else:
    raise ValueError(
        'reduction_method must be set to SUM, MEAN or RUNNING_MEAN')

  return accum_scale, grad_scale


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
               reduction_method=GradientAccumulationReductionMethod.SUM,
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
      replicated_optimizer_state_sharding: If True, any `tf.Variable` which is
        offloaded (for example the accumulator variable when using the
        `tf.MomentumOptimizer`), will be partitioned across the replicas.
        This can exploit the additional bandwidth of the IPU-Links to improve
        overall throughput, however it might increase the code size and hence
        the model might need adjusting (for example the PopLibs option
        `availableMemoryProportion` might need to be changed).
      dtype: The data type used for the gradient accumulation buffer.
        One of:

        - `None`: Use an accumulator of the same type as the variable type.
        - A `DType`: Use this type for all the accumulators.
          For example `tf.float32`.
        - A callable that takes the variable and returns a `DType`: Allows
          specifying the accumulator type on a per-variable basis.

        The gradients passed to `Optimizer.apply_gradients` will have the dtype
        requested here. If that dtype is different from the variable dtype
        a cast is needed at some point to make them compatible. If you want
        to cast the gradients immediately, you can wrap your optimizer in the
        `MapGradientOptimizer` with a `tf.cast`.
      reduction_method: Reduction method to use when accumulating gradients.
        During the iterations in each optimizer step, the computed gradients can
        either be directly summed up or scaled such that we compute a mean of
        all gradients for each variable. Computing a mean avoids potential
        issues with overflow during accumulation especially when using float16,
        but gives smaller gradients and might require adjusting the
        learning-rate accordingly.
        Defaults to `GradientAccumulationReductionMethod.SUM`
        (see :class:`~tensorflow.python.ipu.gradient_accumulation.GradientAccumulationReductionMethod`)  # pylint: disable=line-too-long
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientAccumulationOptimizerV2".
    """
    super().__init__(opt, name=name)
    self._opt = opt

    if num_mini_batches < 1:
      raise ValueError("num_mini_batches must be a positive number.")

    self._num_mini_batches = num_mini_batches

    self._offload_weight_update_variables = offload_weight_update_variables
    self._replicated_optimizer_state_sharding = \
        replicated_optimizer_state_sharding

    self._dtype = dtype

    self._reduction_method = GradientAccumulationReductionMethod.parse(
        reduction_method)

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
    accum_scale, grad_scale = _compute_scales(self._reduction_method,
                                              self._num_mini_batches)

    accumulated_grads_and_vars = op_util.accumulate_gradients(
        grads_and_vars, self._dtype, accum_scale, grad_scale)

    # Create an explicit function call for the apply gradients - note that we
    # allow external captures here.
    apply_grad_ops = []

    def resource_update_(accumulation_count):
      gen_poputil_ops.gradient_accumulation_count(accumulation_count)
      apply_grads = self._opt.apply_gradients(accumulated_grads_and_vars,
                                              global_step, name)
      if apply_grads is not None:
        apply_grad_ops.append(apply_grads)

    return op_util.create_resource_update(
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
               reduction_method=GradientAccumulationReductionMethod.SUM,
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
      replicated_optimizer_state_sharding: If True, any `tf.Variable` which
        is offloaded will be partitioned across the replicas. A collective
        all-gather will be inserted to restore the tensor on each replica.
        If `None`, this value will match the value of
        `offload_weight_update_variables`.
      dtype: The data type used for the gradient accumulation buffer.
        One of:

        - `None`: Use an accumulator of the same type as the variable type.
        - A `DType`: Use this type for all the accumulators.
          For example `tf.float32`.
        - A callable that takes the variable and returns a `DType`: Allows
          specifying the accumulator type on a per-variable basis.

        The gradients passed to `Optimizer.apply_gradients` will have the dtype
        requested here. If that dtype is different from the variable dtype
        a cast is needed at some point to make them compatible. If you want
        to cast the gradients immediately, you can wrap your optimizer in the
        `MapGradientOptimizer` with a `tf.cast`.
      reduction_method: Reduction method to use when accumulating gradients.
        During the iterations in each optimizer step, the computed gradients can
        either be directly summed up or scaled such that we compute a mean of
        all gradients for each variable. Computing a mean avoids potential
        issues with overflow during accumulation especially when using float16,
        but gives smaller gradients and might require adjusting the
        learning-rate accordingly.
        Defaults to `GradientAccumulationReductionMethod.SUM`
        (see :class:`~tensorflow.python.ipu.gradient_accumulation.GradientAccumulationReductionMethod`)  # pylint: disable=line-too-long
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "CrossReplicaGradientAccumulationOptimizerV2".
    """
    if num_mini_batches < 1:
      raise ValueError("num_mini_batches must be a positive number.")

    # Internally we just wrap the optimizer in a GradientAccumulationOptimizer and CrossReplicaOptimizer.
    opt = GradientAccumulationOptimizerV2(
        cross_replica_optimizer.CrossReplicaOptimizer(opt), num_mini_batches,
        offload_weight_update_variables, replicated_optimizer_state_sharding,
        dtype, reduction_method, name)

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
