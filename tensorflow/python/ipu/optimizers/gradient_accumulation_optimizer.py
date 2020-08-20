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

from tensorflow.compiler.plugin.poplar.ops import gen_functional_ops
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ipu import functional_ops
from tensorflow.python.ipu.optimizers import cross_replica_optimizer


class GradientAccumulationOptimizerV2(optimizer.Optimizer):  # pylint: disable=abstract-method
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
               offload_weight_update_variables=True,
               name="GradientAccumulationOptimizerV2"):
    """Construct a Gradient Accumulation Optimizer V2.

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
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientAccumulationOptimizerV2".
    """
    super().__init__(False, name)
    self._opt = opt

    if num_mini_batches < 1:
      raise ValueError("num_mini_batches must be a positive number.")

    self._num_mini_batches = num_mini_batches
    self._offload_weight_update_variables = offload_weight_update_variables

  def compute_gradients(self, *args, **kwargs):  #pylint: disable=arguments-differ
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps the compute_gradients() from the real optimizer. The
    gradients will be aggregated in the apply_gradients() so that user can
    modify the gradients like clipping.

    Args:
      *args: Arguments for compute_gradients().
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.
    """

    return self._opt.compute_gradients(*args, **kwargs)

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
    accumulated_grads_and_vars = []
    for grad, var in grads_and_vars:
      if grad is not None:
        with ops.colocate_with(grad):
          # Create an accumulator - variable is used as reference for shape/layout.
          accumulator = gen_poputil_ops.gradient_accumulator_create(var)
          # Add the gradients to the accumulator.
          accumulator = gen_poputil_ops.gradient_accumulator_add(
              accumulator, grad)
          # Sink the accumulators.
          grad = gen_poputil_ops.gradient_accumulator_sink(
              accumulator, num_mini_batches=self._num_mini_batches)
      # Use the accumulated gradients.
      accumulated_grads_and_vars.append((grad, var))

    # Create an explicit function call for the apply gradients - note that we
    # allow external caputres here.
    apply_grad_ops = []

    def resource_update_():
      apply_grads = self._opt.apply_gradients(accumulated_grads_and_vars,
                                              global_step, name)
      apply_grad_ops.append(apply_grads)

    with ops.name_scope(self._opt.get_name() + "/WU") as scope:
      func_graph, captured_args = functional_ops._compile_function(  # pylint: disable=protected-access
          resource_update_, [], scope, apply_grad_ops, True)

    # Create the resource update and lower the function into XLA.
    with ops.control_dependencies(list(func_graph.control_captures)):
      outputs = gen_functional_ops.resource_update(
          captured_args,
          to_apply=util.create_new_tf_function(func_graph),
          Tout=func_graph.output_types,
          output_shapes=func_graph.output_shapes,
          offload_weight_update_variables=self.
          _offload_weight_update_variables,
          num_batches_to_accumulate=self._num_mini_batches)

    return outputs

  def get_slot(self, *args, **kwargs):  #pylint: disable=arguments-differ
    """Return a slot named "name" created for "var" by the Optimizer.

    This simply wraps the get_slot() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    """
    return self._opt.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):  #pylint: disable=arguments-differ
    """Return a list of the names of slots created by the `Optimizer`.

    This simply wraps the get_slot_names() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      A list of strings.
    """
    return self._opt.get_slot_names(*args, **kwargs)

  def variables(self):
    """Forwarding the variables from the underlying optimizer."""
    return self._opt.variables()


class CrossReplicaGradientAccumulationOptimizerV2(optimizer.Optimizer):  # pylint: disable=abstract-method
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
               offload_weight_update_variables=True,
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
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "CrossReplicaGradientAccumulationOptimizerV2".
    """

    super().__init__(False, name)

    if num_mini_batches < 1:
      raise ValueError("num_mini_batches must be a positive number.")

    # Internally we just wrap the optimizer in a GradientAccumulationOptimizer and CrossReplicaOptimizer.
    self._opt = GradientAccumulationOptimizerV2(
        cross_replica_optimizer.CrossReplicaOptimizer(opt), num_mini_batches,
        offload_weight_update_variables, name)

  def compute_gradients(self, *args, **kwargs):  #pylint: disable=arguments-differ
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps the compute_gradients() from the real optimizer. The
    gradients will be aggregated in the apply_gradients() so that user can
    modify the gradients like clipping.

    Args:
      *args: Arguments for compute_gradients().
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.
    """

    return self._opt.compute_gradients(*args, **kwargs)

  def apply_gradients(self, *args, **kwargs):  #pylint: disable=arguments-differ
    """Apply gradients to variables.

    Args:
      *args: Arguments for apply_gradients().
      **kwargs: Keyword arguments for apply_gradients().

    Returns:
      An `Operation` that applies the gradients. If `global_step` was not None,
      that operation also increments `global_step`.

    """

    return self._opt.apply_gradients(*args, **kwargs)

  def get_slot(self, *args, **kwargs):  #pylint: disable=arguments-differ
    """Return a slot named "name" created for "var" by the Optimizer.

    This simply wraps the get_slot() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    """
    return self._opt.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):  #pylint: disable=arguments-differ
    """Return a list of the names of slots created by the `Optimizer`.

    This simply wraps the get_slot_names() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      A list of strings.
    """
    return self._opt.get_slot_names(*args, **kwargs)

  def variables(self):
    """Forwarding the variables from the underlying optimizer."""
    return self._opt.variables()


class GradientAccumulationOptimizer(optimizer.Optimizer):
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

    super(GradientAccumulationOptimizer, self).__init__(False, name)
    self._opt = opt

    if num_mini_batches < 1:
      raise ValueError("num_mini_batches must be a positive number.")

    self._num_mini_batches = num_mini_batches
    self._verify_usage = verify_usage

  def compute_gradients(self, loss, var_list=None, **kwargs):
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps the compute_gradients() from the real optimizer. The
    gradients will be aggregated in the apply_gradients() so that user can
    modify the gradients like clipping.

    Args:
      loss: A Tensor containing the value to minimize.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKey.TRAINABLE_VARIABLES`.
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.
    """

    return self._opt.compute_gradients(loss, var_list=var_list, **kwargs)

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

  def get_slot(self, *args, **kwargs):
    """Return a slot named "name" created for "var" by the Optimizer.

    This simply wraps the get_slot() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    """
    return self._opt.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):
    """Return a list of the names of slots created by the `Optimizer`.

    This simply wraps the get_slot_names() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      A list of strings.
    """
    return self._opt.get_slot_names(*args, **kwargs)

  def variables(self):
    """Forwarding the variables from the underlying optimizer."""
    return self._opt.variables()


class CrossReplicaGradientAccumulationOptimizer(optimizer.Optimizer):
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

    super(CrossReplicaGradientAccumulationOptimizer,
          self).__init__(False, name)

    if num_mini_batches < 1:
      raise ValueError("num_mini_batches must be a positive number.")

    # Internally we just wrap the optimizer in a GradientAccumulationOptimizer and CrossReplicaOptimizer.
    self._opt = cross_replica_optimizer.CrossReplicaOptimizer(
        GradientAccumulationOptimizer(opt, num_mini_batches, verify_usage))

  def compute_gradients(self, loss, var_list=None, **kwargs):
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps the compute_gradients() from the real optimizer. The
    gradients will be aggregated in the apply_gradients() so that user can
    modify the gradients like clipping.

    Args:
      loss: A Tensor containing the value to minimize.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKey.TRAINABLE_VARIABLES`.
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.
    """

    return self._opt.compute_gradients(loss, var_list=var_list, **kwargs)

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

    """

    return self._opt.apply_gradients(grads_and_vars, global_step, name)

  def get_slot(self, *args, **kwargs):
    """Return a slot named "name" created for "var" by the Optimizer.

    This simply wraps the get_slot() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    """
    return self._opt.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):
    """Return a list of the names of slots created by the `Optimizer`.

    This simply wraps the get_slot_names() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      A list of strings.
    """
    return self._opt.get_slot_names(*args, **kwargs)

  def variables(self):
    """Forwarding the variables from the underlying optimizer."""
    return self._opt.variables()
