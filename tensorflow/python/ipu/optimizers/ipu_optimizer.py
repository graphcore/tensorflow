# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""
Convenience wrapper optimizer to help create new custom optimizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.training import optimizer
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2


class IpuOptimizer(optimizer.Optimizer):
  """
  The wrapper interface for optimizer.Optimizer optimizers. Custom wrappers
  written for IPU can inherit from this class and
  override the appropriate functions.

  This provides the convenience of automatically passing on functions that
  have not been overwritten to the sub class and also allows you to define
  custom APIs specifically for the IPU.
  """
  def __init__(self, opt, name=None):
    """
    Construct a new IpuOptimizer

    Args:
      opt: The optimizer to be wrapped.
      name: The name to be passed to Optimizer constructor.
    """
    if isinstance(opt, OptimizerV2):
      raise ValueError("Should use optimizer in "
                       "ipu.keras.optimizers "
                       "to wrap V2 optimizers")
    super(IpuOptimizer, self).__init__(False, name)
    self._opt = opt

  def get_name(self):
    """
    Return the name of the underlying optimizer
    """
    return self._opt.get_name()

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

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    Applies gradients from underlying optimizer.

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
    return self._opt.apply_gradients(grads_and_vars, global_step, name)

  def compute_gradients(self, loss, var_list=None, **kwargs):  #pylint: disable=arguments-differ
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps the compute_gradients() from the real optimizer. The
    gradients will be aggregated in the apply_gradients() so that user can
    modify the gradients like clipping with per replica global norm if needed.

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

  def _apply_dense(self, grad, var):
    """Add ops to apply dense gradients to `var`.

    Args:
      grad: A `Tensor`.
      var: A `Variable` object.

    Returns:
      An `Operation`.
    """
    return self._opt._apply_dense(self, grad, var)  # pylint: disable=protected-access

  def _resource_apply_dense(self, grad, handle):
    """Add ops to apply dense gradients to the variable `handle`.

    Args:
      grad: a `Tensor` representing the gradient.
      handle: a `Tensor` of dtype `resource` which points to the variable
       to be updated.

    Returns:
      An `Operation` which updates the value of the variable.
    """
    return self._opt._resource_apply_dense(grad, handle)  # pylint: disable=protected-access

  def _apply_sparse(self, grad, var):
    """Add ops to apply sparse gradients to `var`.

    Args:
      grad: `IndexedSlices`, with no repeated indices.
      var: A `Variable` object.

    Returns:
      An `Operation`.
    """
    return self._opt._apply_sparse(grad, var)  # pylint: disable=protected-access

  def _resource_apply_sparse(self, grad, handle, indices):
    """Add ops to apply sparse gradients to the variable `handle`.

    Args:
      grad: a `Tensor` representing the gradient for the affected indices.
      handle: a `Tensor` of dtype `resource` which points to the variable
       to be updated.
      indices: a `Tensor` of integral type representing the indices for
       which the gradient is nonzero. Indices are unique.

    Returns:
      An `Operation` which updates the value of the variable.
    """
    return self._opt._resource_apply_sparse(grad, handle, indices)  # pylint: disable=protected-access
