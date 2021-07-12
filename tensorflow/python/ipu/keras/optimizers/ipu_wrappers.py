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
# =============================================================================
"""
Convenience wrappers for v2 optimizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.training.optimizer import Optimizer


class IpuOptimizer(OptimizerV2):
  """
  The wrapper interface for ipu.keras v2 optimizers. Any custom wrappers
  written for IPU keras applications should inherit from this class and
  override the appropriate functions.

  This provides the convenience of automatically passing on functions that
  have not been overwritten to the sub class and also allows you to define
  custom APIs specifically for the IPU.
  """
  def __init__(self, opt, name=None, **kwargs):
    """
    Construct a new IpuOptimizer

    Args:
      opt: The optimizer to be wrapped.
      name: The name to be passed to OptimizerV2 constructor.
      kwargs: The keyword arguments to be passed to OptimizerV2 constructor.
    """
    super(IpuOptimizer, self).__init__(name, **kwargs)
    self._opt = opt

  def _create_slots(self, var_list):
    """
    Default wrapper that calls the wrapped optimizer's _create_slots.

    Args:
      var_list: The var_list to be passed to wrapped optimizer's _create_slots.
    """
    return self._opt._create_slots(var_list)  # pylint: disable=protected-access

  def _resource_apply_dense(self, grad, handle, apply_state):
    """
    Default wrapper that calls the wrapped
    optimizer's _resource_apply_dense.

    Args:
      grad: A `Tensor` representing the gradient.
      handle: A `Tensor` of dtype `resource` which points to the variable to be
        updated.
      apply_state: A dict which is used across multiple apply calls.

    Returns:
      An `Operation` which updates the value of the variable.
    """
    return self._opt._resource_apply_dense(grad, handle, apply_state)  # pylint: disable=protected-access

  def _resource_apply_sparse(self, grad, handle, indices, apply_state):
    """
    Default wrapper to call through to wrapped
    optimizers _resource_apply_sparse.

    Args:
      grad: A `Tensor` representing the gradient for the affected indices.
      handle: A `Tensor` of dtype `resource` which points to the variable to be
        updated.
      indices: A `Tensor` of integral type representing the indices for which
        the gradient is nonzero. Indices are unique.
      apply_state: A dict which is used across multiple apply calls.

    Returns:
      An `Operation` which updates the value of the variable.
    """
    return self._opt._resource_apply_sparse(  # pylint: disable=protected-access
        grad, handle, indices, apply_state)

  def get_config(self):
    """
    Default wrapper to call through to wrapped optimizers get_config.
    """
    return self._opt.get_config()  # pylint: disable=protected-access

  def preprocess_gradients(self, grad, var):
    """
    Default wrapper to call through to wrapped
    optimizers preprocess_gradients if it has it.
    """
    if isinstance(self._opt, IpuOptimizer):
      return self._opt.preprocess_gradients(grad, var)
    return (grad, var)


class _TensorflowOptimizerWrapper(Optimizer):
  """A class which wraps a standard TensorFlow optimizer,
  giving it a TensorFlow optimizer interface,
  but generating gradients against the Keras Model.
  """
  def __init__(self, model, opt):
    super(_TensorflowOptimizerWrapper, self).__init__(use_locking=False,
                                                      name="optimizer_shim")
    self._model = model
    self._optimizer = opt

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    return self._optimizer.compute_gradients(loss,
                                             self._model.trainable_weights)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    return self._optimizer.apply_gradients(grads_and_vars, global_step, name)

  def _apply_sparse(self, grad, var):
    raise NotImplementedError()

  def _apply_dense(self, grad, var):
    raise NotImplementedError()

  def _resource_apply_dense(self, grad, handle):
    raise NotImplementedError()

  def _resource_apply_sparse(self, grad, handle, indices):
    raise NotImplementedError()


class _KerasOptimizerWrapper(Optimizer):
  """A class which wraps a Keras optimizer,
  giving it a TensorFlow optimizer interface.
  """
  def __init__(self, model, opt):
    super(_KerasOptimizerWrapper, self).__init__(use_locking=False,
                                                 name="optimizer_shim")
    self._model = model
    self._optimizer = opt

  def preprocess_gradients(self, x):
    (grad, var) = x
    if isinstance(self._optimizer, IpuOptimizer):
      return self._optimizer.preprocess_gradients(grad, var)
    return (grad, var)

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    if not self._model and not var_list:
      raise ValueError(
          "When _KerasOptimizerWrapper has been instantiated with it's model "
          "set to None, var_list must be provided.")

    v = var_list if not self._model else self._model.trainable_weights

    grads = self._optimizer.get_gradients(loss, v)
    grads_and_vars = zip(grads, v)
    return list(map(self.preprocess_gradients, grads_and_vars))

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    return self._optimizer.apply_gradients(grads_and_vars)

  def _apply_sparse(self, grad, var):
    raise NotImplementedError()

  def _apply_dense(self, grad, var):
    raise NotImplementedError()

  def _resource_apply_dense(self, grad, handle):
    raise NotImplementedError()

  def _resource_apply_sparse(self, grad, handle, indices):
    raise NotImplementedError()
