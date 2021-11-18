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
from tensorflow.python.keras.optimizer_v1 import TFOptimizer
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
    self._opt = opt
    super().__init__(name=name, **kwargs)

  def __setattr__(self, name, value):
    """
    Default wrapper for setattr to support dynamic hyperparameter setting
    for the wrapped optimizer.

    Args:
      name: The name of the attribute to set.
      value: The value to set.
    """
    # Delegate setting hyperparameter to inner optimizer if the attribute does
    # not exist on the LossScaleOptimizer
    try:
      # We cannot check for the 'iterations' attribute as it cannot be set after
      # it is accessed.
      if name != 'iterations':
        object.__getattribute__(self, name)
      has_attribute = True
    except AttributeError:
      has_attribute = False

    if (name != '_opt' and name in self._opt._hyper and not has_attribute):  # pylint: disable=protected-access
      self._opt._set_hyper(name, value)  # pylint: disable=protected-access
    else:
      super().__setattr__(name, value)

  def __getattribute__(self, name):
    """
    Default wrapper to support hyperparameter access for the
    wrapped optimizer.

    Args:
      name: The name of the attribute to get.
    """
    try:
      return object.__getattribute__(self, name)
    except AttributeError as e:
      if name == '_opt' or name == '_hyper':
        # Avoid infinite recursion
        raise e

      # Delegate hyperparameter accesses to inner optimizer.
      if name in self._opt._hyper:  # pylint: disable=protected-access
        return self._opt._get_hyper(name)  # pylint: disable=protected-access
      raise e

  def __dir__(self):
    """
    Default wrapper to support listing attributes of the wrapped optimizer.
    """
    result = set(super().__dir__())

    if '_opt' in result:
      result |= self._opt._hyper.keys()  # pylint: disable=protected-access
    return list(result)

  def _create_slots(self, var_list):
    """
    Default wrapper that calls the wrapped optimizer's `_create_slots`.
    """
    return self._opt._create_slots(var_list)  # pylint: disable=protected-access

  def _create_hypers(self):
    """
    Default wrapper that calls the wrapped optimizer's `_create_hypers`.
    """
    return self._opt._create_hypers()  # pylint: disable=protected-access

  def _create_all_weights(self, var_list):
    """
    Default wrapper that calls the wrapped optimizer's `_create_all_weights`.
    """
    _ = self.iterations
    return self._opt._create_all_weights(var_list)  # pylint: disable=protected-access

  def _prepare(self, var_list):
    """
    Default wrapper that calls the wrapped optimizer's `_prepare`.
    """
    self._opt._prepare(var_list)  # pylint: disable=protected-access

  def _transform_unaggregated_gradients(self, grads_and_vars):
    """
    Default wrapper that calls the wrapped optimizer's
    `_transform_unaggregated_gradients`.
    """
    return self._opt._transform_unaggregated_gradients(grads_and_vars)  # pylint: disable=protected-access

  def _aggregate_gradients(self, grads_and_vars):
    """
    Default wrapper that calls the wrapped optimizer's
    `_aggregate_gradients`.
    """
    return self._opt._aggregate_gradients(grads_and_vars)  # pylint: disable=protected-access

  def _transform_gradients(self, grads_and_vars):
    """
    Default wrapper that calls the wrapped optimizer's
    `_transform_gradients`.
    """
    return self._opt._transform_gradients(grads_and_vars)  # pylint: disable=protected-access

  def get_gradients(self, loss, params):
    """
    Default wrapper that calls the wrapped optimizer's `get_gradients`.

    Args:
      loss: A loss to compute gradients of.
      params: A list of variables to compute gradients with respect to.
    """
    return self._opt.get_gradients(loss, params)  # pylint: disable=protected-access

  def _resource_apply_dense(self, grad, handle, apply_state):
    """
    Default wrapper that calls the wrapped optimizer's `_resource_apply_dense`.

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
    optimizers `_resource_apply_sparse`.

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

  @staticmethod
  def _verify_config(config):
    """Verifies an `IpuOptimizer` configuration.
    """
    if not 'inner_optimizer_config' in config:
      raise ValueError(
          "IpuOptimizer configuration dicts must contain a configuration "
          "dict for the wrapped optimizer, inner_optimizer_config.")

    if not 'inner_optimizer_type' in config:
      raise ValueError("IpuOptimizer configuration dicts must contain a type "
                       "for the wrapped optimizer, inner_optimizer_type.")

    opt_type = config['inner_optimizer_type']
    if not issubclass(opt_type, OptimizerV2):
      raise ValueError(
          "inner_optimizer_type must be a class derived from Keras OptimizerV2."
      )

  def get_config(self):
    """Returns the config of the `IpuOptimizer` instance.

    An optimizer config is a Python dictionary (serializable)
    containing the configuration of an optimizer.
    The same optimizer can be reinstantiated later
    (without any saved state) from this configuration.

    The returned config will contain at a minimum, `inner_optimizer_config`,
    `inner_optimizer_type` and `name`.

    Returns:
        Python dictionary.
    """
    return {
        'inner_optimizer_config': self._opt.get_config(),
        'inner_optimizer_type': self._opt.__class__,
        'name': self._name
    }

  def get_slot(self, var, slot_name):
    """
    Default wrapper that calls the wrapped optimizer's `get_slot`.

    Args:
      var: A variable to look up.
      slot_name: The name of the slot.
    """
    return self._opt.get_slot(var, slot_name)

  def add_slot(self, var, slot_name, initializer="zeros"):
    """
    Default wrapper that calls the wrapped optimizer's `add_slot`.

    Args:
      var: A variable to add.
      slot_name: The name of the slot.
      initializer: Default initializer for `var`.
    """
    return self._opt.add_slot(var, slot_name, initializer=initializer)

  def get_slot_names(self):
    """
    Default wrapper that calls the wrapped optimizer's `get_slot_names`.
    """
    return self._opt.get_slot_names()

  def get_weights(self):
    """
    Default wrapper that calls the wrapped optimizer's `get_weights`.
    """
    return self._opt.get_weights()

  def set_weights(self, weights):
    """
    Default wrapper that calls the wrapped optimizer's `set_weights`.

    Args:
      weights: The weights to set.
    """
    return self._opt.set_weights(weights)

  def _restore_slot_variable(self, slot_name, variable, slot_variable):
    """
    Default wrapper that calls the wrapped optimizer's
    `_restore_slot_variable`.
    """
    return self._opt._restore_slot_variable(slot_name, variable, slot_variable)  # pylint: disable=protected-access

  def _create_or_restore_slot_variable(self, slot_variable_position, slot_name,
                                       variable):
    """
    Default wrapper that calls the wrapped optimizer's
    `_create_or_restore_slot_variable`.
    """
    return self._opt._create_or_restore_slot_variable(  # pylint: disable=protected-access
        slot_variable_position, slot_name, variable)

  def preprocess_gradients(self, grad, var):
    """
    Default wrapper to call through to wrapped
    optimizers preprocess_gradients if it has it.
    """
    if isinstance(self._opt, IpuOptimizer):
      return self._opt.preprocess_gradients(grad, var)
    return (grad, var)

  def variables(self):
    """
    Returns the variables of the wrapped optimizer.
    """
    return self.weights

  @property
  def weights(self):
    return self._opt.weights

  @property
  def clipnorm(self):
    return self._opt.clipnorm

  @clipnorm.setter
  def clipnorm(self, val):
    self._opt.clipnorm = val

  @property
  def global_clipnorm(self):
    return self._opt.global_clipnorm

  @global_clipnorm.setter
  def global_clipnorm(self, val):
    self._opt.global_clipnorm = val

  @property
  def clipvalue(self):
    return self._opt.clipvalue

  @clipvalue.setter
  def clipvalue(self, val):
    self._opt.clipvalue = val


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

    if isinstance(self._optimizer, TFOptimizer):
      grads_and_vars = self._optimizer.get_grads(loss, v)
    else:
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
