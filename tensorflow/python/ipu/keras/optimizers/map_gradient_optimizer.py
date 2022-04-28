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
Map gradient optimizer wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.ipu.keras.optimizers import IpuOptimizer
from tensorflow.python.util import deprecation


class MapGradientOptimizer(IpuOptimizer):
  """
  Removed, please use MapGradientOptimizerInvertedChaining.
  """
  @deprecation.deprecated(None, "This optimizer will be removed.")
  def __init__(self,
               opt,
               gradient_mapping_function,
               name="MapGradientOptimizer"):
    """
    Construct a MapGradientOptimizer.

    Args:
      opt: The optimizer to be wrapped.
      gradient_mapping_function: The function to be applied to the gradients.
             Mapping functions should be of form fn(grad, var) and return
             the updated gradient.
    """
    raise NotImplementedError(
        "Use MapGradientOptimizerInvertedChaining instead")


class MapGradientOptimizerInvertedChaining(IpuOptimizer):
  """
  Apply a function to gradients before they are applied to the variables.

  If wrapping multiple optimizers then the outer mapping
  functions will be applied first  (this is
  the opposite way to `MapGradientOptimizer`).
  If used with `MapGradientOptimizer` wrapper then
  the `MapGradientOptimziers` will always be applied first.
  """
  @deprecation.deprecated(None, "This optimizer will be removed.")
  def __init__(self,
               opt,
               gradient_mapping_function,
               name="MapGradientOptimizerInvertedChaining"):
    """
    Construct a MapGradientOptimizerInvertedChaining.

    Args:
      opt: The optimizer to be wrapped.
      gradient_mapping_function: The function to be applied to the gradients.
              Mapping functions should be of the form fn(grad, var) and return
              the updated gradient.
    """
    super().__init__(opt, name=name)
    self.gradient_mapping_function = gradient_mapping_function

  def _resource_apply_dense(self, grad, handle, apply_state):
    """Apply gradient to variable referenced by `handle`.

    Args:
      grad: The gradient to be applied.
      handle: A handle to the variable to apply the gradient to.
      apply_state: State passed down to the wrapped
                   optimizer's apply functions.
    Returns:
      The updated variable.
    """
    mapped_grad = self._gradient_mapping_function(grad, handle)
    return super()._resource_apply_dense(  # pylint: disable=protected-access
        mapped_grad,
        handle,
        apply_state=apply_state)

  def get_config(self):
    """
    Returns the config of the `MapGradientOptimizer` instance.
    """
    config = super().get_config()
    config.update(
        {'gradient_mapping_function': self.gradient_mapping_function})
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    """Creates a `MapGradientOptimizer` from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same optimizer from the config
    dictionary.

    Arguments:
        config: A Python dictionary, typically the output of get_config.
        custom_objects: A Python dictionary mapping names to additional Python
          objects used to create this optimizer, such as a function used for a
          hyperparameter.

    Returns:
        A `MapGradientOptimizer` instance.
    """
    config = config.copy()
    IpuOptimizer._verify_config(config)
    inner_config = config.pop('inner_optimizer_config')
    inner_type = config.pop('inner_optimizer_type')
    inner_opt = inner_type(**inner_config)

    return MapGradientOptimizerInvertedChaining(inner_opt, **config)

  @property
  def gradient_mapping_function(self):
    return self._gradient_mapping_function

  @gradient_mapping_function.setter
  def gradient_mapping_function(self, f):
    if not callable(f):
      raise ValueError("gradient_mapping_function must be a callable.")

    self._gradient_mapping_function = f
