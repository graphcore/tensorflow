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


class MapGradientOptimizer(IpuOptimizer):
  """
  Apply a function to gradients before they are applied to the variables.

  If wrapping multiple MapGradientOptimizers, inner map functions will be
  applied first.
  """
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
    super(MapGradientOptimizer, self).__init__(opt, name=name)
    self._gradient_mapping_function = gradient_mapping_function

  def preprocess_gradients(self, grad, var):
    """
    Apply the gradient mapping function to the gradient.

    Args:
      grad: The gradient to apply function to.
      var: The variable gradient corresponds to.
    """
    (grad, var) = super(MapGradientOptimizer,
                        self).preprocess_gradients(grad, var)
    return (self._gradient_mapping_function(grad, var), var)
